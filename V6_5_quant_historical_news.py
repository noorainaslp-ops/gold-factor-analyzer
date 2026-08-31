```
# V6_5_3_quant_gemini_leakage_proof.py
#
# V6.5.3 — LEAKAGE-PROOF QUANT + HISTORICAL GEMINI EXPERIMENT
#
# Purpose:
#   1. Build strictly backward-looking OHLCV features.
#   2. Generate future-return targets separately.
#   3. Perform purged chronological walk-forward validation.
#   4. Compare:
#        A. QUANT
#        B. GEMINI/NEWS ONLY (when historical scores exist)
#        C. HYBRID QUANT + GEMINI (when historical scores exist)
#   5. Never call Gemini retrospectively during a historical backtest.
#   6. Never use OOS observations for model/weight/threshold selection.
#
# Historical Gemini file:
#   historical_gemini.csv
#
# Required columns:
#   ticker,published_at,gemini_score
#
# gemini_score must represent information available AT THAT TIME.
# Range recommended: -1 to +1.
#
# Example:
#   ticker,published_at,gemini_score
#   RELIANCE.NS,2024-01-15 08:30:00,0.42
#
# IMPORTANT:
#   A current Gemini API call is NOT used as a historical feature.
#   Doing that would contaminate the backtest.
#
# Production use:
#   Run this historical experiment first.
#   Only after a genuine OOS improvement is demonstrated should
#   live Gemini scoring be considered.

from __future__ import annotations

import os
import math
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "V6.5.3"
SOURCE_REVISION = "2026-08-31-LEAKAGE-PROOF-HYBRID"

YEARS = 6
ROUND_TRIP_COST = 0.0030

RANDOM_STATE = 42

# Minimum training observations
MIN_TRAIN = 1000

# Purge:
# remove observations whose forward-return window can overlap
# the prediction date / validation boundary.
PURGE_DAYS = 10

# Walk-forward retraining frequency
RETRAIN_EVERY = 20

# Trade probability threshold
TRADE_PMIN = 0.55

# Minimum predicted net return
TRADE_RMIN = 0.0020

# Hybrid weights tested only on VALIDATION
NEWS_WEIGHTS = np.array([
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
])

HORIZONS = [1, 3, 5, 10]

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(exist_ok=True)

# ============================================================
# UNIVERSE
# ============================================================

TICKERS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "INDUSINDBK.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "SHRIRAMFIN.NS",
    "LT.NS",
    "TMPV.NS",
    "TMCV.NS",
    "EICHERMOT.NS",
    "MARUTI.NS",
    "HEROMOTOCO.NS",
    "M&M.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    "TCS.NS",
    "INFY.NS",
    "HCLTECH.NS",
    "WIPRO.NS",
    "TECHM.NS",
    "BHARTIARTL.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "BPCL.NS",
    "COALINDIA.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "BEL.NS",
    "HAL.NS",
    "BHEL.NS",
    "TRENT.NS",
    "PIDILITIND.NS",
    "SIEMENS.NS",
    "ABB.NS",
    "GRASIM.NS",
    "ULTRACEMCO.NS",
    "JSWSTEEL.NS",
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "IOC.NS",
    "VEDL.NS",
    "DLF.NS",
    "LODHA.NS",
    "INDIGO.NS",
    "ETERNAL.NS",
    "NAUKRI.NS",
    "COFORGE.NS",
    "JIOFIN.NS",
    "IRFC.NS",
    "IREDA.NS",
    "POLYCAB.NS",
]

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly normalise yfinance output.

    Handles:
      - ordinary single-level OHLCV columns
      - MultiIndex columns
      - Series returned under a column
    """

    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # Prefer the OHLCV field level.
        wanted = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

        new_cols = []

        for col in df.columns:
            parts = [str(x) for x in col]

            found = None

            for p in parts:
                if p in wanted:
                    found = p
                    break

            if found is None:
                found = parts[-1]

            new_cols.append(found)

        df.columns = new_cols

        # Remove duplicate columns by taking the first valid one.
        df = df.loc[:, ~df.columns.duplicated()]

    else:
        df.columns = [str(c) for c in df.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]

    for c in required:
        if c not in df.columns:
            return pd.DataFrame()

    for c in required:
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]

        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[required].copy()

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    return df


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ============================================================
# TECHNICAL FEATURES
# ============================================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(period).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    close = x["Close"]
    volume = x["Volume"]

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_5"] = close.pct_change(5)
    x["ret_10"] = close.pct_change(10)
    x["ret_20"] = close.pct_change(20)

    x["vol_5"] = x["ret_1"].rolling(5).std()
    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_20"] = x["ret_1"].rolling(20).std()

    x["rsi_7"] = rsi(close, 7)
    x["rsi_14"] = rsi(close, 14)
    x["rsi_21"] = rsi(close, 21)

    x["atr_14"] = atr(x, 14)
    x["atr_pct"] = x["atr_14"] / close

    ema_10 = close.ewm(span=10, adjust=False).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    x["ema10_dist"] = close / ema_10 - 1
    x["ema20_dist"] = close / ema_20 - 1
    x["ema50_dist"] = close / ema_50 - 1

    x["ema10_20"] = ema_10 / ema_20 - 1
    x["ema20_50"] = ema_20 / ema_50 - 1

    high20 = x["High"].rolling(20).max()
    low20 = x["Low"].rolling(20).min()

    x["breakout_20"] = close / high20.shift(1) - 1
    x["breakdown_20"] = close / low20.shift(1) - 1

    vol_mean20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std()

    x["volume_z"] = (
        (volume - vol_mean20) /
        vol_std20.replace(0, np.nan)
    )

    x["range_pct"] = (x["High"] - x["Low"]) / close

    x["close_location"] = (
        (close - x["Low"]) /
        (x["High"] - x["Low"]).replace(0, np.nan)
    )

    # Market-relative momentum proxy
    x["momentum_accel"] = x["ret_5"] - x["ret_20"] / 4

    return x


# ============================================================
# TARGETS
# ============================================================

def add_targets(df: pd.DataFrame) -> pd.DataFrame:

    x = df.copy()

    for h in HORIZONS:

        # Future close return.
        future_return = (
            x["Close"].shift(-h) / x["Close"] - 1
        )

        x[f"target_{h}d"] = future_return

        # Binary target.
        x[f"up_{h}d"] = (
            future_return > 0
        ).astype(float)

    return x


# ============================================================
# HISTORICAL GEMINI
# ============================================================

def load_historical_gemini() -> pd.DataFrame:

    candidates = [
        Path("historical_gemini.csv"),
        Path("historical_news.csv"),
        Path("data/historical_gemini.csv"),
        Path("data/historical_news.csv"),
    ]

    path = None

    for p in candidates:
        if p.exists():
            path = p
            break

    if path is None:

        print(
            "HISTORICAL GEMINI: NOT FOUND — "
            "RUNNING QUANT-ONLY BASELINE"
        )

        return pd.DataFrame(
            columns=[
                "ticker",
                "published_at",
                "gemini_score"
            ]
        )

    news = pd.read_csv(path)

    required = {
        "ticker",
        "published_at",
        "gemini_score"
    }

    missing = required - set(news.columns)

    if missing:
        raise ValueError(
            f"Historical Gemini file missing columns: {sorted(missing)}"
        )

    news = news.copy()

    news["ticker"] = news["ticker"].astype(str)

    news["published_at"] = pd.to_datetime(
        news["published_at"],
        errors="coerce",
        utc=True
    )

    news["gemini_score"] = pd.to_numeric(
        news["gemini_score"],
        errors="coerce"
    )

    news = news.dropna(
        subset=[
            "ticker",
            "published_at",
            "gemini_score"
        ]
    )

    news["gemini_score"] = (
        news["gemini_score"]
        .clip(-1, 1)
    )

    news = news.sort_values(
        ["ticker", "published_at"]
    )

    print(
        f"HISTORICAL GEMINI: FOUND — "
        f"{len(news):,} timestamped observations"
    )

    return news


def attach_historical_gemini(
    market: pd.DataFrame,
    news: pd.DataFrame
) -> pd.DataFrame:

    if news.empty:

        market["gemini_score"] = 0.0
        market["gemini_available"] = 0

        return market

    market = market.copy()

    market["signal_dt"] = pd.to_datetime(
        market["date"],
        utc=True
    )

    news = news.copy()

    news["published_at"] = pd.to_datetime(
        news["published_at"],
        utc=True
    )

    market = market.sort_values(
        ["ticker", "signal_dt"]
    )

    news = news.sort_values(
        ["ticker", "published_at"]
    )

    # CRITICAL:
    # only news published on or before the signal timestamp
    # is eligible.
    merged = pd.merge_asof(
        market,
        news,
        left_on="signal_dt",
        right_on="published_at",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
        suffixes=("", "_news")
    )

    merged["gemini_score"] = (
        pd.to_numeric(
            merged["gemini_score"],
            errors="coerce"
        )
        .fillna(0.0)
        .clip(-1, 1)
    )

    merged["gemini_available"] = (
        merged["published_at"].notna()
    ).astype(int)

    return merged


# ============================================================
# DATA BUILD
# ============================================================

def download_symbol(ticker: str) -> pd.DataFrame:

    try:

        df = yf.download(
            ticker,
            period=f"{YEARS}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        df = clean_columns(df)

        if df.empty:
            return pd.DataFrame()

        df.index = pd.to_datetime(
            df.index,
            errors="coerce"
        )

        df = df[~df.index.isna()]

        df["date"] = df.index

        return df.reset_index(drop=True)

    except Exception as e:

        print(
            f"WARNING: {ticker} failed: {e}"
        )

        return pd.DataFrame()


def build_dataset(news: pd.DataFrame) -> pd.DataFrame:

    rows = []

    print(
        f"Loading {len(TICKERS)} symbols..."
    )

    for i, ticker in enumerate(TICKERS, 1):

        print(
            f"Loading [{i}/{len(TICKERS)}] {ticker}"
        )

        df = download_symbol(ticker)

        if df.empty:

            print(
                f"WARNING: insufficient data for {ticker}; skipping."
            )

            continue

        if len(df) < 300:

            print(
                f"WARNING: insufficient history for "
                f"{ticker}; skipping."
            )

            continue

        df = build_features(df)
        df = add_targets(df)

        df["ticker"] = ticker

        rows.append(df)

    if not rows:
        raise RuntimeError(
            "No valid market data were loaded."
        )

    data = pd.concat(
        rows,
        ignore_index=True
    )

    data = attach_historical_gemini(
        data,
        news
    )

    data = data.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    return data


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

QUANT_FEATURES = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_5",
    "vol_10",
    "vol_20",
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "atr_pct",
    "ema10_dist",
    "ema20_dist",
    "ema50_dist",
    "ema10_20",
    "ema20_50",
    "breakout_20",
    "breakdown_20",
    "volume_z",
    "range_pct",
    "close_location",
    "momentum_accel",
]

GEMINI_FEATURE = [
    "gemini_score"
]


# ============================================================
# FEATURE/TARGET LEAKAGE CHECK
# ============================================================

def leakage_check():

    target_terms = [
        "target_",
        "up_",
        "future",
        "forward",
    ]

    for feature in QUANT_FEATURES:

        low = feature.lower()

        for term in target_terms:

            if term in low:

                raise RuntimeError(
                    f"LEAKAGE FAILURE: {feature}"
                )

    print(
        "FEATURE/TARGET LEAKAGE CHECK: PASS"
    )

    print(
        "Forward-return targets are excluded "
        "from FEATURES."
    )

    print(
        "Historical Gemini timestamps are required "
        "to be <= signal date."
    )


# ============================================================
# MODEL FITTING
# ============================================================

def prepare_xy(
    df: pd.DataFrame,
    features: list[str],
    horizon: int
):

    cols = features + [
        f"target_{horizon}d",
        f"up_{horizon}d"
    ]

    x = df[cols].copy()

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = x.notna().all(axis=1)

    x = x.loc[valid]

    X = x[features]

    y_return = x[f"target_{horizon}d"]

    y_up = x[f"up_{horizon}d"].astype(int)

    return X, y_up, y_return


def fit_models(
    train: pd.DataFrame,
    features: list[str],
    horizon: int
):

    X, y_up, y_return = prepare_xy(
        train,
        features,
        horizon
    )

    if len(X) < MIN_TRAIN:

        raise ValueError(
            f"Insufficient training observations: {len(X)}"
        )

    classifier = Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "logistic",
            LogisticRegression(
                C=0.5,
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])

    classifier.fit(
        X,
        y_up
    )

    gb_classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE
    )

    gb_classifier.fit(
        X,
        y_up
    )

    ridge = Pipeline([
        (
            "scale",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(alpha=10.0)
        )
    ])

    ridge.fit(
        X,
        y_return
    )

    gb_regressor = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
        loss="squared_error"
    )

    gb_regressor.fit(
        X,
        y_return
    )

    return {
        "logistic": classifier,
        "gb_classifier": gb_classifier,
        "ridge": ridge,
        "gb_regressor": gb_regressor,
    }


def predict_models(
    models,
    df: pd.DataFrame,
    features: list[str]
):

    X = (
        df[features]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    valid = X.notna().all(axis=1)

    p = np.full(len(df), np.nan)
    r = np.full(len(df), np.nan)

    if valid.any():

        xv = X.loc[valid]

        p1 = models["logistic"].predict_proba(xv)[:, 1]
        p2 = models["gb_classifier"].predict_proba(xv)[:, 1]

        r1 = models["ridge"].predict(xv)
        r2 = models["gb_regressor"].predict(xv)

        # Ensemble.
        p[valid.to_numpy()] = (
            0.50 * p1 +
            0.50 * p2
        )

        r[valid.to_numpy()] = (
            0.50 * r1 +
            0.50 * r2
        )

    return p, r


# ============================================================
# DATA SPLIT
# ============================================================

def chronological_split(data: pd.DataFrame):

    dates = np.sort(
        data["date"].dropna().unique()
    )

    n = len(dates)

    dev_end = dates[
        int(n * 0.50)
    ]

    val_end = dates[
        int(n * 0.75)
    ]

    development = data[
        data["date"] < dev_end
    ].copy()

    validation = data[
        (data["date"] >= dev_end) &
        (data["date"] < val_end)
    ].copy()

    oos = data[
        data["date"] >= val_end
    ].copy()

    return (
        development,
        validation,
        oos
    )


# ============================================================
# PURGED TRAINING
# ============================================================

def purge_training(
    train: pd.DataFrame,
    prediction_date
):

    cutoff = pd.Timestamp(
        prediction_date
    ) - pd.Timedelta(
        days=PURGE_DAYS
    )

    return train[
        train["date"] < cutoff
    ].copy()


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward(
    development: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    horizon: int
):

    evaluation_dates = np.sort(
        evaluation["date"].unique()
    )

    outputs = []

    models = None

    for i, date in enumerate(
        evaluation_dates
    ):

        if (
            models is None
            or i % RETRAIN_EVERY == 0
        ):

            train = development[
                development["date"] < date
            ].copy()

            train = purge_training(
                train,
                date
            )

            if len(train) < MIN_TRAIN:
                continue

            try:

                models = fit_models(
                    train,
                    features,
                    horizon
                )

            except Exception as e:

                print(
                    f"WARNING: model fit failed "
                    f"on {date}: {e}"
                )

                continue

        day = evaluation[
            evaluation["date"] == date
        ].copy()

        if day.empty:
            continue

        p, r = predict_models(
            models,
            day,
            features
        )

        day["pred_probability"] = p
        day["pred_return"] = r

        day = day.dropna(
            subset=[
                "pred_probability",
                "pred_return"
            ]
        )

        if not day.empty:
            outputs.append(day)

    if not outputs:
        return pd.DataFrame()

    return pd.concat(
        outputs,
        ignore_index=True
    )


# ============================================================
# GEMINI HYBRID
# ============================================================

def add_hybrid_predictions(
    df: pd.DataFrame,
    news_weight: float
):

    out = df.copy()

    q_p = out["pred_probability"].clip(
        0.001,
        0.999
    )

    # Convert probability to log-odds.
    q_logit = np.log(
        q_p / (1 - q_p)
    )

    # Gemini score is mapped to directional probability.
    g_score = out["gemini_score"].clip(
        -1,
        1
    )

    g_prob = 0.5 + 0.5 * g_score

    g_prob = g_prob.clip(
        0.001,
        0.999
    )

    g_logit = np.log(
        g_prob / (1 - g_prob)
    )

    # Convex logit combination.
    hybrid_logit = (
        (1 - news_weight) * q_logit +
        news_weight * g_logit
    )

    out["hybrid_probability"] = sigmoid(
        hybrid_logit
    )

    # Historical Gemini should influence expected return
    # conservatively rather than mechanically.
    #
    # The factor below is deliberately small. Validation
    # decides whether the weight is useful.
    news_return_signal = (
        0.005 * g_score
    )

    out["hybrid_return"] = (
        (1 - news_weight) *
        out["pred_return"] +
        news_weight *
        (
            out["pred_return"] +
            news_return_signal
        )
    )

    return out


# ============================================================
# VALIDATION WEIGHT SELECTION
# ============================================================

def select_news_weight(
    validation: pd.DataFrame,
    horizon: int
):

    if validation.empty:
        return 0.0

    has_news = (
        validation["gemini_available"].sum()
        > 0
    )

    if not has_news:
        return 0.0

    best_weight = 0.0
    best_score = -np.inf

    for w in NEWS_WEIGHTS:

        x = add_hybrid_predictions(
            validation,
            float(w)
        )

        signal = (
            (x["hybrid_probability"] >= TRADE_PMIN) &
            (x["hybrid_return"] >= TRADE_RMIN)
        )

        selected = x.loc[signal]

        if len(selected) < 20:
            continue

        ret = (
            selected[f"target_{horizon}d"]
            - ROUND_TRIP_COST
        )

        avg = ret.mean()

        win = (
            ret > 0
        ).mean()

        # Stability-oriented validation objective.
        score = (
            avg *
            math.sqrt(len(selected))
        )

        if (
            np.isfinite(score)
            and score > best_score
        ):

            best_score = score
            best_weight = float(w)

    return best_weight


# ============================================================
# PERFORMANCE
# ============================================================

def performance_table(
    df: pd.DataFrame,
    horizon: int,
    probability_col: str,
    return_col: str,
    model_name: str
):

    target = f"target_{horizon}d"

    x = df[
        [
            target,
            probability_col,
            return_col
        ]
    ].dropna()

    if x.empty:
        return None

    actual_up = (
        x[target] > 0
    ).astype(int)

    p = x[probability_col].clip(
        0.001,
        0.999
    )

    pred_r = x[return_col]

    directional_accuracy = (
        ((p >= 0.5) == (actual_up == 1))
        .mean()
    )

    try:
        brier = brier_score_loss(
            actual_up,
            p
        )
    except Exception:
        brier = np.nan

    try:
        ll = log_loss(
            actual_up,
            p,
            labels=[0, 1]
        )
    except Exception:
        ll = np.nan

    mae = mean_absolute_error(
        x[target],
        pred_r
    )

    signal = (
        (p >= TRADE_PMIN) &
        (pred_r >= TRADE_RMIN)
    )

    selected = x.loc[signal]

    if selected.empty:

        selected_win = np.nan
        selected_avg = np.nan
        pf = np.nan
        n_selected = 0

    else:

        net = (
            selected[target]
            - ROUND_TRIP_COST
        )

        selected_win = (
            net > 0
        ).mean()

        selected_avg = net.mean()

        gains = net[net > 0].sum()
        losses = -net[net < 0].sum()

        pf = (
            gains / losses
            if losses > 0
            else np.inf
        )

        n_selected = len(selected)

    return {
        "model": model_name,
        "horizon": horizon,
        "observations": len(x),
        "directional_accuracy":
            directional_accuracy,
        "brier_score": brier,
        "log_loss": ll,
        "return_mae": mae,
        "mean_predicted_return":
            pred_r.mean(),
        "mean_actual_return":
            x[target].mean(),
        "selected_n_p>=55":
            n_selected,
        "selected_win_rate":
            selected_win,
        "selected_average_net_return":
            selected_avg,
        "selected_profit_factor":
            pf,
    }


# ============================================================
# NON-OVERLAPPING TRADE TEST
# ============================================================

def nonoverlap_test(
    df: pd.DataFrame,
    horizon: int,
    probability_col: str,
    return_col: str
):

    x = df.sort_values(
        ["date", "ticker"]
    ).copy()

    x = x[
        (x[probability_col] >= TRADE_PMIN) &
        (x[return_col] >= TRADE_RMIN)
    ].copy()

    if x.empty:
        return None

    accepted = []

    last_date = None

    for _, row in x.iterrows():

        d = pd.Timestamp(
            row["date"]
        )

        if (
            last_date is None
            or d >= last_date +
            pd.Timedelta(days=horizon + 1)
        ):

            accepted.append(row)
            last_date = d

    if not accepted:
        return None

    z = pd.DataFrame(
        accepted
    )

    net = (
        z[f"target_{horizon}d"]
        - ROUND_TRIP_COST
    )

    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()

    pf = (
        gains / losses
        if losses > 0
        else np.inf
    )

    return {
        "trades": len(z),
        "win_rate":
            (net > 0).mean(),
        "average_net":
            net.mean(),
        "median_net":
            net.median(),
        "best":
            net.max(),
        "worst":
            net.min(),
        "gross_sum_return":
            z[f"target_{horizon}d"].sum(),
        "net_sum_return":
            net.sum(),
        "profit_factor":
            pf,
    }


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def portfolio_test(
    df: pd.DataFrame,
    horizon: int,
    probability_col: str,
    return_col: str
):

    x = df.copy()

    x = x[
        (x[probability_col] >= TRADE_PMIN) &
        (x[return_col] >= TRADE_RMIN)
    ].copy()

    if x.empty:
        return None

    # One best candidate per date.
    x["score"] = (
        x[probability_col]
        * x[return_col].clip(lower=0)
    )

    x = (
        x.sort_values(
            ["date", "score"],
            ascending=[True, False]
        )
        .groupby("date")
        .head(1)
    )

    capital = 100000.0

    equity = [capital]

    completed = 0

    for _, row in x.iterrows():

        gross = safe_float(
            row[f"target_{horizon}d"],
            0.0
        )

        net = gross - ROUND_TRIP_COST

        capital *= (
            1 + net
        )

        equity.append(capital)

        completed += 1

    eq = np.array(equity)

    total_return = (
        capital / 100000.0 - 1
    )

    trading_dates = len(x)

    if trading_dates > 0:

        years = (
            trading_dates / 252
        )

        cagr = (
            (capital / 100000.0)
            ** (1 / years)
            - 1
            if years > 0
            else np.nan
        )

    else:
        cagr = np.nan

    running_max = np.maximum.accumulate(eq)

    drawdown = (
        eq / running_max - 1
    )

    max_drawdown = drawdown.min()

    daily_returns = pd.Series(
        eq
    ).pct_change().dropna()

    if (
        len(daily_returns) > 1
        and daily_returns.std() > 0
    ):

        sharpe = (
            daily_returns.mean() /
            daily_returns.std()
        ) * np.sqrt(252)

    else:

        sharpe = np.nan

    return {
        "starting_capital":
            100000.0,
        "ending_equity":
            capital,
        "total_return":
            total_return,
        "CAGR":
            cagr,
        "max_drawdown":
            max_drawdown,
        "Sharpe":
            sharpe,
        "completed_trades":
            completed,
    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def run_backtest():

    print("=" * 78)
    print(
        f"{VERSION} — "
        "LEAKAGE-PROOF QUANT + GEMINI EXPERIMENT"
    )
    print("=" * 78)

    print(
        f"Source revision: {SOURCE_REVISION}"
    )

    print(
        f"yfinance version: {yf.__version__}"
    )

    print(
        f"Backtest period: {YEARS}y"
    )

    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST:.3%}"
    )

    news = load_historical_gemini()

    data = build_dataset(
        news
    )

    leakage_check()

    print()
    print("DATASET")

    print(
        f"Total candidate observations: "
        f"{len(data):,}"
    )

    print(
        f"Successful symbols: "
        f"{data['ticker'].nunique()}"
    )

    print(
        f"Unique signal dates: "
        f"{data['date'].nunique()}"
    )

    coverage = (
        data["gemini_available"].mean()
        if "gemini_available" in data
        else 0
    )

    print(
        f"Historical Gemini coverage: "
        f"{coverage:.2%}"
    )

    development, validation, oos = (
        chronological_split(data)
    )

    print(
        f"Development observations: "
        f"{len(development):,}"
    )

    print(
        f"Validation observations: "
        f"{len(validation):,}"
    )

    print(
        f"OOS observations: "
        f"{len(oos):,}"
    )

    print(
        f"Development end: "
        f"{development['date'].max()}"
    )

    print(
        f"Validation end: "
        f"{validation['date'].max()}"
    )

    print(
        f"OOS start: "
        f"{oos['date'].min()}"
    )

    # --------------------------------------------------------
    # OOS COMPARISON
    # --------------------------------------------------------

    all_results = []
    nonoverlap_results = []
    portfolio_results = []

    for horizon in HORIZONS:

        print()
        print(
            "=" * 60
        )

        print(
            f"HORIZON {horizon}D"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # QUANT WALK FORWARD
        # ----------------------------------------------------

        quant_oos = walk_forward(
            development,
            oos,
            QUANT_FEATURES,
            horizon
        )

        if quant_oos.empty:

            print(
                "WARNING: no OOS quant predictions."
            )

            continue

        # ----------------------------------------------------
        # VALIDATION WEIGHT
        # ----------------------------------------------------

        quant_val = walk_forward(
            development,
            validation,
            QUANT_FEATURES,
            horizon
        )

        if quant_val.empty:

            selected_weight = 0.0

        else:

            selected_weight = (
                select_news_weight(
                    quant_val,
                    horizon
                )
            )

        print(
            f"Selected historical Gemini weight: "
            f"{selected_weight:.2f}"
        )

        # ----------------------------------------------------
        # QUANT RESULT
        # ----------------------------------------------------

        result = performance_table(
            quant_oos,
            horizon,
            "pred_probability",
            "pred_return",
            "quant"
        )

        if result:
            all_results.append(
                result
            )

        no = nonoverlap_test(
            quant_oos,
            horizon,
            "pred_probability",
            "pred_return"
        )

        if no:

            no["model"] = "quant"
            no["horizon"] = horizon

            nonoverlap_results.append(no)

        po = portfolio_test(
            quant_oos,
            horizon,
            "pred_probability",
            "pred_return"
        )

        if po:

            po["model"] = "quant"
            po["horizon"] = horizon

            portfolio_results.append(po)

        # ----------------------------------------------------
        # GEMINI / HYBRID
        # ----------------------------------------------------

        has_news = (
            quant_oos["gemini_available"].sum()
            > 0
        )

        if not has_news:

            print(
                "Gemini experiment: NOT RUN "
                "for this horizon — no historical scores."
            )

            continue

        hybrid_oos = add_hybrid_predictions(
            quant_oos,
            selected_weight
        )

        # Gemini-only probability
        hybrid_oos["gemini_probability"] = (
            0.5 +
            0.5 *
            hybrid_oos["gemini_score"]
        )

        # Gemini-only return signal
        hybrid_oos["gemini_return"] = (
            0.005 *
            hybrid_oos["gemini_score"]
        )

        # ----------------------------------------------------
        # GEMINI ONLY
        # ----------------------------------------------------

        result = performance_table(
            hybrid_oos,
            horizon,
            "gemini_probability",
            "gemini_return",
            "gemini_only"
        )

        if result:
            all_results.append(
                result
            )

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        result = performance_table(
            hybrid_oos,
            horizon,
            "hybrid_probability",
            "hybrid_return",
            "hybrid"
        )

        if result:
            all_results.append(
                result
            )

        no = nonoverlap_test(
            hybrid_oos,
            horizon,
            "hybrid_probability",
            "hybrid_return"
        )

        if no:

            no["model"] = "hybrid"
            no["horizon"] = horizon

            nonoverlap_results.append(no)

        po = portfolio_test(
            hybrid_oos,
            horizon,
            "hybrid_probability",
            "hybrid_return"
        )

        if po:

            po["model"] = "hybrid"
            po["horizon"] = horizon

            portfolio_results.append(po)

        # Save horizon-level OOS file.
        hybrid_oos.to_csv(
            AUDIT_DIR /
            f"v6_5_3_oos_h{horizon}.csv",
            index=False
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    if not results_df.empty:

        results_df.to_csv(
            AUDIT_DIR /
            "v6_5_3_oos_model_comparison.csv",
            index=False
        )

    nonoverlap_df = pd.DataFrame(
        nonoverlap_results
    )

    if not nonoverlap_df.empty:

        nonoverlap_df.to_csv(
            AUDIT_DIR /
            "v6_5_3_nonoverlap_oos.csv",
            index=False
        )

    portfolio_df = pd.DataFrame(
        portfolio_results
    )

    if not portfolio_df.empty:

        portfolio_df.to_csv(
            AUDIT_DIR /
            "v6_5_3_portfolio_oos.csv",
            index=False
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 78)
    print(
        "V6.5.3 OOS MODEL COMPARISON"
    )
    print("=" * 78)

    if results_df.empty:

        print(
            "No valid OOS results generated."
        )

    else:

        print(
            results_df.to_string(
                index=False
            )
        )

    print()
    print(
        "NON-OVERLAPPING OOS TRADE TEST"
    )

    if nonoverlap_df.empty:

        print(
            "No qualifying non-overlapping trades."
        )

    else:

        print(
            nonoverlap_df.to_string(
                index=False
            )
        )

    print()
    print(
        "PORTFOLIO OOS TEST"
    )

    if portfolio_df.empty:

        print(
            "No qualifying portfolio trades."
        )

    else:

        print(
            portfolio_df.to_string(
                index=False
            )
        )

    # ========================================================
    # FINAL AUDIT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "V6.5.3 BACKTEST COMPLETED"
    )
    print("=" * 78)

    print(
        "AUDIT GUARANTEES:"
    )

    print(
        "1. OHLCV features are backward-looking."
    )

    print(
        "2. Forward returns are targets only."
    )

    print(
        "3. Training observations are purged "
        f"by {PURGE_DAYS} days."
    )

    print(
        "4. Validation determines Gemini weight."
    )

    print(
        "5. OOS observations are not used "
        "for model/weight selection."
    )

    print(
        "6. Current Gemini calls are NEVER "
        "used as historical information."
    )

    print(
        "7. Historical Gemini timestamps must "
        "precede or equal the signal date."
    )

    print(
        "8. Non-overlapping and portfolio tests "
        "are reported separately."
    )

    print(
        "9. Historical performance does not "
        "guarantee future returns."
    )

    print()
    print(
        "IMPORTANT: A Gemini improvement is only "
        "credible if HYBRID beats QUANT on untouched "
        "OOS data with sufficient observations and "
        "reasonable stability."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_backtest()
```
