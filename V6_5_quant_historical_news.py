# V6_5_quant_historical_news.py
#
# V6.5.3 — LEAKAGE-PROOF QUANT + HISTORICAL GEMINI EXPERIMENT
#
# IMPORTANT:
# This file intentionally does NOT call Gemini during the historical
# backtest. A historical Gemini experiment requires timestamped
# Gemini/news scores that were generated from information available
# at that historical time.
#
# Optional historical file:
#   historical_gemini.csv
#
# Required columns:
#   ticker,published_at,gemini_score
#
# Example:
#   RELIANCE.NS,2025-01-15 08:30:00,0.42
#
# gemini_score must be between -1 and +1.
#
# If the file is absent, the program performs a QUANT-ONLY baseline.
# It will NOT pretend that Gemini was tested.

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "V6.5.3"

SOURCE_REVISION = (
    "2026-08-31-LEAKAGE-PROOF-HYBRID-FINAL"
)

YEARS = 6

ROUND_TRIP_COST = 0.003

RANDOM_STATE = 42

MIN_TRAIN_OBS = 1000

# Five trading-day purge.
# We use 10 calendar days to safely cover weekends/holidays.
PURGE_CALENDAR_DAYS = 10

# Retrain every 20 evaluation dates.
RETRAIN_EVERY = 20

TRADE_PROBABILITY = 0.55

MIN_EXPECTED_RETURN = 0.002

HORIZONS = [1, 3, 5, 10]

# Gemini weights are selected ONLY on validation.
NEWS_WEIGHTS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


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
# BASIC UTILITIES
# ============================================================

def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return default


# ============================================================
# ROBUST YFINANCE OHLCV HANDLING
# ============================================================

def normalise_yfinance_columns(df):
    """
    Handles both normal yfinance DataFrames and MultiIndex
    DataFrames.

    This specifically prevents the earlier:
        'Series' object has no attribute 'Close'
    failure.
    """

    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    x = df.copy()

    wanted = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ]

    # --------------------------------------------------------
    # MultiIndex columns
    # --------------------------------------------------------

    if isinstance(x.columns, pd.MultiIndex):

        flattened = []

        for col in x.columns:

            pieces = [
                str(part)
                for part in col
            ]

            selected = None

            for part in pieces:
                if part in wanted:
                    selected = part
                    break

            if selected is None:
                selected = pieces[-1]

            flattened.append(selected)

        x.columns = flattened

        # If duplicate Close/Open/etc. columns exist,
        # keep the first one.
        x = x.loc[
            :,
            ~x.columns.duplicated()
        ]

    else:

        x.columns = [
            str(c)
            for c in x.columns
        ]

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        c for c in required
        if c not in x.columns
    ]

    if missing:
        return pd.DataFrame()

    for col in required:

        series = x[col]

        # Extremely defensive:
        # if duplicate handling somehow left a DataFrame,
        # use its first column.
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        x[col] = pd.to_numeric(
            series,
            errors="coerce"
        )

    x = x[
        required
    ].copy()

    x = x.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    return x


def download_history(ticker):

    try:

        raw = yf.download(
            ticker,
            period=f"{YEARS}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        df = normalise_yfinance_columns(
            raw
        )

        if df.empty:
            return pd.DataFrame()

        dates = pd.to_datetime(
            df.index,
            errors="coerce"
        )

        df = df.copy()

        df["date"] = dates

        df = df.dropna(
            subset=["date"]
        )

        df = df.sort_values(
            "date"
        )

        df = df.drop_duplicates(
            subset=["date"],
            keep="last"
        )

        df = df.reset_index(
            drop=True
        )

        return df

    except Exception as exc:

        print(
            f"WARNING: {ticker} failed: {exc}"
        )

        return pd.DataFrame()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(
    close,
    period=14
):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    result = (
        100 -
        100 / (1 + rs)
    )

    return result


def calculate_atr(
    df,
    period=14
):

    previous_close = (
        df["Close"].shift(1)
    )

    tr1 = (
        df["High"] -
        df["Low"]
    )

    tr2 = (
        df["High"] -
        previous_close
    ).abs()

    tr3 = (
        df["Low"] -
        previous_close
    ).abs()

    tr = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        period
    ).mean()


# ============================================================
# FEATURE ENGINEERING
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


def make_features(df):

    x = df.copy()

    close = x["Close"]
    volume = x["Volume"]

    # --------------------------------------------------------
    # Historical returns
    # --------------------------------------------------------

    x["ret_1"] = (
        close.pct_change(1)
    )

    x["ret_3"] = (
        close.pct_change(3)
    )

    x["ret_5"] = (
        close.pct_change(5)
    )

    x["ret_10"] = (
        close.pct_change(10)
    )

    x["ret_20"] = (
        close.pct_change(20)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    daily_return = (
        close.pct_change()
    )

    x["vol_5"] = (
        daily_return
        .rolling(5)
        .std()
    )

    x["vol_10"] = (
        daily_return
        .rolling(10)
        .std()
    )

    x["vol_20"] = (
        daily_return
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    x["rsi_7"] = calculate_rsi(
        close,
        7
    )

    x["rsi_14"] = calculate_rsi(
        close,
        14
    )

    x["rsi_21"] = calculate_rsi(
        close,
        21
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    x["atr_14"] = calculate_atr(
        x,
        14
    )

    x["atr_pct"] = (
        x["atr_14"] /
        close
    )

    # --------------------------------------------------------
    # EMA structure
    # --------------------------------------------------------

    ema10 = (
        close
        .ewm(
            span=10,
            adjust=False
        )
        .mean()
    )

    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    ema50 = (
        close
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    x["ema10_dist"] = (
        close / ema10 - 1
    )

    x["ema20_dist"] = (
        close / ema20 - 1
    )

    x["ema50_dist"] = (
        close / ema50 - 1
    )

    x["ema10_20"] = (
        ema10 / ema20 - 1
    )

    x["ema20_50"] = (
        ema20 / ema50 - 1
    )

    # --------------------------------------------------------
    # Breakouts
    #
    # IMPORTANT:
    # shift(1) means today's feature cannot use today's
    # future information.
    # --------------------------------------------------------

    high20 = (
        x["High"]
        .rolling(20)
        .max()
    )

    low20 = (
        x["Low"]
        .rolling(20)
        .min()
    )

    x["breakout_20"] = (
        close /
        high20.shift(1) - 1
    )

    x["breakdown_20"] = (
        close /
        low20.shift(1) - 1
    )

    # --------------------------------------------------------
    # Volume anomaly
    # --------------------------------------------------------

    volume_mean = (
        volume
        .rolling(20)
        .mean()
    )

    volume_std = (
        volume
        .rolling(20)
        .std()
    )

    x["volume_z"] = (
        (volume - volume_mean) /
        volume_std.replace(
            0,
            np.nan
        )
    )

    # --------------------------------------------------------
    # Candle/range structure
    # --------------------------------------------------------

    x["range_pct"] = (
        (x["High"] - x["Low"]) /
        close
    )

    candle_range = (
        x["High"] - x["Low"]
    ).replace(
        0,
        np.nan
    )

    x["close_location"] = (
        (close - x["Low"]) /
        candle_range
    )

    # --------------------------------------------------------
    # Momentum acceleration
    # --------------------------------------------------------

    x["momentum_accel"] = (
        x["ret_5"] -
        x["ret_20"] / 4
    )

    return x


# ============================================================
# TARGET ENGINEERING
# ============================================================

def add_targets(df):

    x = df.copy()

    for horizon in HORIZONS:

        future_return = (
            x["Close"].shift(-horizon) /
            x["Close"] -
            1
        )

        x[
            f"target_{horizon}d"
        ] = future_return

        x[
            f"up_{horizon}d"
        ] = (
            future_return > 0
        ).astype(float)

    return x


# ============================================================
# HISTORICAL GEMINI DATA
# ============================================================

def load_historical_gemini():

    candidates = [
        Path("historical_gemini.csv"),
        Path("historical_news.csv"),
        Path("data/historical_gemini.csv"),
        Path("data/historical_news.csv"),
    ]

    selected = None

    for path in candidates:

        if path.exists():

            selected = path
            break

    if selected is None:

        print(
            "HISTORICAL GEMINI: NOT FOUND — "
            "QUANT-ONLY BASELINE"
        )

        return pd.DataFrame(
            columns=[
                "ticker",
                "published_at",
                "gemini_score",
            ]
        )

    news = pd.read_csv(
        selected
    )

    required = {
        "ticker",
        "published_at",
        "gemini_score",
    }

    missing = (
        required -
        set(news.columns)
    )

    if missing:

        raise ValueError(
            "Historical Gemini file is missing "
            f"columns: {sorted(missing)}"
        )

    news = news.copy()

    news["ticker"] = (
        news["ticker"]
        .astype(str)
        .str.strip()
    )

    news["published_at"] = (
        pd.to_datetime(
            news["published_at"],
            errors="coerce",
            utc=True,
        )
    )

    news["gemini_score"] = (
        pd.to_numeric(
            news["gemini_score"],
            errors="coerce",
        )
    )

    news = news.dropna(
        subset=[
            "ticker",
            "published_at",
            "gemini_score",
        ]
    )

    news["gemini_score"] = (
        news["gemini_score"]
        .clip(-1, 1)
    )

    news = news[
        news["ticker"].isin(
            TICKERS
        )
    ]

    news = news.sort_values(
        [
            "ticker",
            "published_at",
        ]
    )

    print(
        f"HISTORICAL GEMINI: FOUND — "
        f"{len(news):,} timestamped records"
    )

    return news


# ============================================================
# ATTACH HISTORICAL GEMINI
# ============================================================

def attach_gemini(
    market,
    news
):

    market = market.copy()

    if news.empty:

        market["gemini_score"] = 0.0

        market[
            "gemini_available"
        ] = 0

        return market

    market["signal_timestamp"] = (
        pd.to_datetime(
            market["date"],
            errors="coerce",
            utc=True,
        )
    )

    news = news.copy()

    news = news.sort_values(
        [
            "ticker",
            "published_at",
        ]
    )

    market = market.sort_values(
        [
            "ticker",
            "signal_timestamp",
        ]
    )

    # --------------------------------------------------------
    # CRITICAL LEAKAGE RULE
    #
    # direction="backward" means only a Gemini/news score
    # published BEFORE the signal can be used.
    #
    # tolerance limits the news lookback to seven days.
    # --------------------------------------------------------

    merged = pd.merge_asof(
        market,
        news,
        left_on="signal_timestamp",
        right_on="published_at",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(
            days=7
        ),
    )

    merged["gemini_available"] = (
        merged["published_at"]
        .notna()
        .astype(int)
    )

    merged["gemini_score"] = (
        pd.to_numeric(
            merged["gemini_score"],
            errors="coerce",
        )
        .fillna(0.0)
        .clip(-1, 1)
    )

    # Explicit timestamp audit.
    invalid = (
        merged["published_at"].notna()
        &
        (
            merged["published_at"] >
            merged["signal_timestamp"]
        )
    )

    if invalid.any():

        raise RuntimeError(
            "LEAKAGE FAILURE: historical Gemini "
            "timestamp occurs after signal timestamp."
        )

    return merged


# ============================================================
# BUILD COMPLETE DATASET
# ============================================================

def build_dataset(news):

    all_frames = []

    print(
        f"Loading [{len(TICKERS)}] symbols..."
    )

    for index, ticker in enumerate(
        TICKERS,
        start=1,
    ):

        print(
            f"Loading [{index}/{len(TICKERS)}] "
            f"{ticker}"
        )

        raw = download_history(
            ticker
        )

        if raw.empty:

            print(
                f"WARNING: {ticker} "
                "has no usable history; skipping."
            )

            continue

        if len(raw) < 300:

            print(
                f"WARNING: insufficient history "
                f"for {ticker}; skipping."
            )

            continue

        frame = make_features(
            raw
        )

        frame = add_targets(
            frame
        )

        frame["ticker"] = ticker

        all_frames.append(
            frame
        )

    if not all_frames:

        raise RuntimeError(
            "No valid OHLCV datasets were created."
        )

    data = pd.concat(
        all_frames,
        ignore_index=True
    )

    data = attach_gemini(
        data,
        news
    )

    data = data.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    return data


# ============================================================
# LEAKAGE AUDIT
# ============================================================

def leakage_check(data):

    forbidden = [
        "target_",
        "up_",
        "future",
        "forward",
    ]

    for feature in QUANT_FEATURES:

        low = feature.lower()

        for term in forbidden:

            if term in low:

                raise RuntimeError(
                    "FEATURE/TARGET LEAKAGE FAILURE: "
                    f"{feature}"
                )

    if "published_at" in data.columns:

        bad = (
            data["published_at"].notna()
            &
            (
                data["published_at"] >
                data["signal_timestamp"]
            )
        )

        if bad.any():

            raise RuntimeError(
                "HISTORICAL GEMINI TIMESTAMP LEAKAGE."
            )

    print(
        "FEATURE/TARGET LEAKAGE CHECK: PASS"
    )

    print(
        "Forward-return targets are excluded "
        "from FEATURES."
    )

    print(
        "Historical Gemini/news timestamps must be "
        "<= signal timestamp."
    )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_data(data):

    dates = np.sort(
        data["date"]
        .dropna()
        .unique()
    )

    if len(dates) < 300:

        raise RuntimeError(
            "Not enough unique dates for "
            "chronological split."
        )

    n = len(dates)

    dev_cut = dates[
        int(n * 0.50)
    ]

    validation_cut = dates[
        int(n * 0.75)
    ]

    development = data[
        data["date"] < dev_cut
    ].copy()

    validation = data[
        (data["date"] >= dev_cut)
        &
        (data["date"] < validation_cut)
    ].copy()

    oos = data[
        data["date"] >= validation_cut
    ].copy()

    return (
        development,
        validation,
        oos,
    )


# ============================================================
# PURGING
# ============================================================

def purge_training(
    training,
    evaluation_date,
):

    cutoff = (
        pd.Timestamp(
            evaluation_date
        )
        -
        pd.Timedelta(
            days=PURGE_CALENDAR_DAYS
        )
    )

    return training[
        training["date"] < cutoff
    ].copy()


# ============================================================
# PREPARE TRAINING MATRIX
# ============================================================

def prepare_training(
    df,
    features,
    horizon,
):

    required = (
        features +
        [
            f"target_{horizon}d",
            f"up_{horizon}d",
        ]
    )

    x = df[
        required
    ].copy()

    x = x.replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = (
        x.notna()
        .all(axis=1)
    )

    x = x.loc[
        valid
    ]

    X = x[
        features
    ]

    y_return = x[
        f"target_{horizon}d"
    ]

    y_direction = x[
        f"up_{horizon}d"
    ].astype(int)

    return (
        X,
        y_direction,
        y_return,
    )


# ============================================================
# FIT ENSEMBLE
# ============================================================

def fit_ensemble(
    training,
    features,
    horizon,
):

    X, y_direction, y_return = (
        prepare_training(
            training,
            features,
            horizon,
        )
    )

    if len(X) < MIN_TRAIN_OBS:

        raise ValueError(
            "Insufficient training observations: "
            f"{len(X)}"
        )

    # --------------------------------------------------------
    # Logistic classifier
    # --------------------------------------------------------

    logistic = Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    max_iter=1500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic.fit(
        X,
        y_direction
    )

    # --------------------------------------------------------
    # Gradient boosting classifier
    # --------------------------------------------------------

    gb_classifier = (
        HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    )

    gb_classifier.fit(
        X,
        y_direction
    )

    # --------------------------------------------------------
    # Ridge return model
    # --------------------------------------------------------

    ridge = Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                Ridge(
                    alpha=10.0
                ),
            ),
        ]
    )

    ridge.fit(
        X,
        y_return
    )

    # --------------------------------------------------------
    # Gradient boosting return model
    # --------------------------------------------------------

    gb_return = (
        HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
            loss="squared_error",
        )
    )

    gb_return.fit(
        X,
        y_return
    )

    return {
        "logistic": logistic,
        "gb_classifier": gb_classifier,
        "ridge": ridge,
        "gb_return": gb_return,
    }


# ============================================================
# PREDICT
# ============================================================

def predict_ensemble(
    models,
    df,
    features,
):

    X = df[
        features
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = (
        X.notna()
        .all(axis=1)
    )

    probability = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    expected_return = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    if valid.any():

        xv = X.loc[
            valid
        ]

        p1 = (
            models["logistic"]
            .predict_proba(xv)[:, 1]
        )

        p2 = (
            models["gb_classifier"]
            .predict_proba(xv)[:, 1]
        )

        r1 = (
            models["ridge"]
            .predict(xv)
        )

        r2 = (
            models["gb_return"]
            .predict(xv)
        )

        probability[
            valid.to_numpy()
        ] = (
            0.50 * p1 +
            0.50 * p2
        )

        expected_return[
            valid.to_numpy()
        ] = (
            0.50 * r1 +
            0.50 * r2
        )

    return (
        probability,
        expected_return,
    )


# ============================================================
# WALK-FORWARD ENGINE
# ============================================================

def walk_forward(
    development,
    evaluation,
    features,
    horizon,
):

    evaluation_dates = np.sort(
        evaluation["date"]
        .dropna()
        .unique()
    )

    outputs = []

    models = None

    for position, date in enumerate(
        evaluation_dates
    ):

        if (
            models is None
            or
            position % RETRAIN_EVERY == 0
        ):

            training = development[
                development["date"] < date
            ].copy()

            training = purge_training(
                training,
                date
            )

            if len(training) < MIN_TRAIN_OBS:

                continue

            try:

                models = fit_ensemble(
                    training,
                    features,
                    horizon,
                )

            except Exception as exc:

                print(
                    f"WARNING: fit failed "
                    f"for {date}: {exc}"
                )

                continue

        day = evaluation[
            evaluation["date"] == date
        ].copy()

        if day.empty:
            continue

        p, r = predict_ensemble(
            models,
            day,
            features,
        )

        day[
            "pred_probability"
        ] = p

        day[
            "pred_return"
        ] = r

        day = day.dropna(
            subset=[
                "pred_probability",
                "pred_return",
            ]
        )

        if not day.empty:

            outputs.append(
                day
            )

    if not outputs:

        return pd.DataFrame()

    return pd.concat(
        outputs,
        ignore_index=True
    )


# ============================================================
# GEMINI/HYBRID PREDICTION
# ============================================================

def create_hybrid(
    df,
    news_weight,
):

    x = df.copy()

    quant_probability = (
        x["pred_probability"]
        .clip(
            0.001,
            0.999
        )
    )

    quant_logit = np.log(
        quant_probability /
        (1 - quant_probability)
    )

    gemini_score = (
        x["gemini_score"]
        .clip(-1, 1)
    )

    gemini_probability = (
        0.5 +
        0.5 *
        gemini_score
    )

    gemini_probability = (
        gemini_probability
        .clip(
            0.001,
            0.999
        )
    )

    gemini_logit = np.log(
        gemini_probability /
        (1 - gemini_probability)
    )

    hybrid_logit = (
        (1 - news_weight) *
        quant_logit
        +
        news_weight *
        gemini_logit
    )

    x[
        "hybrid_probability"
    ] = sigmoid(
        hybrid_logit
    )

    # Conservative Gemini return contribution.
    #
    # Validation decides whether Gemini should have
    # any weight at all.
    gemini_return_signal = (
        0.005 *
        gemini_score
    )

    x[
        "hybrid_return"
    ] = (
        (1 - news_weight) *
        x["pred_return"]
        +
        news_weight *
        (
            x["pred_return"]
            +
            gemini_return_signal
        )
    )

    x[
        "gemini_probability"
    ] = gemini_probability

    x[
        "gemini_return"
    ] = gemini_return_signal

    return x


# ============================================================
# VALIDATION GEMINI WEIGHT SELECTION
# ============================================================

def select_gemini_weight(
    validation,
    horizon,
):

    if validation.empty:
        return 0.0

    available = (
        validation[
            "gemini_available"
        ] == 1
    )

    if available.sum() < 50:

        print(
            "Historical Gemini coverage too low "
            "for reliable weight selection."
        )

        return 0.0

    best_weight = 0.0
    best_score = -np.inf

    for weight in NEWS_WEIGHTS:

        candidate = create_hybrid(
            validation,
            weight
        )

        candidate = candidate[
            candidate[
                "gemini_available"
            ] == 1
        ]

        signal = (
            (
                candidate[
                    "hybrid_probability"
                ]
                >= TRADE_PROBABILITY
            )
            &
            (
                candidate[
                    "hybrid_return"
                ]
                >= MIN_EXPECTED_RETURN
            )
        )

        selected = candidate[
            signal
        ]

        if len(selected) < 20:
            continue

        net = (
            selected[
                f"target_{horizon}d"
            ]
            -
            ROUND_TRIP_COST
        )

        average_return = (
            net.mean()
        )

        win_rate = (
            net > 0
        ).mean()

        # Stability-aware objective.
        score = (
            average_return *
            math.sqrt(
                len(selected)
            )
        )

        # Small penalty for extremely low win rate.
        if win_rate < 0.50:

            score *= 0.75

        if (
            np.isfinite(score)
            and
            score > best_score
        ):

            best_score = score
            best_weight = float(
                weight
            )

    return best_weight


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_performance(
    df,
    horizon,
    probability_column,
    return_column,
    model_name,
):

    target_column = (
        f"target_{horizon}d"
    )

    required = [
        target_column,
        probability_column,
        return_column,
    ]

    x = df[
        required
    ].dropna()

    if x.empty:
        return None

    actual_direction = (
        x[target_column] > 0
    ).astype(int)

    probability = (
        x[probability_column]
        .clip(
            0.001,
            0.999
        )
    )

    predicted_return = (
        x[return_column]
    )

    directional_accuracy = (
        (
            (
                probability >= 0.5
            )
            ==
            (
                actual_direction == 1
            )
        )
        .mean()
    )

    try:

        brier = brier_score_loss(
            actual_direction,
            probability
        )

    except Exception:

        brier = np.nan

    try:

        logloss = log_loss(
            actual_direction,
            probability,
            labels=[0, 1]
        )

    except Exception:

        logloss = np.nan

    return_mae = (
        mean_absolute_error(
            x[target_column],
            predicted_return
        )
    )

    signal = (
        (
            probability >=
            TRADE_PROBABILITY
        )
        &
        (
            predicted_return >=
            MIN_EXPECTED_RETURN
        )
    )

    selected = x[
        signal
    ]

    if selected.empty:

        selected_n = 0
        selected_win = np.nan
        selected_average = np.nan
        profit_factor = np.nan

    else:

        net = (
            selected[target_column]
            -
            ROUND_TRIP_COST
        )

        selected_n = len(
            selected
        )

        selected_win = (
            net > 0
        ).mean()

        selected_average = (
            net.mean()
        )

        gross_profit = (
            net[net > 0].sum()
        )

        gross_loss = (
            -net[net < 0].sum()
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit /
                gross_loss
            )

        elif gross_profit > 0:

            profit_factor = np.inf

        else:

            profit_factor = np.nan

    return {
        "model": model_name,
        "horizon": horizon,
        "observations": len(x),
        "directional_accuracy":
            directional_accuracy,
        "brier_score": brier,
        "log_loss": logloss,
        "return_mae": return_mae,
        "mean_predicted_return":
            predicted_return.mean(),
        "mean_actual_return":
            x[target_column].mean(),
        "selected_n_p>=55":
            selected_n,
        "selected_win_rate":
            selected_win,
        "selected_average_net_return":
            selected_average,
        "selected_profit_factor":
            profit_factor,
    }


# ============================================================
# NON-OVERLAPPING TRADE TEST
# ============================================================

def nonoverlapping_test(
    df,
    horizon,
    probability_column,
    return_column,
):

    x = df.copy()

    x = x[
        (
            x[probability_column]
            >= TRADE_PROBABILITY
        )
        &
        (
            x[return_column]
            >= MIN_EXPECTED_RETURN
        )
    ].copy()

    if x.empty:
        return None

    x = x.sort_values(
        [
            "date",
            "ticker",
        ]
    )

    accepted = []

    last_trade_date = None

    for _, row in x.iterrows():

        current_date = pd.Timestamp(
            row["date"]
        )

        if (
            last_trade_date is None
            or
            current_date >=
            (
                last_trade_date
                +
                pd.Timedelta(
                    days=horizon + 1
                )
            )
        ):

            accepted.append(
                row
            )

            last_trade_date = (
                current_date
            )

    if not accepted:
        return None

    selected = pd.DataFrame(
        accepted
    )

    net = (
        selected[
            f"target_{horizon}d"
        ]
        -
        ROUND_TRIP_COST
    )

    positive = (
        net[net > 0].sum()
    )

    negative = (
        -net[net < 0].sum()
    )

    if negative > 0:

        profit_factor = (
            positive /
            negative
        )

    else:

        profit_factor = np.inf

    return {
        "trades": len(selected),
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
        "profit_factor":
            profit_factor,
        "gross_sum_return":
            selected[
                f"target_{horizon}d"
            ].sum(),
        "net_sum_return":
            net.sum(),
    }


# ============================================================
# PORTFOLIO TEST
# ============================================================

def portfolio_test(
    df,
    horizon,
    probability_column,
    return_column,
):

    x = df.copy()

    x = x[
        (
            x[probability_column]
            >= TRADE_PROBABILITY
        )
        &
        (
            x[return_column]
            >= MIN_EXPECTED_RETURN
        )
    ].copy()

    if x.empty:
        return None

    x["selection_score"] = (
        x[probability_column] *
        x[return_column].clip(
            lower=0
        )
    )

    # One strongest candidate per signal date.
    x = (
        x.sort_values(
            [
                "date",
                "selection_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .groupby(
            "date",
            as_index=False
        )
        .head(1)
    )

    x = x.sort_values(
        "date"
    )

    capital = 100000.0

    equity_curve = [
        capital
    ]

    for _, row in x.iterrows():

        future_return = safe_float(
            row[
                f"target_{horizon}d"
            ],
            0.0
        )

        net_return = (
            future_return
            -
            ROUND_TRIP_COST
        )

        capital *= (
            1 + net_return
        )

        equity_curve.append(
            capital
        )

    equity = np.asarray(
        equity_curve,
        dtype=float
    )

    total_return = (
        capital /
        100000.0
        -
        1
    )

    trades = len(x)

    if trades > 0:

        years = (
            trades /
            252.0
        )

        if years > 0:

            cagr = (
                (
                    capital /
                    100000.0
                )
                **
                (
                    1 /
                    years
                )
                -
                1
            )

        else:

            cagr = np.nan

    else:

        cagr = np.nan

    running_max = np.maximum.accumulate(
        equity
    )

    drawdown = (
        equity /
        running_max
        -
        1
    )

    max_drawdown = (
        drawdown.min()
    )

    daily_returns = (
        pd.Series(
            equity
        )
        .pct_change()
        .dropna()
    )

    if (
        len(daily_returns) > 1
        and
        daily_returns.std() > 0
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
            trades,
    }


# ============================================================
# MAIN
# ============================================================

def run_backtest():

    print("=" * 78)

    print(
        f"{VERSION} — "
        "LEAKAGE-PROOF QUANT + GEMINI"
    )

    print("=" * 78)

    print(
        f"Source revision: "
        f"{SOURCE_REVISION}"
    )

    print(
        f"yfinance version: "
        f"{yf.__version__}"
    )

    print(
        f"Backtest period: "
        f"{YEARS}y"
    )

    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST:.3%}"
    )

    print()

    # --------------------------------------------------------
    # Load historical Gemini scores
    # --------------------------------------------------------

    news = (
        load_historical_gemini()
    )

    # --------------------------------------------------------
    # Build OHLCV dataset
    # --------------------------------------------------------

    data = build_dataset(
        news
    )

    # --------------------------------------------------------
    # Leakage audit
    # --------------------------------------------------------

    leakage_check(
        data
    )

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

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

    gemini_coverage = (
        data[
            "gemini_available"
        ].mean()
    )

    print(
        f"Historical Gemini coverage: "
        f"{gemini_coverage:.2%}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    (
        development,
        validation,
        oos,
    ) = split_data(
        data
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
    # Results
    # --------------------------------------------------------

    results = []

    nonoverlap_results = []

    portfolio_results = []

    weight_records = []

    # ========================================================
    # EACH HORIZON
    # ========================================================

    for horizon in HORIZONS:

        print()
        print(
            "=" * 78
        )

        print(
            f"HORIZON {horizon}D"
        )

        print(
            "=" * 78
        )

        # ----------------------------------------------------
        # Validation quant predictions
        # ----------------------------------------------------

        validation_predictions = (
            walk_forward(
                development,
                validation,
                QUANT_FEATURES,
                horizon,
            )
        )

        if validation_predictions.empty:

            print(
                "WARNING: validation produced "
                "no predictions."
            )

            continue

        # ----------------------------------------------------
        # Select Gemini weight on VALIDATION ONLY
        # ----------------------------------------------------

        selected_weight = (
            select_gemini_weight(
                validation_predictions,
                horizon,
            )
        )

        print(
            f"Selected Gemini weight: "
            f"{selected_weight:.2f}"
        )

        weight_records.append(
            {
                "horizon": horizon,
                "selected_gemini_weight":
                    selected_weight,
            }
        )

        # ----------------------------------------------------
        # OOS predictions
        #
        # OOS is never used for selecting weight.
        # ----------------------------------------------------

        oos_predictions = (
            walk_forward(
                development,
                oos,
                QUANT_FEATURES,
                horizon,
            )
        )

        if oos_predictions.empty:

            print(
                "WARNING: OOS produced "
                "no predictions."
            )

            continue

        # ----------------------------------------------------
        # Save raw quant OOS
        # ----------------------------------------------------

        oos_predictions.to_csv(
            AUDIT_DIR /
            f"v6_5_3_oos_h{horizon}.csv",
            index=False,
        )

        # ----------------------------------------------------
        # QUANT
        # ----------------------------------------------------

        quant_result = (
            calculate_performance(
                oos_predictions,
                horizon,
                "pred_probability",
                "pred_return",
                "quant",
            )
        )

        if quant_result:

            results.append(
                quant_result
            )

        quant_nonoverlap = (
            nonoverlapping_test(
                oos_predictions,
                horizon,
                "pred_probability",
                "pred_return",
            )
        )

        if quant_nonoverlap:

            quant_nonoverlap[
                "model"
            ] = "quant"

            quant_nonoverlap[
                "horizon"
            ] = horizon

            nonoverlap_results.append(
                quant_nonoverlap
            )

        quant_portfolio = (
            portfolio_test(
                oos_predictions,
                horizon,
                "pred_probability",
                "pred_return",
            )
        )

        if quant_portfolio:

            quant_portfolio[
                "model"
            ] = "quant"

            quant_portfolio[
                "horizon"
            ] = horizon

            portfolio_results.append(
                quant_portfolio
            )

        # ----------------------------------------------------
        # HISTORICAL GEMINI EXPERIMENT
        # ----------------------------------------------------

        has_gemini = (
            oos_predictions[
                "gemini_available"
            ].sum()
            >= 50
        )

        if not has_gemini:

            print(
                "Gemini experiment: NOT RUN — "
                "insufficient timestamped historical "
                "Gemini coverage."
            )

            continue

        hybrid = create_hybrid(
            oos_predictions,
            selected_weight,
        )

        # ----------------------------------------------------
        # GEMINI ONLY
        # ----------------------------------------------------

        gemini_result = (
            calculate_performance(
                hybrid,
                horizon,
                "gemini_probability",
                "gemini_return",
                "gemini_only",
            )
        )

        if gemini_result:

            results.append(
                gemini_result
            )

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        hybrid_result = (
            calculate_performance(
                hybrid,
                horizon,
                "hybrid_probability",
                "hybrid_return",
                "hybrid",
            )
        )

        if hybrid_result:

            results.append(
                hybrid_result
            )

        hybrid_nonoverlap = (
            nonoverlapping_test(
                hybrid,
                horizon,
                "hybrid_probability",
                "hybrid_return",
            )
        )

        if hybrid_nonoverlap:

            hybrid_nonoverlap[
                "model"
            ] = "hybrid"

            hybrid_nonoverlap[
                "horizon"
            ] = horizon

            nonoverlap_results.append(
                hybrid_nonoverlap
            )

        hybrid_portfolio = (
            portfolio_test(
                hybrid,
                horizon,
                "hybrid_probability",
                "hybrid_return",
            )
        )

        if hybrid_portfolio:

            hybrid_portfolio[
                "model"
            ] = "hybrid"

            hybrid_portfolio[
                "horizon"
            ] = horizon

            portfolio_results.append(
                hybrid_portfolio
            )

        hybrid.to_csv(
            AUDIT_DIR /
            f"v6_5_3_hybrid_oos_h{horizon}.csv",
            index=False,
        )

    # ========================================================
    # SAVE AUDIT FILES
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if not results_df.empty:

        results_df.to_csv(
            AUDIT_DIR /
            "v6_5_3_oos_model_comparison.csv",
            index=False,
        )

    weights_df = pd.DataFrame(
        weight_records
    )

    weights_df.to_csv(
        AUDIT_DIR /
        "v6_5_3_selected_gemini_weights.csv",
        index=False,
    )

    nonoverlap_df = pd.DataFrame(
        nonoverlap_results
    )

    if not nonoverlap_df.empty:

        nonoverlap_df.to_csv(
            AUDIT_DIR /
            "v6_5_3_nonoverlap_oos.csv",
            index=False,
        )

    portfolio_df = pd.DataFrame(
        portfolio_results
    )

    if not portfolio_df.empty:

        portfolio_df.to_csv(
            AUDIT_DIR /
            "v6_5_3_portfolio_oos.csv",
            index=False,
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 78)
    print(
        "V6.5.3 OOS MODEL COMPARISON"
    )
    print("=" * 78)

    if results_df.empty:

        print(
            "No valid OOS results."
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
        "1. OHLCV features are strictly "
        "backward-looking."
    )

    print(
        "2. Forward returns are targets only."
    )

    print(
        "3. Training data are purged before "
        "each prediction period."
    )

    print(
        "4. Gemini weight is selected on "
        "validation only."
    )

    print(
        "5. OOS data are never used for "
        "model or weight selection."
    )

    print(
        "6. Current Gemini API calls are "
        "never substituted for historical scores."
    )

    print(
        "7. Historical Gemini information must "
        "exist before the signal timestamp."
    )

    print(
        "8. Non-overlapping and portfolio tests "
        "are reported separately."
    )

    print(
        "9. Historical results do not guarantee "
        "future performance."
    )

    print()

    if gemini_coverage < 0.01:

        print(
            "GEMINI STATUS: NOT TESTED."
        )

        print(
            "To perform the genuine Gemini experiment, "
            "provide historical_gemini.csv containing "
            "timestamped scores."
        )

    else:

        print(
            "GEMINI STATUS: HISTORICAL DATA FOUND."
        )

        print(
            "Compare QUANT vs HYBRID on untouched "
            "OOS observations."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_backtest()
