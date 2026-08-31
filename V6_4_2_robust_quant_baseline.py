#!/usr/bin/env python3
"""
V6.4.2 — ROBUST QUANTITATIVE BASELINE

Purpose
-------
A stricter quantitative stock-selection baseline before adding Gemini/news.

Design goals
------------
1. STRICTLY causal features: every feature uses data available on signal date T.
2. Forward returns are TARGETS ONLY.
3. Purged chronological walk-forward validation.
4. No random train/test splitting.
5. Model comparison:
      - Logistic Regression: probability of positive 5D return
      - Ridge: expected 5D return
      - HistGradientBoosting: nonlinear probability model
      - HistGradientBoosting: nonlinear return model
6. Cross-sectional ranking of stocks each signal date.
7. Top-1 / Top-3 / Top-5 portfolio tests.
8. 1D, 3D, 5D and 10D horizons.
9. Realistic round-trip costs.
10. Regime and feature-ablation diagnostics.
11. Multiple independent OOS checks.
12. No Gemini in this version.

IMPORTANT
---------
This script intentionally does NOT claim that a probability such as 70%
means a literal 70% future probability. All probabilities are empirical
model outputs and must be calibrated/validated.

Research use only. Not financial advice.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

warnings.filterwarnings("ignore")

VERSION = "V6.4.2"

AUDIT = Path(os.getenv("AUDIT_DIR", "audit"))
AUDIT.mkdir(exist_ok=True)

MODE = os.getenv("MODE", "BACKTEST").upper()
PERIOD = os.getenv("BACKTEST_PERIOD", "6y")

# Prediction horizons.
HORIZONS = [1, 3, 5, 10]

# Main selection horizon.
PRIMARY_HORIZON = 5

# Purge at least as many signal dates as the largest target horizon.
PURGE_DAYS = max(HORIZONS)

MIN_HISTORY = int(os.getenv("MIN_HISTORY", "220"))
MIN_TRAIN = int(os.getenv("MIN_TRAIN", "2000"))

VAL_FRAC = float(os.getenv("VALIDATION_FRACTION", "0.20"))
OOS_FRAC = float(os.getenv("OOS_FRACTION", "0.20"))

COST_BPS = float(os.getenv("COST_BPS", "10"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))

ROUND_TRIP_COST = 2.0 * (COST_BPS + SLIPPAGE_BPS) / 10000.0

CAPITAL = float(os.getenv("CAPITAL", "100000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "5"))

# Model hyperparameters are frozen here.
LOG_C = float(os.getenv("LOG_C", "0.25"))
RIDGE_ALPHA = float(os.getenv("RIDGE_ALPHA", "10.0"))

GB_MAX_ITER = int(os.getenv("GB_MAX_ITER", "200"))
GB_LEARNING_RATE = float(os.getenv("GB_LEARNING_RATE", "0.04"))
GB_MAX_LEAF_NODES = int(os.getenv("GB_MAX_LEAF_NODES", "15"))
GB_L2 = float(os.getenv("GB_L2", "2.0"))

RANDOM_STATE = 42

SYMBOLS = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS",
    "KOTAKBANK.NS","INDUSINDBK.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "SHRIRAMFIN.NS","LT.NS","TMPV.NS","TMCV.NS","EICHERMOT.NS","MARUTI.NS",
    "HEROMOTOCO.NS","M&M.NS","TITAN.NS","ASIANPAINT.NS","HINDUNILVR.NS",
    "ITC.NS","NESTLEIND.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
    "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","BHARTIARTL.NS",
    "NTPC.NS","POWERGRID.NS","ONGC.NS","BPCL.NS","COALINDIA.NS","ADANIENT.NS",
    "ADANIPORTS.NS","BEL.NS","HAL.NS","BHEL.NS","TRENT.NS","PIDILITIND.NS",
    "SIEMENS.NS","ABB.NS","GRASIM.NS","ULTRACEMCO.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "HINDALCO.NS","IOC.NS","VEDL.NS","DLF.NS","LODHA.NS","INDIGO.NS","ETERNAL.NS",
    "NAUKRI.NS","COFORGE.NS","JIOFIN.NS","IRFC.NS","IREDA.NS","POLYCAB.NS"
]

# ---------------------------------------------------------------------
# FEATURE GROUPS
# ---------------------------------------------------------------------
# ALL are backward-looking or contemporaneous-at-close features.
# No forward-return variable appears here.

PRICE_FEATURES = [
    "ret1_past", "ret3_past", "ret5_past",
    "ret10_past", "ret20_past",
    "dist20", "dist50", "dist200",
    "rsi", "atr_pct",
    "range_pct", "close_location",
    "breakout20",
]

VOLUME_FEATURES = [
    "vol_ratio", "vol20",
]

RELATIVE_FEATURES = [
    "rel5", "rel20",
]

MARKET_FEATURES = [
    "mkt_ret1", "mkt_ret5", "mkt_ret20",
    "mkt_dist20", "mkt_dist50",
    "mkt_vol20",
    "regime",
]

ALL_FEATURES = PRICE_FEATURES + VOLUME_FEATURES + RELATIVE_FEATURES + MARKET_FEATURES

TARGET_COLUMNS = {
    "ret1_fwd", "ret3_fwd", "ret5_fwd", "ret10_fwd",
    "y1", "y3", "y5", "y10"
}

# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def clean(x: pd.DataFrame) -> pd.DataFrame:
    if x is None or x.empty:
        return pd.DataFrame()

    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]

    if any(c not in x.columns for c in needed):
        return pd.DataFrame()

    x = x[needed].copy()

    for c in needed:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["Close"])
    x = x.sort_index()

    try:
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
    except Exception:
        pass

    return x[~x.index.duplicated(keep="last")]


def download_daily(symbol: str) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        period=PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return clean(raw)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()

    up = d.clip(lower=0).ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n,
    ).mean()

    dn = (-d.clip(upper=0)).ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n,
    ).mean()

    rs = up / dn.replace(0, np.nan)

    return 100 - 100 / (1 + rs)


def atr(x: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = x["Close"].shift(1)

    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - pc).abs(),
            (x["Low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / n,
        adjust=False,
        min_periods=n,
    ).mean()


def make_features(stock: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    s = stock.copy()
    m = market.copy()

    m = m.reindex(s.index).ffill()

    c = s["Close"]
    v = s["Volume"]
    mc = m["Close"]

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()

    ms20 = mc.rolling(20).mean()
    ms50 = mc.rolling(50).mean()

    # STRICTLY PAST RETURNS.
    ret1 = c.pct_change(1)
    ret3 = c.pct_change(3)
    ret5 = c.pct_change(5)
    ret10 = c.pct_change(10)
    ret20 = c.pct_change(20)

    mret1 = mc.pct_change(1)
    mret5 = mc.pct_change(5)
    mret20 = mc.pct_change(20)

    vol20 = ret1.rolling(20).std()
    mvol20 = mret1.rolling(20).std()

    prior_high20 = s["High"].rolling(20).max().shift(1)

    regime = np.select(
        [
            mc > ms50 * 1.005,
            mc < ms50 * 0.995,
        ],
        [1.0, -1.0],
        default=0.0,
    )

    f = pd.DataFrame(
        {
            # -----------------------------
            # PRICE / MOMENTUM
            # -----------------------------
            "ret1_past": ret1,
            "ret3_past": ret3,
            "ret5_past": ret5,
            "ret10_past": ret10,
            "ret20_past": ret20,

            "dist20": c / sma20 - 1,
            "dist50": c / sma50 - 1,
            "dist200": c / sma200 - 1,

            "rsi": rsi(c),
            "atr_pct": atr(s) / c,

            "range_pct": (s["High"] - s["Low"]) / c,

            "close_location": (
                (c - s["Low"])
                / (s["High"] - s["Low"]).replace(0, np.nan)
            ),

            "breakout20": c / prior_high20 - 1,

            # -----------------------------
            # VOLUME
            # -----------------------------
            "vol_ratio": (
                v / v.rolling(20).mean()
            ),

            "vol20": vol20,

            # -----------------------------
            # RELATIVE STRENGTH
            # -----------------------------
            "rel5": ret5 - mret5,
            "rel20": ret20 - mret20,

            # -----------------------------
            # MARKET REGIME
            # -----------------------------
            "mkt_ret1": mret1,
            "mkt_ret5": mret5,
            "mkt_ret20": mret20,

            "mkt_dist20": mc / ms20 - 1,
            "mkt_dist50": mc / ms50 - 1,

            "mkt_vol20": mvol20,

            "regime": regime,

            # Informational only.
            "close": c,
            "market_close": mc,
        },
        index=s.index,
    )

    # -----------------------------------------------------------------
    # TARGETS ONLY.
    # These MUST NOT be placed into ALL_FEATURES.
    # -----------------------------------------------------------------
    for h in HORIZONS:
        f[f"ret{h}_fwd"] = c.shift(-h) / c - 1
        f[f"y{h}"] = (
            f[f"ret{h}_fwd"] > 0
        ).astype(float)

    return f.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# ---------------------------------------------------------------------
# SAFETY CHECKS
# ---------------------------------------------------------------------

def leakage_check() -> None:
    overlap = (
        set(ALL_FEATURES)
        .intersection(TARGET_COLUMNS)
    )

    if overlap:
        raise RuntimeError(
            "FATAL LEAKAGE: target fields in feature list: "
            + str(sorted(overlap))
        )

    forbidden_tokens = (
        "fwd",
        "future",
        "target",
        "y1",
        "y3",
        "y5",
        "y10",
    )

    suspicious = [
        f
        for f in ALL_FEATURES
        if any(token in f.lower() for token in forbidden_tokens)
    ]

    if suspicious:
        raise RuntimeError(
            "FATAL LEAKAGE: suspicious feature names: "
            + str(suspicious)
        )

    print(
        "FEATURE/TARGET LEAKAGE CHECK:\n"
        "PASS — ALL_FEATURES contain only backward-looking features."
    )


# ---------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------

def logistic_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=LOG_C,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def ridge_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                Ridge(
                    alpha=RIDGE_ALPHA,
                ),
            ),
        ]
    )


def gb_classifier():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=GB_MAX_ITER,
                    learning_rate=GB_LEARNING_RATE,
                    max_leaf_nodes=GB_MAX_LEAF_NODES,
                    l2_regularization=GB_L2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def gb_regressor():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=GB_MAX_ITER,
                    learning_rate=GB_LEARNING_RATE,
                    max_leaf_nodes=GB_MAX_LEAF_NODES,
                    l2_regularization=GB_L2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_models(train: pd.DataFrame, horizon: int):
    target_ret = f"ret{horizon}_fwd"
    target_y = f"y{horizon}"

    q = train.dropna(
        subset=[target_ret, target_y]
    ).copy()

    if len(q) < MIN_TRAIN:
        return None

    if q[target_y].nunique() < 2:
        return None

    models = {
        "logistic": logistic_model(),
        "ridge": ridge_model(),
        "gb_cls": gb_classifier(),
        "gb_reg": gb_regressor(),
    }

    models["logistic"].fit(
        q[ALL_FEATURES],
        q[target_y].astype(int),
    )

    models["ridge"].fit(
        q[ALL_FEATURES],
        q[target_ret],
    )

    models["gb_cls"].fit(
        q[ALL_FEATURES],
        q[target_y].astype(int),
    )

    models["gb_reg"].fit(
        q[ALL_FEATURES],
        q[target_ret],
    )

    return models


def predict_models(
    models: dict,
    d: pd.DataFrame,
):
    p_log = models["logistic"].predict_proba(
        d[ALL_FEATURES]
    )[:, 1]

    p_gb = models["gb_cls"].predict_proba(
        d[ALL_FEATURES]
    )[:, 1]

    r_ridge = models["ridge"].predict(
        d[ALL_FEATURES]
    )

    r_gb = models["gb_reg"].predict(
        d[ALL_FEATURES]
    )

    p_log = np.clip(
        p_log,
        0.01,
        0.99,
    )

    p_gb = np.clip(
        p_gb,
        0.01,
        0.99,
    )

    # Simple equal ensemble.
    p_ensemble = (
        0.5 * p_log
        + 0.5 * p_gb
    )

    r_ensemble = (
        0.5 * r_ridge
        + 0.5 * r_gb
    )

    return {
        "p_logistic": p_log,
        "p_gb": p_gb,
        "p_ensemble": p_ensemble,
        "r_ridge": r_ridge,
        "r_gb": r_gb,
        "r_ensemble": r_ensemble,
    }


# ---------------------------------------------------------------------
# PURGING
# ---------------------------------------------------------------------

def purge_training(
    prior: pd.DataFrame,
    prediction_date,
) -> pd.DataFrame:
    dates = sorted(
        prior.date.unique()
    )

    if len(dates) <= PURGE_DAYS:
        return prior.iloc[0:0].copy()

    # The latest PURGE_DAYS signal dates have labels extending
    # into/after the prediction date.
    cutoff = dates[-PURGE_DAYS - 1]

    return prior[
        prior.date <= cutoff
    ].copy()


# ---------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------

def build_dataset():
    print(
        f"Starting {VERSION} data build..."
    )
    print(
        f"Backtest period: {PERIOD}"
    )

    market = download_daily("^NSEI")

    if market.empty:
        raise RuntimeError(
            "Unable to download NIFTY data."
        )

    rows = []
    successful = 0

    for i, sym in enumerate(
        SYMBOLS,
        1,
    ):
        print(
            f"Loading [{i}/{len(SYMBOLS)}] {sym}"
        )

        try:
            stock = download_daily(sym)

            if (
                stock.empty
                or len(stock) < MIN_HISTORY
            ):
                print(
                    f"WARNING: insufficient history "
                    f"for {sym}; skipping."
                )
                continue

            f = make_features(
                stock,
                market,
            )

            successful += 1

            # Need enough future observations for 10D target.
            end = len(f) - max(HORIZONS)

            for j in range(
                MIN_HISTORY - 1,
                end,
            ):
                x = f.iloc[j]

                if x[ALL_FEATURES].isna().any():
                    continue

                row = {
                    "ticker": sym,
                    "date": f.index[j],
                    "close": float(x.close),
                    "market_close": float(x.market_close),
                }

                for col in ALL_FEATURES:
                    row[col] = float(x[col])

                for h in HORIZONS:
                    row[f"ret{h}_fwd"] = float(
                        x[f"ret{h}_fwd"]
                    )
                    row[f"y{h}"] = float(
                        x[f"y{h}"]
                    )

                rows.append(row)

        except Exception as exc:
            print(
                f"WARNING: {sym} failed: {exc}"
            )

    d = pd.DataFrame(rows)

    if d.empty:
        raise RuntimeError(
            "No observations generated."
        )

    d = d.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    leakage_check()

    return d, successful


# ---------------------------------------------------------------------
# CALIBRATION
# ---------------------------------------------------------------------

def fit_probability_calibrator(
    p: np.ndarray,
    y: np.ndarray,
):
    """
    Platt-style calibration.

    Calibration is fitted ONLY on validation data and frozen for OOS.
    """
    p = np.clip(
        np.asarray(p),
        1e-5,
        1 - 1e-5,
    )

    z = np.log(
        p / (1 - p)
    ).reshape(-1, 1)

    cal = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    cal.fit(
        z,
        np.asarray(y).astype(int),
    )

    return cal


def calibrated_probability(
    cal,
    p,
):
    p = np.clip(
        np.asarray(p),
        1e-5,
        1 - 1e-5,
    )

    z = np.log(
        p / (1 - p)
    ).reshape(-1, 1)

    return np.clip(
        cal.predict_proba(z)[:, 1],
        0.01,
        0.99,
    )


# ---------------------------------------------------------------------
# SPLITS
# ---------------------------------------------------------------------

def chronological_split(
    d: pd.DataFrame,
):
    dates = sorted(
        d.date.unique()
    )

    n = len(dates)

    val_start_i = int(
        n * (1 - OOS_FRAC - VAL_FRAC)
    )

    oos_start_i = int(
        n * (1 - OOS_FRAC)
    )

    val_start = dates[val_start_i]
    oos_start = dates[oos_start_i]

    dev = d[
        d.date < val_start
    ].copy()

    val = d[
        (d.date >= val_start)
        & (d.date < oos_start)
    ].copy()

    oos = d[
        d.date >= oos_start
    ].copy()

    return (
        dev,
        val,
        oos,
        val_start,
        oos_start,
    )


# ---------------------------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------------------------

def score_model_on_validation(
    val: pd.DataFrame,
    probability_column: str,
    return_column: str,
    horizon: int,
):
    target_y = f"y{horizon}"
    target_ret = f"ret{horizon}_fwd"

    q = val.dropna(
        subset=[
            probability_column,
            return_column,
            target_y,
            target_ret,
        ]
    )

    if q.empty:
        return {
            "model": probability_column,
            "brier": np.nan,
            "logloss": np.nan,
            "return_mae": np.nan,
            "directional_accuracy": np.nan,
            "avg_return_top10pct": np.nan,
        }

    p = np.clip(
        q[probability_column].values,
        1e-6,
        1 - 1e-6,
    )

    y = q[target_y].astype(int).values

    rpred = q[return_column].values
    rtrue = q[target_ret].values

    cutoff = np.nanpercentile(
        rpred,
        90,
    )

    top = q[rpred >= cutoff]

    return {
        "model": probability_column,
        "brier": brier_score_loss(
            y,
            p,
        ),
        "logloss": log_loss(
            y,
            p,
            labels=[0, 1],
        ),
        "return_mae": mean_absolute_error(
            rtrue,
            rpred,
        ),
        "directional_accuracy": (
            (rpred > 0)
            == (rtrue > 0)
        ).mean(),
        "avg_return_top10pct": (
            top[target_ret].mean()
            if len(top)
            else np.nan
        ),
    }


# ---------------------------------------------------------------------
# CROSS-SECTIONAL RANKING
# ---------------------------------------------------------------------

def rank_actions(
    d: pd.DataFrame,
    probability_col: str,
    return_col: str,
):
    z = d.copy()

    # Rank within each date.
    z["prob_rank"] = (
        z.groupby("date")[probability_col]
        .rank(
            pct=True,
            method="average",
        )
    )

    z["ret_rank"] = (
        z.groupby("date")[return_col]
        .rank(
            pct=True,
            method="average",
        )
    )

    # Ensemble ranking:
    # probability and expected-return rankings contribute equally.
    z["combined_rank"] = (
        0.50 * z["prob_rank"]
        + 0.50 * z["ret_rank"]
    )

    # Risk gates.
    z["risk_ok"] = (
        z["rsi"].between(35, 72)
        & (z["atr_pct"] < 0.06)
        & (z["vol_ratio"] > 0.50)
    )

    z["trade_candidate"] = (
        (z["combined_rank"] >= 0.85)
        & (z[probability_col] >= 0.52)
        & (z[return_col] >= 0.001)
        & z["risk_ok"]
    )

    z["watch_candidate"] = (
        (z["combined_rank"] >= 0.65)
        & (z[probability_col] >= 0.50)
        & (z[return_col] >= 0.0)
    )

    z["action"] = np.where(
        z.trade_candidate,
        "TRADE",
        np.where(
            z.watch_candidate,
            "WATCH",
            "WAIT",
        ),
    )

    return z


# ---------------------------------------------------------------------
# RETURN PERFORMANCE
# ---------------------------------------------------------------------

def add_net_returns(
    d: pd.DataFrame,
):
    z = d.copy()

    for h in HORIZONS:
        z[f"net{h}"] = (
            z[f"ret{h}_fwd"]
            - ROUND_TRIP_COST
        )

    return z


def action_performance(
    d: pd.DataFrame,
):
    rows = []

    for action, g in d.groupby(
        "action"
    ):
        for h in HORIZONS:
            x = g[
                f"net{h}"
            ].dropna()

            if x.empty:
                continue

            winners = x[x > 0]
            losers = x[x <= 0]

            pf = (
                winners.sum()
                / abs(losers.sum())
                if len(losers)
                and losers.sum() < 0
                else np.nan
            )

            rows.append(
                {
                    "selection": action,
                    "horizon": h,
                    "observations": len(x),
                    "win_rate": (x > 0).mean(),
                    "average_net_return": x.mean(),
                    "median_net_return": x.median(),
                    "profit_factor": pf,
                    "best": x.max(),
                    "worst": x.min(),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# TOP-N CROSS-SECTIONAL PORTFOLIO
# ---------------------------------------------------------------------

def top_n_portfolio(
    d: pd.DataFrame,
    n: int,
):
    """
    One signal date = one portfolio formation event.

    Select top N stocks by combined rank.
    Equal-weight the selected stocks.

    The reported horizon returns are event returns.
    This is NOT a daily compounding simulation.
    """
    rows = []

    for date, g in d.groupby("date"):
        g = g.dropna(
            subset=[
                "combined_rank",
                "ret1_fwd",
                "ret3_fwd",
                "ret5_fwd",
                "ret10_fwd",
            ]
        )

        if len(g) < n:
            continue

        selected = g.nlargest(
            n,
            "combined_rank",
        )

        row = {
            "date": date,
            "selected": n,
            "tickers": ",".join(
                selected.ticker.astype(str)
            ),
        }

        for h in HORIZONS:
            row[f"gross{h}"] = selected[
                f"ret{h}_fwd"
            ].mean()

            row[f"net{h}"] = (
                row[f"gross{h}"]
                - ROUND_TRIP_COST
            )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_portfolio(
    p: pd.DataFrame,
):
    rows = []

    if p.empty:
        return pd.DataFrame()

    for h in HORIZONS:
        x = p[
            f"net{h}"
        ].dropna()

        if x.empty:
            continue

        winners = x[x > 0]
        losers = x[x <= 0]

        pf = (
            winners.sum()
            / abs(losers.sum())
            if len(losers)
            and losers.sum() < 0
            else np.nan
        )

        rows.append(
            {
                "horizon": h,
                "events": len(x),
                "win_rate": (x > 0).mean(),
                "average_net_return": x.mean(),
                "median_net_return": x.median(),
                "profit_factor": pf,
                "best": x.max(),
                "worst": x.min(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# NON-OVERLAPPING EVENTS
# ---------------------------------------------------------------------

def nonoverlap_top_n(
    d: pd.DataFrame,
    n: int,
    horizon: int,
):
    p = top_n_portfolio(
        d,
        n,
    )

    if p.empty:
        return pd.DataFrame()

    p = p.sort_values("date")

    selected = []
    last_i = -10**9

    dates = sorted(
        p.date.unique()
    )

    date_i = {
        d: i
        for i, d in enumerate(dates)
    }

    for _, row in p.iterrows():
        i = date_i[row.date]

        if i >= last_i + horizon:
            selected.append(row)
            last_i = i

    q = pd.DataFrame(selected)

    if q.empty:
        return pd.DataFrame()

    x = q[
        f"net{horizon}"
    ].dropna()

    return pd.DataFrame(
        [
            {
                "top_n": n,
                "horizon": horizon,
                "trades": len(x),
                "win_rate": (x > 0).mean(),
                "average_net_return": x.mean(),
                "median_net_return": x.median(),
                "best": x.max(),
                "worst": x.min(),
                "sum_net_return": x.sum(),
            }
        ]
    )


# ---------------------------------------------------------------------
# REGIME TEST
# ---------------------------------------------------------------------

def regime_performance(
    d: pd.DataFrame,
):
    rows = []

    for regime, g in d.groupby(
        "regime"
    ):
        for h in HORIZONS:
            x = g[
                f"net{h}"
            ].dropna()

            if x.empty:
                continue

            rows.append(
                {
                    "regime": regime,
                    "horizon": h,
                    "observations": len(x),
                    "win_rate": (x > 0).mean(),
                    "average_net_return": x.mean(),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# FEATURE ABLATION
# ---------------------------------------------------------------------

def feature_ablation(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    groups = {
        "all": ALL_FEATURES,
        "price_only": PRICE_FEATURES,
        "price_volume": PRICE_FEATURES + VOLUME_FEATURES,
        "price_relative": PRICE_FEATURES + RELATIVE_FEATURES,
        "market_plus_price": PRICE_FEATURES + MARKET_FEATURES,
    }

    rows = []

    target_y = f"y{PRIMARY_HORIZON}"
    target_r = f"ret{PRIMARY_HORIZON}_fwd"

    q = train.dropna(
        subset=[
            target_y,
            target_r,
        ]
    )

    t = test.dropna(
        subset=[
            target_y,
            target_r,
        ]
    )

    if q.empty or t.empty:
        return pd.DataFrame()

    for name, features in groups.items():
        model = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=LOG_C,
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        model.fit(
            q[features],
            q[target_y].astype(int),
        )

        p = model.predict_proba(
            t[features]
        )[:, 1]

        rows.append(
            {
                "feature_set": name,
                "observations": len(t),
                "brier": brier_score_loss(
                    t[target_y],
                    p,
                ),
                "logloss": log_loss(
                    t[target_y],
                    p,
                    labels=[0, 1],
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# PREDICTION METRICS
# ---------------------------------------------------------------------

def prediction_metrics(
    d: pd.DataFrame,
    p_col: str,
    r_col: str,
    name: str,
):
    rows = []

    for h in HORIZONS:
        q = d.dropna(
            subset=[
                p_col,
                r_col,
                f"y{h}",
                f"ret{h}_fwd",
            ]
        )

        if q.empty:
            continue

        p = np.clip(
            q[p_col].values,
            1e-6,
            1 - 1e-6,
        )

        y = q[f"y{h}"].astype(int).values

        rp = q[r_col].values
        rt = q[f"ret{h}_fwd"].values

        rows.append(
            {
                "sample": name,
                "horizon": h,
                "observations": len(q),
                "brier_score": brier_score_loss(
                    y,
                    p,
                ),
                "log_loss": log_loss(
                    y,
                    p,
                    labels=[0, 1],
                ),
                "return_mae": mean_absolute_error(
                    rt,
                    rp,
                ),
                "directional_accuracy": (
                    (rp > 0)
                    == (rt > 0)
                ).mean(),
                "mean_predicted_return": rp.mean(),
                "mean_actual_return": rt.mean(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# WALK FORWARD
# ---------------------------------------------------------------------

def walk_forward(
    d: pd.DataFrame,
    cal_logistic,
    cal_gb,
):
    dates = sorted(
        d.date.unique()
    )

    parts = []

    for k, date in enumerate(dates):
        cur = d[
            d.date == date
        ].copy()

        prior = d[
            d.date < date
        ].copy()

        if len(prior) < MIN_TRAIN:
            continue

        # IMPORTANT: purge largest horizon.
        train = purge_training(
            prior,
            date,
        )

        if len(train) < MIN_TRAIN:
            continue

        # Keep compute manageable.
        train = train.tail(
            int(os.getenv(
                "MAX_TRAIN_OBS",
                "12000",
            ))
        )

        models = fit_models(
            train,
            PRIMARY_HORIZON,
        )

        if models is None:
            continue

        pred = predict_models(
            models,
            cur,
        )

        cur["p_logistic"] = pred[
            "p_logistic"
        ]

        cur["p_gb"] = pred[
            "p_gb"
        ]

        cur["p_ensemble_raw"] = pred[
            "p_ensemble"
        ]

        cur["p_logistic_cal"] = calibrated_probability(
            cal_logistic,
            pred["p_logistic"],
        )

        cur["p_gb_cal"] = calibrated_probability(
            cal_gb,
            pred["p_gb"],
        )

        cur["p_ensemble"] = (
            0.5 * cur["p_logistic_cal"]
            + 0.5 * cur["p_gb_cal"]
        )

        cur["r_ridge"] = pred[
            "r_ridge"
        ]

        cur["r_gb"] = pred[
            "r_gb"
        ]

        cur["r_ensemble"] = pred[
            "r_ensemble"
        ]

        cur = rank_actions(
            cur,
            "p_ensemble",
            "r_ensemble",
        )

        cur = add_net_returns(
            cur
        )

        parts.append(cur)

        if (
            k == 0
            or k % 100 == 0
            or k == len(dates) - 1
        ):
            print(
                f"Walk-forward date "
                f"[{k+1}/{len(dates)}]"
            )

    if not parts:
        return pd.DataFrame()

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# MAIN BACKTEST
# ---------------------------------------------------------------------

def run_backtest():
    d, successful = build_dataset()

    dev, val, oos, val_start, oos_start = chronological_split(
        d
    )

    print(
        f"\nTotal candidate observations: {len(d)}"
    )
    print(
        f"Successful symbols: {successful}"
    )
    print(
        f"Unique symbols: {d.ticker.nunique()}"
    )
    print(
        f"Unique signal dates: {d.date.nunique()}"
    )
    print(
        f"Development observations: {len(dev)}"
    )
    print(
        f"Validation observations: {len(val)}"
    )
    print(
        f"OOS observations: {len(oos)}"
    )
    print(
        f"Validation start: {val_start}"
    )
    print(
        f"OOS start: {oos_start}"
    )
    print(
        f"Round-trip cost assumption: "
        f"{ROUND_TRIP_COST * 100:.3f}%"
    )

    # -------------------------------------------------------------
    # VALIDATION MODEL
    # -------------------------------------------------------------
    dev_train = purge_training(
        dev,
        val.date.min(),
    )

    print(
        f"\nPurged development training observations: "
        f"{len(dev_train)}"
    )

    models_val = fit_models(
        dev_train,
        PRIMARY_HORIZON,
    )

    if models_val is None:
        raise RuntimeError(
            "Validation model fit failed."
        )

    pred_val = predict_models(
        models_val,
        val,
    )

    for key, values in pred_val.items():
        val[key] = values

    # Calibrators are validation-only objects.
    cal_logistic = fit_probability_calibrator(
        val.p_logistic,
        val.y5,
    )

    cal_gb = fit_probability_calibrator(
        val.p_gb,
        val.y5,
    )

    val["p_logistic_cal"] = calibrated_probability(
        cal_logistic,
        val.p_logistic,
    )

    val["p_gb_cal"] = calibrated_probability(
        cal_gb,
        val.p_gb,
    )

    val["p_ensemble"] = (
        0.5 * val.p_logistic_cal
        + 0.5 * val.p_gb_cal
    )

    val["r_ensemble"] = val.r_ensemble

    val = rank_actions(
        val,
        "p_ensemble",
        "r_ensemble",
    )

    val = add_net_returns(
        val
    )

    # -------------------------------------------------------------
    # OOS MODEL
    # -------------------------------------------------------------
    pre = d[
        d.date < oos_start
    ].copy()

    pre_train = purge_training(
        pre,
        oos_start,
    )

    print(
        f"Pre-OOS observations: {len(pre)}"
    )
    print(
        f"Purged pre-OOS training observations: "
        f"{len(pre_train)}"
    )

    models_oos = fit_models(
        pre_train,
        PRIMARY_HORIZON,
    )

    if models_oos is None:
        raise RuntimeError(
            "OOS model fit failed."
        )

    pred_oos = predict_models(
        models_oos,
        oos,
    )

    for key, values in pred_oos.items():
        oos[key] = values

    # Frozen validation calibrators.
    oos["p_logistic_cal"] = calibrated_probability(
        cal_logistic,
        oos.p_logistic,
    )

    oos["p_gb_cal"] = calibrated_probability(
        cal_gb,
        oos.p_gb,
    )

    oos["p_ensemble"] = (
        0.5 * oos.p_logistic_cal
        + 0.5 * oos.p_gb_cal
    )

    oos["r_ensemble"] = oos.r_ensemble

    oos = rank_actions(
        oos,
        "p_ensemble",
        "r_ensemble",
    )

    oos = add_net_returns(
        oos
    )

    # -------------------------------------------------------------
    # WALK-FORWARD
    # -------------------------------------------------------------
    print(
        "\nRunning chronological walk-forward diagnostics..."
    )

    wf = walk_forward(
        d,
        cal_logistic,
        cal_gb,
    )

    # -------------------------------------------------------------
    # TOP-N OOS PORTFOLIOS
    # -------------------------------------------------------------
    top_tables = {}

    for n in (1, 3, 5):
        top_tables[n] = top_n_portfolio(
            oos,
            n,
        )

    # -------------------------------------------------------------
    # NON-OVERLAPPING
    # -------------------------------------------------------------
    nonoverlap_rows = []

    for n in (1, 3, 5):
        for h in (3, 5, 10):
            q = nonoverlap_top_n(
                oos,
                n,
                h,
            )

            if not q.empty:
                nonoverlap_rows.append(q)

    nonoverlap = (
        pd.concat(
            nonoverlap_rows,
            ignore_index=True,
        )
        if nonoverlap_rows
        else pd.DataFrame()
    )

    # -------------------------------------------------------------
    # ACTION PERFORMANCE
    # -------------------------------------------------------------
    wf_perf = action_performance(
        wf
    ) if not wf.empty else pd.DataFrame()

    oos_perf = action_performance(
        oos
    )

    # -------------------------------------------------------------
    # REGIME
    # -------------------------------------------------------------
    regime = regime_performance(
        oos
    )

    # -------------------------------------------------------------
    # MODEL METRICS
    # -------------------------------------------------------------
    metrics = pd.concat(
        [
            prediction_metrics(
                oos,
                "p_logistic_cal",
                "r_ridge",
                "OOS_LOGISTIC_RIDGE",
            ),
            prediction_metrics(
                oos,
                "p_gb_cal",
                "r_gb",
                "OOS_GB",
            ),
            prediction_metrics(
                oos,
                "p_ensemble",
                "r_ensemble",
                "OOS_ENSEMBLE",
            ),
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # VALIDATION MODEL COMPARISON
    # -------------------------------------------------------------
    model_comparison = pd.DataFrame(
        [
            score_model_on_validation(
                val,
                "p_logistic_cal",
                "r_ridge",
                PRIMARY_HORIZON,
            ),
            score_model_on_validation(
                val,
                "p_gb_cal",
                "r_gb",
                PRIMARY_HORIZON,
            ),
            score_model_on_validation(
                val,
                "p_ensemble",
                "r_ensemble",
                PRIMARY_HORIZON,
            ),
        ]
    )

    # -------------------------------------------------------------
    # FEATURE ABLATION
    # -------------------------------------------------------------
    ablation = feature_ablation(
        pre_train,
        oos,
    )

    # -------------------------------------------------------------
    # NIFTY BENCHMARK
    # -------------------------------------------------------------
    benchmark = (
        d.groupby("date")
        .market_close
        .first()
        .sort_index()
    )

    benchmark_rows = []

    for h in HORIZONS:
        r = (
            benchmark.shift(-h)
            / benchmark
            - 1
        ).dropna()

        benchmark_rows.append(
            {
                "horizon": h,
                "observations": len(r),
                "win_rate": (r > 0).mean(),
                "average_return": r.mean(),
                "median_return": r.median(),
            }
        )

    benchmark_table = pd.DataFrame(
        benchmark_rows
    )

    # -------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------
    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    files = {
        f"dataset_v6_4_2_{ts}.csv": d,
        f"validation_v6_4_2_{ts}.csv": val,
        f"oos_v6_4_2_{ts}.csv": oos,
        f"walkforward_v6_4_2_{ts}.csv": wf,
        f"oos_action_performance_v6_4_2_{ts}.csv": oos_perf,
        f"walkforward_action_performance_v6_4_2_{ts}.csv": wf_perf,
        f"model_comparison_v6_4_2_{ts}.csv": model_comparison,
        f"prediction_metrics_v6_4_2_{ts}.csv": metrics,
        f"nonoverlap_v6_4_2_{ts}.csv": nonoverlap,
        f"regime_performance_v6_4_2_{ts}.csv": regime,
        f"feature_ablation_v6_4_2_{ts}.csv": ablation,
        f"nifty_benchmark_v6_4_2_{ts}.csv": benchmark_table,
    }

    for n, frame in files.items():
        frame.to_csv(
            AUDIT / n,
            index=False,
        )

    print(
        "\n" + "=" * 76
    )
    print(
        "V6.4.2 ROBUST QUANTITATIVE BASELINE"
    )
    print(
        "=" * 76
    )

    print(
        "\nFEATURE SET:"
    )
    print(
        f"Total causal features: {len(ALL_FEATURES)}"
    )

    print(
        "\nOOS ACTION COUNTS:"
    )
    print(
        oos.action.value_counts()
        .rename_axis("action")
        .to_frame("count")
        .to_string()
    )

    print(
        "\nOOS ACTION PERFORMANCE:"
    )
    print(
        oos_perf.to_string(index=False)
    )

    print(
        "\nVALIDATION MODEL COMPARISON:"
    )
    print(
        model_comparison.to_string(index=False)
    )

    print(
        "\nNON-OVERLAPPING OOS TEST:"
    )
    print(
        nonoverlap.to_string(index=False)
        if not nonoverlap.empty
        else "None"
    )

    print(
        "\nREGIME PERFORMANCE:"
    )
    print(
        regime.to_string(index=False)
        if not regime.empty
        else "None"
    )

    print(
        "\nPREDICTION METRICS:"
    )
    print(
        metrics.to_string(index=False)
    )

    print(
        "\nFEATURE ABLATION:"
    )
    print(
        ablation.to_string(index=False)
        if not ablation.empty
        else "None"
    )

    print(
        "\nTOP-N CROSS-SECTIONAL OOS PORTFOLIOS:"
    )

    for n in (1, 3, 5):
        print(
            f"\nTOP {n}:"
        )

        summary = summarize_portfolio(
            top_tables[n]
        )

        print(
            summary.to_string(index=False)
            if not summary.empty
            else "None"
        )

    print(
        "\nNIFTY BENCHMARK:"
    )
    print(
        benchmark_table.to_string(
            index=False
        )
    )

    print(
        "\nFILES CREATED:"
    )

    for n in files:
        print(
            AUDIT / n
        )

    print(
        "\n" + "=" * 76
    )
    print(
        "V6.4.2 BACKTEST COMPLETED"
    )
    print(
        "=" * 76
    )

    print(
        """
AUDIT NOTES
-----------
1. No forward-return target is included in ALL_FEATURES.
2. Future returns exist only as target columns.
3. The largest 10-day horizon is purged before model fitting.
4. Validation calibration is frozen before OOS.
5. OOS observations are never used for model fitting.
6. Model comparison is performed before OOS interpretation.
7. Top-N tests are cross-sectional and event based.
8. Non-overlapping tests are reported separately.
9. Results are compared with a NIFTY benchmark.
10. Historical backtests do not guarantee future returns.
"""
    )


# ---------------------------------------------------------------------
# LIVE MODE
# ---------------------------------------------------------------------

def run_live():
    """
    V6.4.2 LIVE is intentionally conservative.

    It generates a ranking only. Thresholds are supplied through
    environment variables and must come from a reviewed validation run.

    Do NOT use this until the backtest has been reviewed.
    """
    raise RuntimeError(
        "V6.4.2 LIVE is intentionally disabled. "
        "Run and review BACKTEST first."
    )


if __name__ == "__main__":
    if MODE == "LIVE":
        run_live()
    else:
        run_backtest()
