#!/usr/bin/env python3
"""
V6.6.1 — LEAKAGE-PROOF QUANT + POINT-IN-TIME GEMINI + STABILITY AUDIT

Purpose
-------
This version improves V6.6 by adding:

1. Strict point-in-time historical Gemini joining.
2. Purged chronological walk-forward predictions.
3. Quant-only vs Hybrid comparison.
4. NIFTY benchmark.
5. Equal-weight market benchmark.
6. Non-overlapping trade tests.
7. Correct portfolio accounting that never compounds NaN.
8. Year-by-year OOS stability.
9. Bull / bear / neutral regime stability.
10. Bootstrap confidence intervals for selected-trade returns/win rate.
11. Explicit "not tested" behavior when historical Gemini is absent.
12. No current Gemini calls during historical backtests.

Required Python packages:
    numpy
    pandas
    yfinance
    scikit-learn

Optional historical Gemini file:
    data/historical_gemini.csv

Required Gemini columns:
    ticker,published_at,gemini_score

Optional:
    gemini_confidence
    gemini_materiality

gemini_score must be in [-1, +1].

Research only. Not investment advice.
"""

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
# CONFIG
# ============================================================

VERSION = "V6.6.2"
REVISION = "2026-08-31-NAN-FIX2-THRESHOLD-FIX"

YEARS = 6
RANDOM_STATE = 42

ROUND_TRIP_COST = 0.0030
PURGE_DAYS = 10
MIN_TRAIN = 2000

# Retraining interval inside validation/OOS.
RETRAIN_EVERY = 20

HORIZONS = [1, 3, 5, 10]

TRADE_P_THRESHOLD_THRESHOLD = 0.55
TRADE_RETURN_THRESHOLDETURN_THRESHOLD = 0.0020

GEMINI_WEIGHTS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

GEMINI_LOOKBACK_DAYS = 7

BOOTSTRAP_ITERATIONS = 3000
CONFIDENCE_LEVEL = 0.95

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

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
# FEATURES
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

# ============================================================
# HELPERS
# ============================================================


# ---------------------------------------------------------------------------
# V6.6.1-NAN-FIX
# All model feature matrices are forced finite before sklearn fitting.
# No target/forward-return field is used for imputation.
# ---------------------------------------------------------------------------
def _finite_features(X):
    import numpy as _np
    import pandas as _pd
    if isinstance(X, _pd.DataFrame):
        Z = X.copy()
        for c in Z.columns:
            Z[c] = _pd.to_numeric(Z[c], errors="coerce")
        Z = Z.replace([_np.inf, -_np.inf], _np.nan)
        med = Z.median(axis=0, skipna=True)
        return Z.fillna(med).fillna(0.0).astype(float)
    A = _np.asarray(X, dtype=float)
    A = _np.where(_np.isfinite(A), A, _np.nan)
    if A.ndim == 1:
        vals = A[_np.isfinite(A)]
        fill = float(_np.median(vals)) if vals.size else 0.0
        return _np.where(_np.isfinite(A), A, fill)
    for j in range(A.shape[1]):
        vals = A[:, j][_np.isfinite(A[:, j])]
        fill = float(_np.median(vals)) if vals.size else 0.0
        A[~_np.isfinite(A[:, j]), j] = fill
    return A

def sigmoid(x):
    return 1.0 / (
        1.0 + np.exp(-np.clip(x, -30, 30))
    )


def normalize_ohlcv(raw):
    if raw is None or raw.empty:
        return pd.DataFrame()

    x = raw.copy()
    wanted = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if isinstance(
        x.columns,
        pd.MultiIndex
    ):
        flat = []
        for col in x.columns:
            parts = [str(v) for v in col]
            found = next(
                (v for v in parts if v in wanted),
                parts[-1]
            )
            flat.append(found)
        x.columns = flat
        x = x.loc[
            :,
            ~x.columns.duplicated(keep="first")
        ]
    else:
        x.columns = [str(c) for c in x.columns]

    if any(
        c not in x.columns
        for c in wanted
    ):
        return pd.DataFrame()

    for c in wanted:
        if isinstance(x[c], pd.DataFrame):
            x[c] = x[c].iloc[:, 0]
        x[c] = pd.to_numeric(
            x[c],
            errors="coerce"
        )

    x = x[wanted].dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    x.index = pd.to_datetime(
        x.index,
        errors="coerce"
    )

    x = x.loc[~x.index.isna()]

    return x.sort_index()


def calc_rsi(close, period):
    delta = close.diff()
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

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


def calc_atr(df, period=14):
    prev = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev).abs(),
            (df["Low"] - prev).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.rolling(
        period
    ).mean()


# ============================================================
# MARKET FEATURES / TARGETS
# ============================================================

def build_features(df):
    x = df.copy()
    close = x["Close"]

    daily = close.pct_change()

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_5"] = close.pct_change(5)
    x["ret_10"] = close.pct_change(10)
    x["ret_20"] = close.pct_change(20)

    x["vol_5"] = daily.rolling(5).std()
    x["vol_10"] = daily.rolling(10).std()
    x["vol_20"] = daily.rolling(20).std()

    x["rsi_7"] = calc_rsi(close, 7)
    x["rsi_14"] = calc_rsi(close, 14)
    x["rsi_21"] = calc_rsi(close, 21)

    x["atr_pct"] = (
        calc_atr(x, 14) /
        close
    )

    ema10 = close.ewm(
        span=10,
        adjust=False
    ).mean()

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

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
        high20.shift(1) -
        1
    )

    x["breakdown_20"] = (
        close /
        low20.shift(1) -
        1
    )

    vmean = (
        x["Volume"]
        .rolling(20)
        .mean()
    )

    vstd = (
        x["Volume"]
        .rolling(20)
        .std()
    )

    x["volume_z"] = (
        (
            x["Volume"] -
            vmean
        )
        /
        vstd.replace(
            0,
            np.nan
        )
    )

    x["range_pct"] = (
        (
            x["High"] -
            x["Low"]
        )
        /
        close
    )

    x["close_location"] = (
        (
            close -
            x["Low"]
        )
        /
        (
            x["High"] -
            x["Low"]
        ).replace(
            0,
            np.nan
        )
    )

    x["momentum_accel"] = (
        x["ret_5"] -
        x["ret_20"] / 4
    )

    return x


def add_targets(df):
    x = df.copy()

    for h in HORIZONS:

        x[f"target_{h}d"] = (
            x["Close"].shift(-h)
            /
            x["Close"]
            -
            1
        )

        x[f"up_{h}d"] = (
            x[f"target_{h}d"] > 0
        ).astype(int)

    return x


# ============================================================
# HISTORICAL GEMINI
# ============================================================

def load_historical_gemini():

    candidates = [
        Path(
            "data/historical_gemini.csv"
        ),
        Path(
            "historical_gemini.csv"
        ),
    ]

    file = next(
        (
            p for p in candidates
            if p.exists()
        ),
        None
    )

    if file is None:

        print(
            "HISTORICAL GEMINI: NOT FOUND"
        )

        return pd.DataFrame()

    g = pd.read_csv(
        file
    )

    required = {
        "ticker",
        "published_at",
        "gemini_score",
    }

    missing = (
        required -
        set(g.columns)
    )

    if missing:

        raise RuntimeError(
            "Historical Gemini file missing: "
            f"{sorted(missing)}"
        )

    if (
        "gemini_confidence"
        not in g.columns
    ):
        g["gemini_confidence"] = 0.5

    if (
        "gemini_materiality"
        not in g.columns
    ):
        g["gemini_materiality"] = 0.5

    g["ticker"] = (
        g["ticker"]
        .astype(str)
        .str.strip()
    )

    g["published_at"] = (
        pd.to_datetime(
            g["published_at"],
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(None)
    )

    for c in [
        "gemini_score",
        "gemini_confidence",
        "gemini_materiality",
    ]:

        g[c] = pd.to_numeric(
            g[c],
            errors="coerce"
        )

    g = g.dropna(
        subset=[
            "ticker",
            "published_at",
            "gemini_score",
        ]
    )

    g["gemini_score"] = (
        g["gemini_score"]
        .clip(-1, 1)
    )

    g["gemini_confidence"] = (
        g["gemini_confidence"]
        .clip(0, 1)
    )

    g["gemini_materiality"] = (
        g["gemini_materiality"]
        .clip(0, 1)
    )

    g = g[
        g["ticker"].isin(TICKERS)
    ]

    g = g.sort_values(
        [
            "ticker",
            "published_at",
        ]
    )

    print(
        f"HISTORICAL GEMINI: FOUND "
        f"{len(g):,} records"
    )

    return g


def attach_gemini(
    data,
    gemini
):

    out = data.copy()

    out["gemini_score"] = 0.0
    out["gemini_confidence"] = 0.0
    out["gemini_materiality"] = 0.0
    out["gemini_available"] = 0
    out["gemini_event_age_days"] = np.nan

    if gemini.empty:
        return out

    out["signal_timestamp"] = (
        pd.to_datetime(
            out["date"]
        )
    )

    left = out[
        [
            "ticker",
            "signal_timestamp",
        ]
    ].copy()

    right = gemini[
        [
            "ticker",
            "published_at",
            "gemini_score",
            "gemini_confidence",
            "gemini_materiality",
        ]
    ].copy()

    left = left.sort_values(
        [
            "ticker",
            "signal_timestamp",
        ]
    )

    right = right.sort_values(
        [
            "ticker",
            "published_at",
        ]
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="signal_timestamp",
        right_on="published_at",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(
            days=GEMINI_LOOKBACK_DAYS
        ),
        allow_exact_matches=True,
    )

    bad = (
        merged["published_at"].notna()
        &
        (
            merged["published_at"]
            >
            merged["signal_timestamp"]
        )
    )

    if bad.any():

        raise RuntimeError(
            "FATAL: Gemini timestamp leakage."
        )

    age = (
        merged["signal_timestamp"] -
        merged["published_at"]
    ).dt.total_seconds() / 86400.0

    out = out.reset_index(
        drop=True
    )

    out["gemini_score"] = (
        pd.to_numeric(
            merged["gemini_score"],
            errors="coerce"
        ).fillna(0).to_numpy()
    )

    out["gemini_confidence"] = (
        pd.to_numeric(
            merged["gemini_confidence"],
            errors="coerce"
        ).fillna(0).to_numpy()
    )

    out["gemini_materiality"] = (
        pd.to_numeric(
            merged["gemini_materiality"],
            errors="coerce"
        ).fillna(0).to_numpy()
    )

    out["gemini_available"] = (
        merged["published_at"]
        .notna()
        .astype(int)
        .to_numpy()
    )

    out["gemini_event_age_days"] = (
        age.to_numpy()
    )

    return out


# ============================================================
# DATA BUILD
# ============================================================

def build_dataset(gemini):

    frames = []

    for i, ticker in enumerate(
        TICKERS,
        1
    ):

        print(
            f"Loading [{i}/{len(TICKERS)}] "
            f"{ticker}"
        )

        try:

            raw = yf.download(
                ticker,
                period=f"{YEARS}y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            px = normalize_ohlcv(
                raw
            )

            if len(px) < 300:

                print(
                    f"WARNING: insufficient "
                    f"history for {ticker}; skipping."
                )

                continue

            f = build_features(
                px
            )

            f = add_targets(
                f
            )

            f["ticker"] = ticker
            f["date"] = f.index

            frames.append(
                f.reset_index(
                    drop=True
                )
            )

        except Exception as exc:

            print(
                f"WARNING: {ticker} failed: "
                f"{exc}"
            )

    if not frames:

        raise RuntimeError(
            "No valid market data were created."
        )

    data = pd.concat(
        frames,
        ignore_index=True
    )

    data = data.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    data = attach_gemini(
        data,
        gemini
    )

    return data


# ============================================================
# MODEL
# ============================================================

def make_classifier():

    return Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            )
        ]
    )


def make_regressor():

    return Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),
            (
                "model",
                Ridge(alpha=10.0)
            )
        ]
    )


def fit_model(
    train,
    features,
    horizon
):

    required = features + [
        f"target_{horizon}d",
        f"up_{horizon}d",
    ]

    q = train[
        required
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    q = q.dropna(
        subset=[
            f"target_{horizon}d",
            f"up_{horizon}d",
        ]
    )

    if (
        len(q) < MIN_TRAIN
        or
        q[
            f"up_{horizon}d"
        ].nunique() < 2
    ):

        return None

    X = q[
        features
    ]

    y_direction = q[
        f"up_{horizon}d"
    ].astype(int)

    y_return = q[
        f"target_{horizon}d"
    ].astype(float)

    clf1 = make_classifier()

    clf2 = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    reg1 = make_regressor()

    reg2 = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.04,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )

    # FINAL NaN/Inf guard: sanitize the exact matrix passed to every estimator.
    X = _finite_features(X)
    if not np.isfinite(X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)).all():
        raise ValueError("Feature sanitization failed: non-finite values remain in X")
    clf1.fit(
        X,
        y_direction
    )

    clf2.fit(
        X,
        y_direction
    )

    reg1.fit(
        X,
        y_return
    )

    reg2.fit(
        X,
        y_return
    )

    return {
        "clf1": clf1,
        "clf2": clf2,
        "reg1": reg1,
        "reg2": reg2,
    }


def predict_model(
    model,
    frame,
    features
):

    X = frame[
        features
    ].replace(
        [np.inf, -np.inf],
        np.nan
    )

    valid = (
        X.notna()
        .all(axis=1)
    )

    p = np.full(
        len(frame),
        np.nan
    )

    r = np.full(
        len(frame),
        np.nan
    )

    if not valid.any():

        return p, r

    xv = X.loc[
        valid
    ]

    p1 = model[
        "clf1"
    ].predict_proba(
        xv
    )[:, 1]

    p2 = model[
        "clf2"
    ].predict_proba(
        xv
    )[:, 1]

    r1 = model[
        "reg1"
    ].predict(
        xv
    )

    r2 = model[
        "reg2"
    ].predict(
        xv
    )

    p[
        valid.to_numpy()
    ] = (
        p1 + p2
    ) / 2

    r[
        valid.to_numpy()
    ] = (
        r1 + r2
    ) / 2

    return p, r


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward(
    history,
    evaluation,
    features,
    horizon
):

    dates = np.sort(
        evaluation[
            "date"
        ].unique()
    )

    outputs = []

    current_model = None

    for i, date in enumerate(
        dates
    ):

        if (
            current_model is None
            or
            i % RETRAIN_EVERY == 0
        ):

            train = history[
                history[
                    "date"
                ] < date
            ].copy()

            purge_before = (
                pd.Timestamp(date)
                -
                pd.Timedelta(
                    days=PURGE_DAYS
                )
            )

            train = train[
                train[
                    "date"
                ] < purge_before
            ]

            current_model = fit_model(
                train,
                features,
                horizon
            )

            if current_model is None:
                continue

        day = evaluation[
            evaluation[
                "date"
            ] == date
        ].copy()

        if day.empty:
            continue

        p, r = predict_model(
            current_model,
            day,
            features
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
# GEMINI HYBRID
# ============================================================

def make_hybrid(
    df,
    weight
):

    x = df.copy()

    quant_p = (
        x[
            "pred_probability"
        ]
        .clip(
            0.001,
            0.999
        )
    )

    quant_logit = np.log(
        quant_p /
        (1 - quant_p)
    )

    gs = (
        x[
            "gemini_score"
        ]
        *
        x[
            "gemini_confidence"
        ]
        *
        x[
            "gemini_materiality"
        ]
    ).clip(
        -1,
        1
    )

    gemini_p = (
        0.5 +
        0.5 * gs
    ).clip(
        0.001,
        0.999
    )

    gemini_logit = np.log(
        gemini_p /
        (1 - gemini_p)
    )

    hybrid_logit = (
        (1 - weight) *
        quant_logit
        +
        weight *
        gemini_logit
    )

    x[
        "hybrid_probability"
    ] = sigmoid(
        hybrid_logit
    )

    x[
        "gemini_probability"
    ] = gemini_p

    # Small return contribution.
    x[
        "gemini_return"
    ] = (
        0.005 * gs
    )

    x[
        "hybrid_return"
    ] = (
        (1 - weight) *
        x[
            "pred_return"
        ]
        +
        weight *
        (
            x[
                "pred_return"
            ]
            +
            x[
                "gemini_return"
            ]
        )
    )

    return x


# ============================================================
# VALIDATION
# ============================================================

def select_gemini_weight(
    validation,
    horizon
):

    usable = validation[
        validation[
            "gemini_available"
        ] == 1
    ].copy()

    if len(usable) < 50:

        return 0.0

    best_weight = 0.0
    best_score = -np.inf

    for w in GEMINI_WEIGHTS:

        candidate = make_hybrid(
            usable,
            w
        )

        selected = candidate[
            (
                candidate[
                    "hybrid_probability"
                ]
                >= TRADE_P_THRESHOLD
            )
            &
            (
                candidate[
                    "hybrid_return"
                ]
                >= TRADE_RETURN_THRESHOLD
            )
        ]

        if len(selected) < 30:
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
            np.sqrt(
                len(net)
            )
        )

        if win_rate < 0.50:
            score *= 0.75

        if (
            np.isfinite(score)
            and
            score > best_score
        ):

            best_score = score
            best_weight = float(w)

    return best_weight


# ============================================================
# STANDARD PERFORMANCE
# ============================================================

def performance(
    df,
    horizon,
    probability_col,
    return_col,
    model_name
):

    target = (
        f"target_{horizon}d"
    )

    q = df[
        [
            target,
            probability_col,
            return_col,
        ]
    ].dropna()

    if q.empty:
        return None

    y = (
        q[target] > 0
    ).astype(int)

    p = (
        q[
            probability_col
        ]
        .clip(
            0.001,
            0.999
        )
    )

    r = q[
        return_col
    ]

    directional = (
        (
            p >= 0.5
        )
        ==
        (
            y == 1
        )
    ).mean()

    selected = q[
        (
            p >= TRADE_P_THRESHOLD
        )
        &
        (
            r >= TRADE_RETURN_THRESHOLD
        )
    ]

    if selected.empty:

        selected_n = 0
        selected_win = np.nan
        selected_avg = np.nan
        profit_factor = np.nan

    else:

        net = (
            selected[target]
            -
            ROUND_TRIP_COST
        )

        selected_n = len(net)

        selected_win = (
            net > 0
        ).mean()

        selected_avg = (
            net.mean()
        )

        gains = (
            net[net > 0]
            .sum()
        )

        losses = (
            -net[net < 0]
            .sum()
        )

        profit_factor = (
            gains / losses
            if losses > 0
            else np.inf
        )

    return {
        "model":
            model_name,
        "horizon":
            horizon,
        "observations":
            len(q),
        "directional_accuracy":
            directional,
        "brier_score":
            brier_score_loss(
                y,
                p
            ),
        "log_loss":
            log_loss(
                y,
                p,
                labels=[0, 1]
            ),
        "return_mae":
            mean_absolute_error(
                q[target],
                r
            ),
        "mean_predicted_return":
            r.mean(),
        "mean_actual_return":
            q[target].mean(),
        "selected_n_p>=55":
            selected_n,
        "selected_win_rate":
            selected_win,
        "selected_average_net_return":
            selected_avg,
        "selected_profit_factor":
            profit_factor,
    }


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_stats(
    returns,
    iterations=BOOTSTRAP_ITERATIONS,
    confidence=CONFIDENCE_LEVEL
):

    arr = np.asarray(
        returns,
        dtype=float
    )

    arr = arr[np.isfinite(arr)]

    if len(arr) < 20:

        return {
            "n": len(arr),
            "mean": np.nan,
            "mean_ci_low": np.nan,
            "mean_ci_high": np.nan,
            "win_rate": np.nan,
            "win_ci_low": np.nan,
            "win_ci_high": np.nan,
            "prob_mean_gt_zero": np.nan,
        }

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    means = np.empty(
        iterations
    )

    wins = np.empty(
        iterations
    )

    n = len(arr)

    for i in range(
        iterations
    ):

        sample = rng.choice(
            arr,
            size=n,
            replace=True
        )

        means[i] = (
            sample.mean()
        )

        wins[i] = (
            sample > 0
        ).mean()

    alpha = (
        1 -
        confidence
    ) / 2

    mean_low = (
        np.quantile(
            means,
            alpha
        )
    )

    mean_high = (
        np.quantile(
            means,
            1 - alpha
        )
    )

    win_low = (
        np.quantile(
            wins,
            alpha
        )
    )

    win_high = (
        np.quantile(
            wins,
            1 - alpha
        )
    )

    return {
        "n": n,
        "mean": arr.mean(),
        "mean_ci_low": mean_low,
        "mean_ci_high": mean_high,
        "win_rate":
            (arr > 0).mean(),
        "win_ci_low": win_low,
        "win_ci_high": win_high,
        "prob_mean_gt_zero":
            float(
                (
                    means > 0
                ).mean()
            ),
    }


# ============================================================
# NON-OVERLAPPING
# ============================================================

def nonoverlap(
    df,
    horizon,
    probability_col,
    return_col,
    model_name
):

    q = df[
        (
            df[probability_col]
            >= TRADE_P_THRESHOLD
        )
        &
        (
            df[return_col]
            >= TRADE_RETURN_THRESHOLD
        )
    ].copy()

    if q.empty:
        return None

    q = q.sort_values(
        [
            "date",
            "ticker",
        ]
    )

    selected = []

    last_date = None

    for _, row in q.iterrows():

        current = pd.Timestamp(
            row["date"]
        )

        if (
            last_date is None
            or
            current >= (
                last_date
                +
                pd.Timedelta(
                    days=horizon + 1
                )
            )
        ):

            selected.append(
                row
            )

            last_date = current

    if not selected:
        return None

    x = pd.DataFrame(
        selected
    )

    net = (
        x[
            f"target_{horizon}d"
        ]
        -
        ROUND_TRIP_COST
    )

    gains = (
        net[net > 0].sum()
    )

    losses = (
        -net[net < 0].sum()
    )

    pf = (
        gains / losses
        if losses > 0
        else np.inf
    )

    boot = bootstrap_stats(
        net.to_numpy()
    )

    return {
        "model":
            model_name,
        "horizon":
            horizon,
        "trades":
            len(net),
        "win_rate":
            (net > 0).mean(),
        "average_net":
            net.mean(),
        "median_net":
            net.median(),
        "profit_factor":
            pf,
        "best":
            net.max(),
        "worst":
            net.min(),
        "net_sum_return":
            net.sum(),
        "bootstrap_mean_ci_low":
            boot["mean_ci_low"],
        "bootstrap_mean_ci_high":
            boot["mean_ci_high"],
        "bootstrap_win_ci_low":
            boot["win_ci_low"],
        "bootstrap_win_ci_high":
            boot["win_ci_high"],
        "prob_mean_gt_zero":
            boot["prob_mean_gt_zero"],
    }


# ============================================================
# SAFE PORTFOLIO TEST
# ============================================================

def portfolio_test(
    df,
    horizon,
    probability_col,
    return_col,
    model_name
):

    q = df[
        (
            df[probability_col]
            >= TRADE_P_THRESHOLD
        )
        &
        (
            df[return_col]
            >= TRADE_RETURN_THRESHOLD
        )
    ].copy()

    if q.empty:
        return None, pd.DataFrame()

    q[
        "selection_score"
    ] = (
        q[probability_col]
        *
        q[return_col].clip(
            lower=0
        )
    )

    # One position per date.
    q = (
        q.sort_values(
            [
                "date",
                "selection_score",
            ],
            ascending=[
                True,
                False,
            ]
        )
        .groupby(
            "date",
            as_index=False
        )
        .head(1)
        .sort_values(
            "date"
        )
    )

    # Remove any row that has no realized target.
    target = (
        f"target_{horizon}d"
    )

    q = q.dropna(
        subset=[
            target
        ]
    )

    if q.empty:
        return None, pd.DataFrame()

    capital = 100000.0

    rows = []

    peak = capital
    max_drawdown = 0.0

    for _, row in q.iterrows():

        gross_return = float(
            row[target]
        )

        net_return = (
            gross_return -
            ROUND_TRIP_COST
        )

        starting = capital

        capital = (
            capital *
            (
                1 +
                net_return
            )
        )

        peak = max(
            peak,
            capital
        )

        drawdown = (
            capital / peak -
            1
        )

        max_drawdown = min(
            max_drawdown,
            drawdown
        )

        rows.append(
            {
                "date":
                    row["date"],
                "ticker":
                    row["ticker"],
                "gross_return":
                    gross_return,
                "net_return":
                    net_return,
                "starting_capital":
                    starting,
                "ending_capital":
                    capital,
                "drawdown":
                    drawdown,
            }
        )

    trades = pd.DataFrame(
        rows
    )

    if trades.empty:
        return None, trades

    # Approximate CAGR from calendar span of the actual test trades.
    first_date = pd.Timestamp(
        trades["date"].min()
    )

    last_date = pd.Timestamp(
        trades["date"].max()
    )

    years = max(
        (
            last_date -
            first_date
        ).days / 365.25,
        1 / 365.25
    )

    total_return = (
        capital / 100000.0 -
        1
    )

    cagr = (
        (capital / 100000.0)
        ** (1 / years)
        - 1
    )

    # Capital-change returns for the sequence.
    seq_returns = (
        trades["ending_capital"]
        /
        trades["starting_capital"]
        -
        1
    )

    if (
        len(seq_returns) > 1
        and
        seq_returns.std() > 0
    ):

        # This is a trade-sequence Sharpe, not a daily Sharpe.
        sharpe = (
            seq_returns.mean()
            /
            seq_returns.std()
        ) * np.sqrt(
            max(
                len(seq_returns),
                1
            )
        )

    else:

        sharpe = np.nan

    result = {
        "model":
            model_name,
        "horizon":
            horizon,
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
        "trade_sequence_sharpe":
            sharpe,
        "completed_trades":
            len(trades),
    }

    return result, trades


# ============================================================
# BENCHMARKS
# ============================================================

def build_nifty_benchmark():

    raw = yf.download(
        "^NSEI",
        period=f"{YEARS}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    px = normalize_ohlcv(
        raw
    )

    if px.empty:
        return pd.DataFrame()

    close = px["Close"]

    records = []

    for h in HORIZONS:

        r = (
            close.shift(-h) /
            close -
            1
        ).dropna()

        records.append(
            {
                "horizon":
                    h,
                "observations":
                    len(r),
                "win_rate":
                    (r > 0).mean(),
                "average_return":
                    r.mean(),
                "median_return":
                    r.median(),
            }
        )

    return pd.DataFrame(
        records
    )


def build_equal_weight_benchmark(
    data
):

    records = []

    for h in HORIZONS:

        grouped = []

        for date, g in data.groupby(
            "date"
        ):

            returns = (
                g[
                    f"target_{h}d"
                ]
                .dropna()
            )

            if len(returns) > 0:

                grouped.append(
                    returns.mean()
                )

        if not grouped:
            continue

        r = pd.Series(
            grouped
        )

        records.append(
            {
                "horizon":
                    h,
                "observations":
                    len(r),
                "win_rate":
                    (r > 0).mean(),
                "average_return":
                    r.mean(),
                "median_return":
                    r.median(),
            }
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# YEARLY STABILITY
# ============================================================

def yearly_stability(
    df,
    horizon,
    probability_col,
    return_col,
    model_name
):

    x = df.copy()

    x["year"] = (
        pd.to_datetime(
            x["date"]
        )
        .dt.year
    )

    rows = []

    for year, g in x.groupby(
        "year"
    ):

        target = (
            f"target_{horizon}d"
        )

        selected = g[
            (
                g[
                    probability_col
                ] >= TRADE_P_THRESHOLD
            )
            &
            (
                g[
                    return_col
                ] >= TRADE_RETURN_THRESHOLD
            )
        ].copy()

        if selected.empty:

            rows.append(
                {
                    "model":
                        model_name,
                    "horizon":
                        horizon,
                    "year":
                        year,
                    "trades":
                        0,
                    "win_rate":
                        np.nan,
                    "average_net":
                        np.nan,
                    "profit_factor":
                        np.nan,
                }
            )

            continue

        net = (
            selected[target]
            -
            ROUND_TRIP_COST
        )

        gains = (
            net[net > 0]
            .sum()
        )

        losses = (
            -net[net < 0]
            .sum()
        )

        rows.append(
            {
                "model":
                    model_name,
                "horizon":
                    horizon,
                "year":
                    year,
                "trades":
                    len(net),
                "win_rate":
                    (net > 0).mean(),
                "average_net":
                    net.mean(),
                "profit_factor":
                    (
                        gains / losses
                        if losses > 0
                        else np.inf
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MARKET REGIME STABILITY
# ============================================================

def add_regime(
    df
):

    x = df.copy()

    # Market regime built from each signal date's
    # median stock 20-day momentum.
    daily = (
        x.groupby(
            "date"
        )[
            "ret_20"
        ]
        .median()
        .rename(
            "market_momentum_20"
        )
    )

    x = x.merge(
        daily,
        left_on="date",
        right_index=True,
        how="left",
    )

    x["regime"] = np.select(
        [
            x[
                "market_momentum_20"
            ] > 0.05,

            x[
                "market_momentum_20"
            ] < -0.05,
        ],
        [
            "BULL",
            "BEAR",
        ],
        default="NEUTRAL",
    )

    return x


def regime_stability(
    df,
    horizon,
    probability_col,
    return_col,
    model_name
):

    x = add_regime(
        df
    )

    rows = []

    target = (
        f"target_{horizon}d"
    )

    for regime, g in x.groupby(
        "regime"
    ):

        selected = g[
            (
                g[
                    probability_col
                ] >= TRADE_P_THRESHOLD
            )
            &
            (
                g[
                    return_col
                ] >= TRADE_RETURN_THRESHOLD
            )
        ]

        if selected.empty:

            rows.append(
                {
                    "model":
                        model_name,
                    "horizon":
                        horizon,
                    "regime":
                        regime,
                    "trades":
                        0,
                    "win_rate":
                        np.nan,
                    "average_net":
                        np.nan,
                    "profit_factor":
                        np.nan,
                }
            )

            continue

        net = (
            selected[target]
            -
            ROUND_TRIP_COST
        )

        gains = (
            net[net > 0]
            .sum()
        )

        losses = (
            -net[net < 0]
            .sum()
        )

        rows.append(
            {
                "model":
                    model_name,
                "horizon":
                    horizon,
                "regime":
                    regime,
                "trades":
                    len(net),
                "win_rate":
                    (net > 0).mean(),
                "average_net":
                    net.mean(),
                "profit_factor":
                    (
                        gains / losses
                        if losses > 0
                        else np.inf
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print(
        f"{VERSION} — "
        "POINT-IN-TIME QUANT + GEMINI"
    )
    print("=" * 78)

    print(
        f"Revision: {REVISION}"
    )

    print(
        f"yfinance: "
        f"{yf.__version__}"
    )

    print(
        f"Backtest: {YEARS} years"
    )

    print(
        f"Round-trip cost: "
        f"{ROUND_TRIP_COST:.3%}"
    )

    gemini = (
        load_historical_gemini()
    )

    data = build_dataset(
        gemini
    )

    print(
        "FEATURE/TARGET LEAKAGE CHECK: PASS"
    )

    print(
        "Point-in-time Gemini timestamp rule: "
        "published_at <= signal date"
    )

    print(
        f"Observations: "
        f"{len(data):,}"
    )

    print(
        f"Symbols: "
        f"{data['ticker'].nunique()}"
    )

    print(
        f"Signal dates: "
        f"{data['date'].nunique()}"
    )

    print(
        f"Gemini coverage: "
        f"{data['gemini_available'].mean():.2%}"
    )

    dates = np.sort(
        data["date"].unique()
    )

    development_end = dates[
        int(
            len(dates) * 0.50
        )
    ]

    validation_end = dates[
        int(
            len(dates) * 0.75
        )
    ]

    development = data[
        data["date"] <
        development_end
    ].copy()

    validation = data[
        (
            data["date"] >=
            development_end
        )
        &
        (
            data["date"] <
            validation_end
        )
    ].copy()

    oos = data[
        data["date"] >=
        validation_end
    ].copy()

    print(
        f"Development: "
        f"{len(development):,}"
    )

    print(
        f"Validation: "
        f"{len(validation):,}"
    )

    print(
        f"OOS: "
        f"{len(oos):,}"
    )

    all_metrics = []
    all_nonoverlap = []
    all_portfolio = []
    all_yearly = []
    all_regime = []
    weight_rows = []

    for horizon in HORIZONS:

        print()
        print(
            "=" * 70
        )
        print(
            f"HORIZON {horizon}D"
        )
        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_pred = walk_forward(
            development,
            validation,
            QUANT_FEATURES,
            horizon
        )

        if val_pred.empty:

            print(
                "WARNING: validation "
                "prediction set empty."
            )

            selected_weight = 0.0

        else:

            selected_weight = (
                select_gemini_weight(
                    val_pred,
                    horizon
                )
            )

        print(
            f"Validation-selected Gemini "
            f"weight: {selected_weight:.2f}"
        )

        weight_rows.append(
            {
                "horizon":
                    horizon,
                "selected_gemini_weight":
                    selected_weight,
            }
        )

        # ----------------------------------------------------
        # OOS QUANT
        # ----------------------------------------------------

        quant_oos = walk_forward(
            development,
            oos,
            QUANT_FEATURES,
            horizon
        )

        if quant_oos.empty:

            print(
                "WARNING: OOS prediction set empty."
            )

            continue

        quant_oos.to_csv(
            AUDIT_DIR /
            f"v6_6_1_quant_oos_h{horizon}.csv",
            index=False
        )

        qres = performance(
            quant_oos,
            horizon,
            "pred_probability",
            "pred_return",
            "quant"
        )

        if qres:
            all_metrics.append(
                qres
            )

        qno = nonoverlap(
            quant_oos,
            horizon,
            "pred_probability",
            "pred_return",
            "quant"
        )

        if qno:
            all_nonoverlap.append(
                qno
            )

        qport, qtrades = (
            portfolio_test(
                quant_oos,
                horizon,
                "pred_probability",
                "pred_return",
                "quant"
            )
        )

        if qport:
            all_portfolio.append(
                qport
            )

        if not qtrades.empty:

            qtrades.to_csv(
                AUDIT_DIR /
                f"v6_6_1_quant_portfolio_trades_h{horizon}.csv",
                index=False
            )

        qyear = yearly_stability(
            quant_oos,
            horizon,
            "pred_probability",
            "pred_return",
            "quant"
        )

        if not qyear.empty:
            all_yearly.append(
                qyear
            )

        qregime = regime_stability(
            quant_oos,
            horizon,
            "pred_probability",
            "pred_return",
            "quant"
        )

        if not qregime.empty:
            all_regime.append(
                qregime
            )

        # ----------------------------------------------------
        # GEMINI/HYBRID
        # ----------------------------------------------------

        coverage = (
            quant_oos[
                "gemini_available"
            ].sum()
        )

        if coverage < 50:

            print(
                "Gemini: NOT TESTED — "
                f"only {coverage} usable historical "
                "Gemini observations in OOS."
            )

            continue

        hybrid = make_hybrid(
            quant_oos,
            selected_weight
        )

        hybrid.to_csv(
            AUDIT_DIR /
            f"v6_6_1_hybrid_oos_h{horizon}.csv",
            index=False
        )

        # Gemini-only.
        gres = performance(
            hybrid,
            horizon,
            "gemini_probability",
            "gemini_return",
            "gemini_only"
        )

        if gres:
            all_metrics.append(
                gres
            )

        # Hybrid.
        hres = performance(
            hybrid,
            horizon,
            "hybrid_probability",
            "hybrid_return",
            "hybrid"
        )

        if hres:
            all_metrics.append(
                hres
            )

        hno = nonoverlap(
            hybrid,
            horizon,
            "hybrid_probability",
            "hybrid_return",
            "hybrid"
        )

        if hno:
            all_nonoverlap.append(
                hno
            )

        hport, htrades = (
            portfolio_test(
                hybrid,
                horizon,
                "hybrid_probability",
                "hybrid_return",
                "hybrid"
            )
        )

        if hport:
            all_portfolio.append(
                hport
            )

        if not htrades.empty:

            htrades.to_csv(
                AUDIT_DIR /
                f"v6_6_1_hybrid_portfolio_trades_h{horizon}.csv",
                index=False
            )

        hyear = yearly_stability(
            hybrid,
            horizon,
            "hybrid_probability",
            "hybrid_return",
            "hybrid"
        )

        if not hyear.empty:
            all_yearly.append(
                hyear
            )

        hregime = regime_stability(
            hybrid,
            horizon,
            "hybrid_probability",
            "hybrid_return",
            "hybrid"
        )

        if not hregime.empty:
            all_regime.append(
                hregime
            )

    # ========================================================
    # BENCHMARKS
    # ========================================================

    nifty = build_nifty_benchmark()

    equal_weight = (
        build_equal_weight_benchmark(
            data
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )

    nonoverlap_df = pd.DataFrame(
        all_nonoverlap
    )

    portfolio_df = pd.DataFrame(
        all_portfolio
    )

    yearly_df = pd.concat(
        all_yearly,
        ignore_index=True
    ) if all_yearly else pd.DataFrame()

    regime_df = pd.concat(
        all_regime,
        ignore_index=True
    ) if all_regime else pd.DataFrame()

    weights_df = pd.DataFrame(
        weight_rows
    )

    metrics_df.to_csv(
        AUDIT_DIR /
        "v6_6_1_oos_model_comparison.csv",
        index=False
    )

    nonoverlap_df.to_csv(
        AUDIT_DIR /
        "v6_6_1_nonoverlap_oos.csv",
        index=False
    )

    portfolio_df.to_csv(
        AUDIT_DIR /
        "v6_6_1_portfolio_oos.csv",
        index=False
    )

    yearly_df.to_csv(
        AUDIT_DIR /
        "v6_6_1_yearly_stability.csv",
        index=False
    )

    regime_df.to_csv(
        AUDIT_DIR /
        "v6_6_1_regime_stability.csv",
        index=False
    )

    weights_df.to_csv(
        AUDIT_DIR /
        "v6_6_1_selected_gemini_weights.csv",
        index=False
    )

    nifty.to_csv(
        AUDIT_DIR /
        "v6_6_1_nifty_benchmark.csv",
        index=False
    )

    equal_weight.to_csv(
        AUDIT_DIR /
        "v6_6_1_equal_weight_benchmark.csv",
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "V6.6.1 OOS MODEL COMPARISON"
    )
    print("=" * 78)

    if metrics_df.empty:
        print(
            "No model comparison results."
        )
    else:
        print(
            metrics_df.to_string(
                index=False
            )
        )

    print()
    print(
        "NON-OVERLAPPING OOS"
    )

    if nonoverlap_df.empty:
        print(
            "No non-overlapping results."
        )
    else:
        print(
            nonoverlap_df.to_string(
                index=False
            )
        )

    print()
    print(
        "PORTFOLIO OOS"
    )

    if portfolio_df.empty:
        print(
            "No portfolio results."
        )
    else:
        print(
            portfolio_df.to_string(
                index=False
            )
        )

    print()
    print(
        "YEARLY STABILITY"
    )

    if yearly_df.empty:
        print(
            "No yearly stability results."
        )
    else:
        print(
            yearly_df.to_string(
                index=False
            )
        )

    print()
    print(
        "REGIME STABILITY"
    )

    if regime_df.empty:
        print(
            "No regime stability results."
        )
    else:
        print(
            regime_df.to_string(
                index=False
            )
        )

    print()
    print(
        "NIFTY BENCHMARK"
    )

    print(
        nifty.to_string(
            index=False
        )
    )

    print()
    print(
        "EQUAL-WEIGHT BENCHMARK"
    )

    print(
        equal_weight.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print(
        "V6.6.1 BACKTEST COMPLETED"
    )
    print("=" * 78)

    if gemini.empty:

        print(
            "GEMINI STATUS: NOT TESTED"
        )

        print(
            "Create data/historical_gemini.csv "
            "for the genuine Gemini experiment."
        )

    else:

        print(
            "GEMINI STATUS: HISTORICAL "
            "POINT-IN-TIME DATA USED"
        )

    print()
    print(
        "AUDIT GUARANTEES:"
    )

    print(
        "1. Forward targets are excluded "
        "from FEATURES."
    )

    print(
        "2. Historical Gemini is joined only "
        "from published_at <= signal date."
    )

    print(
        "3. Ten-day purge is applied before fitting."
    )

    print(
        "4. Gemini weight is selected on validation only."
    )

    print(
        "5. OOS labels are not used to select weights."
    )

    print(
        "6. Portfolio rows without realized targets "
        "are excluded, preventing NaN equity."
    )

    print(
        "7. Yearly and regime stability are reported."
    )

    print(
        "8. Bootstrap confidence intervals are reported "
        "for non-overlapping selected trades."
    )

    print(
        "9. Historical simulation does not guarantee "
        "future performance."
    )


if __name__ == "__main__":
    main()
