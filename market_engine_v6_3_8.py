"""
============================================================
MULTI-FACTOR MARKET ALERT V6.3.8
============================================================

PURPOSE
-------
Short-term Indian equity research screener with:

1. Empirical analogue probabilities
2. 3-session and 5-session expected returns
3. ATR-based risk management
4. Risk/reward validation
5. Market regime filter
6. TRADE / WATCH / REJECT classification
7. Probability calibration audit
8. Historical outcome recording
9. Filter diagnostics
10. Safe environment-variable handling
11. Weekend protection
12. IPO retrieval
13. Telegram delivery
14. CSV audit history

IMPORTANT
---------
This is a probabilistic research system.
It does NOT guarantee profit.

Do not interpret P3/P5 as guaranteed probabilities
of profit.

V6.3.8 deliberately does NOT loosen the trade filters
just to create more trades.

Instead, it measures whether the model's probabilities
actually correspond to future outcomes.

============================================================
"""

from __future__ import annotations

import os
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# VERSION
# ============================================================

VERSION = "V6.3.8"


# ============================================================
# INDIA TIME
# ============================================================

IST = timezone(
    timedelta(hours=5, minutes=30)
)


def now_ist():
    return datetime.now(IST)


# ============================================================
# SAFE ENVIRONMENT VARIABLES
# ============================================================

def safe_env_float(name, default):
    raw = os.getenv(name)

    if raw is None:
        return float(default)

    raw = str(raw).strip()

    if raw == "":
        return float(default)

    try:
        value = float(raw)

        if not np.isfinite(value):
            return float(default)

        return value

    except Exception:
        print(
            f"WARNING: invalid {name}={raw!r}; "
            f"using default {default}"
        )
        return float(default)


# ============================================================
# CAPITAL / RISK
# ============================================================

CAPITAL = safe_env_float(
    "TRADING_CAPITAL",
    100000
)

MAX_RISK_PCT = safe_env_float(
    "MAX_RISK_PCT",
    0.01
)

if CAPITAL <= 0:
    CAPITAL = 100000

if MAX_RISK_PCT <= 0:
    MAX_RISK_PCT = 0.01

if MAX_RISK_PCT > 0.05:
    MAX_RISK_PCT = 0.05


# ============================================================
# MODEL PARAMETERS
# ============================================================

LOOKBACK_DAYS = 756

N_ANALOGUES = 30

MIN_ANALOGUES = 15

MIN_P3 = 0.52
MIN_P5 = 0.51

MIN_ER3 = 0.0010
MIN_ER5 = 0.0015

MIN_RR1 = 1.00
MIN_RR2 = 1.20

MIN_QUALITY = 0.25

MIN_VOLUME_RATIO = 0.80

MAX_RSI = 75.0

MAX_WATCH_RESULTS = 5

AUDIT_DIR = Path("audit")

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STOCK UNIVERSE
# ============================================================

SYMBOLS = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "BPCL",
    "BRITANNIA",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "ICICIGI",
    "ICICIPRULI",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "TATAMOTORS",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
    "WIPRO",
    "VEDL",
    "AUROPHARMA",
    "BOSCHLTD",
    "SAIL",
    "DLF",
    "NAUKRI",
]


FEATURES = [
    "rsi",
    "ret3",
    "ret5",
    "ret10",
    "dist_sma20",
    "dist_sma50",
    "sma50_slope5",
    "atr_pct",
    "vol20",
    "volume_ratio",
]


# ============================================================
# HELPERS
# ============================================================

def ticker(symbol):
    return f"{symbol}.NS"


def safe_float(value, default=np.nan):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def format_money(value):

    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"₹{value:,.2f}"


def is_trading_day():

    return now_ist().weekday() < 5


# ============================================================
# DATA CLEANING
# ============================================================

def clean_frame(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for column in required:

        if column not in df.columns:
            df[column] = np.nan

    df = df[required].copy()

    df.index = pd.to_datetime(
        df.index,
        errors="coerce"
    )

    df = df[
        ~df.index.isna()
    ]

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df.sort_index()


def split_download(raw):

    result = {}

    if raw is None or raw.empty:
        return result

    if not isinstance(
        raw.columns,
        pd.MultiIndex
    ):

        cleaned = clean_frame(raw)

        if cleaned is not None:
            result["SINGLE"] = cleaned

        return result

    level0 = list(
        raw.columns
        .get_level_values(0)
        .unique()
    )

    level1 = list(
        raw.columns
        .get_level_values(1)
        .unique()
    )

    if "Close" in level0:

        for name in level1:

            try:

                part = raw.xs(
                    name,
                    axis=1,
                    level=1,
                    drop_level=True
                )

                cleaned = clean_frame(part)

                if cleaned is not None:
                    result[str(name)] = cleaned

            except Exception:
                continue

    else:

        for name in level0:

            try:

                part = raw.xs(
                    name,
                    axis=1,
                    level=0,
                    drop_level=True
                )

                cleaned = clean_frame(part)

                if cleaned is not None:
                    result[str(name)] = cleaned

            except Exception:
                continue

    return result


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    return rsi.where(
        avg_loss != 0,
        100.0
    )


# ============================================================
# FEATURES
# ============================================================

def add_features(df):

    x = df.copy()

    close = x["Close"]

    high = x["High"]

    low = x["Low"]

    volume = x["Volume"]

    x["sma20"] = (
        close
        .rolling(20)
        .mean()
    )

    x["sma50"] = (
        close
        .rolling(50)
        .mean()
    )

    x["sma200"] = (
        close
        .rolling(200)
        .mean()
    )

    x["ema10"] = (
        close
        .ewm(
            span=10,
            adjust=False
        )
        .mean()
    )

    x["rsi"] = calculate_rsi(
        close
    )

    x["ret1"] = close.pct_change(1)
    x["ret3"] = close.pct_change(3)
    x["ret5"] = close.pct_change(5)
    x["ret10"] = close.pct_change(10)
    x["ret20"] = close.pct_change(20)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,

            (
                high
                -
                previous_close
            ).abs(),

            (
                low
                -
                previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    x["atr14"] = (
        true_range
        .rolling(14)
        .mean()
    )

    x["atr_pct"] = (
        x["atr14"]
        /
        close
    )

    x["vol20"] = (
        x["ret1"]
        .rolling(20)
        .std()
    )

    volume_mean = (
        volume
        .rolling(20)
        .mean()
    )

    x["volume_ratio"] = (
        volume
        /
        volume_mean
    )

    x["dist_sma20"] = (
        close
        /
        x["sma20"]
        -
        1
    )

    x["dist_sma50"] = (
        close
        /
        x["sma50"]
        -
        1
    )

    x["dist_sma200"] = (
        close
        /
        x["sma200"]
        -
        1
    )

    x["sma50_slope5"] = (
        x["sma50"]
        /
        x["sma50"].shift(5)
        -
        1
    )

    x["ema10_slope5"] = (
        x["ema10"]
        /
        x["ema10"].shift(5)
        -
        1
    )

    # Explicitly retained to prevent
    # KeyError: 'trend'
    x["trend"] = x["dist_sma50"]

    return x


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime(
    frames,
    signal_date
):

    nifty = frames.get("^NSEI")

    vix = frames.get("^INDIAVIX")

    if nifty is None or nifty.empty:

        return {
            "regime": "UNKNOWN",
            "nifty": np.nan,
            "sma50": np.nan,
            "vix": np.nan,
            "reason":
                "Nifty data unavailable",
        }

    n = nifty.copy()

    n.index = pd.to_datetime(
        n.index,
        errors="coerce"
    )

    n = n[
        ~n.index.isna()
    ]

    n = n[
        n.index <= pd.Timestamp(
            signal_date
        )
    ]

    n = n.dropna(
        subset=["Close"]
    )

    if len(n) < 55:

        return {
            "regime": "UNKNOWN",
            "nifty":
                safe_float(
                    n["Close"].iloc[-1]
                )
                if not n.empty
                else np.nan,
            "sma50": np.nan,
            "vix": np.nan,
            "reason":
                "Insufficient Nifty history",
        }

    close = n["Close"]

    sma50 = (
        close
        .rolling(50)
        .mean()
    )

    valid = pd.DataFrame(
        {
            "close": close,
            "sma50": sma50,
        }
    ).dropna()

    if len(valid) < 6:

        return {
            "regime": "UNKNOWN",
            "nifty":
                safe_float(
                    close.iloc[-1]
                ),
            "sma50": np.nan,
            "vix": np.nan,
            "reason":
                "SMA50 unavailable",
        }

    latest = valid.iloc[-1]

    previous = valid.iloc[-6]

    nifty_value = safe_float(
        latest["close"]
    )

    sma50_value = safe_float(
        latest["sma50"]
    )

    previous_sma50 = safe_float(
        previous["sma50"]
    )

    if (
        not np.isfinite(nifty_value)
        or
        not np.isfinite(sma50_value)
        or
        not np.isfinite(previous_sma50)
    ):

        regime = "UNKNOWN"

        reason = (
            "Invalid Nifty/SMA50 data"
        )

    else:

        above = (
            nifty_value
            >=
            sma50_value
        )

        rising = (
            sma50_value
            >
            previous_sma50
        )

        if above and rising:

            regime = "FAVORABLE"

            reason = (
                "Nifty above rising SMA50"
            )

        elif (
            not above
            and
            not rising
        ):

            regime = "UNFAVORABLE"

            reason = (
                "Nifty below falling SMA50"
            )

        else:

            regime = "MIXED"

            reason = (
                "Nifty/SMA50 signals mixed"
            )

    vix_value = np.nan

    if vix is not None and not vix.empty:

        vv = vix.copy()

        vv.index = pd.to_datetime(
            vv.index,
            errors="coerce"
        )

        vv = vv[
            ~vv.index.isna()
        ]

        vv = vv[
            vv.index <= pd.Timestamp(
                signal_date
            )
        ]

        vv = vv.dropna(
            subset=["Close"]
        )

        if not vv.empty:

            vix_value = safe_float(
                vv["Close"].iloc[-1]
            )

    return {
        "regime": regime,
        "nifty": nifty_value,
        "sma50": sma50_value,
        "vix": vix_value,
        "reason": reason,
    }


# ============================================================
# EMPIRICAL ANALOGUE MODEL
# ============================================================

def analogue_model(
    features,
    signal_date
):

    data = features.loc[
        features.index
        <=
        pd.Timestamp(signal_date)
    ].copy()

    data = data.dropna(
        subset=FEATURES + ["Close"]
    )

    if len(data) < 300:
        return None

    candidate = data.iloc[-1]

    # Exclude recent observations to reduce
    # overlapping future-window leakage.
    train = data.iloc[:-6].copy()

    if len(train) < 100:
        return None

    train = train.tail(
        LOOKBACK_DAYS
    ).copy()

    train["future1"] = (
        train["Close"].shift(-1)
        /
        train["Close"]
        -
        1
    )

    train["future3"] = (
        train["Close"].shift(-3)
        /
        train["Close"]
        -
        1
    )

    train["future5"] = (
        train["Close"].shift(-5)
        /
        train["Close"]
        -
        1
    )

    train = train.dropna(
        subset=[
            "future1",
            "future3",
            "future5",
        ]
    )

    if len(train) < 80:
        return None

    distances = np.zeros(
        len(train),
        dtype=float
    )

    for col in FEATURES:

        series = train[col]

        median = series.median()

        mad = (
            series
            -
            median
        ).abs().median()

        if (
            np.isfinite(mad)
            and
            mad > 0
        ):

            scale = (
                1.4826
                *
                mad
            )

        else:

            scale = series.std()

        if (
            not np.isfinite(scale)
            or
            scale <= 0
        ):

            scale = 1.0

        candidate_value = safe_float(
            candidate[col],
            median
        )

        distances += (
            (
                (
                    series.values
                    -
                    candidate_value
                )
                /
                scale
            )
            ** 2
        )

    distances = np.sqrt(
        distances
        /
        len(FEATURES)
    )

    train["distance"] = distances

    train = (
        train
        .sort_values(
            "distance"
        )
        .head(
            N_ANALOGUES
        )
    )

    if len(train) < MIN_ANALOGUES:
        return None

    weights = (
        1.0
        /
        (
            train["distance"]
            +
            0.10
        )
    )

    weights = (
        weights
        /
        weights.sum()
    )

    raw_p1 = float(
        (
            weights
            *
            (
                train["future1"]
                >
                0
            )
        ).sum()
    )

    raw_p3 = float(
        (
            weights
            *
            (
                train["future3"]
                >
                0
            )
        ).sum()
    )

    raw_p5 = float(
        (
            weights
            *
            (
                train["future5"]
                >
                0
            )
        ).sum()
    )

    er1 = float(
        (
            weights
            *
            train["future1"]
        ).sum()
    )

    er3 = float(
        (
            weights
            *
            train["future3"]
        ).sum()
    )

    er5 = float(
        (
            weights
            *
            train["future5"]
        ).sum()
    )

    std1 = safe_float(
        train["future1"].std(),
        0.05
    )

    std3 = safe_float(
        train["future3"].std(),
        0.05
    )

    std5 = safe_float(
        train["future5"].std(),
        0.07
    )

    # Probability shrinkage toward 50%.
    shrinkage = min(
        0.45,
        12.0
        /
        max(
            12.0,
            len(train)
        )
    )

    p1 = (
        (1 - shrinkage)
        * raw_p1
        +
        shrinkage
        * 0.50
    )

    p3 = (
        (1 - shrinkage)
        * raw_p3
        +
        shrinkage
        * 0.50
    )

    p5 = (
        (1 - shrinkage)
        * raw_p5
        +
        shrinkage
        * 0.50
    )

    sample_quality = min(
        1.0,
        len(train)
        /
        N_ANALOGUES
    )

    dispersion3 = min(
        1.0,
        std3
        /
        0.035
    )

    dispersion5 = min(
        1.0,
        std5
        /
        0.055
    )

    quality = (
        0.40
        *
        sample_quality
        +
        0.30
        *
        (1 - dispersion3)
        +
        0.30
        *
        (1 - dispersion5)
    )

    quality = max(
        0.0,
        min(
            1.0,
            quality
        )
    )

    return {

        "raw_p1": raw_p1,
        "raw_p3": raw_p3,
        "raw_p5": raw_p5,

        "p1": p1,
        "p3": p3,
        "p5": p5,

        "er1": er1,
        "er3": er3,
        "er5": er5,

        "std1": std1,
        "std3": std3,
        "std5": std5,

        "quality": quality,

        "analogues": len(train),

        "nearest_distance":
            float(
                train[
                    "distance"
                ].iloc[0]
            ),
    }


# ============================================================
# TRADE LEVELS
# ============================================================

def calculate_trade_levels(
    price,
    atr_pct,
    er3,
    er5
):

    price = safe_float(price)

    atr_pct = safe_float(
        atr_pct
    )

    er3 = safe_float(
        er3,
        0.0
    )

    er5 = safe_float(
        er5,
        0.0
    )

    if (
        not np.isfinite(price)
        or
        price <= 0
        or
        not np.isfinite(atr_pct)
        or
        atr_pct <= 0
    ):

        return None

    stop_distance = min(
        0.05,
        max(
            0.012,
            1.5 * atr_pct
        )
    )

    stop = (
        price
        *
        (
            1
            -
            stop_distance
        )
    )

    if (
        not np.isfinite(stop)
        or
        stop <= 0
        or
        stop >= price
    ):

        return None

    risk_per_share = (
        price
        -
        stop
    )

    if risk_per_share <= 0:
        return None

    target1_return = max(
        0.0,
        min(
            0.08,
            er3
        )
    )

    target2_return = max(
        target1_return,
        min(
            0.12,
            er5
        )
    )

    target1 = (
        price
        *
        (
            1
            +
            target1_return
        )
    )

    target2 = (
        price
        *
        (
            1
            +
            target2_return
        )
    )

    rr1 = (
        target1
        -
        price
    ) / risk_per_share

    rr2 = (
        target2
        -
        price
    ) / risk_per_share

    if not np.isfinite(rr1):
        rr1 = 0.0

    if not np.isfinite(rr2):
        rr2 = 0.0

    return {

        "stop": stop,

        "target1": target1,

        "target2": target2,

        "risk_per_share":
            risk_per_share,

        "rr1": rr1,

        "rr2": rr2,

    }


# ============================================================
# POSITION SIZE
# ============================================================

def position_size(
    price,
    stop
):

    price = safe_float(price)

    stop = safe_float(stop)

    if (
        not np.isfinite(price)
        or
        not np.isfinite(stop)
        or
        price <= 0
        or
        stop <= 0
        or
        price <= stop
    ):

        return (
            0,
            0.0,
            0.0
        )

    risk_budget = (
        CAPITAL
        *
        MAX_RISK_PCT
    )

    risk_per_share = (
        price
        -
        stop
    )

    if risk_per_share <= 0:

        return (
            0,
            0.0,
            0.0
        )

    shares_by_risk = math.floor(
        risk_budget
        /
        risk_per_share
    )

    shares_by_capital = math.floor(
        CAPITAL
        /
        price
    )

    shares = max(
        0,
        min(
            shares_by_risk,
            shares_by_capital
        )
    )

    exposure = (
        shares
        *
        price
    )

    planned_loss = (
        shares
        *
        risk_per_share
    )

    return (
        shares,
        exposure,
        planned_loss
    )


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    row,
    regime
):

    p3 = safe_float(
        row.get("p3"),
        0.50
    )

    p5 = safe_float(
        row.get("p5"),
        0.50
    )

    er3 = safe_float(
        row.get("er3"),
        0.0
    )

    er5 = safe_float(
        row.get("er5"),
        0.0
    )

    rsi = safe_float(
        row.get("rsi"),
        50.0
    )

    volume = safe_float(
        row.get(
            "volume_ratio"
        ),
        1.0
    )

    trend = safe_float(
        row.get(
            "trend",
            row.get(
                "dist_sma50",
                0.0
            )
        ),
        0.0
    )

    quality = safe_float(
        row.get(
            "quality",
            0.5
        ),
        0.5
    )

    score = 0.0

    score += (
        p3
        -
        0.50
    ) * 3.0

    score += (
        p5
        -
        0.50
    ) * 3.0

    score += (
        er3
        *
        20.0
    )

    score += (
        er5
        *
        15.0
    )

    score += (
        quality
        *
        0.30
    )

    if trend > 0:
        score += 0.20

    elif trend < -0.03:
        score -= 0.20

    if volume >= 1.20:
        score += 0.15

    elif volume >= 1.00:
        score += 0.05

    elif volume < 0.80:
        score -= 0.10

    if 50 <= rsi <= 68:
        score += 0.10

    elif rsi > 72:
        score -= 0.15

    elif rsi < 40:
        score -= 0.05

    if regime == "FAVORABLE":
        score += 0.15

    elif regime == "MIXED":
        score -= 0.05

    elif regime == "UNFAVORABLE":
        score -= 0.50

    elif regime == "UNKNOWN":
        score -= 0.20

    if not np.isfinite(score):
        score = -999.0

    return float(score)


# ============================================================
# CANDIDATE EVALUATION
# ============================================================

def evaluate_candidate(
    symbol,
    row,
    regime
):

    price = safe_float(
        row.get("Close")
    )

    rsi = safe_float(
        row.get("rsi")
    )

    volume = safe_float(
        row.get("volume_ratio")
    )

    trend = safe_float(
        row.get(
            "trend",
            row.get(
                "dist_sma50",
                0.0
            )
        ),
        0.0
    )

    p3 = safe_float(
        row.get("p3")
    )

    p5 = safe_float(
        row.get("p5")
    )

    er3 = safe_float(
        row.get("er3")
    )

    er5 = safe_float(
        row.get("er5")
    )

    quality = safe_float(
        row.get("quality")
    )

    levels = calculate_trade_levels(
        price,
        row.get("atr_pct"),
        er3,
        er5
    )

    if levels is None:
        return None

    (
        shares,
        exposure,
        planned_loss
    ) = position_size(
        price,
        levels["stop"]
    )

    score = calculate_score(
        row,
        regime
    )

    checks = {

        "p3":
            np.isfinite(p3)
            and
            p3 >= MIN_P3,

        "p5":
            np.isfinite(p5)
            and
            p5 >= MIN_P5,

        "er3":
            np.isfinite(er3)
            and
            er3 >= MIN_ER3,

        "er5":
            np.isfinite(er5)
            and
            er5 >= MIN_ER5,

        "rr1":
            levels["rr1"]
            >=
            MIN_RR1,

        "rr2":
            levels["rr2"]
            >=
            MIN_RR2,

        "quality":
            np.isfinite(quality)
            and
            quality >= MIN_QUALITY,

        "trend":
            np.isfinite(trend)
            and
            trend > -0.03,

        "rsi":
            np.isfinite(rsi)
            and
            rsi <= MAX_RSI,

        "volume":
            np.isfinite(volume)
            and
            volume >= MIN_VOLUME_RATIO,

        "regime":
            regime
            in
            (
                "FAVORABLE",
                "MIXED"
            ),
    }

    failed = [
        key
        for key, value
        in checks.items()
        if not value
    ]

    passed = sum(
        bool(value)
        for value in checks.values()
    )

    pass_rate = (
        passed
        /
        len(checks)
    )

    if all(checks.values()):

        action = "TRADE"

    elif pass_rate >= 0.60:

        action = "WATCH"

    else:

        action = "REJECT"

    if regime in (
        "UNKNOWN",
        "UNFAVORABLE"
    ):

        action = "REJECT"

    return {

        "symbol": symbol,

        "price": price,

        "p1": safe_float(
            row.get("p1")
        ),

        "p3": p3,

        "p5": p5,

        "er1": safe_float(
            row.get("er1")
        ),

        "er3": er3,

        "er5": er5,

        "rsi": rsi,

        "volume_ratio":
            volume,

        "trend":
            trend,

        "quality":
            quality,

        "score":
            score,

        "stop":
            levels["stop"],

        "target1":
            levels["target1"],

        "target2":
            levels["target2"],

        "rr1":
            levels["rr1"],

        "rr2":
            levels["rr2"],

        "shares":
            shares,

        "exposure":
            exposure,

        "planned_loss":
            planned_loss,

        "action":
            action,

        "failed":
            ",".join(failed),

        "pass_rate":
            pass_rate,

    }


# ============================================================
# IPO DATA
# ============================================================

def get_ipo_information():

    url = (
        "https://www.nseindia.com/"
        "api/ipo-current-issue"
    )

    headers = {

        "User-Agent":
            (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120 Safari/537.36"
            ),

        "Accept":
            "application/json,text/plain,*/*",

        "Referer":
            "https://www.nseindia.com/",
    }

    try:

        session = requests.Session()

        session.get(
            "https://www.nseindia.com/",
            headers=headers,
            timeout=10
        )

        response = session.get(
            url,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:

            return []

        data = response.json()

        if not isinstance(
            data,
            list
        ):

            return []

        results = []

        for item in data:

            if not isinstance(
                item,
                dict
            ):
                continue

            name = (
                item.get("companyName")
                or
                item.get("symbol")
                or
                item.get("name")
            )

            if name:
                results.append(
                    str(name)
                )

        return list(
            dict.fromkeys(
                results
            )
        )[:10]

    except Exception as exc:

        print(
            f"IPO retrieval error: {exc}"
        )

        return []


# ============================================================
# TELEGRAM
# ============================================================

def telegram_send(message):

    bot_token = (
        os.getenv(
            "TELEGRAM_BOT_TOKEN"
        )
        or
        os.getenv(
            "BOT_TOKEN"
        )
    )

    chat_id = (
        os.getenv(
            "TELEGRAM_CHAT_ID"
        )
        or
        os.getenv(
            "CHAT_ID"
        )
    )

    if not bot_token or not chat_id:

        print(
            "Telegram credentials not configured."
        )

        return False

    url = (
        "https://api.telegram.org/"
        f"bot{str(bot_token).strip()}/sendMessage"
    )

    try:

        response = requests.post(
            url,
            json={
                "chat_id":
                    str(chat_id).strip(),
                "text":
                    message,
            },
            timeout=20
        )

        if response.status_code != 200:

            print(
                "Telegram error:",
                response.status_code,
                response.text
            )

            return False

        print(
            "Telegram alert sent successfully."
        )

        return True

    except Exception as exc:

        print(
            f"Telegram error: {exc}"
        )

        return False


# ============================================================
# ALERT BUILDER
# ============================================================

def build_alert(
    market,
    candidates,
    ipo_names,
    trading_day
):

    timestamp = now_ist().strftime(
        "%d %b %Y, %H:%M IST"
    )

    lines = [

        (
            f"MULTI-FACTOR MARKET ALERT "
            f"{VERSION}"
        ),

        timestamp,

        "",

        (
            "MARKET STATUS: "
            +
            (
                "TRADING DAY"
                if trading_day
                else
                "WEEKEND / NON-TRADING DAY"
            )
        ),

        (
            f"MARKET REGIME: "
            f"{market['regime']}"
        ),

        (
            f"NIFTY: "
            f"{format_money(market['nifty'])}"
            f" | SMA50: "
            f"{format_money(market['sma50'])}"
        ),

        (
            f"INDIA VIX: "
            f"{format_money(market['vix'])}"
        ),

        (
            f"REGIME REASON: "
            f"{market['reason']}"
        ),

        "",

        "--- TOP SHORT-TERM TRADE SETUPS ---",
    ]

    trades = [
        x
        for x in candidates
        if x["action"] == "TRADE"
    ]

    watches = sorted(
        [
            x
            for x in candidates
            if x["action"] == "WATCH"
        ],
        key=lambda x: x["score"],
        reverse=True
    )

    if not trading_day:

        lines.extend(
            [
                "",
                "MARKET IS CLOSED.",
                (
                    "No new long position should "
                    "be initiated today."
                ),
            ]
        )

    if (
        trades
        and
        trading_day
        and
        market["regime"]
        in
        (
            "FAVORABLE",
            "MIXED"
        )
    ):

        for i, item in enumerate(
            trades[:3],
            start=1
        ):

            lines.extend(
                [
                    "",
                    (
                        f"{i}. "
                        f"{item['symbol']} — TRADE"
                    ),
                    (
                        f"Price: "
                        f"{format_money(item['price'])}"
                    ),
                    (
                        f"P(UP) 3D / 5D: "
                        f"{item['p3']:.1%} / "
                        f"{item['p5']:.1%}"
                    ),
                    (
                        f"Expected return 3D / 5D: "
                        f"{item['er3']:.2%} / "
                        f"{item['er5']:.2%}"
                    ),
                    (
                        f"Score: "
                        f"{item['score']:.3f}"
                        f" | RSI: "
                        f"{item['rsi']:.1f}"
                        f" | Volume: "
                        f"{item['volume_ratio']:.2f}x"
                    ),
                    (
                        f"Stop Loss: "
                        f"{format_money(item['stop'])}"
                    ),
                    (
                        f"Target 1: "
                        f"{format_money(item['target1'])}"
                    ),
                    (
                        f"Target 2: "
                        f"{format_money(item['target2'])}"
                    ),
                    (
                        f"Risk/Reward: "
                        f"{item['rr1']:.2f} / "
                        f"{item['rr2']:.2f}"
                    ),
                    "Expected holding: 1–5 sessions",
                    (
                        f"Suggested position: "
                        f"{item['shares']} shares "
                        f"≈ "
                        f"{format_money(item['exposure'])}"
                    ),
                    (
                        f"Maximum planned loss: "
                        f"{format_money(item['planned_loss'])}"
                    ),
                ]
            )

    else:

        if trading_day:

            lines.extend(
                [
                    "",
                    "NO VALID LONG TRADE TODAY",
                    (
                        "No candidate currently satisfies "
                        "all V6.3.8 probability, expected-return, "
                        "risk/reward and market filters."
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "--- BEST WATCHLIST SETUPS ---",
        ]
    )

    if watches:

        for i, item in enumerate(
            watches[:MAX_WATCH_RESULTS],
            start=1
        ):

            failed = (
                item["failed"]
                or
                "none"
            )

            lines.extend(
                [
                    "",
                    (
                        f"{i}. "
                        f"{item['symbol']} — WATCH"
                    ),
                    (
                        f"Price: "
                        f"{format_money(item['price'])}"
                    ),
                    (
                        f"P3 / P5: "
                        f"{item['p3']:.1%} / "
                        f"{item['p5']:.1%}"
                    ),
                    (
                        f"ER3 / ER5: "
                        f"{item['er3']:.2%} / "
                        f"{item['er5']:.2%}"
                    ),
                    (
                        f"RR1 / RR2: "
                        f"{item['rr1']:.2f} / "
                        f"{item['rr2']:.2f}"
                    ),
                    (
                        f"RSI: "
                        f"{item['rsi']:.1f}"
                        f" | Volume: "
                        f"{item['volume_ratio']:.2f}x"
                    ),
                    (
                        f"Failed filters: "
                        f"{failed}"
                    ),
                    "Action: WATCH / WAIT",
                ]
            )

    else:

        lines.append(
            "None."
        )

    lines.extend(
        [
            "",
            "--- MODEL VALIDATION STATUS ---",
            (
                "V6.3.8 records historical outcomes "
                "for later probability calibration."
            ),
            (
                "P(UP) is an empirical model estimate, "
                "not a guaranteed probability."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "--- IPO OPEN / UPCOMING ---",
        ]
    )

    if ipo_names:

        lines.append(
            (
                "IPO records retrieved. "
                "Verify issue dates, price band "
                "and subscription status directly "
                "before applying."
            )
        )

        for name in ipo_names:

            lines.append(
                f"• {name}"
            )

    else:

        lines.extend(
            [
                "IPO DATA UNAVAILABLE | RETRIEVAL FAILED",
                (
                    "Verify current/upcoming issues "
                    "directly on NSE."
                ),
            ]
        )

    lines.extend(
        [
            "",
            (
                f"{VERSION} is a probabilistic "
                "research screen and does not "
                "guarantee profit."
            ),
            (
                "Verify live price, liquidity, "
                "corporate news, market status "
                "and execution before trading."
            ),
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# LIVE AUDIT
# ============================================================

def save_live_audit(
    candidates,
    market
):

    if not candidates:
        return None

    timestamp = now_ist().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        AUDIT_DIR
        /
        f"live_v6_3_8_{timestamp}.csv"
    )

    df = pd.DataFrame(
        candidates
    )

    df["market_regime"] = (
        market["regime"]
    )

    df["signal_date"] = (
        now_ist().date().isoformat()
    )

    df.to_csv(
        filename,
        index=False
    )

    return filename


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        f"MARKET ALERT {VERSION}"
    )

    print(
        "=" * 70
    )

    print(
        f"Configured capital: "
        f"{format_money(CAPITAL)}"
    )

    print(
        f"Maximum risk/trade: "
        f"{MAX_RISK_PCT:.2%}"
    )

    signal_date = pd.Timestamp(
        now_ist().date()
    )

    trading_day = is_trading_day()

    tickers = [
        ticker(symbol)
        for symbol in SYMBOLS
    ]

    tickers.extend(
        [
            "^NSEI",
            "^INDIAVIX",
        ]
    )

    print(
        f"Downloading data for "
        f"{len(tickers)} instruments..."
    )

    raw = yf.download(
        tickers=tickers,
        period="3y",
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
        timeout=30,
    )

    frames = split_download(
        raw
    )

    if not frames:

        raise RuntimeError(
            "No market data received."
        )

    print(
        f"Downloaded frames: "
        f"{len(frames)}"
    )

    market = market_regime(
        frames,
        signal_date
    )

    print(
        f"Market regime: "
        f"{market['regime']}"
    )

    print(
        f"Reason: "
        f"{market['reason']}"
    )

    feature_frames = {}

    for symbol in SYMBOLS:

        df = frames.get(
            ticker(symbol)
        )

        if df is None or df.empty:
            continue

        try:

            feature_frames[
                symbol
            ] = add_features(df)

        except Exception as exc:

            print(
                f"Feature error "
                f"{symbol}: {exc}"
            )

    print(
        f"Stocks with features: "
        f"{len(feature_frames)}"
    )

    candidates = []

    for symbol, features in (
        feature_frames.items()
    ):

        hist = features.loc[
            features.index
            <=
            signal_date
        ].copy()

        if len(hist) < 300:
            continue

        valid = hist.dropna(
            subset=[
                "Close",
                "rsi",
                "atr_pct",
                "dist_sma50",
                "volume_ratio",
            ]
        )

        if valid.empty:
            continue

        latest = valid.iloc[-1]

        model = analogue_model(
            features,
            signal_date
        )

        if model is None:
            continue

        row = {

            "Close":
                latest["Close"],

            "rsi":
                latest["rsi"],

            "atr_pct":
                latest["atr_pct"],

            "dist_sma50":
                latest["dist_sma50"],

            "trend":
                latest["trend"],

            "volume_ratio":
                latest["volume_ratio"],

            **model,
        }

        result = evaluate_candidate(
            symbol,
            row,
            market["regime"]
        )

        if result is not None:

            candidates.append(
                result
            )

    candidates = sorted(
        candidates,
        key=lambda x: (
            1
            if x["action"] == "TRADE"
            else 0,
            x["score"],
        ),
        reverse=True
    )

    trade_count = sum(
        x["action"] == "TRADE"
        for x in candidates
    )

    watch_count = sum(
        x["action"] == "WATCH"
        for x in candidates
    )

    reject_count = sum(
        x["action"] == "REJECT"
        for x in candidates
    )

    print(
        f"Candidates: "
        f"{len(candidates)}"
    )

    print(
        f"TRADE={trade_count} "
        f"WATCH={watch_count} "
        f"REJECT={reject_count}"
    )

    audit_file = save_live_audit(
        candidates,
        market
    )

    if audit_file:

        print(
            f"Audit saved: "
            f"{audit_file}"
        )

    ipo_names = (
        get_ipo_information()
    )

    message = build_alert(
        market,
        candidates,
        ipo_names,
        trading_day
    )

    print("")
    print(message)
    print("")

    telegram_send(
        message
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
