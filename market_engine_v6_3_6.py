"""
============================================================
MULTI-FACTOR MARKET ALERT V6.3.6
============================================================

Short-term Indian equity research screener.

V6.3.6 improvements:
- Safe environment-variable handling
- No crash on empty GitHub variables
- Safe trend handling
- Conservative empirical analogue model
- 3-session and 5-session P(UP)
- Expected return estimation
- Risk/reward calculation
- Position sizing
- Market regime filter
- Weekend protection
- IPO retrieval
- Telegram alerts
- CSV audit trail
- No negative stop-loss values
- No artificial BUY signal when market regime is poor

IMPORTANT:
This is a probabilistic research model.
It does not guarantee profit.

P(UP) is an empirical/calibrated estimate,
not a guaranteed probability of profit.
============================================================
"""

from __future__ import annotations

import os
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# VERSION
# ============================================================

VERSION = "V6.3.6"


# ============================================================
# SAFE ENVIRONMENT VARIABLES
# ============================================================

def safe_env_float(name, default):
    """
    Safely read a numeric environment variable.

    GitHub Actions may provide an environment variable as
    an empty string. float("") causes ValueError.

    This function safely handles:
    - missing variable
    - empty variable
    - invalid text
    - NaN
    - infinity
    """

    raw = os.getenv(name)

    if raw is None:
        return float(default)

    raw = str(raw).strip()

    if raw == "":
        return float(default)

    try:

        value = float(raw)

        if not math.isfinite(value):
            return float(default)

        return value

    except (TypeError, ValueError):

        print(
            f"WARNING: Invalid {name}={raw!r}. "
            f"Using default {default}."
        )

        return float(default)


# ============================================================
# CAPITAL / RISK CONFIGURATION
# ============================================================

CAPITAL = safe_env_float(
    "TRADING_CAPITAL",
    100000
)

MAX_RISK_PCT = safe_env_float(
    "MAX_RISK_PCT",
    0.01
)


# ============================================================
# SAFETY VALIDATION
# ============================================================

if CAPITAL <= 0:

    print(
        "WARNING: TRADING_CAPITAL must be positive. "
        "Using ₹1,00,000."
    )

    CAPITAL = 100000


if MAX_RISK_PCT <= 0:

    print(
        "WARNING: MAX_RISK_PCT must be positive. "
        "Using 1%."
    )

    MAX_RISK_PCT = 0.01


if MAX_RISK_PCT > 0.05:

    print(
        "WARNING: MAX_RISK_PCT above 5%. "
        "Capping at 5%."
    )

    MAX_RISK_PCT = 0.05


# ============================================================
# MODEL CONFIGURATION
# ============================================================

N_ANALOGUES = 30

MIN_ANALOGUES = 15

LOOKBACK_DAYS = 756

# Probability requirements
MIN_CALIBRATED_P3 = 0.52
MIN_CALIBRATED_P5 = 0.51

# Expected return requirements
MIN_NET_ER3 = 0.0010
MIN_NET_ER5 = 0.0015

# Risk/reward requirements
MIN_RR1 = 1.00
MIN_RR2 = 1.20

# Technical safety filters
MAX_RSI = 75.0
MIN_VOLUME_RATIO = 0.80

MAX_TOP_PICKS = 5

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


# ============================================================
# BASIC HELPERS
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


def clean_date(value):

    ts = pd.Timestamp(value)

    if ts.tzinfo is not None:

        ts = ts.tz_localize(None)

    return ts.normalize()


def is_trading_day():

    return datetime.now().weekday() < 5


def format_money(value):

    value = safe_float(value)

    if not np.isfinite(value):

        return "N/A"

    return f"₹{value:,.2f}"


# ============================================================
# DATA CLEANING
# ============================================================

def clean_frame(df):

    if df is None or df.empty:

        return None

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        return None

    for column in required:

        if column not in df.columns:

            df[column] = np.nan

    df = df[
        required
    ].copy()

    df.index = pd.to_datetime(
        df.index
    )

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

    # --------------------------------------------------------
    # Structure:
    # Price -> ticker
    # --------------------------------------------------------

    if "Close" in level0:

        for name in level1:

            try:

                part = raw.xs(
                    name,
                    axis=1,
                    level=1,
                    drop_level=True
                )

                cleaned = clean_frame(
                    part
                )

                if cleaned is not None:

                    result[str(name)] = cleaned

            except Exception:

                continue

    # --------------------------------------------------------
    # Structure:
    # ticker -> Price
    # --------------------------------------------------------

    else:

        for name in level0:

            try:

                part = raw.xs(
                    name,
                    axis=1,
                    level=0,
                    drop_level=True
                )

                cleaned = clean_frame(
                    part
                )

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
        100 / (1 + rs)
    )

    rsi = rsi.where(
        avg_loss != 0,
        100.0
    )

    return rsi


# ============================================================
# FEATURE ENGINEERING
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

    x["ret1"] = (
        close.pct_change(1)
    )

    x["ret3"] = (
        close.pct_change(3)
    )

    x["ret5"] = (
        close.pct_change(5)
    )

    x["ret10"] = (
        close.pct_change(10)
    )

    x["ret20"] = (
        close.pct_change(20)
    )

    previous_close = (
        close.shift(1)
    )

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
        - 1
    )

    x["dist_sma50"] = (
        close
        /
        x["sma50"]
        - 1
    )

    x["dist_sma200"] = (
        close
        /
        x["sma200"]
        - 1
    )

    x["sma50_slope5"] = (
        x["sma50"]
        /
        x["sma50"].shift(5)
        - 1
    )

    x["ema10_slope5"] = (
        x["ema10"]
        /
        x["ema10"].shift(5)
        - 1
    )

    return x


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
# MARKET REGIME
# ============================================================

def market_regime(
    frames,
    signal_date
):

    nifty = frames.get(
        "^NSEI"
    )

    vix = frames.get(
        "^INDIAVIX"
    )

    if nifty is None or nifty.empty:

        return {

            "regime":
                "UNKNOWN",

            "nifty":
                np.nan,

            "sma50":
                np.nan,

            "vix":
                np.nan,

        }

    n = nifty.loc[
        nifty.index <= signal_date
    ]

    if len(n) < 60:

        return {

            "regime":
                "UNKNOWN",

            "nifty":
                np.nan,

            "sma50":
                np.nan,

            "vix":
                np.nan,

        }

    close = n["Close"]

    sma50 = (
        close
        .rolling(50)
        .mean()
    )

    current = safe_float(
        close.iloc[-1]
    )

    current_sma50 = safe_float(
        sma50.iloc[-1]
    )

    previous_sma50 = safe_float(
        sma50.iloc[-6]
    )

    above = (

        np.isfinite(current)

        and

        np.isfinite(
            current_sma50
        )

        and

        current >= current_sma50

    )

    rising = (

        np.isfinite(
            current_sma50
        )

        and

        np.isfinite(
            previous_sma50
        )

        and

        current_sma50
        >
        previous_sma50

    )

    if above and rising:

        regime = "FAVORABLE"

    elif (
        not above
        and
        not rising
    ):

        regime = "UNFAVORABLE"

    else:

        regime = "MIXED"

    vix_value = np.nan

    if vix is not None:

        vv = vix.loc[
            vix.index <= signal_date
        ]

        if not vv.empty:

            vix_value = safe_float(
                vv["Close"].iloc[-1]
            )

    return {

        "regime":
            regime,

        "nifty":
            current,

        "sma50":
            current_sma50,

        "vix":
            vix_value,

    }


# ============================================================
# EMPIRICAL ANALOGUE MODEL
# ============================================================

def analogue_model(
    features,
    signal_date
):

    data = features.loc[
        features.index <= signal_date
    ].copy()

    required = (
        FEATURES
        +
        ["Close"]
    )

    data = data.dropna(
        subset=required
    )

    if len(data) < 300:

        return None

    candidate = data.iloc[-1]

    # --------------------------------------------------------
    # Avoid using the immediately preceding overlapping
    # observations as training analogues.
    # --------------------------------------------------------

    train = data.iloc[
        :-6
    ].copy()

    if len(train) < 100:

        return None

    train = train.tail(
        LOOKBACK_DAYS
    ).copy()

    # --------------------------------------------------------
    # Future returns
    # --------------------------------------------------------

    train["future3"] = (

        train["Close"].shift(-3)

        /

        train["Close"]

        - 1

    )

    train["future5"] = (

        train["Close"].shift(-5)

        /

        train["Close"]

        - 1

    )

    train = train.dropna(
        subset=[
            "future3",
            "future5",
        ]
    )

    if len(train) < 80:

        return None

    # --------------------------------------------------------
    # Distance calculation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Nearest historical analogues
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Distance weights
    # --------------------------------------------------------

    weights = (

        1
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

    # --------------------------------------------------------
    # Raw empirical probabilities
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Expected returns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Dispersion
    # --------------------------------------------------------

    std3 = safe_float(

        train["future3"].std(),

        0.05

    )

    std5 = safe_float(

        train["future5"].std(),

        0.07

    )

    # --------------------------------------------------------
    # Conservative probability shrinkage
    # --------------------------------------------------------

    shrinkage = min(

        0.45,

        12.0
        /
        max(
            12.0,
            len(train)
        )

    )

    calibrated_p3 = (

        (
            1
            -
            shrinkage
        )
        *
        raw_p3

        +

        shrinkage
        *
        0.50

    )

    calibrated_p5 = (

        (
            1
            -
            shrinkage
        )
        *
        raw_p5

        +

        shrinkage
        *
        0.50

    )

    # --------------------------------------------------------
    # Model quality
    # --------------------------------------------------------

    dispersion_penalty3 = min(

        1.0,

        std3 / 0.035

    )

    dispersion_penalty5 = min(

        1.0,

        std5 / 0.055

    )

    sample_quality = min(

        1.0,

        len(train)
        /
        N_ANALOGUES

    )

    quality = (

        0.40
        *
        sample_quality

        +

        0.30
        *
        (
            1
            -
            dispersion_penalty3
        )

        +

        0.30
        *
        (
            1
            -
            dispersion_penalty5
        )

    )

    quality = max(
        0.0,
        min(
            1.0,
            quality
        )
    )

    return {

        "raw_p3":
            raw_p3,

        "raw_p5":
            raw_p5,

        "p3":
            calibrated_p3,

        "p5":
            calibrated_p5,

        "er3":
            er3,

        "er5":
            er5,

        "std3":
            std3,

        "std5":
            std5,

        "quality":
            quality,

        "analogues":
            len(train),

        "nearest_distance":
            float(
                train[
                    "distance"
                ].iloc[0]
            ),

    }


# ============================================================
# STOP / TARGET / RISK REWARD
# ============================================================

def calculate_trade_levels(
    price,
    atr_pct,
    er3,
    er5
):

    price = safe_float(
        price
    )

    atr_pct = safe_float(
        atr_pct
    )

    er3 = safe_float(
        er3
    )

    er5 = safe_float(
        er5
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

    # --------------------------------------------------------
    # Stop distance
    # --------------------------------------------------------

    stop_distance = min(

        0.05,

        max(

            0.012,

            1.5
            *
            atr_pct

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

    # NEVER allow zero or negative stop.
    stop = max(
        0.01,
        stop
    )

    risk_per_share = (

        price
        -
        stop

    )

    if risk_per_share <= 0:

        return None

    # --------------------------------------------------------
    # Expected-return-based targets
    #
    # These are projections, not guarantees.
    # --------------------------------------------------------

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

        "stop":
            stop,

        "target1":
            target1,

        "target2":
            target2,

        "risk_per_share":
            risk_per_share,

        "rr1":
            rr1,

        "rr2":
            rr2,

    }


# ============================================================
# POSITION SIZING
# ============================================================

def position_size(
    price,
    stop
):

    price = safe_float(
        price
    )

    stop = safe_float(
        stop
    )

    if (

        not np.isfinite(price)

        or

        not np.isfinite(stop)

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

    """
    Calculate the V6.3.6 ranking score.

    IMPORTANT:
    Older versions sometimes expected row["trend"],
    while the engine actually stores the value as
    dist_sma50.

    This function accepts either name so that a missing
    'trend' field cannot crash the engine.
    """

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    p3 = safe_float(

        row.get(
            "p3"
        ),

        0.50

    )

    p5 = safe_float(

        row.get(
            "p5"
        ),

        0.50

    )

    # --------------------------------------------------------
    # Expected return
    # --------------------------------------------------------

    er3 = safe_float(

        row.get(
            "er3"
        ),

        0.0

    )

    er5 = safe_float(

        row.get(
            "er5"
        ),

        0.0

    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_value = safe_float(

        row.get(
            "rsi"
        ),

        50.0

    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume = safe_float(

        row.get(
            "volume_ratio"
        ),

        1.0

    )

    # --------------------------------------------------------
    # TREND FIX
    #
    # Prefer 'trend'.
    # Fall back to 'dist_sma50'.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Base score
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    if trend > 0:

        score += 0.20

    elif trend < -0.03:

        score -= 0.20

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if volume >= 1.20:

        score += 0.15

    elif volume >= 1.00:

        score += 0.05

    elif volume < 0.80:

        score -= 0.10

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if (

        50.0
        <=
        rsi_value
        <=
        68.0

    ):

        score += 0.10

    elif rsi_value > 72.0:

        score -= 0.15

    elif rsi_value < 40.0:

        score -= 0.05

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------

    if regime == "FAVORABLE":

        score += 0.15

    elif regime == "MIXED":

        score -= 0.05

    elif regime == "UNFAVORABLE":

        score -= 0.50

    # --------------------------------------------------------
    # Numerical safety
    # --------------------------------------------------------

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
        row.get(
            "Close"
        )
    )

    rsi_value = safe_float(
        row.get(
            "rsi"
        )
    )

    volume = safe_float(
        row.get(
            "volume_ratio"
        )
    )

    # --------------------------------------------------------
    # Trend fix
    # --------------------------------------------------------

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
        row.get(
            "p3"
        )
    )

    p5 = safe_float(
        row.get(
            "p5"
        )
    )

    er3 = safe_float(
        row.get(
            "er3"
        )
    )

    er5 = safe_float(
        row.get(
            "er5"
        )
    )

    quality = safe_float(
        row.get(
            "quality"
        )
    )

    levels = calculate_trade_levels(

        price,

        row.get(
            "atr_pct"
        ),

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

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    checks = {

        "p3":
            (
                np.isfinite(p3)
                and
                p3
                >=
                MIN_CALIBRATED_P3
            ),

        "p5":
            (
                np.isfinite(p5)
                and
                p5
                >=
                MIN_CALIBRATED_P5
            ),

        "er3":
            (
                np.isfinite(er3)
                and
                er3
                >=
                MIN_NET_ER3
            ),

        "er5":
            (
                np.isfinite(er5)
                and
                er5
                >=
                MIN_NET_ER5
            ),

        "rr1":
            (
                levels["rr1"]
                >=
                MIN_RR1
            ),

        "rr2":
            (
                levels["rr2"]
                >=
                MIN_RR2
            ),

        "quality":
            (
                np.isfinite(quality)
                and
                quality
                >=
                0.25
            ),

        "trend":
            (
                np.isfinite(trend)
                and
                trend
                >
                -0.03
            ),

        "rsi":
            (
                np.isfinite(rsi_value)
                and
                rsi_value
                <=
                MAX_RSI
            ),

        "volume":
            (
                np.isfinite(volume)
                and
                volume
                >=
                MIN_VOLUME_RATIO
            ),

        "regime":
            (
                regime
                !=
                "UNFAVORABLE"
            ),

    }

    failed = [

        key

        for key, value
        in checks.items()

        if not bool(value)

    ]

    all_pass = all(

        bool(value)

        for value
        in checks.values()

    )

    pass_rate = (

        sum(

            bool(value)

            for value
            in checks.values()

        )

        /

        len(checks)

    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if all_pass:

        action = "TRADE"

    elif pass_rate >= 0.60:

        action = "WATCH"

    else:

        action = "REJECT"

    # --------------------------------------------------------
    # Extra safety:
    #
    # No TRADE if the market is unfavourable.
    # --------------------------------------------------------

    if regime == "UNFAVORABLE":

        action = "REJECT"

    return {

        "symbol":
            symbol,

        "price":
            price,

        "raw_p3":
            row.get(
                "raw_p3",
                np.nan
            ),

        "raw_p5":
            row.get(
                "raw_p5",
                np.nan
            ),

        "p3":
            p3,

        "p5":
            p5,

        "er3":
            er3,

        "er5":
            er5,

        "rsi":
            rsi_value,

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
            ",".join(
                failed
            ),

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

            print(

                "IPO HTTP status:",

                response.status_code

            )

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

                item.get(
                    "companyName"
                )

                or

                item.get(
                    "symbol"
                )

                or

                item.get(
                    "name"
                )

            )

            if name:

                results.append(
                    str(name)
                )

        # Remove duplicates
        results = list(
            dict.fromkeys(
                results
            )
        )

        return results[:10]

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

    bot_token = str(
        bot_token
    ).strip()

    chat_id = str(
        chat_id
    ).strip()

    if (

        not bot_token
        or
        not chat_id

    ):

        print(
            "Telegram credentials are empty."
        )

        return False

    url = (

        "https://api.telegram.org/"
        f"bot{bot_token}/sendMessage"

    )

    payload = {

        "chat_id":
            chat_id,

        "text":
            message,

    }

    try:

        response = requests.post(

            url,

            json=payload,

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

    now = datetime.now().strftime(

        "%d %b %Y, %H:%M IST"

    )

    lines = [

        f"MULTI-FACTOR MARKET ALERT {VERSION}",

        now,

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

        "",

        "--- TOP SHORT-TERM TRADE SETUPS ---",

    ]

    trades = [

        item

        for item
        in candidates

        if item["action"]
        ==
        "TRADE"

    ]

    watches = [

        item

        for item
        in candidates

        if item["action"]
        ==
        "WATCH"

    ]

    # ========================================================
    # WEEKEND
    # ========================================================

    if not trading_day:

        lines += [

            "",

            "MARKET IS CLOSED.",

            (
                "No new long position should "
                "be initiated today."
            ),

        ]

    # ========================================================
    # TRADE SETUPS
    # ========================================================

    if trades:

        for index, item in enumerate(

            trades[
                :MAX_TOP_PICKS
            ],

            start=1

        ):

            lines += [

                "",

                (
                    f"{index}. "
                    f"{item['symbol']} — TRADE"
                ),

                (
                    f"Price: "
                    f"{format_money(item['price'])}"
                ),

                (
                    "Calibrated P(UP) "
                    "3D / 5D: "
                    f"{item['p3']:.1%} / "
                    f"{item['p5']:.1%}"
                ),

                (
                    "Raw analogue P(UP) "
                    "3D / 5D: "
                    f"{item['raw_p3']:.1%} / "
                    f"{item['raw_p5']:.1%}"
                ),

                (
                    "Expected return "
                    "3D / 5D: "
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
                    f"Entry: "
                    f"{format_money(item['price'])}"
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

    else:

        lines += [

            "",

            "NO VALID LONG TRADE TODAY",

            (
                "No candidate currently satisfies "
                "the V6.3.6 probability, expected-return, "
                "risk/reward and market filters."
            ),

        ]

    # ========================================================
    # WATCHLIST
    # ========================================================

    lines += [

        "",

        "--- BEST WATCHLIST SETUPS ---",

    ]

    if watches:

        watches = sorted(

            watches,

            key=lambda x:
                x["score"],

            reverse=True

        )

        for index, item in enumerate(

            watches[:3],

            start=1

        ):

            lines += [

                "",

                (
                    f"{index}. "
                    f"{item['symbol']} — WATCH"
                ),

                (
                    f"Price: "
                    f"{format_money(item['price'])}"
                ),

                (
                    "Calibrated P(UP) "
                    "3D / 5D: "
                    f"{item['p3']:.1%} / "
                    f"{item['p5']:.1%}"
                ),

                (
                    "Expected return "
                    "3D / 5D: "
                    f"{item['er3']:.2%} / "
                    f"{item['er5']:.2%}"
                ),

                (
                    f"RR1: "
                    f"{item['rr1']:.2f}"
                    f" | RR2: "
                    f"{item['rr2']:.2f}"
                ),

                (
                    f"RSI: "
                    f"{item['rsi']:.1f}"
                    f" | Volume: "
                    f"{item['volume_ratio']:.2f}x"
                ),

                (
                    f"Score: "
                    f"{item['score']:.3f}"
                ),

                "Action: WATCH / WAIT",

            ]

    else:

        lines.append(
            "None."
        )

    # ========================================================
    # IPO
    # ========================================================

    lines += [

        "",

        "--- IPO OPEN / UPCOMING ---",

    ]

    if ipo_names:

        lines.append(

            "IPO records retrieved. "
            "Verify issue dates, price band "
            "and subscription status before applying."

        )

        for name in ipo_names:

            lines.append(

                f"• {name}"

            )

    else:

        lines += [

            (
                "IPO DATA UNAVAILABLE | "
                "RETRIEVAL FAILED"
            ),

            (
                "Verify current/upcoming issues "
                "directly on NSE."
            ),

        ]

    # ========================================================
    # DISCLAIMER
    # ========================================================

    lines += [

        "",

        (
            f"{VERSION} is a probabilistic "
            "research screen and does not "
            "guarantee profit."
        ),

        (
            "P(UP) is an empirical calibrated "
            "estimate, not a guaranteed "
            "probability of profit."
        ),

        (
            "Verify live price, liquidity, "
            "corporate news, market status "
            "and order execution before trading."
        ),

    ]

    return "\n".join(
        lines
    )


# ============================================================
# MAIN ENGINE
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

    signal_date = clean_date(
        datetime.now()
    )

    trading_day = is_trading_day()

    tickers = [

        ticker(symbol)

        for symbol
        in SYMBOLS

    ]

    tickers += [

        "^NSEI",
        "^INDIAVIX",

    ]

    print(

        f"Downloading data for "
        f"{len(tickers)} instruments..."

    )

    try:

        raw = yf.download(

            tickers=tickers,

            period="3y",

            interval="1d",

            auto_adjust=True,

            group_by="column",

            progress=False,

            threads=True,

        )

    except Exception as exc:

        raise RuntimeError(

            f"Market data download failed: {exc}"

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

    # ========================================================
    # MARKET REGIME
    # ========================================================

    market = market_regime(

        frames,

        signal_date

    )

    print(

        f"Market regime: "
        f"{market['regime']}"

    )

    # ========================================================
    # FEATURE CALCULATION
    # ========================================================

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
            ] = add_features(
                df
            )

        except Exception as exc:

            print(

                f"Feature error "
                f"{symbol}: {exc}"

            )

    print(

        f"Stocks with features: "
        f"{len(feature_frames)}"

    )

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

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
                latest["dist_sma50"],

            "volume_ratio":
                latest["volume_ratio"],

            **model,

        }

        candidate = evaluate_candidate(

            symbol,

            row,

            market["regime"]

        )

        if candidate is not None:

            candidates.append(
                candidate
            )

    # ========================================================
    # SORT
    # ========================================================

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

    print(

        f"Candidates evaluated: "
        f"{len(candidates)}"

    )

    # ========================================================
    # AUDIT
    # ========================================================

    audit = pd.DataFrame(
        candidates
    )

    timestamp = datetime.now().strftime(

        "%Y%m%d_%H%M%S"

    )

    audit_file = (

        AUDIT_DIR
        /
        f"live_v6_3_6_{timestamp}.csv"

    )

    if not audit.empty:

        audit.to_csv(

            audit_file,

            index=False

        )

        print(

            f"Audit saved: "
            f"{audit_file}"

        )

    else:

        print(
            "No candidate audit rows generated."
        )

    # ========================================================
    # IPO
    # ========================================================

    ipo_names = (
        get_ipo_information()
    )

    # ========================================================
    # ALERT
    # ========================================================

    message = build_alert(

        market,

        candidates,

        ipo_names,

        trading_day

    )

    print("")
    print(message)
    print("")

    # ========================================================
    # TELEGRAM
    # ========================================================

    telegram_send(
        message
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
