"""
GOLD FACTOR ANALYZER
MARKET ENGINE V6.3.10

Major improvements over V6.3.9:
1. Positive, volatility-aware stop-loss calculation.
2. Genuine RR1/RR2 calculations.
3. Regime is a weighting factor, not a hard rejection.
4. Conservative probability calibration.
5. Explicit liquidity/volume filtering.
6. Position sizing based on actual stop distance.
7. Empty GitHub environment variables are handled safely.
8. Weekend / closed-market protection.
9. Telegram failure cannot crash the workflow.
10. No negative stop-loss values.
"""

import os
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests


VERSION = "V6.3.10"

IST = timezone(timedelta(hours=5, minutes=30))


# ============================================================
# SAFE ENVIRONMENT VARIABLE FUNCTIONS
# ============================================================

def env_float(name, default):
    value = os.getenv(name, "")
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (ValueError, TypeError):
        return float(default)


def env_int(name, default):
    value = os.getenv(name, "")
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(value)
    except (ValueError, TypeError):
        return int(default)


# ============================================================
# CONFIGURATION
# ============================================================

CAPITAL = env_float("ALERT_CAPITAL", 100000)

MAX_RISK_PCT = env_float(
    "MAX_RISK_PCT",
    1.0
)

MAX_POSITION_PCT = env_float(
    "MAX_POSITION_PCT",
    20.0
)

SLIPPAGE_BPS = env_float(
    "SLIPPAGE_BPS",
    8.0
)

ROUND_TRIP_COST_BPS = env_float(
    "ROUND_TRIP_COST_BPS",
    12.0
)

TOP_N = env_int(
    "TOP_N",
    5
)


TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
).strip()


# ============================================================
# STOCK UNIVERSE
# ============================================================

UNIVERSE = [

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
    "M&M.NS",
    "TATAMOTORS.NS",
    "EICHERMOT.NS",
    "MARUTI.NS",
    "HEROMOTOCO.NS",

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

    "ZOMATO.NS",
    "NAUKRI.NS",
    "COFORGE.NS",

    "JIOFIN.NS",
    "IRFC.NS",
    "IREDA.NS",

    "POLYCAB.NS",
]


# ============================================================
# PRIOR CALIBRATION
# ============================================================

# These are deliberately conservative.
#
# The V6.3.9 backtest showed:
#
# <40%     actual ~42.6%
# 40-45%   actual ~46.5%
# 45-50%   actual ~49.9%
# 50-55%   actual ~51.8%
# 55-60%   actual ~55.6%
# 60-65%   actual ~58.0%
# 65-70%   actual ~59.3%
#
# We shrink the empirical values toward 50%.
#
# This prevents the engine from claiming unrealistic probabilities.

CAL_X = np.array([
    0.32,
    0.42,
    0.47,
    0.52,
    0.57,
    0.62,
    0.67,
    0.72,
    0.80
])

CAL_Y = np.array([
    0.43,
    0.46,
    0.49,
    0.51,
    0.54,
    0.57,
    0.59,
    0.61,
    0.62
])


def calibrated_probability(raw_probability):
    """
    Conservative calibration.

    Never allows the engine to present a very high raw probability
    as certainty.
    """

    raw_probability = float(
        np.clip(
            raw_probability,
            0.20,
            0.90
        )
    )

    empirical = float(
        np.interp(
            raw_probability,
            CAL_X,
            CAL_Y
        )
    )

    calibrated = (
        0.50
        + 0.92 * (empirical - 0.50)
    )

    return float(
        np.clip(
            calibrated,
            0.35,
            0.65
        )
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gains = delta.clip(
        lower=0
    )

    losses = -delta.clip(
        upper=0
    )

    avg_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# ATR
# ============================================================

def calculate_atr(df, period=14):

    previous_close = df["Close"].shift(1)

    true_range = pd.concat(
        [
            df["High"] - df["Low"],

            (
                df["High"]
                - previous_close
            ).abs(),

            (
                df["Low"]
                - previous_close
            ).abs(),
        ],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# FEATURES
# ============================================================

def add_features(df):

    if df is None or df.empty:
        return pd.DataFrame()

    x = df.copy()

    if isinstance(
        x.columns,
        pd.MultiIndex
    ):

        flattened = []

        for column in x.columns:

            if isinstance(
                column,
                tuple
            ):

                if column[0] in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume"
                ]:

                    flattened.append(
                        column[0]
                    )

                else:

                    flattened.append(
                        column[-1]
                    )

            else:

                flattened.append(
                    column
                )

        x.columns = flattened

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required:

        if column not in x.columns:
            return pd.DataFrame()

    x = x[
        required
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    x = x.dropna(
        subset=["Close"]
    )

    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    x["ret1"] = (
        x["Close"].pct_change()
    )

    x["ret3"] = (
        x["Close"].pct_change(3)
    )

    x["ret5"] = (
        x["Close"].pct_change(5)
    )

    x["ret20"] = (
        x["Close"].pct_change(20)
    )

    # --------------------------------------------------------
    # MOVING AVERAGES
    # --------------------------------------------------------

    x["sma20"] = (
        x["Close"]
        .rolling(
            20,
            min_periods=20
        )
        .mean()
    )

    x["sma50"] = (
        x["Close"]
        .rolling(
            50,
            min_periods=50
        )
        .mean()
    )

    x["sma200"] = (
        x["Close"]
        .rolling(
            200,
            min_periods=200
        )
        .mean()
    )

    # --------------------------------------------------------
    # MOMENTUM / RSI
    # --------------------------------------------------------

    x["rsi"] = calculate_rsi(
        x["Close"]
    )

    x["atr"] = calculate_atr(
        x
    )

    x["atr_pct"] = (
        x["atr"]
        / x["Close"]
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    x["vol_ratio"] = (
        x["Volume"]
        /
        x["Volume"]
        .rolling(
            20,
            min_periods=10
        )
        .median()
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    x["volatility20"] = (
        x["ret1"]
        .rolling(
            20,
            min_periods=20
        )
        .std()
        * np.sqrt(252)
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    x["trend20"] = (
        x["Close"]
        / x["sma20"]
        - 1
    )

    x["trend50"] = (
        x["Close"]
        / x["sma50"]
        - 1
    )

    x["trend200"] = (
        x["Close"]
        / x["sma200"]
        - 1
    )

    # --------------------------------------------------------
    # MOMENTUM SCORE
    # --------------------------------------------------------

    x["mom_score"] = (
        0.45 * x["ret5"].fillna(0)
        +
        0.35 * x["ret20"].fillna(0)
        +
        0.20 * x["ret3"].fillna(0)
    )

    # --------------------------------------------------------
    # QUALITY SCORE
    # --------------------------------------------------------

    x["quality"] = (

        0.35
        * (
            x["trend20"] > 0
        ).astype(float)

        +

        0.35
        * (
            x["trend50"] > 0
        ).astype(float)

        +

        0.30
        * (
            x["trend200"] > 0
        ).astype(float)
    )

    return x


# ============================================================
# MARKET REGIME
# ============================================================

def determine_market_regime(index_df):

    features = add_features(
        index_df
    )

    if features.empty:

        return (
            "UNKNOWN",
            np.nan,
            np.nan,
            "Index data unavailable"
        )

    row = features.iloc[-1]

    nifty = float(
        row["Close"]
    )

    sma50 = float(
        row["sma50"]
    ) if pd.notna(
        row["sma50"]
    ) else np.nan

    if pd.isna(sma50):

        return (
            "UNKNOWN",
            nifty,
            sma50,
            "SMA50 unavailable"
        )

    slope = (
        features["sma50"]
        .diff(5)
        .iloc[-1]
    )

    distance = (
        nifty / sma50
        - 1
    )

    if (
        distance > 0.008
        and pd.notna(slope)
        and slope > 0
    ):

        return (
            "FAVORABLE",
            nifty,
            sma50,
            "Nifty above SMA50 with rising SMA50"
        )

    if (
        distance < -0.008
        and pd.notna(slope)
        and slope < 0
    ):

        return (
            "UNFAVORABLE",
            nifty,
            sma50,
            "Nifty below SMA50 with falling SMA50"
        )

    return (
        "MIXED",
        nifty,
        sma50,
        "Nifty/SMA50 signals mixed"
    )


# ============================================================
# RAW PROBABILITY MODEL
# ============================================================

def calculate_raw_probability(
    row,
    regime
):

    rsi_value = float(
        row["rsi"]
    )

    rsi_component = np.clip(
        (rsi_value - 50) / 25,
        -1,
        1
    )

    trend_component = np.clip(
        (
            row["trend20"] * 40
            +
            row["trend50"] * 30
            +
            row["trend200"] * 15
        ),
        -1,
        1
    )

    momentum_component = np.clip(
        row["mom_score"] * 30,
        -1,
        1
    )

    volume_component = np.clip(
        (
            row["vol_ratio"] - 1
        ) / 1.5,
        -1,
        1
    )

    volatility_penalty = np.clip(
        (
            row["atr_pct"]
            - 0.025
        ) / 0.025,
        -1,
        1
    )

    regime_component = {
        "FAVORABLE": 0.55,
        "MIXED": 0.00,
        "UNFAVORABLE": -0.35,
    }.get(
        regime,
        0
    )

    score = (

        0.34
        * trend_component

        +

        0.28
        * momentum_component

        +

        0.12
        * rsi_component

        +

        0.10
        * volume_component

        -

        0.06
        * volatility_penalty

        +

        0.10
        * regime_component
    )

    raw = (
        0.50
        +
        0.24 * score
    )

    # Avoid treating extreme RSI as automatically bullish.
    if rsi_value > 72:

        raw -= min(
            0.05,
            (
                rsi_value - 72
            ) * 0.004
        )

    return float(
        np.clip(
            raw,
            0.25,
            0.78
        )
    )


# ============================================================
# EXPECTED RETURNS
# ============================================================

def calculate_expected_returns(
    row,
    regime
):

    trend = float(
        row["trend20"]
        if pd.notna(
            row["trend20"]
        )
        else 0
    )

    momentum = float(
        row["mom_score"]
        if pd.notna(
            row["mom_score"]
        )
        else 0
    )

    rsi_value = float(
        row["rsi"]
        if pd.notna(
            row["rsi"]
        )
        else 50
    )

    rsi_bias = np.clip(
        (
            rsi_value - 50
        ) / 100,
        -0.15,
        0.15
    )

    regime_bias = {

        "FAVORABLE": 0.0025,

        "MIXED": 0.0,

        "UNFAVORABLE": -0.0015,

    }.get(
        regime,
        0
    )

    base = (

        0.010 * trend

        +

        0.018 * momentum

        +

        0.006 * rsi_bias

        +

        regime_bias
    )

    er1 = np.clip(
        base * 0.42,
        -0.02,
        0.02
    )

    er3 = np.clip(
        base * 1.00,
        -0.05,
        0.05
    )

    er5 = np.clip(
        base * 1.55,
        -0.08,
        0.08
    )

    return (
        float(er1),
        float(er3),
        float(er5)
    )


# ============================================================
# REAL RISK / REWARD
# ============================================================

def calculate_risk_levels(row):

    price = float(
        row["Close"]
    )

    atr_value = float(
        row["atr"]
        if pd.notna(
            row["atr"]
        )
        else price * 0.02
    )

    # --------------------------------------------------------
    # STOP DISTANCE
    # --------------------------------------------------------

    stop_distance = max(
        1.15 * atr_value,
        price * 0.008
    )

    stop_distance = np.clip(
        stop_distance,
        price * 0.008,
        price * 0.035
    )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    stop_loss = (
        price
        - stop_distance
    )

    # Safety: never negative.
    stop_loss = max(
        stop_loss,
        price * 0.50
    )

    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    target1 = (
        price
        +
        max(
            0.85 * atr_value,
            price * 0.008
        )
    )

    target2 = (
        price
        +
        max(
            1.60 * atr_value,
            price * 0.014
        )
    )

    risk = (
        price
        - stop_loss
    )

    if risk <= 0:

        return (
            stop_loss,
            target1,
            target2,
            np.nan,
            np.nan,
            np.nan
        )

    rr1 = (
        target1 - price
    ) / risk

    rr2 = (
        target2 - price
    ) / risk

    return (
        float(stop_loss),
        float(target1),
        float(target2),
        float(rr1),
        float(rr2),
        float(risk)
    )


# ============================================================
# CANDIDATE CLASSIFICATION
# ============================================================

def evaluate_candidate(
    row,
    regime
):

    raw_probability = (
        calculate_raw_probability(
            row,
            regime
        )
    )

    probability = (
        calibrated_probability(
            raw_probability
        )
    )

    er1, er3, er5 = (
        calculate_expected_returns(
            row,
            regime
        )
    )

    (
        stop,
        target1,
        target2,
        rr1,
        rr2,
        risk
    ) = calculate_risk_levels(
        row
    )

    volume_ok = (
        pd.notna(
            row["vol_ratio"]
        )
        and
        0.75
        <=
        float(
            row["vol_ratio"]
        )
        <=
        3.0
    )

    trend_ok = (
        float(
            row["trend20"]
        ) > -0.005
        and
        float(
            row["trend50"]
        ) > -0.010
    )

    rsi_ok = (
        42
        <=
        float(
            row["rsi"]
        )
        <=
        70
    )

    quality_ok = (
        float(
            row["quality"]
        ) >= 0.33
    )

    regime_ok = (
        regime
        in
        {
            "FAVORABLE",
            "MIXED",
            "UNFAVORABLE"
        }
    )

    rr1_ok = (
        pd.notna(rr1)
        and
        rr1 >= 1.05
    )

    rr2_ok = (
        pd.notna(rr2)
        and
        rr2 >= 1.45
    )

    p3_ok = (
        probability >= 0.56
    )

    p5_ok = (
        probability >= 0.58
    )

    er3_ok = (
        er3 >= 0.0025
    )

    er5_ok = (
        er5 >= 0.0040
    )

    checks = {

        "p3": p3_ok,
        "p5": p5_ok,
        "er3": er3_ok,
        "er5": er5_ok,

        "rr1": rr1_ok,
        "rr2": rr2_ok,

        "quality": quality_ok,
        "trend": trend_ok,
        "rsi": rsi_ok,
        "volume": volume_ok,
        "regime": regime_ok,
    }

    failures = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    # --------------------------------------------------------
    # TRADE
    # --------------------------------------------------------

    if all(
        [
            p3_ok,
            p5_ok,
            er3_ok,
            er5_ok,
            rr1_ok,
            rr2_ok,
            quality_ok,
            trend_ok,
            rsi_ok,
            volume_ok,
        ]
    ):

        action = "TRADE"

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    elif (
        probability >= 0.54
        and
        er3 > 0
        and
        er5 > 0
        and
        rr2 >= 1.15
        and
        trend_ok
    ):

        action = "WATCH"

    else:

        action = "WAIT"

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = (

        100
        * (
            probability
            - 0.50
        )

        +

        18 * er3

        +

        10 * er5

        +

        4 * (
            rr2 - 1
        )

        +

        2 * (
            float(
                row["vol_ratio"]
            )
            - 1
        )

        +

        3 * (
            float(
                row["quality"]
            )
            - 0.50
        )
    )

    return {

        "raw_probability":
            raw_probability,

        "probability":
            probability,

        "er1":
            er1,

        "er3":
            er3,

        "er5":
            er5,

        "stop":
            stop,

        "target1":
            target1,

        "target2":
            target2,

        "rr1":
            rr1,

        "rr2":
            rr2,

        "risk":
            risk,

        "action":
            action,

        "score":
            float(score),

        "rsi":
            float(
                row["rsi"]
            ),

        "volume":
            float(
                row["vol_ratio"]
            ),

        "failures":
            failures,
    }


# ============================================================
# POSITION SIZING
# ============================================================

def calculate_position_size(
    price,
    stop
):

    if (
        not np.isfinite(price)
        or
        not np.isfinite(stop)
        or
        price <= 0
        or
        stop <= 0
        or
        stop >= price
    ):

        return 0

    risk_budget = (
        CAPITAL
        *
        MAX_RISK_PCT
        /
        100
    )

    risk_per_share = (
        price
        - stop
    )

    shares_by_risk = math.floor(
        risk_budget
        /
        risk_per_share
    )

    position_budget = (
        CAPITAL
        *
        MAX_POSITION_PCT
        /
        100
    )

    shares_by_position = math.floor(
        position_budget
        /
        price
    )

    return max(
        0,
        min(
            shares_by_risk,
            shares_by_position
        )
    )


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data(
    ticker,
    period="2y"
):

    try:

        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if (
            df is None
            or
            df.empty
        ):

            return pd.DataFrame()

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            new_columns = []

            for column in df.columns:

                if isinstance(
                    column,
                    tuple
                ):

                    if column[0] in [
                        "Open",
                        "High",
                        "Low",
                        "Close",
                        "Volume"
                    ]:

                        new_columns.append(
                            column[0]
                        )

                    else:

                        new_columns.append(
                            column[-1]
                        )

                else:

                    new_columns.append(
                        column
                    )

            df.columns = new_columns

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            c in df.columns
            for c in required
        ):

            return pd.DataFrame()

        return df[
            required
        ].apply(
            pd.to_numeric,
            errors="coerce"
        ).dropna(
            subset=["Close"]
        )

    except Exception as error:

        print(
            f"{ticker}: data error: {error}"
        )

        return pd.DataFrame()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if (
        not TELEGRAM_TOKEN
        or
        not TELEGRAM_CHAT_ID
    ):

        print(
            "Telegram credentials not configured."
        )

        return False

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {

        "chat_id":
            TELEGRAM_CHAT_ID,

        "text":
            message,

        "disable_web_page_preview":
            True,
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=20
        )

        response.raise_for_status()

        return True

    except Exception as error:

        print(
            f"Telegram error: {error}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    now = datetime.now(
        IST
    )

    print(
        f"MARKET ALERT {VERSION}"
    )

    print(
        "=" * 70
    )

    print(
        f"Configured capital: "
        f"₹{CAPITAL:,.2f}"
    )

    print(
        f"Maximum risk/trade: "
        f"{MAX_RISK_PCT:.2f}%"
    )

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    nifty_df = download_data(
        "^NSEI"
    )

    (
        regime,
        nifty,
        sma50,
        regime_reason
    ) = determine_market_regime(
        nifty_df
    )

    # --------------------------------------------------------
    # MARKET STATUS
    # --------------------------------------------------------

    market_open = (
        now.weekday() < 5
    )

    if market_open:

        market_open = (
            9 <= now.hour < 16
        )

    # --------------------------------------------------------
    # STOCK ANALYSIS
    # --------------------------------------------------------

    results = []

    print(
        f"Downloading data for "
        f"{len(UNIVERSE)} instruments..."
    )

    for ticker in UNIVERSE:

        df = download_data(
            ticker
        )

        features = add_features(
            df
        )

        if (
            features.empty
            or
            len(features) < 220
        ):

            continue

        row = features.iloc[-1]

        required_features = [
            "atr",
            "rsi",
            "sma50",
            "vol_ratio",
            "trend20",
            "trend50",
        ]

        if any(
            pd.isna(
                row[c]
            )
            for c in required_features
        ):

            continue

        try:

            evaluation = (
                evaluate_candidate(
                    row,
                    regime
                )
            )

            price = float(
                row["Close"]
            )

            shares = 0

            if (
                evaluation["action"]
                == "TRADE"
            ):

                shares = (
                    calculate_position_size(
                        price,
                        evaluation["stop"]
                    )
                )

            evaluation[
                "ticker"
            ] = ticker.replace(
                ".NS",
                ""
            )

            evaluation[
                "price"
            ] = price

            evaluation[
                "shares"
            ] = shares

            evaluation[
                "capital"
            ] = (
                shares
                * price
            )

            evaluation[
                "max_loss"
            ] = (
                shares
                *
                evaluation["risk"]
            )

            results.append(
                evaluation
            )

        except Exception as error:

            print(
                f"{ticker}: "
                f"evaluation error: "
                f"{error}"
            )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    results.sort(
        key=lambda item: (
            item["action"] == "TRADE",
            item["score"]
        ),
        reverse=True
    )

    trades = [
        x for x in results
        if x["action"] == "TRADE"
    ][:TOP_N]

    watches = [
        x for x in results
        if x["action"] == "WATCH"
    ][:TOP_N]

    # ========================================================
    # ALERT
    # ========================================================

    lines = []

    lines.append(
        f"MULTI-FACTOR MARKET ALERT "
        f"{VERSION}"
    )

    lines.append(
        now.strftime(
            "%d %b %Y, %H:%M IST"
        )
    )

    lines.append("")

    lines.append(
        "MARKET STATUS: "
        +
        (
            "OPEN / SESSION"
            if market_open
            else
            "WEEKEND / NON-TRADING DAY"
        )
    )

    lines.append(
        f"MARKET REGIME: {regime}"
    )

    if pd.notna(nifty):

        lines.append(
            f"NIFTY: ₹{nifty:,.2f}"
        )

    else:

        lines.append(
            "NIFTY: N/A"
        )

    if pd.notna(sma50):

        lines.append(
            f"SMA50: ₹{sma50:,.2f}"
        )

    else:

        lines.append(
            "SMA50: N/A"
        )

    lines.append(
        f"REGIME REASON: "
        f"{regime_reason}"
    )

    lines.append("")

    lines.append(
        "--- TOP SHORT-TERM "
        "TRADE SETUPS (1–5 SESSIONS) ---"
    )

    if not market_open:

        lines.append(
            "MARKET IS CLOSED."
        )

        lines.append(
            "No new long position "
            "should be initiated today."
        )

    if (
        market_open
        and
        trades
    ):

        for number, item in enumerate(
            trades,
            start=1
        ):

            lines.append("")

            lines.append(
                f"{number}. "
                f"{item['ticker']} — TRADE"
            )

            lines.append(
                f"Price: "
                f"₹{item['price']:,.2f}"
            )

            lines.append(
                f"Calibrated P(UP): "
                f"{item['probability']*100:.1f}%"
            )

            lines.append(
                f"Expected return "
                f"3D / 5D: "
                f"{item['er3']*100:.2f}% / "
                f"{item['er5']*100:.2f}%"
            )

            lines.append(
                f"RSI: "
                f"{item['rsi']:.1f}"
                f" | Volume: "
                f"{item['volume']:.2f}x"
            )

            lines.append(
                f"Entry zone: "
                f"₹{item['price']*0.998:,.2f}"
                f" – "
                f"₹{item['price']*1.002:,.2f}"
            )

            lines.append(
                f"Stop Loss: "
                f"₹{item['stop']:,.2f}"
            )

            lines.append(
                f"Target 1: "
                f"₹{item['target1']:,.2f}"
            )

            lines.append(
                f"Target 2: "
                f"₹{item['target2']:,.2f}"
            )

            lines.append(
                f"Risk/Reward: "
                f"{item['rr1']:.2f} / "
                f"{item['rr2']:.2f}"
            )

            lines.append(
                f"Suggested position: "
                f"{item['shares']} shares "
                f"≈ ₹{item['capital']:,.0f}"
            )

            lines.append(
                f"Maximum planned loss: "
                f"₹{item['max_loss']:,.0f}"
            )

            lines.append(
                "Action: BUY / TRADE only "
                "if price remains inside "
                "the entry zone."
            )

    else:

        lines.append(
            "NO VALID LONG TRADE TODAY"
        )

        lines.append(
            "No candidate currently "
            "satisfies the full V6.3.10 "
            "probability, expectancy, "
            "risk/reward, liquidity and "
            "technical filters."
        )

    # ========================================================
    # WATCHLIST
    # ========================================================

    lines.append("")

    lines.append(
        "--- BEST WATCHLIST SETUPS ---"
    )

    if watches:

        for number, item in enumerate(
            watches,
            start=1
        ):

            lines.append("")

            lines.append(
                f"{number}. "
                f"{item['ticker']} — WATCH"
            )

            lines.append(
                f"Price: "
                f"₹{item['price']:,.2f}"
            )

            lines.append(
                f"P(UP): "
                f"{item['probability']*100:.1f}%"
            )

            lines.append(
                f"ER3 / ER5: "
                f"{item['er3']*100:.2f}% / "
                f"{item['er5']*100:.2f}%"
            )

            lines.append(
                f"RR1 / RR2: "
                f"{item['rr1']:.2f} / "
                f"{item['rr2']:.2f}"
            )

            lines.append(
                f"RSI: "
                f"{item['rsi']:.1f}"
                f" | Volume: "
                f"{item['volume']:.2f}x"
            )

            if item["failures"]:

                lines.append(
                    "Failed filters: "
                    +
                    ",".join(
                        item["failures"]
                    )
                )

            else:

                lines.append(
                    "Failed filters: none"
                )

            lines.append(
                "Action: WATCH / WAIT"
            )

    else:

        lines.append(
            "None."
        )

    # ========================================================
    # SAFETY
    # ========================================================

    lines.append("")

    lines.append(
        "--- MODEL SAFETY ---"
    )

    lines.append(
        "P(UP) is a conservative "
        "empirically calibrated estimate, "
        "not a guaranteed probability "
        "of profit."
    )

    lines.append(
        "V6.3.10 uses positive, "
        "volatility-aware stop losses "
        "and explicit R/R calculations."
    )

    lines.append(
        "No negative stop-loss values "
        "are permitted."
    )

    lines.append(
        f"Position sizing limits planned "
        f"risk to {MAX_RISK_PCT:.2f}% "
        f"of configured capital."
    )

    lines.append(
        "Verify live price, liquidity, "
        "corporate news, market status "
        "and order execution before trading."
    )

    alert = "\n".join(
        lines
    )

    # ========================================================
    # AUDIT
    # ========================================================

    Path(
        "audit"
    ).mkdir(
        exist_ok=True
    )

    timestamp = now.strftime(
        "%Y%m%d_%H%M%S"
    )

    audit_file = (
        Path("audit")
        /
        f"market_alert_v6_3_10_"
        f"{timestamp}.txt"
    )

    audit_file.write_text(
        alert,
        encoding="utf-8"
    )

    print("")
    print(alert)
    print("")

    send_telegram(
        alert
    )


if __name__ == "__main__":

    main()
