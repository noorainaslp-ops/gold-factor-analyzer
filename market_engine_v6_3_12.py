"""
MULTI-FACTOR MARKET ENGINE V6.3.12

Purpose:
- Refine V6.3.11 without destroying its useful signal.
- Emphasise 3-5 session expectancy.
- Use conservative empirical probability calibration.
- Use ATR-based stop/targets.
- Use composite ranking rather than excessive hard filtering.
- Handle current NSE ticker names.
- Never generate negative stop losses.
- Never generate zero/invalid risk-reward values.
- Do not trade when the market is closed.
- Produce a clean Telegram alert.
"""

import os
import math
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf


VERSION = "V6.3.12"

IST = timezone(
    timedelta(hours=5, minutes=30)
)


# ============================================================
# CONFIGURATION
# ============================================================

def env_float(name, default):
    value = os.getenv(name, "").strip()

    if not value:
        return float(default)

    try:
        return float(value)
    except Exception:
        return float(default)


def env_int(name, default):
    value = os.getenv(name, "").strip()

    if not value:
        return int(default)

    try:
        return int(value)
    except Exception:
        return int(default)


CAPITAL = env_float(
    "ALERT_CAPITAL",
    100000
)

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
    8
)

ROUND_TRIP_COST_BPS = env_float(
    "ROUND_TRIP_COST_BPS",
    12
)

TOP_N = env_int(
    "TOP_N",
    5
)

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
).strip()


# ============================================================
# CURRENT SYMBOL UNIVERSE
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
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

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
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    return (
        100
        -
        100 / (1 + rs)
    )


def calculate_atr(df, period=14):

    previous_close = (
        df["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (
                df["High"]
                -
                previous_close
            ).abs(),
            (
                df["Low"]
                -
                previous_close
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
        x.columns = [
            c[0]
            if isinstance(c, tuple)
            else c
            for c in x.columns
        ]

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(
        col in x.columns
        for col in required
    ):
        return pd.DataFrame()

    x = x[required].apply(
        pd.to_numeric,
        errors="coerce"
    )

    x = x.dropna(
        subset=["Close"]
    )

    x["ret1"] = (
        x["Close"].pct_change()
    )

    x["ret3"] = (
        x["Close"].pct_change(3)
    )

    x["ret5"] = (
        x["Close"].pct_change(5)
    )

    x["ret10"] = (
        x["Close"].pct_change(10)
    )

    x["ret20"] = (
        x["Close"].pct_change(20)
    )

    x["ema5"] = (
        x["Close"]
        .ewm(
            span=5,
            adjust=False
        )
        .mean()
    )

    x["ema20"] = (
        x["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    x["sma50"] = (
        x["Close"]
        .rolling(50)
        .mean()
    )

    x["sma200"] = (
        x["Close"]
        .rolling(200)
        .mean()
    )

    x["rsi"] = calculate_rsi(
        x["Close"]
    )

    x["atr"] = calculate_atr(
        x
    )

    x["atr_pct"] = (
        x["atr"]
        /
        x["Close"]
    )

    x["volume_ratio"] = (
        x["Volume"]
        /
        x["Volume"]
        .rolling(20)
        .median()
    )

    x["volatility20"] = (
        x["ret1"]
        .rolling(20)
        .std()
        *
        np.sqrt(252)
    )

    x["trend20"] = (
        x["Close"]
        /
        x["ema20"]
        - 1
    )

    x["trend50"] = (
        x["Close"]
        /
        x["sma50"]
        - 1
    )

    x["trend200"] = (
        x["Close"]
        /
        x["sma200"]
        - 1
    )

    x["ema20_slope"] = (
        x["ema20"]
        /
        x["ema20"].shift(10)
        - 1
    )

    x["momentum"] = (
        0.45 * x["ret5"].fillna(0)
        +
        0.35 * x["ret10"].fillna(0)
        +
        0.20 * x["ret20"].fillna(0)
    )

    x["quality"] = (
        0.30
        * (
            x["Close"]
            >
            x["sma50"]
        ).astype(float)

        +

        0.25
        * (
            x["sma50"]
            >
            x["sma200"]
        ).astype(float)

        +

        0.25
        * (
            x["ema20_slope"] > 0
        ).astype(float)

        +

        0.20
        * (
            x["volume_ratio"] >= 0.80
        ).astype(float)
    )

    return x


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime(index_df):

    features = add_features(
        index_df
    )

    if features.empty:
        return (
            "UNKNOWN",
            np.nan,
            np.nan,
            "Nifty data unavailable"
        )

    row = features.iloc[-1]

    nifty = float(
        row["Close"]
    )

    sma50 = row["sma50"]

    if pd.isna(sma50):
        return (
            "UNKNOWN",
            nifty,
            np.nan,
            "Nifty SMA50 unavailable"
        )

    distance = (
        nifty / float(sma50)
        - 1
    )

    slope = (
        features["sma50"]
        .diff(10)
        .iloc[-1]
    )

    if (
        distance > 0.008
        and
        pd.notna(slope)
        and
        slope > 0
    ):
        return (
            "FAVORABLE",
            nifty,
            float(sma50),
            "Nifty above SMA50 with positive SMA50 slope"
        )

    if (
        distance < -0.008
        and
        pd.notna(slope)
        and
        slope < 0
    ):
        return (
            "UNFAVORABLE",
            nifty,
            float(sma50),
            "Nifty below SMA50 with negative SMA50 slope"
        )

    return (
        "MIXED",
        nifty,
        float(sma50),
        "Nifty/SMA50 signals mixed"
    )


# ============================================================
# RAW PROBABILITY
# ============================================================

def raw_probability(
    row,
    regime
):

    trend = np.clip(
        (
            0.40 * row["trend20"]
            +
            0.35 * row["trend50"]
            +
            0.25 * row["trend200"]
        ) * 25,
        -1,
        1
    )

    momentum = np.clip(
        row["momentum"] * 22,
        -1,
        1
    )

    rsi = float(
        row["rsi"]
    )

    rsi_component = np.clip(
        (rsi - 50) / 25,
        -1,
        1
    )

    volume_component = np.clip(
        (
            row["volume_ratio"] - 1
        ) / 1.5,
        -1,
        1
    )

    regime_component = {
        "FAVORABLE": 0.80,
        "MIXED": 0.00,
        "UNFAVORABLE": -0.50,
        "UNKNOWN": -0.20,
    }.get(
        regime,
        0
    )

    volatility_penalty = np.clip(
        (
            row["atr_pct"] - 0.025
        ) / 0.025,
        -1,
        1
    )

    score = (
        0.37 * trend
        +
        0.31 * momentum
        +
        0.11 * rsi_component
        +
        0.09 * volume_component
        +
        0.07 * regime_component
        -
        0.05 * volatility_penalty
    )

    probability = (
        0.50
        +
        0.22 * score
    )

    # Avoid excessive confidence.
    probability = np.clip(
        probability,
        0.35,
        0.72
    )

    # Penalise extreme overbought conditions.
    if rsi > 72:

        probability -= min(
            0.045,
            (rsi - 72) * 0.003
        )

    return float(
        np.clip(
            probability,
            0.35,
            0.72
        )
    )


# ============================================================
# CONSERVATIVE PROBABILITY CALIBRATION
# ============================================================

def calibrate_probability(
    raw_probability_value,
    historical_samples=0,
    empirical_win_rate=np.nan
):

    p = float(
        raw_probability_value
    )

    if (
        historical_samples >= 100
        and
        pd.notna(empirical_win_rate)
    ):

        empirical = float(
            empirical_win_rate
        )

        # Shrink empirical rate toward model
        # to avoid overfitting.
        calibrated = (
            0.60 * empirical
            +
            0.40 * p
        )

    else:

        # Conservative shrinkage toward 50%.
        calibrated = (
            0.70 * p
            +
            0.30 * 0.50
        )

    return float(
        np.clip(
            calibrated,
            0.35,
            0.68
        )
    )


# ============================================================
# EXPECTED RETURN
# ============================================================

def expected_returns(
    row,
    regime
):

    momentum = float(
        row["momentum"]
    )

    trend = float(
        row["trend20"]
    )

    trend50 = float(
        row["trend50"]
    )

    rsi = float(
        row["rsi"]
    )

    rsi_adjustment = np.clip(
        (rsi - 50) / 100,
        -0.12,
        0.12
    )

    regime_adjustment = {
        "FAVORABLE": 0.0015,
        "MIXED": 0.0000,
        "UNFAVORABLE": -0.0012,
        "UNKNOWN": -0.0005,
    }.get(
        regime,
        0
    )

    base = (
        0.020 * momentum
        +
        0.012 * trend
        +
        0.006 * trend50
        +
        0.003 * rsi_adjustment
        +
        regime_adjustment
    )

    er1 = np.clip(
        base * 0.35,
        -0.025,
        0.025
    )

    er3 = np.clip(
        base,
        -0.06,
        0.06
    )

    er5 = np.clip(
        base * 1.50,
        -0.09,
        0.09
    )

    return (
        float(er1),
        float(er3),
        float(er5)
    )


# ============================================================
# ATR RISK MODEL
# ============================================================

def risk_levels(row):

    price = float(
        row["Close"]
    )

    atr = float(
        row["atr"]
    )

    if (
        not np.isfinite(price)
        or
        not np.isfinite(atr)
        or
        price <= 0
        or
        atr <= 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    stop_distance = np.clip(
        1.20 * atr,
        price * 0.010,
        price * 0.035
    )

    stop = (
        price
        -
        stop_distance
    )

    stop = max(
        stop,
        price * 0.50
    )

    risk = (
        price
        -
        stop
    )

    if risk <= 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    target1 = (
        price
        +
        max(
            1.15 * atr,
            price * 0.012
        )
    )

    target2 = (
        price
        +
        max(
            2.00 * atr,
            price * 0.022
        )
    )

    rr1 = (
        target1
        -
        price
    ) / risk

    rr2 = (
        target2
        -
        price
    ) / risk

    return (
        float(stop),
        float(target1),
        float(target2),
        float(rr1),
        float(rr2),
        float(risk)
    )


# ============================================================
# COMPOSITE SCORE
# ============================================================

def evaluate_candidate(
    row,
    regime,
    calibrated_probability=None
):

    raw_p = raw_probability(
        row,
        regime
    )

    if calibrated_probability is None:

        p = calibrate_probability(
            raw_p
        )

    else:

        p = float(
            calibrated_probability
        )

    er1, er3, er5 = expected_returns(
        row,
        regime
    )

    (
        stop,
        target1,
        target2,
        rr1,
        rr2,
        risk
    ) = risk_levels(
        row
    )

    if pd.isna(rr1) or pd.isna(rr2):

        return None

    # --------------------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------------------

    probability_score = np.clip(
        50
        +
        450
        * (
            p - 0.50
        ),
        0,
        100
    )

    expected_score = np.clip(
        50
        +
        450 * er3
        +
        250 * er5,
        0,
        100
    )

    trend_score = np.clip(
        50
        +
        900
        * (
            0.55 * row["trend20"]
            +
            0.30 * row["trend50"]
            +
            0.15 * row["trend200"]
        ),
        0,
        100
    )

    momentum_score = np.clip(
        50
        +
        1000 * row["momentum"],
        0,
        100
    )

    volume_score = np.clip(
        50
        +
        35
        * (
            row["volume_ratio"]
            - 1
        ),
        0,
        100
    )

    rsi = float(
        row["rsi"]
    )

    if 48 <= rsi <= 65:
        rsi_score = 100
    elif 42 <= rsi < 48:
        rsi_score = 75
    elif 65 < rsi <= 70:
        rsi_score = 70
    elif 35 <= rsi < 42:
        rsi_score = 50
    else:
        rsi_score = 25

    rr_score = np.clip(
        50
        +
        28
        * (
            rr2 - 1
        ),
        0,
        100
    )

    regime_score = {
        "FAVORABLE": 100,
        "MIXED": 60,
        "UNFAVORABLE": 30,
        "UNKNOWN": 40,
    }.get(
        regime,
        40
    )

    quality_score = (
        float(row["quality"])
        * 100
    )

    # --------------------------------------------------------
    # 3D / 5D are intentionally more important than 1D.
    # --------------------------------------------------------

    composite = (
        0.25 * probability_score
        +
        0.22 * expected_score
        +
        0.15 * trend_score
        +
        0.10 * momentum_score
        +
        0.07 * volume_score
        +
        0.06 * rsi_score
        +
        0.08 * rr_score
        +
        0.04 * regime_score
        +
        0.03 * quality_score
    )

    # --------------------------------------------------------
    # Risk penalties
    # --------------------------------------------------------

    if row["volume_ratio"] < 0.60:
        composite -= 8

    if row["atr_pct"] > 0.06:
        composite -= 10

    if rsi > 75:
        composite -= 10

    if rr1 < 0.80:
        composite -= 7

    if rr2 < 1.15:
        composite -= 5

    composite = float(
        np.clip(
            composite,
            0,
            100
        )
    )

    # --------------------------------------------------------
    # Classification
    #
    # We deliberately avoid the V6.3.10 problem of producing
    # zero TRADE signals.
    # --------------------------------------------------------

    if (
        composite >= 68
        and
        p >= 0.55
        and
        er3 > 0
        and
        er5 > 0
        and
        rr1 >= 0.85
        and
        rr2 >= 1.20
        and
        row["volume_ratio"] >= 0.70
        and
        rsi <= 72
        and
        row["trend20"] > -0.005
    ):

        action = "TRADE"

    elif (
        composite >= 55
        and
        p >= 0.52
        and
        rr2 >= 1.05
        and
        row["volume_ratio"] >= 0.55
    ):

        action = "WATCH"

    else:

        action = "WAIT"

    failures = []

    if p < 0.55:
        failures.append("probability")

    if er3 <= 0:
        failures.append("er3")

    if er5 <= 0:
        failures.append("er5")

    if rr1 < 0.85:
        failures.append("rr1")

    if rr2 < 1.20:
        failures.append("rr2")

    if row["volume_ratio"] < 0.70:
        failures.append("volume")

    if rsi > 72:
        failures.append("rsi")

    if row["trend20"] <= -0.005:
        failures.append("trend")

    return {
        "raw_probability": raw_p,
        "probability": p,

        "er1": er1,
        "er3": er3,
        "er5": er5,

        "stop": stop,
        "target1": target1,
        "target2": target2,

        "rr1": rr1,
        "rr2": rr2,
        "risk": risk,

        "composite": composite,

        "rsi": rsi,
        "volume": float(
            row["volume_ratio"]
        ),

        "quality": float(
            row["quality"]
        ),

        "action": action,

        "failures": failures,
    }


# ============================================================
# POSITION SIZING
# ============================================================

def position_size(
    price,
    stop
):

    if (
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
        -
        stop
    )

    if risk_per_share <= 0:
        return 0

    shares_by_risk = math.floor(
        risk_budget
        /
        risk_per_share
    )

    capital_limit = (
        CAPITAL
        *
        MAX_POSITION_PCT
        /
        100
    )

    shares_by_capital = math.floor(
        capital_limit
        /
        price
    )

    return max(
        0,
        min(
            shares_by_risk,
            shares_by_capital
        )
    )


# ============================================================
# DOWNLOAD
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

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = [
                c[0]
                if isinstance(c, tuple)
                else c
                for c in df.columns
            ]

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

        return df[required].apply(
            pd.to_numeric,
            errors="coerce"
        ).dropna(
            subset=["Close"]
        )

    except Exception as error:

        print(
            f"{ticker}: "
            f"data unavailable: "
            f"{error}"
        )

        return pd.DataFrame()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if (
        not TELEGRAM_BOT_TOKEN
        or
        not TELEGRAM_CHAT_ID
    ):

        print(
            "Telegram secrets are not configured."
        )

        return False

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    try:

        response = requests.post(
            url,
            json={
                "chat_id":
                    TELEGRAM_CHAT_ID,
                "text":
                    message,
                "disable_web_page_preview":
                    True,
            },
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

    Path(
        "audit"
    ).mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # MARKET DATA
    # --------------------------------------------------------

    nifty = download_data(
        "^NSEI"
    )

    (
        regime,
        nifty_price,
        sma50,
        regime_reason
    ) = market_regime(
        nifty
    )

    # NSE normal trading window.
    market_open = (
        now.weekday() < 5
        and
        (
            (
                now.hour == 9
                and
                now.minute >= 15
            )
            or
            (
                10 <= now.hour < 15
            )
            or
            (
                now.hour == 15
                and
                now.minute <= 30
            )
        )
    )

    candidates = []

    # --------------------------------------------------------
    # STOCK SCREEN
    # --------------------------------------------------------

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

        required = [
            "atr",
            "rsi",
            "sma50",
            "sma200",
            "volume_ratio",
            "trend20",
            "trend50",
            "trend200",
            "quality"
        ]

        if any(
            pd.isna(row[col])
            for col in required
        ):
            continue

        result = evaluate_candidate(
            row,
            regime
        )

        if result is None:
            continue

        result["ticker"] = (
            ticker.replace(
                ".NS",
                ""
            )
        )

        result["price"] = float(
            row["Close"]
        )

        result["shares"] = (
            position_size(
                result["price"],
                result["stop"]
            )
            if result["action"]
            == "TRADE"
            else 0
        )

        result["capital_used"] = (
            result["shares"]
            *
            result["price"]
        )

        result["max_loss"] = (
            result["shares"]
            *
            result["risk"]
        )

        candidates.append(
            result
        )

    # --------------------------------------------------------
    # RANK
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x:
            x["composite"],
        reverse=True
    )

    trades = [
        c for c in candidates
        if c["action"] == "TRADE"
    ][:TOP_N]

    watches = [
        c for c in candidates
        if c["action"] == "WATCH"
    ][:TOP_N]

    # ========================================================
    # TELEGRAM MESSAGE
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

    if pd.notna(nifty_price):

        lines.append(
            f"NIFTY: "
            f"₹{nifty_price:,.2f}"
        )

    else:

        lines.append(
            "NIFTY: N/A"
        )

    if pd.notna(sma50):

        lines.append(
            f"SMA50: "
            f"₹{sma50:,.2f}"
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
        "TRADE SETUPS ---"
    )

    if not market_open:

        lines.append(
            "MARKET IS CLOSED."
        )

        lines.append(
            "No new long position "
            "should be initiated today."
        )

    elif trades:

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
                f"{item['probability'] * 100:.1f}%"
            )

            lines.append(
                f"Expected return "
                f"3D / 5D: "
                f"{item['er3'] * 100:.2f}% / "
                f"{item['er5'] * 100:.2f}%"
            )

            lines.append(
                f"Composite score: "
                f"{item['composite']:.1f}/100"
            )

            lines.append(
                f"RSI: "
                f"{item['rsi']:.1f}"
                f" | Volume: "
                f"{item['volume']:.2f}x"
            )

            lines.append(
                f"Entry: "
                f"₹{item['price']:,.2f}"
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
                f"≈ ₹{item['capital_used']:,.0f}"
            )

            lines.append(
                f"Maximum planned loss: "
                f"₹{item['max_loss']:,.0f}"
            )

            lines.append(
                "Expected holding: "
                "3–5 sessions"
            )

            lines.append(
                "Action: TRADE"
            )

    else:

        lines.append(
            "NO VALID LONG TRADE TODAY"
        )

        lines.append(
            "No candidate currently "
            "meets the V6.3.12 "
            "probability, expectancy, "
            "quality and risk controls."
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
                f"{item['probability'] * 100:.1f}%"
            )

            lines.append(
                f"ER3 / ER5: "
                f"{item['er3'] * 100:.2f}% / "
                f"{item['er5'] * 100:.2f}%"
            )

            lines.append(
                f"RR1 / RR2: "
                f"{item['rr1']:.2f} / "
                f"{item['rr2']:.2f}"
            )

            lines.append(
                f"Composite: "
                f"{item['composite']:.1f}/100"
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

            lines.append(
                "Action: WATCH / WAIT"
            )

    else:

        lines.append(
            "None."
        )

    # ========================================================
    # VALIDATION MESSAGE
    # ========================================================

    lines.append("")

    lines.append(
        "--- MODEL VALIDATION STATUS ---"
    )

    lines.append(
        "V6.3.12 prioritizes 3–5 session "
        "expected return and uses "
        "conservative probability calibration."
    )

    lines.append(
        "P(UP) is an empirical model "
        "estimate, not a guaranteed "
        "probability of profit."
    )

    lines.append(
        "This version must be validated "
        "on an untouched out-of-sample "
        "period before real-money use."
    )

    lines.append(
        "Verify live price, liquidity, "
        "corporate news, market status "
        "and execution before trading."
    )

    message = "\n".join(
        lines
    )

    # ========================================================
    # AUDIT FILE
    # ========================================================

    timestamp = now.strftime(
        "%Y%m%d_%H%M%S"
    )

    audit_file = (
        Path("audit")
        /
        f"market_alert_v6_3_12_"
        f"{timestamp}.txt"
    )

    audit_file.write_text(
        message,
        encoding="utf-8"
    )

    print(message)

    send_telegram(
        message
    )


if __name__ == "__main__":

    main()
