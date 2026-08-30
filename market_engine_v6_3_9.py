"""
MULTI-FACTOR MARKET ENGINE V6.3.9

Purpose:
    Short-term Indian equity research/trading screen.

V6.3.9 improvements:
    - Calibrated empirical probabilities
    - 3/5-session horizon emphasis
    - Realistic positive stop-loss calculation
    - Genuine risk/reward calculation
    - Liquidity/volume confirmation
    - Mixed-regime handling
    - Transaction-cost buffer
    - Outlier protection
    - BUY / WATCH / WAIT classification
    - Risk-based position sizing
    - Clear Telegram output
    - IPO retrieval with safe fallback
    - No negative stop losses
    - No zero-RR display for valid candidates

IMPORTANT:
    This is a probabilistic research screen.
    It does not guarantee profit.
"""

from __future__ import annotations

import os
import math
import statistics
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "V6.3.9"

CAPITAL = float(
    os.getenv("ALERT_CAPITAL", "100000")
)

MAX_RISK_PER_TRADE = float(
    os.getenv("MAX_RISK_PER_TRADE", "0.01")
)

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
)

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
)

# ------------------------------------------------------------
# Probability requirements
# ------------------------------------------------------------

MIN_CALIBRATED_P3 = 0.53
MIN_CALIBRATED_P5 = 0.54

WATCH_P3 = 0.51
WATCH_P5 = 0.52

# ------------------------------------------------------------
# Expected return requirements
# ------------------------------------------------------------

MIN_ER3 = 0.0040
MIN_ER5 = 0.0060

WATCH_ER3 = 0.0025
WATCH_ER5 = 0.0035

# ------------------------------------------------------------
# Risk/reward
# ------------------------------------------------------------

MIN_RR1 = 1.20
MIN_RR2 = 1.60

WATCH_RR1 = 1.00
WATCH_RR2 = 1.30

# ------------------------------------------------------------
# Market filters
# ------------------------------------------------------------

MIN_VOLUME_RATIO = 0.80
WATCH_VOLUME_RATIO = 0.60

RSI_MIN = 45
RSI_MAX = 68

# ------------------------------------------------------------
# Trading-cost / safety buffer
# ------------------------------------------------------------

ROUND_TRIP_COST = 0.0020
SAFETY_BUFFER = 0.0015

MIN_NET_EDGE = (
    ROUND_TRIP_COST +
    SAFETY_BUFFER
)

# ------------------------------------------------------------
# Outlier protection
# ------------------------------------------------------------

MAX_REASONABLE_FORWARD_RETURN = 0.40
MIN_REASONABLE_FORWARD_RETURN = -0.40

# ------------------------------------------------------------
# Candidate universe
# ------------------------------------------------------------

SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "INFY.NS",
    "ITC.NS",
    "LT.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "M&M.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "JSWSTEEL.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "BPCL.NS",
    "IOC.NS",
    "SUNPHARMA.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "AUROPHARMA.NS",
    "DIVISLAB.NS",
    "APOLLOHOSP.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "NESTLEIND.NS",
    "HINDUNILVR.NS",
    "TECHM.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "BHARTIARTL.NS",
    "TRENT.NS",
    "DLF.NS",
    "VEDL.NS",
    "SAIL.NS",
    "SHRIRAMFIN.NS",
    "NAUKRI.NS",
    "ABB.NS",
    "BOSCHLTD.NS",
    "HAL.NS",
    "BEL.NS",
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def display_symbol(symbol):
    return symbol.replace(".NS", "")


def fmt_price(value):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"₹{value:,.2f}"


def pct(value):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value * 100:.2f}%"


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, period=14):

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    rsi = rsi.where(
        ~(avg_loss == 0),
        100
    )

    return rsi


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data(
    symbol,
    period="2y"
):

    try:

        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for col in required:

            if col not in df.columns:
                return None

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=["Close"]
        )

        if len(df) < 150:
            return None

        if getattr(
            df.index,
            "tz",
            None
        ) is not None:

            df.index = (
                df.index
                .tz_localize(None)
            )

        return df

    except Exception as exc:

        print(
            f"Download failed "
            f"{symbol}: {exc}"
        )

        return None


# ============================================================
# NIFTY
# ============================================================

def get_nifty():

    return download_data(
        "^NSEI",
        "2y"
    )


# ============================================================
# VIX
# ============================================================

def get_vix():

    try:

        df = yf.download(
            "^INDIAVIX",
            period="1mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return np.nan

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        value = df["Close"].iloc[-1]

        return safe_float(value)

    except Exception:

        return np.nan


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime(nifty):

    close = nifty["Close"]

    sma50 = close.rolling(50).mean()

    current = safe_float(
        close.iloc[-1]
    )

    current_sma50 = safe_float(
        sma50.iloc[-1]
    )

    previous_sma50 = safe_float(
        sma50.iloc[-6]
    )

    if not all(
        np.isfinite(x)
        for x in [
            current,
            current_sma50,
            previous_sma50
        ]
    ):
        return (
            "UNKNOWN",
            current,
            current_sma50,
            "Insufficient Nifty history"
        )

    above = (
        current >
        current_sma50
    )

    rising = (
        current_sma50 >
        previous_sma50
    )

    if above and rising:

        regime = "FAVORABLE"

        reason = (
            "Nifty above rising SMA50"
        )

    elif (
        above
        or rising
    ):

        regime = "MIXED"

        reason = (
            "Nifty/SMA50 signals mixed"
        )

    else:

        regime = "UNFAVORABLE"

        reason = (
            "Nifty below falling SMA50"
        )

    return (
        regime,
        current,
        current_sma50,
        reason
    )


# ============================================================
# FEATURE ENGINE
# ============================================================

def add_features(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["rsi"] = calculate_rsi(
        close,
        14
    )

    df["ema10"] = close.ewm(
        span=10,
        adjust=False
    ).mean()

    df["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["sma20"] = close.rolling(
        20
    ).mean()

    df["sma50"] = close.rolling(
        50
    ).mean()

    df["sma100"] = close.rolling(
        100
    ).mean()

    df["ret1"] = (
        close.pct_change(1)
    )

    df["ret3"] = (
        close.pct_change(3)
    )

    df["ret5"] = (
        close.pct_change(5)
    )

    df["ret10"] = (
        close.pct_change(10)
    )

    df["ret20"] = (
        close.pct_change(20)
    )

    volume_mean = volume.rolling(
        20
    ).mean()

    df["volume_ratio"] = (
        volume /
        volume_mean.replace(
            0,
            np.nan
        )
    )

    daily_return = close.pct_change()

    df["volatility"] = (
        daily_return
        .rolling(20)
        .std()
        * math.sqrt(252)
    )

    # ATR

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(
        14
    ).mean()

    df["atr_pct"] = (
        df["atr"] /
        close
    )

    # Trend

    df["trend_score"] = (
        (close > df["ema20"]).astype(int)
        +
        (df["ema20"] > df["sma50"]).astype(int)
        +
        (df["sma50"] > df["sma100"]).astype(int)
    )

    df["trend_ok"] = (
        df["trend_score"] >= 2
    )

    # Momentum score

    df["momentum"] = (
        0.35 * df["ret5"]
        +
        0.25 * df["ret10"]
        +
        0.20 * df["ret20"]
        +
        0.10 * (
            close / df["ema20"] - 1
        )
        +
        0.10 * (
            close / df["sma50"] - 1
        )
    )

    return df


# ============================================================
# EMPIRICAL PROBABILITY
# ============================================================

def empirical_probability(
    df,
    position,
    horizon
):

    if position < 100:
        return np.nan

    history_start = max(
        50,
        position - 250
    )

    history = df.iloc[
        history_start:position
    ].copy()

    if len(history) < 30:
        return np.nan

    current = df.iloc[position]

    current_rsi = safe_float(
        current["rsi"]
    )

    current_momentum = safe_float(
        current["momentum"]
    )

    current_volume = safe_float(
        current["volume_ratio"]
    )

    if not all(
        np.isfinite(x)
        for x in [
            current_rsi,
            current_momentum
        ]
    ):
        return np.nan

    # Similarity matching.

    rsi_distance = (
        history["rsi"] -
        current_rsi
    ).abs()

    momentum_distance = (
        history["momentum"] -
        current_momentum
    ).abs()

    if np.isfinite(
        current_volume
    ):

        volume_distance = (
            history["volume_ratio"] -
            current_volume
        ).abs()

    else:

        volume_distance = pd.Series(
            0,
            index=history.index
        )

    hist_momentum_std = (
        history["momentum"].std()
    )

    if (
        not np.isfinite(
            hist_momentum_std
        )
        or
        hist_momentum_std == 0
    ):
        hist_momentum_std = 0.01

    distance = (
        rsi_distance / 15
        +
        momentum_distance /
        hist_momentum_std
        +
        volume_distance / 1.5
    )

    selected = history.loc[
        distance.nsmallest(
            min(60, len(distance))
        ).index
    ].copy()

    if len(selected) < 15:
        selected = history.copy()

    future_close = (
        df["Close"]
        .shift(-horizon)
    )

    outcomes = (
        future_close.loc[
            selected.index
        ]
        /
        selected["Close"]
        - 1
    )

    # Remove incomplete observations.

    outcomes = outcomes.dropna()

    # Remove extreme data-provider/corporate-action anomalies
    # from probability estimation.

    outcomes = outcomes[
        (
            outcomes >=
            MIN_REASONABLE_FORWARD_RETURN
        )
        &
        (
            outcomes <=
            MAX_REASONABLE_FORWARD_RETURN
        )
    ]

    if len(outcomes) < 10:
        return np.nan

    # Bayesian smoothing prevents tiny samples from producing
    # unrealistic 0% or 100% probabilities.

    wins = float(
        (outcomes > 0).sum()
    )

    n = float(
        len(outcomes)
    )

    alpha = 3.0
    beta = 3.0

    probability = (
        wins + alpha
    ) / (
        n + alpha + beta
    )

    return float(
        np.clip(
            probability,
            0.05,
            0.95
        )
    )


# ============================================================
# PROBABILITY CALIBRATION
# ============================================================

def calibrate_probability(
    raw_probability,
    horizon
):

    if not np.isfinite(
        raw_probability
    ):
        return np.nan

    # The V6.3.8 calibration showed systematic
    # overconfidence. Compress extreme probabilities
    # toward the empirical center.

    if horizon == 3:

        calibrated = (
            0.50
            +
            0.62 *
            (
                raw_probability -
                0.50
            )
        )

    else:

        calibrated = (
            0.50
            +
            0.64 *
            (
                raw_probability -
                0.50
            )
        )

    return float(
        np.clip(
            calibrated,
            0.35,
            0.75
        )
    )


# ============================================================
# EXPECTED RETURN
# ============================================================

def expected_return(
    probability,
    volatility,
    horizon
):

    if not all(
        np.isfinite(x)
        for x in [
            probability,
            volatility
        ]
    ):
        return np.nan

    daily_vol = (
        volatility /
        math.sqrt(252)
    )

    horizon_vol = (
        daily_vol *
        math.sqrt(horizon)
    )

    # Conservative asymmetric payoff assumption.

    average_win = (
        horizon_vol * 0.85
    )

    average_loss = (
        horizon_vol * 0.75
    )

    gross_expected = (
        probability *
        average_win
        -
        (1 - probability) *
        average_loss
    )

    net_expected = (
        gross_expected -
        ROUND_TRIP_COST
    )

    return float(
        net_expected
    )


# ============================================================
# STOP / TARGET / RR
# ============================================================

def risk_parameters(
    price,
    atr
):

    if (
        not np.isfinite(price)
        or
        price <= 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    if (
        not np.isfinite(atr)
        or
        atr <= 0
    ):
        atr = (
            price * 0.02
        )

    # 1.5 ATR stop.

    risk = max(
        1.50 * atr,
        price * 0.008
    )

    # Never allow stop below 70% of price.

    risk = min(
        risk,
        price * 0.30
    )

    stop = (
        price - risk
    )

    if stop <= 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    target1 = (
        price +
        risk * 1.25
    )

    target2 = (
        price +
        risk * 2.00
    )

    actual_risk = (
        price - stop
    )

    rr1 = (
        target1 - price
    ) / actual_risk

    rr2 = (
        target2 - price
    ) / actual_risk

    return (
        stop,
        target1,
        target2,
        rr1,
        rr2
    )


# ============================================================
# POSITION SIZING
# ============================================================

def position_size(
    price,
    stop
):

    if not all(
        np.isfinite(x)
        for x in [
            price,
            stop
        ]
    ):
        return (
            0,
            0.0,
            0.0
        )

    if (
        price <= 0
        or
        stop <= 0
        or
        stop >= price
    ):
        return (
            0,
            0.0,
            0.0
        )

    risk_per_share = (
        price - stop
    )

    max_loss = (
        CAPITAL *
        MAX_RISK_PER_TRADE
    )

    shares_by_risk = math.floor(
        max_loss /
        risk_per_share
    )

    shares_by_capital = math.floor(
        CAPITAL /
        price
    )

    shares = min(
        shares_by_risk,
        shares_by_capital
    )

    value = (
        shares * price
    )

    planned_loss = (
        shares *
        risk_per_share
    )

    return (
        int(max(0, shares)),
        float(value),
        float(planned_loss)
    )


# ============================================================
# SCORE
# ============================================================

def candidate_score(row):

    components = []

    for field, weight in [
        ("p3", 0.25),
        ("p5", 0.30),
        ("er3", 0.15),
        ("er5", 0.20)
    ]:

        value = safe_float(
            row.get(field)
        )

        if np.isfinite(value):

            if field.startswith("p"):

                normalized = (
                    value - 0.50
                ) / 0.25

            else:

                normalized = (
                    value / 0.02
                )

            components.append(
                weight *
                normalized
            )

    volume = safe_float(
        row.get("volume")
    )

    if np.isfinite(volume):

        components.append(
            0.10 *
            min(
                1.0,
                max(
                    -1.0,
                    volume - 1
                )
            )
        )

    return float(
        sum(components)
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(
    p3,
    p5,
    er3,
    er5,
    rr1,
    rr2,
    rsi,
    volume,
    trend,
    regime
):

    failures = []

    # Probability

    if (
        not np.isfinite(p3)
        or
        p3 < MIN_CALIBRATED_P3
    ):
        failures.append("p3")

    if (
        not np.isfinite(p5)
        or
        p5 < MIN_CALIBRATED_P5
    ):
        failures.append("p5")

    # Expected return

    if (
        not np.isfinite(er3)
        or
        er3 < MIN_ER3
    ):
        failures.append("er3")

    if (
        not np.isfinite(er5)
        or
        er5 < MIN_ER5
    ):
        failures.append("er5")

    # RR

    if (
        not np.isfinite(rr1)
        or
        rr1 < MIN_RR1
    ):
        failures.append("rr1")

    if (
        not np.isfinite(rr2)
        or
        rr2 < MIN_RR2
    ):
        failures.append("rr2")

    # RSI

    if (
        not np.isfinite(rsi)
        or
        rsi < RSI_MIN
        or
        rsi > RSI_MAX
    ):
        failures.append("rsi")

    # Volume

    if (
        not np.isfinite(volume)
        or
        volume < MIN_VOLUME_RATIO
    ):
        failures.append("volume")

    # Trend

    if not trend:
        failures.append("trend")

    # Regime

    # FAVORABLE = normal trade
    # MIXED = possible trade if everything else is strong
    # UNFAVORABLE = never BUY

    if regime == "UNFAVORABLE":
        failures.append("regime")

    # --------------------------------------------------------
    # TRADE
    # --------------------------------------------------------

    trade_ok = (
        len(failures) == 0
    )

    if trade_ok:
        return (
            "TRADE",
            failures
        )

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    watch_score = 0

    if (
        np.isfinite(p3)
        and
        p3 >= WATCH_P3
    ):
        watch_score += 1

    if (
        np.isfinite(p5)
        and
        p5 >= WATCH_P5
    ):
        watch_score += 1

    if (
        np.isfinite(er3)
        and
        er3 >= WATCH_ER3
    ):
        watch_score += 1

    if (
        np.isfinite(er5)
        and
        er5 >= WATCH_ER5
    ):
        watch_score += 1

    if (
        np.isfinite(rr1)
        and
        rr1 >= WATCH_RR1
    ):
        watch_score += 1

    if (
        np.isfinite(rr2)
        and
        rr2 >= WATCH_RR2
    ):
        watch_score += 1

    if watch_score >= 3:

        return (
            "WATCH",
            failures
        )

    return (
        "WAIT",
        failures
    )


# ============================================================
# ANALYSE STOCK
# ============================================================

def analyse_stock(
    symbol,
    nifty,
    regime
):

    df = download_data(
        symbol,
        "2y"
    )

    if df is None:
        return None

    df = add_features(
        df
    )

    # Need sufficient history.

    if len(df) < 150:
        return None

    position = (
        len(df) - 1
    )

    row = df.iloc[
        position
    ]

    price = safe_float(
        row["Close"]
    )

    rsi = safe_float(
        row["rsi"]
    )

    volume = safe_float(
        row["volume_ratio"]
    )

    volatility = safe_float(
        row["volatility"]
    )

    atr = safe_float(
        row["atr"]
    )

    trend = bool(
        row["trend_ok"]
    )

    raw_p3 = empirical_probability(
        df,
        position,
        3
    )

    raw_p5 = empirical_probability(
        df,
        position,
        5
    )

    p3 = calibrate_probability(
        raw_p3,
        3
    )

    p5 = calibrate_probability(
        raw_p5,
        5
    )

    er3 = expected_return(
        p3,
        volatility,
        3
    )

    er5 = expected_return(
        p5,
        volatility,
        5
    )

    (
        stop,
        target1,
        target2,
        rr1,
        rr2
    ) = risk_parameters(
        price,
        atr
    )

    (
        shares,
        position_value,
        max_loss
    ) = position_size(
        price,
        stop
    )

    action, failures = classify(
        p3,
        p5,
        er3,
        er5,
        rr1,
        rr2,
        rsi,
        volume,
        trend,
        regime
    )

    result = {

        "symbol": display_symbol(
            symbol
        ),

        "price": price,

        "raw_p3": raw_p3,
        "raw_p5": raw_p5,

        "p3": p3,
        "p5": p5,

        "er3": er3,
        "er5": er5,

        "stop": stop,
        "target1": target1,
        "target2": target2,

        "rr1": rr1,
        "rr2": rr2,

        "rsi": rsi,

        "volume": volume,

        "trend": trend,

        "regime": regime,

        "action": action,

        "failures": failures,

        "shares": shares,

        "position_value": position_value,

        "max_loss": max_loss
    }

    result["score"] = candidate_score(
        result
    )

    return result


# ============================================================
# IPO
# ============================================================

def get_ipo_data():

    """
    IPO retrieval is deliberately conservative.

    If external sources cannot be reliably retrieved,
    report unavailable rather than inventing an IPO.
    """

    try:

        url = (
            "https://www.nseindia.com/"
            "api/ipo-current-issue"
        )

        headers = {
            "User-Agent":
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)",
            "Accept":
                "application/json,text/plain,*/*",
            "Referer":
                "https://www.nseindia.com/"
        }

        session = requests.Session()

        response = session.get(
            url,
            headers=headers,
            timeout=15
        )

        if response.status_code != 200:
            return []

        payload = response.json()

        if not isinstance(
            payload,
            dict
        ):
            return []

        records = []

        for key in [
            "ipo",
            "data",
            "currentIssue",
            "issues"
        ]:

            value = payload.get(
                key
            )

            if isinstance(
                value,
                list
            ):
                records = value
                break

        cleaned = []

        for item in records:

            if not isinstance(
                item,
                dict
            ):
                continue

            name = (
                item.get("companyName")
                or
                item.get("name")
                or
                item.get("symbol")
            )

            if name:
                cleaned.append(
                    str(name)
                )

        return cleaned[:10]

    except Exception:

        return []


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    message
):

    if not TELEGRAM_BOT_TOKEN:
        print(
            "Telegram token not configured."
        )
        return False

    if not TELEGRAM_CHAT_ID:
        print(
            "Telegram chat ID not configured."
        )
        return False

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "disable_web_page_preview": True
    }

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=20
        )

        if response.ok:
            return True

        print(
            "Telegram error:",
            response.text[:500]
        )

        return False

    except Exception as exc:

        print(
            "Telegram exception:",
            exc
        )

        return False


# ============================================================
# FORMAT ALERT
# ============================================================

def build_alert(
    results,
    regime,
    nifty,
    sma50,
    vix
):

    now = datetime.now().strftime(
        "%d %b %Y, %H:%M IST"
    )

    weekday = datetime.now().weekday()

    market_closed = (
        weekday >= 5
    )

    lines = []

    lines.append(
        f"MULTI-FACTOR MARKET ALERT {VERSION}"
    )

    lines.append(
        now
    )

    lines.append("")

    if market_closed:

        lines.append(
            "MARKET STATUS: "
            "WEEKEND / NON-TRADING DAY"
        )

    else:

        lines.append(
            "MARKET STATUS: "
            "TRADING DAY"
        )

    lines.append(
        f"MARKET REGIME: {regime}"
    )

    lines.append(
        f"NIFTY: {fmt_price(nifty)} | "
        f"SMA50: {fmt_price(sma50)}"
    )

    lines.append(
        f"INDIA VIX: {fmt_price(vix)}"
    )

    lines.append(
        "REGIME REASON: "
        + (
            "Favorable trend backdrop"
            if regime == "FAVORABLE"
            else
            "Mixed trend signals"
            if regime == "MIXED"
            else
            "Unfavorable trend backdrop"
        )
    )

    lines.append("")

    lines.append(
        "--- TOP SHORT-TERM TRADE SETUPS ---"
    )

    trades = [
        x for x in results
        if x["action"] == "TRADE"
    ]

    watches = [
        x for x in results
        if x["action"] == "WATCH"
    ]

    trades.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    watches.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    if market_closed:

        lines.append("")

        lines.append(
            "MARKET IS CLOSED."
        )

        lines.append(
            "No new long position should "
            "be initiated today."
        )

    if not trades:

        lines.append("")

        lines.append(
            "NO VALID LONG TRADE TODAY"
        )

        lines.append(
            "No candidate currently satisfies "
            "the V6.3.9 probability, expected-"
            "return, risk/reward, liquidity "
            "and market requirements."
        )

    else:

        for i, item in enumerate(
            trades[:5],
            start=1
        ):

            lines.append("")
            lines.append(
                f"{i}. {item['symbol']} — "
                f"TRADE"
            )

            lines.append(
                f"Price: "
                f"{fmt_price(item['price'])}"
            )

            lines.append(
                f"Calibrated P3 / P5: "
                f"{pct(item['p3'])} / "
                f"{pct(item['p5'])}"
            )

            lines.append(
                f"Expected 3D / 5D return: "
                f"{pct(item['er3'])} / "
                f"{pct(item['er5'])}"
            )

            lines.append(
                f"RR1 / RR2: "
                f"{item['rr1']:.2f} / "
                f"{item['rr2']:.2f}"
            )

            lines.append(
                f"RSI: {item['rsi']:.1f} | "
                f"Volume: {item['volume']:.2f}x"
            )

            lines.append(
                f"Entry: "
                f"{fmt_price(item['price'])}"
            )

            lines.append(
                f"Stop Loss: "
                f"{fmt_price(item['stop'])}"
            )

            lines.append(
                f"Target 1: "
                f"{fmt_price(item['target1'])}"
            )

            lines.append(
                f"Target 2: "
                f"{fmt_price(item['target2'])}"
            )

            lines.append(
                "Expected holding: "
                "3–5 sessions"
            )

            lines.append(
                f"Suggested position: "
                f"{item['shares']} shares "
                f"≈ {fmt_price(item['position_value'])}"
            )

            lines.append(
                f"Maximum planned loss: "
                f"{fmt_price(item['max_loss'])}"
            )

            if market_closed:

                lines.append(
                    "Action: "
                    "WATCH UNTIL NEXT OPEN"
                )

            else:

                lines.append(
                    "Action: BUY / MANAGE RISK"
                )

    lines.append("")

    lines.append(
        "--- BEST WATCHLIST SETUPS ---"
    )

    if not watches:

        lines.append(
            "None."
        )

    else:

        for i, item in enumerate(
            watches[:5],
            start=1
        ):

            lines.append("")

            lines.append(
                f"{i}. {item['symbol']} — WATCH"
            )

            lines.append(
                f"Price: "
                f"{fmt_price(item['price'])}"
            )

            lines.append(
                f"Calibrated P3 / P5: "
                f"{pct(item['p3'])} / "
                f"{pct(item['p5'])}"
            )

            lines.append(
                f"ER3 / ER5: "
                f"{pct(item['er3'])} / "
                f"{pct(item['er5'])}"
            )

            lines.append(
                f"RR1 / RR2: "
                f"{item['rr1']:.2f} / "
                f"{item['rr2']:.2f}"
            )

            lines.append(
                f"RSI: {item['rsi']:.1f} | "
                f"Volume: {item['volume']:.2f}x"
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

    lines.append("")

    lines.append(
        "--- IPO OPEN / UPCOMING ---"
    )

    ipo = get_ipo_data()

    if not ipo:

        lines.append(
            "IPO DATA UNAVAILABLE | "
            "RETRIEVAL FAILED"
        )

        lines.append(
            "Verify current/upcoming issues "
            "directly on NSE before applying."
        )

    else:

        lines.append(
            "IPO records retrieved. "
            "Verify issue dates, price band "
            "and subscription status directly "
            "before applying."
        )

        for name in ipo[:8]:

            lines.append(
                f"• {name}"
            )

    lines.append("")

    lines.append(
        "--- MODEL NOTES ---"
    )

    lines.append(
        "V6.3.9 uses calibrated empirical "
        "probabilities rather than treating "
        "raw model probabilities as guaranteed."
    )

    lines.append(
        "Primary research horizon: "
        "3–5 trading sessions."
    )

    lines.append(
        "Expected returns include a configurable "
        "transaction-cost buffer."
    )

    lines.append(
        "Position sizing is based on configured "
        "capital and maximum planned loss."
    )

    lines.append(
        "This is a probabilistic research screen "
        "and does not guarantee profit."
    )

    lines.append(
        "Verify live price, liquidity, corporate "
        "news, market status and execution before trading."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Starting {VERSION}..."
    )

    nifty = get_nifty()

    if nifty is None:

        raise RuntimeError(
            "Unable to retrieve Nifty data."
        )

    (
        regime,
        nifty_value,
        sma50,
        regime_reason
    ) = market_regime(
        nifty
    )

    vix = get_vix()

    print(
        f"Regime: {regime}"
    )

    print(
        f"Nifty: {nifty_value}"
    )

    print(
        f"SMA50: {sma50}"
    )

    print(
        f"VIX: {vix}"
    )

    results = []

    for number, symbol in enumerate(
        SYMBOLS,
        start=1
    ):

        print(
            f"[{number}/{len(SYMBOLS)}] "
            f"{symbol}"
        )

        try:

            result = analyse_stock(
                symbol,
                nifty,
                regime
            )

            if result is not None:

                results.append(
                    result
                )

        except Exception as exc:

            print(
                f"Analysis failed "
                f"{symbol}: {exc}"
            )

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    alert = build_alert(
        results,
        regime,
        nifty_value,
        sma50,
        vix
    )

    print("")
    print("=" * 70)
    print(alert)
    print("=" * 70)

    sent = send_telegram(
        alert
    )

    print(
        "Telegram sent:",
        sent
    )

    # --------------------------------------------------------
    # Save current research output
    # --------------------------------------------------------

    os.makedirs(
        "audit",
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output = pd.DataFrame(
        results
    )

    if not output.empty:

        output.to_csv(
            f"audit/"
            f"live_candidates_v6_3_9_"
            f"{timestamp}.csv",
            index=False
        )

    with open(
        f"audit/"
        f"alert_v6_3_9_"
        f"{timestamp}.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(alert)


if __name__ == "__main__":
    main()
