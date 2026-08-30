"""
MULTI-FACTOR MARKET ENGINE V6.3.11

V6.3.11 design:
- V6.3.9 predictive structure retained as baseline.
- Composite ranking instead of an all-or-nothing mega-filter.
- Conservative probability calibration.
- Genuine ATR-based risk/reward.
- Regime is a score component, not a hard rejection.
- Liquidity and trend are explicitly scored.
- Position sizing is risk based.
- Safe handling of empty GitHub secrets.
- Correct current NSE/Yahoo symbols for changed listings.
"""

import os
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


VERSION = "V6.3.13"

IST = timezone(
    timedelta(hours=5, minutes=30)
)


# ============================================================
# ENVIRONMENT
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

TELEGRAM_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
).strip()


# ============================================================
# UNIVERSE
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


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df,
    period=14
):

    previous_close = (
        df["Close"].shift(1)
    )

    tr = pd.concat(
        [

            df["High"]
            - df["Low"],

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
    ).max(
        axis=1
    )

    return tr.ewm(
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
        c in x.columns
        for c in required
    ):

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

    x["sma20"] = (
        x["Close"]
        .rolling(20)
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

    x["vol_ratio"] = (
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
        * np.sqrt(252)
    )

    x["trend20"] = (
        x["Close"]
        /
        x["sma20"]
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

    x["mom_score"] = (
        0.45 * x["ret5"].fillna(0)
        +
        0.35 * x["ret20"].fillna(0)
        +
        0.20 * x["ret3"].fillna(0)
    )

    x["quality"] = (
        0.40
        * (
            x["Close"]
            >
            x["sma50"]
        ).astype(float)

        +

        0.30
        * (
            x["sma50"]
            >
            x["sma200"]
        ).astype(float)

        +

        0.30
        * (
            x["trend20"] > 0
        ).astype(float)
    )

    return x


# ============================================================
# MARKET REGIME
# ============================================================

def determine_market_regime(
    index_df
):

    f = add_features(
        index_df
    )

    if f.empty:
        return (
            "UNKNOWN",
            np.nan,
            np.nan,
            "Nifty data unavailable"
        )

    row = f.iloc[-1]

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

    sma_slope = (
        f["sma50"]
        .diff(5)
        .iloc[-1]
    )

    distance = (
        nifty / sma50
        - 1
    )

    if (
        distance > 0.008
        and
        pd.notna(sma_slope)
        and
        sma_slope > 0
    ):

        return (
            "FAVORABLE",
            nifty,
            sma50,
            "Nifty above SMA50 with positive SMA50 slope"
        )

    if (
        distance < -0.008
        and
        pd.notna(sma_slope)
        and
        sma_slope < 0
    ):

        return (
            "UNFAVORABLE",
            nifty,
            sma50,
            "Nifty below SMA50 with negative SMA50 slope"
        )

    return (
        "MIXED",
        nifty,
        sma50,
        "Nifty/SMA50 signals mixed"
    )


# ============================================================
# RAW MODEL
# ============================================================

def raw_probability(
    row,
    regime
):

    trend = np.clip(
        (
            0.40 * row["trend20"]
            +
            0.40 * row["trend50"]
            +
            0.20 * row["trend200"]
        ) * 30,
        -1,
        1
    )

    momentum = np.clip(
        row["mom_score"] * 25,
        -1,
        1
    )

    rsi = float(
        row["rsi"]
    )

    rsi_score = np.clip(
        (rsi - 50) / 25,
        -1,
        1
    )

    volume = np.clip(
        (
            row["vol_ratio"] - 1
        ) / 1.5,
        -1,
        1
    )

    volatility_penalty = np.clip(
        (
            row["atr_pct"] - 0.025
        ) / 0.025,
        -1,
        1
    )

    regime_score = {
        "FAVORABLE": 1.0,
        "MIXED": 0.0,
        "UNFAVORABLE": -0.60,
    }.get(
        regime,
        0
    )

    score = (

        0.38 * trend
        +
        0.28 * momentum
        +
        0.12 * rsi_score
        +
        0.10 * volume
        -
        0.06 * volatility_penalty
        +
        0.06 * regime_score
    )

    p = (
        0.50
        +
        0.22 * score
    )

    # Avoid over-rewarding very high RSI.
    if rsi > 72:

        p -= min(
            0.05,
            (rsi - 72) * 0.004
        )

    return float(
        np.clip(
            p,
            0.30,
            0.75
        )
    )


# ============================================================
# EXPECTED RETURNS
# ============================================================

def expected_returns(
    row,
    regime
):

    trend = float(
        row["trend20"]
    )

    momentum = float(
        row["mom_score"]
    )

    rsi = float(
        row["rsi"]
    )

    rsi_bias = np.clip(
        (rsi - 50) / 100,
        -0.15,
        0.15
    )

    regime_bias = {
        "FAVORABLE": 0.0015,
        "MIXED": 0.0,
        "UNFAVORABLE": -0.0010,
    }.get(
        regime,
        0
    )

    base = (

        0.012 * trend
        +
        0.022 * momentum
        +
        0.004 * rsi_bias
        +
        regime_bias
    )

    er1 = np.clip(
        base * 0.40,
        -0.02,
        0.02
    )

    er3 = np.clip(
        base,
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
# RISK / REWARD
# ============================================================

def risk_levels(row):

    price = float(
        row["Close"]
    )

    atr = float(
        row["atr"]
    )

    stop_distance = np.clip(
        1.20 * atr,
        price * 0.008,
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

    target1 = (
        price
        +
        max(
            1.05 * atr,
            price * 0.010
        )
    )

    target2 = (
        price
        +
        max(
            1.90 * atr,
            price * 0.018
        )
    )

    risk = (
        price
        -
        stop
    )

    if risk <= 0:
        return (
            stop,
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
        float(stop),
        float(target1),
        float(target2),
        float(rr1),
        float(rr2),
        float(risk)
    )


# ============================================================
# COMPOSITE RANKING
# ============================================================

def evaluate_candidate(
    row,
    regime
):

    p = raw_probability(
        row,
        regime
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

    trend_score = np.clip(
        50
        +
        1000
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
        1000 * row["mom_score"],
        0,
        100
    )

    volume_score = np.clip(
        50
        +
        35
        * (
            row["vol_ratio"]
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

    regime_score = {
        "FAVORABLE": 100,
        "MIXED": 60,
        "UNFAVORABLE": 30,
        "UNKNOWN": 40,
    }.get(
        regime,
        40
    )

    rr_score = np.clip(
        50
        +
        25 * (
            rr2 - 1
        ),
        0,
        100
    )

    probability_score = (
        50
        +
        400
        * (
            p - 0.50
        )
    )

    expected_score = np.clip(
        50
        +
        500
        * er3
        +
        250
        * er5,
        0,
        100
    )

    composite = (

        0.34 * probability_score

        +

        0.20 * expected_score

        +

        0.15 * trend_score

        +

        0.10 * momentum_score

        +

        0.05 * volume_score

        +

        0.06 * rsi_score

        +

        0.07 * rr_score

        +

        0.03 * regime_score
    )

    # --------------------------------------------------------
    # Quality / safety deductions
    # --------------------------------------------------------

    if row["vol_ratio"] < 0.60:

        composite -= 10

    if row["atr_pct"] > 0.06:

        composite -= 10

    if rsi > 75:

        composite -= 8

    if rr1 < 0.80:

        composite -= 8

    if rr2 < 1.20:

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
    # Ranking first, hard minimums second.
    # --------------------------------------------------------

    if (
        composite >= 75
        and
        p >= 0.60
        and
        er3 >= 0.0025
        and
        er5 >= 0.0040
        and
        rr1 >= 1.00
        and
        rr2 >= 1.40
        and
        row["vol_ratio"] >= 0.85
        and
        45 <= rsi <= 68
        and
        row["trend20"] >= 0
        and
        row["trend50"] >= -0.005
        and
        regime != "UNFAVORABLE"
    ):

        action = "TRADE"

    elif (
        composite >= 62
        and
        p >= 0.55
        and
        er3 >= 0.0005
        and
        er5 >= 0.0010
        and
        rr2 >= 1.20
        and
        row["vol_ratio"] >= 0.70
        and
        rsi <= 70
        and
        row["trend20"] >= -0.005
    ):

        action = "WATCH"

    else:

        action = "WAIT"

    failures = []

    if p < 0.60:
        failures.append("probability")

    if er3 < 0.0025:
        failures.append("er3")

    if er5 < 0.0040:
        failures.append("er5")

    if rr1 < 1.00:
        failures.append("rr1")

    if rr2 < 1.40:
        failures.append("rr2")

    if row["vol_ratio"] < 0.85:
        failures.append("volume")

    if rsi > 68:
        failures.append("rsi")

    if row["trend20"] < 0:
        failures.append("trend")

    if row["trend50"] < -0.005:
        failures.append("trend50")

    if regime == "UNFAVORABLE":
        failures.append("regime")

    return {

        "raw_probability": p,

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

        "action": action,

        "rsi": rsi,

        "volume": float(
            row["vol_ratio"]
        ),

        "failures": failures,
    }


# ============================================================
# POSITION SIZE
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

    shares_risk = math.floor(
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

    shares_capital = math.floor(
        capital_limit
        /
        price
    )

    return max(
        0,
        min(
            shares_risk,
            shares_capital
        )
    )


# ============================================================
# DATA
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
            f"{ticker}: data unavailable: "
            f"{error}"
        )

        return pd.DataFrame()


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    message
):

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
        "https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
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
    # INDEX
    # --------------------------------------------------------

    nifty = download_data(
        "^NSEI"
    )

    (
        regime,
        nifty_price,
        sma50,
        regime_reason
    ) = determine_market_regime(
        nifty
    )

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

        needed = [
            "atr",
            "rsi",
            "sma50",
            "vol_ratio",
            "trend20",
            "trend50",
            "trend200"
        ]

        if any(
            pd.isna(
                row[c]
            )
            for c in needed
        ):

            continue

        result = evaluate_candidate(
            row,
            regime
        )

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
        x for x in candidates
        if x["action"] == "TRADE"
    ][:TOP_N]

    watches = [
        x for x in candidates
        if x["action"] == "WATCH"
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

    lines.append(
        f"NIFTY: "
        f"₹{nifty_price:,.2f}"
        if pd.notna(nifty_price)
        else
        "NIFTY: N/A"
    )

    lines.append(
        f"SMA50: "
        f"₹{sma50:,.2f}"
        if pd.notna(sma50)
        else
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

    if market_open and trades:

        for i, item in enumerate(
            trades,
            start=1
        ):

            lines.append("")
            lines.append(
                f"{i}. "
                f"{item['ticker']} — TRADE"
            )

            lines.append(
                f"Price: "
                f"₹{item['price']:,.2f}"
            )

            lines.append(
                f"Model P(UP): "
                f"{item['raw_probability']*100:.1f}%"
            )

            lines.append(
                f"Expected return "
                f"3D / 5D: "
                f"{item['er3']*100:.2f}% / "
                f"{item['er5']*100:.2f}%"
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
                "Action: TRADE"
            )

    else:

        lines.append(
            "NO VALID LONG TRADE TODAY"
        )

        lines.append(
            "No candidate currently "
            "meets the V6.3.13 composite "
            "quality, probability, "
            "expectancy and risk controls."
        )

    # ========================================================
    # WATCHLIST
    # ========================================================

    lines.append("")
    lines.append(
        "--- BEST WATCHLIST SETUPS ---"
    )

    if watches:

        for i, item in enumerate(
            watches,
            start=1
        ):

            lines.append("")
            lines.append(
                f"{i}. "
                f"{item['ticker']} — WATCH"
            )

            lines.append(
                f"Price: "
                f"₹{item['price']:,.2f}"
            )

            lines.append(
                f"P(UP): "
                f"{item['raw_probability']*100:.1f}%"
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
    # MODEL VALIDATION
    # ========================================================

    lines.append("")
    lines.append(
        "--- MODEL VALIDATION STATUS ---"
    )

    lines.append(
        "V6.3.13 uses strict composite ranking, "
        "walk-forward validation and "
        "risk-aware position sizing."
    )

    lines.append(
        "P(UP) is an empirical model "
        "estimate, not a guaranteed "
        "probability of profit."
    )

    lines.append(
        "Backtesting must be completed "
        "before treating a setup as "
        "validated."
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
    # AUDIT
    # ========================================================

    timestamp = now.strftime(
        "%Y%m%d_%H%M%S"
    )

    audit_file = (
        Path("audit")
        /
        f"market_alert_v6_3_11_"
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
