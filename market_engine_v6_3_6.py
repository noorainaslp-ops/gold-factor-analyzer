"""
MARKET ALERT V6.3.6
CALIBRATED SHORT-TERM MARKET SCREENER

Purpose
-------
Generate short-term Indian equity opportunities for approximately
1-5 trading sessions.

V6.3.6 features
---------------
- Correct market-regime handling.
- MIXED regime is not automatically rejected.
- Separates raw probability from calibrated probability.
- Empirical analogue model.
- Expected return is used as an important trade criterion.
- Risk/reward calculated from valid positive stop/target levels.
- Never produces a negative stop loss.
- TRADE / WATCH / REJECT classification.
- Position sizing based on configured capital and risk.
- Weekend/non-trading-day handling.
- Conservative IPO retrieval.
- Telegram alert generation.
- Audit CSV generation.

IMPORTANT
---------
This is a probabilistic research screen.
It does not guarantee profit.
P(UP) is an empirical/calibrated estimate,
not a guaranteed probability of profit.
"""

from __future__ import annotations

import os
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import requests


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "V6.3.6"

CAPITAL = float(
    os.getenv(
        "TRADING_CAPITAL",
        "100000"
    )
)

MAX_RISK_PCT = float(
    os.getenv(
        "MAX_RISK_PCT",
        "0.01"
    )
)

ROUND_TRIP_COST = 0.0015

N_ANALOGUES = 30
MIN_ANALOGUES = 15
LOOKBACK_DAYS = 756

MIN_CALIBRATED_P3 = 0.52
MIN_CALIBRATED_P5 = 0.51

MIN_NET_ER3 = 0.0010
MIN_NET_ER5 = 0.0015

MIN_RR1 = 1.00
MIN_RR2 = 1.20

MAX_RSI = 75.0
MIN_VOLUME_RATIO = 0.80

MAX_TOP_PICKS = 5

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(
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
# HELPERS
# ============================================================

def ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def safe_float(
    value,
    default=np.nan
):
    try:
        x = float(value)

        if np.isfinite(x):
            return x

    except Exception:
        pass

    return default


def clean_date(value):

    ts = pd.Timestamp(value)

    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts.normalize()


def is_trading_day():

    now = datetime.now()

    return now.weekday() < 5


def clean_frame(df):

    if df is None or df.empty:
        return None

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):
        return None

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for col in required:

        if col not in df.columns:
            df[col] = np.nan

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

        cleaned = clean_frame(
            raw
        )

        if cleaned is not None:
            result["SINGLE"] = cleaned

        return result

    levels0 = list(
        raw.columns
        .get_level_values(0)
        .unique()
    )

    levels1 = list(
        raw.columns
        .get_level_values(1)
        .unique()
    )

    if "Close" in levels0:

        for name in levels1:

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
                    result[
                        str(name)
                    ] = cleaned

            except Exception:
                continue

    else:

        for name in levels0:

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
                    result[
                        str(name)
                    ] = cleaned

            except Exception:
                continue

    return result


# ============================================================
# TECHNICAL INDICATORS
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

    result = (
        100
        -
        100 / (1 + rs)
    )

    result = result.where(
        avg_loss != 0,
        100.0
    )

    return result


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
                - previous_close
            ).abs(),
            (
                low
                - previous_close
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
            "regime": "UNKNOWN",
            "nifty": np.nan,
            "sma50": np.nan,
            "vix": np.nan,
        }

    n = nifty.loc[
        nifty.index <= signal_date
    ]

    if len(n) < 60:

        return {
            "regime": "UNKNOWN",
            "nifty": np.nan,
            "sma50": np.nan,
            "vix": np.nan,
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

    sma_current = safe_float(
        sma50.iloc[-1]
    )

    sma_previous = safe_float(
        sma50.iloc[-6]
    )

    above = (
        np.isfinite(current)
        and np.isfinite(sma_current)
        and current >= sma_current
    )

    rising = (
        np.isfinite(sma_current)
        and np.isfinite(sma_previous)
        and sma_current > sma_previous
    )

    if above and rising:

        regime = "FAVORABLE"

    elif (
        not above
        and not rising
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
        "regime": regime,
        "nifty": current,
        "sma50": sma_current,
        "vix": vix_value,
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
        + ["Close"]
    )

    data = data.dropna(
        subset=required
    )

    if len(data) < 300:
        return None

    candidate = data.iloc[-1]

    train = data.iloc[:-6].copy()

    if len(train) < 100:
        return None

    train = train.tail(
        LOOKBACK_DAYS
    ).copy()

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

    distances = np.zeros(
        len(train),
        dtype=float
    )

    for col in FEATURES:

        series = train[col]

        median = (
            series
            .median()
        )

        mad = (
            series
            - median
        ).abs().median()

        if (
            np.isfinite(mad)
            and mad > 0
        ):

            scale = (
                1.4826 * mad
            )

        else:

            scale = (
                series.std()
            )

        if (
            not np.isfinite(scale)
            or scale <= 0
        ):

            scale = 1.0

        candidate_value = (
            safe_float(
                candidate[col],
                median
            )
        )

        distances += (
            (
                (
                    series.values
                    - candidate_value
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
        1
        /
        (
            train["distance"]
            + 0.10
        )
    )

    weights = (
        weights
        /
        weights.sum()
    )

    raw_p3 = float(
        (
            weights
            *
            (
                train["future3"]
                > 0
            )
        ).sum()
    )

    raw_p5 = float(
        (
            weights
            *
            (
                train["future5"]
                > 0
            )
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

    std3 = safe_float(
        train["future3"].std(),
        0.05
    )

    std5 = safe_float(
        train["future5"].std(),
        0.07
    )

    shrinkage = min(
        0.45,
        12.0 / max(
            12.0,
            len(train)
        )
    )

    calibrated_p3 = (
        (
            1 - shrinkage
        )
        * raw_p3
        +
        shrinkage
        * 0.50
    )

    calibrated_p5 = (
        (
            1 - shrinkage
        )
        * raw_p5
        +
        shrinkage
        * 0.50
    )

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
        0.40 * sample_quality
        +
        0.30 * (
            1
            - dispersion_penalty3
        )
        +
        0.30 * (
            1
            - dispersion_penalty5
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
# RISK / TARGET LEVELS
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
        or price <= 0
        or not np.isfinite(atr_pct)
        or atr_pct <= 0
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
            - stop_distance
        )
    )

    stop = max(
        0.01,
        stop
    )

    risk_per_share = (
        price
        - stop
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
            + target1_return
        )
    )

    target2 = (
        price
        *
        (
            1
            + target2_return
        )
    )

    rr1 = (
        target1 - price
    ) / risk_per_share

    rr2 = (
        target2 - price
    ) / risk_per_share

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
# POSITION SIZE
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
        or not np.isfinite(stop)
        or price <= stop
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
        - stop
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
        * price
    )

    planned_loss = (
        shares
        * risk_per_share
    )

    return (
        shares,
        exposure,
        planned_loss
    )


# ============================================================
# SCORING
# ============================================================

def calculate_score(
    row,
    regime
):

    p3 = row["p3"]
    p5 = row["p5"]

    er3 = row["er3"]
    er5 = row["er5"]

    rsi_value = row["rsi"]
    volume = row["volume_ratio"]
    trend = row["trend"]

    score = 0.0

    score += (
        p3 - 0.50
    ) * 3.0

    score += (
        p5 - 0.50
    ) * 3.0

    score += (
        er3
        * 20.0
    )

    score += (
        er5
        * 15.0
    )

    if trend > 0:
        score += 0.20

    if volume >= 1.20:
        score += 0.15

    elif volume >= 1.0:
        score += 0.05

    if (
        50 <= rsi_value <= 68
    ):
        score += 0.10

    elif rsi_value > 72:
        score -= 0.15

    if regime == "FAVORABLE":
        score += 0.15

    elif regime == "MIXED":
        score -= 0.05

    elif regime == "UNFAVORABLE":
        score -= 0.50

    return score


# ============================================================
# CANDIDATE EVALUATION
# ============================================================

def evaluate_candidate(
    symbol,
    row,
    regime
):

    price = safe_float(
        row["Close"]
    )

    rsi_value = safe_float(
        row["rsi"]
    )

    volume = safe_float(
        row["volume_ratio"]
    )

    trend = safe_float(
        row["dist_sma50"]
    )

    p3 = safe_float(
        row["p3"]
    )

    p5 = safe_float(
        row["p5"]
    )

    er3 = safe_float(
        row["er3"]
    )

    er5 = safe_float(
        row["er5"]
    )

    quality = safe_float(
        row["quality"]
    )

    levels = calculate_trade_levels(

        price,

        row["atr_pct"],

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

        "calibrated_p3":
            p3 >= MIN_CALIBRATED_P3,

        "calibrated_p5":
            p5 >= MIN_CALIBRATED_P5,

        "positive_er3":
            er3 >= MIN_NET_ER3,

        "positive_er5":
            er5 >= MIN_NET_ER5,

        "rr1":
            levels["rr1"] >= MIN_RR1,

        "rr2":
            levels["rr2"] >= MIN_RR2,

        "quality":
            quality >= 0.25,

        "trend":
            trend > -0.03,

        "rsi":
            rsi_value <= MAX_RSI,

        "volume":
            volume >= MIN_VOLUME_RATIO,

        "regime":
            regime != "UNFAVORABLE",

    }

    failed = [
        key
        for key, value
        in checks.items()
        if not bool(value)
    ]

    all_pass = all(
        bool(x)
        for x in checks.values()
    )

    watch_score = (
        sum(
            bool(x)
            for x in checks.values()
        )
        /
        len(checks)
    )

    if all_pass:

        action = "TRADE"

    elif watch_score >= 0.60:

        action = "WATCH"

    else:

        action = "REJECT"

    return {

        "symbol":
            symbol,

        "price":
            price,

        "raw_p3":
            row["raw_p3"],

        "raw_p5":
            row["raw_p5"],

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
            failed,

        "watch_score":
            watch_score,

    }


# ============================================================
# IPO INFORMATION
# ============================================================

def get_ipo_information():

    url = (
        "https://www.nseindia.com/api/ipo-current-issue"
    )

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120 Safari/537.36",

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

        return results[:10]

    except Exception:

        return []


# ============================================================
# TELEGRAM SEND
# ============================================================

def telegram_send(
    message
):

    bot_token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if (
        not bot_token
        or not chat_id
    ):

        print(
            "Telegram credentials "
            "not configured."
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
                "Telegram HTTP error:",
                response.status_code,
                response.text
            )

        return (
            response.status_code == 200
        )

    except Exception as exc:

        print(
            f"Telegram error: {exc}"
        )

        return False


# ============================================================
# MONEY FORMAT
# ============================================================

def format_money(
    value
):

    value = safe_float(
        value
    )

    if not np.isfinite(value):
        return "N/A"

    return (
        f"₹{value:,.2f}"
    )


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
            f"{format_money(market['nifty'])} "
            f"| SMA50: "
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
        x
        for x in candidates
        if x["action"] == "TRADE"
    ]

    watches = [
        x
        for x in candidates
        if x["action"] == "WATCH"
    ]

    if not trading_day:

        lines += [

            "",

            "MARKET IS CLOSED.",

            (
                "No new long position should "
                "be initiated today."
            ),

        ]

    if trades:

        for i, x in enumerate(
            trades[:MAX_TOP_PICKS],
            start=1
        ):

            lines += [

                "",

                (
                    f"{i}. "
                    f"{x['symbol']} "
                    f"— TRADE"
                ),

                (
                    f"Price: "
                    f"{format_money(x['price'])}"
                ),

                # =================================================
                # IMPORTANT CORRECTION:
                #
                # The model calculates 3D and 5D probabilities.
                # It does NOT calculate a true 1D probability.
                #
                # Therefore DO NOT display:
                # P(UP) 1D / 3D / 5D
                #
                # =================================================

                (
                    "Calibrated P(UP) "
                    f"3D / 5D: "
                    f"{x['p3']:.1%} / "
                    f"{x['p5']:.1%}"
                ),

                (
                    "Raw analogue P(UP) "
                    f"3D / 5D: "
                    f"{x['raw_p3']:.1%} / "
                    f"{x['raw_p5']:.1%}"
                ),

                (
                    "Expected return "
                    f"3D / 5D: "
                    f"{x['er3']:.2%} / "
                    f"{x['er5']:.2%}"
                ),

                (
                    f"Score: "
                    f"{x['score']:.3f} "
                    f"| RSI: "
                    f"{x['rsi']:.1f} "
                    f"| Volume: "
                    f"{x['volume_ratio']:.2f}x"
                ),

                (
                    f"Entry: "
                    f"{format_money(x['price'])}"
                ),

                (
                    f"Stop Loss: "
                    f"{format_money(x['stop'])}"
                ),

                (
                    f"Target 1: "
                    f"{format_money(x['target1'])}"
                ),

                (
                    f"Target 2: "
                    f"{format_money(x['target2'])}"
                ),

                (
                    f"Risk/Reward: "
                    f"{x['rr1']:.2f} / "
                    f"{x['rr2']:.2f}"
                ),

                "Expected holding: 1–5 sessions",

                (
                    f"Suggested position: "
                    f"{x['shares']} shares "
                    f"≈ "
                    f"{format_money(x['exposure'])}"
                ),

                (
                    f"Maximum planned loss: "
                    f"{format_money(x['planned_loss'])}"
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

    lines += [

        "",

        "--- BEST WATCHLIST SETUPS ---",

    ]

    if watches:

        sorted_watches = sorted(
            watches,
            key=lambda z:
                z["score"],
            reverse=True
        )

        for i, x in enumerate(
            sorted_watches[:3],
            start=1
        ):

            lines += [

                "",

                (
                    f"{i}. "
                    f"{x['symbol']} "
                    f"— WATCH"
                ),

                (
                    f"Price: "
                    f"{format_money(x['price'])}"
                ),

                (
                    "Calibrated P(UP) "
                    f"3D / 5D: "
                    f"{x['p3']:.1%} / "
                    f"{x['p5']:.1%}"
                ),

                (
                    "Expected return "
                    f"3D / 5D: "
                    f"{x['er3']:.2%} / "
                    f"{x['er5']:.2%}"
                ),

                (
                    f"RR1: "
                    f"{x['rr1']:.2f} "
                    f"| RR2: "
                    f"{x['rr2']:.2f}"
                ),

                (
                    f"RSI: "
                    f"{x['rsi']:.1f} "
                    f"| Volume: "
                    f"{x['volume_ratio']:.2f}x"
                ),

                (
                    f"Score: "
                    f"{x['score']:.3f}"
                ),

                "Action: WATCH / WAIT",

            ]

    else:

        lines.append(
            "None."
        )

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
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        f"MARKET ALERT {VERSION}"
    )

    print("=" * 70)

    signal_date = clean_date(
        datetime.now()
    )

    trading_day = is_trading_day()

    tickers = [
        ticker(s)
        for s in SYMBOLS
    ]

    tickers += [
        "^NSEI",
        "^INDIAVIX",
    ]

    print(
        f"Downloading "
        f"{len(tickers)} tickers..."
    )

    raw = yf.download(

        tickers=tickers,

        period="3y",

        interval="1d",

        auto_adjust=True,

        group_by="column",

        progress=False,

        threads=True,

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

    candidates = []

    for symbol, features in (
        feature_frames.items()
    ):

        hist = features.loc[
            features.index <= signal_date
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

    # ========================================================
    # IPO
    # ========================================================

    ipo_names = (
        get_ipo_information()
    )

    # ========================================================
    # BUILD ALERT
    # ========================================================

    message = build_alert(

        market,

        candidates,

        ipo_names,

        trading_day

    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("")
    print(message)
    print("")

    # ========================================================
    # TELEGRAM
    # ========================================================

    telegram_send(
        message
    )


if __name__ == "__main__":

    main()
