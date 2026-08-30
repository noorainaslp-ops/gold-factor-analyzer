import os
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ============================================================
# MARKET ALERT ENGINE V6.3.2
# ============================================================
#
# Short-term Indian equity research screen.
#
# Horizons:
#   1 trading session
#   3 trading sessions
#   5 trading sessions
#
# Features:
#   - Historical analogue model
#   - 1D / 3D / 5D probability estimates
#   - 1D / 3D / 5D expected returns
#   - Market regime detection
#   - RSI
#   - Moving averages
#   - Volume confirmation
#   - ATR volatility
#   - Risk/reward
#   - Position sizing
#   - TRADE / WATCH / REJECT
#   - Telegram alerts
#   - IPO retrieval
#   - Audit TXT + CSV
#
# IMPORTANT:
# This is a probabilistic research model.
# It does NOT guarantee profit.
# ============================================================


# ============================================================
# TIMEZONE
# ============================================================

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# SAFE ENVIRONMENT VARIABLE READER
# ============================================================

def get_float_env(name, default):
    """
    Safely read a floating-point environment variable.

    Handles:
      - missing variable
      - blank variable
      - invalid value
      - NaN
      - infinity

    In all such cases, the configured default is used.
    """

    value = os.getenv(name, "")

    if value is None:
        return float(default)

    value = str(value).strip()

    if value == "":
        return float(default)

    try:

        number = float(value)

        if not math.isfinite(number):
            return float(default)

        return number

    except (TypeError, ValueError):

        return float(default)


# ============================================================
# CONFIGURATION
# ============================================================

CAPITAL = get_float_env(
    "ALERT_CAPITAL",
    100000
)

RISK_PCT = get_float_env(
    "ALERT_RISK_PCT",
    0.01
)

MAX_POSITION_PCT = get_float_env(
    "ALERT_MAX_POSITION_PCT",
    0.20
)

MAX_STOCKS = int(
    get_float_env(
        "ALERT_MAX_STOCKS",
        70
    )
)


# ============================================================
# STOCK UNIVERSE
# ============================================================

SYMBOLS = [

    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "BHARTIARTL",
    "TCS",
    "INFY",
    "ITC",
    "SBIN",
    "LT",
    "AXISBANK",
    "KOTAKBANK",
    "M&M",
    "HINDUNILVR",
    "BAJFINANCE",
    "MARUTI",
    "SUNPHARMA",
    "HCLTECH",
    "NTPC",
    "TITAN",
    "ADANIENT",
    "ULTRACEMCO",
    "ONGC",
    "POWERGRID",
    "TATASTEEL",
    "JSWSTEEL",
    "COALINDIA",
    "ADANIPORTS",
    "BAJAJFINSV",
    "WIPRO",
    "NESTLEIND",
    "TECHM",
    "ASIANPAINT",
    "BEL",
    "TRENT",
    "GRASIM",
    "HINDALCO",
    "EICHERMOT",
    "SHRIRAMFIN",
    "VEDL",
    "TATAMOTORS",
    "CIPLA",
    "DRREDDY",
    "DIVISLAB",
    "APOLLOHOSP",
    "BRITANNIA",
    "HEROMOTOCO",
    "BAJAJ-AUTO",
    "INDUSINDBK",
    "SBILIFE",
    "HDFCLIFE",
    "TATAELXSI",
    "AUROPHARMA",
    "BOSCHLTD",
    "DLF",
    "NAUKRI",
    "SAIL",
    "ABB",
    "GODREJCP",
    "PIDILITIND",
    "SIEMENS",
    "AMBUJACEM",
    "ACC",
    "BPCL",
    "IOC",
    "HAVELLS",

]

TICKERS = [
    symbol + ".NS"
    for symbol in SYMBOLS
]

NIFTY = "^NSEI"
VIX = "^INDIAVIX"

AUDIT_DIR = Path("audit")

AUDIT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def now_ist():

    return datetime.now(IST)


def finite(value):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

        return np.nan

    except Exception:

        return np.nan


def money(value):

    value = finite(value)

    if not np.isfinite(value):

        return "—"

    return f"₹{value:,.2f}"


# ============================================================
# RSI
# ============================================================

def rsi(
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
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    result = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    return result.fillna(50)


# ============================================================
# ATR
# ============================================================

def atr(
    df,
    period=14
):

    previous_close = (
        df["Close"].shift(1)
    )

    true_range = pd.concat(
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
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# YFINANCE MULTI-TICKER DATA
# ============================================================

def split_yf_download(raw):

    if raw is None:

        return {}

    if raw.empty:

        return {}

    if not isinstance(
        raw.columns,
        pd.MultiIndex
    ):

        return {
            "SINGLE": raw
        }

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

        return {

            ticker: raw.xs(
                ticker,
                axis=1,
                level=1,
                drop_level=True,
            )

            for ticker in level1

        }

    return {

        ticker: raw.xs(
            ticker,
            axis=1,
            level=0,
            drop_level=True,
        )

        for ticker in level0

    }


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

def download_history():

    all_tickers = (
        TICKERS
        + [
            NIFTY,
            VIX,
        ]
    )

    print(
        f"Downloading data for "
        f"{len(all_tickers)} symbols..."
    )

    raw = yf.download(

        all_tickers,

        period="2y",

        interval="1d",

        auto_adjust=True,

        group_by="column",

        progress=False,

        threads=True,

    )

    return split_yf_download(
        raw
    )


# ============================================================
# CLEAN INDIVIDUAL FRAME
# ============================================================

def get_frame(
    frames,
    ticker
):

    df = frames.get(
        ticker
    )

    if df is None:

        return None

    if df.empty:

        return None

    if "Close" not in df.columns:

        return None

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

    result = df[
        required
    ].copy()

    result = result.dropna(
        subset=["Close"]
    )

    return result


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features(df):

    result = df.copy()

    close = result[
        "Close"
    ]

    result["rsi"] = rsi(
        close
    )

    result["sma20"] = (
        close.rolling(20)
        .mean()
    )

    result["sma50"] = (
        close.rolling(50)
        .mean()
    )

    result["sma200"] = (
        close.rolling(200)
        .mean()
    )

    result["ema10"] = (
        close.ewm(
            span=10,
            adjust=False,
        ).mean()
    )

    result["ret1"] = (
        close.pct_change(1)
    )

    result["ret3"] = (
        close.pct_change(3)
    )

    result["ret5"] = (
        close.pct_change(5)
    )

    result["ret20"] = (
        close.pct_change(20)
    )

    result["vol20"] = (
        close.pct_change()
        .rolling(20)
        .std()
        * math.sqrt(252)
    )

    volume_average = (
        result["Volume"]
        .rolling(20)
        .mean()
    )

    result["vol_ratio"] = (
        result["Volume"]
        / volume_average.replace(
            0,
            np.nan
        )
    )

    result["atr"] = atr(
        result
    )

    result["atr_pct"] = (
        result["atr"]
        / close
    )

    result["dist_sma50"] = (
        close
        / result["sma50"]
        - 1
    )

    result["dist_ema10"] = (
        close
        / result["ema10"]
        - 1
    )

    # Forward returns.
    #
    # These are only used on historical observations
    # when building the analogue model.

    for horizon in (
        1,
        3,
        5
    ):

        result[
            f"fwd{horizon}"
        ] = (
            close.shift(
                -horizon
            )
            / close
            - 1
        )

    return result


# ============================================================
# HISTORICAL ANALOGUE MODEL
# ============================================================

def empirical_analog_model(
    features
):

    model_columns = [

        "rsi",
        "ret3",
        "ret5",
        "dist_sma50",
        "dist_ema10",
        "vol_ratio",
        "atr_pct",

    ]

    if len(features) < 120:

        return None

    # Exclude the most recent five observations.
    #
    # This prevents the current decision area from
    # contaminating the historical forward-return sample.

    historical = (
        features.iloc[:-5]
        .dropna(
            subset=(
                model_columns
                + [
                    "fwd1",
                    "fwd3",
                    "fwd5",
                ]
            )
        )
        .copy()
    )

    if len(historical) < 80:

        return None

    historical_vectors = []

    current_vector = []

    for column in model_columns:

        median = (
            historical[column]
            .median()
        )

        mad = (
            historical[column]
            - median
        ).abs().median()

        scale = (
            1.4826
            * mad
        )

        if (
            not np.isfinite(scale)
            or scale < 1e-8
        ):

            scale = (
                historical[column]
                .std(
                    ddof=0
                )
            )

        if (
            not np.isfinite(scale)
            or scale < 1e-8
        ):

            scale = 1.0

        historical_vectors.append(

            (
                (
                    historical[column]
                    - median
                )
                / scale
            ).to_numpy()

        )

        current_value = finite(
            features[column].iloc[-1]
        )

        if not np.isfinite(
            current_value
        ):

            return None

        current_vector.append(

            (
                current_value
                - median
            )
            / scale

        )

    matrix = np.column_stack(
        historical_vectors
    )

    query = np.asarray(
        current_vector,
        dtype=float
    )

    distances = np.sqrt(

        (
            matrix - query
        ) ** 2

    ).mean(axis=1)

    distances = np.sqrt(
        distances
    )

    k = min(

        40,

        max(
            20,
            len(historical)
            // 20
        ),

    )

    nearest_indices = (
        np.argsort(
            distances
        )[:k]
    )

    neighbours = (
        historical.iloc[
            nearest_indices
        ]
    )

    weights = (
        1
        / (
            distances[
                nearest_indices
            ]
            + 0.35
        )
    )

    weights = (
        weights
        / weights.sum()
    )

    result = {}

    for horizon in (
        1,
        3,
        5
    ):

        returns = (
            neighbours[
                f"fwd{horizon}"
            ]
            .to_numpy(
                dtype=float
            )
        )

        result[
            f"p{horizon}"
        ] = float(

            np.sum(
                weights
                * (
                    returns
                    > 0
                )
            )

        )

        result[
            f"er{horizon}"
        ] = float(

            np.sum(
                weights
                * returns
            )

        )

        result[
            f"median{horizon}"
        ] = float(
            np.median(
                returns
            )
        )

    result["analogs"] = int(
        k
    )

    result["quality"] = float(

        np.clip(

            1
            - (
                np.mean(
                    distances[
                        nearest_indices
                    ]
                )
                / 4
            ),

            0,
            1,

        )

    )

    return result


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime(
    frames
):

    nifty_df = get_frame(
        frames,
        NIFTY
    )

    vix_df = get_frame(
        frames,
        VIX
    )

    if nifty_df is None:

        return {

            "label": "UNKNOWN",
            "nifty": np.nan,
            "sma50": np.nan,
            "vix": np.nan,

        }

    nifty_features = (
        make_features(
            nifty_df
        )
    )

    nifty = finite(
        nifty_features[
            "Close"
        ].iloc[-1]
    )

    sma50 = finite(
        nifty_features[
            "sma50"
        ].iloc[-1]
    )

    if len(
        nifty_features
    ) >= 6:

        previous_sma50 = finite(

            nifty_features[
                "sma50"
            ].iloc[-6]

        )

    else:

        previous_sma50 = np.nan

    if vix_df is not None:

        vix = finite(
            vix_df[
                "Close"
            ].iloc[-1]
        )

    else:

        vix = np.nan

    above_sma = (

        np.isfinite(nifty)

        and np.isfinite(sma50)

        and nifty > sma50

    )

    rising_sma = (

        np.isfinite(sma50)

        and np.isfinite(
            previous_sma50
        )

        and sma50
        > previous_sma50

    )

    if (
        above_sma
        and rising_sma
    ):

        label = "FAVORABLE"

    elif (
        not above_sma
        and not rising_sma
    ):

        label = "UNFAVORABLE"

    else:

        label = "MIXED"

    return {

        "label": label,

        "nifty": nifty,

        "sma50": sma50,

        "vix": vix,

    }


# ============================================================
# CANDIDATE SCORE
# ============================================================

def score_candidate(
    row,
    regime_label
):

    score = (

        0.30
        * (
            row["p3"]
            - 0.50
        )

        + 0.25
        * (
            row["p5"]
            - 0.50
        )

        + 0.20
        * np.tanh(
            row["er3"]
            / 0.01
        )

        + 0.15
        * np.tanh(
            row["er5"]
            / 0.02
        )

        + 0.10
        * np.tanh(
            row["trend"]
            / 0.03
        )

    )

    if (
        regime_label
        == "FAVORABLE"
    ):

        score += 0.02

    elif (
        regime_label
        == "UNFAVORABLE"
    ):

        score -= 0.04

    # Avoid chasing extremely overbought stocks.
    if row["rsi"] > 75:

        score -= 0.08

    # Very weak volume confirmation.
    if row["vol_ratio"] < 0.60:

        score -= 0.04

    return float(
        score
    )


# ============================================================
# BUILD CANDIDATES
# ============================================================

def build_candidates(
    frames,
    regime
):

    rows = []

    for symbol in SYMBOLS[
        :MAX_STOCKS
    ]:

        ticker = (
            symbol
            + ".NS"
        )

        df = get_frame(
            frames,
            ticker
        )

        if df is None:

            continue

        if len(df) < 230:

            continue

        features = (
            make_features(
                df
            )
        )

        features = (
            features.dropna(
                subset=[

                    "rsi",
                    "sma50",
                    "ema10",
                    "atr_pct",
                    "ret3",
                    "ret5",

                ]
            )
        )

        if len(features) < 220:

            continue

        model = (
            empirical_analog_model(
                features
            )
        )

        if model is None:

            continue

        last = features.iloc[-1]

        price = finite(
            last["Close"]
        )

        atr_pct = finite(
            last["atr_pct"]
        )

        if (
            not np.isfinite(
                price
            )
            or
            not np.isfinite(
                atr_pct
            )
            or
            atr_pct <= 0
        ):

            continue

        trend = finite(
            last[
                "dist_sma50"
            ]
        )

        row = {

            "symbol": symbol,

            "price": price,

            "rsi": finite(
                last["rsi"]
            ),

            "vol_ratio": finite(
                last["vol_ratio"]
            ),

            "atr_pct": atr_pct,

            "trend": trend,

            **model,

        }

        row["score"] = (
            score_candidate(
                row,
                regime["label"]
            )
        )

        # ----------------------------------------------------
        # STOP LOSS
        # ----------------------------------------------------

        stop_distance = max(

            0.012,

            min(
                0.05,
                1.5
                * atr_pct
            ),

        )

        stop = (
            price
            * (
                1
                - stop_distance
            )
        )

        # ----------------------------------------------------
        # TARGET 1
        # ----------------------------------------------------

        expected_3 = max(

            row["er3"],

            row["median3"],

            0.006,

        )

        target1 = (
            price
            * (
                1
                + max(
                    0.006,
                    min(
                        0.06,
                        expected_3
                    )
                )
            )
        )

        # ----------------------------------------------------
        # TARGET 2
        # ----------------------------------------------------

        expected_5 = max(

            row["er5"],

            row["median5"],

            0.012,

        )

        target2 = (
            price
            * (
                1
                + max(
                    0.012,
                    min(
                        0.10,
                        expected_5
                    )
                )
            )
        )

        if target2 <= target1:

            target2 = (
                target1
                * 1.006
            )

        # ----------------------------------------------------
        # RISK / REWARD
        # ----------------------------------------------------

        risk_per_share = (
            price
            - stop
        )

        if risk_per_share <= 0:

            continue

        rr1 = (
            target1
            - price
        ) / risk_per_share

        rr2 = (
            target2
            - price
        ) / risk_per_share

        row.update({

            "stop": stop,

            "target1": target1,

            "target2": target2,

            "rr1": rr1,

            "rr2": rr2,

        })

        rows.append(
            row
        )

    if not rows:

        return pd.DataFrame()

    return pd.DataFrame(
        rows
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_candidates(
    df,
    regime
):

    if df.empty:

        return df

    result = df.copy()

    # --------------------------------------------------------
    # EXPECTED RETURN
    # --------------------------------------------------------

    result[
        "valid_return"
    ] = (

        (result["er3"] > 0.002)

        &

        (result["er5"] > 0.004)

    )

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    result[
        "valid_probability"
    ] = (

        (result["p3"] >= 0.54)

        &

        (result["p5"] >= 0.53)

    )

    # --------------------------------------------------------
    # RISK / REWARD
    # --------------------------------------------------------

    result[
        "valid_rr"
    ] = (

        (result["rr1"] >= 1.0)

        &

        (result["rr2"] >= 1.5)

    )

    # --------------------------------------------------------
    # ANALOGUE QUALITY
    # --------------------------------------------------------

    result[
        "valid_quality"
    ] = (
        result["quality"]
        >= 0.35
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    result[
        "valid_trend"
    ] = (
        result["trend"]
        > -0.02
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    result[
        "not_extreme"
    ] = (
        result["rsi"]
        < 76
    )

    # --------------------------------------------------------
    # TRADE
    # --------------------------------------------------------

    result[
        "trade"
    ] = (

        result[
            "valid_return"
        ]

        &

        result[
            "valid_probability"
        ]

        &

        result[
            "valid_rr"
        ]

        &

        result[
            "valid_quality"
        ]

        &

        result[
            "valid_trend"
        ]

        &

        result[
            "not_extreme"
        ]

    )

    # Don't issue long trades in an outright
    # unfavorable broad market regime.

    if (
        regime["label"]
        == "UNFAVORABLE"
    ):

        result[
            "trade"
        ] = False

    # --------------------------------------------------------
    # WATCH
    # --------------------------------------------------------

    result[
        "watch"
    ] = (

        ~result["trade"]

        &

        (result["er3"] > 0)

        &

        (result["p3"] >= 0.51)

        &

        (result["p5"] >= 0.50)

        &

        (result["quality"] >= 0.25)

        &

        (result["rr2"] >= 1.0)

    )

    result[
        "action"
    ] = np.where(

        result["trade"],

        "TRADE",

        np.where(

            result["watch"],

            "WATCH",

            "REJECT"

        )

    )

    return result.sort_values(

        [
            "trade",
            "watch",
            "score",
            "er3",
        ],

        ascending=[
            False,
            False,
            False,
            False,
        ],

    )


# ============================================================
# POSITION SIZING
# ============================================================

def position_size(
    row
):

    risk_budget = (
        CAPITAL
        * RISK_PCT
    )

    risk_per_share = max(

        row["price"]
        - row["stop"],

        row["price"]
        * 0.005,

    )

    shares_by_risk = math.floor(

        risk_budget
        / risk_per_share

    )

    shares_by_capital = math.floor(

        (
            CAPITAL
            * MAX_POSITION_PCT
        )
        / row["price"]

    )

    shares = max(

        0,

        min(

            shares_by_risk,

            shares_by_capital,

        )

    )

    value = (
        shares
        * row["price"]
    )

    maximum_loss = (
        shares
        * risk_per_share
    )

    return (
        shares,
        value,
        maximum_loss,
    )


# ============================================================
# IPO DATA
# ============================================================

def fetch_ipo_data():

    urls = [

        "https://www.nseindia.com/api/all-upcoming-issues",

        "https://www.nseindia.com/api/ipo-current-issue",

    ]

    headers = {

        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),

        "Accept": (
            "application/json,"
            "text/plain,*/*"
        ),

        "Referer": (
            "https://www.nseindia.com/"
        ),

    }

    session = (
        requests.Session()
    )

    try:

        session.get(

            "https://www.nseindia.com/",

            headers=headers,

            timeout=12,

        )

        records = []

        for url in urls:

            try:

                response = session.get(

                    url,

                    headers=headers,

                    timeout=12,

                )

            except Exception:

                continue

            if not response.ok:

                continue

            try:

                payload = (
                    response.json()
                )

            except Exception:

                continue

            if isinstance(
                payload,
                list
            ):

                records.extend(
                    payload
                )

            elif isinstance(
                payload,
                dict
            ):

                for key in [

                    "data",
                    "records",
                    "items",

                ]:

                    value = (
                        payload.get(
                            key
                        )
                    )

                    if isinstance(
                        value,
                        list
                    ):

                        records.extend(
                            value
                        )

        unique = []

        seen = set()

        for record in records:

            if not isinstance(
                record,
                dict
            ):

                continue

            key = str(
                sorted(
                    record.items()
                )
            )

            if key in seen:

                continue

            seen.add(
                key
            )

            unique.append(
                record
            )

        if not unique:

            return (
                [],
                "NSE returned no IPO records."
            )

        return (
            unique[:20],
            None,
        )

    except Exception as exc:

        return (
            [],
            str(exc),
        )


# ============================================================
# ALERT MESSAGE
# ============================================================

def alert_text(

    regime,

    df,

    ipo_records,

    ipo_error,

):

    current_time = (
        now_ist()
    )

    is_weekday = (
        current_time.weekday()
        < 5
    )

    lines = [

        "MULTI-FACTOR MARKET ALERT V6.3.2",

        current_time.strftime(
            "%d %b %Y, %H:%M IST"
        ),

        "",

        (
            "MARKET STATUS: "
            "TRADING DAY"
            if is_weekday

            else

            "MARKET STATUS: "
            "WEEKEND / NON-TRADING DAY"
        ),

        (
            f"MARKET REGIME: "
            f"{regime['label']}"
        ),

        (
            f"NIFTY: "
            f"{money(regime['nifty'])} | "
            f"SMA50: "
            f"{money(regime['sma50'])}"
        ),

        (
            f"INDIA VIX: "
            f"{money(regime['vix'])}"
        ),

        "",

        "--- TOP SHORT-TERM TRADE SETUPS (1–5 SESSIONS) ---",

    ]

    trades = (

        df[
            df["action"]
            == "TRADE"
        ].head(3)

        if not df.empty

        else

        pd.DataFrame()

    )

    watches = (

        df[
            df["action"]
            == "WATCH"
        ].head(5)

        if not df.empty

        else

        pd.DataFrame()

    )

    # ========================================================
    # TRADE SETUPS
    # ========================================================

    if trades.empty:

        lines.append(
            "NO VALID LONG TRADE TODAY"
        )

    else:

        for number, (
            _,
            row
        ) in enumerate(
            trades.iterrows(),
            1
        ):

            (
                shares,
                value,
                maximum_loss
            ) = position_size(
                row
            )

            lines.extend([

                "",

                (
                    f"{number}. "
                    f"{row['symbol']} "
                    f"— TRADE"
                ),

                (
                    f"Price: "
                    f"{money(row['price'])}"
                ),

                (
                    "P(UP) 1D / 3D / 5D: "
                    f"{row['p1'] * 100:.1f}% / "
                    f"{row['p3'] * 100:.1f}% / "
                    f"{row['p5'] * 100:.1f}%"
                ),

                (
                    "Expected return 1D / 3D / 5D: "
                    f"{row['er1'] * 100:.2f}% / "
                    f"{row['er3'] * 100:.2f}% / "
                    f"{row['er5'] * 100:.2f}%"
                ),

                (
                    f"Score: "
                    f"{row['score']:.3f} | "
                    f"RSI: "
                    f"{row['rsi']:.1f} | "
                    f"Volume: "
                    f"{row['vol_ratio']:.2f}x"
                ),

                (
                    f"Entry: "
                    f"{money(row['price'])}"
                ),

                (
                    f"Stop Loss: "
                    f"{money(row['stop'])}"
                ),

                (
                    f"Target 1: "
                    f"{money(row['target1'])} | "
                    f"Target 2: "
                    f"{money(row['target2'])}"
                ),

                (
                    f"Risk/Reward: "
                    f"{row['rr1']:.2f} / "
                    f"{row['rr2']:.2f}"
                ),

                "Expected holding: 1–5 sessions",

                (
                    f"Suggested position: "
                    f"{shares} shares "
                    f"≈ {money(value)}"
                ),

                (
                    f"Maximum planned loss: "
                    f"{money(maximum_loss)}"
                ),

            ])


    # ========================================================
    # WATCHLIST
    # ========================================================

    lines.extend([

        "",

        "--- BEST WATCHLIST SETUPS ---",

    ])

    if watches.empty:

        lines.append(
            "None."
        )

    else:

        for number, (
            _,
            row
        ) in enumerate(
            watches.iterrows(),
            1
        ):

            lines.append(

                (
                    f"{number}. "
                    f"{row['symbol']} | "
                    f"Price "
                    f"{money(row['price'])} | "
                    f"P3 "
                    f"{row['p3'] * 100:.1f}% | "
                    f"E3 "
                    f"{row['er3'] * 100:.2f}% | "
                    f"RR2 "
                    f"{row['rr2']:.2f} | "
                    f"RSI "
                    f"{row['rsi']:.1f}"
                )

            )


    # ========================================================
    # IPO
    # ========================================================

    lines.extend([

        "",

        "--- IPO OPEN / UPCOMING ---",

    ])

    if ipo_records:

        lines.append(

            "IPO records retrieved. "
            "Verify issue dates, price band "
            "and subscription status before applying."

        )

        for record in ipo_records[:8]:

            name = (

                record.get(
                    "companyName"
                )

                or

                record.get(
                    "issueName"
                )

                or

                record.get(
                    "symbol"
                )

                or

                record.get(
                    "name"
                )

                or

                "Unnamed issue"

            )

            lines.append(
                f"• {name}"
            )

    elif ipo_error:

        lines.extend([

            (
                "IPO DATA UNAVAILABLE | "
                "RETRIEVAL FAILED"
            ),

            (
                "Verify current/upcoming "
                "issues directly on NSE."
            ),

            (
                f"Retrieval note: "
                f"{ipo_error[:160]}"
            ),

        ])

    else:

        lines.append(

            "No current/upcoming IPO records "
            "were returned by the configured source."

        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    lines.extend([

        "",

        (
            "V6.3.2 is a probabilistic "
            "research screen; it does not "
            "guarantee profit."
        ),

        (
            "P(UP) is an empirical model "
            "estimate, not a guaranteed "
            "probability of profit."
        ),

        (
            "Verify live price, liquidity, "
            "corporate news, market status "
            "and order execution before trading."
        ),

    ])

    return "\n".join(
        lines
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    text
):

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        ""
    ).strip()

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID",
        ""
    ).strip()

    if not token:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
        )

    if not chat_id:

        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing."
        )

    url = (

        "https://api.telegram.org/"
        f"bot{token}/sendMessage"

    )

    response = requests.post(

        url,

        json={

            "chat_id": chat_id,

            "text": text,

        },

        timeout=20,

    )

    response.raise_for_status()


# ============================================================
# SAVE AUDIT
# ============================================================

def save_audit(

    alert,

    candidates,

):

    timestamp = (
        now_ist()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    alert_file = (

        AUDIT_DIR
        / (
            f"alert_"
            f"{timestamp}.txt"
        )

    )

    alert_file.write_text(

        alert,

        encoding="utf-8",

    )

    if not candidates.empty:

        candidates.to_csv(

            AUDIT_DIR
            / (
                f"candidates_"
                f"{timestamp}.csv"
            ),

            index=False,

        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=========================================="
    )

    print(
        "MARKET ALERT ENGINE V6.3.2"
    )

    print(
        "=========================================="
    )

    print(
        f"Capital configured: "
        f"{money(CAPITAL)}"
    )

    print(
        f"Risk per trade: "
        f"{RISK_PCT * 100:.2f}%"
    )

    print(
        f"Maximum position: "
        f"{MAX_POSITION_PCT * 100:.2f}%"
    )

    print("")

    # --------------------------------------------------------
    # MARKET DATA
    # --------------------------------------------------------

    frames = download_history()

    if not frames:

        raise RuntimeError(
            "No market data was downloaded."
        )

    print(
        f"Received data for "
        f"{len(frames)} symbols."
    )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    regime = market_regime(
        frames
    )

    print(
        f"Market regime: "
        f"{regime['label']}"
    )

    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    candidates = (
        build_candidates(
            frames,
            regime
        )
    )

    print(
        f"Candidate count: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    candidates = (
        classify_candidates(
            candidates,
            regime
        )
    )

    # --------------------------------------------------------
    # IPO
    # --------------------------------------------------------

    (
        ipo_records,
        ipo_error
    ) = fetch_ipo_data()

    # --------------------------------------------------------
    # ALERT
    # --------------------------------------------------------

    alert = alert_text(

        regime,

        candidates,

        ipo_records,

        ipo_error,

    )

    print("")
    print(alert)
    print("")

    # --------------------------------------------------------
    # AUDIT
    # --------------------------------------------------------

    save_audit(

        alert,

        candidates,

    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    send_telegram(
        alert
    )

    print(
        "Telegram alert sent successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
