"""
MARKET ALERT V6.3.5
DIAGNOSTIC WALK-FORWARD BACKTEST

Purpose:
    Diagnose why V6.3.2/V6.3.4 produced too few or zero trades.

Important:
    This is a research/backtest program. It does NOT replace the
    live market alert engine.

Key fixes versus V6.3.4:
    1. Correct boolean PASS/FAIL counting.
    2. No negative failure counts.
    3. No dependency on V6.3.2 internals.
    4. Uses a standalone, leakage-controlled empirical analogue model.
    5. Calculates forward outcomes using positional trading-day logic.
    6. Evaluates ALL candidates, including rejected candidates.
    7. Separately measures each filter's incremental effect.
    8. Does not manufacture a positive target merely to satisfy RR.
    9. Skips incomplete forward outcomes from performance statistics.
   10. Produces CSV files that can be inspected after the GitHub run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

START_SIGNAL_DATE = "2026-08-10"
END_SIGNAL_DATE = "2026-08-28"

DATA_START_DATE = "2024-01-01"

HORIZONS = [1, 3, 5]

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(exist_ok=True)

N_ANALOGUES = 30
MIN_ANALOGUES = 12
LOOKBACK_DAYS = 756

# Conservative research assumption for round-trip costs.
ROUND_TRIP_COST = 0.0015


# ============================================================
# NSE UNIVERSE
# ============================================================

SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT",
    "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
    "BEL", "BHARTIARTL", "BPCL", "BRITANNIA", "CIPLA",
    "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO",
    "HINDALCO", "HINDUNILVR", "ICICIBANK", "ICICIGI",
    "ICICIPRULI", "INDUSINDBK", "INFY", "ITC", "JIOFIN",
    "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI",
    "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE",
    "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
    "TRENT", "ULTRACEMCO", "WIPRO", "VEDL", "APOLLOTYRE",
    "AUROPHARMA", "BOSCHLTD", "SAIL", "DLF", "NAUKRI",
]

NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"


# ============================================================
# THRESHOLD PROFILES
# ============================================================

PROFILES = {

    "STRICT_V6_3_2_STYLE": {
        "p3": 0.54,
        "p5": 0.53,
        "er3": 0.002,
        "er5": 0.004,
        "rr1": 1.00,
        "rr2": 1.50,
        "quality": 0.35,
        "trend": -0.02,
        "max_rsi": 76.0,
    },

    "BALANCED": {
        "p3": 0.53,
        "p5": 0.52,
        "er3": 0.001,
        "er5": 0.002,
        "rr1": 0.90,
        "rr2": 1.25,
        "quality": 0.30,
        "trend": -0.025,
        "max_rsi": 78.0,
    },

    "MODERATE": {
        "p3": 0.52,
        "p5": 0.51,
        "er3": 0.000,
        "er5": 0.001,
        "rr1": 0.80,
        "rr2": 1.10,
        "quality": 0.25,
        "trend": -0.03,
        "max_rsi": 80.0,
    },
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def norm_date(x):
    ts = pd.Timestamp(x)

    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts.normalize()


def ticker_for(symbol):
    return symbol + ".NS"


def clean_frame(df):

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        return None

    needed = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    for c in needed:

        if c not in df.columns:
            df[c] = np.nan

    df = df[
        needed
    ].copy()

    df.index = pd.to_datetime(
        df.index
    )

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    df = df.sort_index()

    return df


def split_download(raw):

    if raw is None or raw.empty:
        return {}

    if not isinstance(
        raw.columns,
        pd.MultiIndex
    ):
        return {
            "SINGLE":
                clean_frame(raw)
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

    result = {}

    if "Close" in level0:

        for ticker in level1:

            try:

                part = raw.xs(
                    ticker,
                    axis=1,
                    level=1,
                    drop_level=True
                )

                result[str(ticker)] = (
                    clean_frame(part)
                )

            except Exception:
                pass

    else:

        for ticker in level0:

            try:

                part = raw.xs(
                    ticker,
                    axis=1,
                    level=0,
                    drop_level=True
                )

                result[str(ticker)] = (
                    clean_frame(part)
                )

            except Exception:
                pass

    return {
        k: v
        for k, v in result.items()
        if v is not None
        and not v.empty
    }


# ============================================================
# TECHNICAL FEATURES
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
        min_periods=period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
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
        (
            100
            /
            (1 + rs)
        )
    )

    result = result.where(
        avg_loss != 0,
        100.0
    )

    return result


def make_features(df):

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

    x["rsi"] = rsi(
        close,
        14
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

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (
                high
                - prev_close
            ).abs(),
            (
                low
                - prev_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = (
        tr
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

    x["vol_ratio"] = (
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


# ============================================================
# MARKET REGIME
# ============================================================

def get_market_regime(
    frames,
    signal_date
):

    nifty = frames.get(
        NIFTY_TICKER
    )

    vix = frames.get(
        VIX_TICKER
    )

    if nifty is None:

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

    last = safe_float(
        close.iloc[-1]
    )

    sma_now = safe_float(
        sma50.iloc[-1]
    )

    sma_old = safe_float(
        sma50.iloc[-6]
    )

    vix_value = np.nan

    if vix is not None:

        vv = vix.loc[
            vix.index <= signal_date
        ]

        if not vv.empty:

            vix_value = safe_float(
                vv["Close"].iloc[-1]
            )

    above = (
        np.isfinite(last)
        and np.isfinite(sma_now)
        and last > sma_now
    )

    rising = (
        np.isfinite(sma_now)
        and np.isfinite(sma_old)
        and sma_now > sma_old
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

    return {
        "regime": regime,
        "nifty": last,
        "sma50": sma_now,
        "vix": vix_value,
    }


# ============================================================
# EMPIRICAL ANALOGUE MODEL
# ============================================================

FEATURE_COLUMNS = [

    "rsi",
    "ret3",
    "ret5",
    "ret10",
    "dist_sma20",
    "dist_sma50",
    "sma50_slope5",
    "atr_pct",
    "vol20",
    "vol_ratio",

]


def empirical_model(
    features,
    signal_date
):

    clean = features.loc[
        features.index <= signal_date
    ].copy()

    required = (
        FEATURE_COLUMNS
        + ["Close"]
    )

    clean = clean.dropna(
        subset=required
    )

    if len(clean) < 260:
        return None

    candidate = clean.iloc[-1]

    # Historical observations must be old enough for their
    # subsequent 3/5-day returns to already be known.
    train = clean.iloc[:-5].copy()

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

    for col in FEATURE_COLUMNS:

        series = train[col]

        median = series.median()

        mad = (
            series - median
        ).abs().median()

        if (
            not np.isfinite(mad)
            or mad == 0
        ):

            scale = series.std()

        else:

            scale = (
                1.4826 * mad
            )

        if (
            not np.isfinite(scale)
            or scale == 0
        ):

            scale = 1.0

        candidate_value = safe_float(
            candidate[col],
            median
        )

        distances += (

            (
                series.values
                - candidate_value
            )
            /
            scale

        ) ** 2

    distances = np.sqrt(
        distances
        /
        len(FEATURE_COLUMNS)
    )

    train["distance"] = (
        distances
    )

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

    p3 = float(
        (
            weights
            *
            (
                train["future3"]
                > 0
            )
        ).sum()
    )

    p5 = float(
        (
            weights
            *
            (
                train["future5"]
                > 0
            )
        ).sum()
    )

    med3 = float(
        train["future3"].median()
    )

    med5 = float(
        train["future5"].median()
    )

    consistency3 = 1 - min(
        1.0,
        float(
            train["future3"].std()
        )
        /
        max(
            0.01,
            abs(er3) + 0.01
        )
    )

    consistency5 = 1 - min(
        1.0,
        float(
            train["future5"].std()
        )
        /
        max(
            0.015,
            abs(er5) + 0.015
        )
    )

    sample_quality = min(
        1.0,
        len(train)
        /
        N_ANALOGUES
    )

    quality = (

        0.40
        * sample_quality

        +

        0.30
        * max(
            0.0,
            consistency3
        )

        +

        0.30
        * max(
            0.0,
            consistency5
        )

    )

    return {

        "p3": p3,
        "p5": p5,

        "er3": er3,
        "er5": er5,

        "median3": med3,
        "median5": med5,

        "quality": quality,

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
# RISK / TARGET MODEL
# ============================================================

def risk_model(row):

    price = safe_float(
        row["price"]
    )

    atr_pct = safe_float(
        row["atr_pct"]
    )

    er3 = safe_float(
        row["er3"],
        0.0
    )

    er5 = safe_float(
        row["er5"],
        0.0
    )

    if (
        not np.isfinite(price)
        or
        not np.isfinite(atr_pct)
        or
        atr_pct <= 0
    ):

        return {
            "stop": np.nan,
            "target1": np.nan,
            "target2": np.nan,
            "rr1": np.nan,
            "rr2": np.nan,
        }

    stop_distance = max(
        0.012,
        min(
            0.05,
            1.5 * atr_pct
        )
    )

    stop = (
        price
        * (1 - stop_distance)
    )

    # IMPORTANT:
    # Do not manufacture a positive target when expected
    # return is negative.
    target1_return = max(
        0.0,
        min(
            0.06,
            er3
        )
    )

    target2_return = max(
        0.0,
        min(
            0.10,
            er5
        )
    )

    target1 = (
        price
        * (1 + target1_return)
    )

    target2 = (
        price
        * (1 + target2_return)
    )

    risk = (
        price
        - stop
    )

    if risk <= 0:

        rr1 = np.nan
        rr2 = np.nan

    else:

        rr1 = (
            target1
            - price
        ) / risk

        rr2 = (
            target2
            - price
        ) / risk

    return {

        "stop": stop,
        "target1": target1,
        "target2": target2,
        "rr1": rr1,
        "rr2": rr2,

    }


# ============================================================
# FILTER EVALUATION
# ============================================================

def evaluate_profile(
    row,
    regime,
    profile
):

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

    rr1 = safe_float(
        row["rr1"]
    )

    rr2 = safe_float(
        row["rr2"]
    )

    quality = safe_float(
        row["quality"]
    )

    trend = safe_float(
        row["trend"]
    )

    rsi_value = safe_float(
        row["rsi"]
    )

    checks = {

        "p3":
            (
                np.isfinite(p3)
                and p3 >= profile["p3"]
            ),

        "p5":
            (
                np.isfinite(p5)
                and p5 >= profile["p5"]
            ),

        "er3":
            (
                np.isfinite(er3)
                and er3 > profile["er3"]
            ),

        "er5":
            (
                np.isfinite(er5)
                and er5 > profile["er5"]
            ),

        "rr1":
            (
                np.isfinite(rr1)
                and rr1 >= profile["rr1"]
            ),

        "rr2":
            (
                np.isfinite(rr2)
                and rr2 >= profile["rr2"]
            ),

        "quality":
            (
                np.isfinite(quality)
                and quality >= profile["quality"]
            ),

        "trend":
            (
                np.isfinite(trend)
                and trend > profile["trend"]
            ),

        "rsi":
            (
                np.isfinite(rsi_value)
                and rsi_value < profile["max_rsi"]
            ),

        "regime":
            (
                regime
                != "UNFAVORABLE"
            ),

    }

    failed = [

        name

        for name, passed
        in checks.items()

        if not bool(passed)

    ]

    passed = bool(
        all(
            bool(v)
            for v in checks.values()
        )
    )

    return (
        passed,
        checks,
        failed,
    )


# ============================================================
# FORWARD OUTCOMES
# ============================================================

def forward_outcome(
    df,
    signal_date,
    horizon
):

    if df is None or df.empty:
        return None

    dates = [

        norm_date(x)

        for x in df.index

    ]

    dates = sorted(
        set(dates)
    )

    signal_date = norm_date(
        signal_date
    )

    future_dates = [

        d

        for d in dates

        if d > signal_date

    ]

    if len(future_dates) < horizon:
        return None

    entry_date = future_dates[0]

    exit_date = future_dates[
        horizon - 1
    ]

    try:

        entry = safe_float(
            df.loc[
                entry_date,
                "Open"
            ]
        )

    except Exception:

        return None

    if not np.isfinite(entry):

        try:

            entry = safe_float(
                df.loc[
                    entry_date,
                    "Close"
                ]
            )

        except Exception:

            return None

    try:

        exit_price = safe_float(
            df.loc[
                exit_date,
                "Close"
            ]
        )

    except Exception:

        return None

    if (

        not np.isfinite(entry)

        or

        not np.isfinite(
            exit_price
        )

        or

        entry <= 0

    ):

        return None

    gross = (
        exit_price
        /
        entry
        - 1.0
    )

    net = (
        gross
        -
        ROUND_TRIP_COST
    )

    return {

        "entry_date":
            entry_date.strftime(
                "%Y-%m-%d"
            ),

        "exit_date":
            exit_date.strftime(
                "%Y-%m-%d"
            ),

        "entry":
            entry,

        "exit":
            exit_price,

        "gross_return":
            gross,

        "net_return":
            net,

    }


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def summarize_returns(
    df,
    selection_name,
    selection_mask
):

    rows = []

    selected = df[
        selection_mask
    ].copy()

    for horizon in HORIZONS:

        col = (
            f"net_return_{horizon}d"
        )

        gross_col = (
            f"gross_return_{horizon}d"
        )

        if col not in selected.columns:
            continue

        net = pd.to_numeric(
            selected[col],
            errors="coerce"
        ).dropna()

        gross = pd.to_numeric(
            selected[gross_col],
            errors="coerce"
        ).dropna()

        if net.empty:
            continue

        winners = net[
            net > 0
        ]

        losers = net[
            net <= 0
        ]

        gross_profit = (
            winners.sum()
        )

        gross_loss = (
            -losers.sum()
        )

        if gross_loss > 0:

            profit_factor = (
                gross_profit
                /
                gross_loss
            )

        else:

            profit_factor = np.inf

        rows.append({

            "selection":
                selection_name,

            "horizon":
                horizon,

            "observations":
                len(net),

            "win_rate":
                float(
                    (
                        net > 0
                    ).mean()
                ),

            "average_net_return":
                float(
                    net.mean()
                ),

            "median_net_return":
                float(
                    net.median()
                ),

            "average_gross_return":
                float(
                    gross.mean()
                ),

            "average_winner":
                float(
                    winners.mean()
                )
                if not winners.empty
                else np.nan,

            "average_loser":
                float(
                    losers.mean()
                )
                if not losers.empty
                else np.nan,

            "profit_factor":
                float(
                    profit_factor
                ),

            "best":
                float(
                    net.max()
                ),

            "worst":
                float(
                    net.min()
                ),

        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 72)
    print(
        "MARKET ALERT V6.3.5 "
        "DIAGNOSTIC WALK-FORWARD BACKTEST"
    )
    print("=" * 72)

    print("")

    print(
        f"Signal window: "
        f"{START_SIGNAL_DATE} "
        f"to "
        f"{END_SIGNAL_DATE}"
    )

    print(
        f"Historical data starts: "
        f"{DATA_START_DATE}"
    )

    print(
        f"Stocks in diagnostic universe: "
        f"{len(SYMBOLS)}"
    )

    print("")

    tickers = [
        ticker_for(symbol)
        for symbol in SYMBOLS
    ]

    tickers += [
        NIFTY_TICKER,
        VIX_TICKER,
    ]

    tickers = list(
        dict.fromkeys(
            tickers
        )
    )

    print(
        f"Downloading "
        f"{len(tickers)} tickers..."
    )

    raw = yf.download(

        tickers=tickers,

        start=DATA_START_DATE,

        end="2026-08-30",

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
            "No market data "
            "was downloaded."
        )

    print(
        f"Downloaded frames: "
        f"{len(frames)}"
    )

    # --------------------------------------------------------
    # Precompute technical features.
    # --------------------------------------------------------

    feature_frames = {}

    for symbol in SYMBOLS:

        ticker = ticker_for(
            symbol
        )

        df = frames.get(
            ticker
        )

        if df is None or df.empty:
            continue

        try:

            feature_frames[
                symbol
            ] = make_features(
                df
            )

        except Exception:

            continue

    nifty = frames.get(
        NIFTY_TICKER
    )

    if nifty is None or nifty.empty:

        raise RuntimeError(
            "NIFTY data unavailable."
        )

    signal_dates = sorted({

        norm_date(x)

        for x in nifty.index

        if (

            norm_date(x)
            >= norm_date(
                START_SIGNAL_DATE
            )

            and

            norm_date(x)
            <= norm_date(
                END_SIGNAL_DATE
            )

        )

    })

    print(
        f"Signal dates available: "
        f"{len(signal_dates)}"
    )

    print("")

    records = []

    # ========================================================
    # WALK FORWARD
    # ========================================================

    for signal_date in signal_dates:

        market = get_market_regime(
            frames,
            signal_date
        )

        if np.isfinite(
            market["nifty"]
        ):

            print(

                f"{signal_date.date()} | "
                f"{market['regime']} | "
                f"NIFTY="
                f"{market['nifty']:.2f}"

            )

        else:

            print(

                f"{signal_date.date()} | "
                f"{market['regime']}"

            )

        for symbol in SYMBOLS:

            features = (
                feature_frames.get(
                    symbol
                )
            )

            if features is None:
                continue

            hist = features.loc[
                features.index <= signal_date
            ].copy()

            if len(hist) < 260:
                continue

            clean = hist.dropna(
                subset=[
                    "Close",
                    "rsi",
                    "atr_pct",
                    "dist_sma50",
                    "ret3",
                    "ret5",
                ]
            )

            if clean.empty:
                continue

            latest = clean.iloc[-1]

            model = empirical_model(
                features,
                signal_date
            )

            if model is None:
                continue

            row = {

                "signal_date":
                    signal_date.strftime(
                        "%Y-%m-%d"
                    ),

                "symbol":
                    symbol,

                "regime":
                    market["regime"],

                "nifty":
                    market["nifty"],

                "sma50":
                    market["sma50"],

                "vix":
                    market["vix"],

                "price":
                    safe_float(
                        latest["Close"]
                    ),

                "rsi":
                    safe_float(
                        latest["rsi"]
                    ),

                "trend":
                    safe_float(
                        latest["dist_sma50"]
                    ),

                "atr_pct":
                    safe_float(
                        latest["atr_pct"]
                    ),

                "vol_ratio":
                    safe_float(
                        latest["vol_ratio"]
                    ),

                "ret3":
                    safe_float(
                        latest["ret3"]
                    ),

                "ret5":
                    safe_float(
                        latest["ret5"]
                    ),

                **model,

            }

            row.update(
                risk_model(row)
            )

            # ------------------------------------------------
            # Evaluate all profiles.
            # ------------------------------------------------

            for (
                profile_name,
                profile
            ) in PROFILES.items():

                (
                    passed,
                    checks,
                    failed
                ) = evaluate_profile(

                    row,

                    market["regime"],

                    profile

                )

                row[
                    f"{profile_name}_trade"
                ] = bool(passed)

                row[
                    f"{profile_name}_failed"
                ] = (

                    ",".join(failed)

                    if failed

                    else ""

                )

                for (
                    name,
                    value
                ) in checks.items():

                    row[
                        f"{profile_name}_{name}"
                    ] = bool(value)

            # ------------------------------------------------
            # Final diagnostic classification.
            # ------------------------------------------------

            if row[
                "STRICT_V6_3_2_STYLE_trade"
            ]:

                action = "TRADE"

            elif row[
                "BALANCED_trade"
            ]:

                action = (
                    "WATCH_BALANCED"
                )

            elif row[
                "MODERATE_trade"
            ]:

                action = (
                    "WATCH_MODERATE"
                )

            else:

                action = "REJECT"

            row[
                "classification"
            ] = action

            # ------------------------------------------------
            # Forward outcomes for EVERY candidate.
            # ------------------------------------------------

            stock_df = frames.get(
                ticker_for(symbol)
            )

            for horizon in HORIZONS:

                outcome = forward_outcome(

                    stock_df,

                    signal_date,

                    horizon

                )

                if outcome is None:
                    continue

                row[
                    f"entry_date_{horizon}d"
                ] = outcome[
                    "entry_date"
                ]

                row[
                    f"exit_date_{horizon}d"
                ] = outcome[
                    "exit_date"
                ]

                row[
                    f"entry_{horizon}d"
                ] = outcome[
                    "entry"
                ]

                row[
                    f"exit_{horizon}d"
                ] = outcome[
                    "exit"
                ]

                row[
                    f"gross_return_{horizon}d"
                ] = outcome[
                    "gross_return"
                ]

                row[
                    f"net_return_{horizon}d"
                ] = outcome[
                    "net_return"
                ]

            records.append(
                row
            )

    audit = pd.DataFrame(
        records
    )

    if audit.empty:

        raise RuntimeError(
            "No diagnostic observations "
            "were created."
        )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    # ========================================================
    # FULL CANDIDATE AUDIT
    # ========================================================

    audit_file = (

        AUDIT_DIR
        /
        f"diagnostic_v6_3_5_{timestamp}.csv"

    )

    audit.to_csv(
        audit_file,
        index=False
    )

    # ========================================================
    # CORRECT FILTER COUNTS
    # ========================================================

    filter_rows = []

    strict_prefix = (
        "STRICT_V6_3_2_STYLE_"
    )

    for filter_name in [

        "p3",
        "p5",
        "er3",
        "er5",
        "rr1",
        "rr2",
        "quality",
        "trend",
        "rsi",
        "regime",

    ]:

        col = (
            strict_prefix
            + filter_name
        )

        if col not in audit.columns:
            continue

        values = (
            audit[col]
            .fillna(False)
            .astype(bool)
        )

        passed = int(
            values.sum()
        )

        total = len(values)

        failed = (
            total
            - passed
        )

        filter_rows.append({

            "filter":
                filter_name,

            "passed":
                passed,

            "failed":
                failed,

            "pass_rate":
                passed / total
                if total
                else np.nan,

        })

    filter_df = pd.DataFrame(
        filter_rows
    )

    filter_file = (

        AUDIT_DIR
        /
        f"filter_failures_v6_3_5_{timestamp}.csv"

    )

    filter_df.to_csv(
        filter_file,
        index=False
    )

    # ========================================================
    # PROFILE PERFORMANCE
    # ========================================================

    profile_rows = []

    for profile_name in PROFILES:

        col = (
            f"{profile_name}_trade"
        )

        if col not in audit.columns:
            continue

        profile_rows += summarize_returns(

            audit,

            profile_name,

            audit[col]
            .fillna(False)
            .astype(bool)

        )

    profile_df = pd.DataFrame(
        profile_rows
    )

    profile_file = (

        AUDIT_DIR
        /
        f"profile_comparison_v6_3_5_{timestamp}.csv"

    )

    profile_df.to_csv(
        profile_file,
        index=False
    )

    # ========================================================
    # ALL-CANDIDATE BASELINE
    # ========================================================

    baseline_rows = (
        summarize_returns(

            audit,

            "ALL_CANDIDATES",

            pd.Series(
                True,
                index=audit.index
            )

        )
    )

    baseline_df = pd.DataFrame(
        baseline_rows
    )

    baseline_file = (

        AUDIT_DIR
        /
        f"all_candidate_baseline_v6_3_5_{timestamp}.csv"

    )

    baseline_df.to_csv(
        baseline_file,
        index=False
    )

    # ========================================================
    # ACTION GROUP PERFORMANCE
    # ========================================================

    action_rows = []

    for action in [

        "TRADE",
        "WATCH_BALANCED",
        "WATCH_MODERATE",
        "REJECT",

    ]:

        action_rows += (
            summarize_returns(

                audit,

                action,

                audit[
                    "classification"
                ]
                == action

            )
        )

    action_df = pd.DataFrame(
        action_rows
    )

    action_file = (

        AUDIT_DIR
        /
        f"action_group_performance_v6_3_5_{timestamp}.csv"

    )

    action_df.to_csv(
        action_file,
        index=False
    )

    # ========================================================
    # INCREMENTAL FILTER ANALYSIS
    # ========================================================

    incremental_rows = []

    filters_in_order = [

        "p3",
        "p5",
        "er3",
        "er5",
        "rr1",
        "rr2",
        "quality",
        "trend",
        "rsi",
        "regime",

    ]

    mask = pd.Series(
        True,
        index=audit.index
    )

    for filter_name in filters_in_order:

        col = (
            strict_prefix
            + filter_name
        )

        if col not in audit.columns:
            continue

        mask = (

            mask

            &

            audit[col]
            .fillna(False)
            .astype(bool)

        )

        selected = audit[
            mask
        ]

        for horizon in HORIZONS:

            return_col = (
                f"net_return_{horizon}d"
            )

            if return_col not in selected.columns:
                continue

            returns = pd.to_numeric(

                selected[
                    return_col
                ],

                errors="coerce"

            ).dropna()

            if returns.empty:
                continue

            incremental_rows.append({

                "filter_added":
                    filter_name,

                "remaining_candidates":
                    len(selected),

                "horizon":
                    horizon,

                "observations":
                    len(returns),

                "win_rate":
                    float(
                        (
                            returns > 0
                        ).mean()
                    ),

                "average_net_return":
                    float(
                        returns.mean()
                    ),

                "median_net_return":
                    float(
                        returns.median()
                    ),

                "best":
                    float(
                        returns.max()
                    ),

                "worst":
                    float(
                        returns.min()
                    ),

            })

    incremental_df = pd.DataFrame(
        incremental_rows
    )

    incremental_file = (

        AUDIT_DIR
        /
        f"incremental_filter_analysis_v6_3_5_{timestamp}.csv"

    )

    incremental_df.to_csv(
        incremental_file,
        index=False
    )

    # ========================================================
    # SIGNAL-DATE SUMMARY
    # ========================================================

    date_rows = []

    for signal_date, group in (
        audit.groupby(
            "signal_date"
        )
    ):

        date_rows.append({

            "signal_date":
                signal_date,

            "candidates":
                len(group),

            "strict_trades":
                int(

                    group[
                        "STRICT_V6_3_2_STYLE_trade"
                    ]
                    .fillna(False)
                    .astype(bool)
                    .sum()

                ),

            "balanced_trades":
                int(

                    group[
                        "BALANCED_trade"
                    ]
                    .fillna(False)
                    .astype(bool)
                    .sum()

                ),

            "moderate_trades":
                int(

                    group[
                        "MODERATE_trade"
                    ]
                    .fillna(False)
                    .astype(bool)
                    .sum()

                ),

            "rejected":
                int(

                    (
                        group[
                            "classification"
                        ]
                        == "REJECT"
                    ).sum()

                ),

            "regime":
                group[
                    "regime"
                ].iloc[0],

        })

    date_df = pd.DataFrame(
        date_rows
    )

    date_file = (

        AUDIT_DIR
        /
        f"signal_date_summary_v6_3_5_{timestamp}.csv"

    )

    date_df.to_csv(
        date_file,
        index=False
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("")
    print("=" * 72)
    print(
        "V6.3.5 DIAGNOSTIC SUMMARY"
    )
    print("=" * 72)

    print(
        f"Total candidate observations: "
        f"{len(audit)}"
    )

    print("")
    print("ACTION COUNTS:")

    print(

        audit[
            "classification"
        ]
        .value_counts()
        .to_string()

    )

    print("")
    print("STRICT FILTER COUNTS:")

    if not filter_df.empty:

        print(

            filter_df.to_string(
                index=False
            )

        )

    else:

        print(
            "No filter diagnostics."
        )

    print("")
    print("ALL-CANDIDATE BASELINE:")

    if not baseline_df.empty:

        print(

            baseline_df.to_string(
                index=False
            )

        )

    else:

        print(
            "No completed forward outcomes."
        )

    print("")
    print("PROFILE COMPARISON:")

    if not profile_df.empty:

        print(

            profile_df.to_string(
                index=False
            )

        )

    else:

        print(
            "No completed profile outcomes."
        )

    print("")
    print("ACTION GROUP PERFORMANCE:")

    if not action_df.empty:

        print(

            action_df.to_string(
                index=False
            )

        )

    else:

        print(
            "No completed action-group outcomes."
        )

    print("")
    print("INCREMENTAL FILTER ANALYSIS:")

    if not incremental_df.empty:

        print(

            incremental_df.to_string(
                index=False
            )

        )

    else:

        print(
            "No incremental results."
        )

    print("")
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    for path in [

        audit_file,
        filter_file,
        profile_file,
        baseline_file,
        action_file,
        incremental_file,
        date_file,

    ]:

        print(path)

    print("")
    print(
        "V6.3.5 diagnostic completed."
    )

    print(
        "Do NOT promote any threshold profile "
        "to the live engine from this short test alone."
    )


if __name__ == "__main__":
    main()
