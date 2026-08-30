import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import market_engine_v6_3_2 as engine


# ============================================================
# MARKET ALERT V6.3.4
# DIAGNOSTIC WALK-FORWARD BACKTEST
# ============================================================
#
# This is an AUDIT/RESEARCH engine.
# It DOES NOT replace the live V6.3.2 engine.
#
# It answers four questions:
#
# 1) Which V6.3.2 filter rejected each candidate?
# 2) What happened afterward to rejected/watch candidates?
# 3) Are the current thresholds too strict?
# 4) Which threshold profile has the best historical
#    risk-adjusted expectancy?
#
# Anti-look-ahead design:
# - Each historical signal uses only data available up to the
#   signal date.
# - Hypothetical entry = next trading day's OPEN.
# - Outcomes = 1/3/5 trading-session CLOSE.
#
# Validation window:
#   2026-08-10 through 2026-08-28
# ============================================================


START_DATE = "2026-08-10"
END_DATE = "2026-08-29"


# ============================================================
# TEST HORIZONS
# ============================================================

HORIZONS = [1, 3, 5]


# ============================================================
# THRESHOLD PROFILES
# ============================================================
#
# STRICT:
# Current V6.3.2-style filters.
#
# BALANCED:
# Slightly more permissive.
#
# MODERATE:
# Diagnostic only.
# NOT a recommendation for live trading.
# ============================================================

PROFILES = {

    "STRICT_V6_3_2": {

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
# OUTPUT DIRECTORY
# ============================================================

AUDIT_DIR = Path("audit")

AUDIT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# YFINANCE DATA HANDLING
# ============================================================

def split_yf_download(raw):

    if raw is None or raw.empty:
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
                drop_level=True
            )

            for ticker in level1

        }

    return {

        ticker: raw.xs(
            ticker,
            axis=1,
            level=0,
            drop_level=True
        )

        for ticker in level0

    }


# ============================================================
# CLEAN DATAFRAME
# ============================================================

def get_frame(
    frames,
    ticker
):

    df = frames.get(
        ticker
    )

    if df is None or df.empty:
        return None

    if "Close" not in df.columns:
        return None

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:

        if col not in df.columns:

            df[col] = np.nan

    return df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    ].dropna(
        subset=["Close"]
    ).copy()


# ============================================================
# DATE NORMALIZATION
# ============================================================

def normalize_date(
    value
):

    ts = pd.Timestamp(
        value
    )

    if ts.tzinfo is not None:

        ts = ts.tz_localize(
            None
        )

    return ts.normalize()


# ============================================================
# SAFE FLOAT
# ============================================================

def finite_float(
    value,
    default=np.nan
):

    try:

        result = float(
            value
        )

        if np.isfinite(
            result
        ):

            return result

    except Exception:
        pass

    return default


# ============================================================
# HISTORICAL MARKET REGIME
# ============================================================

def historical_regime(
    frames,
    signal_date
):

    nifty_df = get_frame(
        frames,
        engine.NIFTY
    )

    vix_df = get_frame(
        frames,
        engine.VIX
    )

    if nifty_df is None:

        return {

            "label": "UNKNOWN",

            "nifty": np.nan,

            "sma50": np.nan,

            "vix": np.nan

        }

    nifty_df = nifty_df.loc[
        nifty_df.index <= signal_date
    ]

    if len(nifty_df) < 60:

        return {

            "label": "UNKNOWN",

            "nifty": np.nan,

            "sma50": np.nan,

            "vix": np.nan

        }

    close = nifty_df[
        "Close"
    ]

    sma50 = (
        close
        .rolling(50)
        .mean()
    )

    nifty = finite_float(
        close.iloc[-1]
    )

    sma50_now = finite_float(
        sma50.iloc[-1]
    )

    sma50_old = np.nan

    if len(sma50) >= 6:

        sma50_old = finite_float(
            sma50.iloc[-6]
        )

    vix = np.nan

    if vix_df is not None:

        vix_df = vix_df.loc[
            vix_df.index <= signal_date
        ]

        if not vix_df.empty:

            vix = finite_float(
                vix_df[
                    "Close"
                ].iloc[-1]
            )

    above = (

        np.isfinite(
            nifty
        )

        and

        np.isfinite(
            sma50_now
        )

        and

        nifty > sma50_now

    )

    rising = (

        np.isfinite(
            sma50_now
        )

        and

        np.isfinite(
            sma50_old
        )

        and

        sma50_now
        > sma50_old

    )

    if above and rising:

        label = "FAVORABLE"

    elif (
        not above
        and not rising
    ):

        label = "UNFAVORABLE"

    else:

        label = "MIXED"

    return {

        "label": label,

        "nifty": nifty,

        "sma50": sma50_now,

        "vix": vix

    }


# ============================================================
# BUILD HISTORICAL CANDIDATE
# ============================================================

def build_candidate(
    frames,
    symbol,
    signal_date,
    regime_label
):

    ticker = (
        symbol
        + ".NS"
    )

    raw = get_frame(
        frames,
        ticker
    )

    if raw is None:

        return None

    historical = raw.loc[
        raw.index <= signal_date
    ].copy()

    if len(historical) < 230:

        return None

    try:

        features = (
            engine.make_features(
                historical
            )
        )

    except Exception as exc:

        return {

            "symbol": symbol,

            "data_error":
                "make_features: "
                + str(exc)

        }

    required = [

        "rsi",
        "sma50",
        "ema10",
        "atr_pct",
        "ret3",
        "ret5"

    ]

    missing = [

        col
        for col in required
        if col not in features.columns

    ]

    if missing:

        return {

            "symbol": symbol,

            "data_error":
                "Missing feature columns: "
                + ",".join(missing)

        }

    features = (
        features.dropna(
            subset=required
        )
    )

    if len(features) < 220:

        return {

            "symbol": symbol,

            "data_error":
                "Insufficient clean feature history"

        }

    try:

        model = (
            engine.empirical_analog_model(
                features
            )
        )

    except Exception as exc:

        return {

            "symbol": symbol,

            "data_error":
                "empirical_analog_model: "
                + str(exc)

        }

    if model is None:

        return {

            "symbol": symbol,

            "data_error":
                "Model returned no result"

        }

    last = (
        features.iloc[-1]
    )

    price = finite_float(
        last["Close"]
    )

    atr_pct = finite_float(
        last["atr_pct"]
    )

    if (

        not np.isfinite(price)

        or

        not np.isfinite(atr_pct)

        or

        atr_pct <= 0

    ):

        return {

            "symbol": symbol,

            "data_error":
                "Invalid price or ATR"

        }

    row = {

        "symbol":
            symbol,

        "price":
            price,

        "rsi":
            finite_float(
                last["rsi"]
            ),

        "vol_ratio":
            finite_float(
                last["vol_ratio"]
            ),

        "atr_pct":
            atr_pct,

        "trend":
            finite_float(
                last["dist_sma50"]
            ),

    }

    # --------------------------------------------------------
    # MODEL VALUES
    # --------------------------------------------------------

    for key, value in model.items():

        row[key] = finite_float(
            value
        )

    # Make sure all diagnostic fields exist.

    for key in [

        "p1",
        "p3",
        "p5",

        "er1",
        "er3",
        "er5",

        "median3",
        "median5",

        "quality"

    ]:

        if key not in row:

            row[key] = np.nan

    # --------------------------------------------------------
    # MODEL SCORE
    # --------------------------------------------------------

    try:

        row["score"] = finite_float(

            engine.score_candidate(

                row,

                regime_label

            )

        )

    except Exception:

        row["score"] = np.nan

    # ========================================================
    # RISK MODEL
    # ========================================================

    stop_distance = max(

        0.012,

        min(

            0.05,

            1.5 * atr_pct

        )

    )

    stop = (

        price

        * (

            1
            - stop_distance

        )

    )

    expected3 = max(

        finite_float(
            row["er3"],
            0.0
        ),

        finite_float(
            row["median3"],
            0.0
        ),

        0.006

    )

    target1 = (

        price

        * (

            1

            + max(

                0.006,

                min(

                    0.06,

                    expected3

                )

            )

        )

    )

    expected5 = max(

        finite_float(
            row["er5"],
            0.0
        ),

        finite_float(
            row["median5"],
            0.0
        ),

        0.012

    )

    target2 = (

        price

        * (

            1

            + max(

                0.012,

                min(

                    0.10,

                    expected5

                )

            )

        )

    )

    if target2 <= target1:

        target2 = (
            target1
            * 1.006
        )

    risk = (
        price
        - stop
    )

    if risk <= 0:

        row[
            "risk_error"
        ] = (
            "Non-positive risk"
        )

        return row

    row["stop"] = stop

    row["target1"] = target1

    row["target2"] = target2

    row["rr1"] = (

        target1
        - price

    ) / risk

    row["rr2"] = (

        target2
        - price

    ) / risk

    return row


# ============================================================
# FILTER DIAGNOSTICS
# ============================================================

def evaluate_profile(
    row,
    regime_label,
    profile
):

    p3 = finite_float(
        row.get("p3")
    )

    p5 = finite_float(
        row.get("p5")
    )

    er3 = finite_float(
        row.get("er3")
    )

    er5 = finite_float(
        row.get("er5")
    )

    rr1 = finite_float(
        row.get("rr1")
    )

    rr2 = finite_float(
        row.get("rr2")
    )

    quality = finite_float(
        row.get("quality")
    )

    trend = finite_float(
        row.get("trend")
    )

    rsi = finite_float(
        row.get("rsi")
    )

    checks = {}

    checks["p3"] = (

        np.isfinite(p3)

        and

        p3 >= profile["p3"]

    )

    checks["p5"] = (

        np.isfinite(p5)

        and

        p5 >= profile["p5"]

    )

    checks["er3"] = (

        np.isfinite(er3)

        and

        er3 > profile["er3"]

    )

    checks["er5"] = (

        np.isfinite(er5)

        and

        er5 > profile["er5"]

    )

    checks["rr1"] = (

        np.isfinite(rr1)

        and

        rr1 >= profile["rr1"]

    )

    checks["rr2"] = (

        np.isfinite(rr2)

        and

        rr2 >= profile["rr2"]

    )

    checks["quality"] = (

        np.isfinite(quality)

        and

        quality >= profile["quality"]

    )

    checks["trend"] = (

        np.isfinite(trend)

        and

        trend > profile["trend"]

    )

    checks["rsi"] = (

        np.isfinite(rsi)

        and

        rsi < profile["max_rsi"]

    )

    checks["regime"] = (

        regime_label
        != "UNFAVORABLE"

    )

    passed = all(
        checks.values()
    )

    failed = [

        key

        for key, ok
        in checks.items()

        if not ok

    ]

    return (
        passed,
        checks,
        failed
    )


# ============================================================
# FORWARD PERFORMANCE
# ============================================================

def forward_result(
    df,
    signal_date,
    horizon
):

    dates = sorted(

        {
            normalize_date(x)
            for x in df.index
        }

    )

    signal_date = (
        normalize_date(
            signal_date
        )
    )

    future = [

        d

        for d in dates

        if d > signal_date

    ]

    if len(future) < horizon:

        return None

    entry_date = future[0]

    exit_date = future[
        horizon - 1
    ]

    entry_open = finite_float(

        df.loc[
            entry_date,
            "Open"
        ]

    )

    if not np.isfinite(
        entry_open
    ):

        entry_open = finite_float(

            df.loc[
                entry_date,
                "Close"
            ]

        )

    exit_close = finite_float(

        df.loc[
            exit_date,
            "Close"
        ]

    )

    if (

        not np.isfinite(
            entry_open
        )

        or

        not np.isfinite(
            exit_close
        )

        or

        entry_open <= 0

    ):

        return None

    return {

        "entry_date":
            entry_date.date().isoformat(),

        "exit_date":
            exit_date.date().isoformat(),

        "entry":
            entry_open,

        "exit":
            exit_close,

        "return":
            (
                exit_close
                / entry_open
                - 1.0
            )

    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 58)
    print(
        "MARKET ALERT V6.3.4 "
        "DIAGNOSTIC BACKTEST"
    )
    print("=" * 58)
    print("")

    # --------------------------------------------------------
    # SYMBOL LIST
    # --------------------------------------------------------

    symbols = list(
        getattr(
            engine,
            "SYMBOLS",
            []
        )
    )

    if not symbols:

        symbols = list(
            getattr(
                engine,
                "TICKERS",
                []
            )
        )

        symbols = [

            x

            for x in symbols

            if x not in [

                getattr(
                    engine,
                    "NIFTY",
                    None
                ),

                getattr(
                    engine,
                    "VIX",
                    None
                ),

            ]

        ]

    max_stocks = int(

        getattr(

            engine,

            "MAX_STOCKS",

            len(symbols)

        )

    )

    symbols = symbols[
        :max_stocks
    ]

    print(
        f"Symbols tested: "
        f"{len(symbols)}"
    )

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

    tickers = list(
        dict.fromkeys(

            [

                x + ".NS"

                for x in symbols

            ]

            +

            [

                engine.NIFTY,

                engine.VIX,

            ]

        )
    )

    print(
        f"Downloading "
        f"{len(tickers)} symbols..."
    )

    raw = yf.download(

        tickers,

        start="2025-08-01",

        end=END_DATE,

        interval="1d",

        auto_adjust=True,

        group_by="column",

        progress=False,

        threads=True,

    )

    frames = (
        split_yf_download(
            raw
        )
    )

    if not frames:

        raise RuntimeError(
            "No historical "
            "data downloaded."
        )

    nifty_df = get_frame(

        frames,

        engine.NIFTY

    )

    if nifty_df is None:

        raise RuntimeError(
            "NIFTY data unavailable."
        )

    # --------------------------------------------------------
    # TEST DATES
    # --------------------------------------------------------

    dates = sorted(

        {

            normalize_date(x)

            for x in nifty_df.index

            if (

                pd.Timestamp(
                    START_DATE
                )

                <= normalize_date(x)

                < pd.Timestamp(
                    END_DATE
                )

            )

        }

    )

    print(
        f"Historical test dates: "
        f"{len(dates)}"
    )

    print("")

    records = []

    # ========================================================
    # HISTORICAL WALK FORWARD
    # ========================================================

    for signal_date in dates:

        regime = (
            historical_regime(
                frames,
                signal_date
            )
        )

        print(

            f"{signal_date.date()} | "
            f"Regime={regime['label']}"

        )

        for symbol in symbols:

            row = build_candidate(

                frames,

                symbol,

                signal_date,

                regime["label"]

            )

            if row is None:
                continue

            record = {

                "signal_date":
                    signal_date.date().isoformat(),

                "regime":
                    regime["label"],

                "nifty":
                    regime["nifty"],

                "sma50":
                    regime["sma50"],

                "vix":
                    regime["vix"],

                **row,

            }

            if "data_error" in row:

                record[
                    "strict_action"
                ] = "DATA_ERROR"

                records.append(
                    record
                )

                continue

            # ------------------------------------------------
            # TEST EVERY THRESHOLD PROFILE
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

                    regime["label"],

                    profile

                )

                record[
                    f"{profile_name}_trade"
                ] = bool(
                    passed
                )

                record[
                    f"{profile_name}_failed"
                ] = (

                    ",".join(
                        failed
                    )

                    if failed

                    else ""

                )

                for (
                    check_name,
                    check_value
                ) in checks.items():

                    record[
                        f"{profile_name}_{check_name}"
                    ] = bool(
                        check_value
                    )

            # ------------------------------------------------
            # CURRENT STRICT ACTION
            # ------------------------------------------------

            strict_pass = bool(

                record[
                    "STRICT_V6_3_2_trade"
                ]

            )

            balanced_pass = bool(

                record[
                    "BALANCED_trade"
                ]

            )

            moderate_pass = bool(

                record[
                    "MODERATE_trade"
                ]

            )

            if strict_pass:

                strict_action = (
                    "TRADE"
                )

            elif balanced_pass:

                strict_action = (
                    "WATCH_BALANCED"
                )

            elif moderate_pass:

                strict_action = (
                    "WATCH_MODERATE"
                )

            else:

                strict_action = (
                    "REJECT"
                )

            record[
                "strict_action"
            ] = strict_action

            # ------------------------------------------------
            # FORWARD OUTCOMES
            #
            # IMPORTANT:
            # We calculate outcomes for EVERY candidate,
            # including rejected candidates.
            #
            # This lets us discover whether filters are
            # rejecting stocks that actually perform well.
            # ------------------------------------------------

            stock_df = get_frame(

                frames,

                symbol + ".NS"

            )

            if stock_df is not None:

                for horizon in HORIZONS:

                    outcome = (
                        forward_result(

                            stock_df,

                            signal_date,

                            horizon

                        )
                    )

                    if outcome is None:

                        continue

                    record[
                        f"return_{horizon}d"
                    ] = outcome[
                        "return"
                    ]

                    record[
                        f"entry_{horizon}d"
                    ] = outcome[
                        "entry"
                    ]

                    record[
                        f"exit_{horizon}d"
                    ] = outcome[
                        "exit"
                    ]

                    record[
                        f"entry_date_{horizon}d"
                    ] = outcome[
                        "entry_date"
                    ]

                    record[
                        f"exit_date_{horizon}d"
                    ] = outcome[
                        "exit_date"
                    ]

            records.append(
                record
            )

    audit = pd.DataFrame(
        records
    )

    if audit.empty:

        raise RuntimeError(
            "Diagnostic produced "
            "no records."
        )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    # ========================================================
    # FULL AUDIT
    # ========================================================

    audit_file = (

        AUDIT_DIR

        /

        (
            "diagnostic_v6_3_4_"
            + timestamp
            + ".csv"
        )

    )

    audit.to_csv(

        audit_file,

        index=False

    )

    # ========================================================
    # FILTER FAILURE COUNTS
    # ========================================================

    strict = audit[
        audit["strict_action"]
        != "DATA_ERROR"
    ].copy()

    failure_rows = []

    for field in [

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

        column = (

            "STRICT_V6_3_2_"

            + field

        )

        if column not in strict.columns:

            continue

        failed_count = int(

            (
                ~strict[column]
            ).sum()

        )

        passed_count = int(

            strict[column]
            .sum()

        )

        failure_rows.append({

            "filter":
                field,

            "passed":
                passed_count,

            "failed":
                failed_count,

        })

    failure_df = pd.DataFrame(
        failure_rows
    )

    failure_file = (

        AUDIT_DIR

        /

        (
            "filter_failures_v6_3_4_"
            + timestamp
            + ".csv"
        )

    )

    failure_df.to_csv(

        failure_file,

        index=False

    )

    # ========================================================
    # PROFILE PERFORMANCE
    # ========================================================

    profile_summary = []

    for profile_name in PROFILES:

        trade_column = (
            f"{profile_name}_trade"
        )

        candidates = audit[
            audit[trade_column]
            == True
        ].copy()

        for horizon in HORIZONS:

            return_column = (
                f"return_{horizon}d"
            )

            if (
                return_column
                not in candidates.columns
            ):

                continue

            returns = pd.to_numeric(

                candidates[
                    return_column
                ],

                errors="coerce"

            ).dropna()

            if returns.empty:

                continue

            winners = returns[
                returns > 0
            ]

            losers = returns[
                returns <= 0
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
                    / gross_loss

                )

            else:

                profit_factor = np.inf

            profile_summary.append({

                "profile":
                    profile_name,

                "horizon":
                    horizon,

                "signals":
                    len(returns),

                "win_rate":
                    float(
                        (
                            returns > 0
                        ).mean()
                    ),

                "average_return":
                    float(
                        returns.mean()
                    ),

                "median_return":
                    float(
                        returns.median()
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
                        returns.max()
                    ),

                "worst":
                    float(
                        returns.min()
                    ),

            })

    profile_df = pd.DataFrame(
        profile_summary
    )

    profile_file = (

        AUDIT_DIR

        /

        (
            "profile_comparison_v6_3_4_"
            + timestamp
            + ".csv"
        )

    )

    profile_df.to_csv(

        profile_file,

        index=False

    )

    # ========================================================
    # GROUP PERFORMANCE
    # ========================================================

    group_rows = []

    groups = {

        "STRICT_TRADE":
            audit[
                audit[
                    "strict_action"
                ]
                == "TRADE"
            ],

        "BALANCED_ONLY":
            audit[
                audit[
                    "strict_action"
                ]
                == "WATCH_BALANCED"
            ],

        "MODERATE_ONLY":
            audit[
                audit[
                    "strict_action"
                ]
                == "WATCH_MODERATE"
            ],

        "REJECT":
            audit[
                audit[
                    "strict_action"
                ]
                == "REJECT"
            ],

    }

    for (
        group_name,
        group
    ) in groups.items():

        for horizon in HORIZONS:

            column = (
                f"return_{horizon}d"
            )

            if column not in group.columns:

                continue

            returns = pd.to_numeric(

                group[column],

                errors="coerce"

            ).dropna()

            if returns.empty:

                continue

            group_rows.append({

                "group":
                    group_name,

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

                "average_return":
                    float(
                        returns.mean()
                    ),

                "median_return":
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

    group_df = pd.DataFrame(
        group_rows
    )

    group_file = (

        AUDIT_DIR

        /

        (
            "group_performance_v6_3_4_"
            + timestamp
            + ".csv"
        )

    )

    group_df.to_csv(

        group_file,

        index=False

    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print("")
    print("=" * 58)
    print(
        "V6.3.4 DIAGNOSTIC SUMMARY"
    )
    print("=" * 58)

    print(
        f"Total candidate observations: "
        f"{len(strict)}"
    )

    print("")
    print(
        "STRICT V6.3.2 ACTION COUNTS:"
    )

    print(

        strict[
            "strict_action"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()

    )

    print("")
    print(
        "FILTER FAILURES:"
    )

    if not failure_df.empty:

        print(

            failure_df.to_string(
                index=False
            )

        )

    else:

        print(
            "No filter diagnostics."
        )

    print("")
    print(
        "PROFILE COMPARISON:"
    )

    if not profile_df.empty:

        print(

            profile_df.to_string(
                index=False
            )

        )

    else:

        print(
            "No profile completed outcomes."
        )

    print("")
    print(
        "REJECT / WATCH / TRADE OUTCOMES:"
    )

    if not group_df.empty:

        print(

            group_df.to_string(
                index=False
            )

        )

    else:

        print(
            "No completed forward outcomes."
        )

    print("")
    print("=" * 58)
    print(
        "FILES CREATED"
    )
    print("=" * 58)

    print(
        audit_file
    )

    print(
        failure_file
    )

    print(
        profile_file
    )

    print(
        group_file
    )

    print("")
    print(
        "V6.3.4 completed successfully."
    )

    print(
        "Do NOT change the live V6.3.2 "
        "engine based on this run alone."
    )


if __name__ == "__main__":

    main()
