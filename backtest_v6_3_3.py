import os
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import market_engine_v6_3_2 as engine


# ============================================================
# MARKET ALERT V6.3.3
# WALK-FORWARD BACKTEST ENGINE
# ============================================================
#
# Purpose:
# Validate the V6.3.2 short-term model historically.
#
# IMPORTANT:
# This is NOT a live trading engine.
#
# The model is reconstructed separately for each historical
# date using only information available up to that date.
#
# Signal:
#     Generated after historical day's close
#
# Hypothetical entry:
#     Next trading day's OPEN
#
# Exit:
#     Close after 1 / 3 / 5 trading sessions
#
# This helps reduce look-ahead bias.
# ============================================================


START_DATE = "2026-08-10"
END_DATE = "2026-08-29"

HORIZONS = [1, 3, 5]

AUDIT_DIR = Path("audit")

AUDIT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# YFINANCE DATA HANDLING
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
# GET CLEAN DATAFRAME
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
        "Volume"

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
# MARKET REGIME AS OF HISTORICAL DATE
# ============================================================

def historical_market_regime(
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

    nifty = float(
        close.iloc[-1]
    )

    sma50_value = float(
        sma50.iloc[-1]
    )

    previous_sma50 = np.nan

    if len(sma50) >= 6:

        candidate = (
            sma50.iloc[-6]
        )

        if np.isfinite(
            candidate
        ):

            previous_sma50 = float(
                candidate
            )

    vix = np.nan

    if vix_df is not None:

        vix_df = vix_df.loc[
            vix_df.index <= signal_date
        ]

        if not vix_df.empty:

            vix = float(
                vix_df[
                    "Close"
                ].iloc[-1]
            )

    above_sma = (

        np.isfinite(nifty)

        and np.isfinite(
            sma50_value
        )

        and nifty > sma50_value

    )

    rising_sma = (

        np.isfinite(
            sma50_value
        )

        and np.isfinite(
            previous_sma50
        )

        and sma50_value
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

        "sma50": sma50_value,

        "vix": vix

    }


# ============================================================
# BUILD HISTORICAL CANDIDATE
# ============================================================

def build_historical_candidate(
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

    features = (
        engine.make_features(
            historical
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
                "ret5"

            ]
        )
    )

    if len(features) < 220:

        return None

    model = (
        engine.empirical_analog_model(
            features
        )
    )

    if model is None:

        return None

    last = (
        features.iloc[-1]
    )

    price = float(
        last["Close"]
    )

    atr_pct = float(
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

        return None

    row = {

        "symbol": symbol,

        "price": price,

        "rsi": float(
            last["rsi"]
        ),

        "vol_ratio": float(
            last["vol_ratio"]
        ),

        "atr_pct": atr_pct,

        "trend": float(
            last["dist_sma50"]
        ),

        **model

    }

    row["score"] = (
        engine.score_candidate(
            row,
            regime_label
        )
    )

    # --------------------------------------------------------
    # STOP LOSS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TARGET 1
    # --------------------------------------------------------

    expected_3 = max(

        row["er3"],

        row["median3"],

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
                    expected_3
                )
            )

        )

    )

    # --------------------------------------------------------
    # TARGET 2
    # --------------------------------------------------------

    expected_5 = max(

        row["er5"],

        row["median5"],

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

    risk = (
        price
        - stop
    )

    if risk <= 0:

        return None

    rr1 = (
        target1
        - price
    ) / risk

    rr2 = (
        target2
        - price
    ) / risk

    row.update({

        "stop": stop,

        "target1": target1,

        "target2": target2,

        "rr1": rr1,

        "rr2": rr2

    })

    # ========================================================
    # SAME V6.3.2 FILTERS
    # ========================================================

    valid_return = (

        row["er3"] > 0.002

        and

        row["er5"] > 0.004

    )

    valid_probability = (

        row["p3"] >= 0.54

        and

        row["p5"] >= 0.53

    )

    valid_rr = (

        row["rr1"] >= 1.0

        and

        row["rr2"] >= 1.5

    )

    valid_quality = (

        row["quality"] >= 0.35

    )

    valid_trend = (

        row["trend"] > -0.02

    )

    not_extreme = (

        row["rsi"] < 76

    )

    trade = (

        valid_return

        and

        valid_probability

        and

        valid_rr

        and

        valid_quality

        and

        valid_trend

        and

        not_extreme

    )

    if regime_label == "UNFAVORABLE":

        trade = False

    watch = (

        not trade

        and

        row["er3"] > 0

        and

        row["p3"] >= 0.51

        and

        row["p5"] >= 0.50

        and

        row["quality"] >= 0.25

        and

        row["rr2"] >= 1.0

    )

    if trade:

        action = "TRADE"

    elif watch:

        action = "WATCH"

    else:

        action = "REJECT"

    row["action"] = action

    return row


# ============================================================
# FORWARD TEST
# ============================================================

def get_forward_result(
    df,
    signal_date,
    horizon
):

    dates = [

        normalize_date(
            value
        )

        for value in df.index

    ]

    dates = sorted(
        set(dates)
    )

    signal_date = normalize_date(
        signal_date
    )

    future = [

        value

        for value in dates

        if value > signal_date

    ]

    if len(future) < horizon:

        return None

    entry_date = future[0]

    exit_date = future[
        horizon - 1
    ]

    entry_open = df.loc[
        entry_date,
        "Open"
    ]

    if np.isfinite(
        entry_open
    ):

        entry = float(
            entry_open
        )

    else:

        entry = float(
            df.loc[
                entry_date,
                "Close"
            ]
        )

    exit_price = float(
        df.loc[
            exit_date,
            "Close"
        ]
    )

    return {

        "entry_date":
            entry_date.date().isoformat(),

        "exit_date":
            exit_date.date().isoformat(),

        "entry":
            entry,

        "exit":
            exit_price,

        "return":
            (
                exit_price
                / entry
                - 1
            )

    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def main():

    print("")
    print(
        "================================================"
    )
    print(
        "MARKET ALERT V6.3.3 WALK-FORWARD BACKTEST"
    )
    print(
        "================================================"
    )
    print("")

    tickers = (

        engine.TICKERS

        + [

            engine.NIFTY,
            engine.VIX

        ]

    )

    print(
        f"Downloading historical data "
        f"for {len(tickers)} symbols..."
    )

    raw = yf.download(

        tickers,

        start="2025-08-01",

        end=END_DATE,

        interval="1d",

        auto_adjust=True,

        group_by="column",

        progress=False,

        threads=True

    )

    frames = split_yf_download(
        raw
    )

    if not frames:

        raise RuntimeError(
            "No historical market data "
            "was downloaded."
        )

    nifty_frame = get_frame(
        frames,
        engine.NIFTY
    )

    if nifty_frame is None:

        raise RuntimeError(
            "NIFTY data unavailable."
        )

    available_dates = [

        normalize_date(
            value
        )

        for value in nifty_frame.index

    ]

    start = pd.Timestamp(
        START_DATE
    )

    end = pd.Timestamp(
        END_DATE
    )

    test_dates = sorted(

        {

            value

            for value in available_dates

            if (
                start
                <= value
                < end
            )

        }

    )

    print("")
    print(
        f"Validation dates: "
        f"{len(test_dates)}"
    )
    print("")

    all_records = []

    for signal_date in test_dates:

        print(
            f"Testing "
            f"{signal_date.date()}..."
        )

        regime = (
            historical_market_regime(
                frames,
                signal_date
            )
        )

        candidates = []

        for symbol in (
            engine.SYMBOLS[
                :engine.MAX_STOCKS
            ]
        ):

            candidate = (
                build_historical_candidate(

                    frames,

                    symbol,

                    signal_date,

                    regime["label"]

                )
            )

            if candidate is not None:

                candidates.append(
                    candidate
                )

        if not candidates:

            continue

        day_df = pd.DataFrame(
            candidates
        )

        priority = {

            "TRADE": 0,
            "WATCH": 1,
            "REJECT": 2

        }

        day_df["priority"] = (
            day_df["action"]
            .map(priority)
        )

        day_df = (
            day_df.sort_values(

                [
                    "priority",
                    "score"

                ],

                ascending=[

                    True,
                    False

                ]

            )
        )

        # Save the top 10 candidates
        # for each historical day.

        for _, row in (
            day_df.head(10)
            .iterrows()
        ):

            record = (
                row.to_dict()
            )

            record.update({

                "signal_date":
                    signal_date.date().isoformat(),

                "regime":
                    regime["label"],

                "nifty":
                    regime["nifty"],

                "sma50":
                    regime["sma50"],

                "vix":
                    regime["vix"]

            })

            stock_df = get_frame(

                frames,

                row["symbol"]
                + ".NS"

            )

            # Only actual TRADE signals
            # are included in the main
            # profitability calculation.

            if (

                row["action"]
                == "TRADE"

                and

                stock_df is not None

            ):

                for horizon in HORIZONS:

                    outcome = (
                        get_forward_result(

                            stock_df,

                            signal_date,

                            horizon

                        )
                    )

                    if outcome is None:

                        continue

                    record[
                        f"entry_{horizon}d"
                    ] = outcome["entry"]

                    record[
                        f"exit_{horizon}d"
                    ] = outcome["exit"]

                    record[
                        f"return_{horizon}d"
                    ] = outcome["return"]

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

            all_records.append(
                record
            )

    audit = pd.DataFrame(
        all_records
    )

    if audit.empty:

        print("")
        print(
            "No candidates were generated."
        )
        return

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    audit_file = (

        AUDIT_DIR
        /

        (
            "walkforward_v6_3_3_"
            + timestamp
            + ".csv"
        )

    )

    audit.to_csv(
        audit_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    trades = audit[
        audit["action"]
        == "TRADE"
    ].copy()

    watches = audit[
        audit["action"]
        == "WATCH"
    ].copy()

    rejects = audit[
        audit["action"]
        == "REJECT"
    ].copy()

    print("")
    print(
        "================================================"
    )
    print(
        "V6.3.3 BACKTEST SUMMARY"
    )
    print(
        "================================================"
    )

    print(
        f"TRADE signals : "
        f"{len(trades)}"
    )

    print(
        f"WATCH signals : "
        f"{len(watches)}"
    )

    print(
        f"REJECT signals: "
        f"{len(rejects)}"
    )

    summary_rows = []

    for horizon in HORIZONS:

        column = (
            f"return_{horizon}d"
        )

        if column not in trades.columns:

            continue

        returns = pd.to_numeric(

            trades[column],

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

        summary_rows.append({

            "horizon_sessions":
                horizon,

            "completed_trades":
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

            "best_trade":
                float(
                    returns.max()
                ),

            "worst_trade":
                float(
                    returns.min()
                )

        })

    summary = pd.DataFrame(
        summary_rows
    )

    summary_file = (

        AUDIT_DIR
        /

        (
            "walkforward_summary_v6_3_3_"
            + timestamp
            + ".csv"
        )

    )

    summary.to_csv(

        summary_file,

        index=False

    )

    if summary.empty:

        print("")
        print(
            "No completed TRADE outcomes "
            "were available in the "
            "validation period."
        )

    else:

        print("")
        print(
            summary.to_string(
                index=False
            )
        )

    print("")
    print(
        "================================================"
    )
    print(
        "FILES CREATED"
    )
    print(
        "================================================"
    )

    print(
        audit_file
    )

    print(
        summary_file
    )

    print("")
    print(
        "Backtest completed."
    )

    print("")
    print(
        "IMPORTANT: This is historical "
        "research only. It does not "
        "guarantee future profitability."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
