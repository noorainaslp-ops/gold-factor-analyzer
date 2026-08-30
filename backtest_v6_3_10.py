"""
V6.3.10 WALK-FORWARD BACKTEST

Important:
- Signal is generated using information available at signal close.
- Entry occurs at NEXT trading day's OPEN.
- Slippage is applied.
- Round-trip costs are applied.
- Stop/target are checked using future OHLC.
- Probability calibration is updated only AFTER the future outcome.
- This prevents future-label leakage into earlier predictions.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from market_engine_v6_3_10 import (
    UNIVERSE,
    add_features,
    calculate_raw_probability,
    calculate_expected_returns,
    calculate_risk_levels,
    SLIPPAGE_BPS,
    ROUND_TRIP_COST_BPS,
)


VERSION = "V6.3.10"

MIN_HISTORY = 220

HORIZONS = (
    1,
    3,
    5
)


# ============================================================
# DATA
# ============================================================

def download_data(
    ticker,
    period="6y"
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

            columns = []

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

                        columns.append(
                            column[0]
                        )

                    else:

                        columns.append(
                            column[-1]
                        )

                else:

                    columns.append(
                        column
                    )

            df.columns = columns

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
            f"{ticker}: {error}"
        )

        return pd.DataFrame()


# ============================================================
# REGIME HISTORY
# ============================================================

def build_regime_history(
    index_features
):

    regimes = {}

    for date, row in (
        index_features.iterrows()
    ):

        if pd.isna(
            row["sma50"]
        ):

            regimes[
                pd.Timestamp(
                    date
                ).date()
            ] = "UNKNOWN"

            continue

        history = (
            index_features
            .loc[:date]
        )

        if len(history) > 6:

            slope = (
                history["sma50"]
                .diff(5)
                .iloc[-1]
            )

        else:

            slope = np.nan

        distance = (
            row["Close"]
            /
            row["sma50"]
            - 1
        )

        if (
            distance > 0.008
            and
            pd.notna(slope)
            and
            slope > 0
        ):

            regime = "FAVORABLE"

        elif (
            distance < -0.008
            and
            pd.notna(slope)
            and
            slope < 0
        ):

            regime = "UNFAVORABLE"

        else:

            regime = "MIXED"

        regimes[
            pd.Timestamp(
                date
            ).date()
        ] = regime

    return regimes


# ============================================================
# ONLINE CALIBRATOR
# ============================================================

class OnlineCalibrator:

    """
    Calibration uses only outcomes that were already known.

    A signal NEVER gets to use its own future outcome
    to determine its probability.
    """

    def __init__(self):

        self.edges = np.array([
            0.00,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            1.01
        ])

        self.observations = np.zeros(
            9,
            dtype=float
        )

        self.wins = np.zeros(
            9,
            dtype=float
        )

    def bucket(
        self,
        probability
    ):

        return int(
            np.clip(
                np.digitize(
                    [probability],
                    self.edges,
                    right=False
                )[0] - 1,
                0,
                8
            )
        )

    def transform(
        self,
        probability
    ):

        bucket = self.bucket(
            probability
        )

        n = (
            self.observations[
                bucket
            ]
        )

        wins = (
            self.wins[
                bucket
            ]
        )

        # Small-sample protection.
        if n < 25:

            return float(
                np.clip(
                    0.50
                    +
                    0.55
                    * (
                        probability
                        - 0.50
                    ),
                    0.35,
                    0.65
                )
            )

        empirical = (
            wins
            +
            8 * probability
        ) / (
            n + 8
        )

        return float(
            np.clip(
                0.50
                +
                0.92
                * (
                    empirical
                    - 0.50
                ),
                0.35,
                0.70
            )
        )

    def update(
        self,
        probability,
        outcome
    ):

        bucket = self.bucket(
            probability
        )

        self.observations[
            bucket
        ] += 1

        self.wins[
            bucket
        ] += float(
            outcome
        )


# ============================================================
# TRANSACTION COST
# ============================================================

def transaction_cost_fraction():

    return (
        2 * SLIPPAGE_BPS
        +
        ROUND_TRIP_COST_BPS
    ) / 10000.0


# ============================================================
# REALISTIC TRADE TEST
# ============================================================

def simulate_trade(
    df,
    signal_index,
    stop_loss,
    target1,
    horizon
):

    # --------------------------------------------------------
    # IMPORTANT:
    # Entry is NEXT trading day's OPEN.
    # --------------------------------------------------------

    if (
        signal_index + 1
        >= len(df)
    ):

        return None

    if (
        signal_index + horizon
        >= len(df)
    ):

        return None

    entry = (
        float(
            df["Open"].iloc[
                signal_index + 1
            ]
        )
        *
        (
            1
            +
            SLIPPAGE_BPS
            /
            10000
        )
    )

    exit_price = (
        float(
            df["Close"].iloc[
                signal_index + horizon
            ]
        )
        *
        (
            1
            -
            SLIPPAGE_BPS
            /
            10000
        )
    )

    stop_hit = False
    target_hit = False

    # --------------------------------------------------------
    # Check future sessions.
    #
    # Conservative rule:
    # If both stop and target are touched in one candle,
    # stop is assumed to have occurred first.
    # --------------------------------------------------------

    for future_index in range(

        signal_index + 1,

        min(
            signal_index
            + horizon
            + 1,
            len(df)
        )

    ):

        low = float(
            df["Low"].iloc[
                future_index
            ]
        )

        high = float(
            df["High"].iloc[
                future_index
            ]
        )

        if low <= stop_loss:

            stop_hit = True

            exit_price = (
                stop_loss
                *
                (
                    1
                    -
                    SLIPPAGE_BPS
                    /
                    10000
                )
            )

            break

        if high >= target1:

            target_hit = True

            exit_price = (
                target1
                *
                (
                    1
                    -
                    SLIPPAGE_BPS
                    /
                    10000
                )
            )

            break

    net_return = (
        (
            exit_price
            -
            entry
        )
        /
        entry
        -
        transaction_cost_fraction()
    )

    return (
        float(net_return),
        stop_hit,
        target_hit
    )


# ============================================================
# MAIN BACKTEST
# ============================================================

def main():

    Path(
        "audit"
    ).mkdir(
        exist_ok=True
    )

    print(
        f"Starting {VERSION} "
        "walk-forward backtest..."
    )

    # --------------------------------------------------------
    # MARKET INDEX
    # --------------------------------------------------------

    index = download_data(
        "^NSEI",
        "6y"
    )

    if index.empty:

        raise RuntimeError(
            "Unable to download ^NSEI"
        )

    index_features = add_features(
        index
    )

    regimes = build_regime_history(
        index_features
    )

    calibrator = (
        OnlineCalibrator()
    )

    observations = []

    # ========================================================
    # STOCK LOOP
    # ========================================================

    for number, ticker in enumerate(
        UNIVERSE,
        start=1
    ):

        print(
            f"[{number}/{len(UNIVERSE)}] "
            f"{ticker}"
        )

        df = download_data(
            ticker,
            "6y"
        )

        if (
            df.empty
            or
            len(df) <
            MIN_HISTORY + 10
        ):

            continue

        features = add_features(
            df
        )

        if features.empty:

            continue

        # ----------------------------------------------------
        # SIGNAL DAYS
        # ----------------------------------------------------

        for i in range(
            MIN_HISTORY,
            len(features) - 6
        ):

            row = features.iloc[i]

            date = (
                pd.Timestamp(
                    features.index[i]
                ).date()
            )

            regime = regimes.get(
                date,
                "UNKNOWN"
            )

            if regime == "UNKNOWN":

                continue

            required = [
                "atr",
                "rsi",
                "sma50",
                "vol_ratio",
                "trend20",
                "trend50"
            ]

            if any(
                pd.isna(
                    row[column]
                )
                for column
                in required
            ):

                continue

            # ------------------------------------------------
            # RAW MODEL
            # ------------------------------------------------

            raw_probability = (
                calculate_raw_probability(
                    row,
                    regime
                )
            )

            # ------------------------------------------------
            # PAST-ONLY CALIBRATION
            # ------------------------------------------------

            probability = (
                calibrator.transform(
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
                stop_loss,
                target1,
                target2,
                rr1,
                rr2,
                risk
            ) = calculate_risk_levels(
                row
            )

            # ------------------------------------------------
            # FILTERS
            # ------------------------------------------------

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

            quality_ok = (
                float(
                    row["quality"]
                ) >= 0.33
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

            volume_ok = (
                0.75
                <=
                float(
                    row["vol_ratio"]
                )
                <=
                3.0
            )

            # Regime is deliberately NOT a hard filter.
            regime_ok = True

            # ------------------------------------------------
            # CLASSIFICATION
            # ------------------------------------------------

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
                    volume_ok
                ]
            ):

                action = "TRADE"

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

            # ------------------------------------------------
            # FUTURE OUTCOME
            #
            # Use 5D result for updating calibration only AFTER
            # the signal's future outcome is known.
            # ------------------------------------------------

            five_day = simulate_trade(
                df,
                i,
                stop_loss,
                target1,
                5
            )

            if five_day is None:

                continue

            net5, stop_hit, target_hit = (
                five_day
            )

            # ------------------------------------------------
            # UPDATE CALIBRATOR AFTER OUTCOME
            # ------------------------------------------------

            outcome = (
                1
                if net5 > 0
                else 0
            )

            calibrator.update(
                raw_probability,
                outcome
            )

            # ------------------------------------------------
            # SIMPLE HORIZON RETURNS
            #
            # For clean comparison, these use next-open entry
            # and exit at the future close, after costs.
            # ------------------------------------------------

            future_returns = {}

            for horizon in (
                1,
                3,
                5
            ):

                if (
                    i + horizon
                    >= len(df)
                ):

                    future_returns[
                        horizon
                    ] = np.nan

                    continue

                entry = (
                    float(
                        df["Open"].iloc[
                            i + 1
                        ]
                    )
                    *
                    (
                        1
                        +
                        SLIPPAGE_BPS
                        /
                        10000
                    )
                )

                exit_price = (
                    float(
                        df["Close"].iloc[
                            i + horizon
                        ]
                    )
                    *
                    (
                        1
                        -
                        SLIPPAGE_BPS
                        /
                        10000
                    )
                )

                future_returns[
                    horizon
                ] = (

                    (
                        exit_price
                        -
                        entry
                    )
                    /
                    entry

                    -
                    transaction_cost_fraction()
                )

            observations.append(

                {

                    "date":
                        str(
                            features.index[
                                i
                            ].date()
                        ),

                    "ticker":
                        ticker.replace(
                            ".NS",
                            ""
                        ),

                    "regime":
                        regime,

                    "raw_probability":
                        raw_probability,

                    "calibrated_probability":
                        probability,

                    "er1":
                        er1,

                    "er3":
                        er3,

                    "er5":
                        er5,

                    "rr1":
                        rr1,

                    "rr2":
                        rr2,

                    "rsi":
                        float(
                            row["rsi"]
                        ),

                    "volume":
                        float(
                            row["vol_ratio"]
                        ),

                    "action":
                        action,

                    "ret1":
                        future_returns[
                            1
                        ],

                    "ret3":
                        future_returns[
                            3
                        ],

                    "ret5":
                        future_returns[
                            5
                        ],

                    "stop_hit_5d":
                        stop_hit,

                    "target_hit_5d":
                        target_hit,
                }
            )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = pd.DataFrame(
        observations
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    if output.empty:

        print(
            "No observations generated."
        )

        return

    raw_file = (
        Path("audit")
        /
        f"walkforward_v6_3_10_"
        f"{timestamp}.csv"
    )

    output.to_csv(
        raw_file,
        index=False
    )

    print("")
    print(
        "=" * 70
    )

    print(
        "V6.3.10 WALK-FORWARD BACKTEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Total candidate observations: "
        f"{len(output)}"
    )

    # ========================================================
    # ACTION COUNTS
    # ========================================================

    print("")
    print(
        "ACTION COUNTS:"
    )

    print(
        output[
            "action"
        ].value_counts().to_string()
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    summaries = []

    for selection in [
        "TRADE",
        "WATCH",
        "WAIT",
        "ALL"
    ]:

        if selection == "ALL":

            subset = output

        else:

            subset = output[
                output["action"]
                ==
                selection
            ]

        for horizon in [
            1,
            3,
            5
        ]:

            series = (
                subset[
                    f"ret{horizon}"
                ]
                .dropna()
            )

            if series.empty:

                continue

            winners = (
                series[
                    series > 0
                ]
            )

            losers = (
                series[
                    series <= 0
                ]
            )

            if (
                not losers.empty
                and
                losers.sum() != 0
            ):

                profit_factor = (
                    winners.sum()
                    /
                    abs(
                        losers.sum()
                    )
                )

            else:

                profit_factor = np.nan

            summaries.append(

                {

                    "selection":
                        selection,

                    "horizon":
                        horizon,

                    "observations":
                        len(series),

                    "win_rate":
                        float(
                            (
                                series > 0
                            ).mean()
                        ),

                    "average_net_return":
                        float(
                            series.mean()
                        ),

                    "median_net_return":
                        float(
                            series.median()
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
                        profit_factor,

                    "best":
                        float(
                            series.max()
                        ),

                    "worst":
                        float(
                            series.min()
                        ),
                }
            )

    summary = pd.DataFrame(
        summaries
    )

    print("")
    print(
        "PERFORMANCE:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary_file = (
        Path("audit")
        /
        f"walkforward_summary_"
        f"v6_3_10_"
        f"{timestamp}.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # PROBABILITY CALIBRATION
    # ========================================================

    output["probability_bucket"] = (
        pd.cut(
            output[
                "calibrated_probability"
            ],
            [
                0,
                0.40,
                0.45,
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                1.01
            ],
            right=False,
            labels=[
                "<40%",
                "40-45%",
                "45-50%",
                "50-55%",
                "55-60%",
                "60-65%",
                "65-70%",
                "70-75%",
                "75%+"
            ]
        )
    )

    calibration = (
        output
        .groupby(
            "probability_bucket",
            observed=False
        )
        .agg(

            observations=(
                "ret5",
                "size"
            ),

            average_model_probability=(
                "calibrated_probability",
                "mean"
            ),

            actual_win_rate=(
                "ret5",
                lambda x:
                (
                    x > 0
                ).mean()
            ),

            average_return=(
                "ret5",
                "mean"
            )
        )
        .reset_index()
    )

    print("")
    print(
        "PROBABILITY CALIBRATION:"
    )

    print(
        calibration.to_string(
            index=False
        )
    )

    calibration_file = (
        Path("audit")
        /
        f"probability_calibration_"
        f"v6_3_10_"
        f"{timestamp}.csv"
    )

    calibration.to_csv(
        calibration_file,
        index=False
    )

    # ========================================================
    # STOCK-LEVEL RESULTS
    # ========================================================

    trade_data = output[
        output["action"]
        ==
        "TRADE"
    ]

    if not trade_data.empty:

        symbols = (
            trade_data
            .groupby(
                "ticker"
            )
            .agg(

                observations=(
                    "ret5",
                    "size"
                ),

                win_rate=(
                    "ret5",
                    lambda x:
                    (
                        x > 0
                    ).mean()
                ),

                average_return=(
                    "ret5",
                    "mean"
                )
            )
            .reset_index()
        )

        # IMPORTANT:
        # Only consider symbols with >=20 observations.
        symbols = symbols[
            symbols[
                "observations"
            ] >= 20
        ]

        symbols = symbols.sort_values(
            [
                "average_return",
                "win_rate"
            ],
            ascending=False
        )

        print("")
        print(
            "TRADE SYMBOLS "
            "WITH >=20 OBSERVATIONS:"
        )

        print(
            symbols
            .head(25)
            .to_string(
                index=False
            )
        )

        symbol_file = (
            Path("audit")
            /
            f"trade_symbol_performance_"
            f"v6_3_10_"
            f"{timestamp}.csv"
        )

        symbols.to_csv(
            symbol_file,
            index=False
        )

    print("")
    print(
        "=" * 70
    )

    print(
        "V6.3.10 BACKTEST COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT promote V6.3.10 to "
        "real-money trading solely from "
        "this backtest."
    )

    print(
        "A separate untouched "
        "out-of-sample period should "
        "be used before live deployment."
    )


if __name__ == "__main__":

    main()
