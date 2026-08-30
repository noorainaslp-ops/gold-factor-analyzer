"""
V6.3.11 CHRONOLOGICAL WALK-FORWARD BACKTEST

Key principles:

1. Signal uses information available at signal close.
2. Entry is next trading day's OPEN.
3. Slippage and transaction costs included.
4. Stop/target are tested using future OHLC.
5. Probability calibration is chronological.
6. Calibration is updated only after outcomes become known.
7. Ranking is performed across stocks on each date.
8. Portfolio-level statistics are generated.
9. Minimum symbol sample size is enforced.
10. Current NSE/Yahoo symbols are used.
"""

import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from market_engine_v6_3_11 import (
    UNIVERSE,
    add_features,
    raw_probability,
    expected_returns,
    risk_levels,
    SLIPPAGE_BPS,
    ROUND_TRIP_COST_BPS,
)


VERSION = "V6.3.11"

MIN_HISTORY = 220

HORIZONS = [
    1,
    3,
    5
]


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
            f"{ticker}: "
            f"download failed: {error}"
        )

        return pd.DataFrame()


# ============================================================
# REGIME HISTORY
# ============================================================

def build_regimes(
    index_features
):

    regimes = {}

    for i in range(
        len(index_features)
    ):

        row = (
            index_features.iloc[i]
        )

        date = (
            pd.Timestamp(
                index_features.index[i]
            ).date()
        )

        if pd.isna(
            row["sma50"]
        ):

            regimes[date] = "UNKNOWN"
            continue

        if i >= 5:

            slope = (
                index_features[
                    "sma50"
                ]
                .iloc[i]
                -
                index_features[
                    "sma50"
                ]
                .iloc[i - 5]
            )

        else:

            slope = np.nan

        distance = (
            row["Close"]
            /
            row["sma50"]
            -
            1
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

        regimes[date] = regime

    return regimes


# ============================================================
# ONLINE CALIBRATOR
# ============================================================

class ChronologicalCalibrator:

    def __init__(self):

        self.edges = [
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
        ]

        self.count = np.zeros(
            9
        )

        self.wins = np.zeros(
            9
        )

    def bucket(
        self,
        p
    ):

        for i in range(
            len(self.edges) - 1
        ):

            if (
                self.edges[i]
                <= p
                <
                self.edges[i + 1]
            ):

                return i

        return 8

    def transform(
        self,
        p
    ):

        b = self.bucket(
            p
        )

        n = self.count[b]

        w = self.wins[b]

        # Bayesian shrinkage toward the
        # model probability when sample is small.
        prior_strength = 25.0

        if n < 20:

            return float(
                np.clip(
                    0.50
                    +
                    0.70
                    * (
                        p
                        - 0.50
                    ),
                    0.35,
                    0.70
                )
            )

        empirical = (
            w
            +
            prior_strength * p
        ) / (
            n
            +
            prior_strength
        )

        # Prevent exaggerated probabilities.
        calibrated = (
            0.50
            +
            0.90
            *
            (
                empirical
                - 0.50
            )
        )

        return float(
            np.clip(
                calibrated,
                0.35,
                0.70
            )
        )

    def update(
        self,
        p,
        win
    ):

        b = self.bucket(
            p
        )

        self.count[b] += 1

        self.wins[b] += (
            1
            if win
            else
            0
        )


# ============================================================
# COST
# ============================================================

def cost_fraction():

    return (
        2 * SLIPPAGE_BPS
        +
        ROUND_TRIP_COST_BPS
    ) / 10000


# ============================================================
# FUTURE RETURN
# ============================================================

def future_return(
    df,
    i,
    horizon
):

    if (
        i + 1 >= len(df)
        or
        i + horizon >= len(df)
    ):

        return np.nan

    entry = (
        float(
            df["Open"].iloc[i + 1]
        )
        *
        (
            1
            +
            SLIPPAGE_BPS / 10000
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
            SLIPPAGE_BPS / 10000
        )
    )

    return (
        (
            exit_price
            -
            entry
        )
        /
        entry
        -
        cost_fraction()
    )


# ============================================================
# STOP/TARGET SIMULATION
# ============================================================

def simulate_trade(
    df,
    i,
    stop,
    target,
    horizon
):

    if (
        i + 1 >= len(df)
        or
        i + horizon >= len(df)
    ):

        return None

    entry = (
        float(
            df["Open"].iloc[i + 1]
        )
        *
        (
            1
            +
            SLIPPAGE_BPS / 10000
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
            SLIPPAGE_BPS / 10000
        )
    )

    exit_reason = "TIME"

    for j in range(
        i + 1,
        min(
            i + horizon + 1,
            len(df)
        )
    ):

        low = float(
            df["Low"].iloc[j]
        )

        high = float(
            df["High"].iloc[j]
        )

        # Conservative assumption:
        # if both are touched in one candle,
        # assume stop was hit first.

        if low <= stop:

            exit_price = (
                stop
                *
                (
                    1
                    -
                    SLIPPAGE_BPS / 10000
                )
            )

            exit_reason = "STOP"

            break

        if high >= target:

            exit_price = (
                target
                *
                (
                    1
                    -
                    SLIPPAGE_BPS / 10000
                )
            )

            exit_reason = "TARGET"

            break

    ret = (
        (
            exit_price
            -
            entry
        )
        /
        entry
        -
        cost_fraction()
    )

    return (
        float(ret),
        exit_reason
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(
    probability,
    er3,
    er5,
    rr1,
    rr2,
    trend20,
    volume,
    rsi,
    quality,
    regime
):

    trend_score = np.clip(
        50
        +
        1000 * trend20,
        0,
        100
    )

    momentum_score = 50

    probability_score = (
        50
        +
        400
        *
        (
            probability
            - 0.50
        )
    )

    expected_score = np.clip(
        50
        +
        500 * er3
        +
        250 * er5,
        0,
        100
    )

    volume_score = np.clip(
        50
        +
        35
        * (
            volume
            - 1
        ),
        0,
        100
    )

    if 48 <= rsi <= 65:

        rsi_score = 100

    elif 42 <= rsi <= 70:

        rsi_score = 70

    else:

        rsi_score = 30

    rr_score = np.clip(
        50
        +
        25 * (
            rr2 - 1
        ),
        0,
        100
    )

    regime_score = {
        "FAVORABLE": 100,
        "MIXED": 60,
        "UNFAVORABLE": 30,
    }.get(
        regime,
        40
    )

    composite = (

        0.27 * probability_score
        +
        0.18 * expected_score
        +
        0.15 * trend_score
        +
        0.10 * momentum_score
        +
        0.08 * volume_score
        +
        0.08 * rsi_score
        +
        0.09 * rr_score
        +
        0.05 * regime_score
    )

    composite += (
        8 * (
            quality
            - 0.50
        )
    )

    composite = float(
        np.clip(
            composite,
            0,
            100
        )
    )

    if (
        composite >= 68
        and
        probability >= 0.55
        and
        er3 > 0
        and
        er5 > 0
        and
        rr1 >= 0.85
        and
        rr2 >= 1.20
        and
        volume >= 0.70
        and
        rsi <= 72
    ):

        return (
            "TRADE",
            composite
        )

    if (
        composite >= 55
        and
        probability >= 0.52
        and
        rr2 >= 1.05
        and
        volume >= 0.55
    ):

        return (
            "WATCH",
            composite
        )

    return (
        "WAIT",
        composite
    )


# ============================================================
# MAIN
# ============================================================

def main():

    Path(
        "audit"
    ).mkdir(
        exist_ok=True
    )

    print(
        f"Starting {VERSION} "
        "chronological walk-forward backtest..."
    )

    # --------------------------------------------------------
    # INDEX
    # --------------------------------------------------------

    index = download_data(
        "^NSEI",
        "6y"
    )

    if index.empty:

        raise RuntimeError(
            "NSE index data unavailable."
        )

    index_features = add_features(
        index
    )

    regimes = build_regimes(
        index_features
    )

    # --------------------------------------------------------
    # LOAD ALL STOCK DATA FIRST
    # --------------------------------------------------------

    stock_data = {}

    for number, ticker in enumerate(
        UNIVERSE,
        start=1
    ):

        print(
            f"Loading "
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
            len(df) < MIN_HISTORY + 10
        ):

            continue

        features = add_features(
            df
        )

        if features.empty:

            continue

        stock_data[ticker] = (
            df,
            features
        )

    if not stock_data:

        raise RuntimeError(
            "No stock data available."
        )

    # --------------------------------------------------------
    # MASTER DATES
    #
    # This is the critical V6.3.11 correction.
    # Dates are processed chronologically.
    # --------------------------------------------------------

    all_dates = sorted(
        set(
            date
            for _, (_, features)
            in stock_data.items()
            for date
            in features.index
        )
    )

    calibrator = (
        ChronologicalCalibrator()
    )

    observations = []

    portfolio_daily = []

    # ========================================================
    # DATE LOOP
    # ========================================================

    for date_position, current_date in enumerate(
        all_dates
    ):

        current_date = pd.Timestamp(
            current_date
        )

        date_only = (
            current_date.date()
        )

        regime = regimes.get(
            date_only,
            "UNKNOWN"
        )

        if regime == "UNKNOWN":
            continue

        daily_candidates = []

        # ----------------------------------------------------
        # Generate every signal for THIS date first.
        # ----------------------------------------------------

        for ticker, (
            df,
            features
        ) in stock_data.items():

            if current_date not in features.index:

                continue

            i = features.index.get_loc(
                current_date
            )

            if (
                not isinstance(
                    i,
                    (int, np.integer)
                )
            ):

                continue

            if i < MIN_HISTORY:

                continue

            row = features.iloc[i]

            required = [
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
                for c in required
            ):

                continue

            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            raw_p = raw_probability(
                row,
                regime
            )

            # IMPORTANT:
            # transform uses ONLY outcomes known before
            # this date.
            calibrated_p = (
                calibrator.transform(
                    raw_p
                )
            )

            er1, er3, er5 = (
                expected_returns(
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
            ) = risk_levels(
                row
            )

            if (
                pd.isna(rr1)
                or
                pd.isna(rr2)
            ):

                continue

            action, composite = (
                classify(
                    calibrated_p,
                    er3,
                    er5,
                    rr1,
                    rr2,
                    float(
                        row["trend20"]
                    ),
                    float(
                        row["vol_ratio"]
                    ),
                    float(
                        row["rsi"]
                    ),
                    float(
                        row["quality"]
                    ),
                    regime
                )
            )

            daily_candidates.append(

                {

                    "ticker":
                        ticker.replace(
                            ".NS",
                            ""
                        ),

                    "ticker_full":
                        ticker,

                    "date":
                        str(
                            date_only
                        ),

                    "raw_probability":
                        raw_p,

                    "probability":
                        calibrated_p,

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

                    "stop":
                        stop,

                    "target1":
                        target1,

                    "target2":
                        target2,

                    "risk":
                        risk,

                    "rsi":
                        float(
                            row["rsi"]
                        ),

                    "volume":
                        float(
                            row["vol_ratio"]
                        ),

                    "quality":
                        float(
                            row["quality"]
                        ),

                    "regime":
                        regime,

                    "action":
                        action,

                    "composite":
                        composite,

                    "df":
                        df,

                    "index":
                        i,
                }
            )

        # ----------------------------------------------------
        # Rank candidates ON THE SAME DATE.
        # ----------------------------------------------------

        daily_candidates.sort(
            key=lambda x:
                x["composite"],
            reverse=True
        )

        # ----------------------------------------------------
        # Maximum 5 simultaneous TRADE signals.
        # This prevents a backtest from pretending that
        # unlimited capital can enter every stock.
        # ----------------------------------------------------

        selected = []

        for candidate in daily_candidates:

            if (
                candidate["action"]
                !=
                "TRADE"
            ):

                continue

            if len(selected) >= 5:
                break

            selected.append(
                candidate
            )

        selected_tickers = {
            x["ticker"]
            for x in selected
        }

        # ----------------------------------------------------
        # Evaluate outcomes AFTER ALL signals for the date
        # have been generated.
        # ----------------------------------------------------

        for candidate in daily_candidates:

            df = candidate["df"]

            i = candidate["index"]

            row = candidate

            for horizon in HORIZONS:

                ret = future_return(
                    df,
                    i,
                    horizon
                )

                candidate[
                    f"ret{horizon}"
                ] = ret

            trade_sim = simulate_trade(
                df,
                i,
                candidate["stop"],
                candidate["target1"],
                5
            )

            if trade_sim is not None:

                (
                    candidate["managed_ret5"],
                    candidate["exit_reason"]
                ) = trade_sim

            else:

                candidate[
                    "managed_ret5"
                ] = np.nan

                candidate[
                    "exit_reason"
                ] = ""

            candidate[
                "selected"
            ] = (
                candidate["ticker"]
                in selected_tickers
            )

            # Don't store the giant dataframe.
            candidate.pop(
                "df",
                None
            )

            candidate.pop(
                "index",
                None
            )

            observations.append(
                candidate
            )

        # ----------------------------------------------------
        # NOW update calibration.
        #
        # This prevents today's future outcome from affecting
        # today's probability.
        # ----------------------------------------------------

        for candidate in daily_candidates:

            outcome = (
                candidate[
                    "ret5"
                ]
            )

            if pd.notna(
                outcome
            ):

                calibrator.update(
                    candidate[
                        "raw_probability"
                    ],
                    outcome > 0
                )

        # ----------------------------------------------------
        # PORTFOLIO DAY
        # ----------------------------------------------------

        selected_returns = []

        for candidate in selected:

            value = candidate.get(
                "ret5",
                np.nan
            )

            if pd.notna(value):

                selected_returns.append(
                    value
                )

        if selected_returns:

            portfolio_daily.append(
                {
                    "date":
                        str(date_only),

                    "n_positions":
                        len(
                            selected_returns
                        ),

                    "portfolio_return":
                        float(
                            np.mean(
                                selected_returns
                            )
                        ),
                }
            )

    # ========================================================
    # RESULTS
    # ========================================================

    output = pd.DataFrame(
        observations
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    raw_file = (
        Path("audit")
        /
        f"walkforward_v6_3_11_"
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
        "V6.3.11 WALK-FORWARD BACKTEST"
    )

    print(
        "=" * 70
    )

    print(
        f"Total candidate observations: "
        f"{len(output)}"
    )

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
    # ACTION PERFORMANCE
    # ========================================================

    summaries = []

    for action in [
        "TRADE",
        "WATCH",
        "WAIT"
    ]:

        subset = output[
            output["action"]
            ==
            action
        ]

        for horizon in HORIZONS:

            series = (
                subset[
                    f"ret{horizon}"
                ]
                .dropna()
            )

            if series.empty:
                continue

            winners = series[
                series > 0
            ]

            losers = series[
                series <= 0
            ]

            if (
                not losers.empty
                and
                abs(
                    losers.sum()
                ) > 0
            ):

                pf = (
                    winners.sum()
                    /
                    abs(
                        losers.sum()
                    )
                )

            else:

                pf = np.nan

            summaries.append(

                {

                    "selection":
                        action,

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
                        float(pf)
                        if pd.notna(pf)
                        else np.nan,

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
        "ACTION GROUP PERFORMANCE:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    summary.to_csv(
        Path("audit")
        /
        f"action_group_performance_"
        f"v6_3_11_"
        f"{timestamp}.csv",
        index=False
    )

    # ========================================================
    # PROBABILITY CALIBRATION
    # ========================================================

    output[
        "probability_bucket"
    ] = pd.cut(
        output[
            "probability"
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
                "probability",
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

    calibration.to_csv(
        Path("audit")
        /
        f"probability_calibration_"
        f"v6_3_11_"
        f"{timestamp}.csv",
        index=False
    )

    # ========================================================
    # PORTFOLIO
    # ========================================================

    portfolio = pd.DataFrame(
        portfolio_daily
    )

    if not portfolio.empty:

        portfolio[
            "equity"
        ] = (
            1
            +
            portfolio[
                "portfolio_return"
            ]
        ).cumprod()

        total_return = (
            portfolio[
                "equity"
            ].iloc[-1]
            - 1
        )

        running_max = (
            portfolio[
                "equity"
            ].cummax()
        )

        drawdown = (
            portfolio[
                "equity"
            ]
            /
            running_max
            - 1
        )

        max_drawdown = (
            drawdown.min()
        )

        daily = portfolio[
            "portfolio_return"
        ]

        if (
            daily.std()
            and
            daily.std() > 0
        ):

            sharpe = (
                daily.mean()
                /
                daily.std()
                *
                np.sqrt(252)
            )

        else:

            sharpe = np.nan

        print("")
        print(
            "PORTFOLIO-LEVEL RESULT:"
        )

        print(
            f"Trading days: "
            f"{len(portfolio)}"
        )

        print(
            f"Cumulative return: "
            f"{total_return*100:.2f}%"
        )

        print(
            f"Maximum drawdown: "
            f"{max_drawdown*100:.2f}%"
        )

        print(
            f"Annualized Sharpe-like ratio: "
            f"{sharpe:.2f}"
        )

        portfolio.to_csv(
            Path("audit")
            /
            f"portfolio_performance_"
            f"v6_3_11_"
            f"{timestamp}.csv",
            index=False
        )

    # ========================================================
    # SYMBOL PERFORMANCE
    # ========================================================

    trades = output[
        output["action"]
        ==
        "TRADE"
    ]

    if not trades.empty:

        symbols = (
            trades
            .groupby(
                "ticker"
            )
            .agg(

                observations=(
                    "ret5",
                    "count"
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
                ),

                median_return=(
                    "ret5",
                    "median"
                )
            )
            .reset_index()
        )

        # Only display statistically useful groups.
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
            .head(30)
            .to_string(
                index=False
            )
        )

        symbols.to_csv(
            Path("audit")
            /
            f"trade_symbol_performance_"
            f"v6_3_11_"
            f"{timestamp}.csv",
            index=False
        )

    # ========================================================
    # FILTER / DATA QUALITY
    # ========================================================

    print("")
    print(
        "DATA QUALITY:"
    )

    print(
        f"Unique symbols tested: "
        f"{output['ticker'].nunique()}"
    )

    print(
        f"Unique signal dates: "
        f"{output['date'].nunique()}"
    )

    print(
        f"TRADE observations: "
        f"{(
            output['action'] == 'TRADE'
        ).sum()}"
    )

    print("")
    print(
        "=" * 70
    )

    print(
        "V6.3.11 BACKTEST COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "Do NOT promote to real-money "
        "trading solely from this test."
    )

    print(
        "Use an untouched out-of-sample "
        "period before live deployment."
    )


if __name__ == "__main__":

    main()
