"""
V6.3.12 WALK-FORWARD BACKTEST

Important design:

- Signals are generated chronologically.
- Calibration only uses outcomes known BEFORE the signal date.
- Entry occurs at next trading day's open.
- Slippage and round-trip costs are included.
- Stop/target simulation is conservative.
- Maximum five simultaneous new trade selections per day.
- 3D and 5D results are emphasised.
- Portfolio-level results are reported.
- Probability calibration is reported.
- No future outcome is used to generate the same day's signal.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from market_engine_v6_3_12 import (
    UNIVERSE,
    add_features,
    raw_probability,
    calibrate_probability,
    expected_returns,
    risk_levels,
    SLIPPAGE_BPS,
    ROUND_TRIP_COST_BPS,
)


VERSION = "V6.3.12"

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
            f"download failed: "
            f"{error}"
        )

        return pd.DataFrame()


# ============================================================
# HISTORICAL REGIME
# ============================================================

def build_regimes(
    index_features
):

    regimes = {}

    for i in range(
        len(index_features)
    ):

        row = index_features.iloc[i]

        date = pd.Timestamp(
            index_features.index[i]
        ).date()

        if pd.isna(
            row["sma50"]
        ):

            regimes[date] = "UNKNOWN"
            continue

        if i >= 10:

            slope = (
                index_features[
                    "sma50"
                ].iloc[i]
                -
                index_features[
                    "sma50"
                ].iloc[i - 10]
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
# CHRONOLOGICAL CALIBRATOR
# ============================================================

class Calibrator:

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
            1.01,
        ]

        self.count = np.zeros(9)
        self.wins = np.zeros(9)

    def bucket(
        self,
        probability
    ):

        for i in range(
            len(self.edges) - 1
        ):

            if (
                self.edges[i]
                <= probability
                <
                self.edges[i + 1]
            ):

                return i

        return 8

    def get_empirical(
        self,
        probability
    ):

        b = self.bucket(
            probability
        )

        n = self.count[b]

        if n <= 0:

            return (
                0,
                np.nan
            )

        return (
            int(n),
            float(
                self.wins[b]
                /
                n
            )
        )

    def transform(
        self,
        probability
    ):

        n, empirical = (
            self.get_empirical(
                probability
            )
        )

        return calibrate_probability(
            probability,
            n,
            empirical
        )

    def update(
        self,
        probability,
        win
    ):

        b = self.bucket(
            probability
        )

        self.count[b] += 1

        if win:

            self.wins[b] += 1


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
# FORWARD RETURN
# ============================================================

def forward_return(
    df,
    index,
    horizon
):

    if (
        index + 1 >= len(df)
        or
        index + horizon >= len(df)
    ):

        return np.nan

    entry = (
        float(
            df["Open"]
            .iloc[index + 1]
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
            df["Close"]
            .iloc[
                index + horizon
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
# MANAGED TRADE
# ============================================================

def simulate_trade(
    df,
    index,
    stop,
    target,
    horizon=5
):

    if (
        index + 1 >= len(df)
        or
        index + horizon >= len(df)
    ):

        return None

    entry = (
        float(
            df["Open"]
            .iloc[index + 1]
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
            df["Close"]
            .iloc[
                index + horizon
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
        index + 1,
        min(
            index + horizon + 1,
            len(df)
        )
    ):

        low = float(
            df["Low"].iloc[j]
        )

        high = float(
            df["High"].iloc[j]
        )

        # Conservative same-candle assumption:
        # stop is considered first if both are touched.

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

    net_return = (
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
        float(net_return),
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
    trend50,
    volume,
    rsi,
    quality,
    regime
):

    probability_score = np.clip(
        50
        +
        450
        * (
            probability
            - 0.50
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
            0.60 * trend20
            +
            0.40 * trend50
        ),
        0,
        100
    )

    momentum_score = np.clip(
        50
        +
        500
        * trend20,
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

    elif 42 <= rsi < 48:

        rsi_score = 75

    elif 65 < rsi <= 70:

        rsi_score = 70

    else:

        rsi_score = 35

    rr_score = np.clip(
        50
        +
        28
        * (
            rr2
            -
            1
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
        quality
        * 100
    )

    composite = (

        0.25
        * probability_score

        +

        0.22
        * expected_score

        +

        0.15
        * trend_score

        +

        0.10
        * momentum_score

        +

        0.07
        * volume_score

        +

        0.06
        * rsi_score

        +

        0.08
        * rr_score

        +

        0.04
        * regime_score

        +

        0.03
        * quality_score
    )

    if volume < 0.60:
        composite -= 8

    if rr1 < 0.80:
        composite -= 7

    if rr2 < 1.15:
        composite -= 5

    if rsi > 75:
        composite -= 10

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
        and
        trend20 > -0.005
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
# PERFORMANCE SUMMARY
# ============================================================

def performance_table(
    output,
    actions
):

    rows = []

    for action in actions:

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

                profit_factor = (
                    winners.sum()
                    /
                    abs(
                        losers.sum()
                    )
                )

            else:

                profit_factor = np.nan

            rows.append(
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
                        float(
                            profit_factor
                        )
                        if pd.notna(
                            profit_factor
                        )
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

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Starting {VERSION} "
        "chronological walk-forward backtest..."
    )

    Path(
        "audit"
    ).mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # NIFTY
    # --------------------------------------------------------

    nifty = download_data(
        "^NSEI",
        "6y"
    )

    if nifty.empty:

        raise RuntimeError(
            "Unable to download Nifty data."
        )

    nifty_features = add_features(
        nifty
    )

    regimes = build_regimes(
        nifty_features
    )

    # --------------------------------------------------------
    # STOCK DATA
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
            "No stock data was downloaded."
        )

    # --------------------------------------------------------
    # ALL AVAILABLE DATES
    # --------------------------------------------------------

    dates = sorted(
        set(
            date
            for (
                _ticker,
                (
                    _df,
                    features
                )
            )
            in stock_data.items()
            for date in features.index
        )
    )

    calibrator = Calibrator()

    observations = []

    portfolio_rows = []

    # ========================================================
    # CHRONOLOGICAL LOOP
    # ========================================================

    for current_date in dates:

        current_date = pd.Timestamp(
            current_date
        )

        regime = regimes.get(
            current_date.date(),
            "UNKNOWN"
        )

        if regime == "UNKNOWN":
            continue

        daily = []

        # ----------------------------------------------------
        # Generate ALL signals for this date.
        # ----------------------------------------------------

        for ticker, (
            df,
            features
        ) in stock_data.items():

            if current_date not in features.index:
                continue

            index = features.index.get_loc(
                current_date
            )

            if not isinstance(
                index,
                (int, np.integer)
            ):
                continue

            if index < MIN_HISTORY:
                continue

            row = features.iloc[index]

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

            raw_p = raw_probability(
                row,
                regime
            )

            # IMPORTANT:
            # Only prior outcomes are available here.
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

            action, composite = classify(
                calibrated_p,
                er3,
                er5,
                rr1,
                rr2,
                float(
                    row["trend20"]
                ),
                float(
                    row["trend50"]
                ),
                float(
                    row["volume_ratio"]
                ),
                float(
                    row["rsi"]
                ),
                float(
                    row["quality"]
                ),
                regime
            )

            daily.append(
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
                            current_date.date()
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
                            row["volume_ratio"]
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

                    "_df":
                        df,

                    "_index":
                        index,
                }
            )

        # ----------------------------------------------------
        # Rank on same date.
        # ----------------------------------------------------

        daily.sort(
            key=lambda x:
                x["composite"],
            reverse=True
        )

        selected = [
            x for x in daily
            if x["action"] == "TRADE"
        ][:5]

        selected_tickers = {
            x["ticker"]
            for x in selected
        }

        # ----------------------------------------------------
        # Calculate outcomes.
        # ----------------------------------------------------

        for candidate in daily:

            df = candidate["_df"]

            index = candidate["_index"]

            for horizon in HORIZONS:

                candidate[
                    f"ret{horizon}"
                ] = forward_return(
                    df,
                    index,
                    horizon
                )

            managed = simulate_trade(
                df,
                index,
                candidate["stop"],
                candidate["target1"],
                5
            )

            if managed is not None:

                (
                    candidate[
                        "managed_ret5"
                    ],
                    candidate[
                        "exit_reason"
                    ]
                ) = managed

            else:

                candidate[
                    "managed_ret5"
                ] = np.nan

                candidate[
                    "exit_reason"
                ] = ""

            candidate["selected"] = (
                candidate["ticker"]
                in selected_tickers
            )

            candidate.pop(
                "_df",
                None
            )

            candidate.pop(
                "_index",
                None
            )

            observations.append(
                candidate
            )

        # ----------------------------------------------------
        # Update calibration AFTER today's outcomes.
        #
        # This is essential to prevent look-ahead bias.
        # ----------------------------------------------------

        for candidate in daily:

            outcome = candidate.get(
                "ret5",
                np.nan
            )

            if pd.notna(outcome):

                calibrator.update(
                    candidate[
                        "raw_probability"
                    ],
                    outcome > 0
                )

        # ----------------------------------------------------
        # Portfolio result for this signal date.
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

            portfolio_rows.append(
                {
                    "date":
                        str(
                            current_date.date()
                        ),

                    "positions":
                        len(
                            selected_returns
                        ),

                    "portfolio_return":
                        float(
                            np.mean(
                                selected_returns
                            )
                        )
                }
            )

    # ========================================================
    # DATAFRAME
    # ========================================================

    output = pd.DataFrame(
        observations
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        Path("audit")
        /
        f"walkforward_v6_3_12_"
        f"{timestamp}.csv"
    )

    output.to_csv(
        output_file,
        index=False
    )

    print("")
    print(
        "=" * 70
    )

    print(
        "V6.3.12 WALK-FORWARD BACKTEST"
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

    summary = performance_table(
        output,
        [
            "TRADE",
            "WATCH",
            "WAIT"
        ]
    )

    print("")
    print(
        "ACTION GROUP PERFORMANCE:"
    )

    if not summary.empty:

        print(
            summary.to_string(
                index=False
            )
        )

    summary.to_csv(
        Path("audit")
        /
        f"action_group_performance_"
        f"v6_3_12_"
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
        f"v6_3_12_"
        f"{timestamp}.csv",
        index=False
    )

    # ========================================================
    # PORTFOLIO
    # ========================================================

    portfolio = pd.DataFrame(
        portfolio_rows
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

        cumulative_return = (
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

        daily_returns = (
            portfolio[
                "portfolio_return"
            ]
        )

        if (
            daily_returns.std()
            and
            daily_returns.std() > 0
        ):

            sharpe = (
                daily_returns.mean()
                /
                daily_returns.std()
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
            f"{cumulative_return * 100:.2f}%"
        )

        print(
            f"Maximum drawdown: "
            f"{max_drawdown * 100:.2f}%"
        )

        print(
            f"Annualized Sharpe-like ratio: "
            f"{sharpe:.2f}"
        )

        portfolio.to_csv(
            Path("audit")
            /
            f"portfolio_performance_"
            f"v6_3_12_"
            f"{timestamp}.csv",
            index=False
        )

    # ========================================================
    # REGIME PERFORMANCE
    # ========================================================

    regime_rows = []

    for regime in [
        "FAVORABLE",
        "MIXED",
        "UNFAVORABLE"
    ]:

        subset = output[
            output["regime"]
            ==
            regime
        ]

        series = (
            subset["ret5"]
            .dropna()
        )

        if series.empty:
            continue

        regime_rows.append(
            {
                "regime":
                    regime,

                "observations":
                    len(series),

                "win_rate":
                    float(
                        (
                            series > 0
                        ).mean()
                    ),

                "average_return":
                    float(
                        series.mean()
                    ),

                "median_return":
                    float(
                        series.median()
                    )
            }
        )

    regime_table = pd.DataFrame(
        regime_rows
    )

    print("")
    print(
        "REGIME PERFORMANCE - 5D:"
    )

    if not regime_table.empty:

        print(
            regime_table.to_string(
                index=False
            )
        )

    regime_table.to_csv(
        Path("audit")
        /
        f"regime_performance_"
        f"v6_3_12_"
        f"{timestamp}.csv",
        index=False
    )

    # ========================================================
    # TRADE SYMBOL PERFORMANCE
    # ========================================================

    trades = output[
        output["action"]
        ==
        "TRADE"
    ]

    if not trades.empty:

        symbol_table = (
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

        reliable = symbol_table[
            symbol_table[
                "observations"
            ] >= 20
        ].sort_values(
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

        if not reliable.empty:

            print(
                reliable
                .head(30)
                .to_string(
                    index=False
                )
            )

        else:

            print(
                "No symbol has yet reached "
                "20 trade observations."
            )

        symbol_table.to_csv(
            Path("audit")
            /
            f"trade_symbol_performance_"
            f"v6_3_12_"
            f"{timestamp}.csv",
            index=False
        )

    # ========================================================
    # DATA QUALITY
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

    print(
        f"WATCH observations: "
        f"{(
            output['action'] == 'WATCH'
        ).sum()}"
    )

    print(
        f"WAIT observations: "
        f"{(
            output['action'] == 'WAIT'
        ).sum()}"
    )

    print("")
    print(
        "=" * 70
    )

    print(
        "V6.3.12 BACKTEST COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "Do NOT promote V6.3.12 "
        "to real-money trading solely "
        "from this backtest."
    )

    print(
        "Use an untouched out-of-sample "
        "period before live deployment."
    )


if __name__ == "__main__":

    main()
