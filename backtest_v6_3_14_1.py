"""
V6.3.14.1 CHRONOLOGICAL WALK-FORWARD BACKTEST

V6.3.14.1 is a validation/engineering revision of V6.3.14.

IMPORTANT:
- The V6.3.14 strategy thresholds are preserved.
- No strategy loosening is performed.
- Chronological probability calibration is preserved.
- Today's future outcome is never used to determine today's probability.
- Maximum five simultaneous TRADE selections are retained.
- Audit output is versioned as V6.3.14.1.
- Additional concentration and exposure diagnostics are included.

This is a research/backtesting script.
It is NOT a live trading approval.
"""

import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from market_engine_v6_3_14 import (
    UNIVERSE,
    add_features,
    raw_probability,
    expected_returns,
    risk_levels,
    SLIPPAGE_BPS,
    ROUND_TRIP_COST_BPS,
)


VERSION = "V6.3.14.1"

MIN_HISTORY = 220

HORIZONS = [
    1,
    3,
    5,
]

MAX_PORTFOLIO_POSITIONS = 5

MIN_SYMBOL_OBSERVATIONS = 20


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data(ticker, period="6y"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                c[0] if isinstance(c, tuple) else c
                for c in df.columns
            ]

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        if not all(c in df.columns for c in required):
            return pd.DataFrame()

        df = (
            df[required]
            .apply(pd.to_numeric, errors="coerce")
            .dropna(subset=["Close"])
        )

        return df

    except Exception as error:
        print(
            f"{ticker}: download failed: {error}"
        )
        return pd.DataFrame()


# ============================================================
# HISTORICAL MARKET REGIME
# ============================================================

def build_regimes(index_features):
    regimes = {}

    for i in range(len(index_features)):

        row = index_features.iloc[i]

        date = pd.Timestamp(
            index_features.index[i]
        ).date()

        if pd.isna(row["sma50"]):
            regimes[date] = "UNKNOWN"
            continue

        if i >= 5:
            slope = (
                index_features["sma50"].iloc[i]
                -
                index_features["sma50"].iloc[i - 5]
            )
        else:
            slope = np.nan

        distance = (
            row["Close"] /
            row["sma50"]
            - 1
        )

        if (
            distance > 0.008
            and pd.notna(slope)
            and slope > 0
        ):
            regime = "FAVORABLE"

        elif (
            distance < -0.008
            and pd.notna(slope)
            and slope < 0
        ):
            regime = "UNFAVORABLE"

        else:
            regime = "MIXED"

        regimes[date] = regime

    return regimes


# ============================================================
# CHRONOLOGICAL CALIBRATOR
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
            1.01,
        ]

        self.count = np.zeros(9)

        self.wins = np.zeros(9)

    def bucket(self, p):

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

    def transform(self, p):

        b = self.bucket(p)

        n = self.count[b]

        w = self.wins[b]

        prior_strength = 25.0

        if n < 20:

            return float(
                np.clip(
                    0.50
                    +
                    0.70 * (p - 0.50),
                    0.35,
                    0.70,
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

        calibrated = (
            0.50
            +
            0.90
            * (empirical - 0.50)
        )

        return float(
            np.clip(
                calibrated,
                0.35,
                0.70,
            )
        )

    def update(self, p, win):

        b = self.bucket(p)

        self.count[b] += 1

        self.wins[b] += (
            1 if win else 0
        )


# ============================================================
# TRANSACTION COST
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
    horizon,
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
            exit_price - entry
        )
        /
        entry
        -
        cost_fraction()
    )


# ============================================================
# MANAGED STOP/TARGET TRADE
# ============================================================

def simulate_trade(
    df,
    i,
    stop,
    target,
    horizon,
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
            len(df),
        ),
    ):

        low = float(
            df["Low"].iloc[j]
        )

        high = float(
            df["High"].iloc[j]
        )

        # Conservative:
        # if stop and target are both
        # touched in the same candle,
        # assume STOP occurred first.

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
            exit_price - entry
        )
        /
        entry
        -
        cost_fraction()
    )

    return (
        float(ret),
        exit_reason,
    )


# ============================================================
# V6.3.14 CLASSIFICATION
#
# IMPORTANT:
# These thresholds are intentionally preserved
# from the V6.3.14 strategy.
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
    regime,
):

    trend_score = np.clip(
        50
        +
        1000 * trend20,
        0,
        100,
    )

    momentum_score = 50

    probability_score = (
        50
        +
        400
        * (
            probability - 0.50
        )
    )

    expected_score = np.clip(
        50
        +
        500 * er3
        +
        250 * er5,
        0,
        100,
    )

    volume_score = np.clip(
        50
        +
        35
        * (
            volume - 1
        ),
        0,
        100,
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
        100,
    )

    regime_score = {
        "FAVORABLE": 100,
        "MIXED": 60,
        "UNFAVORABLE": 30,
        "UNKNOWN": 40,
    }.get(
        regime,
        40,
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
        8
        *
        (
            quality - 0.50
        )
    )

    composite = float(
        np.clip(
            composite,
            0,
            100,
        )
    )

    # --------------------------------------------------------
    # V6.3.14 TRADE GATE
    # --------------------------------------------------------

    if (
        composite >= 72
        and probability >= 0.56
        and er3 > 0
        and er5 > 0
        and rr1 >= 0.85
        and rr2 >= 1.25
        and volume >= 0.70
        and rsi <= 72
    ):

        action = "TRADE"

    # --------------------------------------------------------
    # V6.3.14 WATCH GATE
    # --------------------------------------------------------

    elif (
        composite >= 60
        and probability >= 0.53
        and er3 > -0.001
        and rr2 >= 1.10
        and volume >= 0.55
    ):

        action = "WATCH"

    else:

        action = "WAIT"

    failures = []

    if probability < 0.56:
        failures.append(
            "probability"
        )

    if er3 <= 0:
        failures.append("er3")

    if er5 <= 0:
        failures.append("er5")

    if rr1 < 0.85:
        failures.append("rr1")

    if rr2 < 1.25:
        failures.append("rr2")

    if volume < 0.70:
        failures.append("volume")

    if rsi > 72:
        failures.append("rsi")

    if trend20 < -0.005:
        failures.append("trend")

    return {
        "action": action,
        "composite": composite,
        "failures": failures,
    }


# ============================================================
# MAIN BACKTEST
# ============================================================

def main():

    Path("audit").mkdir(
        exist_ok=True
    )

    print(
        f"Starting {VERSION} "
        "chronological walk-forward backtest..."
    )

    # ========================================================
    # NIFTY INDEX
    # ========================================================

    index = download_data(
        "^NSEI",
        "6y",
    )

    if index.empty:

        raise RuntimeError(
            "NSE index data unavailable."
        )

    index_features = add_features(
        index
    )

    if index_features.empty:

        raise RuntimeError(
            "Unable to calculate NIFTY features."
        )

    regimes = build_regimes(
        index_features
    )

    # ========================================================
    # LOAD ALL STOCK DATA FIRST
    # ========================================================

    stock_data = {}

    for number, ticker in enumerate(
        UNIVERSE,
        start=1,
    ):

        print(
            f"Loading "
            f"[{number}/{len(UNIVERSE)}] "
            f"{ticker}"
        )

        df = download_data(
            ticker,
            "6y",
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
            features,
        )

    if not stock_data:

        raise RuntimeError(
            "No stock data available."
        )

    # ========================================================
    # MASTER CHRONOLOGICAL DATES
    # ========================================================

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
            "UNKNOWN",
        )

        if regime == "UNKNOWN":
            continue

        daily_candidates = []

        # ====================================================
        # GENERATE ALL SIGNALS FOR THE DATE
        # ====================================================

        for ticker, (
            df,
            features,
        ) in stock_data.items():

            if current_date not in features.index:
                continue

            i = features.index.get_loc(
                current_date
            )

            if not isinstance(
                i,
                (int, np.integer),
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
                "trend200",
            ]

            if any(
                pd.isna(row[c])
                for c in required
            ):
                continue

            # ------------------------------------------------
            # RAW MODEL
            # ------------------------------------------------

            raw_p = raw_probability(
                row,
                regime,
            )

            # ------------------------------------------------
            # CRITICAL:
            # calibration uses only outcomes
            # known BEFORE this date.
            # ------------------------------------------------

            calibrated_p = (
                calibrator.transform(
                    raw_p
                )
            )

            er1, er3, er5 = (
                expected_returns(
                    row,
                    regime,
                )
            )

            (
                stop,
                target1,
                target2,
                rr1,
                rr2,
                risk,
            ) = risk_levels(
                row
            )

            if (
                pd.isna(rr1)
                or
                pd.isna(rr2)
            ):
                continue

            classification = (
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
                    regime,
                )
            )

            daily_candidates.append(
                {
                    "ticker":
                        ticker.replace(
                            ".NS",
                            "",
                        ),

                    "ticker_full":
                        ticker,

                    "date":
                        str(date_only),

                    "raw_probability":
                        float(raw_p),

                    "probability":
                        float(calibrated_p),

                    "er1":
                        float(er1),

                    "er3":
                        float(er3),

                    "er5":
                        float(er5),

                    "stop":
                        float(stop),

                    "target1":
                        float(target1),

                    "target2":
                        float(target2),

                    "rr1":
                        float(rr1),

                    "rr2":
                        float(rr2),

                    "risk":
                        float(risk),

                    "rsi":
                        float(row["rsi"]),

                    "volume":
                        float(row["vol_ratio"]),

                    "quality":
                        float(row["quality"]),

                    "trend20":
                        float(row["trend20"]),

                    "trend50":
                        float(row["trend50"]),

                    "trend200":
                        float(row["trend200"]),

                    "regime":
                        regime,

                    "action":
                        classification["action"],

                    "composite":
                        classification["composite"],

                    "failures":
                        ",".join(
                            classification[
                                "failures"
                            ]
                        ),

                    "df":
                        df,

                    "index":
                        i,
                }
            )

        # ====================================================
        # RANK ON SAME DATE
        # ====================================================

        daily_candidates.sort(
            key=lambda x:
                x["composite"],
            reverse=True,
        )

        # ====================================================
        # MAXIMUM FIVE TRADE POSITIONS
        # ====================================================

        selected = []

        for candidate in daily_candidates:

            if (
                candidate["action"]
                !=
                "TRADE"
            ):
                continue

            if (
                len(selected)
                >=
                MAX_PORTFOLIO_POSITIONS
            ):
                break

            selected.append(
                candidate
            )

        selected_tickers = {
            x["ticker"]
            for x in selected
        }

        # ====================================================
        # EVALUATE FUTURE OUTCOMES
        # ====================================================

        for candidate in daily_candidates:

            df = candidate["df"]

            i = candidate["index"]

            for horizon in HORIZONS:

                candidate[
                    f"ret{horizon}"
                ] = future_return(
                    df,
                    i,
                    horizon,
                )

            managed = simulate_trade(
                df,
                i,
                candidate["stop"],
                candidate["target1"],
                5,
            )

            if managed is not None:

                (
                    candidate[
                        "managed_ret5"
                    ],
                    candidate[
                        "exit_reason"
                    ],
                ) = managed

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

            candidate.pop(
                "df",
                None,
            )

            candidate.pop(
                "index",
                None,
            )

            observations.append(
                candidate
            )

        # ====================================================
        # ONLY NOW UPDATE CALIBRATION
        # ====================================================

        for candidate in daily_candidates:

            outcome = candidate[
                "ret5"
            ]

            if pd.notna(outcome):

                calibrator.update(
                    candidate[
                        "raw_probability"
                    ],
                    outcome > 0,
                )

        # ====================================================
        # PORTFOLIO DAY
        # ====================================================

        selected_returns = []

        for candidate in selected:

            value = candidate.get(
                "ret5",
                np.nan,
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
    # RESULTS DATAFRAME
    # ========================================================

    output = pd.DataFrame(
        observations
    )

    if output.empty:

        raise RuntimeError(
            "No observations were generated."
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # ========================================================
    # RAW WALK-FORWARD AUDIT
    # ========================================================

    raw_file = (
        Path("audit")
        /
        f"walkforward_v6_3_14_1_"
        f"{timestamp}.csv"
    )

    output.to_csv(
        raw_file,
        index=False,
    )

    print("")
    print("=" * 70)
    print(
        f"{VERSION} WALK-FORWARD BACKTEST"
    )
    print("=" * 70)

    print(
        f"Total candidate observations: "
        f"{len(output)}"
    )

    print("")
    print("ACTION COUNTS:")
    print(
        output["action"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # ACTION PERFORMANCE
    # ========================================================

    summaries = []

    for action in [
        "TRADE",
        "WATCH",
        "WAIT",
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
        f"v6_3_14_1_"
        f"{timestamp}.csv",
        index=False,
    )

    # ========================================================
    # PROBABILITY CALIBRATION
    # ========================================================

    output[
        "probability_bucket"
    ] = pd.cut(
        output["probability"],
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
            1.01,
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
            "75%+",
        ],
    )

    calibration = (
        output
        .groupby(
            "probability_bucket",
            observed=False,
        )
        .agg(
            observations=(
                "ret5",
                "size",
            ),

            average_model_probability=(
                "probability",
                "mean",
            ),

            actual_win_rate=(
                "ret5",
                lambda x:
                    (
                        x > 0
                    ).mean(),
            ),

            average_return=(
                "ret5",
                "mean",
            ),
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
        f"v6_3_14_1_"
        f"{timestamp}.csv",
        index=False,
    )

    # ========================================================
    # PORTFOLIO PERFORMANCE
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
            f"{total_return * 100:.2f}%"
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
            f"v6_3_14_1_"
            f"{timestamp}.csv",
            index=False,
        )

    else:

        print("")
        print(
            "PORTFOLIO-LEVEL RESULT:"
        )

        print(
            "No completed portfolio days."
        )

    # ========================================================
    # TRADE STATISTICS
    # ========================================================

    trades = output[
        output["action"]
        ==
        "TRADE"
    ].copy()

    print("")
    print(
        "TRADE STATISTICS:"
    )

    print(
        f"TRADE observations: "
        f"{len(trades)}"
    )

    if not trades.empty:

        trade_rate = (
            len(trades)
            /
            len(output)
            *
            100
        )

        print(
            f"TRADE percentage of "
            f"all candidates: "
            f"{trade_rate:.3f}%"
        )

        unique_trade_dates = (
            trades["date"]
            .nunique()
        )

        print(
            f"Unique TRADE dates: "
            f"{unique_trade_dates}"
        )

        selected = trades[
            trades["selected"]
            == True
        ]

        print(
            f"Selected TRADE observations: "
            f"{len(selected)}"
        )

        if len(selected) > 0:

            print(
                f"Average selected "
                f"positions/day: "
                f"{selected.groupby('date').size().mean():.2f}"
            )

            print(
                f"Maximum selected "
                f"positions/day: "
                f"{selected.groupby('date').size().max()}"
            )

    # ========================================================
    # ALL-SYMBOL PERFORMANCE
    #
    # This is intentionally NOT restricted to >=20.
    # ========================================================

    if not trades.empty:

        all_symbols = (
            trades
            .groupby("ticker")
            .agg(
                observations=(
                    "ret5",
                    "count",
                ),

                win_rate=(
                    "ret5",
                    lambda x:
                        (
                            x > 0
                        ).mean(),
                ),

                average_return=(
                    "ret5",
                    "mean",
                ),

                median_return=(
                    "ret5",
                    "median",
                ),

                average_probability=(
                    "probability",
                    "mean",
                ),

                average_composite=(
                    "composite",
                    "mean",
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "average_return",
                    "win_rate",
                ],
                ascending=False,
            )
        )

        print("")
        print(
            "ALL TRADE SYMBOL PERFORMANCE:"
        )

        print(
            all_symbols.to_string(
                index=False
            )
        )

        all_symbols.to_csv(
            Path("audit")
            /
            f"trade_symbol_performance_all_"
            f"v6_3_14_1_"
            f"{timestamp}.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Statistically useful symbols
        # ----------------------------------------------------

        useful_symbols = (
            all_symbols[
                all_symbols[
                    "observations"
                ]
                >=
                MIN_SYMBOL_OBSERVATIONS
            ]
        )

        print("")
        print(
            f"TRADE SYMBOLS WITH "
            f">={MIN_SYMBOL_OBSERVATIONS} "
            f"OBSERVATIONS:"
        )

        if useful_symbols.empty:

            print(
                "No symbol currently has "
                f">={MIN_SYMBOL_OBSERVATIONS} "
                "TRADE observations."
            )

        else:

            print(
                useful_symbols.to_string(
                    index=False
                )
            )

        useful_symbols.to_csv(
            Path("audit")
            /
            f"trade_symbol_performance_ge20_"
            f"v6_3_14_1_"
            f"{timestamp}.csv",
            index=False,
        )

    # ========================================================
    # TRADE CONCENTRATION
    # ========================================================

    if not trades.empty:

        symbol_counts = (
            trades["ticker"]
            .value_counts()
        )

        total_trade_obs = len(
            trades
        )

        top1_share = (
            symbol_counts.iloc[0]
            /
            total_trade_obs
            *
            100
        )

        top3_share = (
            symbol_counts.head(3).sum()
            /
            total_trade_obs
            *
            100
        )

        top5_share = (
            symbol_counts.head(5).sum()
            /
            total_trade_obs
            *
            100
        )

        print("")
        print(
            "TRADE CONCENTRATION:"
        )

        print(
            f"Unique TRADE symbols: "
            f"{trades['ticker'].nunique()}"
        )

        print(
            f"Top 1 symbol share: "
            f"{top1_share:.2f}%"
        )

        print(
            f"Top 3 symbol share: "
            f"{top3_share:.2f}%"
        )

        print(
            f"Top 5 symbol share: "
            f"{top5_share:.2f}%"
        )

        concentration = pd.DataFrame(
            {
                "metric": [
                    "unique_trade_symbols",
                    "top1_trade_share_pct",
                    "top3_trade_share_pct",
                    "top5_trade_share_pct",
                ],

                "value": [
                    trades["ticker"].nunique(),
                    top1_share,
                    top3_share,
                    top5_share,
                ],
            }
        )

        concentration.to_csv(
            Path("audit")
            /
            f"trade_concentration_"
            f"v6_3_14_1_"
            f"{timestamp}.csv",
            index=False,
        )

    # ========================================================
    # EXIT REASON ANALYSIS
    # ========================================================

    if not trades.empty:

        exits = (
            trades[
                "exit_reason"
            ]
            .value_counts()
            .rename_axis(
                "exit_reason"
            )
            .reset_index(
                name="observations"
            )
        )

        print("")
        print(
            "TRADE EXIT REASONS:"
        )

        print(
            exits.to_string(
                index=False
            )
        )

        exits.to_csv(
            Path("audit")
            /
            f"trade_exit_reasons_"
            f"v6_3_14_1_"
            f"{timestamp}.csv",
            index=False,
        )

    # ========================================================
    # REGIME PERFORMANCE
    # ========================================================

    regime_rows = []

    for regime in [
        "FAVORABLE",
        "MIXED",
        "UNFAVORABLE",
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
                    ),
            }
        )

    regime_performance = pd.DataFrame(
        regime_rows
    )

    print("")
    print(
        "REGIME PERFORMANCE - 5D:"
    )

    if regime_performance.empty:

        print(
            "No regime statistics."
        )

    else:

        print(
            regime_performance.to_string(
                index=False
            )
        )

        regime_performance.to_csv(
            Path("audit")
            /
            f"regime_performance_"
            f"v6_3_14_1_"
            f"{timestamp}.csv",
            index=False,
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
            output["action"]
            ==
            "TRADE"
        ).sum()}"
    )

    print(
        f"WATCH observations: "
        f"{(
            output["action"]
            ==
            "WATCH"
        ).sum()}"
    )

    print(
        f"WAIT observations: "
        f"{(
            output["action"]
            ==
            "WAIT"
        ).sum()}"
    )

    # ========================================================
    # VERSION / VALIDATION MANIFEST
    # ========================================================

    manifest = pd.DataFrame(
        [
            {
                "version":
                    VERSION,

                "base_strategy":
                    "V6.3.14",

                "strategy_thresholds_changed":
                    False,

                "chronological_calibration":
                    True,

                "maximum_simultaneous_positions":
                    MAX_PORTFOLIO_POSITIONS,

                "minimum_symbol_observations":
                    MIN_SYMBOL_OBSERVATIONS,

                "slippage_bps":
                    SLIPPAGE_BPS,

                "round_trip_cost_bps":
                    ROUND_TRIP_COST_BPS,

                "candidate_observations":
                    len(output),

                "trade_observations":
                    int(
                        (
                            output["action"]
                            ==
                            "TRADE"
                        ).sum()
                    ),

                "watch_observations":
                    int(
                        (
                            output["action"]
                            ==
                            "WATCH"
                        ).sum()
                    ),

                "wait_observations":
                    int(
                        (
                            output["action"]
                            ==
                            "WAIT"
                        ).sum()
                    ),

                "unique_symbols":
                    output[
                        "ticker"
                    ].nunique(),

                "unique_signal_dates":
                    output[
                        "date"
                    ].nunique(),
            }
        ]
    )

    manifest.to_csv(
        Path("audit")
        /
        f"validation_manifest_"
        f"v6_3_14_1_"
        f"{timestamp}.csv",
        index=False,
    )

    # ========================================================
    # COMPLETION
    # ========================================================

    print("")
    print("=" * 70)

    print(
        f"{VERSION} BACKTEST COMPLETED"
    )

    print("=" * 70)

    print(
        "Strategy thresholds were "
        "NOT loosened or optimized "
        "during this validation revision."
    )

    print(
        "Chronological calibration was "
        "updated only after future outcomes "
        "became known."
    )

    print(
        "Do NOT promote to real-money "
        "trading solely from this test."
    )

    print(
        "Use a completely untouched "
        "out-of-sample period before "
        "live deployment."
    )

    print("")
    print(
        "Audit files written to: "
        "audit/"
    )


if __name__ == "__main__":
    main()
