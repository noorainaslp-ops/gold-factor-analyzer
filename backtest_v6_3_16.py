#!/usr/bin/env python3

"""
V6.3.16 CHRONOLOGICAL WALK-FORWARD BACKTEST

Research-only NSE long-side model.

IMPORTANT DESIGN RULES
----------------------
1. Stock and market data are explicitly date-aligned.
2. No future observations are used when generating a signal.
3. Probability calibration is chronological.
4. Calibration uses only observations available before the signal date.
5. A minimum calibration sample is required.
6. The final chronological portion is reported separately as OOS.
7. Transaction costs and slippage are included.
8. Probability calibration statistics are reported.
9. Missing/delisted Yahoo Finance symbols do not crash the run.
10. This program does NOT guarantee trading profitability.
"""

from pathlib import Path
from datetime import datetime
import os
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "V6.3.16"

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BACKTEST_PERIOD = os.getenv(
    "BACKTEST_PERIOD",
    "6y",
)

MIN_HISTORY = int(
    os.getenv(
        "MIN_HISTORY",
        "220",
    )
)

CAL_WINDOW = int(
    os.getenv(
        "CAL_WINDOW",
        "8000",
    )
)

MIN_CAL = int(
    os.getenv(
        "MIN_CAL",
        "250",
    )
)

OOS_FRACTION = float(
    os.getenv(
        "OOS_FRACTION",
        "0.20",
    )
)

COST_BPS = float(
    os.getenv(
        "COST_BPS",
        "10",
    )
)

SLIPPAGE_BPS = float(
    os.getenv(
        "SLIPPAGE_BPS",
        "5",
    )
)


# ============================================================
# SYMBOL UNIVERSE
# ============================================================

SYMBOLS = [
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
# DATA CLEANING
# ============================================================

def clean_download(data):
    """
    Normalize Yahoo Finance output.

    Handles:
    - MultiIndex columns
    - numeric conversion
    - missing rows
    - invalid values
    """

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if isinstance(
        data.columns,
        pd.MultiIndex,
    ):
        data.columns = (
            data.columns
            .get_level_values(0)
        )

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        return pd.DataFrame()

    data = data[
        required
    ].copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    data = data.dropna(
        subset=[
            "Close",
        ]
    )

    data = data.sort_index()

    # Remove duplicate timestamps.
    data = data[
        ~data.index.duplicated(
            keep="last"
        )
    ]

    return data


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    series,
    period=14,
):

    delta = series.diff()

    gain = (
        delta
        .clip(
            lower=0
        )
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    loss = (
        -delta
        .clip(
            upper=0
        )
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    rs = (
        gain
        /
        loss.replace(
            0,
            np.nan,
        )
    )

    rsi = (
        100
        -
        (
            100
            /
            (
                1 + rs
            )
        )
    )

    return rsi


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    data,
    period=14,
):

    previous_close = (
        data["Close"]
        .shift(1)
    )

    tr1 = (
        data["High"]
        -
        data["Low"]
    )

    tr2 = (
        data["High"]
        -
        previous_close
    ).abs()

    tr3 = (
        data["Low"]
        -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(
        axis=1
    )

    atr = (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    return atr


# ============================================================
# DATE ALIGNMENT HELPER
# ============================================================

def align_market_to_stock(
    stock_data,
    market_data,
):
    """
    IMPORTANT V6.3.16 FIX.

    Yahoo Finance can return different numbers of
    observations for individual stocks and NIFTY.

    Example:
        stock = 1489 rows
        market = 1484 rows

    Direct NumPy operations therefore fail.

    This function aligns the market series to the
    stock's actual trading dates.
    """

    stock_index = stock_data.index

    market = market_data.copy()

    # Normalize timestamps when possible.
    try:

        if getattr(
            stock_index,
            "tz",
            None,
        ) is not None:

            stock_index = (
                stock_index
                .tz_localize(None)
            )

        if getattr(
            market.index,
            "tz",
            None,
        ) is not None:

            market.index = (
                market.index
                .tz_localize(None)
            )

    except Exception:
        pass

    stock_copy = stock_data.copy()

    stock_copy.index = stock_index

    market = market.sort_index()

    aligned = market.reindex(
        stock_index
    )

    # Only carry forward already-known
    # market observations.
    aligned = aligned.ffill()

    return (
        stock_copy,
        aligned,
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    stock_data,
    market_data,
):

    stock, market = (
        align_market_to_stock(
            stock_data,
            market_data,
        )
    )

    close = stock[
        "Close"
    ]

    # --------------------------------------------------------
    # STOCK INDICATORS
    # --------------------------------------------------------

    atr = calculate_atr(
        stock
    )

    rsi = calculate_rsi(
        close
    )

    sma20 = (
        close
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    sma50 = (
        close
        .rolling(
            50,
            min_periods=50,
        )
        .mean()
    )

    sma200 = (
        close
        .rolling(
            200,
            min_periods=200,
        )
        .mean()
    )

    volume20 = (
        stock["Volume"]
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    return5 = (
        close
        .pct_change(5)
    )

    return20 = (
        close
        .pct_change(20)
    )

    # --------------------------------------------------------
    # MARKET FEATURES
    # --------------------------------------------------------

    market_close = (
        market["Close"]
    )

    market_return20 = (
        market_close
        .pct_change(20)
    )

    market_sma50 = (
        market_close
        .rolling(
            50,
            min_periods=50,
        )
        .mean()
    )

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend = (
        (
            close > sma20
        ).astype(float)
        +
        (
            close > sma50
        ).astype(float)
        +
        (
            close > sma200
        ).astype(float)
    ) / 3.0

    trend = (
        trend * 2.0
        - 1.0
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    momentum5 = np.tanh(
        return5
        /
        0.03
    )

    momentum20 = np.tanh(
        return20
        /
        0.08
    )

    momentum = (
        0.50 * momentum5
        +
        0.50 * momentum20
    )

    # --------------------------------------------------------
    # RELATIVE STRENGTH
    # --------------------------------------------------------

    relative_strength = np.tanh(
        (
            return20
            -
            market_return20
        )
        /
        0.08
    )

    # --------------------------------------------------------
    # RSI QUALITY COMPONENT
    # --------------------------------------------------------

    rsi_component = (
        1.0
        -
        (
            (
                rsi
                -
                55.0
            ).abs()
            /
            35.0
        ).clip(
            lower=0.0,
            upper=1.0,
        )
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ratio = (
        stock["Volume"]
        /
        volume20
    )

    volume_component = (
        volume_ratio
        -
        1.0
    ).clip(
        lower=-1.0,
        upper=1.0,
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    atr_pct = (
        atr
        /
        close
    )

    volatility_component = (
        1.0
        -
        (
            atr_pct
            /
            0.045
        )
    ).clip(
        lower=-1.0,
        upper=1.0,
    )

    # --------------------------------------------------------
    # MARKET REGIME
    #
    # THIS IS A SERIES INDEXED TO THE STOCK DATES.
    #
    # This is the central V6.3.16 fix for:
    #
    # ValueError:
    # operands could not be broadcast together
    # with shapes (1489,) (1484,)
    # --------------------------------------------------------

    regime = pd.Series(
        0.0,
        index=stock.index,
        dtype=float,
    )

    favorable = (
        market_close
        >
        market_sma50
        *
        1.002
    )

    unfavorable = (
        market_close
        <
        market_sma50
        *
        0.998
    )

    regime.loc[
        favorable.fillna(
            False
        )
    ] = 1.0

    regime.loc[
        unfavorable.fillna(
            False
        )
    ] = -1.0

    # --------------------------------------------------------
    # FINAL EXPLICIT ALIGNMENT
    # --------------------------------------------------------

    trend = trend.reindex(
        stock.index
    )

    momentum = momentum.reindex(
        stock.index
    )

    relative_strength = (
        relative_strength.reindex(
            stock.index
        )
    )

    rsi_component = (
        rsi_component.reindex(
            stock.index
        )
    )

    volume_component = (
        volume_component.reindex(
            stock.index
        )
    )

    volatility_component = (
        volatility_component.reindex(
            stock.index
        )
    )

    regime = regime.reindex(
        stock.index
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = (
        0.26 * trend
        +
        0.22 * momentum
        +
        0.16 * relative_strength
        +
        0.10
        *
        (
            2.0
            *
            rsi_component
            -
            1.0
        )
        +
        0.08 * volume_component
        +
        0.08 * volatility_component
        +
        0.10 * regime
    )

    # --------------------------------------------------------
    # RAW MODEL PROBABILITY
    # --------------------------------------------------------

    raw_probability = (
        1.0
        /
        (
            1.0
            +
            np.exp(
                -3.0 * score
            )
        )
    )

    raw_probability = (
        raw_probability
        .clip(
            lower=0.35,
            upper=0.75,
        )
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    features = pd.DataFrame(
        {
            "raw_p": raw_probability,
            "trend": trend,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "atr_pct": atr_pct,
            "regime": regime,
            "close": close,
        },
        index=stock.index,
    )

    features = features.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    return features


# ============================================================
# PROBABILITY CALIBRATION
# ============================================================

def fit_probability_calibrator(
    training,
):

    if training is None:
        return None

    training = training.dropna(
        subset=[
            "raw_p",
            "y",
        ]
    )

    if len(training) < MIN_CAL:
        return None

    if training["y"].nunique() < 2:
        return None

    calibrator = (
        IsotonicRegression(
            y_min=0.25,
            y_max=0.70,
            out_of_bounds="clip",
        )
    )

    calibrator.fit(
        training[
            "raw_p"
        ].to_numpy(
            dtype=float
        ),
        training[
            "y"
        ].to_numpy(
            dtype=float
        ),
    )

    return calibrator


# ============================================================
# SIGNAL CLASSIFICATION
# ============================================================

def classify_signal(
    row,
    probability,
):

    failures = []

    atr_pct = float(
        row["atr_pct"]
    )

    # --------------------------------------------------------
    # Risk model
    # --------------------------------------------------------

    stop_distance = max(
        1.5 * atr_pct,
        0.012,
    )

    target_distance = max(
        2.0 * stop_distance,
        0.025,
    )

    risk_reward = (
        target_distance
        /
        stop_distance
    )

    expected_return = (
        probability
        *
        target_distance
        -
        (
            1.0
            -
            probability
        )
        *
        stop_distance
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    if probability < 0.60:

        failures.append(
            "probability"
        )

    if expected_return < 0.0025:

        failures.append(
            "expected_return"
        )

    if risk_reward < 1.50:

        failures.append(
            "risk_reward"
        )

    rsi = float(
        row["rsi"]
    )

    if not (
        42.0
        <= rsi
        <= 68.0
    ):

        failures.append(
            "rsi"
        )

    volume_ratio = float(
        row["volume_ratio"]
    )

    if volume_ratio < 0.70:

        failures.append(
            "volume"
        )

    trend = float(
        row["trend"]
    )

    if trend < -0.05:

        failures.append(
            "trend"
        )

    regime = float(
        row["regime"]
    )

    if regime < -0.75:

        failures.append(
            "regime"
        )

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    if not failures:

        action = "TRADE"

    elif (
        probability >= 0.55
        and trend >= -0.20
        and 35.0 <= rsi <= 72.0
    ):

        action = "WATCH"

    else:

        action = "WAIT"

    return (
        action,
        expected_return,
        risk_reward,
        failures,
    )


# ============================================================
# PERFORMANCE TABLE
# ============================================================

def performance_table(
    data,
):

    rows = []

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    for action, group in data.groupby(
        "action"
    ):

        for horizon in [
            1,
            3,
            5,
        ]:

            column = (
                f"net{horizon}"
            )

            if column not in group:
                continue

            values = (
                group[column]
                .dropna()
            )

            if values.empty:
                continue

            winners = (
                values[
                    values > 0
                ]
            )

            losers = (
                values[
                    values <= 0
                ]
            )

            gross_wins = (
                winners.sum()
            )

            gross_losses = (
                abs(
                    losers.sum()
                )
            )

            if gross_losses > 0:

                profit_factor = (
                    gross_wins
                    /
                    gross_losses
                )

            else:

                profit_factor = np.nan

            rows.append(
                {
                    "selection": action,
                    "horizon": horizon,
                    "observations": len(values),
                    "win_rate": float(
                        (
                            values > 0
                        ).mean()
                    ),
                    "average_net_return": float(
                        values.mean()
                    ),
                    "median_net_return": float(
                        values.median()
                    ),
                    "average_winner": float(
                        winners.mean()
                    )
                    if not winners.empty
                    else np.nan,
                    "average_loser": float(
                        losers.mean()
                    )
                    if not losers.empty
                    else np.nan,
                    "profit_factor": float(
                        profit_factor
                    )
                    if np.isfinite(
                        profit_factor
                    )
                    else np.nan,
                    "best": float(
                        values.max()
                    ),
                    "worst": float(
                        values.min()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROBABILITY CALIBRATION TABLE
# ============================================================

def probability_calibration_table(
    data,
):

    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    valid = data.dropna(
        subset=[
            "p_cal",
            "y",
        ]
    ).copy()

    if valid.empty:
        return pd.DataFrame()

    bins = [
        -0.01,
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

    labels = [
        "<40%",
        "40-45%",
        "45-50%",
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75%+",
    ]

    valid["probability_bucket"] = pd.cut(
        valid["p_cal"],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []

    for (
        bucket,
        group,
    ) in valid.groupby(
        "probability_bucket",
        observed=False,
    ):

        if group.empty:
            continue

        rows.append(
            {
                "probability_bucket": str(
                    bucket
                ),
                "observations": len(
                    group
                ),
                "average_model_probability": float(
                    group[
                        "p_cal"
                    ].mean()
                ),
                "actual_win_rate": float(
                    group[
                        "y"
                    ].mean()
                ),
                "average_return": float(
                    group[
                        "net5"
                    ].mean()
                ),
                "brier": float(
                    (
                        (
                            group[
                                "p_cal"
                            ]
                            -
                            group[
                                "y"
                            ]
                        )
                        ** 2
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FILTER PERFORMANCE
# ============================================================

def filter_performance_table(
    data,
):

    filters = [
        "probability",
        "expected_return",
        "risk_reward",
        "rsi",
        "volume",
        "trend",
        "regime",
    ]

    rows = []

    for filter_name in filters:

        if filter_name == "probability":

            passed = (
                data["p_cal"]
                >= 0.60
            )

        elif filter_name == "expected_return":

            passed = (
                data["er5"]
                >= 0.0025
            )

        elif filter_name == "risk_reward":

            passed = (
                data["rr"]
                >= 1.50
            )

        elif filter_name == "rsi":

            passed = (
                data["rsi"]
                .between(
                    42,
                    68,
                )
            )

        elif filter_name == "volume":

            passed = (
                data[
                    "volume_ratio"
                ]
                >= 0.70
            )

        elif filter_name == "trend":

            passed = (
                data["trend"]
                >= -0.05
            )

        elif filter_name == "regime":

            passed = (
                data["regime"]
                >= -0.75
            )

        passed_count = int(
            passed.fillna(
                False
            ).sum()
        )

        failed_count = (
            len(data)
            -
            passed_count
        )

        rows.append(
            {
                "filter": filter_name,
                "passed": passed_count,
                "failed": failed_count,
                "pass_rate": (
                    passed_count
                    /
                    len(data)
                    if len(data)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN BACKTEST
# ============================================================

def main():

    print(
        f"Starting {VERSION} "
        "chronological walk-forward "
        "backtest..."
    )

    print(
        f"Backtest period: "
        f"{BACKTEST_PERIOD}"
    )

    print(
        f"Calibration window: "
        f"{CAL_WINDOW}"
    )

    print(
        f"Minimum calibration sample: "
        f"{MIN_CAL}"
    )

    # --------------------------------------------------------
    # DOWNLOAD NIFTY
    # --------------------------------------------------------

    try:

        market_raw = yf.download(
            "^NSEI",
            period=BACKTEST_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        market = clean_download(
            market_raw
        )

    except Exception as error:

        raise RuntimeError(
            "Unable to download NIFTY data: "
            f"{error}"
        )

    if market.empty:

        raise RuntimeError(
            "NIFTY data unavailable."
        )

    # --------------------------------------------------------
    # DOWNLOAD STOCK DATA
    # --------------------------------------------------------

    all_rows = []

    successful_symbols = 0

    for number, symbol in enumerate(
        SYMBOLS,
        1,
    ):

        print(
            f"Loading "
            f"[{number}/{len(SYMBOLS)}] "
            f"{symbol}"
        )

        try:

            raw = yf.download(
                symbol,
                period=BACKTEST_PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )

            stock = clean_download(
                raw
            )

        except Exception as error:

            print(
                f"WARNING: failed to "
                f"download {symbol}: "
                f"{error}"
            )

            continue

        if stock.empty:

            print(
                f"WARNING: no usable "
                f"data for {symbol}; "
                "skipping."
            )

            continue

        if len(stock) < MIN_HISTORY + 5:

            print(
                f"WARNING: insufficient "
                f"history for {symbol}; "
                "skipping."
            )

            continue

        try:

            features = build_features(
                stock,
                market,
            )

        except Exception as error:

            print(
                f"WARNING: feature "
                f"calculation failed "
                f"for {symbol}: "
                f"{error}"
            )

            continue

        successful_symbols += 1

        # ----------------------------------------------------
        # CREATE OBSERVATIONS
        # ----------------------------------------------------

        max_index = (
            len(features)
            - 5
        )

        for index in range(
            MIN_HISTORY - 1,
            max_index,
        ):

            signal_date = (
                features.index[
                    index
                ]
            )

            current_close = float(
                features[
                    "close"
                ].iloc[
                    index
                ]
            )

            if not np.isfinite(
                current_close
            ):
                continue

            row = (
                features.iloc[
                    index
                ].copy()
            )

            row_dict = {
                "ticker": symbol,
                "date": signal_date,
                "raw_p": row[
                    "raw_p"
                ],
                "trend": row[
                    "trend"
                ],
                "rsi": row[
                    "rsi"
                ],
                "volume_ratio": row[
                    "volume_ratio"
                ],
                "atr_pct": row[
                    "atr_pct"
                ],
                "regime": row[
                    "regime"
                ],
                "close": current_close,
            }

            # ------------------------------------------------
            # FORWARD RETURNS
            #
            # These are deliberately calculated only for
            # outcome measurement. They are NEVER used when
            # generating the signal itself.
            # ------------------------------------------------

            for horizon in [
                1,
                3,
                5,
            ]:

                future_close = float(
                    features[
                        "close"
                    ].iloc[
                        index
                        +
                        horizon
                    ]
                )

                forward_return = (
                    future_close
                    /
                    current_close
                    -
                    1.0
                )

                row_dict[
                    f"ret{horizon}"
                ] = forward_return

            # Five-day outcome used for calibration.
            row_dict["y"] = float(
                row_dict["ret5"] > 0
            )

            all_rows.append(
                row_dict
            )

    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    data = pd.DataFrame(
        all_rows
    )

    if data.empty:

        raise RuntimeError(
            "No candidate observations "
            "were generated."
        )

    # --------------------------------------------------------
    # CHRONOLOGICAL SORT
    # --------------------------------------------------------

    data = (
        data
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # INITIALIZE OUTPUT COLUMNS
    # --------------------------------------------------------

    data["p_cal"] = np.nan

    data["cal_n"] = 0

    data["calibrated"] = False

    data["action"] = "WAIT"

    data["er5"] = np.nan

    data["rr"] = np.nan

    data["fails"] = ""

    # --------------------------------------------------------
    # CHRONOLOGICAL WALK-FORWARD CALIBRATION
    # --------------------------------------------------------

    unique_dates = (
        sorted(
            data[
                "date"
            ].dropna().unique()
        )
    )

    print(
        "\nRunning chronological "
        "walk-forward calibration..."
    )

    for date_number, date in enumerate(
        unique_dates,
        1,
    ):

        current_mask = (
            data["date"]
            ==
            date
        )

        current_indices = (
            data.index[
                current_mask
            ]
        )

        # ----------------------------------------------------
        # CRITICAL:
        #
        # ONLY observations STRICTLY BEFORE today's signal
        # date may be used for calibration.
        # ----------------------------------------------------

        training = data[
            data["date"]
            <
            date
        ]

        if (
            CAL_WINDOW > 0
            and len(training)
            >
            CAL_WINDOW
        ):

            training = training.tail(
                CAL_WINDOW
            )

        calibrator = (
            fit_probability_calibrator(
                training
            )
        )

        for row_index in (
            current_indices
        ):

            raw_probability = float(
                data.at[
                    row_index,
                    "raw_p",
                ]
            )

            if not np.isfinite(
                raw_probability
            ):

                probability = 0.50

            elif calibrator is None:

                # Conservative shrinkage when
                # insufficient historical data
                # exists for calibration.
                probability = (
                    0.50
                    +
                    0.35
                    *
                    (
                        raw_probability
                        -
                        0.50
                    )
                )

            else:

                probability = float(
                    calibrator.predict(
                        [
                            raw_probability
                        ]
                    )[0]
                )

                data.at[
                    row_index,
                    "calibrated",
                ] = True

            # ------------------------------------------------
            # Conservative probability range.
            #
            # This prevents the system from reporting
            # unrealistic certainty.
            # ------------------------------------------------

            probability = float(
                np.clip(
                    probability,
                    0.30,
                    0.70,
                )
            )

            data.at[
                row_index,
                "p_cal",
            ] = probability

            data.at[
                row_index,
                "cal_n",
            ] = len(training)

            (
                action,
                expected_return,
                risk_reward,
                failures,
            ) = classify_signal(
                data.loc[
                    row_index
                ],
                probability,
            )

            data.at[
                row_index,
                "action",
            ] = action

            data.at[
                row_index,
                "er5",
            ] = expected_return

            data.at[
                row_index,
                "rr",
            ] = risk_reward

            data.at[
                row_index,
                "fails",
            ] = ",".join(
                failures
            )

        if (
            date_number == 1
            or
            date_number % 100 == 0
            or
            date_number == len(
                unique_dates
            )
        ):

            print(
                f"Calibration date "
                f"[{date_number}/"
                f"{len(unique_dates)}]"
            )

    # --------------------------------------------------------
    # TRANSACTION COSTS
    # --------------------------------------------------------

    # Cost is charged once for entry and once for exit.
    total_round_trip_cost = (
        2.0
        *
        (
            COST_BPS
            +
            SLIPPAGE_BPS
        )
        /
        10000.0
    )

    for horizon in [
        1,
        3,
        5,
    ]:

        data[
            f"net{horizon}"
        ] = (
            data[
                f"ret{horizon}"
            ]
            -
            total_round_trip_cost
        )

    # --------------------------------------------------------
    # OOS SPLIT
    #
    # Chronological, not random.
    # --------------------------------------------------------

    cutoff = data[
        "date"
    ].quantile(
        1.0
        -
        OOS_FRACTION
    )

    data["sample"] = np.where(
        data["date"] >= cutoff,
        "OOS",
        "WALK_FORWARD",
    )

    oos = data[
        data["sample"]
        ==
        "OOS"
    ].copy()

    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    # --------------------------------------------------------
    # SAVE MAIN WALK-FORWARD AUDIT
    # --------------------------------------------------------

    walkforward_file = (
        AUDIT_DIR
        /
        f"walkforward_v6_3_16_"
        f"{timestamp}.csv"
    )

    data.to_csv(
        walkforward_file,
        index=False,
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    performance = (
        performance_table(
            data
        )
    )

    performance_file = (
        AUDIT_DIR
        /
        f"action_group_performance_"
        f"v6_3_16_"
        f"{timestamp}.csv"
    )

    performance.to_csv(
        performance_file,
        index=False,
    )

    # --------------------------------------------------------
    # OOS PERFORMANCE
    # --------------------------------------------------------

    oos_performance = (
        performance_table(
            oos
        )
    )

    oos_file = (
        AUDIT_DIR
        /
        f"oos_performance_"
        f"v6_3_16_"
        f"{timestamp}.csv"
    )

    oos_performance.to_csv(
        oos_file,
        index=False,
    )

    # --------------------------------------------------------
    # PROBABILITY CALIBRATION
    # --------------------------------------------------------

    calibration = (
        probability_calibration_table(
            data
        )
    )

    calibration_file = (
        AUDIT_DIR
        /
        f"probability_calibration_"
        f"v6_3_16_"
        f"{timestamp}.csv"
    )

    calibration.to_csv(
        calibration_file,
        index=False,
    )

    # --------------------------------------------------------
    # FILTER PERFORMANCE
    # --------------------------------------------------------

    filter_table = (
        filter_performance_table(
            data
        )
    )

    filter_file = (
        AUDIT_DIR
        /
        f"filter_performance_"
        f"v6_3_16_"
        f"{timestamp}.csv"
    )

    filter_table.to_csv(
        filter_file,
        index=False,
    )

    # --------------------------------------------------------
    # CALIBRATION QUALITY
    # --------------------------------------------------------

    valid = data.dropna(
        subset=[
            "p_cal",
            "y",
        ]
    )

    if valid.empty:

        brier = np.nan
        logloss = np.nan

    else:

        brier = brier_score_loss(
            valid["y"].astype(
                int
            ),
            valid["p_cal"],
        )

        try:

            logloss = log_loss(
                valid["y"].astype(
                    int
                ),
                np.clip(
                    valid["p_cal"],
                    1e-6,
                    1.0 - 1e-6,
                ),
                labels=[
                    0,
                    1,
                ],
            )

        except Exception:

            logloss = np.nan

    # --------------------------------------------------------
    # ACTION COUNTS
    # --------------------------------------------------------

    action_counts = (
        data[
            "action"
        ]
        .value_counts()
        .rename_axis(
            "action"
        )
        .reset_index(
            name="count"
        )
    )

    action_counts_file = (
        AUDIT_DIR
        /
        f"action_counts_"
        f"v6_3_16_"
        f"{timestamp}.csv"
    )

    action_counts.to_csv(
        action_counts_file,
        index=False,
    )

    # --------------------------------------------------------
    # OOS COUNTS
    # --------------------------------------------------------

    oos_action_counts = (
        oos[
            "action"
        ]
        .value_counts()
        .rename_axis(
            "action"
        )
        .reset_index(
            name="count"
        )
    )

    oos_action_counts_file = (
        AUDIT_DIR
        /
        f"oos_action_counts_"
        f"v6_3_16_"
        f"{timestamp}.csv"
    )

    oos_action_counts.to_csv(
        oos_action_counts_file,
        index=False,
    )

    # ========================================================
    # PRINT REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{VERSION} WALK-FORWARD BACKTEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nTotal candidate observations:",
        len(data),
    )

    print(
        "Successful symbols:",
        successful_symbols,
    )

    print(
        "Unique symbols:",
        data[
            "ticker"
        ].nunique(),
    )

    print(
        "Unique signal dates:",
        data[
            "date"
        ].nunique(),
    )

    # --------------------------------------------------------
    # ACTION COUNTS
    # --------------------------------------------------------

    print(
        "\nACTION COUNTS:"
    )

    if action_counts.empty:

        print(
            "No actions."
        )

    else:

        print(
            action_counts.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # FILTER PERFORMANCE
    # --------------------------------------------------------

    print(
        "\nFILTER PERFORMANCE:"
    )

    if filter_table.empty:

        print(
            "No filter statistics."
        )

    else:

        print(
            filter_table.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # FULL PERFORMANCE
    # --------------------------------------------------------

    print(
        "\nACTION GROUP PERFORMANCE:"
    )

    if performance.empty:

        print(
            "No completed outcomes."
        )

    else:

        print(
            performance.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # OOS PERFORMANCE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "UNTOUCHED CHRONOLOGICAL OOS "
        "PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print(
        f"OOS cutoff date: {cutoff}"
    )

    print(
        f"OOS observations: {len(oos)}"
    )

    print(
        "\nOOS ACTION COUNTS:"
    )

    if oos_action_counts.empty:

        print(
            "No OOS actions."
        )

    else:

        print(
            oos_action_counts.to_string(
                index=False
            )
        )

    print(
        "\nOOS PERFORMANCE:"
    )

    if oos_performance.empty:

        print(
            "No completed OOS outcomes."
        )

    else:

        print(
            oos_performance.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # CALIBRATION
    # --------------------------------------------------------

    print(
        "\nPROBABILITY CALIBRATION:"
    )

    if calibration.empty:

        print(
            "No calibration data."
        )

    else:

        print(
            calibration.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # CALIBRATION METRICS
    # --------------------------------------------------------

    print(
        "\nCALIBRATION QUALITY:"
    )

    if np.isfinite(
        brier
    ):

        print(
            f"Brier score: "
            f"{brier:.6f}"
        )

    else:

        print(
            "Brier score: N/A"
        )

    if np.isfinite(
        logloss
    ):

        print(
            f"Log loss:    "
            f"{logloss:.6f}"
        )

    else:

        print(
            "Log loss:    N/A"
        )

    print(
        "Calibrated observations:",
        int(
            data[
                "calibrated"
            ].sum()
        ),
    )

    print(
        "Minimum calibration sample:",
        MIN_CAL,
    )

    print(
        "Calibration window:",
        CAL_WINDOW,
    )

    print(
        "Transaction cost:",
        f"{COST_BPS:.1f} bps per side",
    )

    print(
        "Slippage:",
        f"{SLIPPAGE_BPS:.1f} bps per side",
    )

    # --------------------------------------------------------
    # TRADE QUALITY
    # --------------------------------------------------------

    trade_data = data[
        data["action"]
        ==
        "TRADE"
    ]

    print(
        "\nTRADE OBSERVATIONS:",
        len(trade_data),
    )

    if not trade_data.empty:

        print(
            "\nTRADE 5-DAY SUMMARY:"
        )

        trade5 = (
            trade_data[
                "net5"
            ]
            .dropna()
        )

        if not trade5.empty:

            print(
                f"Observations: "
                f"{len(trade5)}"
            )

            print(
                f"Win rate: "
                f"{(
                    trade5 > 0
                ).mean():.4f}"
            )

            print(
                f"Average net return: "
                f"{trade5.mean():.6f}"
            )

            print(
                f"Median net return: "
                f"{trade5.median():.6f}"
            )

            winners = (
                trade5[
                    trade5 > 0
                ]
            )

            losers = (
                trade5[
                    trade5 <= 0
                ]
            )

            if not losers.empty:

                pf = (
                    winners.sum()
                    /
                    abs(
                        losers.sum()
                    )
                )

                print(
                    f"Profit factor: "
                    f"{pf:.4f}"
                )

    # --------------------------------------------------------
    # SYMBOL-LEVEL TRADE STATISTICS
    # --------------------------------------------------------

    print(
        "\nTRADE SYMBOLS "
        "WITH >=20 OBSERVATIONS:"
    )

    if trade_data.empty:

        symbol_table = pd.DataFrame()

    else:

        grouped = (
            trade_data
            .groupby(
                "ticker"
            )
        )

        symbol_rows = []

        for ticker, group in grouped:

            returns = (
                group[
                    "net5"
                ]
                .dropna()
            )

            if len(returns) < 20:
                continue

            symbol_rows.append(
                {
                    "ticker": ticker.replace(
                        ".NS",
                        "",
                    ),
                    "observations": len(
                        returns
                    ),
                    "win_rate": (
                        returns > 0
                    ).mean(),
                    "average_return": (
                        returns.mean()
                    ),
                    "median_return": (
                        returns.median()
                    ),
                }
            )

        symbol_table = (
            pd.DataFrame(
                symbol_rows
            )
        )

    if symbol_table.empty:

        print(
            "No symbols with "
            ">=20 TRADE observations."
        )

    else:

        symbol_table = (
            symbol_table
            .sort_values(
                "average_return",
                ascending=False,
            )
        )

        print(
            symbol_table.to_string(
                index=False
            )
        )

    symbol_file = (
        AUDIT_DIR
        /
        f"trade_symbol_performance_"
        f"v6_3_16_"
        f"{timestamp}.csv"
    )

    symbol_table.to_csv(
        symbol_file,
        index=False,
    )

    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    print(
        "\nDATA QUALITY:"
    )

    print(
        "Unique symbols tested:",
        data[
            "ticker"
        ].nunique(),
    )

    print(
        "Unique signal dates:",
        data[
            "date"
        ].nunique(),
    )

    print(
        "Total observations:",
        len(data),
    )

    print(
        "TRADE observations:",
        int(
            (
                data[
                    "action"
                ]
                ==
                "TRADE"
            ).sum()
        ),
    )

    print(
        "WATCH observations:",
        int(
            (
                data[
                    "action"
                ]
                ==
                "WATCH"
            ).sum()
        ),
    )

    print(
        "WAIT observations:",
        int(
            (
                data[
                    "action"
                ]
                ==
                "WAIT"
            ).sum()
        ),
    )

    # --------------------------------------------------------
    # FILES CREATED
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FILES CREATED"
    )

    print(
        "=" * 70
    )

    print(
        walkforward_file
    )

    print(
        performance_file
    )

    print(
        oos_file
    )

    print(
        calibration_file
    )

    print(
        filter_file
    )

    print(
        action_counts_file
    )

    print(
        oos_action_counts_file
    )

    print(
        symbol_file
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{VERSION} BACKTEST COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is historical research only."
    )

    print(
        "Do NOT promote this model "
        "to real-money trading solely "
        "from this backtest."
    )

    print(
        "Use a completely untouched "
        "future period for independent "
        "validation before live deployment."
    )

    print(
        "P(UP) is an empirical calibrated "
        "model estimate and is NOT a "
        "guarantee of profit."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
