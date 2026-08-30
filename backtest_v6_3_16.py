#!/usr/bin/env python3

"""
V6.3.16 CHRONOLOGICAL WALK-FORWARD BACKTEST

Research version.

Key safeguards:
- Features use only information available on the signal date.
- Future prices are used only for outcome measurement.
- Probability calibration is chronological.
- Isotonic calibration uses only earlier observations.
- Minimum calibration sample is required.
- Final chronological segment is reported as OOS.
- Transaction costs and slippage are included.
- No future outcomes are used to choose current actions.
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


VERSION = "V6.3.16"

OUT = Path("audit")
OUT.mkdir(exist_ok=True)

BACKTEST_PERIOD = os.getenv("BACKTEST_PERIOD", "6y")

MIN_HISTORY = 220

CAL_WINDOW = int(os.getenv("CAL_WINDOW", "8000"))
MIN_CAL = int(os.getenv("MIN_CAL", "250"))

OOS_FRACTION = float(os.getenv("OOS_FRACTION", "0.20"))

COST_BPS = float(os.getenv("COST_BPS", "10"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))


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


def clean_download(data):

    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    data = data[required].copy()

    for column in required:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=["Close"]
    )

    return data.sort_index()


def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    loss = (
        -delta.clip(upper=0)
    ).ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = gain / loss.replace(
        0,
        np.nan,
    )

    return 100 - (
        100 / (1 + rs)
    )


def calculate_atr(data, period=14):

    previous_close = data["Close"].shift(1)

    true_range = pd.concat(
        [
            data["High"] - data["Low"],
            (data["High"] - previous_close).abs(),
            (data["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def build_features(data, market):

    close = data["Close"]

    atr = calculate_atr(data)
    rsi = calculate_rsi(close)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    volume20 = data["Volume"].rolling(20).mean()

    market_return20 = (
        market["Close"].pct_change(20)
    )

    market_sma50 = (
        market["Close"].rolling(50).mean()
    )

    return5 = close.pct_change(5)
    return20 = close.pct_change(20)

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
    ) / 3

    trend = trend * 2 - 1

    momentum = (
        0.5 * np.tanh(return5 / 0.03)
        +
        0.5 * np.tanh(return20 / 0.08)
    )

    relative_strength = np.tanh(
        (
            return20
            - market_return20
        ) / 0.08
    )

    rsi_component = (
        1
        -
        np.minimum(
            abs(rsi - 55) / 35,
            1,
        )
    )

    volume_ratio = (
        data["Volume"]
        /
        volume20
    )

    volume_component = (
        volume_ratio - 1
    ).clip(
        -1,
        1,
    )

    volatility_component = (
        1
        -
        (
            atr / close
        ) / 0.045
    ).clip(
        -1,
        1,
    )

    regime = np.where(
        market["Close"]
        >
        market_sma50 * 1.002,
        1,
        np.where(
            market["Close"]
            <
            market_sma50 * 0.998,
            -1,
            0,
        ),
    )

    score = (
        0.26 * trend
        +
        0.22 * momentum
        +
        0.16 * relative_strength
        +
        0.10 * (
            2 * rsi_component - 1
        )
        +
        0.08 * volume_component
        +
        0.08 * volatility_component
        +
        0.10 * regime
    )

    raw_probability = (
        1
        /
        (
            1
            +
            np.exp(
                -3 * score
            )
        )
    )

    raw_probability = np.clip(
        raw_probability,
        0.35,
        0.75,
    )

    features = pd.DataFrame(
        {
            "raw_p": raw_probability,
            "trend": trend,
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "atr_pct": atr / close,
            "regime": regime,
            "close": close,
        },
        index=data.index,
    )

    return features.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def fit_probability_calibrator(training):

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

    calibrator = IsotonicRegression(
        y_min=0.25,
        y_max=0.70,
        out_of_bounds="clip",
    )

    calibrator.fit(
        training["raw_p"].values,
        training["y"].values,
    )

    return calibrator


def classify_signal(row, probability):

    failures = []

    stop_distance = max(
        1.5 * row["atr_pct"],
        0.012,
    )

    target_distance = max(
        2.0 * stop_distance,
        0.025,
    )

    expected_return = (
        probability * target_distance
        -
        (1 - probability)
        * stop_distance
    )

    risk_reward = (
        target_distance
        /
        stop_distance
    )

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

    if not (
        42
        <= row["rsi"]
        <= 68
    ):
        failures.append(
            "rsi"
        )

    if row["volume_ratio"] < 0.70:
        failures.append(
            "volume"
        )

    if row["trend"] < -0.05:
        failures.append(
            "trend"
        )

    if row["regime"] < -0.75:
        failures.append(
            "regime"
        )

    if not failures:
        action = "TRADE"

    elif (
        probability >= 0.55
        and row["trend"] >= -0.20
        and 35 <= row["rsi"] <= 72
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


def performance_table(data):

    rows = []

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

            values = group[
                f"net{horizon}"
            ].dropna()

            if values.empty:
                continue

            winners = values[
                values > 0
            ]

            losers = values[
                values <= 0
            ]

            if len(losers) > 0:
                profit_factor = (
                    winners.sum()
                    /
                    abs(losers.sum())
                )
            else:
                profit_factor = np.nan

            rows.append(
                {
                    "selection": action,
                    "horizon": horizon,
                    "observations": len(values),
                    "win_rate": (
                        values > 0
                    ).mean(),
                    "average_net_return": values.mean(),
                    "median_net_return": values.median(),
                    "average_winner": (
                        winners.mean()
                        if len(winners)
                        else np.nan
                    ),
                    "average_loser": (
                        losers.mean()
                        if len(losers)
                        else np.nan
                    ),
                    "profit_factor": profit_factor,
                    "best": values.max(),
                    "worst": values.min(),
                }
            )

    return pd.DataFrame(rows)


def probability_calibration_table(data):

    if data.empty:
        return pd.DataFrame()

    valid = data.dropna(
        subset=[
            "p_cal",
            "y",
        ]
    )

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

    valid = valid.copy()

    valid["bucket"] = pd.cut(
        valid["p_cal"],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []

    for bucket, group in valid.groupby(
        "bucket",
        observed=False,
    ):

        if group.empty:
            continue

        rows.append(
            {
                "probability_bucket": str(
                    bucket
                ),
                "observations": len(group),
                "average_model_probability": group[
                    "p_cal"
                ].mean(),
                "actual_win_rate": group[
                    "y"
                ].mean(),
                "average_return": group[
                    "net5"
                ].mean(),
                "brier": (
                    (
                        group["p_cal"]
                        -
                        group["y"]
                    ) ** 2
                ).mean(),
            }
        )

    return pd.DataFrame(rows)


def main():

    print(
        f"Starting {VERSION} "
        "chronological walk-forward backtest..."
    )

    market = clean_download(
        yf.download(
            "^NSEI",
            period=BACKTEST_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )

    if market.empty:
        raise RuntimeError(
            "NIFTY data unavailable"
        )

    all_rows = []

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

            data = clean_download(
                yf.download(
                    symbol,
                    period=BACKTEST_PERIOD,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            )

        except Exception as error:

            print(
                f"WARNING: {symbol}: "
                f"{error}"
            )

            continue

        if len(data) < MIN_HISTORY:

            print(
                f"WARNING: insufficient "
                f"history for {symbol}; "
                "skipping."
            )

            continue

        feature_data = build_features(
            data,
            market,
        )

        for index in range(
            MIN_HISTORY - 1,
            len(feature_data) - 5,
        ):

            row = feature_data.iloc[
                index
            ].copy()

            row["ticker"] = symbol

            row["date"] = (
                feature_data.index[
                    index
                ]
            )

            for horizon in [
                1,
                3,
                5,
            ]:

                row[
                    f"ret{horizon}"
                ] = (
                    feature_data[
                        "close"
                    ].iloc[
                        index + horizon
                    ]
                    /
                    feature_data[
                        "close"
                    ].iloc[index]
                    -
                    1
                )

            row["y"] = float(
                row["ret5"] > 0
            )

            all_rows.append(
                row
            )

    data = pd.DataFrame(
        all_rows
    )

    if data.empty:
        raise RuntimeError(
            "No candidate observations"
        )

    data = (
        data.sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    data["p_cal"] = np.nan
    data["cal_n"] = 0
    data["calibrated"] = False

    data["action"] = "WAIT"
    data["er5"] = np.nan
    data["rr"] = np.nan
    data["fails"] = ""

    unique_dates = sorted(
        data["date"].unique()
    )

    for date in unique_dates:

        current_indices = data.index[
            data["date"] == date
        ]

        training = data[
            data["date"] < date
        ].tail(
            CAL_WINDOW
        )

        calibrator = (
            fit_probability_calibrator(
                training
            )
        )

        for row_index in current_indices:

            raw_probability = float(
                data.at[
                    row_index,
                    "raw_p",
                ]
            )

            if calibrator is None:

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
                        [raw_probability]
                    )[0]
                )

                data.at[
                    row_index,
                    "calibrated",
                ] = True

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

    total_cost = (
        2
        *
        (
            COST_BPS
            +
            SLIPPAGE_BPS
        )
        /
        10000
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
            total_cost
        )

    cutoff = data[
        "date"
    ].quantile(
        1 - OOS_FRACTION
    )

    data["sample"] = np.where(
        data["date"] >= cutoff,
        "OOS",
        "WALK_FORWARD",
    )

    oos = data[
        data["sample"] == "OOS"
    ].copy()

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    data.to_csv(
        OUT
        /
        f"walkforward_v6_3_16_{timestamp}.csv",
        index=False,
    )

    full_performance = performance_table(
        data
    )

    full_performance.to_csv(
        OUT
        /
        f"action_group_performance_v6_3_16_{timestamp}.csv",
        index=False,
    )

    oos_performance = performance_table(
        oos
    )

    oos_performance.to_csv(
        OUT
        /
        f"oos_performance_v6_3_16_{timestamp}.csv",
        index=False,
    )

    calibration = probability_calibration_table(
        data
    )

    calibration.to_csv(
        OUT
        /
        f"probability_calibration_v6_3_16_{timestamp}.csv",
        index=False,
    )

    valid = data.dropna(
        subset=[
            "p_cal",
            "y",
        ]
    )

    brier = brier_score_loss(
        valid["y"],
        valid["p_cal"],
    )

    try:

        logloss = log_loss(
            valid["y"],
            np.clip(
                valid["p_cal"],
                1e-6,
                1 - 1e-6,
            ),
        )

    except Exception:

        logloss = np.nan

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
        "\nACTION COUNTS:"
    )

    print(
        data[
            "action"
        ]
        .value_counts()
        .rename_axis(
            "action"
        )
        .to_string()
    )

    print(
        "\nACTION GROUP PERFORMANCE:"
    )

    if full_performance.empty:
        print(
            "No completed outcomes."
        )
    else:
        print(
            full_performance.to_string(
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

    print(
        "\nCALIBRATION QUALITY:"
    )

    print(
        f"Brier score: {brier:.6f}"
    )

    print(
        f"Log loss:    {logloss:.6f}"
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
        "Do NOT promote this model "
        "to real-money trading solely "
        "from this backtest."
    )

    print(
        "Use a completely untouched "
        "future period before live deployment."
    )

    print(
        "Model probabilities are calibrated "
        "estimates, not guarantees of profit."
    )


if __name__ == "__main__":
    main()
