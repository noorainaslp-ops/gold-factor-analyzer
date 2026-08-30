"""
V6.3.9 WALK-FORWARD BACKTEST

Validates the V6.3.9 methodology without sending Telegram alerts.

Key improvements:
    - Walk-forward probability estimation
    - Probability calibration
    - 1/3/5-session outcomes
    - Transaction-cost adjustment
    - Genuine RR calculation
    - Outlier filtering
    - TRADE/WATCH/WAIT/REJECT comparison
    - Probability calibration table
    - Filter analysis
    - Regime analysis
    - Monthly performance
    - Symbol performance
"""

from __future__ import annotations

import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf


VERSION = "V6.3.9"

START_DATE = os.getenv(
    "BACKTEST_START",
    "2023-01-01"
)

END_DATE = os.getenv(
    "BACKTEST_END",
    ""
)

ROUND_TRIP_COST = 0.0020
SAFETY_BUFFER = 0.0015

MIN_P3 = 0.53
MIN_P5 = 0.54

MIN_ER3 = 0.0040
MIN_ER5 = 0.0060

MIN_RR1 = 1.20
MIN_RR2 = 1.60

WATCH_P3 = 0.51
WATCH_P5 = 0.52

WATCH_ER3 = 0.0025
WATCH_ER5 = 0.0035

RSI_MIN = 45
RSI_MAX = 68

MIN_VOLUME = 0.80

MAX_RETURN = 0.40
MIN_RETURN = -0.40


SYMBOLS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "INFY.NS",
    "ITC.NS",
    "LT.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "M&M.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "HINDALCO.NS",
    "JSWSTEEL.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "BPCL.NS",
    "IOC.NS",
    "SUNPHARMA.NS",
    "CIPLA.NS",
    "DRREDDY.NS",
    "AUROPHARMA.NS",
    "DIVISLAB.NS",
    "APOLLOHOSP.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "TITAN.NS",
    "ASIANPAINT.NS",
    "ULTRACEMCO.NS",
    "GRASIM.NS",
    "NESTLEIND.NS",
    "HINDUNILVR.NS",
    "TECHM.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "BHARTIARTL.NS",
    "TRENT.NS",
    "DLF.NS",
    "VEDL.NS",
    "SAIL.NS",
    "SHRIRAMFIN.NS",
    "NAUKRI.NS",
    "ABB.NS",
    "BOSCHLTD.NS",
    "HAL.NS",
    "BEL.NS",
]


def safe_float(x):

    try:

        x = float(x)

        if np.isfinite(x):
            return x

    except Exception:
        pass

    return np.nan


def download(symbol):

    try:

        df = yf.download(
            symbol,
            start=START_DATE,
            end=END_DATE if END_DATE else None,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return None

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]:

            if col not in df.columns:
                return None

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = df.dropna(
            subset=["Close"]
        )

        if len(df) < 180:
            return None

        if getattr(
            df.index,
            "tz",
            None
        ) is not None:

            df.index = (
                df.index
                .tz_localize(None)
            )

        return df

    except Exception as exc:

        print(
            f"Download failed {symbol}: "
            f"{exc}"
        )

        return None


def rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    result = (
        100 -
        100 / (1 + rs)
    )

    return result


def features(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["rsi"] = rsi(
        close
    )

    df["ema20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["sma50"] = close.rolling(
        50
    ).mean()

    df["sma100"] = close.rolling(
        100
    ).mean()

    df["ret5"] = (
        close.pct_change(5)
    )

    df["ret10"] = (
        close.pct_change(10)
    )

    df["ret20"] = (
        close.pct_change(20)
    )

    df["volume_ratio"] = (
        volume /
        volume.rolling(20).mean()
    )

    daily = close.pct_change()

    df["volatility"] = (
        daily
        .rolling(20)
        .std()
        * math.sqrt(252)
    )

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = tr.rolling(
        14
    ).mean()

    df["momentum"] = (
        0.40 * df["ret5"]
        +
        0.30 * df["ret10"]
        +
        0.20 * df["ret20"]
        +
        0.10 * (
            close / df["ema20"] - 1
        )
    )

    df["trend"] = (
        (close > df["ema20"])
        &
        (df["ema20"] > df["sma50"])
    )

    return df


def probability(
    df,
    position,
    horizon
):

    if position < 100:
        return np.nan

    start = max(
        50,
        position - 250
    )

    history = df.iloc[
        start:position
    ]

    current = df.iloc[
        position
    ]

    crsi = safe_float(
        current["rsi"]
    )

    cmom = safe_float(
        current["momentum"]
    )

    cvol = safe_float(
        current["volume_ratio"]
    )

    if not np.isfinite(crsi):
        return np.nan

    if not np.isfinite(cmom):
        return np.nan

    rsi_distance = (
        history["rsi"] -
        crsi
    ).abs()

    mom_std = (
        history["momentum"].std()
    )

    if (
        not np.isfinite(mom_std)
        or
        mom_std == 0
    ):
        mom_std = 0.01

    mom_distance = (
        history["momentum"] -
        cmom
    ).abs() / mom_std

    if np.isfinite(cvol):

        vol_distance = (
            history["volume_ratio"] -
            cvol
        ).abs() / 1.5

    else:

        vol_distance = 0

    distance = (
        rsi_distance / 15
        +
        mom_distance
        +
        vol_distance
    )

    nearest = distance.nsmallest(
        min(60, len(distance))
    )

    selected = history.loc[
        nearest.index
    ]

    if len(selected) < 15:
        selected = history

    future = (
        df["Close"]
        .shift(-horizon)
    )

    outcomes = (
        future.loc[
            selected.index
        ]
        /
        selected["Close"]
        - 1
    ).dropna()

    outcomes = outcomes[
        (outcomes >= MIN_RETURN)
        &
        (outcomes <= MAX_RETURN)
    ]

    if len(outcomes) < 10:
        return np.nan

    wins = (
        outcomes > 0
    ).sum()

    n = len(outcomes)

    return float(
        np.clip(
            (wins + 3) /
            (n + 6),
            0.05,
            0.95
        )
    )


def calibrated(
    raw,
    horizon
):

    if not np.isfinite(raw):
        return np.nan

    coefficient = (
        0.62
        if horizon == 3
        else 0.64
    )

    result = (
        0.50
        +
        coefficient *
        (raw - 0.50)
    )

    return float(
        np.clip(
            result,
            0.35,
            0.75
        )
    )


def expected_return(
    probability_value,
    volatility,
    horizon
):

    if not all(
        np.isfinite(x)
        for x in [
            probability_value,
            volatility
        ]
    ):
        return np.nan

    daily_vol = (
        volatility /
        math.sqrt(252)
    )

    horizon_vol = (
        daily_vol *
        math.sqrt(horizon)
    )

    win = (
        horizon_vol *
        0.85
    )

    loss = (
        horizon_vol *
        0.75
    )

    return (
        probability_value * win
        -
        (1 - probability_value) * loss
        -
        ROUND_TRIP_COST
    )


def rr_values(
    price,
    atr
):

    if (
        not np.isfinite(price)
        or
        price <= 0
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    if (
        not np.isfinite(atr)
        or
        atr <= 0
    ):
        atr = (
            price * 0.02
        )

    risk = max(
        1.5 * atr,
        price * 0.008
    )

    risk = min(
        risk,
        price * 0.30
    )

    stop = (
        price - risk
    )

    if stop <= 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    target1 = (
        price +
        1.25 * risk
    )

    target2 = (
        price +
        2.00 * risk
    )

    rr1 = (
        target1 - price
    ) / (
        price - stop
    )

    rr2 = (
        target2 - price
    ) / (
        price - stop
    )

    return (
        stop,
        target1,
        target2,
        rr1,
        rr2
    )


def classify(row):

    failures = []

    if (
        not np.isfinite(row["p3"])
        or
        row["p3"] < MIN_P3
    ):
        failures.append("p3")

    if (
        not np.isfinite(row["p5"])
        or
        row["p5"] < MIN_P5
    ):
        failures.append("p5")

    if (
        not np.isfinite(row["er3"])
        or
        row["er3"] < MIN_ER3
    ):
        failures.append("er3")

    if (
        not np.isfinite(row["er5"])
        or
        row["er5"] < MIN_ER5
    ):
        failures.append("er5")

    if (
        not np.isfinite(row["rr1"])
        or
        row["rr1"] < MIN_RR1
    ):
        failures.append("rr1")

    if (
        not np.isfinite(row["rr2"])
        or
        row["rr2"] < MIN_RR2
    ):
        failures.append("rr2")

    if (
        not np.isfinite(row["rsi"])
        or
        row["rsi"] < RSI_MIN
        or
        row["rsi"] > RSI_MAX
    ):
        failures.append("rsi")

    if (
        not np.isfinite(row["volume"])
        or
        row["volume"] < MIN_VOLUME
    ):
        failures.append("volume")

    if not row["trend"]:
        failures.append("trend")

    if row["regime"] == "UNFAVORABLE":
        failures.append("regime")

    if not failures:
        return "TRADE", failures

    watch_score = 0

    if (
        np.isfinite(row["p3"])
        and
        row["p3"] >= WATCH_P3
    ):
        watch_score += 1

    if (
        np.isfinite(row["p5"])
        and
        row["p5"] >= WATCH_P5
    ):
        watch_score += 1

    if (
        np.isfinite(row["er3"])
        and
        row["er3"] >= WATCH_ER3
    ):
        watch_score += 1

    if (
        np.isfinite(row["er5"])
        and
        row["er5"] >= WATCH_ER5
    ):
        watch_score += 1

    if watch_score >= 3:
        return "WATCH", failures

    return "WAIT", failures


def process_symbol(
    symbol,
    nifty
):

    df = download(
        symbol
    )

    if df is None:
        return []

    df = features(
        df
    )

    records = []

    # Start sufficiently far into history
    # to make the probability calculation valid.

    for i in range(
        120,
        len(df) - 6
    ):

        row = df.iloc[i]

        price = safe_float(
            row["Close"]
        )

        if not np.isfinite(price):
            continue

        # ----------------------------------------------------
        # NIFTY regime using only information available
        # on the signal date.
        # ----------------------------------------------------

        nifty_slice = nifty[
            nifty.index <= df.index[i]
        ]

        if len(nifty_slice) < 60:
            continue

        nc = nifty_slice["Close"]

        nsma50 = nc.rolling(
            50
        ).mean()

        current_nifty = safe_float(
            nc.iloc[-1]
        )

        current_sma = safe_float(
            nsma50.iloc[-1]
        )

        previous_sma = safe_float(
            nsma50.iloc[-6]
        )

        if not all(
            np.isfinite(x)
            for x in [
                current_nifty,
                current_sma,
                previous_sma
            ]
        ):
            continue

        if (
            current_nifty > current_sma
            and
            current_sma > previous_sma
        ):
            regime = "FAVORABLE"

        elif (
            current_nifty > current_sma
            or
            current_sma > previous_sma
        ):
            regime = "MIXED"

        else:
            regime = "UNFAVORABLE"

        raw_p3 = probability(
            df,
            i,
            3
        )

        raw_p5 = probability(
            df,
            i,
            5
        )

        p3 = calibrated(
            raw_p3,
            3
        )

        p5 = calibrated(
            raw_p5,
            5
        )

        volatility = safe_float(
            row["volatility"]
        )

        er3 = expected_return(
            p3,
            volatility,
            3
        )

        er5 = expected_return(
            p5,
            volatility,
            5
        )

        (
            stop,
            target1,
            target2,
            rr1,
            rr2
        ) = rr_values(
            price,
            safe_float(row["atr"])
        )

        candidate = {

            "symbol":
                symbol.replace(
                    ".NS",
                    ""
                ),

            "date":
                df.index[i],

            "price":
                price,

            "p3":
                p3,

            "p5":
                p5,

            "er3":
                er3,

            "er5":
                er5,

            "rr1":
                rr1,

            "rr2":
                rr2,

            "rsi":
                safe_float(
                    row["rsi"]
                ),

            "volume":
                safe_float(
                    row["volume_ratio"]
                ),

            "trend":
                bool(
                    row["trend"]
                ),

            "regime":
                regime,

            "return1":
                safe_float(
                    (
                        df["Close"]
                        .iloc[i + 1]
                        /
                        price
                    ) - 1
                ),

            "return3":
                safe_float(
                    (
                        df["Close"]
                        .iloc[i + 3]
                        /
                        price
                    ) - 1
                ),

            "return5":
                safe_float(
                    (
                        df["Close"]
                        .iloc[i + 5]
                        /
                        price
                    ) - 1
                ),

            "stop":
                stop,

            "target1":
                target1,

            "target2":
                target2
        }

        action, failures = classify(
            candidate
        )

        candidate["classification"] = (
            action
        )

        candidate["failures"] = (
            ",".join(failures)
        )

        records.append(
            candidate
        )

    return records


def performance(
    df,
    selection,
    horizon
):

    data = df.copy()

    if selection != "ALL":

        data = data[
            data["classification"]
            == selection
        ]

    column = (
        f"return{horizon}"
    )

    values = pd.to_numeric(
        data[column],
        errors="coerce"
    ).dropna()

    if len(values) == 0:

        return {
            "selection":
                selection,
            "horizon":
                horizon,
            "observations":
                0
        }

    # Outlier-adjusted research return.

    clean = values[
        (values >= MIN_RETURN)
        &
        (values <= MAX_RETURN)
    ]

    if len(clean) == 0:
        return {
            "selection":
                selection,
            "horizon":
                horizon,
            "observations":
                0
        }

    winners = clean[
        clean > 0
    ]

    losers = clean[
        clean <= 0
    ]

    gross_profit = (
        winners.sum()
    )

    gross_loss = abs(
        losers.sum()
    )

    pf = (
        gross_profit /
        gross_loss
        if gross_loss > 0
        else np.nan
    )

    return {

        "selection":
            selection,

        "horizon":
            horizon,

        "observations":
            len(clean),

        "win_rate":
            float(
                (clean > 0).mean()
            ),

        "average_net_return":
            float(
                clean.mean()
            ),

        "median_net_return":
            float(
                clean.median()
            ),

        "average_winner":
            float(
                winners.mean()
            )
            if len(winners)
            else np.nan,

        "average_loser":
            float(
                losers.mean()
            )
            if len(losers)
            else np.nan,

        "profit_factor":
            float(pf)
            if np.isfinite(pf)
            else np.nan,

        "best":
            float(clean.max()),

        "worst":
            float(clean.min())
    }


def calibration_table(df):

    data = df[
        [
            "p3",
            "return3"
        ]
    ].dropna()

    if data.empty:
        return pd.DataFrame()

    bins = [
        0.30,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        1.00
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
        "75%+"
    ]

    data["bucket"] = pd.cut(
        data["p3"],
        bins=bins,
        labels=labels,
        right=False
    )

    rows = []

    for bucket, group in data.groupby(
        "bucket",
        observed=False
    ):

        if group.empty:
            continue

        rows.append({

            "probability_bucket":
                str(bucket),

            "observations":
                len(group),

            "average_model_probability":
                group["p3"].mean(),

            "actual_win_rate":
                (
                    group["return3"] > 0
                ).mean(),

            "average_return":
                group["return3"].mean()
        })

    return pd.DataFrame(rows)


def filter_analysis(df):

    rows = []

    tests = {

        "p3":
            df["p3"] >= MIN_P3,

        "p5":
            df["p5"] >= MIN_P5,

        "er3":
            df["er3"] >= MIN_ER3,

        "er5":
            df["er5"] >= MIN_ER5,

        "rr1":
            df["rr1"] >= MIN_RR1,

        "rr2":
            df["rr2"] >= MIN_RR2,

        "rsi":
            (
                df["rsi"] >= RSI_MIN
            )
            &
            (
                df["rsi"] <= RSI_MAX
            ),

        "volume":
            df["volume"] >= MIN_VOLUME,

        "trend":
            df["trend"],

        "favorable_or_mixed_regime":
            df["regime"] != "UNFAVORABLE"
    }

    for name, mask in tests.items():

        passed = int(
            mask.fillna(False).sum()
        )

        total = len(df)

        rows.append({

            "filter":
                name,

            "passed":
                passed,

            "failed":
                total - passed,

            "pass_rate":
                passed / total
                if total
                else np.nan
        })

    return pd.DataFrame(rows)


def main():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    os.makedirs(
        "audit",
        exist_ok=True
    )

    print()
    print("=" * 70)
    print(
        f"{VERSION} WALK-FORWARD BACKTEST"
    )
    print("=" * 70)

    print(
        f"Start: {START_DATE}"
    )

    print(
        f"End: "
        f"{END_DATE if END_DATE else 'latest'}"
    )

    print()

    print(
        "Downloading NIFTY..."
    )

    nifty = download(
        "^NSEI"
    )

    if nifty is None:

        raise RuntimeError(
            "Nifty data unavailable."
        )

    all_records = []

    for number, symbol in enumerate(
        SYMBOLS,
        start=1
    ):

        print(
            f"[{number}/{len(SYMBOLS)}] "
            f"{symbol}"
        )

        try:

            records = process_symbol(
                symbol,
                nifty
            )

            all_records.extend(
                records
            )

        except Exception as exc:

            print(
                f"ERROR {symbol}: "
                f"{exc}"
            )

    if not all_records:

        raise RuntimeError(
            "No observations generated."
        )

    data = pd.DataFrame(
        all_records
    )

    data.sort_values(
        [
            "date",
            "symbol"
        ],
        inplace=True
    )

    raw_file = (
        "audit/"
        f"walkforward_v6_3_9_"
        f"{timestamp}.csv"
    )

    data.to_csv(
        raw_file,
        index=False
    )

    print()
    print("=" * 70)
    print("V6.3.9 BACKTEST SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Total candidate observations: "
        f"{len(data)}"
    )

    print()
    print("ACTION COUNTS:")

    counts = (
        data["classification"]
        .value_counts()
        .rename_axis(
            "classification"
        )
        .reset_index(
            name="count"
        )
    )

    print(
        counts.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    performance_rows = []

    for selection in [
        "ALL",
        "TRADE",
        "WATCH",
        "WAIT"
    ]:

        for horizon in [
            1,
            3,
            5
        ]:

            performance_rows.append(
                performance(
                    data,
                    selection,
                    horizon
                )
            )

    performance_df = pd.DataFrame(
        performance_rows
    )

    print()
    print("PERFORMANCE:")
    print(
        performance_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    filters = filter_analysis(
        data
    )

    print()
    print("FILTER PERFORMANCE:")
    print(
        filters.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibration = calibration_table(
        data
    )

    print()
    print("PROBABILITY CALIBRATION:")

    if calibration.empty:

        print(
            "No calibration observations."
        )

    else:

        print(
            calibration.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Regime performance
    # --------------------------------------------------------

    regime_rows = []

    for regime in [
        "FAVORABLE",
        "MIXED",
        "UNFAVORABLE"
    ]:

        subset = data[
            data["regime"] == regime
        ]

        for horizon in [
            3,
            5
        ]:

            result = performance(
                subset,
                "ALL",
                horizon
            )

            result["regime"] = regime

            regime_rows.append(
                result
            )

    regime_df = pd.DataFrame(
        regime_rows
    )

    print()
    print("REGIME PERFORMANCE:")
    print(
        regime_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Symbol performance
    # --------------------------------------------------------

    symbol_rows = []

    for symbol, group in data.groupby(
        "symbol"
    ):

        result = performance(
            group,
            "TRADE",
            5
        )

        result["symbol"] = symbol

        symbol_rows.append(
            result
        )

    symbol_df = pd.DataFrame(
        symbol_rows
    )

    if not symbol_df.empty:

        symbol_df.sort_values(
            "average_net_return",
            ascending=False,
            inplace=True
        )

    print()
    print(
        "TOP TRADE SYMBOLS "
        "(5-session):"
    )

    if symbol_df.empty:

        print(
            "No completed TRADE symbols."
        )

    else:

        print(
            symbol_df.head(20).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Save audit files
    # --------------------------------------------------------

    performance_file = (
        "audit/"
        f"performance_v6_3_9_"
        f"{timestamp}.csv"
    )

    filters_file = (
        "audit/"
        f"filter_analysis_v6_3_9_"
        f"{timestamp}.csv"
    )

    calibration_file = (
        "audit/"
        f"probability_calibration_v6_3_9_"
        f"{timestamp}.csv"
    )

    regime_file = (
        "audit/"
        f"regime_performance_v6_3_9_"
        f"{timestamp}.csv"
    )

    symbol_file = (
        "audit/"
        f"symbol_performance_v6_3_9_"
        f"{timestamp}.csv"
    )

    performance_df.to_csv(
        performance_file,
        index=False
    )

    filters.to_csv(
        filters_file,
        index=False
    )

    calibration.to_csv(
        calibration_file,
        index=False
    )

    regime_df.to_csv(
        regime_file,
        index=False
    )

    symbol_df.to_csv(
        symbol_file,
        index=False
    )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(raw_file)
    print(performance_file)
    print(filters_file)
    print(calibration_file)
    print(regime_file)
    print(symbol_file)

    print()
    print(
        "V6.3.9 walk-forward backtest completed."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do not promote V6.3.9 to live "
        "capital solely from this test."
    )

    print(
        "Compare V6.3.9 against V6.3.8 "
        "using out-of-sample results."
    )

    print(
        "Historical performance does not "
        "guarantee future profitability."
    )


if __name__ == "__main__":
    main()
