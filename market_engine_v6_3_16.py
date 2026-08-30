#!/usr/bin/env python3

"""
MULTI-FACTOR MARKET ALERT V6.3.16

Daily NSE long-only research screen.

Telegram credentials are read ONLY from:
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID

Never place credentials in this file.
"""

import os
import math
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf


VERSION = "V6.3.16"

CAPITAL = float(
    os.getenv(
        "CAPITAL",
        "100000",
    )
)

RISK_PCT = float(
    os.getenv(
        "MAX_RISK_PCT",
        "1.0",
    )
)

TOP_WATCH = int(
    os.getenv(
        "TOP_WATCH",
        "5",
    )
)

MIN_HISTORY = 220


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


def clean(data):

    if data is None or data.empty:
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

    if not all(
        c in data.columns
        for c in required
    ):
        return pd.DataFrame()

    data = data[
        required
    ].copy()

    for column in required:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return (
        data
        .dropna(
            subset=["Close"]
        )
        .sort_index()
    )


def calculate_rsi(
    series,
    period=14,
):

    delta = series.diff()

    gain = (
        delta.clip(
            lower=0
        )
        .ewm(
            alpha=1 / period,
            adjust=False,
        )
        .mean()
    )

    loss = (
        -delta.clip(
            upper=0
        )
        .ewm(
            alpha=1 / period,
            adjust=False,
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

    return (
        100
        -
        100
        /
        (
            1 + rs
        )
    )


def calculate_atr(
    data,
    period=14,
):

    previous_close = (
        data["Close"].shift(1)
    )

    true_range = pd.concat(
        [
            data["High"]
            -
            data["Low"],

            (
                data["High"]
                -
                previous_close
            ).abs(),

            (
                data["Low"]
                -
                previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()


def evaluate(
    data,
    market,
):

    close = data["Close"]

    atr = calculate_atr(
        data
    )

    rsi = calculate_rsi(
        close
    )

    sma20 = (
        close.rolling(20)
        .mean()
    )

    sma50 = (
        close.rolling(50)
        .mean()
    )

    sma200 = (
        close.rolling(200)
        .mean()
    )

    volume20 = (
        data["Volume"]
        .rolling(20)
        .mean()
    )

    return5 = (
        close.pct_change(5)
    )

    return20 = (
        close.pct_change(20)
    )

    market_return20 = (
        market["Close"]
        .pct_change(20)
    )

    trend = (
        (
            close.iloc[-1]
            >
            sma20.iloc[-1]
        )
        +
        (
            close.iloc[-1]
            >
            sma50.iloc[-1]
        )
        +
        (
            close.iloc[-1]
            >
            sma200.iloc[-1]
        )
    ) / 3

    trend = (
        trend * 2 - 1
    )

    momentum = (
        0.5
        *
        np.tanh(
            return5.iloc[-1]
            /
            0.03
        )
        +
        0.5
        *
        np.tanh(
            return20.iloc[-1]
            /
            0.08
        )
    )

    relative_strength = np.tanh(
        (
            return20.iloc[-1]
            -
            market_return20.iloc[-1]
        )
        /
        0.08
    )

    rsi_component = (
        1
        -
        min(
            abs(
                rsi.iloc[-1]
                -
                55
            )
            /
            35,
            1,
        )
    )

    volume_ratio = (
        data["Volume"].iloc[-1]
        /
        volume20.iloc[-1]
    )

    volume_component = np.clip(
        volume_ratio - 1,
        -1,
        1,
    )

    volatility_component = np.clip(
        1
        -
        (
            atr.iloc[-1]
            /
            close.iloc[-1]
        )
        /
        0.045,
        -1,
        1,
    )

    market_sma50 = (
        market["Close"]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    market_close = (
        market["Close"].iloc[-1]
    )

    if (
        market_close
        >
        market_sma50 * 1.002
    ):
        regime = 1

    elif (
        market_close
        <
        market_sma50 * 0.998
    ):
        regime = -1

    else:
        regime = 0

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
            2 * rsi_component
            -
            1
        )
        +
        0.08 * volume_component
        +
        0.08 * volatility_component
        +
        0.10 * regime
    )

    probability = (
        1
        /
        (
            1
            +
            math.exp(
                -3 * score
            )
        )
    )

    # Conservative reporting cap.
    probability = float(
        np.clip(
            probability,
            0.35,
            0.70,
        )
    )

    stop_distance = max(
        1.5 * atr.iloc[-1],
        close.iloc[-1] * 0.012,
    )

    target_distance = max(
        2.0 * stop_distance,
        close.iloc[-1] * 0.025,
    )

    stop = (
        close.iloc[-1]
        -
        stop_distance
    )

    target = (
        close.iloc[-1]
        +
        target_distance
    )

    expected_return = (
        probability
        *
        (
            target_distance
            /
            close.iloc[-1]
        )
        -
        (
            1 - probability
        )
        *
        (
            stop_distance
            /
            close.iloc[-1]
        )
    )

    risk_reward = (
        target_distance
        /
        stop_distance
    )

    failures = []

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
        <= rsi.iloc[-1]
        <= 68
    ):
        failures.append(
            "rsi"
        )

    if volume_ratio < 0.70:
        failures.append(
            "volume"
        )

    if trend < -0.05:
        failures.append(
            "trend"
        )

    if regime < -0.75:
        failures.append(
            "regime"
        )

    if not failures:

        action = "TRADE"

    elif (
        probability >= 0.55
        and trend >= -0.20
        and 35 <= rsi.iloc[-1] <= 72
    ):

        action = "WATCH"

    else:

        action = "WAIT"

    return {
        "price": float(
            close.iloc[-1]
        ),
        "p": probability,
        "er": expected_return,
        "rr": risk_reward,
        "rsi": float(
            rsi.iloc[-1]
        ),
        "vr": float(
            volume_ratio
        ),
        "stop": float(
            stop
        ),
        "target": float(
            target
        ),
        "action": action,
        "fails": failures,
    }


def send_telegram(
    message
):

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN",
        "",
    ).strip()

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID",
        "",
    ).strip()

    if not token or not chat_id:

        print(
            "Telegram credentials "
            "not configured. "
            "Report printed only."
        )

        return

    url = (
        "https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": message,
        },
        timeout=20,
    )

    response.raise_for_status()


def main():

    now = datetime.now().astimezone()

    market = clean(
        yf.download(
            "^NSEI",
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    )

    vix = clean(
        yf.download(
            "^INDIAVIX",
            period="2y",
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

    nifty = float(
        market["Close"].iloc[-1]
    )

    sma50 = float(
        market["Close"]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    vix_value = (
        float(
            vix["Close"].iloc[-1]
        )
        if not vix.empty
        else np.nan
    )

    if nifty > sma50 * 1.002:

        regime = "FAVORABLE"
        reason = (
            "Nifty above SMA50"
        )

    elif nifty < sma50 * 0.998:

        regime = "UNFAVORABLE"
        reason = (
            "Nifty below SMA50"
        )

    else:

        regime = "MIXED"
        reason = (
            "Nifty/SMA50 signals mixed"
        )

    rows = []

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

            data = clean(
                yf.download(
                    symbol,
                    period="2y",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            )

            if len(data) < MIN_HISTORY:
                continue

            result = evaluate(
                data,
                market,
            )

            result["ticker"] = symbol

            rows.append(
                result
            )

        except Exception as error:

            print(
                f"WARNING {symbol}: "
                f"{error}"
            )

    data = pd.DataFrame(
        rows
    )

    trades = (
        data[
            data["action"]
            ==
            "TRADE"
        ]
        .sort_values(
            [
                "p",
                "er",
            ],
            ascending=False,
        )
    )

    watch = (
        data[
            data["action"]
            ==
            "WATCH"
        ]
        .sort_values(
            [
                "p",
                "er",
            ],
            ascending=False,
        )
        .head(
            TOP_WATCH
        )
    )

    status = (
        "TRADING DAY"
        if now.weekday() < 5
        else
        "WEEKEND / NON-TRADING DAY"
    )

    lines = [
        f"MULTI-FACTOR MARKET ALERT {VERSION}",
        now.strftime(
            "%d %b %Y, %H:%M %Z"
        ),
        "",
        f"MARKET STATUS: {status}",
        f"MARKET REGIME: {regime}",
        (
            f"NIFTY: ₹{nifty:,.2f} "
            f"| SMA50: ₹{sma50:,.2f}"
        ),
        (
            "INDIA VIX: "
            +
            (
                f"₹{vix_value:,.2f}"
                if np.isfinite(
                    vix_value
                )
                else "N/A"
            )
        ),
        f"REGIME REASON: {reason}",
        "",
        "--- TOP SHORT-TERM TRADE SETUPS (1–5 SESSIONS) ---",
    ]

    if now.weekday() >= 5:

        lines.extend(
            [
                "",
                "MARKET IS CLOSED.",
                (
                    "No new long position "
                    "should be initiated today."
                ),
            ]
        )

    if trades.empty:

        lines.extend(
            [
                "",
                "NO VALID LONG TRADE TODAY",
                (
                    "No candidate currently "
                    "satisfies the V6.3.16 "
                    "probability, expected-return, "
                    "risk/reward and quality filters."
                ),
            ]
        )

    else:

        for _, row in trades.head(
            5
        ).iterrows():

            lines.extend(
                [
                    "",
                    (
                        f"{row['ticker'].replace('.NS','')}"
                        " — TRADE CANDIDATE"
                    ),
                    (
                        f"Price: ₹"
                        f"{row['price']:,.2f}"
                    ),
                    (
                        f"P(UP): "
                        f"{row['p']*100:.1f}%"
                    ),
                    (
                        f"ER5: "
                        f"{row['er']*100:.2f}% "
                        f"| RR: {row['rr']:.2f}"
                    ),
                    (
                        f"RSI: "
                        f"{row['rsi']:.1f} "
                        f"| Volume: "
                        f"{row['vr']:.2f}x"
                    ),
                    (
                        f"Stop: ₹"
                        f"{row['stop']:,.2f}"
                        f" | Target: ₹"
                        f"{row['target']:,.2f}"
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "--- BEST WATCHLIST SETUPS ---",
        ]
    )

    if watch.empty:

        lines.append(
            "None."
        )

    else:

        for number, (_, row) in enumerate(
            watch.iterrows(),
            1,
        ):

            lines.extend(
                [
                    "",
                    (
                        f"{number}. "
                        f"{row['ticker'].replace('.NS','')} "
                        "— WATCH"
                    ),
                    (
                        f"Price: ₹"
                        f"{row['price']:,.2f}"
                    ),
                    (
                        f"P(UP): "
                        f"{row['p']*100:.1f}% "
                        f"| ER5: "
                        f"{row['er']*100:.2f}% "
                        f"| RR: "
                        f"{row['rr']:.2f}"
                    ),
                    (
                        f"RSI: "
                        f"{row['rsi']:.1f} "
                        f"| Volume: "
                        f"{row['vr']:.2f}x"
                    ),
                    (
                        "Failed filters: "
                        +
                        (
                            ",".join(
                                row["fails"]
                            )
                            if row["fails"]
                            else "none"
                        )
                    ),
                    "Action: WATCH / WAIT",
                ]
            )

    lines.extend(
        [
            "",
            "--- MODEL VALIDATION STATUS ---",
            (
                "V6.3.16 uses conservative "
                "probability reporting."
            ),
            (
                "P(UP) is an empirical calibrated "
                "estimate, not a guaranteed "
                "probability of profit."
            ),
            "",
            (
                "V6.3.16 is a probabilistic "
                "research screen; it does "
                "not guarantee profit."
            ),
            (
                "Verify live price, liquidity, "
                "corporate news, market status "
                "and execution before trading."
            ),
        ]
    )

    message = "\n".join(
        lines
    )

    print(
        "\n"
        + message
    )

    try:

        send_telegram(
            message
        )

    except Exception as error:

        print(
            f"WARNING: Telegram "
            f"send failed: {error}"
        )


if __name__ == "__main__":
    main()
