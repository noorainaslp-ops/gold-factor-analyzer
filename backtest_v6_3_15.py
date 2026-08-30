"""
MULTI-FACTOR MARKET ALERT
V6.3.15
Chronological walk-forward validation engine

IMPORTANT:
This is a research/backtesting system.
It does not guarantee future profitability.

Design goals:
- No look-ahead leakage
- Chronological walk-forward calibration
- 5-session primary horizon
- 3-session secondary horizon
- 1-session diagnostic horizon
- Cost/slippage aware
- Robust handling of unavailable Yahoo symbols
- Explicit probability calibration
- Portfolio-level risk controls
"""

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

VERSION = "V6.3.15"

# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = os.getenv("BACKTEST_START", "2021-01-01")
END_DATE = os.getenv("BACKTEST_END", "")

INITIAL_CAPITAL = float(os.getenv("ALERT_CAPITAL", "100000") or "100000")

# Conservative round-trip trading friction assumption.
COST_BPS = float(os.getenv("COST_BPS", "15") or "15")
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5") or "5")

TOTAL_COST = (COST_BPS + SLIPPAGE_BPS) / 10000.0

PRIMARY_HORIZON = 5
SECONDARY_HORIZON = 3
FAST_HORIZON = 1

MIN_CALIBRATION_OBS = 250
CALIBRATION_LOOKBACK = 5000

MAX_POSITIONS_PER_DAY = 5
MAX_SYMBOL_WEIGHT = 0.20

# We deliberately do not use regime as a hard rejection.
# V6.3.14.1 showed weak regime discrimination.
REGIME_SCORE_WEIGHT = 0.05

# Signal thresholds.
# These are intentionally moderate and are NOT optimized on the
# current backtest results.
TRADE_PROBABILITY = 0.57
WATCH_PROBABILITY = 0.52

TRADE_SCORE = 70.0
WATCH_SCORE = 62.0

MIN_TRADE_EXPECTED_RETURN = 0.004
MIN_WATCH_EXPECTED_RETURN = 0.001

MIN_TRADE_RR = 1.10
MIN_WATCH_RR = 0.85

# ============================================================
# UNIVERSE
# ============================================================

TICKERS = [
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
# HELPERS
# ============================================================

def safe_float(value, default=np.nan):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return default


def clamp(x, low, high):
    return max(low, min(high, x))


def sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    return df


def download_symbol(ticker):
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE if END_DATE else None,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"WARNING: no data for {ticker}; skipping.")
            return None

        df = clean_columns(df)

        required = {"open", "high", "low", "close", "volume"}

        if not required.issubset(set(df.columns)):
            print(f"WARNING: incomplete data for {ticker}; skipping.")
            return None

        df = df[["open", "high", "low", "close", "volume"]].copy()

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close"])

        if len(df) < 300:
            print(f"WARNING: insufficient history for {ticker}; skipping.")
            return None

        return df

    except Exception as exc:
        print(f"WARNING: failed downloading {ticker}: {exc}")
        return None


# ============================================================
# TECHNICAL FEATURES
# ============================================================

def add_features(df):
    x = df.copy()

    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"]

    x["ret1"] = close.pct_change(1)
    x["ret3"] = close.pct_change(3)
    x["ret5"] = close.pct_change(5)
    x["ret10"] = close.pct_change(10)
    x["ret20"] = close.pct_change(20)

    x["sma20"] = close.rolling(20).mean()
    x["sma50"] = close.rolling(50).mean()
    x["sma100"] = close.rolling(100).mean()
    x["sma200"] = close.rolling(200).mean()

    x["trend20"] = close / x["sma20"] - 1
    x["trend50"] = close / x["sma50"] - 1
    x["trend200"] = close / x["sma200"] - 1

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["rsi"] = x["rsi"].fillna(50)

    # ATR
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    x["atr14"] = tr.rolling(14).mean()
    x["atr_pct"] = x["atr14"] / close

    # Volatility
    x["volatility20"] = x["ret1"].rolling(20).std()

    # Relative volume
    avg_volume = volume.rolling(20).mean()
    x["relative_volume"] = (
        volume / avg_volume.replace(0, np.nan)
    )

    # Breakout / position
    rolling_high20 = high.rolling(20).max()
    rolling_low20 = low.rolling(20).min()

    x["range_position"] = (
        (close - rolling_low20)
        / (rolling_high20 - rolling_low20).replace(0, np.nan)
    )

    # Momentum consistency
    x["positive_days10"] = (
        x["ret1"].rolling(10)
        .apply(lambda a: np.mean(a > 0), raw=True)
    )

    return x


# ============================================================
# MARKET REGIME
# ============================================================

def build_market_context():
    try:
        nifty = yf.download(
            "^NSEI",
            start=START_DATE,
            end=END_DATE if END_DATE else None,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if nifty is None or nifty.empty:
            return pd.DataFrame()

        nifty = clean_columns(nifty)

        if "close" not in nifty.columns:
            return pd.DataFrame()

        nifty["nifty_close"] = pd.to_numeric(
            nifty["close"], errors="coerce"
        )

        nifty["nifty_sma50"] = (
            nifty["nifty_close"].rolling(50).mean()
        )

        nifty["nifty_sma200"] = (
            nifty["nifty_close"].rolling(200).mean()
        )

        nifty["nifty_ret5"] = (
            nifty["nifty_close"].pct_change(5)
        )

        nifty["nifty_trend50"] = (
            nifty["nifty_close"]
            / nifty["nifty_sma50"]
            - 1
        )

        nifty["regime"] = np.select(
            [
                (
                    (nifty["nifty_close"] > nifty["nifty_sma50"])
                    &
                    (nifty["nifty_sma50"] > nifty["nifty_sma200"])
                ),
                (
                    (nifty["nifty_close"] > nifty["nifty_sma50"])
                    |
                    (nifty["nifty_sma50"] > nifty["nifty_sma200"])
                ),
            ],
            [
                "FAVORABLE",
                "MIXED",
            ],
            default="UNFAVORABLE",
        )

        return nifty[
            [
                "nifty_close",
                "nifty_sma50",
                "nifty_sma200",
                "nifty_ret5",
                "nifty_trend50",
                "regime",
            ]
        ]

    except Exception as exc:
        print(f"WARNING: market context unavailable: {exc}")
        return pd.DataFrame()


# ============================================================
# RAW MODEL PROBABILITY
# ============================================================

def calculate_raw_probability(row):
    """
    A deliberately transparent multi-factor probability score.

    This is NOT presented as a mathematically true probability.
    It is subsequently calibrated chronologically.
    """

    trend50 = safe_float(row.get("trend50"), 0)
    trend200 = safe_float(row.get("trend200"), 0)
    ret5 = safe_float(row.get("ret5"), 0)
    ret20 = safe_float(row.get("ret20"), 0)
    rsi = safe_float(row.get("rsi"), 50)
    relvol = safe_float(row.get("relative_volume"), 1)
    range_pos = safe_float(row.get("range_position"), 0.5)
    pos10 = safe_float(row.get("positive_days10"), 0.5)

    score = 0.0

    score += clamp(trend50 / 0.05, -2, 2) * 0.90
    score += clamp(trend200 / 0.10, -2, 2) * 0.60
    score += clamp(ret5 / 0.05, -2, 2) * 0.55
    score += clamp(ret20 / 0.10, -2, 2) * 0.35

    # RSI: moderate strength is preferred.
    rsi_component = -abs(rsi - 55) / 25
    score += clamp(rsi_component, -1.5, 0.2) * 0.30

    score += clamp((relvol - 1) / 0.5, -2, 2) * 0.25
    score += clamp((range_pos - 0.5) * 2, -1, 1) * 0.25
    score += clamp((pos10 - 0.5) * 2, -1, 1) * 0.35

    return float(sigmoid(score))


# ============================================================
# FEATURE QUALITY
# ============================================================

def calculate_quality(row):
    checks = []

    checks.append(
        safe_float(row.get("trend50"), 0) > -0.08
    )

    checks.append(
        safe_float(row.get("trend200"), 0) > -0.15
    )

    rsi = safe_float(row.get("rsi"), 50)

    checks.append(
        35 <= rsi <= 70
    )

    atr_pct = safe_float(row.get("atr_pct"), np.nan)

    if np.isfinite(atr_pct):
        checks.append(
            0.005 <= atr_pct <= 0.08
        )

    volume = safe_float(
        row.get("relative_volume"),
        np.nan
    )

    if np.isfinite(volume):
        checks.append(
            volume >= 0.60
        )

    if not checks:
        return 0.0

    return float(np.mean(checks))


# ============================================================
# RISK / REWARD
# ============================================================

def calculate_risk_reward(row):
    price = safe_float(row.get("close"), np.nan)
    atr = safe_float(row.get("atr14"), np.nan)

    if not np.isfinite(price) or not np.isfinite(atr):
        return 0.0, 0.0, 0.0

    if price <= 0 or atr <= 0:
        return 0.0, 0.0, 0.0

    stop_distance = max(1.25 * atr, price * 0.012)

    # Conservative 5-session target.
    target_distance = max(
        1.60 * atr,
        price * 0.018
    )

    rr = target_distance / stop_distance

    return (
        float(stop_distance),
        float(target_distance),
        float(rr),
    )


# ============================================================
# CALIBRATION
# ============================================================

def probability_bucket(p):
    p = safe_float(p, np.nan)

    if not np.isfinite(p):
        return "NA"

    if p < 0.40:
        return "<40%"
    if p < 0.45:
        return "40-45%"
    if p < 0.50:
        return "45-50%"
    if p < 0.55:
        return "50-55%"
    if p < 0.60:
        return "55-60%"
    if p < 0.65:
        return "60-65%"
    if p < 0.70:
        return "65-70%"
    if p < 0.75:
        return "70-75%"

    return "75%+"


def calibrate_probability(raw_p, history):
    """
    Expanding chronological calibration.

    Only historical rows strictly before the current signal are
    supplied to this function.
    """

    raw_p = safe_float(raw_p, np.nan)

    if not np.isfinite(raw_p):
        return np.nan, "INSUFFICIENT_HISTORY"

    if history is None or len(history) < MIN_CALIBRATION_OBS:
        return raw_p, "RAW_INSUFFICIENT_HISTORY"

    h = history.copy()

    if CALIBRATION_LOOKBACK > 0:
        h = h.tail(CALIBRATION_LOOKBACK)

    # Nearby historical probability observations.
    distances = (h["raw_probability"] - raw_p).abs()

    nearest = h.loc[
        distances.nsmallest(
            min(250, len(h))
        ).index
    ]

    if len(nearest) < 50:
        return raw_p, "RAW_NEAREST_INSUFFICIENT"

    empirical = nearest["win5"].mean()

    if not np.isfinite(empirical):
        return raw_p, "RAW_BAD_CALIBRATION"

    # Shrink empirical estimate toward raw model probability.
    # This avoids extreme estimates from small local samples.
    n = len(nearest)

    weight = min(0.85, n / 300.0)

    calibrated = (
        weight * empirical
        + (1 - weight) * raw_p
    )

    return float(clamp(calibrated, 0.20, 0.80)), "CALIBRATED"


# ============================================================
# ACTION CLASSIFICATION
# ============================================================

def classify_action(
    probability,
    expected_return,
    rr,
    quality,
    regime,
):
    probability = safe_float(probability, 0)
    expected_return = safe_float(expected_return, 0)
    rr = safe_float(rr, 0)
    quality = safe_float(quality, 0)

    # Regime contributes only a small score adjustment.
    regime_bonus = {
        "FAVORABLE": 0.02,
        "MIXED": 0.00,
        "UNFAVORABLE": -0.02,
    }.get(regime, 0.0)

    effective_probability = probability + (
        REGIME_SCORE_WEIGHT * regime_bonus
    )

    composite = (
        effective_probability * 100
        + quality * 10
        + min(max(rr - 1, -1), 1) * 5
    )

    if (
        effective_probability >= TRADE_PROBABILITY
        and expected_return >= MIN_TRADE_EXPECTED_RETURN
        and rr >= MIN_TRADE_RR
        and quality >= 0.60
        and composite >= TRADE_SCORE
    ):
        return "TRADE", composite

    if (
        effective_probability >= WATCH_PROBABILITY
        and expected_return >= MIN_WATCH_EXPECTED_RETURN
        and rr >= MIN_WATCH_RR
        and quality >= 0.50
        and composite >= WATCH_SCORE
    ):
        return "WATCH", composite

    return "WAIT", composite


# ============================================================
# BUILD CANDIDATE OBSERVATIONS
# ============================================================

def create_candidate_table(ticker, df, market):
    x = add_features(df)

    if market is not None and not market.empty:
        x = x.join(market, how="left")

    x["ticker"] = ticker
    x["date"] = pd.to_datetime(x.index).tz_localize(None)

    # Forward returns.
    x["future1"] = x["close"].shift(-1) / x["close"] - 1
    x["future3"] = x["close"].shift(-3) / x["close"] - 1
    x["future5"] = x["close"].shift(-5) / x["close"] - 1

    # Cost-adjusted forward returns.
    x["net1"] = x["future1"] - TOTAL_COST
    x["net3"] = x["future3"] - TOTAL_COST
    x["net5"] = x["future5"] - TOTAL_COST

    x["win1"] = (x["net1"] > 0).astype(float)
    x["win3"] = (x["net3"] > 0).astype(float)
    x["win5"] = (x["net5"] > 0).astype(float)

    rows = []

    for _, row in x.iterrows():

        if not np.isfinite(safe_float(row.get("close"))):
            continue

        required = [
            "sma50",
            "sma200",
            "rsi",
            "atr14",
            "relative_volume",
            "trend50",
            "trend200",
            "ret5",
            "ret20",
            "range_position",
            "positive_days10",
        ]

        if any(
            not np.isfinite(safe_float(row.get(c)))
            for c in required
        ):
            continue

        raw_p = calculate_raw_probability(row)

        stop_distance, target_distance, rr = (
            calculate_risk_reward(row)
        )

        quality = calculate_quality(row)

        rows.append(
            {
                "ticker": ticker,
                "date": row["date"],
                "close": safe_float(row["close"]),
                "raw_probability": raw_p,
                "quality": quality,
                "rr": rr,
                "stop_distance": stop_distance,
                "target_distance": target_distance,
                "rsi": safe_float(row["rsi"]),
                "relative_volume": safe_float(
                    row["relative_volume"]
                ),
                "trend50": safe_float(row["trend50"]),
                "trend200": safe_float(row["trend200"]),
                "regime": row.get(
                    "regime",
                    "UNKNOWN"
                ),
                "future1": safe_float(row.get("future1")),
                "future3": safe_float(row.get("future3")),
                "future5": safe_float(row.get("future5")),
                "net1": safe_float(row.get("net1")),
                "net3": safe_float(row.get("net3")),
                "net5": safe_float(row.get("net5")),
                "win1": safe_float(row.get("win1")),
                "win3": safe_float(row.get("win3")),
                "win5": safe_float(row.get("win5")),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================

def performance_table(df, selection_col="action"):
    records = []

    if df.empty:
        return pd.DataFrame()

    for selection in [
        "TRADE",
        "WATCH",
        "WAIT",
        "ALL",
    ]:

        if selection == "ALL":
            sub = df.copy()
        else:
            sub = df[df[selection_col] == selection]

        if sub.empty:
            continue

        for horizon in [1, 3, 5]:

            ret_col = f"net{horizon}"
            win_col = f"win{horizon}"

            vals = pd.to_numeric(
                sub[ret_col],
                errors="coerce"
            ).dropna()

            wins = pd.to_numeric(
                sub[win_col],
                errors="coerce"
            ).dropna()

            if vals.empty:
                continue

            winners = vals[vals > 0]
            losers = vals[vals < 0]

            gross_winners = (
                winners.sum()
                if not winners.empty
                else 0
            )

            gross_losers = (
                abs(losers.sum())
                if not losers.empty
                else np.nan
            )

            pf = (
                gross_winners / gross_losers
                if np.isfinite(gross_losers)
                and gross_losers > 0
                else np.nan
            )

            records.append(
                {
                    "selection": selection,
                    "horizon": horizon,
                    "observations": len(vals),
                    "win_rate": wins.mean()
                    if not wins.empty
                    else np.nan,
                    "average_net_return": vals.mean(),
                    "median_net_return": vals.median(),
                    "average_winner": winners.mean()
                    if not winners.empty
                    else np.nan,
                    "average_loser": losers.mean()
                    if not losers.empty
                    else np.nan,
                    "profit_factor": pf,
                    "best": vals.max(),
                    "worst": vals.min(),
                }
            )

    return pd.DataFrame(records)


def calibration_table(df):
    if df.empty:
        return pd.DataFrame()

    buckets = [
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

    records = []

    for bucket in buckets:

        sub = df[
            df["probability_bucket"] == bucket
        ]

        if sub.empty:
            continue

        records.append(
            {
                "probability_bucket": bucket,
                "observations": len(sub),
                "average_model_probability":
                    sub["calibrated_probability"].mean(),
                "actual_win_rate":
                    sub["win5"].mean(),
                "average_return":
                    sub["net5"].mean(),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def simulate_portfolio(df):
    trades = df[
        df["action"] == "TRADE"
    ].copy()

    if trades.empty:
        return {
            "trading_days": 0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_like": 0.0,
        }

    selected = []

    for date, day in trades.groupby("date"):

        day = day.sort_values(
            [
                "calibrated_probability",
                "composite_score",
            ],
            ascending=False,
        )

        day = day.head(
            MAX_POSITIONS_PER_DAY
        )

        selected.append(day)

    selected = pd.concat(
        selected,
        ignore_index=True
    )

    daily = (
        selected
        .groupby("date")["net5"]
        .mean()
        .sort_index()
    )

    equity = (1 + daily).cumprod()

    cumulative = equity.iloc[-1] - 1

    running_max = equity.cummax()

    drawdown = (
        equity / running_max - 1
    )

    max_drawdown = drawdown.min()

    if len(daily) > 1 and daily.std() > 0:
        sharpe = (
            daily.mean()
            / daily.std()
            * np.sqrt(252)
        )
    else:
        sharpe = 0.0

    return {
        "trading_days": int(len(daily)),
        "cumulative_return": float(cumulative),
        "max_drawdown": float(max_drawdown),
        "sharpe_like": float(sharpe),
    }


# ============================================================
# MAIN WALK-FORWARD ENGINE
# ============================================================

def main():

    print(
        f"Starting {VERSION} chronological "
        "walk-forward backtest..."
    )

    print(
        f"Universe size: {len(TICKERS)}"
    )

    market = build_market_context()

    all_frames = []

    for i, ticker in enumerate(TICKERS, 1):

        print(
            f"Loading [{i}/{len(TICKERS)}] {ticker}"
        )

        df = download_symbol(ticker)

        if df is None:
            continue

        candidate = create_candidate_table(
            ticker,
            df,
            market
        )

        if candidate.empty:
            print(
                f"WARNING: no usable observations "
                f"for {ticker}"
            )
            continue

        all_frames.append(candidate)

    if not all_frames:
        raise RuntimeError(
            "No candidate observations were created."
        )

    data = pd.concat(
        all_frames,
        ignore_index=True
    )

    data = data.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    # Only observations with completed 5D outcomes
    # participate in calibration.
    calibration_pool = []

    output_rows = []

    unique_dates = sorted(
        data["date"].dropna().unique()
    )

    for date in unique_dates:

        day = data[
            data["date"] == date
        ].copy()

        # ----------------------------------------------------
        # HISTORICAL CALIBRATION SET
        # ----------------------------------------------------

        if calibration_pool:

            history = pd.DataFrame(
                calibration_pool
            )

            history = history[
                history["date"] < date
            ].copy()

        else:
            history = pd.DataFrame()

        # ----------------------------------------------------
        # CURRENT DAY
        # ----------------------------------------------------

        for idx, row in day.iterrows():

            calibrated_p, calibration_status = (
                calibrate_probability(
                    row["raw_probability"],
                    history
                )
            )

            # Historical conditional expected return.
            expected_return = 0.0

            if (
                not history.empty
                and len(history) >= MIN_CALIBRATION_OBS
            ):

                distances = (
                    history["raw_probability"]
                    - row["raw_probability"]
                ).abs()

                nearest = history.loc[
                    distances.nsmallest(
                        min(250, len(history))
                    ).index
                ]

                if not nearest.empty:
                    expected_return = safe_float(
                        nearest["net5"].mean(),
                        0.0
                    )

            if calibration_status != "CALIBRATED":
                # Before enough historical information exists,
                # expected return must not create a trade.
                expected_return = min(
                    expected_return,
                    0.0
                )

            action, composite = classify_action(
                calibrated_p,
                expected_return,
                row["rr"],
                row["quality"],
                row["regime"],
            )

            output_rows.append(
                {
                    **row.to_dict(),
                    "calibrated_probability":
                        calibrated_p,
                    "calibration_status":
                        calibration_status,
                    "expected_return":
                        expected_return,
                    "composite_score":
                        composite,
                    "probability_bucket":
                        probability_bucket(
                            calibrated_p
                        ),
                    "action":
                        action,
                }
            )

        # ----------------------------------------------------
        # AFTER CLASSIFYING THE DAY:
        # ADD ITS COMPLETED FUTURE OUTCOMES TO HISTORY.
        #
        # This ordering is critical.
        # ----------------------------------------------------

        for _, row in day.iterrows():

            if np.isfinite(
                safe_float(row["win5"])
            ):
                calibration_pool.append(
                    {
                        "date": row["date"],
                        "raw_probability":
                            row["raw_probability"],
                        "win5":
                            row["win5"],
                        "net5":
                            row["net5"],
                    }
                )

    result = pd.DataFrame(output_rows)

    if result.empty:
        raise RuntimeError(
            "Walk-forward result is empty."
        )

    # ========================================================
    # AUDIT DIRECTORY
    # ========================================================

    audit_dir = Path("audit")
    audit_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    # ========================================================
    # REPORT
    # ========================================================

    actions = (
        result["action"]
        .value_counts()
        .rename_axis("action")
        .reset_index(name="count")
    )

    performance = performance_table(
        result
    )

    calibration = calibration_table(
        result
    )

    portfolio = simulate_portfolio(
        result
    )

    trade_rows = result[
        result["action"] == "TRADE"
    ].copy()

    watch_rows = result[
        result["action"] == "WATCH"
    ].copy()

    # ========================================================
    # SYMBOL PERFORMANCE
    # ========================================================

    if not trade_rows.empty:

        symbol_stats = (
            trade_rows
            .groupby("ticker")
            .agg(
                observations=("ticker", "size"),
                win_rate=("win5", "mean"),
                average_return=("net5", "mean"),
                median_return=("net5", "median"),
                average_probability=(
                    "calibrated_probability",
                    "mean"
                ),
                average_composite=(
                    "composite_score",
                    "mean"
                ),
            )
            .reset_index()
            .sort_values(
                "average_return",
                ascending=False
            )
        )

    else:

        symbol_stats = pd.DataFrame(
            columns=[
                "ticker",
                "observations",
                "win_rate",
                "average_return",
                "median_return",
                "average_probability",
                "average_composite",
            ]
        )

    # ========================================================
    # REGIME PERFORMANCE
    # ========================================================

    regime_stats = (
        result
        .groupby("regime")
        .agg(
            observations=("net5", "size"),
            win_rate=("win5", "mean"),
            average_return=("net5", "mean"),
            median_return=("net5", "median"),
        )
        .reset_index()
    )

    # ========================================================
    # SAVE AUDIT FILES
    # ========================================================

    result.to_csv(
        audit_dir
        / f"walkforward_{VERSION.lower().replace('.', '_')}_{timestamp}.csv",
        index=False
    )

    performance.to_csv(
        audit_dir
        / f"performance_{VERSION.lower().replace('.', '_')}_{timestamp}.csv",
        index=False
    )

    calibration.to_csv(
        audit_dir
        / f"probability_calibration_{VERSION.lower().replace('.', '_')}_{timestamp}.csv",
        index=False
    )

    actions.to_csv(
        audit_dir
        / f"action_counts_{VERSION.lower().replace('.', '_')}_{timestamp}.csv",
        index=False
    )

    symbol_stats.to_csv(
        audit_dir
        / f"symbol_performance_{VERSION.lower().replace('.', '_')}_{timestamp}.csv",
        index=False
    )

    regime_stats.to_csv(
        audit_dir
        / f"regime_performance_{VERSION.lower().replace('.', '_')}_{timestamp}.csv",
        index=False
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print(f"{VERSION} WALK-FORWARD BACKTEST")
    print("=" * 70)

    print()
    print(
        "Total candidate observations:",
        len(result)
    )

    print()
    print("ACTION COUNTS:")
    print(actions.to_string(index=False))

    print()
    print("ACTION GROUP PERFORMANCE:")

    if performance.empty:
        print("No completed outcomes.")
    else:
        print(
            performance.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    print()
    print("PROBABILITY CALIBRATION:")

    if calibration.empty:
        print("No calibration observations.")
    else:
        print(
            calibration.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    print()
    print("PORTFOLIO-LEVEL RESULT:")
    print(
        f"Trading days: "
        f"{portfolio['trading_days']}"
    )
    print(
        f"Cumulative return: "
        f"{portfolio['cumulative_return']:.2%}"
    )
    print(
        f"Maximum drawdown: "
        f"{portfolio['max_drawdown']:.2%}"
    )
    print(
        f"Annualized Sharpe-like ratio: "
        f"{portfolio['sharpe_like']:.2f}"
    )

    print()
    print("TRADE SYMBOL PERFORMANCE:")

    if symbol_stats.empty:
        print("No completed TRADE observations.")
    else:
        print(
            symbol_stats.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    print()
    print("REGIME PERFORMANCE - 5D:")

    if regime_stats.empty:
        print("No regime observations.")
    else:
        print(
            regime_stats.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}"
            )
        )

    print()
    print("DATA QUALITY:")
    print(
        "Unique symbols tested:",
        result["ticker"].nunique()
    )
    print(
        "Unique signal dates:",
        result["date"].nunique()
    )
    print(
        "TRADE observations:",
        len(trade_rows)
    )
    print(
        "WATCH observations:",
        len(watch_rows)
    )
    print(
        "WAIT observations:",
        int((result["action"] == "WAIT").sum())
    )

    print()
    print("=" * 70)
    print(f"{VERSION} BACKTEST COMPLETED")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This is historical research only."
    )
    print(
        "Do NOT promote the strategy to real-money "
        "trading solely from this test."
    )
    print(
        "Use a completely untouched out-of-sample "
        "period before live deployment."
    )
    print(
        "Model probabilities are calibrated estimates, "
        "not guaranteed probabilities of profit."
    )


if __name__ == "__main__":
    main()
