"""
V6.3.8 WALK-FORWARD BACKTEST
============================

Purpose
-------
Historical validation of the V6.3.8 short-term trading framework.

Important design principles
---------------------------
1. No future information is used when creating a signal.
2. Features are calculated using data available on the signal date.
3. Forward returns are calculated only after the signal date.
4. 1, 3 and 5-session outcomes are evaluated separately.
5. The script does NOT send Telegram alerts.
6. The script does NOT modify the live market engine.
7. Results are saved under audit/.

This is research/validation code only.
It does not guarantee future profitability.
"""

from __future__ import annotations

import os
import math
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

VERSION = "V6.3.8"

# Backtest period.
# 2 years gives substantially more observations than a very short test.
START_DATE = os.getenv("BACKTEST_START", "2024-01-01")
END_DATE = os.getenv("BACKTEST_END", "")

# Capital is NOT used to fabricate profitability.
# It is only used for illustrative position sizing.
try:
    CAPITAL = float(os.getenv("ALERT_CAPITAL") or "100000")
except Exception:
    CAPITAL = 100000.0

try:
    MAX_RISK_PER_TRADE = float(
        os.getenv("MAX_RISK_PER_TRADE") or "0.01"
    )
except Exception:
    MAX_RISK_PER_TRADE = 0.01

# Minimum quality requirements.
#
# These are deliberately not made so strict that the backtest
# produces zero observations.
MIN_P3 = 0.55
MIN_P5 = 0.55

MIN_ER3 = 0.002
MIN_ER5 = 0.003

MIN_RR1 = 1.00
MIN_RR2 = 1.00

# Maximum RSI for a fresh long entry.
MAX_RSI = 70.0

# Minimum relative volume.
MIN_VOLUME = 0.80

# Trend requirement.
REQUIRE_TREND = True

# Market-regime requirement.
# In the research backtest we evaluate both:
#   1. all candidates
#   2. candidates passing regime
#
# This is useful because an excessively restrictive regime filter
# can hide whether the stock model itself has predictive value.
REQUIRE_REGIME_FOR_TRADE = True

# Candidate universe.
# Keep this reasonably broad so the backtest does not suffer from
# survivorship concentrated in only a handful of stocks.
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


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):
    """Convert a value safely to float."""
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

        if value == "":
            return default

        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except Exception:
        return default


def clean_symbol(symbol: str) -> str:
    """Convert Yahoo symbol into a display symbol."""
    return symbol.replace(".NS", "")


def annualised_volatility(close: pd.Series) -> float:
    """20-session annualised historical volatility."""
    returns = close.pct_change().dropna()

    if len(returns) < 10:
        return np.nan

    value = returns.tail(20).std() * math.sqrt(252)

    return safe_float(value)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI."""
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    # If average loss is zero, RSI is effectively 100.
    result = result.where(
        ~((avg_loss == 0) & (avg_gain > 0)),
        100
    )

    return result


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score without look-ahead."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()

    return (series - mean) / std.replace(0, np.nan)


def download_symbol(symbol: str) -> pd.DataFrame | None:
    """Download one instrument."""
    try:
        df = yf.download(
            symbol,
            start=START_DATE,
            end=END_DATE if END_DATE else None,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return None

        # yfinance can sometimes return MultiIndex columns.
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                return None

        required = ["Open", "High", "Low", "Close", "Volume"]

        if not all(col in df.columns for col in required):
            return None

        df = df[required].copy()

        for col in required:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df.dropna(subset=["Close"], inplace=True)

        if len(df) < 150:
            return None

        df.index = pd.to_datetime(df.index)

        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as exc:
        print(
            f"WARNING: download failed for {symbol}: {exc}"
        )
        return None


# ============================================================
# FEATURE ENGINE
# ============================================================

def add_features(stock: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all features using information available up to
    each individual trading date.
    """

    df = stock.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)
    df["ret_20"] = close.pct_change(20)

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    df["ema_10"] = close.ewm(
        span=10,
        adjust=False
    ).mean()

    df["ema_20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_100"] = close.rolling(100).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    df["rsi"] = rsi(close, 14)

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_average = volume.rolling(20).mean()

    df["relative_volume"] = (
        volume / volume_average.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    daily_returns = close.pct_change()

    df["volatility_20"] = (
        daily_returns
        .rolling(20)
        .std()
        * math.sqrt(252)
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr_14"] = true_range.rolling(14).mean()

    df["atr_pct"] = (
        df["atr_14"] /
        close.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Breakout / trend features
    # --------------------------------------------------------

    df["high_20"] = high.shift(1).rolling(20).max()
    df["low_20"] = low.shift(1).rolling(20).min()

    df["breakout_20"] = (
        close > df["high_20"]
    )

    df["trend_strength"] = (
        (close > df["sma_20"]).astype(int)
        +
        (df["sma_20"] > df["sma_50"]).astype(int)
        +
        (df["sma_50"] > df["sma_100"]).astype(int)
    )

    df["trend"] = (
        (close > df["sma_50"])
        &
        (df["sma_50"] > df["sma_100"])
    )

    # --------------------------------------------------------
    # Nifty regime
    # --------------------------------------------------------

    nifty_close = nifty["Close"].reindex(
        df.index
    ).ffill()

    nifty_sma50 = nifty_close.rolling(50).mean()

    nifty_sma50_previous = nifty_sma50.shift(5)

    df["nifty"] = nifty_close
    df["nifty_sma50"] = nifty_sma50

    df["regime_favorable"] = (
        (nifty_close > nifty_sma50)
        &
        (nifty_sma50 > nifty_sma50_previous)
    )

    df["regime_mixed"] = (
        (nifty_close > nifty_sma50)
        ^
        (nifty_sma50 > nifty_sma50_previous)
    )

    df["regime"] = np.select(
        [
            df["regime_favorable"],
            df["regime_mixed"],
        ],
        [
            "FAVORABLE",
            "MIXED",
        ],
        default="UNFAVORABLE"
    )

    # --------------------------------------------------------
    # Composite momentum score
    # --------------------------------------------------------

    df["momentum_score"] = (
        0.30 * df["ret_5"].fillna(0)
        +
        0.25 * df["ret_10"].fillna(0)
        +
        0.20 * df["ret_20"].fillna(0)
        +
        0.15 * (
            close / df["ema_20"] - 1
        ).fillna(0)
        +
        0.10 * (
            close / df["sma_50"] - 1
        ).fillna(0)
    )

    # Rolling percentile-like strength.
    df["momentum_z"] = rolling_zscore(
        df["momentum_score"],
        60
    )

    return df


# ============================================================
# EMPIRICAL PROBABILITY ESTIMATION
# ============================================================

def historical_probability(
    df: pd.DataFrame,
    index_position: int,
    horizon: int,
    lookback: int = 120,
) -> float:
    """
    Estimate probability of positive future return using only
    observations BEFORE the current signal date.

    This avoids the major look-ahead error of calculating a
    probability from the entire dataset.
    """

    if index_position < 80:
        return np.nan

    start = max(20, index_position - lookback)

    historical = df.iloc[start:index_position].copy()

    if historical.empty:
        return np.nan

    # Historical analogue conditions.
    current = df.iloc[index_position]

    current_rsi = safe_float(current["rsi"])
    current_momentum = safe_float(
        current["momentum_score"]
    )
    current_volume = safe_float(
        current["relative_volume"]
    )

    if not np.isfinite(current_rsi):
        return np.nan

    if not np.isfinite(current_momentum):
        return np.nan

    # Similarity bands.
    rsi_low = current_rsi - 8
    rsi_high = current_rsi + 8

    momentum_std = historical["momentum_score"].std()

    if not np.isfinite(momentum_std) or momentum_std == 0:
        momentum_std = abs(current_momentum) * 0.5 + 0.001

    momentum_low = (
        current_momentum - momentum_std
    )

    momentum_high = (
        current_momentum + momentum_std
    )

    mask = (
        historical["rsi"].between(
            rsi_low,
            rsi_high
        )
        &
        historical["momentum_score"].between(
            momentum_low,
            momentum_high
        )
    )

    if np.isfinite(current_volume):
        volume_low = max(0.25, current_volume * 0.65)
        volume_high = current_volume * 1.50

        mask &= historical["relative_volume"].between(
            volume_low,
            volume_high
        )

    comparable = historical.loc[mask].copy()

    # If too few analogues are available, progressively broaden.
    if len(comparable) < 15:
        comparable = historical[
            historical["rsi"].between(
                rsi_low - 5,
                rsi_high + 5
            )
        ].copy()

    if len(comparable) < 10:
        comparable = historical.copy()

    # Calculate forward returns from historical observations.
    future_close = df["Close"].shift(-horizon)

    outcomes = (
        future_close.loc[comparable.index]
        / comparable["Close"]
        - 1
    )

    outcomes = outcomes.dropna()

    if len(outcomes) < 10:
        return np.nan

    probability = (
        (outcomes > 0).mean()
    )

    return float(probability)


# ============================================================
# EXPECTED RETURN
# ============================================================

def expected_return(
    probability: float,
    horizon: int,
    volatility: float,
) -> float:
    """
    Conservative expected-return estimate.

    This is not a promise of return. It is a model quantity used
    for ranking and filtering.
    """

    if not np.isfinite(probability):
        return np.nan

    if not np.isfinite(volatility):
        return np.nan

    daily_vol = volatility / math.sqrt(252)

    # Conservative payoff asymmetry.
    average_win = daily_vol * math.sqrt(horizon) * 0.85
    average_loss = daily_vol * math.sqrt(horizon) * 0.75

    result = (
        probability * average_win
        -
        (1 - probability) * average_loss
    )

    return float(result)


# ============================================================
# RISK / REWARD
# ============================================================

def calculate_risk_reward(
    price: float,
    atr: float,
) -> tuple[float, float, float, float]:
    """
    Create realistic stop/targets from ATR.

    Unlike earlier versions, stops cannot become negative.
    """

    if not np.isfinite(price) or price <= 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.02

    # Stop approximately 1.5 ATR below entry.
    stop = price - 1.50 * atr

    # Never allow nonsensical/negative stops.
    minimum_stop = price * 0.70

    stop = max(stop, minimum_stop)

    risk = price - stop

    target1 = price + 1.25 * risk
    target2 = price + 2.00 * risk

    rr1 = (
        (target1 - price) / risk
        if risk > 0
        else np.nan
    )

    rr2 = (
        (target2 - price) / risk
        if risk > 0
        else np.nan
    )

    return (
        stop,
        target1,
        target2,
        rr1,
        rr2,
    )


# ============================================================
# POSITION SIZE
# ============================================================

def position_size(
    price: float,
    stop: float,
) -> tuple[int, float]:
    """Risk-based position sizing."""

    if (
        not np.isfinite(price)
        or
        not np.isfinite(stop)
        or
        price <= 0
        or
        stop <= 0
        or
        stop >= price
    ):
        return 0, 0.0

    risk_per_share = price - stop

    allowed_loss = (
        CAPITAL * MAX_RISK_PER_TRADE
    )

    shares = math.floor(
        allowed_loss / risk_per_share
    )

    # Never allocate more than available capital.
    max_by_capital = math.floor(
        CAPITAL / price
    )

    shares = min(
        shares,
        max_by_capital
    )

    value = shares * price

    return int(max(0, shares)), float(value)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_candidate(row: dict) -> tuple[str, list[str]]:
    """
    Apply the V6.3.8 research filters.

    Returns:
        action, failed_filters
    """

    failed = []

    p3 = safe_float(row.get("p3"))
    p5 = safe_float(row.get("p5"))

    er3 = safe_float(row.get("er3"))
    er5 = safe_float(row.get("er5"))

    rr1 = safe_float(row.get("rr1"))
    rr2 = safe_float(row.get("rr2"))

    quality = bool(row.get("quality", False))
    trend = bool(row.get("trend", False))
    rsi_ok = bool(row.get("rsi_ok", False))
    regime_ok = bool(row.get("regime_ok", False))
    volume_ok = bool(row.get("volume_ok", False))

    if not np.isfinite(p3) or p3 < MIN_P3:
        failed.append("p3")

    if not np.isfinite(p5) or p5 < MIN_P5:
        failed.append("p5")

    if not np.isfinite(er3) or er3 < MIN_ER3:
        failed.append("er3")

    if not np.isfinite(er5) or er5 < MIN_ER5:
        failed.append("er5")

    if not np.isfinite(rr1) or rr1 < MIN_RR1:
        failed.append("rr1")

    if not np.isfinite(rr2) or rr2 < MIN_RR2:
        failed.append("rr2")

    if not quality:
        failed.append("quality")

    if REQUIRE_TREND and not trend:
        failed.append("trend")

    if not rsi_ok:
        failed.append("rsi")

    if not volume_ok:
        failed.append("volume")

    if REQUIRE_REGIME_FOR_TRADE and not regime_ok:
        failed.append("regime")

    # --------------------------------------------------------
    # Action logic
    # --------------------------------------------------------

    if not failed:
        return "TRADE", []

    # WATCH means the model sees some evidence of strength but
    # the complete risk filters are not satisfied.
    watch_conditions = (
        np.isfinite(p3)
        and np.isfinite(p5)
        and
        max(p3, p5) >= 0.58
        and
        np.isfinite(er3)
        and
        np.isfinite(er5)
        and
        max(er3, er5) >= 0.003
    )

    if watch_conditions:
        return "WATCH", failed

    return "REJECT", failed


# ============================================================
# FORWARD OUTCOME ENGINE
# ============================================================

def forward_outcome(
    df: pd.DataFrame,
    signal_position: int,
    horizon: int,
) -> float:
    """Calculate simple close-to-close forward return."""

    future_position = (
        signal_position + horizon
    )

    if future_position >= len(df):
        return np.nan

    entry = safe_float(
        df.iloc[signal_position]["Close"]
    )

    exit_price = safe_float(
        df.iloc[future_position]["Close"]
    )

    if (
        not np.isfinite(entry)
        or
        not np.isfinite(exit_price)
        or
        entry <= 0
    ):
        return np.nan

    return (
        exit_price / entry
        - 1
    )


# ============================================================
# BACKTEST ONE SYMBOL
# ============================================================

def backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
) -> list[dict]:

    records = []

    # Need enough history for indicators and probability estimation.
    start_position = 130

    for i in range(
        start_position,
        len(df) - 6
    ):

        row = df.iloc[i]

        price = safe_float(row["Close"])
        rsi_value = safe_float(row["rsi"])
        volume_value = safe_float(
            row["relative_volume"]
        )

        if not np.isfinite(price):
            continue

        if price <= 0:
            continue

        if not np.isfinite(rsi_value):
            continue

        # ----------------------------------------------------
        # Model probabilities
        # ----------------------------------------------------

        p3 = historical_probability(
            df,
            i,
            3
        )

        p5 = historical_probability(
            df,
            i,
            5
        )

        # ----------------------------------------------------
        # Expected returns
        # ----------------------------------------------------

        volatility = safe_float(
            row["volatility_20"]
        )

        er3 = expected_return(
            p3,
            3,
            volatility
        )

        er5 = expected_return(
            p5,
            5,
            volatility
        )

        # ----------------------------------------------------
        # Risk / reward
        # ----------------------------------------------------

        (
            stop,
            target1,
            target2,
            rr1,
            rr2,
        ) = calculate_risk_reward(
            price,
            safe_float(row["atr_14"])
        )

        # ----------------------------------------------------
        # Quality filter
        # ----------------------------------------------------

        quality = (
            np.isfinite(p3)
            and
            np.isfinite(p5)
            and
            np.isfinite(er3)
            and
            np.isfinite(er5)
        )

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        trend = bool(
            row.get("trend", False)
        )

        # ----------------------------------------------------
        # RSI
        # ----------------------------------------------------

        rsi_ok = (
            np.isfinite(rsi_value)
            and
            45 <= rsi_value <= MAX_RSI
        )

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        volume_ok = (
            np.isfinite(volume_value)
            and
            volume_value >= MIN_VOLUME
        )

        # ----------------------------------------------------
        # Regime
        # ----------------------------------------------------

        regime = str(
            row.get("regime", "UNFAVORABLE")
        )

        regime_ok = (
            regime == "FAVORABLE"
        )

        # ----------------------------------------------------
        # Candidate score
        # ----------------------------------------------------

        score_components = []

        if np.isfinite(p3):
            score_components.append(
                (p3 - 0.50) * 2
            )

        if np.isfinite(p5):
            score_components.append(
                (p5 - 0.50) * 2
            )

        if np.isfinite(er3):
            score_components.append(
                er3 * 100
            )

        if np.isfinite(er5):
            score_components.append(
                er5 * 100
            )

        if np.isfinite(volume_value):
            score_components.append(
                min(volume_value - 1, 1)
            )

        if np.isfinite(rsi_value):
            # Prefer middle RSI rather than overbought extremes.
            score_components.append(
                max(
                    0,
                    1 - abs(rsi_value - 55) / 40
                )
            )

        score = (
            float(np.mean(score_components))
            if score_components
            else np.nan
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        candidate = {
            "symbol": clean_symbol(symbol),
            "date": df.index[i],
            "price": price,

            "p3": p3,
            "p5": p5,

            "er3": er3,
            "er5": er5,

            "stop": stop,
            "target1": target1,
            "target2": target2,

            "rr1": rr1,
            "rr2": rr2,

            "rsi": rsi_value,
            "volume": volume_value,

            "quality": quality,
            "trend": trend,
            "rsi_ok": rsi_ok,
            "volume_ok": volume_ok,
            "regime_ok": regime_ok,

            "regime": regime,
            "score": score,
        }

        action, failed = classify_candidate(
            candidate
        )

        candidate["classification"] = action
        candidate["failed_filters"] = ",".join(
            failed
        )

        # ----------------------------------------------------
        # Forward outcomes
        # ----------------------------------------------------

        candidate["return_1"] = forward_outcome(
            df,
            i,
            1
        )

        candidate["return_3"] = forward_outcome(
            df,
            i,
            3
        )

        candidate["return_5"] = forward_outcome(
            df,
            i,
            5
        )

        # ----------------------------------------------------
        # Simulated trade outcome
        #
        # We record it for every candidate so that we can
        # compare ALL / WATCH / TRADE / REJECT.
        # ----------------------------------------------------

        candidate["winner_1"] = (
            candidate["return_1"] > 0
            if np.isfinite(
                candidate["return_1"]
            )
            else np.nan
        )

        candidate["winner_3"] = (
            candidate["return_3"] > 0
            if np.isfinite(
                candidate["return_3"]
            )
            else np.nan
        )

        candidate["winner_5"] = (
            candidate["return_5"] > 0
            if np.isfinite(
                candidate["return_5"]
            )
            else np.nan
        )

        shares, position_value = position_size(
            price,
            stop
        )

        candidate["suggested_shares"] = shares
        candidate["position_value"] = position_value

        candidate["maximum_loss"] = (
            max(0, price - stop) * shares
        )

        records.append(candidate)

    return records


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def performance_table(
    data: pd.DataFrame,
    selection_name: str,
) -> pd.DataFrame:

    rows = []

    selection_data = data.copy()

    if selection_name != "ALL_CANDIDATES":
        selection_data = selection_data[
            selection_data["classification"]
            == selection_name
        ].copy()

    for horizon in [1, 3, 5]:

        return_column = f"return_{horizon}"

        if return_column not in selection_data:
            continue

        values = pd.to_numeric(
            selection_data[return_column],
            errors="coerce"
        ).dropna()

        if values.empty:
            continue

        winners = values[
            values > 0
        ]

        losers = values[
            values <= 0
        ]

        win_rate = (
            len(winners) / len(values)
        )

        average_net_return = values.mean()
        median_net_return = values.median()

        average_winner = (
            winners.mean()
            if not winners.empty
            else np.nan
        )

        average_loser = (
            losers.mean()
            if not losers.empty
            else np.nan
        )

        gross_wins = winners.sum()

        gross_losses = abs(
            losers.sum()
        )

        profit_factor = (
            gross_wins / gross_losses
            if gross_losses > 0
            else np.inf
        )

        rows.append({
            "selection": selection_name,
            "horizon": horizon,
            "observations": len(values),
            "win_rate": win_rate,
            "average_net_return": average_net_return,
            "median_net_return": median_net_return,
            "average_winner": average_winner,
            "average_loser": average_loser,
            "profit_factor": profit_factor,
            "best": values.max(),
            "worst": values.min(),
        })

    return pd.DataFrame(rows)


# ============================================================
# FILTER FAILURE ANALYSIS
# ============================================================

def filter_analysis(
    data: pd.DataFrame,
) -> pd.DataFrame:

    filters = [
        "p3",
        "p5",
        "er3",
        "er5",
        "rr1",
        "rr2",
        "quality",
        "trend",
        "rsi",
        "volume",
        "regime",
    ]

    rows = []

    for name in filters:

        passed = 0
        failed = 0

        for _, row in data.iterrows():

            value = row.get(name)

            if isinstance(value, (bool, np.bool_)):
                ok = bool(value)

            elif name == "p3":
                ok = (
                    np.isfinite(
                        safe_float(value)
                    )
                    and
                    safe_float(value) >= MIN_P3
                )

            elif name == "p5":
                ok = (
                    np.isfinite(
                        safe_float(value)
                    )
                    and
                    safe_float(value) >= MIN_P5
                )

            elif name == "er3":
                ok = (
                    np.isfinite(
                        safe_float(value)
                    )
                    and
                    safe_float(value) >= MIN_ER3
                )

            elif name == "er5":
                ok = (
                    np.isfinite(
                        safe_float(value)
                    )
                    and
                    safe_float(value) >= MIN_ER5
                )

            elif name == "rr1":
                ok = (
                    np.isfinite(
                        safe_float(value)
                    )
                    and
                    safe_float(value) >= MIN_RR1
                )

            elif name == "rr2":
                ok = (
                    np.isfinite(
                        safe_float(value)
                    )
                    and
                    safe_float(value) >= MIN_RR2
                )

            elif name == "rsi":
                number = safe_float(value)

                ok = (
                    np.isfinite(number)
                    and
                    45 <= number <= MAX_RSI
                )

            elif name == "volume":
                number = safe_float(value)

                ok = (
                    np.isfinite(number)
                    and
                    number >= MIN_VOLUME
                )

            elif name == "regime":
                ok = (
                    str(value)
                    == "FAVORABLE"
                )

            else:
                ok = bool(value)

            if ok:
                passed += 1
            else:
                failed += 1

        total = passed + failed

        rows.append({
            "filter": name,
            "passed": passed,
            "failed": failed,
            "pass_rate": (
                passed / total
                if total
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# INCREMENTAL FILTER ANALYSIS
# ============================================================

def incremental_analysis(
    data: pd.DataFrame,
) -> pd.DataFrame:

    filter_order = [
        "p3",
        "p5",
        "er3",
        "er5",
    ]

    current = data.copy()
    rows = []

    for filter_name in filter_order:

        if filter_name == "p3":
            current = current[
                current["p3"].apply(
                    lambda x:
                    np.isfinite(
                        safe_float(x)
                    )
                    and
                    safe_float(x) >= MIN_P3
                )
            ]

        elif filter_name == "p5":
            current = current[
                current["p5"].apply(
                    lambda x:
                    np.isfinite(
                        safe_float(x)
                    )
                    and
                    safe_float(x) >= MIN_P5
                )
            ]

        elif filter_name == "er3":
            current = current[
                current["er3"].apply(
                    lambda x:
                    np.isfinite(
                        safe_float(x)
                    )
                    and
                    safe_float(x) >= MIN_ER3
                )
            ]

        elif filter_name == "er5":
            current = current[
                current["er5"].apply(
                    lambda x:
                    np.isfinite(
                        safe_float(x)
                    )
                    and
                    safe_float(x) >= MIN_ER5
                )
            ]

        for horizon in [1, 3, 5]:

            col = f"return_{horizon}"

            values = pd.to_numeric(
                current[col],
                errors="coerce"
            ).dropna()

            if values.empty:
                continue

            rows.append({
                "filter_added": filter_name,
                "remaining_candidates": len(current),
                "horizon": horizon,
                "observations": len(values),
                "win_rate": (
                    values > 0
                ).mean(),
                "average_net_return": values.mean(),
                "median_net_return": values.median(),
                "best": values.max(),
                "worst": values.min(),
            })

    return pd.DataFrame(rows)


# ============================================================
# WALK-FORWARD CALIBRATION
# ============================================================

def probability_calibration(
    data: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # Use 3-session probability as primary calibration.
    probability = pd.to_numeric(
        data["p3"],
        errors="coerce"
    )

    outcome = pd.to_numeric(
        data["return_3"],
        errors="coerce"
    )

    temp = pd.DataFrame({
        "probability": probability,
        "outcome": outcome,
    }).dropna()

    if temp.empty:
        return pd.DataFrame()

    bins = [
        0.0,
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

    temp["probability_bucket"] = pd.cut(
        temp["probability"],
        bins=bins,
        labels=labels,
        right=False,
    )

    for bucket, group in temp.groupby(
        "probability_bucket",
        observed=False
    ):

        if group.empty:
            continue

        rows.append({
            "probability_bucket": str(bucket),
            "observations": len(group),
            "average_model_probability": group[
                "probability"
            ].mean(),
            "actual_win_rate": (
                group["outcome"] > 0
            ).mean(),
            "average_return": group[
                "outcome"
            ].mean(),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

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
    print(f"{VERSION} WALK-FORWARD BACKTEST")
    print("=" * 70)
    print(f"Backtest start: {START_DATE}")
    print(
        f"Backtest end: "
        f"{END_DATE if END_DATE else 'latest available'}"
    )
    print(
        f"Configured capital: "
        f"₹{CAPITAL:,.2f}"
    )
    print(
        f"Maximum risk/trade: "
        f"{MAX_RISK_PER_TRADE:.2%}"
    )
    print()

    # --------------------------------------------------------
    # Download Nifty
    # --------------------------------------------------------

    print("Downloading NIFTY data...")

    nifty = download_symbol(
        "^NSEI"
    )

    if nifty is None:
        raise RuntimeError(
            "Unable to download NIFTY data."
        )

    print(
        f"NIFTY observations: "
        f"{len(nifty)}"
    )

    # --------------------------------------------------------
    # Download stock universe
    # --------------------------------------------------------

    all_records = []

    successful = 0

    print()
    print(
        f"Downloading data for "
        f"{len(SYMBOLS)} instruments..."
    )

    for number, symbol in enumerate(
        SYMBOLS,
        start=1
    ):

        print(
            f"[{number:02d}/{len(SYMBOLS)}] "
            f"{symbol}"
        )

        stock = download_symbol(
            symbol
        )

        if stock is None:
            continue

        successful += 1

        try:
            featured = add_features(
                stock,
                nifty
            )

            records = backtest_symbol(
                symbol,
                featured
            )

            all_records.extend(
                records
            )

        except Exception as exc:
            print(
                f"WARNING: backtest failed "
                f"for {symbol}: {exc}"
            )

    print()
    print(
        f"Successfully processed: "
        f"{successful}"
    )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    if not all_records:
        raise RuntimeError(
            "No backtest observations were produced."
        )

    data = pd.DataFrame(
        all_records
    )

    # Ensure dates are sorted.
    data.sort_values(
        ["date", "symbol"],
        inplace=True
    )

    data.reset_index(
        drop=True,
        inplace=True
    )

    # --------------------------------------------------------
    # Save raw walk-forward audit
    # --------------------------------------------------------

    raw_file = (
        f"audit/"
        f"walkforward_v6_3_8_"
        f"{timestamp}.csv"
    )

    data.to_csv(
        raw_file,
        index=False
    )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    action_counts = (
        data["classification"]
        .value_counts()
        .rename_axis("classification")
        .reset_index(name="count")
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = performance_table(
        data,
        "ALL_CANDIDATES"
    )

    # --------------------------------------------------------
    # Action groups
    # --------------------------------------------------------

    action_tables = []

    for action in [
        "TRADE",
        "WATCH",
        "REJECT",
    ]:

        result = performance_table(
            data,
            action
        )

        if not result.empty:
            action_tables.append(
                result
            )

    if action_tables:
        action_performance = pd.concat(
            action_tables,
            ignore_index=True
        )
    else:
        action_performance = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # Filter analysis
    # --------------------------------------------------------

    filters = filter_analysis(
        data
    )

    incremental = incremental_analysis(
        data
    )

    calibration = probability_calibration(
        data
    )

    # --------------------------------------------------------
    # Save individual audit files
    # --------------------------------------------------------

    summary_file = (
        f"audit/"
        f"walkforward_summary_v6_3_8_"
        f"{timestamp}.csv"
    )

    baseline_file = (
        f"audit/"
        f"all_candidate_baseline_v6_3_8_"
        f"{timestamp}.csv"
    )

    actions_file = (
        f"audit/"
        f"action_group_performance_v6_3_8_"
        f"{timestamp}.csv"
    )

    filters_file = (
        f"audit/"
        f"filter_failures_v6_3_8_"
        f"{timestamp}.csv"
    )

    incremental_file = (
        f"audit/"
        f"incremental_filter_analysis_v6_3_8_"
        f"{timestamp}.csv"
    )

    calibration_file = (
        f"audit/"
        f"probability_calibration_v6_3_8_"
        f"{timestamp}.csv"
    )

    action_counts.to_csv(
        summary_file,
        index=False
    )

    baseline.to_csv(
        baseline_file,
        index=False
    )

    action_performance.to_csv(
        actions_file,
        index=False
    )

    filters.to_csv(
        filters_file,
        index=False
    )

    incremental.to_csv(
        incremental_file,
        index=False
    )

    calibration.to_csv(
        calibration_file,
        index=False
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V6.3.8 BACKTEST SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Total candidate observations: "
        f"{len(data)}"
    )

    print()
    print("ACTION COUNTS:")
    print(
        action_counts.to_string(
            index=False
        )
    )

    print()
    print("FILTER PERFORMANCE:")
    print(
        filters.to_string(
            index=False
        )
    )

    print()
    print("ALL-CANDIDATE BASELINE:")
    print(
        baseline.to_string(
            index=False
        )
    )

    print()
    print("ACTION GROUP PERFORMANCE:")

    if action_performance.empty:
        print(
            "No completed action outcomes."
        )
    else:
        print(
            action_performance.to_string(
                index=False
            )
        )

    print()
    print("INCREMENTAL FILTER ANALYSIS:")

    if incremental.empty:
        print(
            "No incremental outcomes."
        )
    else:
        print(
            incremental.to_string(
                index=False
            )
        )

    print()
    print("PROBABILITY CALIBRATION:")

    if calibration.empty:
        print(
            "No probability calibration "
            "observations."
        )
    else:
        print(
            calibration.to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(raw_file)
    print(summary_file)
    print(baseline_file)
    print(actions_file)
    print(filters_file)
    print(incremental_file)
    print(calibration_file)

    print()
    print(
        "V6.3.8 walk-forward backtest completed."
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Do NOT promote or loosen live trading "
        "thresholds solely from this backtest."
    )

    print(
        "Historical results do not guarantee "
        "future profitability."
    )

    print(
        "P(UP) is an empirical model estimate, "
        "not a guaranteed probability of profit."
    )

    print()


if __name__ == "__main__":
    main()
