from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# MARKET ENGINE V6.3.1
# ============================================================
#
# Corrective release for V6.3.
#
# Main objectives:
#   1. Robust ATR / volatility calculation
#   2. Valid positive stop-loss
#   3. Realistic entry range
#   4. No negative-return "opportunities"
#   5. Separate TRADE SETUPS from WATCHLIST
#   6. Multi-horizon probability / return model
#   7. Risk-based position sizing
#   8. IPO data failure handling
#   9. Walk-forward backtesting
#
# This is a probabilistic research tool.
# It does NOT guarantee profit.
# ============================================================


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("market_engine_v6_3_1")

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "ret1",
    "ret3",
    "ret5",
    "ret10",
    "ret20",
    "rsi14",
    "dist_ema5",
    "dist_ema20",
    "dist_sma50",
    "ema20_slope10",
    "sma50_slope10",
    "atr_pct",
    "vol20",
    "volume_ratio",
    "range_pct",
    "close_location",
    "relative3",
    "relative5",
    "relative10",
    "relative20",
    "nifty_ret3",
    "nifty_ret10",
    "nifty_above_sma50",
    "nifty_sma50_slope10",
    "vix_level",
    "vix_change5",
]


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:

    bot_token: Optional[str] = None
    chat_id: Optional[str] = None

    capital: float = 100000.0

    top_n: int = 3

    lookback: int = 504

    min_train: int = 700

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    min_price: float = 50.0

    min_turnover_cr: float = 20.0

    # --------------------------------------------------------
    # Actual trade thresholds
    # --------------------------------------------------------

    high_prob: float = 0.60
    trade_prob: float = 0.55

    high_return: float = 0.80
    trade_return: float = 0.50

    high_score: float = 0.30
    trade_score: float = 0.10

    # --------------------------------------------------------
    # Watchlist thresholds
    # --------------------------------------------------------

    watch_prob: float = 0.52
    watch_return: float = 0.15

    # --------------------------------------------------------
    # Technical controls
    # --------------------------------------------------------

    max_rsi_entry: float = 76.0

    # Do not chase a stock more than this many ATR
    # above EMA20.
    max_extension_atr: float = 2.00

    # Absolute safety ceiling for daily ATR.
    #
    # This is deliberately generous enough for volatile
    # stocks but prevents corrupted data such as ATR > price.
    max_atr_pct: float = 12.0

    # Stop-loss ATR multiplier.
    stop_atr: float = 1.25

    min_rr: float = 1.50

    # Maximum entry distance from current market price.
    max_entry_distance_pct: float = 2.50

    # --------------------------------------------------------
    # Position sizing
    # --------------------------------------------------------

    risk_pct: float = 0.75

    max_position_pct: float = 0.25

    # --------------------------------------------------------
    # Backtest assumptions
    # --------------------------------------------------------

    slippage_pct: float = 0.05

    brokerage_pct: float = 0.03

    # --------------------------------------------------------
    # Audit files
    # --------------------------------------------------------

    alert_history: str = (
        "alert_history_v6_3_1.csv"
    )

    candidate_history: str = (
        "candidate_history_v6_3_1.csv"
    )

    rejected_history: str = (
        "rejected_candidates_v6_3_1.csv"
    )

    backtest_file: str = (
        "backtest_v6_3_1.csv"
    )

    # --------------------------------------------------------
    # Market indexes
    # --------------------------------------------------------

    nifty: str = "^NSEI"

    vix: str = "^INDIAVIX"

    # --------------------------------------------------------
    # Liquid Indian equity universe
    # --------------------------------------------------------

    universe: list = field(
        default_factory=lambda: [

            "RELIANCE.NS",
            "TCS.NS",
            "HDFCBANK.NS",
            "ICICIBANK.NS",
            "INFY.NS",
            "BHARTIARTL.NS",
            "SBIN.NS",
            "ITC.NS",
            "HINDUNILVR.NS",
            "LT.NS",
            "BAJFINANCE.NS",
            "HCLTECH.NS",
            "KOTAKBANK.NS",
            "SUNPHARMA.NS",
            "MARUTI.NS",
            "M&M.NS",
            "AXISBANK.NS",
            "TITAN.NS",
            "NTPC.NS",
            "ADANIENT.NS",
            "BAJAJFINSV.NS",
            "ONGC.NS",
            "POWERGRID.NS",
            "ADANIPORTS.NS",
            "COALINDIA.NS",
            "WIPRO.NS",
            "JSWSTEEL.NS",
            "TATASTEEL.NS",
            "NESTLEIND.NS",
            "TATAMOTORS.NS",
            "ASIANPAINT.NS",
            "HAL.NS",
            "BEL.NS",
            "GRASIM.NS",
            "SBILIFE.NS",
            "TECHM.NS",
            "HDFCLIFE.NS",
            "CIPLA.NS",
            "TRENT.NS",
            "DRREDDY.NS",
            "EICHERMOT.NS",
            "BAJAJ-AUTO.NS",
            "APOLLOHOSP.NS",
            "BRITANNIA.NS",
            "DIVISLAB.NS",
            "INDUSINDBK.NS",
            "HEROMOTOCO.NS",
            "SHRIRAMFIN.NS",
            "PIDILITIND.NS",
            "GODREJCP.NS",
            "DABUR.NS",
            "SIEMENS.NS",
            "DLF.NS",
            "VEDL.NS",
            "LTIM.NS",
            "AMBUJACEM.NS",
            "BANKBARODA.NS",
            "PNB.NS",
            "GAIL.NS",
            "IOC.NS",
            "BPCL.NS",
            "TATAPOWER.NS",
            "TATACONSUM.NS",
            "PIIND.NS",
            "HAVELLS.NS",
            "MOTHERSON.NS",
            "BOSCHLTD.NS",
            "CANBK.NS",
            "IDFCFIRSTB.NS",
            "AUROPHARMA.NS",
            "LUPIN.NS",
            "TORNTPHARM.NS",
            "COLPAL.NS",
            "MARICO.NS",
            "SRF.NS",
            "PAGEIND.NS",
            "MUTHOOTFIN.NS",
            "CHOLAFIN.NS",
            "BALKRISIND.NS",
            "ICICIPRULI.NS",
            "ICICIGI.NS",
            "INDIGO.NS",
            "NAUKRI.NS",
            "PFC.NS",
            "RECLTD.NS",
            "JINDALSTEL.NS",
            "SAIL.NS",
            "HINDALCO.NS",
            "NMDC.NS",
            "UPL.NS",
            "BHARATFORG.NS",
            "CUMMINSIND.NS",
            "ABB.NS",
            "POLYCAB.NS",
            "PERSISTENT.NS",
        ]
    )


# ============================================================
# HELPERS
# ============================================================

def now_ist() -> datetime:
    return datetime.now(IST)


def safe_float(
    value,
    default=np.nan,
):
    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return default

    except Exception:
        return default


# ============================================================
# DATA NORMALIZATION
# ============================================================

def extract_ticker_series(
    dataframe,
    ticker,
):
    """
    Robustly extract one ticker from either:
      - normal DataFrame
      - yfinance MultiIndex DataFrame

    This prevents accidental extraction of the wrong
    column/level and is important for ATR correctness.
    """

    if dataframe is None:
        return None

    if isinstance(
        dataframe.columns,
        pd.MultiIndex,
    ):

        # Typical yfinance layout:
        # Price / Ticker
        for level in range(
            dataframe.columns.nlevels
        ):

            try:

                values = (
                    dataframe
                    .columns
                    .get_level_values(level)
                )

                if ticker in values:

                    result = dataframe.xs(
                        ticker,
                        axis=1,
                        level=level,
                    )

                    if isinstance(
                        result,
                        pd.DataFrame,
                    ):

                        if result.shape[1] == 1:

                            return result.iloc[:, 0]

                    return result

            except Exception:
                pass

    if ticker in dataframe.columns:

        return dataframe[ticker]

    return None


def download_market_data(
    tickers,
    period="3y",
    retries=3,
):

    tickers = list(
        dict.fromkeys(tickers)
    )

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            raw = yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )

            if raw.empty:
                raise RuntimeError(
                    "Yahoo Finance returned empty data."
                )

            result = {}

            for field_name in [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]:

                if isinstance(
                    raw.columns,
                    pd.MultiIndex,
                ):

                    level0 = list(
                        raw.columns
                        .get_level_values(0)
                    )

                    if field_name in level0:

                        result[field_name] = (
                            raw[field_name]
                        )

                    else:

                        level1 = list(
                            raw.columns
                            .get_level_values(1)
                        )

                        if field_name in level1:

                            result[field_name] = (
                                raw.xs(
                                    field_name,
                                    axis=1,
                                    level=1,
                                )
                            )

                else:

                    if field_name in raw.columns:

                        result[field_name] = (
                            raw[[field_name]]
                        )

            if "Close" not in result:

                raise RuntimeError(
                    "Close data unavailable."
                )

            return result

        except Exception as exc:

            last_error = exc

            log.warning(
                "Market data attempt %d/%d failed: %s",
                attempt,
                retries,
                exc,
            )

            if attempt < retries:
                time.sleep(
                    attempt * 2
                )

    raise RuntimeError(
        f"Market data unavailable: {last_error}"
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    close,
    period=14,
):

    delta = close.diff()

    gain = (
        delta
        .clip(lower=0)
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    loss = (
        -delta
        .clip(upper=0)
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    rs = (
        gain
        / loss.replace(
            0,
            np.nan,
        )
    )

    return (
        100
        - (
            100
            / (1 + rs)
        )
    )


# ============================================================
# ROBUST ATR
# ============================================================

def calculate_atr(
    high,
    low,
    close,
    period=14,
):

    high = pd.to_numeric(
        high,
        errors="coerce",
    )

    low = pd.to_numeric(
        low,
        errors="coerce",
    )

    close = pd.to_numeric(
        close,
        errors="coerce",
    )

    previous_close = close.shift(1)

    tr1 = (
        high
        - low
    ).abs()

    tr2 = (
        high
        - previous_close
    ).abs()

    tr3 = (
        low
        - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3,
        ],
        axis=1,
    ).max(
        axis=1,
        skipna=True,
    )

    # First calculate normal ATR.
    atr = (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    # --------------------------------------------------------
    # Robust protection against corrupted historical candles.
    #
    # If one candle is abnormally large, a median-based
    # rolling scale prevents that observation from dominating
    # the current risk calculation.
    # --------------------------------------------------------

    median_tr = (
        true_range
        .rolling(
            60,
            min_periods=20,
        )
        .median()
    )

    safe_tr = true_range.copy()

    cap = (
        median_tr
        * 5.0
    )

    valid_cap = cap.notna()

    safe_tr.loc[
        valid_cap
    ] = np.minimum(
        safe_tr.loc[
            valid_cap
        ],
        cap.loc[
            valid_cap
        ],
    )

    robust_atr = (
        safe_tr
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    return robust_atr


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    close,
    high,
    low,
    volume,
    nifty,
    vix,
):

    close = pd.to_numeric(
        close,
        errors="coerce",
    )

    high = pd.to_numeric(
        high,
        errors="coerce",
    )

    low = pd.to_numeric(
        low,
        errors="coerce",
    )

    volume = pd.to_numeric(
        volume,
        errors="coerce",
    )

    close = close.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    high = high.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    low = low.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    volume = volume.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    nifty = (
        pd.to_numeric(
            nifty,
            errors="coerce",
        )
        .reindex(
            close.index
        )
        .ffill()
    )

    vix = (
        pd.to_numeric(
            vix,
            errors="coerce",
        )
        .reindex(
            close.index
        )
        .ffill()
    )

    ema5 = (
        close
        .ewm(
            span=5,
            adjust=False,
        )
        .mean()
    )

    ema20 = (
        close
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    sma50 = (
        close
        .rolling(50)
        .mean()
    )

    nifty_sma50 = (
        nifty
        .rolling(50)
        .mean()
    )

    atr = calculate_atr(
        high,
        low,
        close,
    )

    returns = close.pct_change()

    frame = pd.DataFrame(
        index=close.index
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    for n in [
        1,
        3,
        5,
        10,
        20,
    ]:

        frame[
            f"ret{n}"
        ] = close.pct_change(n)

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    frame["rsi14"] = (
        calculate_rsi(close)
    )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    frame["dist_ema5"] = (
        close / ema5 - 1
    )

    frame["dist_ema20"] = (
        close / ema20 - 1
    )

    frame["dist_sma50"] = (
        close / sma50 - 1
    )

    frame["ema20_slope10"] = (
        ema20.pct_change(10)
    )

    frame["sma50_slope10"] = (
        sma50.pct_change(10)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    frame["atr_pct"] = (
        atr / close
    )

    frame["vol20"] = (
        returns
        .rolling(20)
        .std()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    frame["volume_ratio"] = (
        volume
        / volume
        .rolling(
            20,
            min_periods=10,
        )
        .median()
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    frame["range_pct"] = (
        high - low
    ) / close

    frame["close_location"] = (
        close - low
    ) / (
        high - low
    ).replace(
        0,
        np.nan,
    )

    # --------------------------------------------------------
    # Relative strength
    # --------------------------------------------------------

    nifty_ret3 = (
        nifty.pct_change(3)
    )

    nifty_ret10 = (
        nifty.pct_change(10)
    )

    nifty_ret20 = (
        nifty.pct_change(20)
    )

    frame["relative3"] = (
        frame["ret3"]
        - nifty_ret3
    )

    frame["relative5"] = (
        frame["ret5"]
        - nifty.pct_change(5)
    )

    frame["relative10"] = (
        frame["ret10"]
        - nifty_ret10
    )

    frame["relative20"] = (
        frame["ret20"]
        - nifty_ret20
    )

    # --------------------------------------------------------
    # Market regime features
    # --------------------------------------------------------

    frame["nifty_ret3"] = (
        nifty_ret3
    )

    frame["nifty_ret10"] = (
        nifty_ret10
    )

    frame[
        "nifty_above_sma50"
    ] = (
        nifty
        > nifty_sma50
    ).astype(float)

    frame[
        "nifty_sma50_slope10"
    ] = (
        nifty_sma50
        .pct_change(10)
    )

    # --------------------------------------------------------
    # VIX
    # --------------------------------------------------------

    frame["vix_level"] = vix

    frame["vix_change5"] = (
        vix.pct_change(5)
    )

    # --------------------------------------------------------
    # Raw values
    # --------------------------------------------------------

    frame["price"] = close

    frame["atr"] = atr

    frame["volume"] = volume

    # --------------------------------------------------------
    # Forward targets
    # --------------------------------------------------------

    for horizon in [
        1,
        3,
        5,
    ]:

        frame[
            f"target{horizon}"
        ] = (
            close.shift(-horizon)
            / close
            - 1
        )

    return frame


# ============================================================
# MODEL FITTING
# ============================================================

def fit_model(
    training,
    horizon,
    cfg,
):

    target = (
        f"target{horizon}"
    )

    x = (
        training[
            FEATURES
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    y = pd.to_numeric(
        training[target],
        errors="coerce",
    )

    valid = (
        x.notna().all(axis=1)
        & y.notna()
    )

    x = x.loc[valid]
    y = y.loc[valid]

    if len(x) < cfg.min_train:

        return None

    if y.nunique() < 2:

        return None

    # --------------------------------------------------------
    # Winsorize return target.
    # --------------------------------------------------------

    low = y.quantile(
        0.01
    )

    high = y.quantile(
        0.99
    )

    y_clean = y.clip(
        low,
        high,
    )

    return_model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=10.0
                ),
            ),
        ]
    )

    direction_model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "logit",
                LogisticRegression(
                    C=0.20,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return_model.fit(
        x,
        y_clean,
    )

    direction_model.fit(
        x,
        (
            y > 0
        ).astype(int),
    )

    return (
        return_model,
        direction_model,
    )


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(
    nifty,
    vix,
):

    nifty = pd.to_numeric(
        nifty,
        errors="coerce",
    ).dropna()

    sma50 = (
        nifty
        .rolling(50)
        .mean()
    )

    slope10 = (
        sma50
        .pct_change(10)
    )

    nifty_value = float(
        nifty.iloc[-1]
    )

    sma_value = float(
        sma50.iloc[-1]
    )

    slope_value = float(
        slope10.iloc[-1]
    )

    if len(vix) > 0:

        vix_value = float(
            pd.to_numeric(
                vix,
                errors="coerce",
            )
            .dropna()
            .iloc[-1]
        )

    else:

        vix_value = np.nan

    score = 0

    if nifty_value > sma_value:
        score += 1
    else:
        score -= 1

    if slope_value > 0:
        score += 1
    else:
        score -= 1

    if np.isfinite(vix_value):

        if vix_value < 14:
            score += 1

        elif vix_value > 20:
            score -= 1

    if score >= 2:
        label = "FAVORABLE"

    elif score <= -2:
        label = "UNFAVORABLE"

    else:
        label = "MIXED"

    return {
        "label": label,
        "nifty": nifty_value,
        "sma50": sma_value,
        "slope": slope_value * 100,
        "vix": vix_value,
        "score": score,
    }


# ============================================================
# TIER
# ============================================================

def determine_trade_tier(
    probability,
    predicted_return,
    score,
    risk_reward,
    cfg,
):

    if (
        probability >= cfg.high_prob
        and predicted_return >= cfg.high_return
        and score >= cfg.high_score
        and risk_reward >= cfg.min_rr
    ):

        return "HIGH-CONFIDENCE"

    if (
        probability >= cfg.trade_prob
        and predicted_return >= cfg.trade_return
        and score >= cfg.trade_score
        and risk_reward >= cfg.min_rr
    ):

        return "TRADEABLE"

    return "NO TRADE"


def determine_watchlist_status(
    probability,
    predicted_return,
    score,
    cfg,
):

    if (
        probability >= cfg.watch_prob
        and predicted_return >= cfg.watch_return
        and score >= 0
    ):

        return "WATCHLIST"

    return "REJECTED"


# ============================================================
# VALIDATE ATR
# ============================================================

def validate_volatility(
    price,
    atr,
    atr_pct,
    cfg,
):

    if not np.isfinite(price):
        return False

    if not np.isfinite(atr):
        return False

    if not np.isfinite(atr_pct):
        return False

    if price <= 0:
        return False

    if atr <= 0:
        return False

    if atr_pct <= 0:
        return False

    if atr_pct > cfg.max_atr_pct:
        return False

    # ATR cannot reasonably be larger than price.
    if atr >= price:
        return False

    return True


# ============================================================
# TRADE PLAN
# ============================================================

def build_trade_plan(
    row,
    cfg,
):

    price = float(
        row["price"]
    )

    atr = float(
        row["atr"]
    )

    atr_pct = (
        atr
        / price
        * 100
    )

    if not validate_volatility(
        price,
        atr,
        atr_pct,
        cfg,
    ):

        return None

    # --------------------------------------------------------
    # Current price is the reference entry.
    #
    # We no longer create enormous ATR-based ranges.
    # --------------------------------------------------------

    entry_low = (
        price
        * (
            1
            - cfg.max_entry_distance_pct
            / 100
        )
    )

    entry_high = (
        price
        * (
            1
            + 0.50
            * cfg.max_entry_distance_pct
            / 100
        )
    )

    # --------------------------------------------------------
    # Stop based on ATR.
    # --------------------------------------------------------

    stop = (
        price
        - cfg.stop_atr * atr
    )

    # Safety floor:
    # Stop must remain positive and below entry.
    # --------------------------------------------------------

    minimum_stop = (
        price * 0.90
    )

    stop = max(
        stop,
        minimum_stop,
    )

    if (
        not np.isfinite(stop)
        or stop <= 0
        or stop >= price
    ):

        return None

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk_per_share = (
        price - stop
    )

    risk_pct = (
        risk_per_share
        / price
        * 100
    )

    if risk_pct <= 0:
        return None

    # --------------------------------------------------------
    # Expected return
    # --------------------------------------------------------

    pred3 = float(
        row["pred3"]
    )

    pred5 = float(
        row["pred5"]
    )

    expected_return = max(
        pred3,
        0.0,
    )

    # Target 1:
    # conservative fraction of expected move.
    # --------------------------------------------------------

    target1_pct = max(
        expected_return * 0.55,
        risk_pct * 1.50,
    )

    target2_pct = max(
        expected_return,
        risk_pct * 2.00,
    )

    # Avoid absurd targets.
    target1_pct = min(
        target1_pct,
        8.0,
    )

    target2_pct = min(
        target2_pct,
        12.0,
    )

    target1 = (
        price
        * (
            1
            + target1_pct / 100
        )
    )

    target2 = (
        price
        * (
            1
            + target2_pct / 100
        )
    )

    rr1 = (
        target1_pct
        / risk_pct
    )

    rr2 = (
        target2_pct
        / risk_pct
    )

    if (
        not np.isfinite(rr2)
        or rr2 < cfg.min_rr
    ):

        return None

    # --------------------------------------------------------
    # Chasing detection
    # --------------------------------------------------------

    extension_atr = float(
        row["extension_atr"]
    )

    rsi = float(
        row["rsi"]
    )

    if (
        extension_atr
        > cfg.max_extension_atr
        or rsi
        > cfg.max_rsi_entry
    ):

        action = (
            "WAIT FOR PULLBACK"
        )

    else:

        action = (
            "BUY ON CONFIRMATION"
        )

    # --------------------------------------------------------
    # Holding period
    # --------------------------------------------------------

    if (
        pred5 > pred3
        and float(row["p5"])
        >= float(row["p3"]) - 0.03
    ):

        holding = "3-5 sessions"

    else:

        holding = "1-3 sessions"

    return {
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "risk_pct": risk_pct,
        "rr1": rr1,
        "rr2": rr2,
        "action": action,
        "holding_period": holding,
    }


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    row,
    cfg,
):

    price = float(
        row["price"]
    )

    stop = float(
        row["stop"]
    )

    if (
        price <= 0
        or stop <= 0
        or stop >= price
    ):

        return (
            0,
            0.0,
            0.0,
        )

    risk_per_share = (
        price - stop
    )

    risk_budget = (
        cfg.capital
        * cfg.risk_pct
        / 100
    )

    max_position_value = (
        cfg.capital
        * cfg.max_position_pct
    )

    by_risk = math.floor(
        risk_budget
        / risk_per_share
    )

    by_capital = math.floor(
        max_position_value
        / price
    )

    shares = max(
        0,
        min(
            by_risk,
            by_capital,
        ),
    )

    value = (
        shares * price
    )

    max_loss = (
        shares
        * risk_per_share
    )

    return (
        shares,
        value,
        max_loss,
    )


# ============================================================
# IPO DATA
# ============================================================

def fetch_ipo_data():

    session = requests.Session()

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/151.0 Safari/537.36",

        "Accept":
            "application/json,text/plain,*/*",

        "Referer":
            "https://www.nseindia.com/",
    }

    records = []

    try:

        session.get(
            "https://www.nseindia.com/",
            headers=headers,
            timeout=15,
        )

    except Exception as exc:

        log.warning(
            "NSE session initialization failed: %s",
            exc,
        )

    endpoints = [
        (
            "https://www.nseindia.com/"
            "api/ipo-current-issues",
            "OPEN",
        ),

        (
            "https://www.nseindia.com/"
            "api/ipo-upcoming-issues",
            "UPCOMING",
        ),
    ]

    endpoint_failures = 0

    for url, status in endpoints:

        try:

            response = session.get(
                url,
                headers=headers,
                timeout=20,
            )

            response.raise_for_status()

            data = response.json()

            if isinstance(
                data,
                dict,
            ):

                data = (
                    data.get(
                        "data",
                        data.get(
                            "records",
                            [],
                        ),
                    )
                )

            if not isinstance(
                data,
                list,
            ):

                endpoint_failures += 1

                continue

            for item in data[:25]:

                if not isinstance(
                    item,
                    dict,
                ):

                    continue

                name = (
                    item.get(
                        "companyName"
                    )
                    or item.get(
                        "company"
                    )
                    or item.get(
                        "symbol"
                    )
                    or item.get(
                        "name"
                    )
                )

                if not name:
                    continue

                records.append(
                    {
                        "name":
                            str(name),

                        "status":
                            status,

                        "start":
                            (
                                item.get(
                                    "issueStartDate"
                                )
                                or item.get(
                                    "startDate"
                                )
                                or item.get(
                                    "issueStart"
                                )
                                or "-"
                            ),

                        "end":
                            (
                                item.get(
                                    "issueEndDate"
                                )
                                or item.get(
                                    "endDate"
                                )
                                or item.get(
                                    "issueEnd"
                                )
                                or "-"
                            ),

                        "price":
                            (
                                item.get(
                                    "priceBand"
                                )
                                or item.get(
                                    "issuePrice"
                                )
                                or item.get(
                                    "price"
                                )
                                or "-"
                            ),

                        "subscription":
                            (
                                item.get(
                                    "subscription"
                                )
                                or item.get(
                                    "subscriptionRatio"
                                )
                                or "-"
                            ),
                    }
                )

        except Exception as exc:

            endpoint_failures += 1

            log.warning(
                "IPO %s retrieval failed: %s",
                status,
                exc,
            )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    unique = []

    seen = set()

    for item in records:

        key = (
            item["name"].strip().lower(),
            item["status"],
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            item
        )

    if unique:

        return unique

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Never say "no IPOs" when retrieval failed.
    # --------------------------------------------------------

    return [
        {
            "name":
                "IPO DATA UNAVAILABLE",

            "status":
                "RETRIEVAL FAILED",

            "start":
                "-",

            "end":
                "-",

            "price":
                "-",

            "subscription":
                (
                    "Verify current/upcoming "
                    "issues directly on NSE."
                ),
        }
    ]


# ============================================================
# ENGINE
# ============================================================

class MarketEngineV631:

    def __init__(
        self,
        cfg,
    ):

        self.cfg = cfg

    # ========================================================
    # SCAN
    # ========================================================

    def scan(self):

        tickers = (
            self.cfg.universe
            + [
                self.cfg.nifty,
                self.cfg.vix,
            ]
        )

        data = download_market_data(
            tickers,
            period="3y",
        )

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        nifty = extract_ticker_series(
            close,
            self.cfg.nifty,
        )

        if nifty is None:

            raise RuntimeError(
                "Nifty data unavailable."
            )

        nifty = (
            pd.to_numeric(
                nifty,
                errors="coerce",
            )
            .dropna()
        )

        vix = extract_ticker_series(
            close,
            self.cfg.vix,
        )

        if vix is None:

            vix = pd.Series(
                dtype=float
            )

        else:

            vix = (
                pd.to_numeric(
                    vix,
                    errors="coerce",
                )
                .dropna()
            )

        latest = nifty.index[-1]

        frames = {}

        training_parts = {
            1: [],
            3: [],
            5: [],
        }

        # ----------------------------------------------------
        # Build each stock
        # ----------------------------------------------------

        for ticker in self.cfg.universe:

            stock_close = (
                extract_ticker_series(
                    close,
                    ticker,
                )
            )

            stock_high = (
                extract_ticker_series(
                    high,
                    ticker,
                )
            )

            stock_low = (
                extract_ticker_series(
                    low,
                    ticker,
                )
            )

            stock_volume = (
                extract_ticker_series(
                    volume,
                    ticker,
                )
            )

            if any(
                item is None
                for item in [
                    stock_close,
                    stock_high,
                    stock_low,
                    stock_volume,
                ]
            ):

                continue

            stock_close = (
                pd.to_numeric(
                    stock_close,
                    errors="coerce",
                )
            )

            stock_high = (
                pd.to_numeric(
                    stock_high,
                    errors="coerce",
                )
            )

            stock_low = (
                pd.to_numeric(
                    stock_low,
                    errors="coerce",
                )
            )

            stock_volume = (
                pd.to_numeric(
                    stock_volume,
                    errors="coerce",
                )
            )

            valid_price = (
                stock_close
                .dropna()
            )

            if len(valid_price) < 650:
                continue

            frame = build_features(
                stock_close,
                stock_high,
                stock_low,
                stock_volume,
                nifty,
                vix,
            )

            frames[ticker] = frame

            historical = (
                frame[
                    frame.index < latest
                ]
                .tail(
                    self.cfg.lookback
                )
            )

            for horizon in [
                1,
                3,
                5,
            ]:

                training_parts[
                    horizon
                ].append(
                    historical
                )

        # ----------------------------------------------------
        # Fit models
        # ----------------------------------------------------

        models = {}

        for horizon in [
            1,
            3,
            5,
        ]:

            if not training_parts[
                horizon
            ]:

                continue

            training = pd.concat(
                training_parts[
                    horizon
                ],
                ignore_index=True,
            )

            models[horizon] = (
                fit_model(
                    training,
                    horizon,
                    self.cfg,
                )
            )

        if models.get(3) is None:

            raise RuntimeError(
                "Unable to fit V6.3.1 3-day model."
            )

        trade_candidates = []

        watch_candidates = []

        rejected = []

        # ----------------------------------------------------
        # Score stocks
        # ----------------------------------------------------

        for ticker, frame in frames.items():

            if latest not in frame.index:
                continue

            row = frame.loc[
                [latest]
            ]

            x = row[
                FEATURES
            ]

            if x.isna().any(
                axis=1
            ).iloc[0]:

                continue

            predictions = {}

            probabilities = {}

            for horizon in [
                1,
                3,
                5,
            ]:

                bundle = models.get(
                    horizon
                )

                if bundle is None:
                    continue

                return_model = (
                    bundle[0]
                )

                direction_model = (
                    bundle[1]
                )

                predictions[
                    horizon
                ] = (
                    float(
                        return_model
                        .predict(x)[0]
                    )
                    * 100
                )

                probabilities[
                    horizon
                ] = (
                    float(
                        direction_model
                        .predict_proba(x)[
                            0,
                            1,
                        ]
                    )
                )

            if 3 not in predictions:
                continue

            p1 = probabilities.get(
                1,
                probabilities[3],
            )

            p3 = probabilities[3]

            p5 = probabilities.get(
                5,
                probabilities[3],
            )

            pred1 = predictions.get(
                1,
                predictions[3],
            )

            pred3 = predictions[3]

            pred5 = predictions.get(
                5,
                predictions[3],
            )

            # ------------------------------------------------
            # Basic values
            # ------------------------------------------------

            price = safe_float(
                row[
                    "price"
                ].iloc[0]
            )

            atr = safe_float(
                row[
                    "atr"
                ].iloc[0]
            )

            rsi = safe_float(
                row[
                    "rsi14"
                ].iloc[0]
            )

            volume_ratio = safe_float(
                row[
                    "volume_ratio"
                ].iloc[0]
            )

            dist_ema20 = safe_float(
                row[
                    "dist_ema20"
                ].iloc[0]
            )

            atr_pct = (
                atr
                / price
                * 100
                if price > 0
                else np.nan
            )

            extension_atr = (
                dist_ema20
                * price
                / atr
                if (
                    np.isfinite(
                        dist_ema20
                    )
                    and np.isfinite(atr)
                    and atr > 0
                )
                else np.nan
            )

            turnover_cr = (
                price
                * safe_float(
                    row[
                        "volume"
                    ].iloc[0],
                    0,
                )
                / 1e7
            )

            # ------------------------------------------------
            # Reject invalid market data
            # ------------------------------------------------

            if not validate_volatility(
                price,
                atr,
                atr_pct,
                self.cfg,
            ):

                rejected.append(
                    {
                        "ticker":
                            ticker.replace(
                                ".NS",
                                "",
                            ),

                        "reason":
                            "INVALID_ATR",

                        "price":
                            price,

                        "atr":
                            atr,

                        "atr_pct":
                            atr_pct,
                    }
                )

                continue

            if (
                price
                < self.cfg.min_price
            ):

                rejected.append(
                    {
                        "ticker":
                            ticker.replace(
                                ".NS",
                                "",
                            ),

                        "reason":
                            "PRICE_TOO_LOW",

                        "price":
                            price,
                    }
                )

                continue

            if (
                turnover_cr
                < self.cfg.min_turnover_cr
            ):

                rejected.append(
                    {
                        "ticker":
                            ticker.replace(
                                ".NS",
                                "",
                            ),

                        "reason":
                            "LOW_LIQUIDITY",

                        "turnover_cr":
                            turnover_cr,
                    }
                )

                continue

            # ------------------------------------------------
            # Ensemble
            # ------------------------------------------------

            ensemble_probability = (
                0.20 * p1
                + 0.55 * p3
                + 0.25 * p5
            )

            ensemble_return = (
                0.20 * pred1
                + 0.55 * pred3
                + 0.25 * pred5
            )

            # ------------------------------------------------
            # Risk-adjusted score
            # ------------------------------------------------

            score = (
                ensemble_return
                * (
                    2
                    * ensemble_probability
                    - 1
                )
                - 0.20
                * atr_pct
            )

            # ------------------------------------------------
            # Trend
            # ------------------------------------------------

            trend_aligned = (
                safe_float(
                    row[
                        "dist_ema20"
                    ].iloc[0]
                ) > 0
                and
                safe_float(
                    row[
                        "dist_sma50"
                    ].iloc[0]
                ) > 0
                and
                safe_float(
                    row[
                        "ema20_slope10"
                    ].iloc[0]
                ) > 0
            )

            if trend_aligned:
                score += 0.12

            # ------------------------------------------------
            # Relative strength
            # ------------------------------------------------

            relative5 = (
                safe_float(
                    row[
                        "relative5"
                    ].iloc[0]
                )
                * 100
            )

            if relative5 > 0:

                score += min(
                    0.10,
                    relative5 * 0.02,
                )

            # ------------------------------------------------
            # Volume confirmation
            # ------------------------------------------------

            if volume_ratio > 1.15:
                score += 0.05

            # ------------------------------------------------
            # RSI penalty
            # ------------------------------------------------

            if rsi > 70:
                score -= 0.08

            if rsi > 76:
                score -= 0.08

            # ------------------------------------------------
            # Extension penalty
            # ------------------------------------------------

            if (
                np.isfinite(
                    extension_atr
                )
                and extension_atr > 1.25
            ):

                score -= 0.10

            # ------------------------------------------------
            # Small market-regime modifier.
            #
            # Mixed market does NOT prevent stock-specific
            # trades.
            # ------------------------------------------------

            nifty_above = safe_float(
                row[
                    "nifty_above_sma50"
                ].iloc[0]
            )

            if nifty_above < 0.5:
                score -= 0.03

            # ------------------------------------------------
            # Candidate object
            # ------------------------------------------------

            candidate = {
                "ticker":
                    ticker.replace(
                        ".NS",
                        "",
                    ),

                "price":
                    price,

                "atr":
                    atr,

                "atr_pct":
                    atr_pct,

                "rsi":
                    rsi,

                "volume_ratio":
                    volume_ratio,

                "extension_atr":
                    extension_atr,

                "relative5":
                    relative5,

                "trend":
                    trend_aligned,

                "p1":
                    p1,

                "p3":
                    p3,

                "p5":
                    p5,

                "pred1":
                    pred1,

                "pred3":
                    pred3,

                "pred5":
                    pred5,

                "ensemble_probability":
                    ensemble_probability,

                "ensemble_return":
                    ensemble_return,

                "score":
                    score,

                "turnover_cr":
                    turnover_cr,
            }

            # ------------------------------------------------
            # First create a safe trade plan.
            # ------------------------------------------------

            plan = build_trade_plan(
                candidate,
                self.cfg,
            )

            # ------------------------------------------------
            # No valid trade plan.
            # ------------------------------------------------

            if plan is None:

                watch_status = (
                    determine_watchlist_status(
                        ensemble_probability,
                        ensemble_return,
                        score,
                        self.cfg,
                    )
                )

                candidate[
                    "status"
                ] = watch_status

                candidate[
                    "reason"
                ] = (
                    "No valid risk/reward "
                    "trade plan."
                )

                if watch_status == "WATCHLIST":

                    watch_candidates.append(
                        candidate
                    )

                else:

                    rejected.append(
                        candidate
                    )

                continue

            candidate.update(
                plan
            )

            # ------------------------------------------------
            # Position sizing
            # ------------------------------------------------

            (
                shares,
                position_value,
                max_loss,
            ) = calculate_position_size(
                candidate,
                self.cfg,
            )

            candidate[
                "shares"
            ] = shares

            candidate[
                "position_value"
            ] = position_value

            candidate[
                "max_loss"
            ] = max_loss

            # ------------------------------------------------
            # Actual trade tier
            # ------------------------------------------------

            tier = determine_trade_tier(
                ensemble_probability,
                ensemble_return,
                score,
                candidate["rr2"],
                self.cfg,
            )

            # ------------------------------------------------
            # Critical V6.3.1 rule:
            #
            # A negative expected return can NEVER be a
            # trade candidate or a watchlist "opportunity".
            # ------------------------------------------------

            if (
                ensemble_return <= 0
            ):

                candidate[
                    "status"
                ] = "REJECTED"

                candidate[
                    "reason"
                ] = (
                    "Negative expected return."
                )

                rejected.append(
                    candidate
                )

                continue

            # ------------------------------------------------
            # Actual trade
            # ------------------------------------------------

            if tier in [
                "HIGH-CONFIDENCE",
                "TRADEABLE",
            ]:

                # If the model says WAIT FOR PULLBACK,
                # do not call it an immediate BUY.
                if (
                    candidate["action"]
                    == "WAIT FOR PULLBACK"
                ):

                    candidate[
                        "status"
                    ] = "WATCHLIST"

                    candidate[
                        "reason"
                    ] = (
                        "Good setup but "
                        "currently extended."
                    )

                    watch_candidates.append(
                        candidate
                    )

                else:

                    candidate[
                        "status"
                    ] = tier

                    trade_candidates.append(
                        candidate
                    )

            else:

                watch_status = (
                    determine_watchlist_status(
                        ensemble_probability,
                        ensemble_return,
                        score,
                        self.cfg,
                    )
                )

                candidate[
                    "status"
                ] = watch_status

                if watch_status == "WATCHLIST":

                    candidate[
                        "reason"
                    ] = (
                        "Promising but below "
                        "trade threshold."
                    )

                    watch_candidates.append(
                        candidate
                    )

                else:

                    candidate[
                        "reason"
                    ] = (
                        "Below V6.3.1 "
                        "trade thresholds."
                    )

                    rejected.append(
                        candidate
                    )

        # ----------------------------------------------------
        # Save audit
        # ----------------------------------------------------

        self.save_rejected(
            rejected
        )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        trade_candidates.sort(
            key=lambda x: (
                x["score"],
                x["ensemble_probability"],
                x["ensemble_return"],
            ),
            reverse=True,
        )

        watch_candidates.sort(
            key=lambda x: (
                x["score"],
                x["ensemble_probability"],
                x["ensemble_return"],
            ),
            reverse=True,
        )

        trades = pd.DataFrame(
            trade_candidates[
                : self.cfg.top_n
            ]
        )

        watches = pd.DataFrame(
            watch_candidates[
                : self.cfg.top_n
            ]
        )

        if not trades.empty:

            self.save_candidates(
                trades,
                "TRADE"
            )

        if not watches.empty:

            self.save_candidates(
                watches,
                "WATCHLIST"
            )

        return (
            trades,
            watches,
        )

    # ========================================================
    # SAVE CANDIDATES
    # ========================================================

    def save_candidates(
        self,
        dataframe,
        category,
    ):

        if dataframe.empty:
            return

        try:

            df = dataframe.copy()

            df["category"] = (
                category
            )

            df["date"] = (
                now_ist().strftime(
                    "%Y-%m-%d %H:%M IST"
                )
            )

            if os.path.exists(
                self.cfg.candidate_history
            ):

                old = pd.read_csv(
                    self.cfg.candidate_history
                )

            else:

                old = pd.DataFrame()

            pd.concat(
                [
                    old,
                    df,
                ],
                ignore_index=True,
            ).to_csv(
                self.cfg.candidate_history,
                index=False,
            )

        except Exception as exc:

            log.warning(
                "Candidate history failed: %s",
                exc,
            )

    # ========================================================
    # SAVE REJECTED
    # ========================================================

    def save_rejected(
        self,
        rows,
    ):

        if not rows:
            return

        try:

            df = pd.DataFrame(
                rows
            )

            df["date"] = (
                now_ist().strftime(
                    "%Y-%m-%d %H:%M IST"
                )
            )

            if os.path.exists(
                self.cfg.rejected_history
            ):

                old = pd.read_csv(
                    self.cfg.rejected_history
                )

            else:

                old = pd.DataFrame()

            pd.concat(
                [
                    old,
                    df,
                ],
                ignore_index=True,
            ).to_csv(
                self.cfg.rejected_history,
                index=False,
            )

        except Exception as exc:

            log.warning(
                "Rejected history failed: %s",
                exc,
            )

    # ========================================================
    # TELEGRAM MESSAGE
    # ========================================================

    def build_message(
        self,
        regime,
        trades,
        watches,
        ipos,
    ):

        timestamp = (
            now_ist().strftime(
                "%d %b %Y, %H:%M IST"
            )
        )

        lines = [
            "*MULTI-FACTOR MARKET ALERT V6.3.1*",
            f"_{timestamp}_",
            "",
            (
                f"MARKET REGIME: "
                f"*{regime['label']}*"
            ),
            (
                f"NIFTY: "
                f"{regime['nifty']:.2f} | "
                f"SMA50: "
                f"{regime['sma50']:.2f}"
            ),
        ]

        if np.isfinite(
            regime["vix"]
        ):

            lines.append(
                f"INDIA VIX: "
                f"{regime['vix']:.2f}"
            )

        else:

            lines.append(
                "INDIA VIX: unavailable"
            )

        # ====================================================
        # TRADE SETUPS
        # ====================================================

        lines.extend(
            [
                "",
                "--- TOP SHORT-TERM "
                "TRADE SETUPS ---",
            ]
        )

        if trades.empty:

            lines.extend(
                [
                    "",
                    "*NO VALID LONG TRADE TODAY*",
                    "",
                    (
                        "The model did not find "
                        "a candidate meeting all "
                        "V6.3.1 probability, "
                        "return and risk/reward "
                        "requirements."
                    ),
                ]
            )

        else:

            for number, (
                _,
                row,
            ) in enumerate(
                trades.iterrows(),
                start=1,
            ):

                lines.extend(
                    [
                        "",
                        (
                            f"*{number}. "
                            f"{row['ticker']} "
                            f"— {row['status']}*"
                        ),

                        (
                            f"Price: "
                            f"₹{row['price']:.2f}"
                        ),

                        (
                            "P(UP) 1D / 3D / 5D: "
                            f"{row['p1'] * 100:.1f}% / "
                            f"{row['p3'] * 100:.1f}% / "
                            f"{row['p5'] * 100:.1f}%"
                        ),

                        (
                            "Expected return "
                            "1D / 3D / 5D: "
                            f"{row['pred1']:.2f}% / "
                            f"{row['pred3']:.2f}% / "
                            f"{row['pred5']:.2f}%"
                        ),

                        (
                            f"Score: "
                            f"{row['score']:.3f} | "
                            f"RSI: "
                            f"{row['rsi']:.1f} | "
                            f"ATR: "
                            f"{row['atr_pct']:.2f}%"
                        ),

                        (
                            f"Entry zone: "
                            f"₹{row['entry_low']:.2f}"
                            f" – "
                            f"₹{row['entry_high']:.2f}"
                        ),

                        (
                            f"Stop Loss: "
                            f"₹{row['stop']:.2f}"
                        ),

                        (
                            f"Target 1: "
                            f"₹{row['target1']:.2f}"
                            f" | Target 2: "
                            f"₹{row['target2']:.2f}"
                        ),

                        (
                            f"Risk/Reward: "
                            f"{row['rr1']:.2f} / "
                            f"{row['rr2']:.2f}"
                        ),

                        (
                            f"Expected holding: "
                            f"{row['holding_period']}"
                        ),

                        (
                            f"Action: "
                            f"{row['action']}"
                        ),

                        (
                            f"Suggested position: "
                            f"{int(row['shares'])} "
                            f"shares ≈ "
                            f"₹{row['position_value']:,.0f}"
                        ),

                        (
                            f"Maximum planned loss: "
                            f"₹{row['max_loss']:,.0f}"
                        ),
                    ]
                )

        # ====================================================
        # WATCHLIST
        # ====================================================

        lines.extend(
            [
                "",
                "--- BEST WATCHLIST SETUPS ---",
            ]
        )

        if watches.empty:

            lines.append(
                "None."
            )

        else:

            for number, (
                _,
                row,
            ) in enumerate(
                watches.iterrows(),
                start=1,
            ):

                lines.extend(
                    [
                        "",
                        (
                            f"{number}. "
                            f"*{row['ticker']}*"
                        ),

                        (
                            f"P(UP 3D): "
                            f"{row['p3'] * 100:.1f}%"
                        ),

                        (
                            f"Expected 3D return: "
                            f"{row['pred3']:.2f}%"
                        ),

                        (
                            f"Score: "
                            f"{row['score']:.3f}"
                        ),

                        (
                            f"Status: "
                            f"{row['status']}"
                        ),

                        (
                            f"Reason: "
                            f"{row.get('reason', '-')}"
                        ),
                    ]
                )

        # ====================================================
        # IPO
        # ====================================================

        lines.extend(
            [
                "",
                "--- IPO OPEN / UPCOMING ---",
            ]
        )

        if not ipos:

            lines.append(
                "IPO data unavailable."
            )

        else:

            for ipo in ipos[:10]:

                lines.append(
                    (
                        f"*{ipo['name']}* | "
                        f"{ipo['status']}"
                    )
                )

                lines.append(
                    (
                        f"Dates: "
                        f"{ipo.get('start', '-')}"
                        f" to "
                        f"{ipo.get('end', '-')}"
                    )
                )

                lines.append(
                    (
                        f"Price: "
                        f"{ipo.get('price', '-')}"
                    )
                )

                lines.append(
                    (
                        f"Subscription: "
                        f"{ipo.get('subscription', '-')}"
                    )
                )

        # ====================================================
        # FOOTER
        # ====================================================

        lines.extend(
            [
                "",
                (
                    "_V6.3.1 is a probabilistic "
                    "research screen. It does not "
                    "guarantee profit. Position sizing "
                    "uses configured capital and risk "
                    "limits. Verify market/IPO data "
                    "before trading._"
                ),
            ]
        )

        return "\n".join(
            lines
        )

    # ========================================================
    # SEND TELEGRAM
    # ========================================================

    def send_telegram(
        self,
        regime,
        trades,
        watches,
        ipos,
    ):

        if not self.cfg.bot_token:

            raise RuntimeError(
                "TELEGRAM_BOT_TOKEN is missing."
            )

        if not self.cfg.chat_id:

            raise RuntimeError(
                "TELEGRAM_CHAT_ID is missing."
            )

        message = self.build_message(
            regime,
            trades,
            watches,
            ipos,
        )

        url = (
            "https://api.telegram.org/"
            f"bot{self.cfg.bot_token}/sendMessage"
        )

        response = requests.post(
            url,
            json={
                "chat_id":
                    self.cfg.chat_id,

                "text":
                    message,

                "parse_mode":
                    "Markdown",
            },
            timeout=30,
        )

        response.raise_for_status()

        log.info(
            "Telegram V6.3.1 alert sent."
        )

        # ----------------------------------------------------
        # Audit
        # ----------------------------------------------------

        try:

            timestamp = (
                now_ist().strftime(
                    "%Y-%m-%d %H:%M:%S IST"
                )
            )

            exists = os.path.exists(
                self.cfg.alert_history
            )

            with open(
                self.cfg.alert_history,
                "a",
                encoding="utf-8",
            ) as file:

                if not exists:

                    file.write(
                        "timestamp,message\n"
                    )

                clean = (
                    message
                    .replace(
                        '"',
                        '""',
                    )
                    .replace(
                        "\n",
                        "\\n",
                    )
                )

                file.write(
                    f'"{timestamp}",'
                    f'"{clean}"\n'
                )

        except Exception as exc:

            log.warning(
                "Alert audit save failed: %s",
                exc,
            )


# ============================================================
# REGIME
# ============================================================

def get_regime(
    cfg,
):

    data = download_market_data(
        [
            cfg.nifty,
            cfg.vix,
        ],
        period="1y",
    )

    nifty = extract_ticker_series(
        data["Close"],
        cfg.nifty,
    )

    if nifty is None:

        raise RuntimeError(
            "Nifty data unavailable."
        )

    vix = extract_ticker_series(
        data["Close"],
        cfg.vix,
    )

    if vix is None:

        vix = pd.Series(
            dtype=float
        )

    return calculate_market_regime(
        nifty,
        vix,
    )


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def run_backtest(
    cfg,
    period="5y",
):

    log.info(
        "Starting V6.3.1 walk-forward backtest."
    )

    data = download_market_data(
        cfg.universe
        + [
            cfg.nifty,
            cfg.vix,
        ],
        period=period,
    )

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    nifty = extract_ticker_series(
        close,
        cfg.nifty,
    )

    if nifty is None:

        raise RuntimeError(
            "Nifty data unavailable."
        )

    nifty = (
        pd.to_numeric(
            nifty,
            errors="coerce",
        )
        .dropna()
    )

    vix = extract_ticker_series(
        close,
        cfg.vix,
    )

    if vix is None:

        vix = pd.Series(
            dtype=float
        )

    frames = {}

    for ticker in cfg.universe:

        stock_close = (
            extract_ticker_series(
                close,
                ticker,
            )
        )

        stock_high = (
            extract_ticker_series(
                high,
                ticker,
            )
        )

        stock_low = (
            extract_ticker_series(
                low,
                ticker,
            )
        )

        stock_volume = (
            extract_ticker_series(
                volume,
                ticker,
            )
        )

        if any(
            x is None
            for x in [
                stock_close,
                stock_high,
                stock_low,
                stock_volume,
            ]
        ):

            continue

        if (
            len(
                stock_close.dropna()
            )
            < 750
        ):

            continue

        frames[ticker] = (
            build_features(
                stock_close,
                stock_high,
                stock_low,
                stock_volume,
                nifty,
                vix,
            )
        )

    if not frames:

        return (
            pd.DataFrame(),
            {
                "trades": 0
            },
        )

    # --------------------------------------------------------
    # Common dates
    # --------------------------------------------------------

    all_dates = sorted(
        set.intersection(
            *[
                set(frame.index)
                for frame
                in frames.values()
            ]
        )
    )

    # Every 5th trading day.
    test_dates = all_dates[
        650:-6:5
    ]

    trades = []

    for current_date in test_dates:

        training_parts = {
            1: [],
            3: [],
            5: [],
        }

        for frame in frames.values():

            historical = (
                frame[
                    frame.index
                    < current_date
                ]
                .tail(
                    cfg.lookback
                )
            )

            for horizon in [
                1,
                3,
                5,
            ]:

                training_parts[
                    horizon
                ].append(
                    historical
                )

        models = {}

        for horizon in [
            1,
            3,
            5,
        ]:

            training = pd.concat(
                training_parts[
                    horizon
                ],
                ignore_index=True,
            )

            models[horizon] = (
                fit_model(
                    training,
                    horizon,
                    cfg,
                )
            )

        if models.get(3) is None:

            continue

        candidates = []

        for ticker, frame in frames.items():

            if current_date not in frame.index:
                continue

            row = frame.loc[
                [current_date]
            ]

            x = row[
                FEATURES
            ]

            if x.isna().any(
                axis=1
            ).iloc[0]:

                continue

            predictions = {}
            probabilities = {}

            for horizon in [
                1,
                3,
                5,
            ]:

                bundle = models.get(
                    horizon
                )

                if bundle is None:
                    continue

                predictions[
                    horizon
                ] = (
                    float(
                        bundle[0]
                        .predict(x)[0]
                    )
                    * 100
                )

                probabilities[
                    horizon
                ] = (
                    float(
                        bundle[1]
                        .predict_proba(x)[
                            0,
                            1,
                        ]
                    )
                )

            if 3 not in predictions:
                continue

            price = safe_float(
                row[
                    "price"
                ].iloc[0]
            )

            atr = safe_float(
                row[
                    "atr"
                ].iloc[0]
            )

            atr_pct = (
                atr
                / price
                * 100
                if price > 0
                else np.nan
            )

            if not validate_volatility(
                price,
                atr,
                atr_pct,
                cfg,
            ):

                continue

            turnover_cr = (
                price
                * safe_float(
                    row[
                        "volume"
                    ].iloc[0],
                    0,
                )
                / 1e7
            )

            if (
                price
                < cfg.min_price
                or
                turnover_cr
                < cfg.min_turnover_cr
            ):

                continue

            p1 = probabilities.get(
                1,
                probabilities[3],
            )

            p3 = probabilities[3]

            p5 = probabilities.get(
                5,
                probabilities[3],
            )

            r1 = predictions.get(
                1,
                predictions[3],
            )

            r3 = predictions[3]

            r5 = predictions.get(
                5,
                predictions[3],
            )

            ensemble_probability = (
                0.20 * p1
                + 0.55 * p3
                + 0.25 * p5
            )

            ensemble_return = (
                0.20 * r1
                + 0.55 * r3
                + 0.25 * r5
            )

            score = (
                ensemble_return
                * (
                    2
                    * ensemble_probability
                    - 1
                )
                - 0.20
                * atr_pct
            )

            trend = (
                safe_float(
                    row[
                        "dist_ema20"
                    ].iloc[0]
                ) > 0
                and
                safe_float(
                    row[
                        "dist_sma50"
                    ].iloc[0]
                ) > 0
                and
                safe_float(
                    row[
                        "ema20_slope10"
                    ].iloc[0]
                ) > 0
            )

            if trend:
                score += 0.12

            relative5 = (
                safe_float(
                    row[
                        "relative5"
                    ].iloc[0]
                )
                * 100
            )

            if relative5 > 0:
                score += min(
                    0.10,
                    relative5 * 0.02,
                )

            volume_ratio = safe_float(
                row[
                    "volume_ratio"
                ].iloc[0]
            )

            if volume_ratio > 1.15:
                score += 0.05

            rsi = safe_float(
                row[
                    "rsi14"
                ].iloc[0]
            )

            if rsi > 70:
                score -= 0.08

            if rsi > 76:
                score -= 0.08

            # ------------------------------------------------
            # Only genuine trade candidates enter backtest.
            # ------------------------------------------------

            if (
                ensemble_probability
                < cfg.trade_prob
            ):

                continue

            if (
                ensemble_return
                < cfg.trade_return
            ):

                continue

            if score < cfg.trade_score:
                continue

            candidate = {
                "price":
                    price,

                "atr":
                    atr,

                "pred3":
                    r3,

                "pred5":
                    r5,

                "p3":
                    p3,

                "p5":
                    p5,

                "extension_atr":
                    (
                        safe_float(
                            row[
                                "dist_ema20"
                            ].iloc[0]
                        )
                        * price
                        / atr
                    ),

                "rsi":
                    rsi,
            }

            plan = build_trade_plan(
                candidate,
                cfg,
            )

            if plan is None:
                continue

            if (
                plan["rr2"]
                < cfg.min_rr
            ):
                continue

            ticker_series = (
                pd.to_numeric(
                    close[
                        ticker
                    ],
                    errors="coerce",
                )
                .dropna()
            )

            try:

                position = (
                    ticker_series.index
                    .get_loc(
                        current_date
                    )
                )

            except KeyError:

                continue

            if (
                position + 3
                >= len(
                    ticker_series
                )
            ):

                continue

            entry = price

            exit_price = float(
                ticker_series.iloc[
                    position + 3
                ]
            )

            gross_return = (
                exit_price
                / entry
                - 1
            ) * 100

            transaction_cost = (
                2
                * (
                    cfg.slippage_pct
                    + cfg.brokerage_pct
                )
            )

            net_return = (
                gross_return
                - transaction_cost
            )

            candidates.append(
                {
                    "score":
                        score,

                    "ticker":
                        ticker,

                    "entry":
                        entry,

                    "exit":
                        exit_price,

                    "probability":
                        ensemble_probability
                        * 100,

                    "predicted_return":
                        ensemble_return,

                    "net_3d_pct":
                        net_return,
                }
            )

        if not candidates:
            continue

        candidates.sort(
            key=lambda x: (
                x["score"],
                x["probability"],
                x["predicted_return"],
            ),
            reverse=True,
        )

        best = candidates[0]

        trades.append(
            {
                "date":
                    current_date,

                "ticker":
                    best["ticker"].replace(
                        ".NS",
                        "",
                    ),

                "entry":
                    best["entry"],

                "exit":
                    best["exit"],

                "probability":
                    best["probability"],

                "predicted_return":
                    best["predicted_return"],

                "score":
                    best["score"],

                "net_3d_pct":
                    best["net_3d_pct"],
            }
        )

    trades_df = pd.DataFrame(
        trades
    )

    if trades_df.empty:

        return (
            trades_df,
            {
                "trades": 0
            },
        )

    wins = (
        trades_df[
            "net_3d_pct"
        ]
        > 0
    )

    equity = (
        1
        + trades_df[
            "net_3d_pct"
        ]
        / 100
    ).cumprod()

    drawdown = (
        equity
        / equity.cummax()
        - 1
    ) * 100

    summary = {
        "trades":
            int(
                len(
                    trades_df
                )
            ),

        "hit_rate_pct":
            round(
                wins.mean()
                * 100,
                1,
            ),

        "average_net_3d_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].mean(),
                3,
            ),

        "median_net_3d_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].median(),
                3,
            ),

        "best_trade_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].max(),
                2,
            ),

        "worst_trade_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].min(),
                2,
            ),

        "max_drawdown_pct":
            round(
                drawdown.min(),
                2,
            ),

        "compound_return_pct":
            round(
                (
                    equity.iloc[-1]
                    - 1
                )
                * 100,
                2,
            ),
    }

    return (
        trades_df,
        summary,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Indian Market Engine V6.3.1"
        )
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run walk-forward backtest.",
    )

    parser.add_argument(
        "--backtest-period",
        default="5y",
        help="Historical period for backtest.",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Trading capital.",
    )

    args = parser.parse_args()

    # ========================================================
    # CAPITAL
    # ========================================================

    env_capital = (
        os.getenv(
            "CAPITAL"
        )
        or "100000"
    )

    if args.capital is not None:

        capital = safe_float(
            args.capital,
            100000.0,
        )

    else:

        capital = safe_float(
            env_capital,
            100000.0,
        )

    if (
        not np.isfinite(capital)
        or capital <= 0
    ):

        log.warning(
            "Invalid CAPITAL. "
            "Using ₹100000."
        )

        capital = 100000.0

    # ========================================================
    # CONFIG
    # ========================================================

    cfg = Config(

        bot_token=os.getenv(
            "TELEGRAM_BOT_TOKEN"
        ),

        chat_id=os.getenv(
            "TELEGRAM_CHAT_ID"
        ),

        capital=capital,
    )

    # ========================================================
    # BACKTEST
    # ========================================================

    if args.backtest:

        trades, summary = (
            run_backtest(
                cfg,
                args.backtest_period,
            )
        )

        print()
        print(
            "=========================================="
        )

        print(
            "V6.3.1 WALK-FORWARD BACKTEST"
        )

        print(
            "=========================================="
        )

        for key, value in (
            summary.items()
        ):

            print(
                f"{key}: {value}"
            )

        trades.to_csv(
            cfg.backtest_file,
            index=False,
        )

        print()
        print(
            f"Saved: {cfg.backtest_file}"
        )

        return

    # ========================================================
    # LIVE ALERT
    # ========================================================

    regime = get_regime(
        cfg
    )

    log.info(
        "Market regime: %s",
        regime["label"],
    )

    engine = MarketEngineV631(
        cfg
    )

    trades, watches = (
        engine.scan()
    )

    ipos = fetch_ipo_data()

    engine.send_telegram(
        regime,
        trades,
        watches,
        ipos,
    )

    log.info(
        "V6.3.1 completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
