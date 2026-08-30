"""
Indian Market Alert Engine V6.1

Purpose
-------
Short-term Indian equity screening and Telegram alerts.

Major improvements over the previous version:
- 1 / 3 / 5-session prediction ensemble
- Probability calibration
- Relative-strength analysis
- Volume confirmation
- ATR-based stop loss
- Entry zone
- Target 1 / Target 2
- Risk/reward calculation
- Risk-budget position sizing
- Soft market-regime filter
- Nifty + India VIX context
- Larger stock universe with fallback
- Missed-opportunity tracking
- Persistent alert/candidate history
- Dynamic NSE IPO discovery
- No fabricated GMP
- Backtest mode

IMPORTANT
--------
This is a probabilistic research/paper-trading system.
It does not guarantee profit and is not investment advice.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("market_engine_v6_1")


# ============================================================
# CONSTANTS
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

HORIZONS = (1, 3, 5)

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
    "atr_pct",
    "vol20",
    "volume_ratio",
    "relative3",
    "relative5",
    "relative10",
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

    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    top_n: int = 3

    # Example only. Change through GitHub Actions environment.
    capital_inr: float = 100000.0

    # Maximum planned capital loss per position.
    risk_per_trade_pct: float = 0.50

    # Model horizons.
    horizons: tuple = HORIZONS
    primary_horizon: int = 3

    # Historical training window.
    model_lookback: int = 504
    min_training_samples: int = 1500

    # Selection thresholds.
    min_probability: float = 0.535
    min_expected_return_pct: float = 0.20
    min_score: float = 0.035

    # Liquidity filters.
    min_price: float = 50.0
    min_turnover_cr: float = 25.0

    # Risk management.
    atr_stop_multiple: float = 1.20
    entry_atr_buffer: float = 0.20
    max_entry_extension_atr: float = 1.15

    min_target1_pct: float = 0.70
    min_target2_pct: float = 1.20

    # Persistent files.
    alert_history_path: str = "alert_history_v6_1.csv"
    candidate_history_path: str = "candidate_history_v6_1.csv"
    missed_history_path: str = "missed_opportunities_v6_1.csv"

    # Fallback universe.
    fallback_universe: list = field(
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
# BASIC HELPERS
# ============================================================

def now_ist() -> datetime:
    return datetime.now(IST)


def safe_float(value, default=np.nan) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return default

    except Exception:
        return default


def clean_symbol(symbol: str) -> str:
    return str(symbol).replace(".NS", "").strip().upper()


# ============================================================
# MARKET DATA
# ============================================================

def download_market_data(
    tickers,
    period="3y",
    retries=3,
):
    """
    Download daily OHLCV data using yfinance.

    Returns:
        dict with DataFrames for Open/High/Low/Close/Volume.
    """

    tickers = list(dict.fromkeys(tickers))

    last_error = None

    for attempt in range(1, retries + 1):

        try:

            log.info(
                "Downloading market data for %d tickers...",
                len(tickers),
            )

            raw = yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )

            if raw is None or raw.empty:
                raise RuntimeError(
                    "yfinance returned empty data."
                )

            result = {}

            if isinstance(raw.columns, pd.MultiIndex):

                level0 = list(
                    raw.columns.get_level_values(0)
                )

                level1 = list(
                    raw.columns.get_level_values(1)
                )

                for field in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]:

                    if field in level0:

                        result[field] = raw[field]

                    elif field in level1:

                        result[field] = raw.xs(
                            field,
                            axis=1,
                            level=1,
                        )

            else:

                for field in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]:

                    if field in raw.columns:

                        result[field] = raw[[field]]

            if "Close" not in result:
                raise RuntimeError(
                    "Close price data unavailable."
                )

            return result

        except Exception as exc:

            last_error = exc

            log.warning(
                "Market-data attempt %d/%d failed: %s",
                attempt,
                retries,
                exc,
            )

            if attempt < retries:
                time.sleep(attempt * 2)

    raise RuntimeError(
        f"Market data download failed: {last_error}"
    )


# ============================================================
# UNIVERSE
# ============================================================

def load_current_universe(cfg: Config):
    """
    Try to retrieve the current Nifty 500 constituent list.

    Falls back to a curated liquid universe if NSE blocks
    the request.
    """

    url = (
        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty500list.csv"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/151.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*",
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20,
        )

        response.raise_for_status()

        df = pd.read_csv(
            BytesIO(response.content)
        )

        if "Symbol" in df.columns:
            symbol_column = "Symbol"
        else:
            symbol_column = df.columns[0]

        symbols = []

        for value in df[symbol_column].dropna():

            symbol = str(value).strip().upper()

            if not symbol:
                continue

            if not symbol.endswith(".NS"):
                symbol += ".NS"

            symbols.append(symbol)

        symbols = list(
            dict.fromkeys(symbols)
        )

        if len(symbols) >= 200:

            log.info(
                "Loaded %d current Nifty constituents.",
                len(symbols),
            )

            return symbols

    except Exception as exc:

        log.warning(
            "Could not retrieve current Nifty universe: %s",
            exc,
        )

    log.warning(
        "Using fallback liquid universe with %d stocks.",
        len(cfg.fallback_universe),
    )

    return cfg.fallback_universe


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def calculate_rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:

    delta = series.diff()

    gain = (
        delta.clip(lower=0)
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    loss = (
        -delta.clip(upper=0)
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    rs = gain / loss.replace(0, np.nan)

    return 100 - (
        100 / (1 + rs)
    )


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    market: pd.Series,
    vix: pd.Series,
) -> pd.DataFrame:

    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)

    market = (
        market
        .astype(float)
        .reindex(close.index)
        .ffill()
    )

    vix = (
        vix
        .astype(float)
        .reindex(close.index)
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
        market
        .rolling(50)
        .mean()
    )

    atr_value = calculate_atr(
        high,
        low,
        close,
    )

    daily_return = close.pct_change()

    features = pd.DataFrame(
        index=close.index
    )

    for period in [1, 3, 5, 10, 20]:

        features[f"ret{period}"] = (
            close.pct_change(period)
        )

    features["rsi14"] = calculate_rsi(
        close
    )

    features["dist_ema5"] = (
        close / ema5 - 1
    )

    features["dist_ema20"] = (
        close / ema20 - 1
    )

    features["dist_sma50"] = (
        close / sma50 - 1
    )

    features["ema20_slope10"] = (
        ema20.pct_change(10)
    )

    features["atr_pct"] = (
        atr_value / close
    )

    features["vol20"] = (
        daily_return
        .rolling(20)
        .std()
    )

    volume_median = (
        volume
        .rolling(20)
        .median()
    )

    features["volume_ratio"] = (
        volume /
        volume_median.replace(0, np.nan)
    )

    nifty_ret3 = market.pct_change(3)
    nifty_ret5 = market.pct_change(5)
    nifty_ret10 = market.pct_change(10)

    features["nifty_ret3"] = nifty_ret3
    features["nifty_ret10"] = nifty_ret10

    features["nifty_above_sma50"] = (
        market > nifty_sma50
    ).astype(float)

    features["nifty_sma50_slope10"] = (
        nifty_sma50.pct_change(10)
    )

    features["relative3"] = (
        features["ret3"] -
        nifty_ret3
    )

    features["relative5"] = (
        features["ret5"] -
        nifty_ret5
    )

    features["relative10"] = (
        features["ret10"] -
        nifty_ret10
    )

    features["vix_level"] = vix

    features["vix_change5"] = (
        vix.pct_change(5)
    )

    # Raw values used by trade planning.
    features["price"] = close
    features["atr"] = atr_value
    features["volume"] = volume

    # Future targets.
    for horizon in HORIZONS:

        features[f"target{horizon}"] = (
            close.shift(-horizon) /
            close - 1
        )

    return features


# ============================================================
# MODEL FITTING
# ============================================================

def fit_models(
    training: pd.DataFrame,
    horizon: int,
    cfg: Config,
):
    """
    Fit:
    1. Ridge return model
    2. Logistic direction model
    3. Isotonic probability calibrator
    """

    target_column = f"target{horizon}"

    if target_column not in training.columns:
        return None

    X = (
        training[FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    y = training[target_column]

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid]
    y = y.loc[valid]

    if len(X) < cfg.min_training_samples:
        return None

    if y.nunique() < 10:
        return None

    split = int(
        len(X) * 0.80
    )

    if split < 500:
        return None

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_holdout = X.iloc[split:]
    y_holdout = y.iloc[split:]

    if len(X_holdout) < 100:
        return None

    return_model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=8.0
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
                    C=0.25,
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    lower = y_train.quantile(0.01)
    upper = y_train.quantile(0.99)

    y_train_clipped = (
        y_train.clip(
            lower,
            upper,
        )
    )

    return_model.fit(
        X_train,
        y_train_clipped,
    )

    direction_target = (
        y_train > 0
    ).astype(int)

    if direction_target.nunique() < 2:
        return None

    direction_model.fit(
        X_train,
        direction_target,
    )

    raw_holdout_probability = (
        direction_model
        .predict_proba(
            X_holdout
        )[:, 1]
    )

    holdout_direction = (
        y_holdout > 0
    ).astype(int)
    holdout_direction = (
        holdout_direction
        .to_numpy()
    )

    calibrator = None

    if len(
        np.unique(
            holdout_direction
        )
    ) == 2:

        try:

            calibrator = (
                IsotonicRegression(
                    out_of_bounds="clip"
                )
                .fit(
                    raw_holdout_probability,
                    holdout_direction,
                )
            )

        except Exception:

            calibrator = None

    # Refit models using all historical observations.
    return_model.fit(
        X,
        y.clip(
            y.quantile(0.01),
            y.quantile(0.99),
        ),
    )

    direction_model.fit(
        X,
        (y > 0).astype(int),
    )

    return (
        return_model,
        direction_model,
        calibrator,
    )


def calibrated_probability(
    direction_model,
    calibrator,
    X: pd.DataFrame,
) -> float:

    raw_probability = float(
        direction_model
        .predict_proba(X)[0, 1]
    )

    if calibrator is None:
        return raw_probability

    try:

        calibrated = float(
            calibrator.predict(
                [raw_probability]
            )[0]
        )

        return float(
            np.clip(
                calibrated,
                0.01,
                0.99,
            )
        )

    except Exception:

        return raw_probability


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(
    market: pd.Series,
    vix: pd.Series,
) -> dict:

    sma50 = (
        market
        .rolling(50)
        .mean()
    )

    slope = (
        sma50
        .pct_change(10)
    )

    latest_nifty = safe_float(
        market.iloc[-1]
    )

    latest_sma = safe_float(
        sma50.iloc[-1]
    )

    latest_slope = safe_float(
        slope.iloc[-1]
    )

    if not vix.empty:

        latest_vix = safe_float(
            vix.dropna().iloc[-1]
        )

    else:

        latest_vix = np.nan

    score = 0

    if (
        np.isfinite(latest_nifty)
        and
        np.isfinite(latest_sma)
    ):

        if latest_nifty > latest_sma:
            score += 1
        else:
            score -= 1

    if np.isfinite(latest_slope):

        if latest_slope > 0:
            score += 1
        else:
            score -= 1

    if np.isfinite(latest_vix):

        if latest_vix >= 20:
            score -= 2

        elif latest_vix >= 18:
            score -= 1

        elif latest_vix <= 14:
            score += 1

    if score >= 2:
        label = "FAVORABLE"

    elif score <= -2:
        label = "UNFAVORABLE"

    else:
        label = "MIXED"

    return {
        "label": label,
        "score": score,
        "nifty": latest_nifty,
        "sma50": latest_sma,
        "slope_pct": latest_slope * 100,
        "vix": latest_vix,
    }


# ============================================================
# TRADE PLAN
# ============================================================

def make_trade_plan(
    row: pd.Series,
    cfg: Config,
) -> dict:

    price = safe_float(
        row["Price"]
    )

    atr_value = safe_float(
        row["ATR"]
    )

    if (
        not np.isfinite(atr_value)
        or atr_value <= 0
    ):

        atr_pct = safe_float(
            row["ATR_Pct"],
            default=1.0,
        )

        atr_value = (
            price *
            atr_pct /
            100
        )

    # Entry zone.
    entry_low = max(
        0.01,
        price -
        cfg.entry_atr_buffer *
        atr_value,
    )

    entry_high = (
        price +
        0.10 *
        atr_value
    )

    # Stop.
    stop_loss = max(
        0.01,
        price -
        cfg.atr_stop_multiple *
        atr_value,
    )

    risk_pct = (
        (price - stop_loss) /
        price *
        100
    )

    # Expected model return.
    expected_return_pct = max(
        safe_float(
            row[
                "Ensemble_Return_Pct"
            ]
        ),
        cfg.min_target2_pct,
    )

    # Target 1.
    target1_return_pct = max(
        expected_return_pct * 0.55,
        cfg.min_target1_pct,
    )

    target1 = (
        price *
        (
            1 +
            target1_return_pct /
            100
        )
    )

    # Target 2.
    target2 = (
        price *
        (
            1 +
            expected_return_pct /
            100
        )
    )

    rr1 = (
        target1_return_pct /
        max(risk_pct, 0.01)
    )

    rr2 = (
        expected_return_pct /
        max(risk_pct, 0.01)
    )

    # Position size from risk budget.
    risk_budget = (
        cfg.capital_inr *
        cfg.risk_per_trade_pct /
        100
    )

    risk_per_share = max(
        price - stop_loss,
        0.01,
    )

    shares = int(
        risk_budget /
        risk_per_share
    )

    shares = max(
        shares,
        0,
    )

    allocation = (
        shares *
        price
    )

    probability = safe_float(
        row[
            "Probability_3D_Pct"
        ]
    )

    score = safe_float(
        row["Score"]
    )

    if (
        probability >= 60
        and score >= 0.12
    ):

        tier = "HIGH-CONFIDENCE"

    elif (
        probability >= 55
        and score >= cfg.min_score
    ):

        tier = "GOOD SETUP"

    else:

        tier = "SPECULATIVE / SMALL SIZE"

    # Chasing protection.
    rsi_value = safe_float(
        row["RSI"]
    )

    extension_atr = safe_float(
        row["Extension_ATR"]
    )

    if (
        rsi_value >= 70
        or
        extension_atr >
        cfg.max_entry_extension_atr
    ):

        action = (
            "DO NOT CHASE — "
            "WAIT FOR PULLBACK"
        )

    else:

        action = "BUY ON CONFIRMATION"

    return {
        "Entry_Low": round(
            entry_low,
            2,
        ),
        "Entry_High": round(
            entry_high,
            2,
        ),
        "Stop_Loss": round(
            stop_loss,
            2,
        ),
        "Target_1": round(
            target1,
            2,
        ),
        "Target_2": round(
            target2,
            2,
        ),
        "Risk_Pct": round(
            risk_pct,
            2,
        ),
        "RR_T1": round(
            rr1,
            2,
        ),
        "RR_T2": round(
            rr2,
            2,
        ),
        "Suggested_Shares": shares,
        "Suggested_Allocation": round(
            allocation,
            2,
        ),
        "Tier": tier,
        "Action": action,
    }


# ============================================================
# ENGINE
# ============================================================

class MarketEngineV61:

    def __init__(
        self,
        cfg: Config,
    ):

        self.cfg = cfg

    # ========================================================
    # STOCK SCAN
    # ========================================================

    def stock_scan(self):

        cfg = self.cfg

        symbols = (
            load_current_universe(
                cfg
            )
        )

        tickers = list(
            dict.fromkeys(
                symbols +
                [
                    NIFTY_TICKER,
                    VIX_TICKER,
                ]
            )
        )

        data = download_market_data(
            tickers,
            period="3y",
        )

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        if NIFTY_TICKER not in close.columns:
            raise RuntimeError(
                "Nifty 50 data unavailable."
            )

        market = (
            close[NIFTY_TICKER]
            .dropna()
        )

        if VIX_TICKER in close.columns:

            vix = (
                close[VIX_TICKER]
                .dropna()
            )

        else:

            vix = pd.Series(
                dtype=float
            )

        latest_market_date = (
            market.index[-1]
        )

        feature_frames = {}

        training_parts = {
            horizon: []
            for horizon in HORIZONS
        }

        log.info(
            "Building stock feature sets..."
        )

        for ticker in symbols:

            if (
                ticker not in close.columns
                or
                ticker not in high.columns
                or
                ticker not in low.columns
                or
                ticker not in volume.columns
            ):
                continue

            stock_close = (
                close[ticker]
                .dropna()
            )

            if len(stock_close) < 650:
                continue

            stock_high = (
                high[ticker]
                .reindex(
                    stock_close.index
                )
            )

            stock_low = (
                low[ticker]
                .reindex(
                    stock_close.index
                )
            )

            stock_volume = (
                volume[ticker]
                .reindex(
                    stock_close.index
                )
            )

            features = build_features(
                stock_close,
                stock_high,
                stock_low,
                stock_volume,
                market,
                vix,
            )

            feature_frames[ticker] = (
                features
            )

            historical = (
                features[
                    features.index <
                    latest_market_date
                ]
                .tail(
                    cfg.model_lookback
                )
            )

            for horizon in HORIZONS:

                training_parts[
                    horizon
                ].append(
                    historical
                )

        # ====================================================
        # FIT MODELS
        # ====================================================

        models = {}

        for horizon in HORIZONS:

            if not training_parts[horizon]:
                continue

            combined = pd.concat(
                training_parts[horizon],
                ignore_index=True,
            )

            log.info(
                "Training %d-session model on %d rows...",
                horizon,
                len(combined),
            )

            model = fit_models(
                combined,
                horizon,
                cfg,
            )

            if model is not None:

                models[horizon] = model

        if 3 not in models:

            raise RuntimeError(
                "Could not fit the 3-session model."
            )

        # ====================================================
        # SCORE STOCKS
        # ====================================================

        rows = []

        for ticker, features in (
            feature_frames.items()
        ):

            if (
                latest_market_date
                not in features.index
            ):
                continue

            current = features.loc[
                [latest_market_date]
            ]

            if (
                current[FEATURES]
                .isna()
                .any(axis=1)
                .iloc[0]
            ):
                continue

            X = current[FEATURES]

            predictions = {}
            probabilities = {}

            for horizon in HORIZONS:

                if horizon not in models:
                    continue

                (
                    return_model,
                    direction_model,
                    calibrator,
                ) = models[horizon]

                prediction = float(
                    return_model.predict(X)[0]
                )

                probability = (
                    calibrated_probability(
                        direction_model,
                        calibrator,
                        X,
                    )
                )

                predictions[horizon] = (
                    prediction
                )

                probabilities[horizon] = (
                    probability
                )

            if 3 not in predictions:
                continue

            price = safe_float(
                current[
                    "price"
                ].iloc[0]
            )

            atr_value = safe_float(
                current[
                    "atr"
                ].iloc[0]
            )

            atr_pct = (
                atr_value /
                price *
                100
                if price > 0
                else np.nan
            )

            rsi_value = safe_float(
                current[
                    "rsi14"
                ].iloc[0]
            )

            volume_value = safe_float(
                current[
                    "volume"
                ].iloc[0]
            )

            turnover_cr = (
                price *
                volume_value /
                1e7
            )

            dist_ema20 = safe_float(
                current[
                    "dist_ema20"
                ].iloc[0]
            )

            extension_atr = (
                dist_ema20 *
                price /
                max(
                    atr_value,
                    1e-9,
                )
            )

            trend_aligned = (
                safe_float(
                    current[
                        "dist_ema20"
                    ].iloc[0]
                ) > 0

                and

                safe_float(
                    current[
                        "dist_sma50"
                    ].iloc[0]
                ) > 0

                and

                safe_float(
                    current[
                        "ema20_slope10"
                    ].iloc[0]
                ) > 0
            )

            # =================================================
            # MULTI-HORIZON ENSEMBLE
            # =================================================

            prediction_1d = predictions.get(
                1,
                predictions[3],
            )

            prediction_3d = predictions[3]

            prediction_5d = predictions.get(
                5,
                predictions[3],
            )

            probability_1d = probabilities.get(
                1,
                probabilities[3],
            )

            probability_3d = probabilities[3]

            probability_5d = probabilities.get(
                5,
                probabilities[3],
            )

            ensemble_return = (
                0.20 * prediction_1d
                +
                0.55 * prediction_3d
                +
                0.25 * prediction_5d
            )

            ensemble_probability = (
                0.20 * probability_1d
                +
                0.55 * probability_3d
                +
                0.25 * probability_5d
            )

            # =================================================
            # RISK-ADJUSTED SCORE
            # =================================================

            directional_edge = (
                2 *
                ensemble_probability
                - 1
            )

            raw_edge = (
                ensemble_return *
                directional_edge
            )

            volatility_penalty = (
                0.20 *
                (
                    atr_value /
                    max(price, 1e-9)
                )
            )

            score = (
                raw_edge
                -
                volatility_penalty
            )

            # =================================================
            # RELATIVE STRENGTH
            # =================================================

            relative5 = safe_float(
                current[
                    "relative5"
                ].iloc[0]
            )

            relative10 = safe_float(
                current[
                    "relative10"
                ].iloc[0]
            )

            score += (
                0.05 *
                np.clip(
                    relative5 * 10,
                    -1,
                    1,
                )
            )

            score += (
                0.025 *
                np.clip(
                    relative10 * 8,
                    -1,
                    1,
                )
            )

            # =================================================
            # VOLUME CONFIRMATION
            # =================================================

            volume_ratio = safe_float(
                current[
                    "volume_ratio"
                ].iloc[0]
            )

            score += (
                0.035 *
                np.clip(
                    (
                        volume_ratio -
                        1
                    ) / 1.5,
                    -1,
                    1,
                )
            )

            # =================================================
            # TREND BONUS
            # =================================================

            if trend_aligned:

                score += 0.04

            else:

                score -= 0.025

            # =================================================
            # SOFT MARKET REGIME FILTER
            # =================================================

            nifty_above_sma50 = safe_float(
                current[
                    "nifty_above_sma50"
                ].iloc[0]
            )

            if nifty_above_sma50 < 0.5:

                # Do NOT reject automatically.
                # Simply reduce confidence.
                score *= 0.82

            # =================================================
            # OVERBOUGHT PENALTY
            # =================================================

            if rsi_value > 70:

                score *= 0.75

            elif rsi_value < 42:

                score *= 0.80

            # =================================================
            # EXTENSION PENALTY
            # =================================================

            if extension_atr > 1.25:

                score *= 0.70

            expected_return_pct = (
                ensemble_return *
                100
            )

            probability_pct = (
                ensemble_probability *
                100
            )

            # =================================================
            # LIQUIDITY
            # =================================================

            liquid = (
                price >= cfg.min_price
                and
                turnover_cr >=
                cfg.min_turnover_cr
            )

            eligible = (
                liquid
                and
                expected_return_pct
                >=
                cfg.min_expected_return_pct
                and
                probability_pct
                >=
                cfg.min_probability * 100
                and
                score
                >=
                cfg.min_score
            )

            rejection_reasons = []

            if not liquid:

                rejection_reasons.append(
                    "liquidity"
                )

            if (
                expected_return_pct
                <
                cfg.min_expected_return_pct
            ):

                rejection_reasons.append(
                    "expected return"
                )

            if (
                probability_pct
                <
                cfg.min_probability * 100
            ):

                rejection_reasons.append(
                    "probability"
                )

            if score < cfg.min_score:

                rejection_reasons.append(
                    "risk-adjusted score"
                )

            if rsi_value > 72:

                rejection_reasons.append(
                    "overbought"
                )

            if extension_atr > 1.50:

                rejection_reasons.append(
                    "extended"
                )

            rejection = (
                ", ".join(
                    rejection_reasons
                )
                if rejection_reasons
                else ""
            )

            rows.append(
                {
                    "Ticker":
                        clean_symbol(ticker),

                    "Price":
                        round(
                            price,
                            2,
                        ),

                    "Predicted_1D_Return_Pct":
                        round(
                            prediction_1d * 100,
                            2,
                        ),

                    "Predicted_3D_Return_Pct":
                        round(
                            prediction_3d * 100,
                            2,
                        ),

                    "Predicted_5D_Return_Pct":
                        round(
                            prediction_5d * 100,
                            2,
                        ),

                    "Probability_1D_Pct":
                        round(
                            probability_1d * 100,
                            1,
                        ),

                    "Probability_3D_Pct":
                        round(
                            probability_3d * 100,
                            1,
                        ),

                    "Probability_5D_Pct":
                        round(
                            probability_5d * 100,
                            1,
                        ),

                    "Ensemble_Return_Pct":
                        round(
                            expected_return_pct,
                            2,
                        ),

                    "Ensemble_Probability_Pct":
                        round(
                            probability_pct,
                            1,
                        ),

                    "Score":
                        round(
                            score,
                            4,
                        ),

                    "RSI":
                        round(
                            rsi_value,
                            1,
                        ),

                    "ATR":
                        round(
                            atr_value,
                            2,
                        ),

                    "ATR_Pct":
                        round(
                            atr_pct,
                            2,
                        ),

                    "Extension_ATR":
                        round(
                            extension_atr,
                            2,
                        ),

                    "Volume_Ratio":
                        round(
                            volume_ratio,
                            2,
                        ),

                    "Relative_5D_Pct":
                        round(
                            relative5 * 100,
                            2,
                        ),

                    "Relative_10D_Pct":
                        round(
                            relative10 * 100,
                            2,
                        ),

                    "Trend_Aligned":
                        trend_aligned,

                    "Turnover_Cr":
                        round(
                            turnover_cr,
                            1,
                        ),

                    "Eligible":
                        eligible,

                    "Rejection":
                        rejection,
                }
            )

        all_candidates = pd.DataFrame(
            rows
        )

        if all_candidates.empty:

            no_trade = pd.DataFrame(
                [
                    {
                        "Ticker": "--",
                        "No_Trade": True,
                        "Tier": "NO TRADE",
                        "Action": (
                            "No candidates "
                            "available."
                        ),
                    }
                ]
            )

            return (
                no_trade,
                pd.DataFrame(),
            )

        all_candidates = (
            all_candidates
            .sort_values(
                [
                    "Score",
                    "Ensemble_Probability_Pct",
                    "Ensemble_Return_Pct",
                ],
                ascending=False,
            )
            .reset_index(drop=True)
        )

        # ====================================================
        # SAVE TOP CANDIDATES
        # ====================================================

        self.append_history(
            all_candidates.head(30),
            cfg.candidate_history_path,
        )

        # ====================================================
        # MISSED OPPORTUNITIES
        # ====================================================

        missed = (
            all_candidates[
                ~all_candidates[
                    "Eligible"
                ]
            ]
            .head(10)
            .copy()
        )

        self.append_history(
            missed,
            cfg.missed_history_path,
        )

        # ====================================================
        # FINAL PICKS
        # ====================================================

        picks = (
            all_candidates[
                all_candidates[
                    "Eligible"
                ]
            ]
            .head(cfg.top_n)
            .copy()
            .reset_index(drop=True)
        )

        if picks.empty:

            no_trade = pd.DataFrame(
                [
                    {
                        "Ticker": "--",
                        "No_Trade": True,
                        "Tier": "NO TRADE",
                        "Action": (
                            "No setup cleared "
                            "the minimum "
                            "risk-adjusted edge."
                        ),
                    }
                ]
            )

            return (
                no_trade,
                missed,
            )

        # Add trade plans.
        for index, row in picks.iterrows():

            plan = make_trade_plan(
                row,
                cfg,
            )

            for key, value in plan.items():

                picks.loc[
                    index,
                    key
                ] = value

        picks["No_Trade"] = False

        return (
            picks,
            missed,
        )

    # ========================================================
    # PERSISTENT HISTORY
    # ========================================================

    def append_history(
        self,
        dataframe: pd.DataFrame,
        path: str,
    ):

        if (
            dataframe is None
            or dataframe.empty
        ):
            return

        try:

            output = dataframe.copy()

            output.insert(
                0,
                "alert_timestamp",
                now_ist().strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                ),
            )

            if os.path.exists(path):

                old = pd.read_csv(
                    path
                )

            else:

                old = pd.DataFrame()

            combined = pd.concat(
                [
                    old,
                    output,
                ],
                ignore_index=True,
            )

            combined.to_csv(
                path,
                index=False,
            )

        except Exception as exc:

            log.warning(
                "Could not write history file %s: %s",
                path,
                exc,
            )

    # ========================================================
    # ALERT HISTORY
    # ========================================================

    def save_alert_history(
        self,
        picks: pd.DataFrame,
        regime: dict,
    ):

        if (
            picks is None
            or picks.empty
        ):
            return

        rows = []

        for _, row in picks.iterrows():

            rows.append(
                {
                    "timestamp":
                        now_ist().strftime(
                            "%Y-%m-%d %H:%M:%S %Z"
                        ),

                    "ticker":
                        row.get(
                            "Ticker"
                        ),

                    "price":
                        row.get(
                            "Price"
                        ),

                    "score":
                        row.get(
                            "Score"
                        ),

                    "predicted_1d_pct":
                        row.get(
                            "Predicted_1D_Return_Pct"
                        ),

                    "predicted_3d_pct":
                        row.get(
                            "Predicted_3D_Return_Pct"
                        ),

                    "predicted_5d_pct":
                        row.get(
                            "Predicted_5D_Return_Pct"
                        ),

                    "probability_3d_pct":
                        row.get(
                            "Probability_3D_Pct"
                        ),

                    "entry_low":
                        row.get(
                            "Entry_Low"
                        ),

                    "entry_high":
                        row.get(
                            "Entry_High"
                        ),

                    "stop_loss":
                        row.get(
                            "Stop_Loss"
                        ),

                    "target1":
                        row.get(
                            "Target_1"
                        ),

                    "target2":
                        row.get(
                            "Target_2"
                        ),

                    "risk_pct":
                        row.get(
                            "Risk_Pct"
                        ),

                    "rr_target1":
                        row.get(
                            "RR_T1"
                        ),

                    "rr_target2":
                        row.get(
                            "RR_T2"
                        ),

                    "suggested_shares":
                        row.get(
                            "Suggested_Shares"
                        ),

                    "suggested_allocation":
                        row.get(
                            "Suggested_Allocation"
                        ),

                    "tier":
                        row.get(
                            "Tier"
                        ),

                    "action":
                        row.get(
                            "Action"
                        ),

                    "market_regime":
                        regime.get(
                            "label"
                        ),

                    "nifty":
                        regime.get(
                            "nifty"
                        ),

                    "vix":
                        regime.get(
                            "vix"
                        ),
                }
            )

        self.append_history(
            pd.DataFrame(rows),
            self.cfg.alert_history_path,
        )


# ============================================================
# TELEGRAM
# ============================================================

def format_stock_section(
    picks: pd.DataFrame,
    missed: pd.DataFrame,
) -> list[str]:

    lines = [
        "--- TOP SHORT-TERM OPPORTUNITIES ---"
    ]

    if (
        picks is None
        or picks.empty
    ):

        lines.append(
            "NO TRADE — no candidate cleared "
            "the risk-adjusted filters."
        )

        return lines

    for number, (_, row) in enumerate(
        picks.iterrows(),
        start=1,
    ):

        if bool(
            row.get(
                "No_Trade",
                False,
            )
        ):

            lines.append(
                "NO TRADE — no suitable setup."
            )

            continue

        lines.extend(
            [
                "",
                (
                    f"{number}. "
                    f"{row['Ticker']} "
                    f"| Rs.{row['Price']}"
                ),

                (
                    "   MODEL 1D / 3D / 5D: "
                    f"{row['Predicted_1D_Return_Pct']}% / "
                    f"{row['Predicted_3D_Return_Pct']}% / "
                    f"{row['Predicted_5D_Return_Pct']}%"
                ),

                (
                    "   P(UP) 1D / 3D / 5D: "
                    f"{row['Probability_1D_Pct']}% / "
                    f"{row['Probability_3D_Pct']}% / "
                    f"{row['Probability_5D_Pct']}%"
                ),

                (
                    f"   SCORE: {row['Score']} | "
                    f"RSI: {row['RSI']} | "
                    f"ATR: {row['ATR_Pct']}% | "
                    f"VOL: {row['Volume_Ratio']}x"
                ),

                (
                    f"   RELATIVE STRENGTH 5D: "
                    f"{row['Relative_5D_Pct']}%"
                ),

                (
                    f"   ENTRY: "
                    f"Rs.{row['Entry_Low']} – "
                    f"Rs.{row['Entry_High']}"
                ),

                (
                    f"   TARGET 1: "
                    f"Rs.{row['Target_1']} | "
                    f"TARGET 2: "
                    f"Rs.{row['Target_2']}"
                ),

                (
                    f"   STOP-LOSS: "
                    f"Rs.{row['Stop_Loss']}"
                ),

                (
                    f"   R:R: "
                    f"{row['RR_T1']} / "
                    f"{row['RR_T2']}"
                ),

                (
                    "   HOLDING WINDOW: "
                    "1–5 trading sessions"
                ),

                (
                    f"   POSITION: "
                    f"{int(row['Suggested_Shares'])} shares "
                    f"(~Rs.{row['Suggested_Allocation']})"
                ),

                (
                    f"   {row['Tier']} | "
                    f"{row['Action']}"
                ),
            ]
        )

    if (
        missed is not None
        and not missed.empty
    ):

        lines.extend(
            [
                "",
                "--- MISSED-OPPORTUNITY WATCHLIST ---",
                (
                    "Strong rejected candidates are "
                    "logged so future versions can "
                    "learn whether filters are too strict."
                ),
            ]
        )

        for _, row in missed.head(3).iterrows():

            lines.append(
                (
                    f"{row['Ticker']}: "
                    f"Score {row['Score']} | "
                    f"P3 {row['Probability_3D_Pct']}% | "
                    f"E3 {row['Ensemble_Return_Pct']}% | "
                    f"Rejected: {row['Rejection']}"
                )
            )

    return lines


def send_telegram(
    cfg: Config,
    regime: dict,
    picks: pd.DataFrame,
    missed: pd.DataFrame,
    ipos: list[dict],
):

    if (
        not cfg.telegram_bot_token
        or not cfg.telegram_chat_id
    ):

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN or "
            "TELEGRAM_CHAT_ID is missing."
        )

    # FIXED:
    # Calculate the timestamp separately instead of
    # nesting quotation marks inside an f-string.
    timestamp = now_ist().strftime(
        "%d %b %Y, %H:%M IST"
    )

    nifty_value = safe_float(
        regime.get(
            "nifty"
        )
    )

    sma_value = safe_float(
        regime.get(
            "sma50"
        )
    )

    vix_value = safe_float(
        regime.get(
            "vix"
        )
    )

    lines = [

        "*MULTI-FACTOR MARKET ALERT V6.1*",

        f"_{timestamp}_",

        "",

        (
            f"MARKET REGIME: "
            f"{regime.get('label', 'UNKNOWN')}"
        ),

        (
            f"Nifty 50: "
            f"{nifty_value:.2f} | "
            f"50D average: "
            f"{sma_value:.2f}"
        ),

        (
            f"India VIX: "
            f"{vix_value:.2f}"
        ),

        "",
    ]

    lines.extend(
        format_stock_section(
            picks,
            missed,
        )
    )

    lines.extend(
        [
            "",
            "--- IPO OPEN / UPCOMING ---",
        ]
    )

    if ipos:

        for ipo in ipos:

            name = str(
                ipo.get(
                    "name",
                    "Unknown IPO",
                )
            )

            status = str(
                ipo.get(
                    "status",
                    "UNKNOWN",
                )
            )

            start = ipo.get(
                "start",
                "?",
            )

            end = ipo.get(
                "end",
                "?",
            )

            lines.append(
                (
                    f"{name} "
                    f"[{status}] | "
                    f"{start} → {end}"
                )
            )

            price = ipo.get(
                "price"
            )

            if price:

                lines.append(
                    f"   PRICE: {price}"
                )

            subscription = ipo.get(
                "subscription"
            )

            if subscription:

                lines.append(
                    (
                        f"   SUBSCRIPTION: "
                        f"{subscription}"
                    )
                )

            gmp = ipo.get(
                "gmp"
            )

            if gmp is not None:

                gmp_pct = ipo.get(
                    "gmp_pct",
                    "?",
                )

                lines.append(
                    (
                        f"   GMP: Rs.{gmp} "
                        f"({gmp_pct}%)"
                    )
                )

            else:

                lines.append(
                    "   GMP: NOT VERIFIED"
                )

            lines.append(
                (
                    "   ACTION: "
                    f"{ipo.get("
                    "'action', "
                    "'REVIEW BEFORE APPLYING'"
                    ")}"
                )
            )

    else:

        lines.extend(
            [
                (
                    "No current/upcoming IPO "
                    "records were retrieved "
                    "from NSE."
                ),
                (
                    "This does NOT prove that "
                    "no IPO is available."
                ),
            ]
        )

    lines.extend(
        [
            "",
            (
                "_Probabilistic research screen. "
                "No profit is guaranteed. "
                "Position sizing uses a fixed "
                "risk budget. IPO GMP is unofficial "
                "unless independently verified._"
            ),
        ]
    )

    telegram_url = (
        "https://api.telegram.org/"
        f"bot{cfg.telegram_bot_token}/sendMessage"
    )

    response = requests.post(
        telegram_url,
        json={
            "chat_id":
                cfg.telegram_chat_id,

            "text":
                "\n".join(lines),

            "parse_mode":
                "Markdown",
        },
        timeout=20,
    )

    response.raise_for_status()

    payload = response.json()

    if not payload.get(
        "ok",
        False,
    ):

        raise RuntimeError(
            "Telegram API returned an unsuccessful response: "
            + str(payload)
        )

    log.info(
        "Telegram alert sent successfully."
    )


# ============================================================
# NSE IPO DISCOVERY
# ============================================================

def create_nse_session():

    session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/151.0 Safari/537.36"
        ),
        "Accept": (
            "application/json,"
            "text/plain,"
            "*/*"
        ),
        "Referer":
            "https://www.nseindia.com/",
    }

    try:

        session.get(
            "https://www.nseindia.com/",
            headers=headers,
            timeout=15,
        )

    except Exception as exc:

        log.warning(
            "NSE landing page request failed: %s",
            exc,
        )

    return session, headers


def fetch_current_ipos() -> list[dict]:
    """
    Retrieve current/upcoming IPO information.

    IMPORTANT:
    NSE does not publish unofficial GMP here.
    GMP is therefore left blank unless another verified
    source is deliberately integrated later.
    """

    endpoints = [
        (
            "https://www.nseindia.com/"
            "api/ipo-current-issue"
        ),
        (
            "https://www.nseindia.com/"
            "api/all-upcoming-issues?category=ipo"
        ),
    ]

    try:

        session, headers = (
            create_nse_session()
        )

        records = []

        for endpoint in endpoints:

            try:

                response = session.get(
                    endpoint,
                    headers=headers,
                    timeout=20,
                )

                response.raise_for_status()

                payload = response.json()

            except Exception as exc:

                log.warning(
                    "NSE IPO endpoint failed: %s",
                    exc,
                )

                continue

            if isinstance(
                payload,
                dict,
            ):

                data = payload.get(
                    "data",
                    [],
                )

                if isinstance(
                    data,
                    dict,
                ):

                    data = data.get(
                        "data",
                        [],
                    )

            elif isinstance(
                payload,
                list,
            ):

                data = payload

            else:

                data = []

            if not isinstance(
                data,
                list,
            ):

                continue

            for item in data:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                name = (
                    item.get(
                        "companyName"
                    )
                    or
                    item.get(
                        "companyname"
                    )
                    or
                    item.get(
                        "name"
                    )
                    or
                    item.get(
                        "company"
                    )
                    or
                    item.get(
                        "symbol"
                    )
                    or
                    "Unknown IPO"
                )

                start = (
                    item.get(
                        "issueStartDate"
                    )
                    or
                    item.get(
                        "biddingStartDate"
                    )
                    or
                    item.get(
                        "startDate"
                    )
                )

                end = (
                    item.get(
                        "issueEndDate"
                    )
                    or
                    item.get(
                        "biddingEndDate"
                    )
                    or
                    item.get(
                        "endDate"
                    )
                )

                status = (
                    item.get(
                        "status"
                    )
                    or
                    (
                        "OPEN"
                        if "current"
                        in endpoint
                        else "UPCOMING"
                    )
                )

                price = (
                    item.get(
                        "priceBand"
                    )
                    or
                    item.get(
                        "price"
                    )
                    or
                    item.get(
                        "issuePrice"
                    )
                )

                subscription = (
                    item.get(
                        "subscription"
                    )
                    or
                    item.get(
                        "subscriptionTimes"
                    )
                    or
                    item.get(
                        "subscriptionStatus"
                    )
                )

                records.append(
                    {
                        "name":
                            str(
                                name
                            ).strip(),

                        "start":
                            start,

                        "end":
                            end,

                        "status":
                            str(
                                status
                            ).upper(),

                        "price":
                            price,

                        "subscription":
                            subscription,

                        "gmp":
                            None,

                        "gmp_pct":
                            None,

                        "action":
                            (
                                "REVIEW ISSUE PRICE + "
                                "SUBSCRIPTION; VERIFY GMP "
                                "SEPARATELY"
                            ),
                    }
                )

        # Deduplicate by normalized company name.
        unique = []
        seen = set()

        for record in records:

            key = "".join(
                character
                for character in
                record["name"].upper()
                if character.isalnum()
            )

            if not key:
                continue

            if key in seen:
                continue

            seen.add(key)
            unique.append(record)

        log.info(
            "Retrieved %d current/upcoming IPO records.",
            len(unique),
        )

        return unique[:20]

    except Exception as exc:

        log.warning(
            "IPO discovery failed: %s",
            exc,
        )

        return []


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    cfg: Config,
    period: str = "5y",
):
    """
    Walk-forward 3-session backtest.

    This is deliberately conservative:
    models are trained only on information available
    before each test date.
    """

    symbols = (
        load_current_universe(
            cfg
        )
    )

    tickers = list(
        dict.fromkeys(
            symbols +
            [
                NIFTY_TICKER,
                VIX_TICKER,
            ]
        )
    )

    data = download_market_data(
        tickers,
        period=period,
    )

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    if NIFTY_TICKER not in close.columns:

        raise RuntimeError(
            "Nifty data unavailable for backtest."
        )

    market = (
        close[NIFTY_TICKER]
        .dropna()
    )

    if VIX_TICKER in close.columns:

        vix = (
            close[VIX_TICKER]
            .dropna()
        )

    else:

        vix = pd.Series(
            dtype=float
        )

    feature_frames = {}

    for ticker in symbols:

        required = (
            ticker in close.columns
            and
            ticker in high.columns
            and
            ticker in low.columns
            and
            ticker in volume.columns
        )

        if not required:
            continue

        stock_close = (
            close[ticker]
            .dropna()
        )

        if len(stock_close) < 750:
            continue

        feature_frames[ticker] = (
            build_features(
                stock_close,
                high[ticker].reindex(
                    stock_close.index
                ),
                low[ticker].reindex(
                    stock_close.index
                ),
                volume[ticker].reindex(
                    stock_close.index
                ),
                market,
                vix,
            )
        )

    if not feature_frames:

        return (
            pd.DataFrame(),
            {
                "trades": 0
            },
        )

    common_dates = sorted(
        set.intersection(
            *[
                set(frame.index)
                for frame in
                feature_frames.values()
            ]
        )
    )

    if len(common_dates) < 700:

        return (
            pd.DataFrame(),
            {
                "trades": 0,
                "message":
                    "Insufficient common history.",
            },
        )

    start_index = max(
        550,
        len(common_dates) // 3,
    )

    test_dates = common_dates[
        start_index:-6
    ]

    trades = []

    for test_date in test_dates:

        training_parts = {
            horizon: []
            for horizon in HORIZONS
        }

        for frame in feature_frames.values():

            historical = (
                frame[
                    frame.index <
                    test_date
                ]
                .tail(
                    cfg.model_lookback
                )
            )

            for horizon in HORIZONS:

                training_parts[
                    horizon
                ].append(
                    historical
                )

        models = {}

        for horizon in HORIZONS:

            if not training_parts[horizon]:
                continue

            combined = pd.concat(
                training_parts[horizon],
                ignore_index=True,
            )

            model = fit_models(
                combined,
                horizon,
                cfg,
            )

            if model is not None:

                models[horizon] = model

        if 3 not in models:
            continue

        candidates = []

        for ticker, frame in (
            feature_frames.items()
        ):

            if test_date not in frame.index:
                continue

            current = frame.loc[
                [test_date]
            ]

            if (
                current[FEATURES]
                .isna()
                .any(axis=1)
                .iloc[0]
            ):
                continue

            X = current[FEATURES]

            predictions = {}
            probabilities = {}

            for horizon in HORIZONS:

                if horizon not in models:
                    continue

                (
                    return_model,
                    direction_model,
                    calibrator,
                ) = models[horizon]

                predictions[horizon] = float(
                    return_model.predict(X)[0]
                )

                probabilities[horizon] = (
                    calibrated_probability(
                        direction_model,
                        calibrator,
                        X,
                    )
                )

            if 3 not in predictions:
                continue

            price = safe_float(
                current[
                    "price"
                ].iloc[0]
            )

            atr_value = safe_float(
                current[
                    "atr"
                ].iloc[0]
            )

            turnover_cr = (
                price *
                safe_float(
                    current[
                        "volume"
                    ].iloc[0]
                )
                /
                1e7
            )

            if (
                price < cfg.min_price
                or
                turnover_cr <
                cfg.min_turnover_cr
            ):
                continue

            ensemble_return = (
                0.20 *
                predictions.get(
                    1,
                    predictions[3],
                )
                +
                0.55 *
                predictions[3]
                +
                0.25 *
                predictions.get(
                    5,
                    predictions[3],
                )
            )

            ensemble_probability = (
                0.20 *
                probabilities.get(
                    1,
                    probabilities[3],
                )
                +
                0.55 *
                probabilities[3]
                +
                0.25 *
                probabilities.get(
                    5,
                    probabilities[3],
                )
            )

            score = (
                ensemble_return *
                (
                    2 *
                    ensemble_probability
                    - 1
                )
                -
                0.20 *
                (
                    atr_value /
                    max(
                        price,
                        1e-9,
                    )
                )
                +
                0.05 *
                np.clip(
                    safe_float(
                        current[
                            "relative5"
                        ].iloc[0]
                    ) * 10,
                    -1,
                    1,
                )
            )

            if (
                ensemble_return * 100
                <
                cfg.min_expected_return_pct
            ):
                continue

            if (
                ensemble_probability
                <
                cfg.min_probability
            ):
                continue

            if score < cfg.min_score:
                continue

            candidates.append(
                (
                    score,
                    ticker,
                    price,
                    ensemble_probability,
                    ensemble_return,
                )
            )

        if not candidates:
            continue

        candidates.sort(
            reverse=True
        )

        (
            score,
            ticker,
            entry,
            probability,
            expected,
        ) = candidates[0]

        price_series = (
            close[ticker]
            .dropna()
        )

        if test_date not in price_series.index:
            continue

        location = (
            price_series.index
            .get_loc(
                test_date
            )
        )

        if (
            location + 3
            >= len(price_series)
        ):
            continue

        exit_price = safe_float(
            price_series.iloc[
                location + 3
            ]
        )

        actual_return = (
            exit_price /
            entry -
            1
        ) * 100

        trades.append(
            {
                "date":
                    test_date,

                "ticker":
                    clean_symbol(ticker),

                "entry":
                    entry,

                "predicted_3d_pct":
                    expected * 100,

                "probability_3d_pct":
                    probability * 100,

                "actual_3d_pct":
                    actual_return,
            }
        )

    result = pd.DataFrame(
        trades
    )

    if result.empty:

        return (
            result,
            {
                "trades": 0
            },
        )

    summary = {

        "trades":
            int(
                len(result)
            ),

        "hit_rate_pct":
            round(
                (
                    result[
                        "actual_3d_pct"
                    ] > 0
                ).mean()
                * 100,
                1,
            ),

        "average_3d_return_pct":
            round(
                result[
                    "actual_3d_pct"
                ].mean(),
                2,
            ),

        "median_3d_return_pct":
            round(
                result[
                    "actual_3d_pct"
                ].median(),
                2,
            ),

        "best_trade_pct":
            round(
                result[
                    "actual_3d_pct"
                ].max(),
                2,
            ),

        "worst_trade_pct":
            round(
                result[
                    "actual_3d_pct"
                ].min(),
                2,
            ),

        "sum_of_trade_returns_pct":
            round(
                result[
                    "actual_3d_pct"
                ].sum(),
                2,
            ),
    }

    return (
        result,
        summary,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Indian Market Alert Engine V6.1"
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
        help="yfinance period for backtest.",
    )

    args = parser.parse_args()

    cfg = Config(

        telegram_bot_token=os.getenv(
            "TELEGRAM_BOT_TOKEN"
        ),

        telegram_chat_id=os.getenv(
            "TELEGRAM_CHAT_ID"
        ),

        top_n=int(
            os.getenv(
                "TOP_N",
                "3",
            )
        ),

        capital_inr=float(
            os.getenv(
                "CAPITAL_INR",
                "100000",
            )
        ),

        risk_per_trade_pct=float(
            os.getenv(
                "RISK_PER_TRADE_PCT",
                "0.50",
            )
        ),
    )

    # ========================================================
    # BACKTEST MODE
    # ========================================================

    if args.backtest:

        log.info(
            "Starting V6.1 walk-forward backtest..."
        )

        trades, summary = (
            run_backtest(
                cfg,
                args.backtest_period,
            )
        )

        print(
            json.dumps(
                summary,
                indent=2,
                default=str,
            )
        )

        if (
            trades is not None
            and not trades.empty
        ):

            trades.to_csv(
                "backtest_v6_1_trades.csv",
                index=False,
            )

            log.info(
                "Backtest trades saved to "
                "backtest_v6_1_trades.csv"
            )

        return

    # ========================================================
    # LIVE ALERT MODE
    # ========================================================

    log.info(
        "Starting Market Alert Engine V6.1..."
    )

    engine = MarketEngineV61(
        cfg
    )

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------

    regime_data = download_market_data(
        [
            NIFTY_TICKER,
            VIX_TICKER,
        ],
        period="1y",
    )

    market = (
        regime_data["Close"]
        [NIFTY_TICKER]
        .dropna()
    )

    if VIX_TICKER in (
        regime_data["Close"].columns
    ):

        vix = (
            regime_data["Close"]
            [VIX_TICKER]
            .dropna()
        )

    else:

        vix = pd.Series(
            dtype=float
        )

    regime = calculate_market_regime(
        market,
        vix,
    )

    log.info(
        "Market regime: %s | Nifty %.2f | VIX %.2f",
        regime["label"],
        regime["nifty"],
        regime["vix"],
    )

    # --------------------------------------------------------
    # Stock scan
    # --------------------------------------------------------

    picks, missed = (
        engine.stock_scan()
    )

    # --------------------------------------------------------
    # IPO discovery
    # --------------------------------------------------------

    ipos = fetch_current_ipos()

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    engine.save_alert_history(
        picks,
        regime,
    )

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    send_telegram(
        cfg,
        regime,
        picks,
        missed,
        ipos,
    )

    log.info(
        "Market Alert V6.1 completed successfully."
    )


if __name__ == "__main__":
    main()
