"""
Indian Market Alert Engine V6.1

Purpose:
- Short-term Indian equity opportunity screening
- 1 / 3 / 5-session model ensemble
- Risk-adjusted ranking
- Entry / stop / targets / position sizing
- Dynamic NSE IPO discovery
- Persistent audit history
- Missed-opportunity tracking

IMPORTANT:
This is a probabilistic research/paper-trading system.
It does NOT guarantee profit and is not investment advice.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

log = logging.getLogger("market_engine_v6_1")


@dataclass
class Config:

    # Telegram
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None

    # Number of final stock picks
    top_n: int = 3

    # Capital used ONLY for position-size calculation
    capital_inr: float = 100000.0

    # Maximum capital risk per trade
    risk_per_trade_pct: float = 0.50

    # Prediction horizons
    horizons: tuple = (1, 3, 5)
    primary_horizon: int = 3

    # Training
    model_lookback: int = 504
    min_training_samples: int = 2500

    # Opportunity filters
    min_probability: float = 0.535
    min_expected_return_pct: float = 0.20
    min_score: float = 0.035

    # Liquidity
    min_price: float = 50.0
    min_turnover_cr: float = 25.0

    # Risk management
    atr_stop_multiple: float = 1.20
    entry_atr_buffer: float = 0.20
    max_entry_extension_atr: float = 1.15

    target1_fraction: float = 0.55
    min_t1_pct: float = 0.70
    min_t2_pct: float = 1.20

    # Audit files
    history_path: str = "alert_history_v6_1.csv"
    candidate_history_path: str = "candidate_history_v6_1.csv"
    missed_history_path: str = "missed_opportunities_v6_1.csv"

    # Indices
    nifty: str = "^NSEI"
    vix: str = "^INDIAVIX"

    # Fallback universe
    fallback_universe: list = field(default_factory=lambda: [

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
    ])


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
# HELPERS
# ============================================================

def now_ist():
    return datetime.now(IST)


def download(tickers, period="3y", retries=3):

    tickers = list(dict.fromkeys(tickers))
    last_error = None

    for attempt in range(1, retries + 1):

        try:

            raw = yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )

            if raw.empty:
                raise RuntimeError("Empty yfinance response")

            if isinstance(raw.columns, pd.MultiIndex):

                level0 = set(raw.columns.get_level_values(0))

                fields = {
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                }

                if fields & level0:

                    return {
                        f: raw[f]
                        for f in fields
                        if f in level0
                    }

                return {
                    f: raw.xs(
                        f,
                        axis=1,
                        level=1
                    )
                    for f in fields
                    if f in raw.columns.get_level_values(1)
                }

            return {
                c: raw[[c]]
                for c in raw.columns
                if c in {
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                }
            }

        except Exception as exc:

            last_error = exc

            log.warning(
                "Market data attempt %d/%d failed: %s",
                attempt,
                retries,
                exc,
            )

            if attempt < retries:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Market-data download failed: {last_error}"
    )


# ============================================================
# CURRENT NIFTY UNIVERSE
# ============================================================

def load_universe(cfg):

    url = (
        "https://www.niftyindices.com/"
        "IndexConstituent/ind_nifty500list.csv"
    )

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        df = pd.read_csv(
            pd.io.common.BytesIO(
                response.content
            )
        )

        column = (
            "Symbol"
            if "Symbol" in df.columns
            else df.columns[0]
        )

        symbols = [
            str(x).strip().upper() + ".NS"
            for x in df[column].dropna()
        ]

        symbols = [
            x for x in symbols
            if x != "NAN.NS"
        ]

        if len(symbols) >= 200:

            log.info(
                "Loaded %d current Nifty constituents",
                len(symbols),
            )

            return symbols

    except Exception as exc:

        log.warning(
            "Could not download current Nifty 500 list: %s",
            exc,
        )

    return cfg.fallback_universe


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def rsi(series, period=14):

    delta = series.diff()

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

    rs = gain / loss.replace(0, np.nan)

    return 100 - 100 / (1 + rs)


def atr(high, low, close, period=14):

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(
    close,
    high,
    low,
    volume,
    market,
    vix,
):

    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)

    market = market.reindex(
        close.index
    ).ffill()

    vix = vix.reindex(
        close.index
    ).ffill()

    ema5 = close.ewm(
        span=5,
        adjust=False
    ).mean()

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    sma50 = close.rolling(50).mean()

    market_sma50 = market.rolling(50).mean()

    atr_value = atr(
        high,
        low,
        close
    )

    daily_return = close.pct_change()

    f = pd.DataFrame(
        index=close.index
    )

    for n in (1, 3, 5, 10, 20):

        f[f"ret{n}"] = close.pct_change(n)

    f["rsi14"] = rsi(close)

    f["dist_ema5"] = (
        close / ema5 - 1
    )

    f["dist_ema20"] = (
        close / ema20 - 1
    )

    f["dist_sma50"] = (
        close / sma50 - 1
    )

    f["ema20_slope10"] = (
        ema20.pct_change(10)
    )

    f["atr_pct"] = (
        atr_value / close
    )

    f["vol20"] = (
        daily_return.rolling(20).std()
    )

    f["volume_ratio"] = (
        volume /
        volume.rolling(20).median()
    )

    f["nifty_ret3"] = (
        market.pct_change(3)
    )

    f["nifty_ret10"] = (
        market.pct_change(10)
    )

    f["nifty_above_sma50"] = (
        market > market_sma50
    ).astype(float)

    f["nifty_sma50_slope10"] = (
        market_sma50.pct_change(10)
    )

    f["relative3"] = (
        f["ret3"] -
        f["nifty_ret3"]
    )

    f["relative5"] = (
        f["ret5"] -
        market.pct_change(5)
    )

    f["relative10"] = (
        f["ret10"] -
        f["nifty_ret10"]
    )

    f["vix_level"] = vix

    f["vix_change5"] = (
        vix.pct_change(5)
    )

    f["price"] = close
    f["atr"] = atr_value
    f["volume"] = volume

    for horizon in (1, 3, 5):

        f[f"target{horizon}"] = (
            close.shift(-horizon) /
            close - 1
        )

    return f


# ============================================================
# MODEL
# ============================================================

def fit_models(
    training,
    horizon,
    cfg,
):

    target = f"target{horizon}"

    X = training[FEATURES].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    y = training[target]

    valid = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = X.loc[valid]
    y = y.loc[valid]

    if (
        len(X) < cfg.min_training_samples
        or y.nunique() < 10
    ):
        return None

    split = int(len(X) * 0.80)

    if (
        split < 1000
        or len(X) - split < 200
    ):
        return None

    X_train = X.iloc[:split]
    y_train = y.iloc[:split]

    X_holdout = X.iloc[split:]
    y_holdout = y.iloc[split:]

    return_model = Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge(alpha=8.0)
            ),
        ]
    )

    direction_model = Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),
            (
                "logit",
                LogisticRegression(
                    C=0.25,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    clipped_y = y_train.clip(
        y_train.quantile(0.01),
        y_train.quantile(0.99),
    )

    return_model.fit(
        X_train,
        clipped_y,
    )

    direction_model.fit(
        X_train,
        (y_train > 0).astype(int),
    )

    raw_prob = (
        direction_model
        .predict_proba(X_holdout)[:, 1]
    )

    holdout_y = (
        y_holdout > 0
    ).astype(int).to_numpy()

    calibrator = None

    if len(np.unique(holdout_y)) == 2:

        calibrator = IsotonicRegression(
            out_of_bounds="clip"
        ).fit(
            raw_prob,
            holdout_y,
        )

    # Refit on all available historical data.
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
    X,
):

    raw = float(
        direction_model
        .predict_proba(X)[0, 1]
    )

    if calibrator is None:
        return raw

    return float(
        calibrator.predict([raw])[0]
    )


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime(
    market,
    vix,
):

    sma50 = market.rolling(50).mean()

    slope = sma50.pct_change(10)

    latest = float(
        market.iloc[-1]
    )

    sma = float(
        sma50.iloc[-1]
    )

    slope_value = float(
        slope.iloc[-1]
    )

    if not vix.dropna().empty:
        current_vix = float(
            vix.dropna().iloc[-1]
        )
    else:
        current_vix = np.nan

    score = 0

    score += (
        1 if latest > sma
        else -1
    )

    score += (
        1 if slope_value > 0
        else -1
    )

    if np.isfinite(current_vix):

        if current_vix > 18:
            score -= 1

        elif current_vix < 14:
            score += 1

    if score >= 2:
        label = "FAVORABLE"

    elif score <= -2:
        label = "UNFAVORABLE"

    else:
        label = "MIXED"

    return {
        "label": label,
        "nifty": latest,
        "sma50": sma,
        "slope": slope_value * 100,
        "vix": current_vix,
        "score": score,
    }


# ============================================================
# TRADE PLAN
# ============================================================

def make_trade_plan(
    row,
    cfg,
):

    price = float(row["Price"])

    atr_value = float(row["ATR"])

    if (
        not np.isfinite(atr_value)
        or atr_value <= 0
    ):

        atr_value = (
            price *
            max(
                float(row["ATR_Pct"]) / 100,
                0.01,
            )
        )

    entry_low = max(
        0.01,
        price -
        cfg.entry_atr_buffer *
        atr_value,
    )

    entry_high = (
        price +
        0.10 * atr_value
    )

    stop = max(
        0.01,
        price -
        cfg.atr_stop_multiple *
        atr_value,
    )

    expected_return = max(
        float(row["Ensemble_Return_Pct"]) / 100,
        cfg.min_t2_pct / 100,
    )

    target1_return = max(
        expected_return *
        cfg.target1_fraction,
        cfg.min_t1_pct / 100,
    )

    target1 = (
        price *
        (1 + target1_return)
    )

    target2 = (
        price *
        (1 + expected_return)
    )

    risk_pct = (
        (price - stop) /
        price *
        100
    )

    rr1 = (
        ((target1 - price) /
         price * 100)
        /
        max(risk_pct, 0.01)
    )

    rr2 = (
        ((target2 - price) /
         price * 100)
        /
        max(risk_pct, 0.01)
    )

    # Risk-budget position sizing.
    risk_budget = (
        cfg.capital_inr *
        cfg.risk_per_trade_pct /
        100
    )

    risk_per_share = max(
        price - stop,
        0.01,
    )

    shares = max(
        0,
        int(
            risk_budget /
            risk_per_share
        ),
    )

    allocation = shares * price

    probability = float(
        row["Probability_3D_Pct"]
    )

    score = float(
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

    action = "BUY ON CONFIRMATION"

    if (
        float(row["RSI"]) > 70
        or float(row["Extension_ATR"])
        > cfg.max_entry_extension_atr
    ):

        action = (
            "DO NOT CHASE — "
            "WAIT FOR PULLBACK"
        )

    return {

        "Entry_Low": round(
            entry_low, 2
        ),

        "Entry_High": round(
            entry_high, 2
        ),

        "Stop_Loss": round(
            stop, 2
        ),

        "Target_1": round(
            target1, 2
        ),

        "Target_2": round(
            target2, 2
        ),

        "Risk_Pct": round(
            risk_pct, 2
        ),

        "RR_T1": round(
            rr1, 2
        ),

        "RR_T2": round(
            rr2, 2
        ),

        "Suggested_Shares": shares,

        "Suggested_Allocation": round(
            allocation, 2
        ),

        "Tier": tier,

        "Action": action,
    }


# ============================================================
# ENGINE
# ============================================================

class MarketEngineV61:

    def __init__(self, cfg):

        self.cfg = cfg

    # --------------------------------------------------------
    # STOCK SCAN
    # --------------------------------------------------------

    def stock_scan(self):

        cfg = self.cfg

        symbols = load_universe(cfg)

        tickers = list(
            dict.fromkeys(
                symbols +
                [
                    cfg.nifty,
                    cfg.vix,
                ]
            )
        )

        data = download(
            tickers,
            period="3y",
        )

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        market = (
            close[cfg.nifty]
            .dropna()
        )

        if cfg.vix in close:
            vix = (
                close[cfg.vix]
                .dropna()
            )
        else:
            vix = pd.Series(
                dtype=float
            )

        latest_date = (
            market.index[-1]
        )

        frames = {}

        training_parts = {
            h: []
            for h in cfg.horizons
        }

        # Build feature sets.
        for ticker in symbols:

            if not all(
                ticker in x
                for x in (
                    close,
                    high,
                    low,
                    volume,
                )
            ):
                continue

            if (
                len(
                    close[ticker]
                    .dropna()
                )
                < 650
            ):
                continue

            f = build_features(
                close[ticker],
                high[ticker],
                low[ticker],
                volume[ticker],
                market,
                vix,
            )

            frames[ticker] = f

            historical = (
                f[
                    f.index <
                    latest_date
                ]
                .tail(
                    cfg.model_lookback
                )
            )

            for h in cfg.horizons:
                training_parts[h].append(
                    historical
                )

        # Fit one model per horizon.
        models = {}

        for h in cfg.horizons:

            if not training_parts[h]:
                continue

            training = pd.concat(
                training_parts[h],
                ignore_index=True,
            )

            models[h] = fit_models(
                training,
                h,
                cfg,
            )

        if not models.get(3):

            raise RuntimeError(
                "Could not fit the 3-session model."
            )

        rows = []

        for ticker, f in frames.items():

            if latest_date not in f.index:
                continue

            current = f.loc[
                [latest_date]
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

            for h in cfg.horizons:

                if not models.get(h):
                    continue

                (
                    return_model,
                    direction_model,
                    calibrator,
                ) = models[h]

                predictions[h] = float(
                    return_model
                    .predict(X)[0]
                )

                probabilities[h] = (
                    calibrated_probability(
                        direction_model,
                        calibrator,
                        X,
                    )
                )

            if 3 not in predictions:
                continue

            price = float(
                current["price"].iloc[0]
            )

            atr_value = float(
                current["atr"].iloc[0]
            )

            atr_pct = (
                atr_value /
                price *
                100
            )

            rsi_value = float(
                current["rsi14"].iloc[0]
            )

            turnover_cr = (
                price *
                float(
                    current[
                        "volume"
                    ].iloc[0]
                )
                /
                1e7
            )

            extension_atr = (
                float(
                    current[
                        "dist_ema20"
                    ].iloc[0]
                )
                *
                price
                /
                max(
                    atr_value,
                    1e-9,
                )
            )

            trend_aligned = (

                float(
                    current[
                        "dist_ema20"
                    ].iloc[0]
                ) > 0

                and

                float(
                    current[
                        "dist_sma50"
                    ].iloc[0]
                ) > 0

                and

                float(
                    current[
                        "ema20_slope10"
                    ].iloc[0]
                ) > 0
            )

            # ------------------------------------------------
            # Multi-horizon ensemble
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Risk-adjusted expected edge
            # ------------------------------------------------

            raw_edge = (
                ensemble_return *
                (
                    2 *
                    ensemble_probability
                    - 1
                )
            )

            uncertainty = (
                0.20 *
                (
                    atr_value /
                    price
                )
            )

            score = (
                raw_edge -
                uncertainty
            )

            # ------------------------------------------------
            # Soft market-regime effects
            # ------------------------------------------------

            # IMPORTANT:
            # We DO NOT reject the stock simply
            # because Nifty is mixed/bearish.

            nifty_above = float(
                current[
                    "nifty_above_sma50"
                ].iloc[0]
            )

            if not nifty_above:
                score *= 0.82

            if rsi_value > 70:
                score *= 0.75

            elif rsi_value < 42:
                score *= 0.80

            if extension_atr > 1.25:
                score *= 0.70

            # ------------------------------------------------
            # Relative strength + volume
            # ------------------------------------------------

            score += (
                0.05 *
                np.clip(
                    float(
                        current[
                            "relative5"
                        ].iloc[0]
                    ) * 10,
                    -1,
                    1,
                )
            )

            score += (
                0.035 *
                np.clip(
                    (
                        float(
                            current[
                                "volume_ratio"
                            ].iloc[0]
                        )
                        - 1
                    ) / 1.5,
                    -1,
                    1,
                )
            )

            score += (
                0.04
                if trend_aligned
                else -0.025
            )

            expected_return_pct = (
                ensemble_return * 100
            )

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
                >= cfg.min_expected_return_pct
                and
                ensemble_probability
                >= cfg.min_probability
                and
                score >= cfg.min_score
            )

            rejection = ""

            if not eligible:

                reasons = []

                if not liquid:
                    reasons.append(
                        "liquidity"
                    )

                if (
                    expected_return_pct
                    <
                    cfg.min_expected_return_pct
                ):
                    reasons.append(
                        "expected return"
                    )

                if (
                    ensemble_probability
                    <
                    cfg.min_probability
                ):
                    reasons.append(
                        "probability"
                    )

                if (
                    score <
                    cfg.min_score
                ):
                    reasons.append(
                        "risk-adjusted score"
                    )

                if rsi_value > 72:
                    reasons.append(
                        "overbought"
                    )

                if extension_atr > 1.5:
                    reasons.append(
                        "extended"
                    )

                rejection = (
                    ", ".join(reasons)
                    or
                    "other filter"
                )

            rows.append(
                {
                    "Ticker":
                        ticker.replace(
                            ".NS",
                            "",
                        ),

                    "Price":
                        round(
                            price,
                            2,
                        ),

                    "Predicted_1D_Return_Pct":
                        round(
                            predictions.get(
                                1,
                                np.nan,
                            ) * 100,
                            2,
                        ),

                    "Predicted_3D_Return_Pct":
                        round(
                            predictions[3]
                            * 100,
                            2,
                        ),

                    "Predicted_5D_Return_Pct":
                        round(
                            predictions.get(
                                5,
                                np.nan,
                            ) * 100,
                            2,
                        ),

                    "Probability_1D_Pct":
                        round(
                            probabilities.get(
                                1,
                                np.nan,
                            ) * 100,
                            1,
                        ),

                    "Probability_3D_Pct":
                        round(
                            probabilities[3]
                            * 100,
                            1,
                        ),

                    "Probability_5D_Pct":
                        round(
                            probabilities.get(
                                5,
                                np.nan,
                            ) * 100,
                            1,
                        ),

                    "Ensemble_Return_Pct":
                        round(
                            expected_return_pct,
                            2,
                        ),

                    "Ensemble_Probability_Pct":
                        round(
                            ensemble_probability
                            * 100,
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
                            float(
                                current[
                                    "volume_ratio"
                                ].iloc[0]
                            ),
                            2,
                        ),

                    "Relative_5D_Pct":
                        round(
                            float(
                                current[
                                    "relative5"
                                ].iloc[0]
                            ) * 100,
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

        df = pd.DataFrame(rows)

        if df.empty:

            return (
                pd.DataFrame(
                    [
                        {
                            "Ticker":
                                "--",
                            "No_Trade":
                                True,
                            "Tier":
                                "NO TRADE",
                            "Action":
                                "No candidates available.",
                        }
                    ]
                ),
                pd.DataFrame(),
            )

        df = df.sort_values(
            [
                "Score",
                "Ensemble_Probability_Pct",
                "Ensemble_Return_Pct",
            ],
            ascending=False,
        ).reset_index(drop=True)

        # --------------------------------------------
        # Important:
        # Keep rejected candidates.
        # This allows future filter calibration.
        # --------------------------------------------

        missed = (
            df[
                ~df["Eligible"]
            ]
            .head(10)
            .copy()
        )

        # --------------------------------------------
        # Actual picks
        # --------------------------------------------

        picks = (
            df[
                df["Eligible"]
            ]
            .head(cfg.top_n)
            .copy()
            .reset_index(drop=True)
        )

        if picks.empty:

            picks = pd.DataFrame(
                [
                    {
                        "Ticker":
                            "--",
                        "No_Trade":
                            True,
                        "Tier":
                            "NO TRADE",
                        "Action":
                            (
                                "No setup cleared "
                                "the minimum "
                                "risk-adjusted edge."
                            ),
                    }
                ]
            )

        else:

            for i, row in picks.iterrows():

                plan = make_trade_plan(
                    row,
                    cfg,
                )

                for key, value in plan.items():

                    picks.loc[
                        i,
                        key
                    ] = value

            picks["No_Trade"] = False

        # Persist candidate information.
        self.log_dataframe(
            df.head(20),
            cfg.candidate_history_path,
        )

        self.log_dataframe(
            missed,
            cfg.missed_history_path,
        )

        return picks, missed

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    def log_dataframe(
        self,
        dataframe,
        path,
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
                "date",
                now_ist().strftime(
                    "%Y-%m-%d"
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
                "Could not write %s: %s",
                path,
                exc,
            )

    # --------------------------------------------------------
    # ALERT HISTORY
    # --------------------------------------------------------

    def log_alert(
        self,
        picks,
        regime,
    ):

        rows = []

        for _, row in picks.iterrows():

            rows.append(
                {
                    "date":
                        now_ist().strftime(
                            "%Y-%m-%d"
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

                    "predicted_3d":
                        row.get(
                            "Predicted_3D_Return_Pct"
                        ),

                    "probability_3d":
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

                    "stop":
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

                    "shares":
                        row.get(
                            "Suggested_Shares"
                        ),

                    "allocation":
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

                    "regime":
                        regime[
                            "label"
                        ],
                }
            )

        if rows:

            self.log_dataframe(
                pd.DataFrame(rows),
                self.cfg.history_path,
            )

    # --------------------------------------------------------
    # TELEGRAM STOCK MESSAGE
    # --------------------------------------------------------

    def format_stock_section(
        self,
        picks,
        missed,
    ):

        lines = [
            "--- TOP SHORT-TERM OPPORTUNITIES ---"
        ]

        for i, row in picks.iterrows():

            if row.get(
                "No_Trade"
            ):

                lines.append(
                    "NO TRADE — no setup cleared "
                    "the minimum risk-adjusted edge."
                )

                continue

            lines.extend(
                [
                    "",
                    (
                        f"{i+1}. "
                        f"{row['Ticker']} "
                        f"| Rs.{row['Price']}"
                    ),

                    (
                        "   MODEL 1D/3D/5D: "
                        f"{row['Predicted_1D_Return_Pct']}% / "
                        f"{row['Predicted_3D_Return_Pct']}% / "
                        f"{row['Predicted_5D_Return_Pct']}%"
                    ),

                    (
                        "   P(UP): "
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
                        f"   RELATIVE 5D: "
                        f"{row['Relative_5D_Pct']}%"
                    ),

                    (
                        f"   ENTRY: "
                        f"Rs.{row['Entry_Low']}–"
                        f"{row['Entry_High']}"
                    ),

                    (
                        f"   TARGET 1: "
                        f"Rs.{row['Target_1']} | "
                        f"TARGET 2: "
                        f"Rs.{row['Target_2']}"
                    ),

                    (
                        f"   STOP: "
                        f"Rs.{row['Stop_Loss']}"
                    ),

                    (
                        f"   R:R: "
                        f"{row['RR_T1']} / "
                        f"{row['RR_T2']}"
                    ),

                    (
                        f"   HOLD: "
                        f"1–5 trading sessions"
                    ),

                    (
                        f"   POSITION: "
                        f"{row['Suggested_Shares']} shares "
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
                        "These were strong rejected "
                        "candidates. They are logged "
                        "to test whether filters are "
                        "too strict."
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

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    def send_telegram(
        self,
        regime,
        picks,
        missed,
        ipos,
    ):

        if (
            not self.cfg.bot_token
            or not self.cfg.chat_id
        ):

            raise ValueError(
                "TELEGRAM_BOT_TOKEN / "
                "TELEGRAM_CHAT_ID missing."
            )

        lines = [

            "*MULTI-FACTOR MARKET ALERT V6.1*",

            (
                f"_{now_ist().strftime("
                "'%d %b %Y, %H:%M IST'"
                ")}_"
            ),

            "",

            (
                f"MARKET REGIME: "
                f"{regime['label']}"
            ),

            (
                f"Nifty: "
                f"{regime['nifty']:.2f} | "
                f"50D avg: "
                f"{regime['sma50']:.2f} | "
                f"VIX: "
                f"{regime['vix']:.2f}"
            ),

            "",
        ]

        lines.extend(
            self.format_stock_section(
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

                lines.append(
                    (
                        f"{ipo['name']} "
                        f"[{ipo['status']}] | "
                        f"{ipo.get('start', '?')} → "
                        f"{ipo.get('end', '?')}"
                    )
                )

                if ipo.get("price"):
                    lines.append(
                        f"   PRICE: {ipo['price']}"
                    )

                if ipo.get(
                    "subscription"
                ):

                    lines.append(
                        (
                            "   SUBSCRIPTION: "
                            f"{ipo['subscription']}"
                        )
                    )

                if ipo.get(
                    "gmp"
                ) is not None:

                    lines.append(
                        (
                            f"   GMP: Rs."
                            f"{ipo['gmp']} "
                            f"({ipo.get('gmp_pct', '?')}%)"
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

            lines.append(
                "No current/upcoming IPO data "
                "was retrieved from NSE."
            )

            lines.append(
                "Do NOT interpret this as proof "
                "that no IPO exists."
            )

        lines.extend(
            [
                "",
                (
                    "_Probabilistic research screen. "
                    "No profit is guaranteed. "
                    "Position sizing uses a fixed "
                    "risk budget. GMP is unofficial "
                    "unless independently verified._"
                ),
            ]
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
                    "\n".join(lines),
                "parse_mode":
                    "Markdown",
            },
            timeout=15,
        )

        response.raise_for_status()

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    def get_regime(self):

        data = download(
            [
                self.cfg.nifty,
                self.cfg.vix,
            ],
            period="1y",
        )

        market = (
            data["Close"]
            [self.cfg.nifty]
            .dropna()
        )

        if self.cfg.vix in data["Close"]:

            vix = (
                data["Close"]
                [self.cfg.vix]
                .dropna()
            )

        else:

            vix = pd.Series(
                dtype=float
            )

        return market_regime(
            market,
            vix,
        )


# ============================================================
# NSE IPO DISCOVERY
# ============================================================

def nse_session():

    session = requests.Session()

    headers = {

        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/151 Safari/537.36",

        "Accept":
            "application/json,text/plain,*/*",

        "Referer":
            "https://www.nseindia.com/",
    }

    session.get(
        "https://www.nseindia.com/",
        headers=headers,
        timeout=10,
    )

    return session, headers


def fetch_current_ipos():

    """
    Uses NSE's current/upcoming IPO API endpoints.

    IMPORTANT:
    NSE does not provide unofficial GMP in this data.
    Therefore GMP is never fabricated.
    """

    endpoints = [

        "https://www.nseindia.com/"
        "api/ipo-current-issue",

        "https://www.nseindia.com/"
        "api/all-upcoming-issues?category=ipo",
    ]

    try:

        session, headers = nse_session()

        records = []

        for endpoint in endpoints:

            response = session.get(
                endpoint,
                headers=headers,
                timeout=15,
            )

            response.raise_for_status()

            payload = response.json()

            if isinstance(
                payload,
                dict
            ):

                data = payload.get(
                    "data",
                    []
                )

                if isinstance(
                    data,
                    dict
                ):

                    data = data.get(
                        "data",
                        []
                    )

            elif isinstance(
                payload,
                list
            ):

                data = payload

            else:

                data = []

            for item in data:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                name = (
                    item.get(
                        "companyName"
                    )
                    or
                    item.get("name")
                    or
                    item.get("company")
                    or
                    item.get("symbol")
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
                        else
                        "UPCOMING"
                    )
                )

                price = (
                    item.get(
                        "priceBand"
                    )
                    or
                    item.get(
                        "issuePrice"
                    )
                    or
                    item.get(
                        "price"
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

                        "action":
                            (
                                "REVIEW ISSUE PRICE "
                                "+ SUBSCRIPTION; "
                                "VERIFY GMP SEPARATELY"
                            ),
                    }
                )

        # Deduplicate.
        seen = set()
        output = []

        for record in records:

            key = re.sub(
                r"[^A-Z0-9]",
                "",
                record["name"].upper(),
            )

            if (
                key
                and key not in seen
            ):

                seen.add(key)
                output.append(record)

        return output[:15]

    except Exception as exc:

        log.warning(
            "NSE IPO retrieval failed: %s",
            exc,
        )

        return []


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    cfg,
    period="5y",
):

    symbols = load_universe(
        cfg
    )

    data = download(
        list(
            dict.fromkeys(
                symbols +
                [
                    cfg.nifty,
                    cfg.vix,
                ]
            )
        ),
        period=period,
    )

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    market = (
        close[cfg.nifty]
        .dropna()
    )

    if cfg.vix in close:

        vix = (
            close[cfg.vix]
            .dropna()
        )

    else:

        vix = pd.Series(
            dtype=float
        )

    frames = {}

    for ticker in symbols:

        if not all(
            ticker in x
            for x in (
                close,
                high,
                low,
                volume,
            )
        ):
            continue

        if (
            len(
                close[ticker]
                .dropna()
            )
            < 750
        ):
            continue

        frames[ticker] = build_features(
            close[ticker],
            high[ticker],
            low[ticker],
            volume[ticker],
            market,
            vix,
        )

    if not frames:

        return (
            pd.DataFrame(),
            {"trades": 0},
        )

    common_dates = sorted(
        set.intersection(
            *[
                set(f.index)
                for f in frames.values()
            ]
        )
    )

    trades = []

    test_dates = common_dates[
        max(
            550,
            len(common_dates) // 3,
        ):
        -6
    ]

    for test_date in test_dates:

        parts = {
            h: []
            for h in cfg.horizons
        }

        for f in frames.values():

            historical = (
                f[
                    f.index <
                    test_date
                ]
                .tail(
                    cfg.model_lookback
                )
            )

            for h in cfg.horizons:

                parts[h].append(
                    historical
                )

        models = {}

        for h in cfg.horizons:

            if parts[h]:

                models[h] = fit_models(
                    pd.concat(
                        parts[h],
                        ignore_index=True,
                    ),
                    h,
                    cfg,
                )

        if not models.get(3):
            continue

        candidates = []

        for ticker, f in frames.items():

            if test_date not in f.index:
                continue

            current = f.loc[
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

            preds = {}
            probs = {}

            for h in cfg.horizons:

                if not models.get(h):
                    continue

                (
                    rm,
                    dm,
                    cal,
                ) = models[h]

                preds[h] = float(
                    rm.predict(X)[0]
                )

                probs[h] = (
                    calibrated_probability(
                        dm,
                        cal,
                        X,
                    )
                )

            if 3 not in preds:
                continue

            price = float(
                current[
                    "price"
                ].iloc[0]
            )

            atr_value = float(
                current[
                    "atr"
                ].iloc[0]
            )

            turnover = (
                price *
                float(
                    current[
                        "volume"
                    ].iloc[0]
                )
                /
                1e7
            )

            if (
                price <
                cfg.min_price
                or
                turnover <
                cfg.min_turnover_cr
            ):
                continue

            ensemble_return = (

                0.20 *
                preds.get(
                    1,
                    preds[3],
                )

                +

                0.55 *
                preds[3]

                +

                0.25 *
                preds.get(
                    5,
                    preds[3],
                )
            )

            ensemble_probability = (

                0.20 *
                probs.get(
                    1,
                    probs[3],
                )

                +

                0.55 *
                probs[3]

                +

                0.25 *
                probs.get(
                    5,
                    probs[3],
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
                    price
                )

                +

                0.05 *
                np.clip(
                    float(
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
                >=
                cfg.min_expected_return_pct

                and

                ensemble_probability
                >=
                cfg.min_probability

                and

                score
                >=
                cfg.min_score
            ):

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

        series = (
            close[ticker]
            .dropna()
        )

        index = (
            series.index
            .get_loc(
                test_date
            )
        )

        if (
            index + 3
            >= len(series)
        ):
            continue

        exit_price = float(
            series.iloc[
                index + 3
            ]
        )

        trades.append(
            {
                "date":
                    test_date,

                "ticker":
                    ticker,

                "entry":
                    entry,

                "predicted_3d_pct":
                    expected * 100,

                "probability_pct":
                    probability * 100,

                "actual_3d_pct":
                    (
                        exit_price /
                        entry -
                        1
                    ) * 100,
            }
        )

    result = pd.DataFrame(
        trades
    )

    if result.empty:

        return (
            result,
            {"trades": 0},
        )

    summary = {

        "trades":
            len(result),

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

        "simple_sum_of_returns_pct":
            round(
                result[
                    "actual_3d_pct"
                ].sum(),
                2,
            ),
    }

    return result, summary


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--backtest",
        action="store_true",
    )

    parser.add_argument(
        "--backtest-period",
        default="5y",
    )

    args = parser.parse_args()

    cfg = Config(

        bot_token=os.getenv(
            "TELEGRAM_BOT_TOKEN"
        ),

        chat_id=os.getenv(
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

    if args.backtest:

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

        trades.to_csv(
            "backtest_v6_1_trades.csv",
            index=False,
        )

        return

    engine = MarketEngineV61(
        cfg
    )

    regime = (
        engine.get_regime()
    )

    picks, missed = (
        engine.stock_scan()
    )

    ipos = fetch_current_ipos()

    engine.log_alert(
        picks,
        regime,
    )

    engine.send_telegram(
        regime,
        picks,
        missed,
        ipos,
    )


if __name__ == "__main__":
    main()
