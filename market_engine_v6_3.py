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
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# MARKET ENGINE V6.3
# ============================================================
#
# Purpose:
#   Probabilistic short-term Indian equity screening.
#
# Outputs:
#   HIGH-CONFIDENCE
#   TRADEABLE
#   WATCHLIST
#
# Horizons:
#   1 trading day
#   3 trading days
#   5 trading days
#
# IMPORTANT:
#   This is a research/screening system.
#   It does NOT guarantee profit.
# ============================================================


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("market_engine_v6_3")

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

    min_train: int = 800

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    min_price: float = 50.0

    min_turnover_cr: float = 20.0

    # --------------------------------------------------------
    # V6.3 candidate tiers
    # --------------------------------------------------------

    high_prob: float = 0.60
    trade_prob: float = 0.55
    watch_prob: float = 0.52

    high_return: float = 0.80
    trade_return: float = 0.50
    watch_return: float = 0.25

    high_score: float = 0.30
    trade_score: float = 0.12
    watch_score: float = 0.00

    # --------------------------------------------------------
    # Technical risk controls
    # --------------------------------------------------------

    max_rsi_entry: float = 76.0

    chase_atr: float = 1.35

    min_rr: float = 1.40

    atr_stop: float = 1.25

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
    # Output files
    # --------------------------------------------------------

    alert_history: str = (
        "alert_history_v6_3.csv"
    )

    candidate_history: str = (
        "candidate_history_v6_3.csv"
    )

    rejected_history: str = (
        "rejected_candidates_v6_3.csv"
    )

    backtest_file: str = (
        "backtest_v6_3.csv"
    )

    # --------------------------------------------------------
    # Market indexes
    # --------------------------------------------------------

    nifty: str = "^NSEI"

    vix: str = "^INDIAVIX"

    # --------------------------------------------------------
    # Liquid Indian universe
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
# GENERAL HELPERS
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
# DOWNLOAD MARKET DATA
# ============================================================

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

            output = {}

            if isinstance(
                raw.columns,
                pd.MultiIndex,
            ):

                for field in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]:

                    if field in (
                        raw.columns
                        .get_level_values(0)
                    ):

                        output[field] = raw[field]

                    elif field in (
                        raw.columns
                        .get_level_values(1)
                    ):

                        output[field] = raw.xs(
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

                        output[field] = raw[
                            [field]
                        ]

            if "Close" not in output:

                raise RuntimeError(
                    "Close-price data unavailable."
                )

            return output

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
    series,
    period=14,
):

    delta = series.diff()

    gains = (
        delta
        .clip(lower=0)
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )

    losses = (
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
        gains
        / losses.replace(
            0,
            np.nan,
        )
    )

    return (
        100
        - 100 / (1 + rs)
    )


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    high,
    low,
    close,
    period=14,
):

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (
                high
                - previous_close
            ).abs(),
            (
                low
                - previous_close
            ).abs(),
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
    close,
    high,
    low,
    volume,
    nifty,
    vix,
):

    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)

    nifty = (
        nifty
        .reindex(close.index)
        .ffill()
    )

    vix = (
        vix
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

    frame["rsi14"] = calculate_rsi(
        close
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
        .rolling(20)
        .median()
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    frame["range_pct"] = (
        (high - low)
        / close
    )

    frame["close_location"] = (
        (close - low)
        / (
            high - low
        ).replace(
            0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Market-relative momentum
    # --------------------------------------------------------

    nifty_ret3 = nifty.pct_change(3)
    nifty_ret5 = nifty.pct_change(5)
    nifty_ret10 = nifty.pct_change(10)
    nifty_ret20 = nifty.pct_change(20)

    frame["relative3"] = (
        frame["ret3"]
        - nifty_ret3
    )

    frame["relative5"] = (
        frame["ret5"]
        - nifty_ret5
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
    # Nifty regime
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
        nifty > nifty_sma50
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
    # Targets
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
        training[FEATURES]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    y = training[target]

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

    cut = int(
        len(x) * 0.80
    )

    if cut < 500:

        return None

    x_train = x.iloc[:cut]
    y_train = y.iloc[:cut]

    x_cal = x.iloc[cut:]
    y_cal = y.iloc[cut:]

    # --------------------------------------------------------
    # Return model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Direction model
    # --------------------------------------------------------

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

    lower = y_train.quantile(
        0.01
    )

    upper = y_train.quantile(
        0.99
    )

    return_model.fit(
        x_train,
        y_train.clip(
            lower,
            upper,
        ),
    )

    direction_model.fit(
        x_train,
        (
            y_train > 0
        ).astype(int),
    )

    # --------------------------------------------------------
    # Probability calibration
    # --------------------------------------------------------

    calibration_model = None

    try:

        raw_probability = (
            direction_model
            .predict_proba(
                x_cal
            )[:, 1]
        )

        actual = (
            (y_cal > 0)
            .astype(int)
            .to_numpy()
        )

        if (
            len(raw_probability) >= 100
            and len(
                np.unique(actual)
            ) == 2
        ):

            calibration_model = (
                IsotonicRegression(
                    out_of_bounds="clip"
                )
            )

            calibration_model.fit(
                raw_probability,
                actual,
            )

    except Exception as exc:

        log.warning(
            "Probability calibration skipped for %dD: %s",
            horizon,
            exc,
        )

    # --------------------------------------------------------
    # Refit on complete historical sample
    # --------------------------------------------------------

    return_model.fit(
        x,
        y.clip(
            y.quantile(0.01),
            y.quantile(0.99),
        ),
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
        calibration_model,
    )


# ============================================================
# CALIBRATED PROBABILITY
# ============================================================

def get_probability(
    model_bundle,
    x,
):

    (
        _return_model,
        direction_model,
        calibration_model,
    ) = model_bundle

    raw_probability = float(
        direction_model
        .predict_proba(x)[0, 1]
    )

    if calibration_model is None:

        return raw_probability

    try:

        return float(
            calibration_model
            .predict(
                [raw_probability]
            )[0]
        )

    except Exception:

        return raw_probability


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(
    nifty,
    vix,
):

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

    if not vix.empty:

        vix_value = float(
            vix.dropna().iloc[-1]
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
# CANDIDATE TIER
# ============================================================

def determine_tier(
    probability,
    predicted_return,
    score,
    cfg,
):

    if (
        probability >= cfg.high_prob
        and predicted_return >= cfg.high_return
        and score >= cfg.high_score
    ):

        return "HIGH-CONFIDENCE"

    if (
        probability >= cfg.trade_prob
        and predicted_return >= cfg.trade_return
        and score >= cfg.trade_score
    ):

        return "TRADEABLE"

    if (
        probability >= cfg.watch_prob
        and predicted_return >= cfg.watch_return
        and score >= cfg.watch_score
    ):

        return "WATCHLIST"

    return "REJECTED"


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

    atr_value = max(
        float(row["atr"]),
        price
        * max(
            float(
                row["atr_pct"]
            ),
            0.01,
        ),
    )

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    entry_low = (
        price
        - 0.15 * atr_value
    )

    entry_high = (
        price
        + 0.05 * atr_value
    )

    # --------------------------------------------------------
    # Stop
    # --------------------------------------------------------

    stop = (
        price
        - cfg.atr_stop
        * atr_value
    )

    risk_pct = (
        (price - stop)
        / price
        * 100
    )

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    predicted_return = max(
        float(
            row["pred3"]
        ),
        0.25,
    )

    target2_pct = max(
        predicted_return,
        1.00,
    )

    target1_pct = max(
        target2_pct * 0.55,
        0.60,
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
        / max(
            risk_pct,
            0.01,
        )
    )

    rr2 = (
        target2_pct
        / max(
            risk_pct,
            0.01,
        )
    )

    # --------------------------------------------------------
    # Chase detection
    # --------------------------------------------------------

    chasing = (
        float(row["ext_atr"])
        > cfg.chase_atr
        or float(row["rsi"])
        > cfg.max_rsi_entry
    )

    if chasing:

        action = (
            "WAIT FOR PULLBACK"
        )

    else:

        action = (
            "BUY ON CONFIRMATION"
        )

    if rr2 < cfg.min_rr:

        action = (
            "WATCH / WAIT"
        )

    # --------------------------------------------------------
    # Expected holding period
    # --------------------------------------------------------

    if (
        float(row["pred5"])
        > float(row["pred3"]) * 1.15
        and float(row["p5"])
        >= float(row["p3"]) - 0.03
    ):

        holding_period = (
            "3-5 sessions"
        )

    else:

        holding_period = (
            "1-3 sessions"
        )

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
        "holding_period": holding_period,
    }


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_position_size(
    row,
    cfg,
):

    price = max(
        float(row["price"]),
        0.01,
    )

    stop = float(
        row["stop"]
    )

    risk_per_share = max(
        price - stop,
        0.01,
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

    shares_by_risk = math.floor(
        risk_budget
        / risk_per_share
    )

    shares_by_capital = math.floor(
        max_position_value
        / price
    )

    shares = max(
        0,
        min(
            shares_by_risk,
            shares_by_capital,
        ),
    )

    position_value = (
        shares * price
    )

    max_loss = (
        shares
        * risk_per_share
    )

    return (
        shares,
        position_value,
        max_loss,
    )


# ============================================================
# IPO RETRIEVAL
# ============================================================

def fetch_ipo_data():

    session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/151.0 Safari/537.36"
        ),
        "Accept": (
            "application/json,text/plain,*/*"
        ),
        "Referer": (
            "https://www.nseindia.com/"
        ),
    }

    try:

        session.get(
            "https://www.nseindia.com/",
            headers=headers,
            timeout=15,
        )

    except Exception:

        pass

    records = []

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

                data = data.get(
                    "data",
                    data.get(
                        "records",
                        [],
                    ),
                )

            if not isinstance(
                data,
                list,
            ):

                continue

            for item in data[:20]:

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
                        "name": str(
                            name
                        ),

                        "status": status,

                        "start": (
                            item.get(
                                "issueStartDate"
                            )
                            or item.get(
                                "startDate"
                            )
                            or item.get(
                                "issueStart"
                            )
                        ),

                        "end": (
                            item.get(
                                "issueEndDate"
                            )
                            or item.get(
                                "endDate"
                            )
                            or item.get(
                                "issueEnd"
                            )
                        ),

                        "price": (
                            item.get(
                                "priceBand"
                            )
                            or item.get(
                                "issuePrice"
                            )
                            or item.get(
                                "price"
                            )
                        ),

                        "subscription": (
                            item.get(
                                "subscription"
                            )
                            or item.get(
                                "subscriptionRatio"
                            )
                        ),
                    }
                )

        except Exception as exc:

            log.warning(
                "IPO %s endpoint failed: %s",
                status,
                exc,
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    output = []

    seen = set()

    for item in records:

        key = (
            item["name"].lower(),
            item["status"],
        )

        if key in seen:

            continue

        seen.add(key)

        output.append(
            item
        )

    if output:

        return output

    # IMPORTANT:
    # Never claim that there are no IPOs merely because
    # the data retrieval failed.
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
                "Verify directly on NSE IPO page",
        }
    ]


# ============================================================
# MARKET ENGINE
# ============================================================

class MarketEngineV63:

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

        if (
            self.cfg.nifty
            not in close.columns
        ):

            raise RuntimeError(
                "Nifty 50 data unavailable."
            )

        nifty = (
            close[
                self.cfg.nifty
            ]
            .dropna()
        )

        if (
            self.cfg.vix
            in close.columns
        ):

            vix = (
                close[
                    self.cfg.vix
                ]
                .dropna()
            )

        else:

            vix = pd.Series(
                dtype=float
            )

        latest = nifty.index[-1]

        frames = {}

        training_parts = {
            1: [],
            3: [],
            5: [],
        }

        # ----------------------------------------------------
        # Build features
        # ----------------------------------------------------

        for ticker in self.cfg.universe:

            if not all(
                ticker in dataset.columns
                for dataset in [
                    close,
                    high,
                    low,
                    volume,
                ]
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

            frame = build_features(
                close[ticker],
                high[ticker],
                low[ticker],
                volume[ticker],
                nifty,
                vix,
            )

            frames[ticker] = frame

            historical = (
                frame[
                    frame.index
                    < latest
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
        # Train models
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
                "V6.3 could not fit the 3-day model."
            )

        candidates = []

        rejected = []

        # ----------------------------------------------------
        # Score each stock
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

                predictions[
                    horizon
                ] = float(
                    return_model
                    .predict(x)[0]
                ) * 100

                probabilities[
                    horizon
                ] = get_probability(
                    bundle,
                    x,
                )

            if 3 not in predictions:

                continue

            # ------------------------------------------------
            # Basic values
            # ------------------------------------------------

            price = float(
                row[
                    "price"
                ].iloc[0]
            )

            atr_value = float(
                row[
                    "atr"
                ].iloc[0]
            )

            atr_pct = (
                atr_value
                / price
                * 100
            )

            turnover_cr = (
                price
                * float(
                    row[
                        "volume"
                    ].iloc[0]
                )
                / 1e7
            )

            rsi_value = float(
                row[
                    "rsi14"
                ].iloc[0]
            )

            volume_ratio = float(
                row[
                    "volume_ratio"
                ].iloc[0]
            )

            extension_atr = (
                float(
                    row[
                        "dist_ema20"
                    ].iloc[0]
                )
                * price
                / max(
                    atr_value,
                    1e-9,
                )
            )

            relative5 = float(
                row[
                    "relative5"
                ].iloc[0]
            ) * 100

            trend_aligned = (
                float(
                    row[
                        "dist_ema20"
                    ].iloc[0]
                ) > 0
                and
                float(
                    row[
                        "dist_sma50"
                    ].iloc[0]
                ) > 0
                and
                float(
                    row[
                        "ema20_slope10"
                    ].iloc[0]
                ) > 0
            )

            # ------------------------------------------------
            # Multi-horizon ensemble
            # ------------------------------------------------

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
            # Trend bonus
            # ------------------------------------------------

            if trend_aligned:

                score += 0.12

            # ------------------------------------------------
            # Relative strength bonus
            # ------------------------------------------------

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
            # Overbought penalty
            # ------------------------------------------------

            if rsi_value > 70:

                score -= 0.08

            if rsi_value > 76:

                score -= 0.08

            # ------------------------------------------------
            # Excessive extension penalty
            # ------------------------------------------------

            if extension_atr > 1.25:

                score -= 0.10

            # ------------------------------------------------
            # Nifty below SMA penalty
            #
            # Small penalty only.
            # V6.3 must still find stock-specific
            # opportunities during mixed markets.
            # ------------------------------------------------

            nifty_above = float(
                row[
                    "nifty_above_sma50"
                ].iloc[0]
            )

            if nifty_above < 0.5:

                score -= 0.03

            # ------------------------------------------------
            # Liquidity filters
            # ------------------------------------------------

            if (
                price
                < self.cfg.min_price
            ):

                continue

            if (
                turnover_cr
                < self.cfg.min_turnover_cr
            ):

                continue

            # ------------------------------------------------
            # Tier
            # ------------------------------------------------

            tier = determine_tier(
                ensemble_probability,
                ensemble_return,
                score,
                self.cfg,
            )

            candidate = {
                "ticker":
                    ticker.replace(
                        ".NS",
                        "",
                    ),

                "price":
                    price,

                "atr":
                    atr_value,

                "atr_pct":
                    atr_pct,

                "rsi":
                    rsi_value,

                "volume_ratio":
                    volume_ratio,

                "ext_atr":
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

                "tier":
                    tier,

                "turnover_cr":
                    turnover_cr,
            }

            candidates.append(
                candidate
            )

            if tier == "REJECTED":

                rejected.append(
                    candidate
                )

        # ----------------------------------------------------
        # Save rejected candidates
        # ----------------------------------------------------

        self.save_rejected(
            rejected
        )

        # ----------------------------------------------------
        # No usable candidates
        # ----------------------------------------------------

        if not candidates:

            return pd.DataFrame(
                [
                    {
                        "ticker":
                            "--",

                        "tier":
                            "NO TRADE",

                        "action":
                            (
                                "No usable "
                                "candidates."
                            ),
                    }
                ]
            )

        # ----------------------------------------------------
        # Sort all candidates
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["p3"],
                item["pred3"],
            ),
            reverse=True,
        )

        # ----------------------------------------------------
        # Prefer actual tradeable candidates
        # ----------------------------------------------------

        tradeable = [
            item
            for item in candidates
            if item["tier"]
            in [
                "HIGH-CONFIDENCE",
                "TRADEABLE",
            ]
        ]

        # If none qualify, show the best available
        # candidates as WATCHLIST instead of falsely
        # reporting that the model found nothing.
        # ----------------------------------------------------

        if tradeable:

            selected = tradeable[
                : self.cfg.top_n
            ]

        else:

            selected = candidates[
                : self.cfg.top_n
            ]

            for item in selected:

                if item["tier"] == "REJECTED":

                    item["tier"] = (
                        "WATCHLIST"
                    )

        # ----------------------------------------------------
        # Trade plans
        # ----------------------------------------------------

        output = []

        for item in selected:

            plan = build_trade_plan(
                item,
                self.cfg,
            )

            item.update(
                plan
            )

            (
                shares,
                position_value,
                max_loss,
            ) = calculate_position_size(
                item,
                self.cfg,
            )

            item[
                "shares"
            ] = shares

            item[
                "position_value"
            ] = position_value

            item[
                "max_loss"
            ] = max_loss

            output.append(
                item
            )

        result = pd.DataFrame(
            output
        )

        self.save_candidates(
            result
        )

        return result

    # ========================================================
    # SAVE CANDIDATES
    # ========================================================

    def save_candidates(
        self,
        dataframe,
    ):

        try:

            df = dataframe.copy()

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
        picks,
        ipos,
    ):

        timestamp = (
            now_ist().strftime(
                "%d %b %Y, %H:%M IST"
            )
        )

        lines = [
            "*MULTI-FACTOR MARKET ALERT V6.3*",
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

        lines.extend(
            [
                "",
                "--- TOP SHORT-TERM "
                "OPPORTUNITIES ---",
            ]
        )

        # ----------------------------------------------------
        # No result
        # ----------------------------------------------------

        if (
            picks.empty
            or picks.iloc[0].get(
                "ticker"
            ) == "--"
        ):

            lines.extend(
                [
                    "",
                    "*NO TRADEABLE SETUP*",
                    "",
                    (
                        "No usable candidate "
                        "was produced by the model."
                    ),
                ]
            )

        else:

            for number, (
                _,
                row,
            ) in enumerate(
                picks.iterrows(),
                start=1,
            ):

                lines.extend(
                    [
                        "",
                        (
                            f"*{number}. "
                            f"{row['ticker']} — "
                            f"{row['tier']}*"
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
                            f"Volume: "
                            f"{row['volume_ratio']:.2f}x"
                        ),

                        (
                            f"Entry: "
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

        # ----------------------------------------------------
        # IPO section
        # ----------------------------------------------------

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
                        f"{ipo.get('start') or '-'} "
                        f"to "
                        f"{ipo.get('end') or '-'}"
                    )
                )

                lines.append(
                    (
                        f"Price: "
                        f"{ipo.get('price') or '-'}"
                    )
                )

                lines.append(
                    (
                        f"Subscription: "
                        f"{ipo.get('subscription') or '-'}"
                    )
                )

        lines.extend(
            [
                "",
                (
                    "_V6.3 is a probabilistic "
                    "research screen and does not "
                    "guarantee profit. "
                    "IPO/GMP information should be "
                    "independently verified._"
                ),
            ]
        )

        return "\n".join(
            lines
        )

    # ========================================================
    # TELEGRAM
    # ========================================================

    def send_telegram(
        self,
        regime,
        picks,
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
            picks,
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
            "Telegram V6.3 alert sent."
        )

        # ----------------------------------------------------
        # Save audit copy
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
# MARKET REGIME DATA
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

    nifty = (
        data["Close"][
            cfg.nifty
        ]
        .dropna()
    )

    if (
        cfg.vix
        in data["Close"].columns
    ):

        vix = (
            data["Close"][
                cfg.vix
            ]
            .dropna()
        )

    else:

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
        "Starting V6.3 walk-forward backtest."
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

    nifty = (
        close[
            cfg.nifty
        ]
        .dropna()
    )

    if (
        cfg.vix
        in close.columns
    ):

        vix = (
            close[
                cfg.vix
            ]
            .dropna()
        )

    else:

        vix = pd.Series(
            dtype=float
        )

    frames = {}

    for ticker in cfg.universe:

        if not all(
            ticker in dataset.columns
            for dataset in [
                close,
                high,
                low,
                volume,
            ]
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

        frames[ticker] = (
            build_features(
                close[ticker],
                high[ticker],
                low[ticker],
                volume[ticker],
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

    all_dates = sorted(
        set.intersection(
            *[
                set(
                    frame.index
                )
                for frame
                in frames.values()
            ]
        )
    )

    test_dates = all_dates[
        600:-6:5
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

            if (
                current_date
                not in frame.index
            ):

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
                ] = float(
                    bundle[0]
                    .predict(x)[0]
                ) * 100

                probabilities[
                    horizon
                ] = get_probability(
                    bundle,
                    x,
                )

            if 3 not in predictions:

                continue

            price = float(
                row[
                    "price"
                ].iloc[0]
            )

            atr_value = float(
                row[
                    "atr"
                ].iloc[0]
            )

            atr_pct = (
                atr_value
                / price
                * 100
            )

            turnover_cr = (
                price
                * float(
                    row[
                        "volume"
                    ].iloc[0]
                )
                / 1e7
            )

            if (
                price
                < cfg.min_price
            ):

                continue

            if (
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
                float(
                    row[
                        "dist_ema20"
                    ].iloc[0]
                ) > 0
                and
                float(
                    row[
                        "dist_sma50"
                    ].iloc[0]
                ) > 0
                and
                float(
                    row[
                        "ema20_slope10"
                    ].iloc[0]
                ) > 0
            )

            if trend:

                score += 0.12

            if (
                float(
                    row[
                        "relative5"
                    ].iloc[0]
                )
                > 0
            ):

                score += 0.05

            if (
                float(
                    row[
                        "rsi14"
                    ].iloc[0]
                )
                > 70
            ):

                score -= 0.08

            if (
                float(
                    row[
                        "rsi14"
                    ].iloc[0]
                )
                > 76
            ):

                score -= 0.08

            if (
                score
                >= cfg.trade_score
                and
                ensemble_probability
                >= cfg.trade_prob
                and
                ensemble_return
                >= cfg.trade_return
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
            expected_return,
        ) = candidates[0]

        series = (
            close[ticker]
            .dropna()
        )

        try:

            position = (
                series.index.get_loc(
                    current_date
                )
            )

        except KeyError:

            continue

        if (
            position + 3
            >= len(series)
        ):

            continue

        exit_price = float(
            series.iloc[
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

        trades.append(
            {
                "date":
                    current_date,

                "ticker":
                    ticker.replace(
                        ".NS",
                        "",
                    ),

                "entry":
                    entry,

                "exit":
                    exit_price,

                "predicted_return":
                    expected_return,

                "probability":
                    probability
                    * 100,

                "score":
                    score,

                "gross_3d_pct":
                    gross_return,

                "net_3d_pct":
                    net_return,
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
            "Indian Market Engine V6.3"
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
        help="Backtest period.",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Capital for position sizing.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CAPITAL
    #
    # Handles:
    #   missing secret
    #   empty secret
    #   invalid secret
    # --------------------------------------------------------

    environment_capital = (
        os.getenv("CAPITAL")
        or "100000"
    )

    if args.capital is not None:

        capital = safe_float(
            args.capital,
            100000.0,
        )

    else:

        capital = safe_float(
            environment_capital,
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

    log.info(
        "Capital: ₹%.2f",
        capital,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

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
    # BACKTEST MODE
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
            "===================================="
        )

        print(
            "V6.3 WALK-FORWARD BACKTEST"
        )

        print(
            "===================================="
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
            f"Detailed trades saved to "
            f"{cfg.backtest_file}"
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

    engine = MarketEngineV63(
        cfg
    )

    picks = engine.scan()

    ipos = fetch_ipo_data()

    engine.send_telegram(
        regime,
        picks,
        ipos,
    )

    log.info(
        "V6.3 completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
