"""
Indian Market Engine V6.2
=========================

Short-term Indian equity opportunity scanner.

Main features:
- 1 / 3 / 5-session prediction
- calibrated probability
- technical + OHLCV features
- Nifty 50 market regime
- India VIX
- relative strength
- volume confirmation
- ATR-based risk management
- risk-adjusted ranking
- entry / stop / targets
- position sizing
- correlation warning
- candidate history
- rejected candidate history
- Telegram alerts
- IPO discovery
- historical walk-forward backtest

IMPORTANT:
This is a probabilistic research/screening system.
It does NOT guarantee profits.
"""

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
# GLOBAL SETTINGS
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger("market_engine_v6_2")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class EngineConfig:

    # Telegram
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None

    # Prediction horizons
    short_horizons: tuple = (1, 3, 5)
    primary_horizon: int = 3

    # Model
    model_lookback: int = 504
    min_training_samples: int = 2500

    # Number of final picks
    top_n: int = 3

    # --------------------------------------------------------
    # ENTRY FILTERS
    # --------------------------------------------------------

    min_probability: float = 0.54

    min_pred_return_pct: float = 0.20

    # Score is expressed in percentage points.
    min_score_pct: float = 0.10

    # --------------------------------------------------------
    # RISK / EXECUTION
    # --------------------------------------------------------

    atr_stop_multiple: float = 1.25

    atr_entry_buffer: float = 0.20

    target1_fraction: float = 0.55

    min_target1_pct: float = 0.60

    min_target2_pct: float = 1.00

    max_entry_extension_atr: float = 1.15

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    min_price: float = 50.0

    min_avg_turnover_cr: float = 25.0

    # --------------------------------------------------------
    # POSITION SIZING
    # --------------------------------------------------------

    max_position_pct: float = 0.25

    risk_per_trade_pct: float = 0.75

    total_risk_pct: float = 2.0

    capital: float = 100000.0

    # --------------------------------------------------------
    # BACKTEST COSTS
    # --------------------------------------------------------

    slippage_pct: float = 0.05

    brokerage_pct: float = 0.03

    # --------------------------------------------------------
    # HISTORY FILES
    # --------------------------------------------------------

    history_path: str = (
        "alert_history_v6_2.csv"
    )

    candidate_history_path: str = (
        "candidate_history_v6_2.csv"
    )

    missed_history_path: str = (
        "missed_opportunities_v6_2.csv"
    )

    backtest_path: str = (
        "backtest_v6_2_trades.csv"
    )

    # --------------------------------------------------------
    # FALLBACK UNIVERSE
    # --------------------------------------------------------

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

    nifty_ticker: str = "^NSEI"

    vix_ticker: str = "^INDIAVIX"


# ============================================================
# MODEL FEATURES
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
# GENERAL HELPERS
# ============================================================

def now_ist() -> datetime:
    return datetime.now(IST)


def safe_float(value, default=np.nan):

    try:
        return float(value)

    except Exception:
        return default


# ============================================================
# DOWNLOAD MARKET DATA
# ============================================================

def download_ohlcv(
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
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )

            if raw.empty:
                raise ValueError(
                    "Empty yfinance response"
                )

            fields = {
                "Open",
                "High",
                "Low",
                "Close",
                "Adj Close",
                "Volume",
            }

            result = {}

            if isinstance(
                raw.columns,
                pd.MultiIndex,
            ):

                level0 = set(
                    raw.columns.get_level_values(0)
                )

                if fields & level0:

                    for field in fields:

                        if field in level0:
                            result[field] = raw[field]

                else:

                    level1 = set(
                        raw.columns.get_level_values(1)
                    )

                    for field in fields:

                        if field in level1:

                            result[field] = raw.xs(
                                field,
                                axis=1,
                                level=1,
                            )

            else:

                for field in fields:

                    if field in raw.columns:

                        result[field] = raw[[field]]

            if "Close" not in result:

                raise ValueError(
                    "Close field missing from "
                    "yfinance response"
                )

            return result

        except Exception as exc:

            last_error = exc

            log.warning(
                "Download attempt %d/%d failed: %s",
                attempt,
                retries,
                exc,
            )

            if attempt < retries:

                time.sleep(
                    2 * attempt
                )

    raise RuntimeError(
        f"Market data download failed: {last_error}"
    )


# ============================================================
# NIFTY UNIVERSE
# ============================================================

def load_nifty500_universe(
    fallback,
):

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
            timeout=20,
        )

        response.raise_for_status()

        df = pd.read_csv(
            pd.io.common.BytesIO(
                response.content
            )
        )

        if "Symbol" in df.columns:

            symbol_column = "Symbol"

        else:

            symbol_column = df.columns[0]

        symbols = [

            str(symbol)
            .strip()
            .upper()
            + ".NS"

            for symbol
            in df[symbol_column].dropna()
        ]

        symbols = [
            symbol
            for symbol in symbols
            if symbol != "NAN.NS"
        ]

        if len(symbols) >= 200:

            log.info(
                "Loaded %d current Nifty constituents.",
                len(symbols),
            )

            return symbols

    except Exception as exc:

        log.warning(
            "Could not load current Nifty universe: %s",
            exc,
        )

    log.warning(
        "Using embedded fallback universe."
    )

    return list(fallback)


# ============================================================
# RSI
# ============================================================

def rsi(
    series,
    period=14,
):

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
        (-delta.clip(upper=0))
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
        - 100 / (1 + rs)
    )


# ============================================================
# ATR
# ============================================================

def atr(
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
    market,
    vix,
):

    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)

    market = (
        market
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

    market_sma50 = (
        market
        .rolling(50)
        .mean()
    )

    atr_value = atr(
        high,
        low,
        close,
    )

    returns = close.pct_change()

    frame = pd.DataFrame(
        index=close.index
    )

    for n in (
        1,
        3,
        5,
        10,
        20,
    ):

        frame[
            f"ret{n}"
        ] = close.pct_change(n)

    frame["rsi14"] = rsi(
        close
    )

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

    frame["atr_pct"] = (
        atr_value / close
    )

    frame["vol20"] = (
        returns
        .rolling(20)
        .std()
    )

    frame["volume_ratio"] = (
        volume
        / volume
        .rolling(20)
        .median()
    )

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

    nifty_ret3 = (
        market.pct_change(3)
    )

    nifty_ret5 = (
        market.pct_change(5)
    )

    nifty_ret10 = (
        market.pct_change(10)
    )

    nifty_ret20 = (
        market.pct_change(20)
    )

    frame["nifty_ret3"] = (
        nifty_ret3
    )

    frame["nifty_ret10"] = (
        nifty_ret10
    )

    frame[
        "nifty_above_sma50"
    ] = (
        market > market_sma50
    ).astype(float)

    frame[
        "nifty_sma50_slope10"
    ] = (
        market_sma50
        .pct_change(10)
    )

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

    frame["vix_level"] = vix

    frame["vix_change5"] = (
        vix.pct_change(5)
    )

    # Raw values for trade construction.
    frame["price"] = close

    frame["atr"] = atr_value

    frame["volume"] = volume

    # Future targets.
    for horizon in (
        1,
        3,
        5,
    ):

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

def fit_models(
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
            [np.inf, -np.inf],
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

    if len(x) < cfg.min_training_samples:
        return None

    if y.nunique() < 10:
        return None

    cut = int(
        len(x) * 0.80
    )

    if (
        cut < 1000
        or len(x) - cut < 200
    ):
        return None

    x_train = x.iloc[:cut]
    y_train = y.iloc[:cut]

    x_cal = x.iloc[cut:]
    y_cal = y.iloc[cut:]

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
                    max_iter=1500,
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
        x_train,
        clipped_y,
    )

    direction_model.fit(
        x_train,
        (
            y_train > 0
        ).astype(int),
    )

    holdout_probability = (
        direction_model
        .predict_proba(
            x_cal
        )[:, 1]
    )

    holdout_actual = (
        y_cal > 0
    ).astype(int).to_numpy()

    calibrator = None

    if (
        len(holdout_probability) >= 100
        and len(
            np.unique(
                holdout_actual
            )
        ) == 2
    ):

        calibrator = (
            IsotonicRegression(
                out_of_bounds="clip"
            )
        )

        calibrator.fit(
            holdout_probability,
            holdout_actual,
        )

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
        calibrator,
    )


# ============================================================
# CALIBRATED PROBABILITY
# ============================================================

def calibrated_probability(
    direction_model,
    calibrator,
    x,
):

    raw_probability = float(
        direction_model
        .predict_proba(
            x
        )[0, 1]
    )

    if calibrator is None:
        return raw_probability

    return float(
        calibrator
        .predict(
            [raw_probability]
        )[0]
    )


# ============================================================
# MARKET REGIME
# ============================================================

def market_regime(
    market,
    vix,
):

    market = market.dropna()

    sma50 = (
        market
        .rolling(50)
        .mean()
    )

    slope = (
        sma50
        .pct_change(10)
    )

    latest = float(
        market.iloc[-1]
    )

    latest_sma = float(
        sma50.iloc[-1]
    )

    latest_slope = float(
        slope.iloc[-1]
    )

    if not vix.dropna().empty:

        latest_vix = float(
            vix
            .dropna()
            .iloc[-1]
        )

    else:

        latest_vix = np.nan

    score = 0

    if latest > latest_sma:
        score += 1
    else:
        score -= 1

    if latest_slope > 0:
        score += 1
    else:
        score -= 1

    if np.isfinite(
        latest_vix
    ):

        if latest_vix > 20:
            score -= 1

        elif latest_vix < 14:
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
        "sma50": latest_sma,
        "sma50_slope10_pct":
            latest_slope * 100,
        "vix": latest_vix,
        "score": score,
    }


# ============================================================
# TRADE PLAN
# ============================================================

def candidate_plan(
    row,
    cfg,
):

    price = float(
        row["Price"]
    )

    atr_value = float(
        row["ATR"]
    )

    if (
        not np.isfinite(
            atr_value
        )
        or atr_value <= 0
    ):

        atr_value = (
            price
            * max(
                float(
                    row["ATR_Pct"]
                ) / 100,
                0.01,
            )
        )

    entry_low = max(
        0.01,
        price
        - (
            cfg.atr_entry_buffer
            * atr_value
        ),
    )

    entry_high = (
        price
        + (
            0.10
            * atr_value
        )
    )

    stop = (
        price
        - (
            cfg.atr_stop_multiple
            * atr_value
        )
    )

    predicted_return = max(
        float(
            row[
                "Predicted_3D_Return_Pct"
            ]
        ) / 100,
        0,
    )

    target2_return = max(
        predicted_return,
        cfg.min_target2_pct / 100,
    )

    target1_return = max(
        target2_return
        * cfg.target1_fraction,
        cfg.min_target1_pct / 100,
    )

    target1 = (
        price
        * (
            1
            + target1_return
        )
    )

    target2 = (
        price
        * (
            1
            + target2_return
        )
    )

    risk_pct = (
        (
            price
            - stop
        )
        / price
        * 100
    )

    rr1 = (
        target1_return
        * 100
        / max(
            risk_pct,
            0.01,
        )
    )

    rr2 = (
        target2_return
        * 100
        / max(
            risk_pct,
            0.01,
        )
    )

    if (
        row["Probability_3D_Pct"]
        >= 62
        and row["Score_Pct"]
        >= 0.35
    ):

        tier = "HIGH-CONFIDENCE"

    elif (
        row["Probability_3D_Pct"]
        >= 57
        and row["Score_Pct"]
        >= 0.18
    ):

        tier = "GOOD SETUP"

    else:

        tier = "SPECULATIVE / SMALL SIZE"

    action = (
        "BUY ON CONFIRMATION"
    )

    if (
        row["RSI"] >= 72
        or row["Extension_ATR"]
        > cfg.max_entry_extension_atr
    ):

        action = (
            "WAIT FOR PULLBACK — "
            "DO NOT CHASE"
        )

    return {

        "Entry_Low":
            round(
                entry_low,
                2,
            ),

        "Entry_High":
            round(
                entry_high,
                2,
            ),

        "Stop_Loss":
            round(
                stop,
                2,
            ),

        "Target_1":
            round(
                target1,
                2,
            ),

        "Target_2":
            round(
                target2,
                2,
            ),

        "Risk_Pct":
            round(
                risk_pct,
                2,
            ),

        "RR_T1":
            round(
                rr1,
                2,
            ),

        "RR_T2":
            round(
                rr2,
                2,
            ),

        "Tier":
            tier,

        "Action":
            action,
    }


# ============================================================
# POSITION SIZING
# ============================================================

def add_position_size(
    df,
    cfg,
):

    if df.empty:
        return df

    if df.iloc[0].get(
        "No_Trade",
        False,
    ):
        return df

    total_risk_budget = (
        cfg.capital
        * cfg.total_risk_pct
        / 100
    )

    for index, row in df.iterrows():

        price = float(
            row["Price"]
        )

        stop = float(
            row["Stop_Loss"]
        )

        risk_per_share = max(
            price - stop,
            0.01,
        )

        trade_risk_budget = (
            cfg.capital
            * cfg.risk_per_trade_pct
            / 100
        )

        usable_risk_budget = min(
            trade_risk_budget,
            total_risk_budget,
        )

        shares_by_risk = math.floor(
            usable_risk_budget
            / risk_per_share
        )

        max_position_value = (
            cfg.capital
            * cfg.max_position_pct
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
            shares
            * price
        )

        maximum_loss = (
            shares
            * risk_per_share
        )

        df.loc[
            index,
            "Shares"
        ] = shares

        df.loc[
            index,
            "Position_Value"
        ] = round(
            position_value,
            2,
        )

        df.loc[
            index,
            "Position_Pct"
        ] = round(
            (
                position_value
                / cfg.capital
                * 100
            )
            if cfg.capital
            else 0,
            2,
        )

        df.loc[
            index,
            "Max_Loss_Rs"
        ] = round(
            maximum_loss,
            2,
        )

    return df


# ============================================================
# MAIN ENGINE
# ============================================================

class MarketEngineV62:

    def __init__(
        self,
        cfg,
    ):

        self.cfg = cfg

    # --------------------------------------------------------
    # STOCK SCANNER
    # --------------------------------------------------------

    def get_stock_picks(
        self,
    ):

        universe = (
            load_nifty500_universe(
                self.cfg.fallback_universe
            )
        )

        tickers = list(
            dict.fromkeys(
                universe
                + [
                    self.cfg.nifty_ticker,
                    self.cfg.vix_ticker,
                ]
            )
        )

        log.info(
            "Downloading data for %d tickers.",
            len(tickers),
        )

        data = download_ohlcv(
            tickers,
            period="3y",
        )

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        if (
            self.cfg.nifty_ticker
            not in close.columns
        ):

            raise RuntimeError(
                "Nifty 50 data unavailable."
            )

        market = (
            close[
                self.cfg.nifty_ticker
            ]
            .dropna()
        )

        if (
            self.cfg.vix_ticker
            in close.columns
        ):

            vix = (
                close[
                    self.cfg.vix_ticker
                ]
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
            horizon: []
            for horizon
            in self.cfg.short_horizons
        }

        rejected = []

        # ----------------------------------------------------
        # BUILD FEATURES
        # ----------------------------------------------------

        for ticker in universe:

            if not all(
                ticker in dataset.columns
                for dataset in (
                    close,
                    high,
                    low,
                    volume,
                )
            ):

                continue

            series = (
                close[ticker]
                .dropna()
            )

            if len(series) < 650:
                continue

            frame = build_features(
                close[ticker],
                high[ticker],
                low[ticker],
                volume[ticker],
                market,
                vix,
            )

            frames[ticker] = frame

            historical = (
                frame[
                    frame.index
                    < latest_date
                ]
                .tail(
                    self.cfg.model_lookback
                )
            )

            for horizon in (
                self.cfg.short_horizons
            ):

                training_parts[
                    horizon
                ].append(
                    historical
                )

        # ----------------------------------------------------
        # TRAIN MODELS
        # ----------------------------------------------------

        models = {}

        for horizon in (
            self.cfg.short_horizons
        ):

            if not training_parts[
                horizon
            ]:

                continue

            training = pd.concat(
                training_parts[horizon],
                ignore_index=True,
            )

            models[horizon] = (
                fit_models(
                    training,
                    horizon,
                    self.cfg,
                )
            )

        primary = models.get(
            self.cfg.primary_horizon
        )

        if primary is None:

            raise RuntimeError(
                "Could not fit the "
                "V6.2 3-session model."
            )

        rows = []

        # ----------------------------------------------------
        # SCORE STOCKS
        # ----------------------------------------------------

        for ticker, frame in frames.items():

            if (
                latest_date
                not in frame.index
            ):

                continue

            row = frame.loc[
                [latest_date]
            ]

            if (
                row[
                    FEATURES
                ]
                .isna()
                .any(axis=1)
                .iloc[0]
            ):

                continue

            x = row[
                FEATURES
            ]

            predictions = {}

            probabilities = {}

            for horizon in (
                self.cfg.short_horizons
            ):

                bundle = models.get(
                    horizon
                )

                if bundle is None:
                    continue

                (
                    return_model,
                    direction_model,
                    calibrator,
                ) = bundle

                predictions[horizon] = (
                    float(
                        return_model
                        .predict(x)[0]
                    )
                )

                probabilities[horizon] = (
                    calibrated_probability(
                        direction_model,
                        calibrator,
                        x,
                    )
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
                if price
                else np.nan
            )

            rsi_value = float(
                row[
                    "rsi14"
                ].iloc[0]
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

            # ------------------------------------------------
            # ENSEMBLE
            # ------------------------------------------------

            return_1 = predictions.get(
                1,
                predictions[3],
            )

            return_3 = predictions[3]

            return_5 = predictions.get(
                5,
                predictions[3],
            )

            probability_1 = probabilities.get(
                1,
                probabilities[3],
            )

            probability_3 = probabilities[3]

            probability_5 = probabilities.get(
                5,
                probabilities[3],
            )

            ensemble_return = (
                0.20 * return_1
                + 0.55 * return_3
                + 0.25 * return_5
            )

            ensemble_probability = (
                0.20 * probability_1
                + 0.55 * probability_3
                + 0.25 * probability_5
            )

            predicted_return_pct = (
                ensemble_return * 100
            )

            # ------------------------------------------------
            # RISK-ADJUSTED SCORE
            #
            # All score components are in percentage points.
            # ------------------------------------------------

            directional_edge_pct = (
                predicted_return_pct
                * (
                    2
                    * ensemble_probability
                    - 1
                )
            )

            uncertainty_penalty_pct = (
                0.25 * atr_pct
            )

            score_pct = (
                directional_edge_pct
                - uncertainty_penalty_pct
            )

            # ------------------------------------------------
            # TECHNICAL CONDITIONS
            # ------------------------------------------------

            breadth_ok = (
                float(
                    row[
                        "nifty_above_sma50"
                    ].iloc[0]
                )
                > 0
            )

            trend_aligned = (
                float(
                    row[
                        "dist_ema20"
                    ].iloc[0]
                )
                > 0
                and
                float(
                    row[
                        "dist_sma50"
                    ].iloc[0]
                )
                > 0
                and
                float(
                    row[
                        "ema20_slope10"
                    ].iloc[0]
                )
                > 0
            )

            volume_ratio = float(
                row[
                    "volume_ratio"
                ].iloc[0]
            )

            relative_strength = float(
                row[
                    "relative5"
                ].iloc[0]
            )

            # ------------------------------------------------
            # SOFT PENALTIES / BONUSES
            # ------------------------------------------------

            if not breadth_ok:

                score_pct -= 0.08

            if rsi_value > 70:

                score_pct -= 0.10

            if rsi_value < 42:

                score_pct -= 0.05

            if extension_atr > 1.25:

                score_pct -= 0.12

            score_pct += (
                0.08
                * np.clip(
                    relative_strength * 10,
                    -1,
                    1,
                )
            )

            score_pct += (
                0.06
                * np.clip(
                    (
                        volume_ratio - 1
                    )
                    / 1.5,
                    -1,
                    1,
                )
            )

            score_pct += (
                0.08
                if trend_aligned
                else -0.08
            )

            # ------------------------------------------------
            # HARD FILTERS
            # ------------------------------------------------

            liquid = (
                price
                >= self.cfg.min_price
                and
                turnover_cr
                >= self.cfg.min_avg_turnover_cr
            )

            plausible = (
                predicted_return_pct
                >= self.cfg.min_pred_return_pct
                and
                ensemble_probability
                >= self.cfg.min_probability
            )

            reasons = []

            if not liquid:

                reasons.append(
                    "liquidity"
                )

            if (
                predicted_return_pct
                < self.cfg.min_pred_return_pct
            ):

                reasons.append(
                    "expected return"
                )

            if (
                ensemble_probability
                < self.cfg.min_probability
            ):

                reasons.append(
                    "probability"
                )

            if (
                score_pct
                < self.cfg.min_score_pct
            ):

                reasons.append(
                    "risk-adjusted score"
                )

            if not reasons:

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
                                return_1 * 100,
                                2,
                            ),

                        "Predicted_3D_Return_Pct":
                            round(
                                return_3 * 100,
                                2,
                            ),

                        "Predicted_5D_Return_Pct":
                            round(
                                return_5 * 100,
                                2,
                            ),

                        "Probability_1D_Pct":
                            round(
                                probability_1 * 100,
                                1,
                            ),

                        "Probability_3D_Pct":
                            round(
                                probability_3 * 100,
                                1,
                            ),

                        "Probability_5D_Pct":
                            round(
                                probability_5 * 100,
                                1,
                            ),

                        "Ensemble_Return_Pct":
                            round(
                                predicted_return_pct,
                                2,
                            ),

                        "Ensemble_Probability_Pct":
                            round(
                                ensemble_probability * 100,
                                1,
                            ),

                        "Score_Pct":
                            round(
                                score_pct,
                                3,
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
                                relative_strength * 100,
                                2,
                            ),

                        "Trend_Aligned":
                            trend_aligned,

                        "Turnover_Cr":
                            round(
                                turnover_cr,
                                1,
                            ),
                    }
                )

            else:

                rejected.append(
                    {
                        "date":
                            now_ist().strftime(
                                "%Y-%m-%d"
                            ),

                        "ticker":
                            ticker.replace(
                                ".NS",
                                "",
                            ),

                        "price":
                            round(
                                price,
                                2,
                            ),

                        "predicted_3d_pct":
                            round(
                                return_3 * 100,
                                2,
                            ),

                        "probability_3d_pct":
                            round(
                                probability_3 * 100,
                                1,
                            ),

                        "score_pct":
                            round(
                                score_pct,
                                3,
                            ),

                        "rejection_reason":
                            ", ".join(
                                reasons
                            ),
                    }
                )

        self.log_missed(
            rejected
        )

        # ----------------------------------------------------
        # NO TRADE
        # ----------------------------------------------------

        if not rows:

            no_trade = pd.DataFrame(
                [
                    {
                        "Ticker": "--",

                        "No_Trade": True,

                        "Tier": "NO TRADE",

                        "Action": (
                            "No candidate cleared "
                            "the V6.2 confidence, "
                            "return and risk filters."
                        ),
                    }
                ]
            )

            self.log_candidates(
                no_trade
            )

            return no_trade

        # ----------------------------------------------------
        # RANK FINAL CANDIDATES
        # ----------------------------------------------------

        df = pd.DataFrame(
            rows
        )

        df = (
            df
            .sort_values(
                [
                    "Score_Pct",
                    "Probability_3D_Pct",
                    "Predicted_3D_Return_Pct",
                ],
                ascending=False,
            )
            .head(
                self.cfg.top_n
            )
            .reset_index(
                drop=True
            )
        )

        # ----------------------------------------------------
        # TRADE PLANS
        # ----------------------------------------------------

        for index, row in df.iterrows():

            plan = candidate_plan(
                row,
                self.cfg,
            )

            for key, value in plan.items():

                df.loc[
                    index,
                    key,
                ] = value

        df = add_position_size(
            df,
            self.cfg,
        )

        df["No_Trade"] = False

        # ----------------------------------------------------
        # CORRELATION
        # ----------------------------------------------------

        if len(df) > 1:

            names = [
                ticker + ".NS"
                for ticker
                in df["Ticker"]
            ]

            valid_names = [
                ticker
                for ticker
                in names
                if ticker in close.columns
            ]

            panel = (
                close[valid_names]
                .dropna()
            )

            if panel.shape[1] > 1:

                correlation = (
                    panel
                    .pct_change()
                    .corr()
                )

                values = (
                    correlation.values
                )

                n = len(
                    correlation
                )

                average_corr = (
                    values.sum() - n
                ) / (
                    n * (n - 1)
                )

                df.attrs[
                    "avg_pairwise_correlation"
                ] = round(
                    float(
                        average_corr
                    ),
                    2,
                )

                df.attrs[
                    "high_concentration_warning"
                ] = bool(
                    average_corr > 0.65
                )

            else:

                df.attrs[
                    "avg_pairwise_correlation"
                ] = None

                df.attrs[
                    "high_concentration_warning"
                ] = False

        else:

            df.attrs[
                "avg_pairwise_correlation"
            ] = None

            df.attrs[
                "high_concentration_warning"
            ] = False

        self.log_candidates(
            df
        )

        return df

    # --------------------------------------------------------
    # CANDIDATE HISTORY
    # --------------------------------------------------------

    def log_candidates(
        self,
        df,
    ):

        output = df.copy()

        output["date"] = (
            now_ist().strftime(
                "%Y-%m-%d %H:%M IST"
            )
        )

        try:

            if os.path.exists(
                self.cfg.candidate_history_path
            ):

                old = pd.read_csv(
                    self.cfg.candidate_history_path
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
                self.cfg.candidate_history_path,
                index=False,
            )

        except Exception as exc:

            log.warning(
                "Candidate history failed: %s",
                exc,
            )

    # --------------------------------------------------------
    # MISSED / REJECTED HISTORY
    # --------------------------------------------------------

    def log_missed(
        self,
        rows,
    ):

        if not rows:
            return

        try:

            if os.path.exists(
                self.cfg.missed_history_path
            ):

                old = pd.read_csv(
                    self.cfg.missed_history_path
                )

            else:

                old = pd.DataFrame()

            combined = pd.concat(
                [
                    old,
                    pd.DataFrame(rows),
                ],
                ignore_index=True,
            )

            combined.to_csv(
                self.cfg.missed_history_path,
                index=False,
            )

        except Exception as exc:

            log.warning(
                "Missed history failed: %s",
                exc,
            )

    # --------------------------------------------------------
    # ALERT HISTORY
    # --------------------------------------------------------

    def log_alert(
        self,
        df,
        regime,
    ):

        rows = []

        for _, row in df.iterrows():

            rows.append(
                {
                    "date":
                        now_ist().strftime(
                            "%Y-%m-%d %H:%M IST"
                        ),

                    "ticker":
                        row.get(
                            "Ticker"
                        ),

                    "price":
                        row.get(
                            "Price"
                        ),

                    "score_pct":
                        row.get(
                            "Score_Pct"
                        ),

                    "predicted_3d_return_pct":
                        row.get(
                            "Predicted_3D_Return_Pct"
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

                    "target_1":
                        row.get(
                            "Target_1"
                        ),

                    "target_2":
                        row.get(
                            "Target_2"
                        ),

                    "rr_t1":
                        row.get(
                            "RR_T1"
                        ),

                    "rr_t2":
                        row.get(
                            "RR_T2"
                        ),

                    "shares":
                        row.get(
                            "Shares"
                        ),

                    "position_value":
                        row.get(
                            "Position_Value"
                        ),

                    "max_loss_rs":
                        row.get(
                            "Max_Loss_Rs"
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
                        regime["label"],
                }
            )

        try:

            if os.path.exists(
                self.cfg.history_path
            ):

                old = pd.read_csv(
                    self.cfg.history_path
                )

            else:

                old = pd.DataFrame()

            combined = pd.concat(
                [
                    old,
                    pd.DataFrame(rows),
                ],
                ignore_index=True,
            )

            combined.to_csv(
                self.cfg.history_path,
                index=False,
            )

        except Exception as exc:

            log.warning(
                "Alert history failed: %s",
                exc,
            )

    # --------------------------------------------------------
    # TELEGRAM MESSAGE
    # --------------------------------------------------------

    def format_stock_section(
        self,
        df,
    ):

        lines = [
            "--- SHORT-TERM OPPORTUNITIES "
            "(1–5 SESSIONS) ---"
        ]

        for index, row in df.iterrows():

            if row.get(
                "No_Trade",
                False,
            ):

                lines.append(
                    "NO TRADE — no setup cleared "
                    "the V6.2 confidence/risk filters."
                )

                continue

            lines.extend(
                [

                    (
                        f"{index + 1}. "
                        f"{row['Ticker']} | "
                        f"Rs.{row['Price']}"
                    ),

                    (
                        "MODEL RETURN 1D/3D/5D: "
                        f"{row['Predicted_1D_Return_Pct']}% / "
                        f"{row['Predicted_3D_Return_Pct']}% / "
                        f"{row['Predicted_5D_Return_Pct']}%"
                    ),

                    (
                        "P(UP) 1D/3D/5D: "
                        f"{row['Probability_1D_Pct']}% / "
                        f"{row['Probability_3D_Pct']}% / "
                        f"{row['Probability_5D_Pct']}%"
                    ),

                    (
                        "SCORE: "
                        f"{row['Score_Pct']} pp | "
                        f"RSI {row['RSI']} | "
                        f"ATR {row['ATR_Pct']}% | "
                        f"Volume {row['Volume_Ratio']}x"
                    ),

                    (
                        f"ENTRY: Rs.{row['Entry_Low']}–"
                        f"{row['Entry_High']} | "
                        f"STOP: Rs.{row['Stop_Loss']}"
                    ),

                    (
                        f"TARGET 1: Rs.{row['Target_1']} | "
                        f"TARGET 2: Rs.{row['Target_2']}"
                    ),

                    (
                        f"R:R: {row['RR_T1']} / "
                        f"{row['RR_T2']}"
                    ),

                    (
                        "HOLD: approximately 1–3 sessions; "
                        "extend toward 5 only while the "
                        "risk structure remains valid"
                    ),

                    (
                        f"SIZE for ₹{self.cfg.capital:,.0f}: "
                        f"{int(row['Shares'])} shares | "
                        f"₹{row['Position_Value']:,.0f}"
                    ),

                    (
                        "MAX LOSS AT STOP: "
                        f"₹{row['Max_Loss_Rs']:,.0f}"
                    ),

                    (
                        f"TIER: {row['Tier']} | "
                        f"ACTION: {row['Action']}"
                    ),
                ]
            )

        correlation = df.attrs.get(
            "avg_pairwise_correlation"
        )

        if correlation is not None:

            warning = (
                " — HIGH CONCENTRATION"
                if df.attrs.get(
                    "high_concentration_warning",
                    False,
                )
                else ""
            )

            lines.append(
                f"Avg pick correlation: "
                f"{correlation}{warning}"
            )

        return lines

    # --------------------------------------------------------
    # SEND TELEGRAM
    # --------------------------------------------------------

    def send_telegram(
        self,
        regime,
        picks,
        ipos,
    ):

        if (
            not self.cfg.bot_token
            or not self.cfg.chat_id
        ):

            raise ValueError(
                "TELEGRAM_BOT_TOKEN or "
                "TELEGRAM_CHAT_ID is missing."
            )

        # IMPORTANT:
        # This is the corrected f-string.
        #
        # The previous version had nested quotation marks
        # inside strftime(). This version is syntactically safe.

        timestamp = now_ist().strftime(
            "%d %b %Y, %H:%M IST"
        )

        lines = [

            "*MULTI-FACTOR MARKET ALERT V6.2*",

            f"_{timestamp}_",

            "",

            (
                f"MARKET REGIME: "
                f"{regime['label']} | "
                f"Nifty: {regime['nifty']:.2f} | "
                f"SMA50: {regime['sma50']:.2f} | "
                f"VIX: {regime['vix']:.2f}"
            ),

            "",
        ]

        lines.extend(
            self.format_stock_section(
                picks
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
                    f"{ipo['name']} | "
                    f"{ipo['status']} | "
                    f"{ipo['start']} → "
                    f"{ipo['end']}"
                )

                if ipo.get(
                    "price"
                ):

                    lines.append(
                        f"PRICE: {ipo['price']}"
                    )

                if ipo.get(
                    "subscription"
                ):

                    lines.append(
                        "SUBSCRIPTION: "
                        f"{ipo['subscription']}"
                    )

                lines.append(
                    "GMP: "
                    f"{ipo.get('gmp_text', 'NOT VERIFIED')}"
                )

                lines.append(
                    "-> Verify issue price, dates and "
                    "offer documents before applying."
                )

        else:

            lines.append(
                "No current/upcoming IPO data "
                "was retrieved from the NSE source."
            )

        lines.extend(
            [
                "",
                (
                    "_V6.2 is a probabilistic, "
                    "risk-controlled screen. It does "
                    "not guarantee profit. Position "
                    "size uses the configured capital._"
                ),
            ]
        )

        message = "\n".join(
            lines
        )

        telegram_url = (
            "https://api.telegram.org/"
            f"bot{self.cfg.bot_token}/sendMessage"
        )

        response = requests.post(
            telegram_url,
            json={
                "chat_id":
                    self.cfg.chat_id,

                "text":
                    message,

                "parse_mode":
                    "Markdown",
            },
            timeout=20,
        )

        response.raise_for_status()


# ============================================================
# MARKET REGIME DATA
# ============================================================

def get_market_regime(
    cfg,
):

    data = download_ohlcv(
        [
            cfg.nifty_ticker,
            cfg.vix_ticker,
        ],
        period="1y",
    )

    market = (
        data["Close"][
            cfg.nifty_ticker
        ]
        .dropna()
    )

    if (
        cfg.vix_ticker
        in data["Close"].columns
    ):

        vix = (
            data["Close"][
                cfg.vix_ticker
            ]
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

def fetch_nse_ipos():

    url = (
        "https://www.nseindia.com/"
        "market-data/all-upcoming-issues-ipo"
    )

    headers = {

        "User-Agent":
            "Mozilla/5.0",

        "Accept":
            (
                "text/html,"
                "application/xhtml+xml,"
                "application/xml;q=0.9,"
                "*/*;q=0.8"
            ),

        "Referer":
            "https://www.nseindia.com/",
    }

    try:

        session = requests.Session()

        session.get(
            "https://www.nseindia.com/",
            headers=headers,
            timeout=10,
        )

        response = session.get(
            url,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        tables = pd.read_html(
            response.text
        )

        best = None

        for table in tables:

            columns = (
                " ".join(
                    map(
                        str,
                        table.columns,
                    )
                )
                .lower()
            )

            if (
                "company" in columns
                and "issue" in columns
                and (
                    "date" in columns
                    or "status" in columns
                )
            ):

                best = table

                break

        if (
            best is None
            or best.empty
        ):

            return []

        best.columns = [
            str(column).strip()
            for column
            in best.columns
        ]

        rows = []

        for _, row in best.iterrows():

            values = [
                str(value)
                for value
                in row.tolist()
            ]

            if not values:
                continue

            if all(
                value.lower() == "nan"
                for value in values
            ):

                continue

            rows.append(
                {
                    "name":
                        values[0],

                    "status":
                        "OPEN/UPCOMING",

                    "start":
                        (
                            values[1]
                            if len(values) > 1
                            else "?"
                        ),

                    "end":
                        (
                            values[2]
                            if len(values) > 2
                            else "?"
                        ),

                    "price":
                        None,

                    "subscription":
                        (
                            values[-1]
                            if len(values) > 3
                            else None
                        ),

                    "gmp_text":
                        "NOT VERIFIED",
                }
            )

        return rows[:12]

    except Exception as exc:

        log.warning(
            "NSE IPO retrieval failed: %s",
            exc,
        )

        return []


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

def run_backtest(
    cfg,
    period="5y",
):

    log.info(
        "Starting V6.2 walk-forward backtest."
    )

    universe = (
        load_nifty500_universe(
            cfg.fallback_universe
        )
    )

    tickers = list(
        dict.fromkeys(
            universe
            + [
                cfg.nifty_ticker,
                cfg.vix_ticker,
            ]
        )
    )

    data = download_ohlcv(
        tickers,
        period=period,
    )

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    market = (
        close[
            cfg.nifty_ticker
        ]
        .dropna()
    )

    if (
        cfg.vix_ticker
        in close.columns
    ):

        vix = (
            close[
                cfg.vix_ticker
            ]
            .dropna()
        )

    else:

        vix = pd.Series(
            dtype=float
        )

    frames = {}

    for ticker in universe:

        if not all(
            ticker in dataset.columns
            for dataset in (
                close,
                high,
                low,
                volume,
            )
        ):

            continue

        if (
            close[ticker]
            .dropna()
            .shape[0]
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

    common = sorted(
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

    if len(common) < 800:

        return (
            pd.DataFrame(),
            {"trades": 0},
        )

    test_index = pd.DatetimeIndex(
        common[
            550:-6
        ]
    )

    test_dates = test_index[::5]

    trades = []

    for current_date in test_dates:

        training_parts = []

        for frame in frames.values():

            historical = (
                frame[
                    frame.index
                    < current_date
                ]
                .tail(
                    cfg.model_lookback
                )
            )

            training_parts.append(
                historical
            )

        training = pd.concat(
            training_parts,
            ignore_index=True,
        )

        models = {}

        for horizon in (
            cfg.short_horizons
        ):

            models[horizon] = (
                fit_models(
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

            if (
                row[
                    FEATURES
                ]
                .isna()
                .any(axis=1)
                .iloc[0]
            ):

                continue

            x = row[
                FEATURES
            ]

            predictions = {}

            probabilities = {}

            for horizon in (
                cfg.short_horizons
            ):

                bundle = models.get(
                    horizon
                )

                if bundle is None:
                    continue

                (
                    return_model,
                    direction_model,
                    calibrator,
                ) = bundle

                predictions[horizon] = (
                    float(
                        return_model
                        .predict(x)[0]
                    )
                )

                probabilities[horizon] = (
                    calibrated_probability(
                        direction_model,
                        calibrator,
                        x,
                    )
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

            turnover = (
                price
                * float(
                    row[
                        "volume"
                    ].iloc[0]
                )
                / 1e7
            )

            if price < cfg.min_price:
                continue

            if (
                turnover
                < cfg.min_avg_turnover_cr
            ):

                continue

            ensemble_return = (
                0.20
                * predictions.get(
                    1,
                    predictions[3],
                )
                + 0.55
                * predictions[3]
                + 0.25
                * predictions.get(
                    5,
                    predictions[3],
                )
            )

            ensemble_probability = (
                0.20
                * probabilities.get(
                    1,
                    probabilities[3],
                )
                + 0.55
                * probabilities[3]
                + 0.25
                * probabilities.get(
                    5,
                    probabilities[3],
                )
            )

            predicted_pct = (
                ensemble_return * 100
            )

            if (
                predicted_pct
                < cfg.min_pred_return_pct
            ):

                continue

            if (
                ensemble_probability
                < cfg.min_probability
            ):

                continue

            score_pct = (
                predicted_pct
                * (
                    2
                    * ensemble_probability
                    - 1
                )
                - 0.25
                * atr_pct
            )

            rsi_value = float(
                row[
                    "rsi14"
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

            trend_aligned = (
                float(
                    row[
                        "dist_ema20"
                    ].iloc[0]
                )
                > 0
                and
                float(
                    row[
                        "dist_sma50"
                    ].iloc[0]
                )
                > 0
                and
                float(
                    row[
                        "ema20_slope10"
                    ].iloc[0]
                )
                > 0
            )

            if rsi_value > 70:

                score_pct -= 0.10

            if extension_atr > 1.25:

                score_pct -= 0.12

            score_pct += (
                0.08
                * np.clip(
                    float(
                        row[
                            "relative5"
                        ].iloc[0]
                    )
                    * 10,
                    -1,
                    1,
                )
            )

            score_pct += (
                0.06
                * np.clip(
                    (
                        float(
                            row[
                                "volume_ratio"
                            ].iloc[0]
                        )
                        - 1
                    )
                    / 1.5,
                    -1,
                    1,
                )
            )

            score_pct += (
                0.08
                if trend_aligned
                else -0.08
            )

            if (
                score_pct
                < cfg.min_score_pct
            ):

                continue

            candidates.append(
                (
                    score_pct,
                    ticker,
                    price,
                    ensemble_return,
                    ensemble_probability,
                )
            )

        if not candidates:
            continue

        candidates.sort(
            reverse=True
        )

        (
            score_pct,
            ticker,
            entry_price,
            predicted_return,
            probability,
        ) = candidates[0]

        series = (
            close[ticker]
            .dropna()
        )

        try:

            position_index = (
                series.index.get_loc(
                    current_date
                )
            )

        except KeyError:

            continue

        if (
            position_index + 3
            >= len(series)
        ):

            continue

        exit_price = float(
            series.iloc[
                position_index + 3
            ]
        )

        gross_return = (
            exit_price
            / entry_price
            - 1
        ) * 100

        total_cost = (
            2
            * (
                cfg.slippage_pct
                + cfg.brokerage_pct
            )
        )

        net_return = (
            gross_return
            - total_cost
        )

        trades.append(
            {
                "date":
                    current_date,

                "ticker":
                    ticker,

                "entry":
                    entry_price,

                "predicted_3d_pct":
                    predicted_return * 100,

                "probability_3d_pct":
                    probability * 100,

                "score_pct":
                    score_pct,

                "actual_3d_pct":
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
            {"trades": 0},
        )

    wins = (
        trades_df[
            "net_3d_pct"
        ]
        > 0
    )

    positive = (
        trades_df.loc[
            wins,
            "net_3d_pct"
        ]
    )

    negative = (
        trades_df.loc[
            ~wins,
            "net_3d_pct"
        ]
    )

    gross_profit = positive.sum()

    gross_loss = -negative.sum()

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            / gross_loss
        )

    else:

        profit_factor = None

    equity_curve = (
        1
        + trades_df[
            "net_3d_pct"
        ] / 100
    ).cumprod()

    running_peak = (
        equity_curve
        .cummax()
    )

    drawdown = (
        equity_curve
        / running_peak
        - 1
    ) * 100

    summary = {

        "trades":
            int(
                len(trades_df)
            ),

        "hit_rate_pct":
            round(
                wins.mean()
                * 100,
                1,
            ),

        "avg_net_3d_return_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].mean(),
                3,
            ),

        "median_net_3d_return_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].median(),
                3,
            ),

        "best_net_3d_return_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].max(),
                2,
            ),

        "worst_net_3d_return_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].min(),
                2,
            ),

        "profit_factor":
            (
                round(
                    profit_factor,
                    2,
                )
                if profit_factor is not None
                else None
            ),

        "max_drawdown_pct":
            round(
                drawdown.min(),
                2,
            ),

        "compound_return_pct":
            round(
                (
                    equity_curve.iloc[-1]
                    - 1
                )
                * 100,
                2,
            ),

        "expectancy_pct":
            round(
                trades_df[
                    "net_3d_pct"
                ].mean(),
                3,
            ),
    }

    return (
        trades_df,
        summary,
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Indian Market Engine V6.2"
        )
    )

    parser.add_argument(
        "--backtest",
        action="store_true",
        help=(
            "Run historical walk-forward "
            "backtest instead of Telegram alert."
        ),
    )

    parser.add_argument(
        "--backtest-period",
        default="5y",
        help=(
            "Historical period, e.g. 3y, 5y, 10y."
        ),
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=float(
            os.getenv(
                "CAPITAL",
                "100000",
            )
        ),
        help=(
            "Capital used for position sizing."
        ),
    )

    args = parser.parse_args()

    cfg = EngineConfig(

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

        capital=args.capital,
    )

    # ========================================================
    # BACKTEST MODE
    # ========================================================

    if args.backtest:

        log.info(
            "Running V6.2 backtest..."
        )

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
            "V6.2 BACKTEST SUMMARY"
        )
        print(
            "=========================================="
        )

        for key, value in summary.items():

            print(
                f"{key}: {value}"
            )

        trades.to_csv(
            cfg.backtest_path,
            index=False,
        )

        print()
        print(
            f"Detailed trades saved to: "
            f"{cfg.backtest_path}"
        )

        return

    # ========================================================
    # LIVE ALERT MODE
    # ========================================================

    engine = MarketEngineV62(
        cfg
    )

    regime = get_market_regime(
        cfg
    )

    log.info(
        "Market regime: %s",
        regime,
    )

    picks = (
        engine.get_stock_picks()
    )

    ipos = fetch_nse_ipos()

    engine.log_alert(
        picks,
        regime,
    )

    engine.send_telegram(
        regime,
        picks,
        ipos,
    )

    log.info(
        "V6.2 Telegram alert sent successfully."
    )


if __name__ == "__main__":
    main()
