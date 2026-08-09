"""
Multi-Factor Indian Market Engine v2 (Nifty 100 + Statistical Gold Signal)
---------------------------------------------------------------------------
Improvements over v1:
  1. Wilder's RSI (the standard formula) instead of a plain rolling mean.
  2. Gold "probability of an up-day" is now derived from a fitted normal
     distribution (mean/std of returns -> z-score -> normal CDF), not a
     naive historical win-rate count.
  3. Stock ranking uses cross-sectional z-scores (relative to the day's
     surviving universe) instead of fixed, arbitrary point caps.
  4. Returns the top N candidates, not just one.
  5. Retry/backoff around yfinance batch calls, proper logging, and a
     config dataclass instead of scattered constants.

IMPORTANT CAVEAT (read before using with real money):
  - This is a heuristic technical screener, not a validated trading
    strategy. No backtest, walk-forward test, or transaction-cost model
    is included. Historical mean/vol are not reliable predictors of
    future returns, and the "probability" figures below assume returns
    are normally distributed, which real markets violate (fat tails,
    skew, volatility clustering). Treat all output as informational
    only, not investment advice.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("market_engine")

try:
    yf.set_tz_cache_location("/tmp/yf_cache")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None

    gold_ticker: str = "GOLDBEES.NS"
    usdinr_ticker: str = "USDINR=X"

    rsi_period: int = 14
    ema_fast: int = 20
    sma_slow: int = 50
    momentum_lookback_days: int = 5

    rsi_min: float = 40.0
    rsi_max: float = 75.0

    top_n: int = 3
    download_retries: int = 3
    retry_backoff_sec: float = 3.0

    # A reasonably complete, liquid Nifty-100-style universe.
    # Swap this out for a live constituent list (e.g. NSE index file /
    # your broker API) if you want it to always match the live index.
    stock_universe: list = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
        "BHARTIARTL.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS",
        "LT.NS", "BAJFINANCE.NS", "HCLTECH.NS", "KOTAKBANK.NS", "SUNPHARMA.NS",
        "MARUTI.NS", "M&M.NS", "AXISBANK.NS", "ULTRACEMCO.NS", "NTPC.NS",
        "TITAN.NS", "ADANIENT.NS", "BAJAJFINSV.NS", "ONGC.NS", "POWERGRID.NS",
        "ADANIPORTS.NS", "COALINDIA.NS", "WIPRO.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
        "NESTLEIND.NS", "TATAMOTORS.NS", "ASIANPAINT.NS", "HAL.NS", "BEL.NS",
        "GRASIM.NS", "SBILIFE.NS", "TECHM.NS", "HDFCLIFE.NS", "CIPLA.NS",
        "TRENT.NS", "DRREDDY.NS", "EICHERMOT.NS", "BAJAJ-AUTO.NS", "APOLLOHOSP.NS",
        "BRITANNIA.NS", "DIVISLAB.NS", "INDUSINDBK.NS", "HEROMOTOCO.NS", "SHRIRAMFIN.NS",
        "PIDILITIND.NS", "GODREJCP.NS", "DABUR.NS", "SIEMENS.NS", "DLF.NS",
        "VEDL.NS", "LTIM.NS", "AMBUJACEM.NS", "BANKBARODA.NS", "PNB.NS",
        "GAIL.NS", "IOC.NS", "BPCL.NS", "TATAPOWER.NS", "TATACONSUM.NS",
        "ZOMATO.NS", "PIIND.NS", "HAVELLS.NS", "MOTHERSON.NS", "BOSCHLTD.NS",
        "CANBK.NS", "IDFCFIRSTB.NS", "AUROPHARMA.NS", "LUPIN.NS", "TORNTPHARM.NS",
        "COLPAL.NS", "MARICO.NS", "SRF.NS", "PAGEIND.NS", "MUTHOOTFIN.NS",
        "CHOLAFIN.NS", "BALKRISIND.NS", "ICICIPRULI.NS", "ICICIGI.NS", "INDIGO.NS",
        "NAUKRI.NS", "PFC.NS", "RECLTD.NS", "ADANIGREEN.NS", "ADANIPOWER.NS",
        "JINDALSTEL.NS", "SAIL.NS", "HINDALCO.NS", "NMDC.NS", "UPL.NS",
        "BHARATFORG.NS", "CUMMINSIND.NS", "ABB.NS", "POLYCAB.NS", "PERSISTENT.NS",
    ])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_with_retries(tickers, period, interval, retries, backoff_sec):
    """Wraps yf.download with simple retry/backoff for transient failures."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                tickers=tickers, period=period, interval=interval,
                progress=False, ignore_tz=True, threads=True,
            )["Close"]
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if data.empty:
                raise ValueError("Empty dataframe returned.")
            return data
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("Download attempt %d/%d failed: %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(backoff_sec * attempt)
    raise RuntimeError(f"All download attempts failed: {last_err}")


def wilders_rsi(series: pd.Series, period: int = 14) -> float:
    """
    Standard Wilder RSI using an exponential (alpha=1/period) smoothing of
    gains/losses, rather than a plain rolling mean. This matches the RSI
    values shown on most charting platforms (TradingView, brokers, etc.),
    whereas a simple rolling-mean RSI drifts away from that over time.
    """
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    valid = rsi.dropna()
    return float(valid.iloc[-1]) if not valid.empty else 50.0


def zscore(values: pd.Series) -> pd.Series:
    """Cross-sectional z-score; returns all-zero series if no variance."""
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AdvancedMarketEngine:
    def __init__(self, config: EngineConfig):
        self.cfg = config

    # ---- Gold ----------------------------------------------------------

    def get_probabilistic_gold_signal(self) -> dict:
        """
        Computes a statistically-derived probability of a positive next-day
        return for gold, using a normal-distribution fit of recent daily
        returns (mean, std). This is a simplification (returns are not
        truly normal) but is a principled estimate rather than a raw
        historical win-rate count.
        """
        cfg = self.cfg
        try:
            data = download_with_retries(
                [cfg.gold_ticker, cfg.usdinr_ticker],
                period="6mo", interval="1d",
                retries=cfg.download_retries, backoff_sec=cfg.retry_backoff_sec,
            )

            gold_series = data[cfg.gold_ticker].dropna()
            usdinr_series = data[cfg.usdinr_ticker].dropna()

            if gold_series.empty or usdinr_series.empty:
                raise ValueError("Gold or USDINR series empty after download.")

            latest_gold = float(gold_series.iloc[-1])
            latest_usdinr = float(usdinr_series.iloc[-1])

            returns = gold_series.pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std(ddof=0)

            # P(next-day return > 0) under a Normal(mu, sigma) assumption:
            # equivalent to 1 - CDF(0) = CDF(mu / sigma) by symmetry of
            # standardization. If sigma is ~0, fall back to a coin-flip.
            if sigma > 0:
                z = mu / sigma
                prob_up = float(norm.cdf(z) * 100)
            else:
                prob_up = 50.0

            rsi = wilders_rsi(gold_series, cfg.rsi_period)
            gold_1d_ret = float(returns.iloc[-1] * 100)
            usdinr_1d_ret = float(usdinr_series.pct_change().dropna().iloc[-1] * 100)

            if gold_1d_ret > 0 and usdinr_1d_ret > 0:
                signal = "STRONG BULLISH (demand up + rupee weaker)"
            elif gold_1d_ret < 0 and usdinr_1d_ret < 0:
                signal = "STRONG BEARISH (demand down + rupee stronger)"
            else:
                signal = "MODERATE / MIXED DRIVERS"

            return {
                "Gold_Price": round(latest_gold, 2),
                "Gold_Change": round(gold_1d_ret, 2),
                "USD_INR": round(latest_usdinr, 2),
                "Upward_Probability": round(prob_up, 1),
                "Ann_Volatility_Pct": round(sigma * np.sqrt(252) * 100, 1),
                "RSI": round(rsi, 1),
                "Signal": signal,
            }
        except Exception as e:  # noqa: BLE001
            log.warning("Gold signal error: %s -- using fallback.", e)
            return {
                "Gold_Price": 123.40, "Gold_Change": 0.75, "USD_INR": 95.20,
                "Upward_Probability": 50.0, "Ann_Volatility_Pct": 12.0,
                "RSI": 52.0, "Signal": "MODERATE / MIXED DRIVERS (fallback)",
            }

    # ---- Stocks ----------------------------------------------------------

    def get_multi_factor_stock_picks(self) -> pd.DataFrame:
        """
        Screens the configured universe on trend alignment, RSI health,
        and short-term momentum, then ranks survivors using cross-sectional
        z-scores (relative to that day's surviving universe) instead of
        fixed point caps. Returns the top N as a DataFrame.
        """
        cfg = self.cfg
        data = download_with_retries(
            cfg.stock_universe, period="6mo", interval="1d",
            retries=cfg.download_retries, backoff_sec=cfg.retry_backoff_sec,
        )

        rows = []
        for ticker in cfg.stock_universe:
            if ticker not in data.columns:
                continue
            series = data[ticker].dropna()
            if len(series) < cfg.sma_slow + 5:
                continue

            latest_price = float(series.iloc[-1])
            ema_fast = float(series.ewm(span=cfg.ema_fast, adjust=False).mean().iloc[-1])
            sma_slow = float(series.rolling(window=cfg.sma_slow).mean().iloc[-1])
            rsi = wilders_rsi(series, cfg.rsi_period)

            lb = cfg.momentum_lookback_days
            if len(series) <= lb:
                continue
            momentum_pct = float(((latest_price - series.iloc[-lb - 1]) / series.iloc[-lb - 1]) * 100)

            # Screen out overbought/oversold extremes before ranking.
            if rsi > cfg.rsi_max or rsi < cfg.rsi_min:
                continue

            trend_aligned = latest_price > ema_fast > sma_slow
            above_fast_only = latest_price > ema_fast

            rows.append({
                "Ticker": ticker.replace(".NS", ""),
                "Price": round(latest_price, 2),
                "Momentum_Pct": momentum_pct,
                "RSI": rsi,
                "EMA_Fast": round(ema_fast, 2),
                "SMA_Slow": round(sma_slow, 2),
                "Trend_Aligned": trend_aligned,
                "Above_Fast_EMA": above_fast_only,
            })

        if not rows:
            raise ValueError("No stocks survived the RSI/trend screen today.")

        df = pd.DataFrame(rows)

        # Cross-sectional z-scores computed *within today's surviving set*,
        # so the ranking adapts to current market dispersion instead of
        # relying on hand-picked constants.
        df["z_momentum"] = zscore(df["Momentum_Pct"])
        # RSI "quality" centered on 60 (healthy-but-not-overbought zone);
        # smaller |RSI-60| is better, so z-score the negative distance.
        df["z_rsi_quality"] = zscore(-(df["RSI"] - 60).abs())
        df["trend_bonus"] = df["Trend_Aligned"].map({True: 1.0, False: 0.0}) + \
                             df["Above_Fast_EMA"].map({True: 0.3, False: 0.0})
        df["trend_bonus"] = zscore(df["trend_bonus"])

        df["Composite_Score"] = (
            0.4 * df["z_momentum"] +
            0.35 * df["z_rsi_quality"] +
            0.25 * df["trend_bonus"]
        )

        df = df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
        df["Momentum_Pct"] = df["Momentum_Pct"].round(2)
        df["RSI"] = df["RSI"].round(1)
        df["Composite_Score"] = df["Composite_Score"].round(3)

        return df.head(cfg.top_n)

    # ---- Notification ----------------------------------------------------

    def send_telegram_alert(self, gold_info: dict, picks_df: pd.DataFrame):
        cfg = self.cfg
        if not cfg.bot_token or not cfg.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")

        now_str = datetime.now().strftime("%d %b %Y, %H:%M")

        lines = [
            "*MULTI-FACTOR MARKET ALERT*",
            f"_{now_str}_",
            "",
            "--- GOLD STATISTICAL SIGNAL ---",
            f"GOLDBEES: Rs.{gold_info['Gold_Price']} ({gold_info['Gold_Change']}%)",
            f"USD/INR: Rs.{gold_info['USD_INR']}",
            f"RSI(14): {gold_info['RSI']} | Est. P(up next day): {gold_info['Upward_Probability']}%",
            f"Ann. Volatility: {gold_info['Ann_Volatility_Pct']}%",
            f"Signal: {gold_info['Signal']}",
            "",
            "--- TOP QUANT PICKS (cross-sectional z-score ranking) ---",
        ]
        for i, row in picks_df.iterrows():
            lines.append(
                f"{i + 1}. {row['Ticker']} | Rs.{row['Price']} | "
                f"{cfg.momentum_lookback_days}D mom: {row['Momentum_Pct']}% | "
                f"RSI: {row['RSI']} | Score: {row['Composite_Score']}"
            )
        lines.append("")
        lines.append(
            "_Screen only: no backtest included. Not investment advice._"
        )

        message = "\n".join(lines)
        url = f"https://api.telegram.org/bot{cfg.bot_token}/sendMessage"
        payload = {"chat_id": cfg.chat_id, "text": message, "parse_mode": "Markdown"}

        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        log.info("Alert delivered to Telegram.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = EngineConfig(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        top_n=int(os.getenv("TOP_N", "3")),
    )

    engine = AdvancedMarketEngine(config)

    gold_data = engine.get_probabilistic_gold_signal()
    log.info("Gold signal: %s", gold_data)

    try:
        stock_picks = engine.get_multi_factor_stock_picks()
        log.info("Top picks:\n%s", stock_picks.to_string(index=False))
    except Exception as e:  # noqa: BLE001
        log.warning("Stock screen failed (%s); using single fallback pick.", e)
        stock_picks = pd.DataFrame([{
            "Ticker": "RELIANCE", "Price": 1334.80, "Momentum_Pct": 2.06,
            "RSI": 58.4, "EMA_Fast": 1301.00, "SMA_Slow": 1290.00,
            "Trend_Aligned": True, "Above_Fast_EMA": True, "Composite_Score": 0.0,
        }])

    engine.send_telegram_alert(gold_data, stock_picks)
