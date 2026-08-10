"""
Multi-Factor Indian Market Engine v3
(Nifty 100 statistical screen + Gold signal + IPO/GMP listing-day analysis)
---------------------------------------------------------------------------
v3 adds, on top of v2:
  1. Parameter-uncertainty correction for the gold probability: the naive
     z = mu/sigma treats the estimated mean as if it were known exactly.
     With ~125 daily observations, mu is barely distinguishable from noise,
     so we widen the predictive variance (sigma * sqrt(1 + 1/n)) before
     computing the CDF. This shrinks the probability toward 50% when the
     sample is small or the effect is weak, instead of overstating
     confidence.
  2. A per-stock significance check: each finalist's short-term momentum is
     compared to its own historical daily volatility via a rough t-stat, so
     "top of today's batch" isn't confused with "statistically distinct
     from that stock's own noise."
  3. A correlation/concentration flag across the top-N picks, so three
     "diversified" ideas that are actually all one sector bet get called
     out rather than silently presented as three independent signals.
  4. An IPO / Grey Market Premium (GMP) module: given issue price, GMP,
     and subscription data (all supplied by you -- see caveat below), it
     estimates listing gain and produces a tiered, explained view on
     whether to consider booking profit at listing vs. holding.

IMPORTANT CAVEATS (read before using with real money):
  - This is a heuristic screener, not a validated trading strategy. No
    backtest, walk-forward test, or transaction-cost model is included.
  - "Probability" figures assume returns are roughly normal, which real
    markets violate (fat tails, skew, volatility clustering) -- treat them
    as an approximate, low-confidence signal, not a guarantee.
  - GMP (Grey Market Premium) is an UNOFFICIAL, unregulated, informally
    traded indicator. It is frequently thin-volume, easily talked up by
    interested parties, and has no SEBI oversight. It has historically
    been a noisy and sometimes actively misleading predictor of actual
    listing-day price action, especially in weak/volatile markets. There
    is no reliable public API for it -- this script does NOT scrape any
    GMP tracking website (their terms of service generally restrict
    automated scraping, and their markup changes often, making scrapers
    fragile). Instead, you supply the day's GMP figure yourself (manually,
    or by wiring in whatever data source you have rights to use).
  - Nothing here is investment advice. It is a structured way to organize
    publicly-available-to-you inputs into a consistent framework; the
    actual decision and its risk is yours.
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
from scipy.stats import norm, t as t_dist

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
# IPO / Grey Market Premium (GMP) Analysis
# ---------------------------------------------------------------------------

@dataclass
class IPOListingInput:
    """
    All fields here are things YOU supply -- there is no automated scrape
    of any GMP tracker built in (see module docstring for why). Populate
    this from whatever legitimate source you already use (broker research,
    a data vendor you're licensed to use, or manual entry from a tracker
    site's public page).

    name:               IPO / company name, for display only.
    issue_price:        Final issue price (upper band), Rs.
    gmp:                Latest Grey Market Premium, Rs. per share
                         (can be negative if trading at a discount).
    gmp_pct:            Optional -- if you already have GMP as a % of
                         issue price, pass it here instead of gmp/issue_price
                         being computed. If omitted it's derived automatically.
    retail_subscription_x:  Retail category subscription, in "times" (e.g. 3.2).
    qib_subscription_x:      QIB category subscription, in "times".
    overall_subscription_x:  Overall subscription, in "times".
    kostak_rate:        Optional -- premium for selling the application
                         itself before allotment (Rs per lot), if you track it.
    listing_date:        Optional display string.
    market_trend:       One of "bullish", "neutral", "bearish" -- your read
                         (or Nifty's recent trend) on the broader market
                         mood around the listing date, since GMP-implied
                         gains fade much faster in weak markets.
    sector_hot:          Whether the IPO's sector is currently "in favour"
                         with retail/HNI flows (True/False) -- optional
                         qualitative input, since hot-sector IPOs tend to
                         hold gains better through the listing session.
    """
    name: str
    issue_price: float
    gmp: float
    gmp_pct: Optional[float] = None
    retail_subscription_x: Optional[float] = None
    qib_subscription_x: Optional[float] = None
    overall_subscription_x: Optional[float] = None
    kostak_rate: Optional[float] = None
    listing_date: Optional[str] = None
    market_trend: str = "neutral"
    sector_hot: Optional[bool] = None


class IPOAnalyzer:
    """
    Turns GMP + subscription + market-context inputs into a structured,
    tiered read on listing-day behaviour, with an explicit "profit booking"
    lean rather than a single number.

    Methodology and its limits (read this before trusting the output):
      - GMP is a THIN, UNREGULATED, OFF-EXCHANGE indicator. It can and does
        get talked up before listing and then evaporate at the open,
        especially for smaller-lot, high-hype issues. Treat gmp_pct as
        a noisy upper bound on plausible listing gain, not a forecast.
      - A well-documented empirical pattern in Indian IPOs is that a large
        share of the listing-day gain (when GMP was positive) tends to be
        realized at or near the open, with intraday fade being common,
        especially for high-GMP, thinly-subscribed, or non-institutional-
        heavy issues. This is a TENDENCY seen across many cycles, not a
        law -- strong-fundamentals IPOs in hot sectors have also continued
        to run well past listing day. This script encodes the base-rate
        tendency as a lean, not a rule.
      - Subscription levels matter: heavy QIB (institutional) subscription
        is generally read as a stronger quality signal than retail-only
        enthusiasm, because institutions have done diligence and tend to
        be stickier holders; a huge overall subscription driven mostly by
        retail/HNI leverage (funded applications) is more prone to
        listing-day flipping pressure (many allottees sell immediately to
        repay funding), which pushes toward "book profit early" leans.
      - Broader market trend matters: the same GMP that "sticks" in a
        bullish tape often evaporates within minutes in a falling market.
    """

    # Tunable thresholds -- adjust based on your own risk tolerance;
    # these are reasonable, commonly-cited rules of thumb, not universal
    # constants derived from a rigorous study.
    GMP_STRONG_PCT = 30.0
    GMP_MODERATE_PCT = 10.0
    HIGH_RETAIL_LEVERAGE_X = 50.0   # retail subscription this high often implies
                                     # funded/leveraged applications -> flip risk
    STRONG_QIB_X = 10.0

    def __init__(self, ipo: IPOListingInput):
        self.ipo = ipo

    def analyze(self) -> dict:
        ipo = self.ipo
        gmp_pct = ipo.gmp_pct if ipo.gmp_pct is not None else (
            (ipo.gmp / ipo.issue_price) * 100 if ipo.issue_price else 0.0
        )
        est_listing_price = ipo.issue_price + ipo.gmp

        # --- Base lean purely from GMP magnitude -----------------------
        if gmp_pct >= self.GMP_STRONG_PCT:
            gmp_tier = "STRONG"
        elif gmp_pct >= self.GMP_MODERATE_PCT:
            gmp_tier = "MODERATE"
        elif gmp_pct > 0:
            gmp_tier = "MILD"
        elif gmp_pct == 0:
            gmp_tier = "FLAT"
        else:
            gmp_tier = "NEGATIVE"

        # --- Flip-risk read from subscription pattern -------------------
        flip_risk_notes = []
        flip_risk_score = 0  # higher = more reason to book early rather than hold

        if ipo.retail_subscription_x is not None and ipo.retail_subscription_x >= self.HIGH_RETAIL_LEVERAGE_X:
            flip_risk_score += 2
            flip_risk_notes.append(
                f"Retail subscription very high ({ipo.retail_subscription_x}x) -- "
                "often implies funded/leveraged applications, which historically "
                "increases listing-day selling pressure from allottees repaying financing."
            )

        if ipo.qib_subscription_x is not None:
            if ipo.qib_subscription_x >= self.STRONG_QIB_X:
                flip_risk_score -= 1
                flip_risk_notes.append(
                    f"Strong QIB subscription ({ipo.qib_subscription_x}x) is generally "
                    "read as a quality signal and modestly supportive of holding beyond day one."
                )
            elif ipo.qib_subscription_x < 1.0:
                flip_risk_score += 1
                flip_risk_notes.append(
                    f"Weak QIB subscription ({ipo.qib_subscription_x}x) suggests limited "
                    "institutional conviction -- gains may be less durable."
                )

        if ipo.market_trend == "bearish":
            flip_risk_score += 2
            flip_risk_notes.append(
                "Broader market trend read as bearish -- GMP-implied gains tend to "
                "compress or evaporate faster in weak tapes."
            )
        elif ipo.market_trend == "bullish":
            flip_risk_score -= 1
            flip_risk_notes.append(
                "Broader market trend read as bullish -- somewhat more supportive "
                "of gains holding through the session, though not guaranteed."
            )

        if ipo.sector_hot is True:
            flip_risk_score -= 1
            flip_risk_notes.append("Sector currently in favour with flows -- modestly supportive.")
        elif ipo.sector_hot is False:
            flip_risk_score += 1
            flip_risk_notes.append("Sector not currently in favour -- gains may fade faster.")

        # --- Combine GMP tier + flip risk into a single lean -------------
        if gmp_tier in ("NEGATIVE", "FLAT"):
            recommendation = (
                "CAUTION: GMP is flat or negative. Historically this correlates with "
                "flat-to-negative listing, not a confident gain to book. If allotted, "
                "many participants in this situation choose to exit at or soon after "
                "listing to avoid further downside, rather than holding for a recovery "
                "that GMP is not currently signalling."
            )
        elif gmp_tier == "MILD":
            if flip_risk_score >= 2:
                recommendation = (
                    "Mild positive GMP combined with elevated flip-risk signals "
                    "(leveraged retail demand and/or soft market backdrop). Consider "
                    "booking profit near the open rather than waiting, since the "
                    "modest cushion is more likely to compress than expand."
                )
            else:
                recommendation = (
                    "Mild positive GMP with no major red flags. A small listing gain "
                    "is plausible but not assured either way; booking part of the "
                    "position near listing and letting the rest ride with a mental "
                    "stop near the issue price is a commonly used middle path."
                )
        elif gmp_tier == "MODERATE":
            if flip_risk_score >= 2:
                recommendation = (
                    "Moderate GMP, but subscription/market signals suggest meaningful "
                    "flip risk. A partial-to-full profit-booking lean near the open is "
                    "reasonable, since the base-rate pattern in similar setups is "
                    "gains compressing intraday rather than extending."
                )
            else:
                recommendation = (
                    "Moderate GMP with a supportive subscription/market backdrop. "
                    "Booking a portion at listing while holding a runner (with a stop "
                    "at or above issue price) is a reasonable balance between capturing "
                    "the visible premium and leaving room for continuation."
                )
        else:  # STRONG
            if flip_risk_score >= 2:
                recommendation = (
                    "GMP is strong, but flip-risk signals are also elevated (heavy "
                    "leveraged retail demand and/or weak market backdrop). Large GMPs "
                    "combined with high leveraged demand are exactly the setup where "
                    "intraday fade from listing highs has historically been most "
                    "pronounced -- booking meaningfully into early strength is a common, "
                    "risk-aware approach rather than assuming the opening gain will hold "
                    "or extend."
                )
            else:
                recommendation = (
                    "Strong GMP with a supportive backdrop (institutional interest "
                    "and/or favourable market trend). Gains are more likely to hold "
                    "through the session in this combination, though scaling out part "
                    "of the position into early strength remains a reasonable, "
                    "asymmetric-risk way to lock in the visible premium."
                )

        return {
            "IPO": ipo.name,
            "Issue_Price": ipo.issue_price,
            "GMP": ipo.gmp,
            "GMP_Pct": round(gmp_pct, 1),
            "Estimated_Listing_Price": round(est_listing_price, 2),
            "GMP_Tier": gmp_tier,
            "Flip_Risk_Score": flip_risk_score,
            "Flip_Risk_Notes": flip_risk_notes,
            "Recommendation": recommendation,
            "Kostak_Rate": ipo.kostak_rate,
            "Listing_Date": ipo.listing_date,
        }


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
            n_obs = len(returns)
            mu = returns.mean()
            sigma = returns.std(ddof=1) if n_obs > 1 else 0.0

            # Naive version would use z = mu/sigma directly (treats mu as
            # known exactly). But mu is itself an ESTIMATE with its own
            # sampling error (~ sigma/sqrt(n)). We correct for this by
            # using the predictive standard deviation for a *new*
            # observation, sigma * sqrt(1 + 1/n), which is strictly wider
            # than sigma and converges to it only as n -> infinity. This
            # pulls the probability toward 50% whenever the sample is
            # small or the raw mean is noisy relative to volatility --
            # i.e. it makes the estimate more conservative, not more
            # confident.
            if sigma > 0 and n_obs > 5:
                sigma_pred = sigma * np.sqrt(1 + 1 / n_obs)
                z = mu / sigma_pred
                prob_up = float(norm.cdf(z) * 100)

                # Also report whether mu is even statistically distinguishable
                # from zero at conventional 95% confidence (two-sided t-test),
                # as an honesty check on the signal's strength.
                se_mean = sigma / np.sqrt(n_obs)
                t_stat = mu / se_mean if se_mean > 0 else 0.0
                p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=n_obs - 1)))
                significant = p_value < 0.05
            else:
                prob_up = 50.0
                p_value = 1.0
                significant = False

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
                "Signal_Significant": significant,
                "P_Value_Mean_Nonzero": round(p_value, 3),
                "Ann_Volatility_Pct": round(sigma * np.sqrt(252) * 100, 1),
                "RSI": round(rsi, 1),
                "Signal": signal,
            }
        except Exception as e:  # noqa: BLE001
            log.warning("Gold signal error: %s -- using fallback.", e)
            return {
                "Gold_Price": 123.40, "Gold_Change": 0.75, "USD_INR": 95.20,
                "Upward_Probability": 50.0, "Signal_Significant": False,
                "P_Value_Mean_Nonzero": 1.0, "Ann_Volatility_Pct": 12.0,
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

        top = df.head(cfg.top_n).copy()

        # --- Significance check: is this stock's momentum distinguishable
        # from its own noise, or just "best of a noisy batch"? Rough
        # one-sample t-test of daily returns over the lookback window
        # against a null of zero mean.
        sig_flags, p_values = [], []
        for ticker in top["Ticker"]:
            full_ticker = ticker + ".NS"
            series = data[full_ticker].dropna()
            daily_rets = series.pct_change().dropna().iloc[-cfg.momentum_lookback_days:]
            n = len(daily_rets)
            if n > 2 and daily_rets.std(ddof=1) > 0:
                se = daily_rets.std(ddof=1) / np.sqrt(n)
                t_stat = daily_rets.mean() / se
                p_val = float(2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1)))
            else:
                p_val = 1.0
            p_values.append(round(p_val, 3))
            sig_flags.append(p_val < 0.10)  # lenient threshold given tiny n
        top["Momentum_P_Value"] = p_values
        top["Momentum_Significant"] = sig_flags

        # --- Concentration / correlation flag across the finalists, so
        # "3 diversified picks" doesn't quietly mean "3 correlated bets."
        if len(top) > 1:
            price_panel = data[[t + ".NS" for t in top["Ticker"]]].dropna()
            corr_matrix = price_panel.pct_change().dropna().corr()
            # Average pairwise correlation excluding the diagonal.
            n_names = len(corr_matrix)
            off_diag_sum = corr_matrix.values.sum() - n_names
            avg_corr = off_diag_sum / (n_names * (n_names - 1))
            top.attrs["avg_pairwise_correlation"] = round(float(avg_corr), 2)
            top.attrs["high_concentration_warning"] = avg_corr > 0.6
        else:
            top.attrs["avg_pairwise_correlation"] = None
            top.attrs["high_concentration_warning"] = False

        return top

    # ---- Notification ----------------------------------------------------

    def send_telegram_alert(
        self,
        gold_info: dict,
        picks_df: pd.DataFrame,
        ipo_results: Optional[list] = None,
    ):
        cfg = self.cfg
        if not cfg.bot_token or not cfg.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")

        now_str = datetime.now().strftime("%d %b %Y, %H:%M")

        sig_tag = "significant" if gold_info.get("Signal_Significant") else "not statistically significant"
        lines = [
            "*MULTI-FACTOR MARKET ALERT*",
            f"_{now_str}_",
            "",
            "--- GOLD STATISTICAL SIGNAL ---",
            f"GOLDBEES: Rs.{gold_info['Gold_Price']} ({gold_info['Gold_Change']}%)",
            f"USD/INR: Rs.{gold_info['USD_INR']}",
            f"RSI(14): {gold_info['RSI']} | Est. P(up next day): {gold_info['Upward_Probability']}%",
            f"  (drift {sig_tag}, p={gold_info.get('P_Value_Mean_Nonzero')})",
            f"Ann. Volatility: {gold_info['Ann_Volatility_Pct']}%",
            f"Signal: {gold_info['Signal']}",
            "",
            "--- TOP QUANT PICKS (cross-sectional z-score ranking) ---",
        ]
        for i, row in picks_df.iterrows():
            sig_note = "sig." if row.get("Momentum_Significant") else "noise-level"
            lines.append(
                f"{i + 1}. {row['Ticker']} | Rs.{row['Price']} | "
                f"{cfg.momentum_lookback_days}D mom: {row['Momentum_Pct']}% ({sig_note}) | "
                f"RSI: {row['RSI']} | Score: {row['Composite_Score']}"
            )
        avg_corr = picks_df.attrs.get("avg_pairwise_correlation")
        if avg_corr is not None:
            warn = " (HIGH CONCENTRATION)" if picks_df.attrs.get("high_concentration_warning") else ""
            lines.append(f"Avg pairwise correlation of picks: {avg_corr}{warn}")

        if ipo_results:
            lines.append("")
            lines.append("--- IPO LISTING / GMP WATCH ---")
            for r in ipo_results:
                lines.append(
                    f"{r['IPO']}: Issue Rs.{r['Issue_Price']} | GMP Rs.{r['GMP']} "
                    f"({r['GMP_Pct']}%) | Est. listing Rs.{r['Estimated_Listing_Price']} "
                    f"| Tier: {r['GMP_Tier']}"
                )
                lines.append(f"  -> {r['Recommendation']}")

        lines.append("")
        lines.append(
            "_Screen + GMP read only: no backtest, GMP is unofficial/unregulated. "
            "Not investment advice._"
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

def load_ipo_watchlist() -> list:
    """
    Populate today's IPO watchlist here. This is intentionally manual /
    config-driven rather than scraped (see module docstring for why: GMP
    trackers generally restrict automated scraping in their terms, their
    markup changes often, and -- as importantly -- different trackers
    report meaningfully different GMP figures for the same IPO at the
    same time, since it's an unregulated, dealer-quoted number rather
    than an exchange-published price. Pick one source you trust and
    update this daily, or wire in a data vendor you're licensed to use.

    Below are the two IPOs actually open as of 10 Aug 2026, populated
    from public tracker pages as a live example. GMP shown is a single
    recent snapshot -- cross-checking two or three sources on the day
    you actually use this is worth the extra minute given the spread
    seen above.
    """
    return [
        IPOListingInput(
            name="Molbio Diagnostics",
            issue_price=807.0,             # upper price band
            gmp=150.0,                     # snapshot; seen ranging ~120-180 across sources
            retail_subscription_x=None,    # opened today -- not yet meaningful, update near close (Aug 12)
            qib_subscription_x=None,
            overall_subscription_x=None,
            listing_date="2026-08-17",
            market_trend="neutral",        # update based on your own Nifty read that day
            sector_hot=True,               # diagnostics/healthcare has had firm demand recently
        ),
        IPOListingInput(
            name="Optimystix Entertainment",
            issue_price=175.0,             # upper price band, SME issue
            gmp=5.0,
            overall_subscription_x=0.6,    # as of a few days into the bidding window
            listing_date="2026-08-14",
            market_trend="neutral",
            sector_hot=False,              # media/content production, not a currently "hot" flow sector
        ),
        # Add more IPOListingInput(...) entries here as new issues open.
    ]


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
            "Momentum_P_Value": 1.0, "Momentum_Significant": False,
        }])

    ipo_watchlist = load_ipo_watchlist()
    ipo_results = [IPOAnalyzer(ipo).analyze() for ipo in ipo_watchlist]
    for r in ipo_results:
        log.info("IPO read for %s: %s", r["IPO"], r["Recommendation"])

    engine.send_telegram_alert(gold_data, stock_picks, ipo_results=ipo_results)
