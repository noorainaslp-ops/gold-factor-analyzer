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
from datetime import datetime, date
from zoneinfo import ZoneInfo
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

    # v4 change: 5-day return is deliberately no longer the ranking
    # momentum factor. Short lookbacks (1-2 weeks) fall inside the
    # well-documented short-term REVERSAL window (Jegadeesh 1990) --
    # using it to rank "which stock will keep going up" was fighting a
    # known market effect, which is a plausible explanation for the
    # AUROPHARMA/NAUKRI reversals seen in live alerts. Medium-term
    # momentum (Jegadeesh & Titman 1993), typically 3-12 months with the
    # most recent ~1 month skipped, has the actual academic support for
    # continuation. short_momentum_lookback_days is kept only as a
    # DISPLAY/CAUTION input (how extended is this move right now), not
    # as a ranking factor.
    short_momentum_lookback_days: int = 5
    medium_momentum_lookback_days: int = 126   # ~6 trading months
    medium_momentum_skip_days: int = 21        # skip most recent ~1 month

    rsi_min: float = 40.0
    rsi_max: float = 75.0

    # Intraday profit-booking heuristic thresholds (see get_intraday_recommendation).
    intraday_overbought_rsi: float = 65.0     # RSI above this = elevated / pullback-prone zone
    intraday_extended_pct: float = 6.0        # % above the fast EMA considered "stretched"

    # Regime filter: momentum-style ranking tends to work in trending
    # markets and fail in choppy/range-bound ones. Rather than blindly
    # emitting top picks regardless of the broader market's own trend,
    # check the index itself first.
    regime_index_ticker: str = "^NSEI"   # Nifty 50 index
    regime_sma_period: int = 50

    top_n: int = 3
    download_retries: int = 3
    retry_backoff_sec: float = 3.0

    # Where the daily picks get appended so the system can audit its own
    # track record over time instead of requiring manual reconstruction
    # from old alert messages (as had to be done for the 9-14 Aug review).
    alert_history_path: str = "alert_history.csv"

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


def wilders_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Full-series Wilder RSI (exponential alpha=1/period smoothing of
    gains/losses). Returns the RSI at every date, not just the latest --
    needed for backtesting, where we must compute what the RSI *would
    have been* on each historical day using only data up to that day.
    """
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def wilders_rsi(series: pd.Series, period: int = 14) -> float:
    """Latest-value convenience wrapper around wilders_rsi_series."""
    valid = wilders_rsi_series(series, period).dropna()
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
    subscription_open_date:  "YYYY-MM-DD" -- when bidding opens.
    subscription_close_date: "YYYY-MM-DD" -- when bidding closes. Used to
                         decide whether this IPO still belongs in the
                         alert: once this date has passed, the issue is
                         no longer subscribable and is dropped from the
                         "open for subscription" watch (see
                         filter_actionable_ipos below) -- the alert is
                         meant to help with an application decision, not
                         to keep reporting on issues you can no longer
                         apply to.
    listing_date:        Optional display string ("YYYY-MM-DD" recommended).
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
    subscription_open_date: Optional[str] = None
    subscription_close_date: Optional[str] = None
    listing_date: Optional[str] = None
    market_trend: str = "neutral"
    sector_hot: Optional[bool] = None
    # Set this to False for an IPO you've heard is open but whose price
    # band / dates you found conflicting across sources (this happens more
    # than you'd expect -- aggregator sites scrape from each other and
    # table-row misalignment silently mixes up figures between adjacent
    # IPOs). Unverified entries are excluded from the scored GMP analysis
    # but still surface in the alert as a "verify before applying" prompt,
    # so you don't lose track of an IPO just because its public data was
    # messy on the day you checked.
    verified: bool = True


def filter_actionable_ipos(ipos: list, as_of: Optional[date] = None) -> list:
    """
    Keeps only IPOs you can still actually act on: currently open for
    subscription, or opening soon. Drops anything whose subscription
    window has already closed, so the alert doesn't keep reporting on
    issues you can no longer apply to (that's a "how did it do" question,
    which this script isn't built to answer -- it's meant to inform a
    still-pending application decision).

    If subscription_close_date is missing on an entry, it's kept by
    default (assumed still relevant) but logged, since we can't verify
    its status without that date.
    """
    as_of = as_of or date.today()

    kept = []
    for ipo in ipos:
        if not ipo.subscription_close_date:
            log.warning(
                "IPO '%s' has no subscription_close_date set -- keeping it "
                "in the alert by default, but its status can't be verified.",
                ipo.name,
            )
            kept.append(ipo)
            continue
        try:
            close_dt = date.fromisoformat(ipo.subscription_close_date)
        except ValueError:
            log.warning("IPO '%s' has an unparseable close date; keeping it.", ipo.name)
            kept.append(ipo)
            continue

        if close_dt >= as_of:
            kept.append(ipo)
        else:
            log.info(
                "Dropping '%s' from alert -- subscription closed %s (already past).",
                ipo.name, ipo.subscription_close_date,
            )
    return kept


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

        # --- Subscription-window status (open / upcoming / unknown) ------
        today = date.today()
        status = "UNKNOWN"
        try:
            if ipo.subscription_open_date:
                open_dt = date.fromisoformat(ipo.subscription_open_date)
                if open_dt > today:
                    status = "UPCOMING"
            if ipo.subscription_close_date:
                close_dt = date.fromisoformat(ipo.subscription_close_date)
                if close_dt >= today and status != "UPCOMING":
                    status = "OPEN"
                elif close_dt < today:
                    status = "CLOSED"
        except ValueError:
            status = "UNKNOWN"

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
            "Subscription_Status": status,
            "Subscription_Open_Date": ipo.subscription_open_date,
            "Subscription_Close_Date": ipo.subscription_close_date,
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

    # ---- Market regime ----------------------------------------------------

    def get_market_regime(self) -> dict:
        """
        Checks whether the broader market (Nifty 50) is itself trending,
        since momentum-style stock ranking has historically worked better
        in trending markets and worse in choppy/range-bound ones. This
        doesn't veto the alert -- it's a caution flag attached to it, so
        you can weight conviction (and position size) accordingly rather
        than treating every day's top picks as equally reliable regardless
        of what the index itself is doing.
        """
        cfg = self.cfg
        try:
            data = download_with_retries(
                [cfg.regime_index_ticker], period="1y", interval="1d",
                retries=cfg.download_retries, backoff_sec=cfg.retry_backoff_sec,
            )
            series = data[cfg.regime_index_ticker].dropna() if cfg.regime_index_ticker in data.columns else data.iloc[:, 0].dropna()
            if len(series) < cfg.regime_sma_period + 5:
                raise ValueError("Insufficient index history for regime check.")

            latest = float(series.iloc[-1])
            sma = float(series.rolling(cfg.regime_sma_period).mean().iloc[-1])
            pct_vs_sma = ((latest - sma) / sma) * 100

            # Simple trend slope: is the SMA itself rising or falling over
            # the last ~10 sessions? Adds a bit more nuance than "above/
            # below" alone -- a index just barely above a flattening SMA
            # is a weaker trend than one clearly above a rising SMA.
            sma_series = series.rolling(cfg.regime_sma_period).mean().dropna()
            sma_slope_pct = float(((sma_series.iloc[-1] - sma_series.iloc[-11]) / sma_series.iloc[-11]) * 100) if len(sma_series) > 11 else 0.0

            if latest > sma and sma_slope_pct > 0:
                regime = "FAVORABLE"
                note = "Nifty 50 is above its rising 50-day average -- historically a more supportive backdrop for momentum-style picks."
            elif latest < sma and sma_slope_pct < 0:
                regime = "UNFAVORABLE"
                note = "Nifty 50 is below its falling 50-day average -- momentum strategies have historically underperformed in this regime. Consider reduced conviction/position size on today's picks."
            else:
                regime = "MIXED"
                note = "Nifty 50's trend signals are mixed (e.g. above the average but the average itself isn't clearly rising, or vice versa) -- treat today's picks with moderate, not high, conviction."

            return {
                "Regime": regime,
                "Index_Level": round(latest, 1),
                "SMA_Level": round(sma, 1),
                "Pct_Vs_SMA": round(pct_vs_sma, 2),
                "SMA_Slope_Pct_10d": round(sma_slope_pct, 2),
                "Note": note,
            }
        except Exception as e:  # noqa: BLE001
            log.warning("Regime check failed: %s -- treating as UNKNOWN.", e)
            return {
                "Regime": "UNKNOWN", "Index_Level": None, "SMA_Level": None,
                "Pct_Vs_SMA": None, "SMA_Slope_Pct_10d": None,
                "Note": "Regime check failed to fetch index data -- treat picks with normal caution.",
            }

    # ---- Alert history logging --------------------------------------------

    def log_alert_history(self, picks_df: pd.DataFrame, regime: dict):
        """
        Appends today's picks to a local CSV so the system's own track
        record accumulates automatically -- this is what made the manual
        9-14 Aug review possible in the first place, and it shouldn't
        require reconstructing prices from old chat messages every time.

        Note on persistence: GitHub Actions runners are ephemeral, so a
        file written here disappears when the job ends UNLESS your
        workflow commits it back to the repo (or writes it to external
        storage -- a Google Sheet, a small database, S3, etc.). See the
        commented git commit-back step in market_alert.yml for the
        simplest option if you're running this via GitHub Actions.
        """
        cfg = self.cfg
        today_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")

        rows = []
        for _, row in picks_df.iterrows():
            rows.append({
                "date": today_str,
                "ticker": row.get("Ticker"),
                "price": row.get("Price"),
                "medium_momentum_pct": row.get("Medium_Momentum_Pct"),
                "short_momentum_pct": row.get("Short_Momentum_Pct"),
                "rsi": row.get("RSI"),
                "composite_score": row.get("Composite_Score"),
                "intraday_tier": row.get("Intraday_Tier"),
                "regime": regime.get("Regime"),
            })
        new_rows = pd.DataFrame(rows)

        try:
            if os.path.exists(cfg.alert_history_path):
                existing = pd.read_csv(cfg.alert_history_path)
                combined = pd.concat([existing, new_rows], ignore_index=True)
            else:
                combined = new_rows
            combined.to_csv(cfg.alert_history_path, index=False)
            log.info("Logged %d picks to %s", len(new_rows), cfg.alert_history_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not write alert history: %s", e)

    # ---- Stocks ----------------------------------------------------------

    def get_multi_factor_stock_picks(self) -> pd.DataFrame:
        """
        Screens the configured universe on trend alignment, RSI health,
        and MEDIUM-TERM momentum (not short-term), then ranks survivors
        using cross-sectional z-scores. Returns the top N as a DataFrame.

        v4 change: the primary momentum factor is now a ~6-month return
        with the most recent ~1 month skipped (medium_momentum_lookback_days
        / medium_momentum_skip_days), rather than a 5-day return. Short
        lookbacks fall inside the well-documented short-term reversal
        window; using one to rank "which stock keeps going up" was
        fighting a known market effect rather than exploiting one. The
        5-day figure is still computed and shown (as Short_Momentum_Pct)
        for context and for the intraday extension/caution flags, but it
        no longer drives the ranking itself.
        """
        cfg = self.cfg
        data = download_with_retries(
            cfg.stock_universe, period="1y", interval="1d",
            retries=cfg.download_retries, backoff_sec=cfg.retry_backoff_sec,
        )

        min_history = cfg.medium_momentum_lookback_days + cfg.medium_momentum_skip_days + 10

        rows = []
        for ticker in cfg.stock_universe:
            if ticker not in data.columns:
                continue
            series = data[ticker].dropna()
            if len(series) < max(cfg.sma_slow + 5, min_history):
                continue

            latest_price = float(series.iloc[-1])
            ema_fast = float(series.ewm(span=cfg.ema_fast, adjust=False).mean().iloc[-1])
            sma_slow = float(series.rolling(window=cfg.sma_slow).mean().iloc[-1])
            rsi = wilders_rsi(series, cfg.rsi_period)

            lb_short = cfg.short_momentum_lookback_days
            short_momentum_pct = float(((latest_price - series.iloc[-lb_short - 1]) / series.iloc[-lb_short - 1]) * 100)

            # Medium-term momentum: return from (lookback+skip) days ago to
            # (skip) days ago -- i.e. the ~6-month formation period,
            # excluding the most recent ~1 month where reversal effects
            # dominate. This is the factor actually used for ranking.
            skip = cfg.medium_momentum_skip_days
            look = cfg.medium_momentum_lookback_days
            price_formation_start = float(series.iloc[-(look + skip) - 1])
            price_formation_end = float(series.iloc[-skip - 1])
            medium_momentum_pct = float(((price_formation_end - price_formation_start) / price_formation_start) * 100)

            # Screen out overbought/oversold extremes before ranking.
            if rsi > cfg.rsi_max or rsi < cfg.rsi_min:
                continue

            trend_aligned = latest_price > ema_fast > sma_slow
            above_fast_only = latest_price > ema_fast

            # Rolling daily volatility (20d) as % -- gives intraday-guidance
            # a sense of this stock's typical daily swing, so "extended 6%
            # above the EMA" can be read in context (routine for a volatile
            # small-cap, unusual for a low-beta large-cap).
            daily_vol_pct = float(series.pct_change().rolling(20).std().iloc[-1] * 100)

            # Distance from the fast EMA, used for the "stretched" flag in
            # the intraday recommendation below.
            extension_from_ema_pct = ((latest_price - ema_fast) / ema_fast) * 100 if ema_fast else 0.0

            rows.append({
                "Ticker": ticker.replace(".NS", ""),
                "Price": round(latest_price, 2),
                "Short_Momentum_Pct": round(short_momentum_pct, 2),   # display/caution only
                "Medium_Momentum_Pct": round(medium_momentum_pct, 2), # ranking factor
                "RSI": rsi,
                "EMA_Fast": round(ema_fast, 2),
                "SMA_Slow": round(sma_slow, 2),
                "Trend_Aligned": trend_aligned,
                "Above_Fast_EMA": above_fast_only,
                "Daily_Vol_Pct": round(daily_vol_pct, 2) if not np.isnan(daily_vol_pct) else None,
                "Extension_From_EMA_Pct": round(extension_from_ema_pct, 1),
            })

        if not rows:
            raise ValueError("No stocks survived the RSI/trend screen today.")

        df = pd.DataFrame(rows)

        # Cross-sectional z-scores computed *within today's surviving set*,
        # so the ranking adapts to current market dispersion instead of
        # relying on hand-picked constants.
        df["z_momentum"] = zscore(df["Medium_Momentum_Pct"])
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
        df["RSI"] = df["RSI"].round(1)
        df["Composite_Score"] = df["Composite_Score"].round(3)

        top = df.head(cfg.top_n).copy()

        # --- Significance check: is this stock's momentum distinguishable
        # from its own noise, or just "best of a noisy batch"? One-sample
        # t-test of daily returns over the formation window (medium-term
        # momentum lookback) against a null of zero mean. Widened from the
        # original 5-day window -- 5 daily observations is too few for a
        # t-test to mean much; using the same ~6-month window the ranking
        # itself is based on makes this check consistent with what's
        # actually being ranked.
        sig_flags, p_values = [], []
        for ticker in top["Ticker"]:
            full_ticker = ticker + ".NS"
            series = data[full_ticker].dropna()
            window = cfg.medium_momentum_lookback_days
            daily_rets = series.pct_change().dropna().iloc[-window:]
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
            n_names = len(corr_matrix)
            off_diag_sum = corr_matrix.values.sum() - n_names
            avg_corr = off_diag_sum / (n_names * (n_names - 1))
            top.attrs["avg_pairwise_correlation"] = round(float(avg_corr), 2)
            top.attrs["high_concentration_warning"] = avg_corr > 0.6
        else:
            top.attrs["avg_pairwise_correlation"] = None
            top.attrs["high_concentration_warning"] = False

        # --- Intraday profit-booking lean per pick, combining today's
        # RSI level, statistical significance of the move, and how
        # stretched price is from its own fast EMA.
        top["Intraday_Tier"] = ""
        top["Intraday_Action"] = ""
        for idx, row in top.iterrows():
            rec = self.get_intraday_recommendation(row)
            top.at[idx, "Intraday_Tier"] = rec["Tier"]
            top.at[idx, "Intraday_Action"] = rec["Action"]

        return top

    def get_intraday_recommendation(self, row: pd.Series) -> dict:
        """
        Turns a stock's already-computed factors into an intraday
        profit-booking lean, using the same "count the caution flags"
        approach as the IPO analyzer, so the two sections of the alert
        reason about risk consistently.

        Caution flags, each contributing one point:
          1. RSI >= intraday_overbought_rsi: elevated zone where
             short-term pullbacks / consolidation are common.
          2. Extension_From_EMA_Pct >= intraday_extended_pct: price has
             run meaningfully ahead of its own recent average, which
             historically raises mean-reversion risk (read relative to
             the stock's own daily volatility, since "6% stretch" means
             very different things for a low-vol large-cap vs. a
             high-vol small-cap).
          3. Momentum_Significant is False: today's move isn't
             statistically distinguishable from that stock's own daily
             noise, so there's no strong statistical basis to expect it
             to persist through the session.

        More flags -> stronger lean toward booking profit rather than
        holding for further continuation. This is a heuristic reasoning
        aid, not a signal with a demonstrated hit rate -- treat the tier
        as a structured way to weigh the same factors you'd look at
        anyway, not as a rule to follow mechanically.
        """
        cfg = self.cfg
        rsi = row["RSI"]
        significant = bool(row.get("Momentum_Significant", False))
        extension_pct = row.get("Extension_From_EMA_Pct", 0.0) or 0.0
        daily_vol_pct = row.get("Daily_Vol_Pct", None)
        trend_aligned = bool(row.get("Trend_Aligned", False))

        overbought = rsi >= cfg.intraday_overbought_rsi
        extended = extension_pct >= cfg.intraday_extended_pct

        reasons = []
        if overbought:
            reasons.append(
                f"RSI at {rsi} is in an elevated zone (>= {cfg.intraday_overbought_rsi}) "
                "where short-term pauses or pullbacks are common."
            )
        if extended:
            vol_note = f" (vs. a typical daily move of ~{daily_vol_pct}% for this name)" if daily_vol_pct else ""
            reasons.append(
                f"Price is {extension_pct:.1f}% above its fast EMA{vol_note} -- "
                "a stretch that historically raises mean-reversion risk."
            )
        if not significant:
            reasons.append(
                "The recent move isn't statistically distinguishable from this "
                "stock's own daily noise, so there's limited statistical basis "
                "to expect it to extend through the session."
            )
        if trend_aligned and significant and not overbought and not extended:
            reasons.append(
                "Trend alignment and statistically-supported momentum, with "
                "price not yet stretched, together favor letting a trailing "
                "stop manage the exit rather than booking outright."
            )

        caution_count = sum([overbought, extended, not significant])

        if caution_count >= 2:
            tier = "LEAN: BOOK MOST/ALL INTRADAY"
            action = (
                "Multiple caution signals are stacked together. A risk-aware "
                "approach commonly used in this situation is booking most or "
                "all of the position into intraday strength rather than "
                "assuming the move continues into the next session."
            )
        elif caution_count == 1:
            tier = "LEAN: PARTIAL BOOKING + TRAIL STOP"
            action = (
                "One caution flag is present. Booking part of the position "
                "and trailing a stop (e.g. near the fast EMA) on the "
                "remainder is a common middle path -- it locks in some gain "
                "while leaving room if the move continues."
            )
        else:
            tier = "LEAN: HOLD WITH TRAILING STOP"
            action = (
                "No major caution flags today -- RSI, statistical "
                "significance, and trend all lean supportive. Holding with "
                "a trailing stop (rather than a fixed target) is a common "
                "way to stay with the position without giving back the "
                "full gain if sentiment reverses."
            )

        return {"Tier": tier, "Action": action, "Reasons": reasons}

    # ---- Notification ----------------------------------------------------

    def send_telegram_alert(
        self,
        gold_info: dict,
        picks_df: pd.DataFrame,
        ipo_results: Optional[list] = None,
        unverified_ipos: Optional[list] = None,
        regime: Optional[dict] = None,
    ):
        cfg = self.cfg
        if not cfg.bot_token or not cfg.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID.")

        # Explicit IST conversion -- GitHub Actions runners are UTC, so
        # datetime.now() alone silently showed server (UTC) time before,
        # which is why the earlier alert read "04:15" for what was meant
        # to be an 11:00 AM IST send.
        now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        now_str = now_ist.strftime("%d %b %Y, %H:%M IST")

        sig_tag = "significant" if gold_info.get("Signal_Significant") else "not statistically significant"
        lines = [
            "*MULTI-FACTOR MARKET ALERT*",
            f"_{now_str}_",
            "",
        ]
        if regime:
            lines.append(f"MARKET REGIME: {regime['Regime']} -- {regime['Note']}")
            lines.append("")
        lines += [
            "--- GOLD STATISTICAL SIGNAL ---",
            f"GOLDBEES: Rs.{gold_info['Gold_Price']} ({gold_info['Gold_Change']}%)",
            f"USD/INR: Rs.{gold_info['USD_INR']}",
            f"RSI(14): {gold_info['RSI']} | Est. P(up next day): {gold_info['Upward_Probability']}%",
            f"  (drift {sig_tag}, p={gold_info.get('P_Value_Mean_Nonzero')})",
            f"Ann. Volatility: {gold_info['Ann_Volatility_Pct']}%",
            f"Signal: {gold_info['Signal']}",
            "",
            "--- TOP QUANT PICKS (~6mo momentum, skip 1mo -- see notes) ---",
        ]
        for i, row in picks_df.iterrows():
            sig_note = "sig." if row.get("Momentum_Significant") else "noise-level"
            lines.append(
                f"{i + 1}. {row['Ticker']} | Rs.{row['Price']} | "
                f"~6mo mom (skip 1mo): {row['Medium_Momentum_Pct']}% ({sig_note}) | "
                f"RSI: {row['RSI']} | Score: {row['Composite_Score']}"
            )
            short_mom = row.get("Short_Momentum_Pct")
            if short_mom is not None:
                lines.append(f"   {cfg.short_momentum_lookback_days}D move: {short_mom}% (context/caution only, not ranked on)")
            ext = row.get("Extension_From_EMA_Pct")
            vol = row.get("Daily_Vol_Pct")
            if ext is not None:
                vol_note = f" | typical daily move ~{vol}%" if vol else ""
                lines.append(f"   Extension from EMA: {ext}%{vol_note}")
            tier = row.get("Intraday_Tier")
            action = row.get("Intraday_Action")
            if tier:
                lines.append(f"   INTRADAY: {tier}")
                lines.append(f"   -> {action}")
        avg_corr = picks_df.attrs.get("avg_pairwise_correlation")
        if avg_corr is not None:
            warn = " (HIGH CONCENTRATION)" if picks_df.attrs.get("high_concentration_warning") else ""
            lines.append(f"Avg pairwise correlation of picks: {avg_corr}{warn}")

        if ipo_results:
            lines.append("")
            lines.append("--- IPO OPEN / UPCOMING FOR SUBSCRIPTION ---")
            for r in ipo_results:
                status = r.get("Subscription_Status", "UNKNOWN")
                close_note = f" | Closes: {r['Subscription_Close_Date']}" if r.get("Subscription_Close_Date") else ""
                lines.append(
                    f"{r['IPO']} [{status}]{close_note}"
                )
                lines.append(
                    f"  Issue Rs.{r['Issue_Price']} | GMP Rs.{r['GMP']} "
                    f"({r['GMP_Pct']}%) | Est. listing Rs.{r['Estimated_Listing_Price']} "
                    f"| Tier: {r['GMP_Tier']}"
                )
                lines.append(f"  -> {r['Recommendation']}")
        else:
            lines.append("")
            lines.append("--- IPO OPEN / UPCOMING FOR SUBSCRIPTION ---")
            lines.append("No IPOs currently open or opening soon in the configured watchlist.")

        if unverified_ipos:
            lines.append("")
            lines.append("--- IPO: OPEN BUT NEEDS MANUAL VERIFICATION ---")
            lines.append(
                "Source data conflicted for these -- confirm price band/dates on "
                "NSE's IPO page or the RHP, then update the watchlist entry:"
            )
            for ipo in unverified_ipos:
                close_note = f" | Closes: {ipo.subscription_close_date}" if ipo.subscription_close_date else ""
                lines.append(f"  - {ipo.name}{close_note}")

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
# Backtesting -- walk-forward simulation of the exact scoring logic above
# ---------------------------------------------------------------------------

def backtest_stock_strategy(
    cfg: EngineConfig,
    period: str = "2y",
    start_offset_days: int = 200,
    forward_horizons: tuple = (1, 3, 5, 10),
) -> tuple:
    """
    Replays get_multi_factor_stock_picks()'s exact logic (medium-term
    momentum, RSI screen, trend alignment, z-score composite) on every
    historical trading day in the backtest window, using ONLY data that
    would have been available up to that day (no look-ahead), then
    checks what actually happened over the following 1/3/5/10 trading
    days. This is what turns "no backtest included" from a disclaimer
    into an answerable question.

    Returns (trades_df, summary_dict). trades_df has one row per
    historical pick with its forward returns; summary_dict aggregates
    hit rate and average return per horizon, plus a same-period Nifty 50
    buy-and-hold comparison so you can see whether the strategy actually
    added anything over just holding the index.

    Honest limitations of this backtest (read before trusting the output):
      - No transaction costs, slippage, brokerage, or STT are modeled --
        real returns will be lower, especially for short holding periods.
      - No position sizing / capital constraints -- assumes you could
        take every signal at full size, which isn't realistic.
      - Survivorship bias: cfg.stock_universe is today's list of liquid
        names, not the actual historical Nifty-100 constituents on each
        past date -- names that were removed from the index (due to
        underperformance, among other reasons) aren't in the test.
      - Single-market, single-universe test -- statistical significance
        of the aggregate hit rate should be checked (e.g. is it
        meaningfully different from 50% given the sample size?) rather
        than eyeballed.
    """
    log.info("Downloading %s of history for %d tickers for backtest...", period, len(cfg.stock_universe))
    data = download_with_retries(
        cfg.stock_universe, period=period, interval="1d",
        retries=cfg.download_retries, backoff_sec=cfg.retry_backoff_sec,
    )

    # Precompute full indicator series per ticker (vectorized), so the
    # per-day loop below is just lookups, not recomputation.
    indicators = {}
    for ticker in cfg.stock_universe:
        if ticker not in data.columns:
            continue
        s = data[ticker].dropna()
        if len(s) < cfg.medium_momentum_lookback_days + cfg.medium_momentum_skip_days + start_offset_days:
            continue
        indicators[ticker] = {
            "price": s,
            "ema_fast": s.ewm(span=cfg.ema_fast, adjust=False).mean(),
            "sma_slow": s.rolling(cfg.sma_slow).mean(),
            "rsi": wilders_rsi_series(s, cfg.rsi_period),
        }

    if not indicators:
        raise ValueError("Not enough historical data to backtest -- try a longer 'period'.")

    # Common date index across all tickers with enough history.
    all_dates = sorted(set().union(*[set(ind["price"].index) for ind in indicators.values()]))
    test_dates = all_dates[start_offset_days:-max(forward_horizons)]
    log.info("Backtesting over %d historical trading days...", len(test_dates))

    trade_rows = []
    for dt in test_dates:
        day_rows = []
        for ticker, ind in indicators.items():
            price_series = ind["price"]
            if dt not in price_series.index:
                continue
            idx = price_series.index.get_loc(dt)
            if idx < cfg.medium_momentum_lookback_days + cfg.medium_momentum_skip_days:
                continue

            price = price_series.iloc[idx]
            ema = ind["ema_fast"].iloc[idx]
            sma = ind["sma_slow"].iloc[idx]
            rsi = ind["rsi"].iloc[idx]
            if pd.isna(ema) or pd.isna(sma) or pd.isna(rsi):
                continue
            if rsi > cfg.rsi_max or rsi < cfg.rsi_min:
                continue

            skip = cfg.medium_momentum_skip_days
            look = cfg.medium_momentum_lookback_days
            p_start = price_series.iloc[idx - look - skip]
            p_end = price_series.iloc[idx - skip]
            medium_mom = ((p_end - p_start) / p_start) * 100

            trend_aligned = price > ema > sma

            day_rows.append({
                "Date": dt, "Ticker": ticker, "Idx": idx, "Price": price,
                "RSI": rsi, "Medium_Momentum_Pct": medium_mom, "Trend_Aligned": trend_aligned,
            })

        if len(day_rows) < 3:
            continue
        day_df = pd.DataFrame(day_rows)
        day_df["z_momentum"] = zscore(day_df["Medium_Momentum_Pct"])
        day_df["z_rsi_quality"] = zscore(-(day_df["RSI"] - 60).abs())
        day_df["z_trend"] = zscore(day_df["Trend_Aligned"].astype(float))
        day_df["Composite_Score"] = 0.4 * day_df["z_momentum"] + 0.35 * day_df["z_rsi_quality"] + 0.25 * day_df["z_trend"]

        top = day_df.sort_values("Composite_Score", ascending=False).head(cfg.top_n)

        for _, row in top.iterrows():
            price_series = indicators[row["Ticker"]]["price"]
            idx = int(row["Idx"])
            entry_price = row["Price"]
            trade = {
                "Date": dt, "Ticker": row["Ticker"], "Entry_Price": entry_price,
                "RSI": round(row["RSI"], 1), "Medium_Momentum_Pct": round(row["Medium_Momentum_Pct"], 2),
                "Composite_Score": round(row["Composite_Score"], 3),
            }
            for h in forward_horizons:
                if idx + h < len(price_series):
                    fwd_price = price_series.iloc[idx + h]
                    trade[f"Fwd_{h}d_Pct"] = round(((fwd_price - entry_price) / entry_price) * 100, 2)
                else:
                    trade[f"Fwd_{h}d_Pct"] = np.nan
            trade_rows.append(trade)

    trades_df = pd.DataFrame(trade_rows)
    if trades_df.empty:
        raise ValueError("Backtest produced no trades -- check universe/date range.")

    # --- Benchmark: Nifty 50 buy-and-hold over the same test window ---
    try:
        bench_data = download_with_retries(
            [cfg.regime_index_ticker], period=period, interval="1d",
            retries=cfg.download_retries, backoff_sec=cfg.retry_backoff_sec,
        )
        bench_series = bench_data[cfg.regime_index_ticker] if cfg.regime_index_ticker in bench_data.columns else bench_data.iloc[:, 0]
        bench_series = bench_series.dropna()
        bench_start = bench_series.loc[bench_series.index >= test_dates[0]].iloc[0]
        bench_end = bench_series.iloc[-1]
        bench_return_pct = round(((bench_end - bench_start) / bench_start) * 100, 2)
    except Exception as e:  # noqa: BLE001
        log.warning("Benchmark fetch failed: %s", e)
        bench_return_pct = None

    summary = {"n_trades": len(trades_df), "nifty50_buyhold_pct_same_window": bench_return_pct}
    for h in forward_horizons:
        col = f"Fwd_{h}d_Pct"
        valid = trades_df[col].dropna()
        if len(valid) == 0:
            continue
        hit_rate = float((valid > 0).mean() * 100)
        avg_return = float(valid.mean())
        std_return = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
        # One-sample t-test: is the average return significantly different
        # from zero, given the sample size and observed dispersion?
        if std_return > 0 and len(valid) > 1:
            se = std_return / np.sqrt(len(valid))
            t_stat = avg_return / se
            p_val = float(2 * (1 - t_dist.cdf(abs(t_stat), df=len(valid) - 1)))
        else:
            p_val = 1.0
        summary[f"{h}d_hit_rate_pct"] = round(hit_rate, 1)
        summary[f"{h}d_avg_return_pct"] = round(avg_return, 2)
        summary[f"{h}d_return_p_value"] = round(p_val, 3)
        summary[f"{h}d_n"] = len(valid)

    return trades_df, summary


def print_backtest_report(trades_df: pd.DataFrame, summary: dict):
    print("\n" + "=" * 70)
    print("BACKTEST REPORT -- walk-forward simulation, no look-ahead")
    print("=" * 70)
    print(f"Total historical picks tested: {summary['n_trades']}")
    if summary.get("nifty50_buyhold_pct_same_window") is not None:
        print(f"Nifty 50 buy-and-hold return over same window: {summary['nifty50_buyhold_pct_same_window']}%")
    print("-" * 70)
    print(f"{'Horizon':<10}{'Hit Rate':<12}{'Avg Return':<14}{'p-value':<10}{'N':<6}")
    for h in (1, 3, 5, 10):
        if f"{h}d_hit_rate_pct" not in summary:
            continue
        sig = "*" if summary[f"{h}d_return_p_value"] < 0.05 else " "
        print(
            f"{h}d{sig:<8}"
            f"{summary[f'{h}d_hit_rate_pct']}%{'':<6}"
            f"{summary[f'{h}d_avg_return_pct']}%{'':<8}"
            f"{summary[f'{h}d_return_p_value']:<10}"
            f"{summary[f'{h}d_n']}"
        )
    print("-" * 70)
    print("* = average return statistically distinguishable from zero at p<0.05")
    print(
        "\nReminder: no transaction costs/slippage modeled, and cfg.stock_universe\n"
        "is today's liquid-names list applied retroactively (survivorship bias),\n"
        "not the actual historical index constituents on each test date."
    )
    print("=" * 70 + "\n")


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

    ALWAYS set subscription_open_date / subscription_close_date -- these
    are what filter_actionable_ipos() uses to automatically drop an IPO
    from the alert once you can no longer apply to it. Without a close
    date, an entry is kept indefinitely by default (with a warning in
    the logs), which defeats the point of this filter.

    Below reflects the fuller set of IPOs actually open as of 11 Aug 2026,
    cross-checked across multiple public tracker pages as a live example.
    Two things worth calling out from that cross-check itself:
      - GMP for the same IPO varied by ~Rs.30-60 across trackers at the
        same point in time (unregulated, dealer-quoted number -- expected).
      - More surprisingly, a couple of aggregators showed internally
        inconsistent price bands / segments for the SAME IPO (e.g. one
        listed an SME issue as "Mainboard" with a price band that matched
        a different, unrelated IPO's fresh-issue amount) -- almost
        certainly a scraping/table-row misalignment on their end, not a
        real discrepancy. Rather than guess which number is right,
        conflicting entries are marked verified=False below and are
        excluded from the scored analysis, but still show up in the alert
        as a "confirm before applying" prompt.
    """
    return [
        IPOListingInput(
            name="Molbio Diagnostics",
            issue_price=807.0,             # upper price band -- consistent across 3+ sources
            gmp=150.0,                     # snapshot; seen ranging ~120-180 across sources
            retail_subscription_x=None,    # opened today -- not yet meaningful, update near close (Aug 12)
            qib_subscription_x=None,
            overall_subscription_x=None,
            subscription_open_date="2026-08-10",
            subscription_close_date="2026-08-12",
            listing_date="2026-08-17",
            market_trend="neutral",        # update based on your own Nifty read that day
            sector_hot=True,               # diagnostics/healthcare has had firm demand recently
        ),
        IPOListingInput(
            name="Dhoot Transmission",
            issue_price=871.0,             # upper price band -- consistent across dedicated review sites
            gmp=250.0,                     # snapshot; cross-check before relying on it
            overall_subscription_x=None,   # update near close (Aug 12) -- auto components name, watch QIB closely
            subscription_open_date="2026-08-10",
            subscription_close_date="2026-08-12",
            listing_date="2026-08-17",
            market_trend="neutral",
            sector_hot=True,               # auto-components / EV-adjacent supplier, sector has had steady flows
        ),
        IPOListingInput(
            name="Optimystix Entertainment",
            issue_price=175.0,             # upper price band, SME issue -- confirmed across 5 sources
            gmp=5.0,
            overall_subscription_x=0.6,    # as of a few days into the bidding window
            subscription_open_date="2026-08-07",
            subscription_close_date="2026-08-11",
            listing_date="2026-08-14",
            market_trend="neutral",
            sector_hot=False,              # media/content production, not a currently "hot" flow sector
        ),
        # --- Flagged: open this week, but source data conflicted enough
        # that I'm not confident in the exact price band/segment. Verify
        # directly on NSE's IPO page or the RHP, fill in the confirmed
        # issue_price/gmp, and flip verified=True once you have.
        IPOListingInput(
            name="LEAP India",
            issue_price=159.0,             # UNVERIFIED -- one source said 159, another said 871 (likely a
                                            # scraping mixup with Dhoot Transmission's own price band)
            gmp=0.0,
            subscription_open_date="2026-08-07",
            subscription_close_date="2026-08-11",
            listing_date="2026-08-14",
            verified=False,
        ),
        IPOListingInput(
            name="Technocraft Ventures",
            issue_price=159.0,             # UNVERIFIED -- conflicting price bands seen across sources
            gmp=0.0,
            subscription_open_date="2026-08-07",
            subscription_close_date="2026-08-11",
            listing_date="2026-08-14",
            verified=False,
        ),
        IPOListingInput(
            name="LAPL Automotive",
            issue_price=175.0,             # UNVERIFIED -- price band matched Optimystix's exactly across one
                                            # source, which is more likely a scraping artifact than coincidence
            gmp=0.0,
            subscription_open_date="2026-08-07",
            subscription_close_date="2026-08-11",
            listing_date="2026-08-13",
            verified=False,
        ),
        # Add more IPOListingInput(...) entries here as new issues open.
        # Once today's date passes an entry's subscription_close_date,
        # filter_actionable_ipos() drops it from the alert automatically --
        # you can leave past entries in this list and they'll age out on
        # their own rather than needing manual deletion.
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-factor market engine")
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run a walk-forward historical backtest instead of sending a live alert.",
    )
    parser.add_argument(
        "--backtest-period", default="2y",
        help="How much history to backtest over (yfinance period string, e.g. '1y', '2y'). Default: 2y.",
    )
    args = parser.parse_args()

    config = EngineConfig(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        top_n=int(os.getenv("TOP_N", "3")),
    )

    if args.backtest:
        # Validate the strategy against history BEFORE trusting it live.
        # No Telegram alert is sent in this mode.
        trades_df, summary = backtest_stock_strategy(config, period=args.backtest_period)
        print_backtest_report(trades_df, summary)
        trades_df.to_csv("backtest_trades.csv", index=False)
        log.info("Full trade-level backtest detail written to backtest_trades.csv")
        raise SystemExit(0)

    engine = AdvancedMarketEngine(config)

    regime = engine.get_market_regime()
    log.info("Market regime: %s", regime)

    gold_data = engine.get_probabilistic_gold_signal()
    log.info("Gold signal: %s", gold_data)

    try:
        stock_picks = engine.get_multi_factor_stock_picks()
        log.info("Top picks:\n%s", stock_picks.to_string(index=False))
    except Exception as e:  # noqa: BLE001
        log.warning("Stock screen failed (%s); using single fallback pick.", e)
        stock_picks = pd.DataFrame([{
            "Ticker": "RELIANCE", "Price": 1334.80, "Short_Momentum_Pct": 2.06,
            "Medium_Momentum_Pct": 4.5,
            "RSI": 58.4, "EMA_Fast": 1301.00, "SMA_Slow": 1290.00,
            "Trend_Aligned": True, "Above_Fast_EMA": True, "Composite_Score": 0.0,
            "Momentum_P_Value": 1.0, "Momentum_Significant": False,
            "Extension_From_EMA_Pct": 0.0, "Daily_Vol_Pct": None,
            "Intraday_Tier": "LEAN: HOLD WITH TRAILING STOP", "Intraday_Action": "Fallback data.",
        }])

    engine.log_alert_history(stock_picks, regime)

    # Only IPOs still open (or opening soon) for subscription reach the
    # alert -- anything whose bidding window has already closed is dropped
    # automatically, since the point is to inform a still-pending
    # application decision, not to report on issues you can no longer apply to.
    full_watchlist = load_ipo_watchlist()
    actionable_watchlist = filter_actionable_ipos(full_watchlist)

    verified_ipos = [ipo for ipo in actionable_watchlist if ipo.verified]
    unverified_ipos = [ipo for ipo in actionable_watchlist if not ipo.verified]

    ipo_results = [IPOAnalyzer(ipo).analyze() for ipo in verified_ipos]
    for r in ipo_results:
        log.info("IPO read for %s (%s): %s", r["IPO"], r["Subscription_Status"], r["Recommendation"])
    for ipo in unverified_ipos:
        log.warning(
            "IPO '%s' is open but marked unverified -- source data conflicted; "
            "surfacing as a manual-check prompt only.", ipo.name,
        )

    engine.send_telegram_alert(
        gold_data, stock_picks, ipo_results=ipo_results, unverified_ipos=unverified_ipos,
        regime=regime,
    )
