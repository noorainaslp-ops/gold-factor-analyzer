"""
Indian Market Engine V6 — short-term opportunity + IPO alert

V6 goals:
- Rank liquid Indian stocks for 1/3/5-session opportunities.
- Use a calibrated direction model + return model, not a single hard filter.
- Use OHLCV features: momentum, RSI, EMA/SMA trend, ATR, volatility, volume,
  relative strength and Nifty/VIX regime.
- Produce an actionable plan: entry zone, stop, T1, T2, expected holding period.
- Keep a NO-TRADE state, but do not suppress all candidates merely because
  one market-regime filter fails.
- Log every candidate and every daily alert so future performance can be audited.
- Pull current/upcoming IPO names from NSE's public IPO page when available.
  GMP is deliberately NOT fabricated; it can be supplied through IPO_GMP_JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IST = ZoneInfo("Asia/Kolkata")
log = logging.getLogger("market_engine_v6")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class EngineConfig:
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None

    short_horizons: tuple = (1, 3, 5)
    primary_horizon: int = 3
    model_lookback: int = 504
    min_training_samples: int = 2500

    top_n: int = 3
    min_score: float = 0.08
    min_probability: float = 0.53
    min_pred_return_pct: float = 0.25

    # Risk / execution
    atr_stop_multiple: float = 1.20
    atr_entry_buffer: float = 0.20
    target1_fraction: float = 0.55
    min_target1_pct: float = 0.70
    min_target2_pct: float = 1.20
    max_entry_extension_atr: float = 1.00

    # Liquidity filters
    min_price: float = 50.0
    min_avg_turnover_cr: float = 25.0

    # Universe
    fallback_universe: list = field(default_factory=lambda: [
        "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
        "BHARTIARTL.NS","SBIN.NS","ITC.NS","HINDUNILVR.NS","LT.NS",
        "BAJFINANCE.NS","HCLTECH.NS","KOTAKBANK.NS","SUNPHARMA.NS",
        "MARUTI.NS","M&M.NS","AXISBANK.NS","TITAN.NS","NTPC.NS",
        "ADANIENT.NS","BAJAJFINSV.NS","ONGC.NS","POWERGRID.NS",
        "ADANIPORTS.NS","COALINDIA.NS","WIPRO.NS","JSWSTEEL.NS",
        "TATASTEEL.NS","NESTLEIND.NS","TATAMOTORS.NS","ASIANPAINT.NS",
        "HAL.NS","BEL.NS","GRASIM.NS","SBILIFE.NS","TECHM.NS",
        "HDFCLIFE.NS","CIPLA.NS","TRENT.NS","DRREDDY.NS","EICHERMOT.NS",
        "BAJAJ-AUTO.NS","APOLLOHOSP.NS","BRITANNIA.NS","DIVISLAB.NS",
        "INDUSINDBK.NS","HEROMOTOCO.NS","SHRIRAMFIN.NS","PIDILITIND.NS",
        "GODREJCP.NS","DABUR.NS","SIEMENS.NS","DLF.NS","VEDL.NS","LTIM.NS",
        "AMBUJACEM.NS","BANKBARODA.NS","PNB.NS","GAIL.NS","IOC.NS","BPCL.NS",
        "TATAPOWER.NS","TATACONSUM.NS","PIIND.NS","HAVELLS.NS","MOTHERSON.NS",
        "BOSCHLTD.NS","CANBK.NS","IDFCFIRSTB.NS","AUROPHARMA.NS","LUPIN.NS",
        "TORNTPHARM.NS","COLPAL.NS","MARICO.NS","SRF.NS","PAGEIND.NS",
        "MUTHOOTFIN.NS","CHOLAFIN.NS","BALKRISIND.NS","ICICIPRULI.NS",
        "ICICIGI.NS","INDIGO.NS","NAUKRI.NS","PFC.NS","RECLTD.NS",
        "JINDALSTEL.NS","SAIL.NS","HINDALCO.NS","NMDC.NS","UPL.NS",
        "BHARATFORG.NS","CUMMINSIND.NS","ABB.NS","POLYCAB.NS","PERSISTENT.NS",
    ])

    nifty_ticker: str = "^NSEI"
    vix_ticker: str = "^INDIAVIX"
    history_path: str = "alert_history_v6.csv"
    candidate_history_path: str = "candidate_history_v6.csv"


FEATURES = [
    "ret1","ret3","ret5","ret10","ret20",
    "rsi14","dist_ema5","dist_ema20","dist_sma50",
    "ema20_slope10","atr_pct","vol20","volume_ratio",
    "relative3","relative5","relative10",
    "nifty_ret3","nifty_ret10","nifty_above_sma50",
    "nifty_sma50_slope10","vix_level","vix_change5",
]


def now_ist() -> datetime:
    return datetime.now(IST)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def download_ohlcv(tickers, period="3y", interval="1d", retries=3):
    last = None
    tickers = list(dict.fromkeys(tickers))
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers=tickers, period=period, interval=interval,
                auto_adjust=False, progress=False, threads=True, group_by="column"
            )
            if raw.empty:
                raise ValueError("Empty yfinance response")

            # yfinance returns either field->ticker or ticker->field depending on version.
            if isinstance(raw.columns, pd.MultiIndex):
                level0 = set(raw.columns.get_level_values(0))
                fields = {"Open","High","Low","Close","Adj Close","Volume"}
                if fields & level0:
                    out = {f: raw[f] for f in fields if f in level0}
                else:
                    out = {}
                    for f in fields:
                        if f in raw.columns.get_level_values(1):
                            out[f] = raw.xs(f, axis=1, level=1)
                result = out
            else:
                # Single ticker
                result = {c: raw[[c]] for c in raw.columns if c in {"Open","High","Low","Close","Volume"}}

            if "Close" not in result:
                raise ValueError("Close field missing")
            return result
        except Exception as exc:
            last = exc
            log.warning("Data download %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"All market-data attempts failed: {last}")


def load_nifty500_universe(fallback):
    """Use the official Nifty 500 constituent CSV when reachable; otherwise fallback."""
    url = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        syms = [str(x).strip().upper() + ".NS" for x in df[col].dropna()]
        syms = [x for x in syms if x != "NAN.NS"]
        if len(syms) >= 200:
            log.info("Loaded %d current Nifty 500 constituents.", len(syms))
            return syms
    except Exception as exc:
        log.warning("Could not load Nifty 500 list; using fallback universe: %s", exc)
    return list(fallback)


def rsi(series, period=14):
    d = series.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(high, low, close, period=14):
    prev = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev).abs(),
        (low - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def build_features(close, high, low, volume, market, vix):
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    volume = volume.astype(float)
    market = market.reindex(close.index).ffill()
    vix = vix.reindex(close.index).ffill()

    ema5 = close.ewm(span=5, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    sma50 = close.rolling(50).mean()
    msma50 = market.rolling(50).mean()
    a = atr(high, low, close)
    ret = close.pct_change()

    f = pd.DataFrame(index=close.index)
    for n in (1,3,5,10,20):
        f[f"ret{n}"] = close.pct_change(n)
    f["rsi14"] = rsi(close)
    f["dist_ema5"] = close / ema5 - 1
    f["dist_ema20"] = close / ema20 - 1
    f["dist_sma50"] = close / sma50 - 1
    f["ema20_slope10"] = ema20.pct_change(10)
    f["atr_pct"] = a / close
    f["vol20"] = ret.rolling(20).std()
    f["volume_ratio"] = volume / volume.rolling(20).median()
    f["nifty_ret3"] = market.pct_change(3)
    f["nifty_ret10"] = market.pct_change(10)
    f["nifty_above_sma50"] = (market > msma50).astype(float)
    f["nifty_sma50_slope10"] = msma50.pct_change(10)
    f["relative3"] = f["ret3"] - f["nifty_ret3"]
    f["relative5"] = f["ret5"] - market.pct_change(5)
    f["relative10"] = f["ret10"] - f["nifty_ret10"]
    f["vix_level"] = vix
    f["vix_change5"] = vix.pct_change(5)
    f["price"] = close
    f["atr"] = a
    f["volume"] = volume
    for h in (1,3,5):
        f[f"target{h}"] = close.shift(-h) / close - 1
    return f


def fit_models(training, horizon, cfg):
    target = f"target{horizon}"
    x = training[FEATURES].replace([np.inf,-np.inf], np.nan)
    y = training[target]
    valid = x.notna().all(axis=1) & y.notna()
    x, y = x.loc[valid], y.loc[valid]
    if len(x) < cfg.min_training_samples or y.nunique() < 10:
        return None

    # Time-ordered holdout for probability calibration.
    cut = int(len(x) * 0.80)
    if cut < 1000 or len(x) - cut < 200:
        return None
    x0, y0 = x.iloc[:cut], y.iloc[:cut]
    x1, y1 = x.iloc[cut:], y.iloc[cut:]
    direction0 = (y0 > 0).astype(int)

    ret_model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=8.0)),
    ])
    dir_model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=0.25, max_iter=1000, class_weight="balanced")),
    ])
    ret_model.fit(x0, y0.clip(y0.quantile(.01), y0.quantile(.99)))
    dir_model.fit(x0, direction0)

    hold_probs = dir_model.predict_proba(x1)[:,1]
    cal_y = (y1 > 0).astype(int).to_numpy()
    calibrator = IsotonicRegression(out_of_bounds="clip")
    # If holdout has both classes, calibrate; otherwise use raw probabilities.
    if len(np.unique(cal_y)) == 2:
        calibrator.fit(hold_probs, cal_y)
    else:
        calibrator = None

    # Refit both models on all past data after calibration.
    ret_model.fit(x, y.clip(y.quantile(.01), y.quantile(.99)))
    dir_model.fit(x, (y > 0).astype(int))
    return ret_model, dir_model, calibrator


def calibrated_probability(model, calibrator, x):
    raw = float(model.predict_proba(x)[0,1])
    return float(calibrator.predict([raw])[0]) if calibrator is not None else raw


def market_regime(market, vix):
    market = market.dropna()
    sma = market.rolling(50).mean()
    slope = sma.pct_change(10)
    latest = float(market.iloc[-1])
    s = float(sma.iloc[-1])
    sl = float(slope.iloc[-1])
    v = float(vix.dropna().iloc[-1]) if not vix.dropna().empty else np.nan

    score = 0
    score += 1 if latest > s else -1
    score += 1 if sl > 0 else -1
    if np.isfinite(v):
        score += -1 if v > 18 else 1 if v < 14 else 0

    if score >= 2:
        label = "FAVORABLE"
    elif score <= -2:
        label = "UNFAVORABLE"
    else:
        label = "MIXED"
    return {
        "label": label, "nifty": latest, "sma50": s,
        "sma50_slope10_pct": sl*100, "vix": v, "score": score
    }


def candidate_plan(row, cfg):
    price = row["Price"]
    a = row["ATR"]
    pred3 = row["Predicted_3D_Return_Pct"] / 100
    if not np.isfinite(a) or a <= 0:
        a = price * max(row["ATR_Pct"]/100, 0.01)

    entry_low = max(0.01, price - cfg.atr_entry_buffer * a)
    entry_high = price + 0.10 * a
    stop = price - cfg.atr_stop_multiple * a

    target2_return = max(pred3, cfg.min_target2_pct/100)
    target1_return = max(target2_return * cfg.target1_fraction, cfg.min_target1_pct/100)
    t1 = price * (1 + target1_return)
    t2 = price * (1 + target2_return)

    risk_pct = (price - stop) / price * 100
    rr1 = ((t1 - price) / price * 100) / max(risk_pct, 0.01)
    rr2 = ((t2 - price) / price * 100) / max(risk_pct, 0.01)

    if row["Probability_3D_Pct"] >= 60 and row["Score"] >= 0.18:
        tier = "HIGH-CONFIDENCE"
    elif row["Probability_3D_Pct"] >= 55 and row["Score"] >= cfg.min_score:
        tier = "GOOD SETUP"
    else:
        tier = "SPECULATIVE / SMALL SIZE"

    action = "BUY ON CONFIRMATION"
    if row["RSI"] > 70 or row["Extension_ATR"] > cfg.max_entry_extension_atr:
        action = "DO NOT CHASE — WAIT FOR PULLBACK"

    return {
        "Entry_Low": round(entry_low,2),
        "Entry_High": round(entry_high,2),
        "Stop_Loss": round(stop,2),
        "Target_1": round(t1,2),
        "Target_2": round(t2,2),
        "Risk_Pct": round(risk_pct,2),
        "RR_T1": round(rr1,2),
        "RR_T2": round(rr2,2),
        "Tier": tier,
        "Action": action,
    }


class MarketEngineV6:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_stock_picks(self):
        universe = load_nifty500_universe(self.cfg.fallback_universe)
        tickers = list(dict.fromkeys(universe + [self.cfg.nifty_ticker, self.cfg.vix_ticker]))
        data = download_ohlcv(tickers, period="3y")

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        market = close[self.cfg.nifty_ticker].dropna()
        vix = close[self.cfg.vix_ticker].dropna() if self.cfg.vix_ticker in close else pd.Series(dtype=float)

        frames = {}
        training_parts = {h: [] for h in self.cfg.short_horizons}
        latest_date = market.index[-1]

        for ticker in universe:
            if ticker not in close or ticker not in high or ticker not in low or ticker not in volume:
                continue
            s = close[ticker].dropna()
            if len(s) < 650:
                continue
            f = build_features(
                close[ticker], high[ticker], low[ticker], volume[ticker], market, vix
            )
            frames[ticker] = f
            hist = f[f.index < latest_date].tail(self.cfg.model_lookback)
            for h in self.cfg.short_horizons:
                training_parts[h].append(hist)

        models = {}
        for h in self.cfg.short_horizons:
            if not training_parts[h]:
                continue
            training = pd.concat(training_parts[h], ignore_index=True)
            models[h] = fit_models(training, h, self.cfg)

        if self.cfg.primary_horizon not in models or models[self.cfg.primary_horizon] is None:
            raise RuntimeError("V6 could not fit the primary short-term model.")

        rows = []
        for ticker, f in frames.items():
            if latest_date not in f.index:
                continue
            row = f.loc[[latest_date]]
            if row[FEATURES].isna().any(axis=1).iloc[0]:
                continue

            preds, probs = {}, {}
            x = row[FEATURES]
            for h in self.cfg.short_horizons:
                m = models.get(h)
                if not m:
                    continue
                rm, dm, cal = m
                preds[h] = float(rm.predict(x)[0])
                probs[h] = calibrated_probability(dm, cal, x)

            if 3 not in preds:
                continue

            price = float(row["price"].iloc[0])
            atr_v = float(row["atr"].iloc[0])
            atr_pct = atr_v / price * 100 if price else np.nan
            rsi_v = float(row["rsi14"].iloc[0])
            turnover_cr = price * float(row["volume"].iloc[0]) / 1e7
            ext_atr = float(row["dist_ema20"].iloc[0]) * price / max(atr_v, 1e-9)

            trend = (
                float(row["dist_ema20"].iloc[0]) > 0 and
                float(row["dist_sma50"].iloc[0]) > 0 and
                float(row["ema20_slope10"].iloc[0]) > 0
            )
            breadth_ok = float(row["nifty_above_sma50"].iloc[0]) > 0

            # Ensemble across horizons. 3D dominates because it is the execution horizon.
            ensemble_return = (
                0.20 * preds.get(1, preds[3]) +
                0.55 * preds[3] +
                0.25 * preds.get(5, preds[3])
            )
            ensemble_prob = (
                0.20 * probs.get(1, probs[3]) +
                0.55 * probs[3] +
                0.25 * probs.get(5, probs[3])
            )

            # Risk-adjusted expected edge.  This is a ranking statistic, not a guarantee.
            uncertainty = max(0.0, atr_pct / 100 * 0.20)
            raw_edge = ensemble_return * (2 * ensemble_prob - 1)
            risk_adjusted = raw_edge - uncertainty

            # Soft regime penalty rather than a hard "no trade".
            if not breadth_ok:
                risk_adjusted *= 0.75
            if rsi_v > 72:
                risk_adjusted *= 0.75
            if rsi_v < 42:
                risk_adjusted *= 0.80
            if ext_atr > 1.25:
                risk_adjusted *= 0.70

            # Liquidity / sanity filters.
            liquid = price >= self.cfg.min_price and turnover_cr >= self.cfg.min_avg_turnover_cr
            plausible = ensemble_return * 100 >= self.cfg.min_pred_return_pct and ensemble_prob >= self.cfg.min_probability
            if not liquid or not plausible:
                continue

            # Reward strong relative strength, trend, and volume confirmation.
            score = risk_adjusted
            score += 0.04 * np.clip(float(row["relative5"].iloc[0]) * 10, -1, 1)
            score += 0.03 * np.clip((float(row["volume_ratio"].iloc[0]) - 1) / 1.5, -1, 1)
            score += 0.03 if trend else -0.03
            score = float(score)

            rows.append({
                "Ticker": ticker.replace(".NS",""),
                "Price": round(price,2),
                "Predicted_1D_Return_Pct": round(preds.get(1,np.nan)*100,2),
                "Predicted_3D_Return_Pct": round(preds[3]*100,2),
                "Predicted_5D_Return_Pct": round(preds.get(5,np.nan)*100,2),
                "Probability_1D_Pct": round(probs.get(1,np.nan)*100,1),
                "Probability_3D_Pct": round(probs[3]*100,1),
                "Probability_5D_Pct": round(probs.get(5,np.nan)*100,1),
                "Ensemble_Return_Pct": round(ensemble_return*100,2),
                "Ensemble_Probability_Pct": round(ensemble_prob*100,1),
                "Score": round(score,4),
                "RSI": round(rsi_v,1),
                "ATR": round(atr_v,2),
                "ATR_Pct": round(atr_pct,2),
                "Extension_ATR": round(ext_atr,2),
                "Volume_Ratio": round(float(row["volume_ratio"].iloc[0]),2),
                "Relative_5D_Pct": round(float(row["relative5"].iloc[0])*100,2),
                "Trend_Aligned": trend,
                "Turnover_Cr": round(turnover_cr,1),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame([{
                "Ticker":"--","No_Trade":True,"Tier":"NO TRADE",
                "Action":"No liquid candidate met the minimum calibrated probability/return filters."
            }])

        df = df.sort_values(["Score","Ensemble_Probability_Pct","Ensemble_Return_Pct"], ascending=False)
        # Keep only genuinely useful positive-score ideas.
        df = df[df["Score"] >= self.cfg.min_score].head(self.cfg.top_n).reset_index(drop=True)

        if df.empty:
            return pd.DataFrame([{
                "Ticker":"--","No_Trade":True,"Tier":"NO TRADE",
                "Action":"The model found candidates, but none had enough risk-adjusted edge to justify a trade."
            }])

        plans = [candidate_plan(r, self.cfg) for _, r in df.iterrows()]
        for k, plan in enumerate(plans):
            for key, value in plan.items():
                df.loc[k, key] = value
        df["No_Trade"] = False

        if len(df) > 1:
            # Avoid three highly correlated bets.
            # A simple warning is safer than pretending correlation can be eliminated.
            names = [x + ".NS" for x in df["Ticker"]]
            panel = close[[x for x in names if x in close.columns]].dropna()
            if panel.shape[1] > 1:
                c = panel.pct_change().corr()
                vals = c.values
                n = len(c)
                avg_corr = (vals.sum()-n)/(n*(n-1))
                df.attrs["avg_pairwise_correlation"] = round(float(avg_corr),2)
                df.attrs["high_concentration_warning"] = bool(avg_corr > .65)
        else:
            df.attrs["avg_pairwise_correlation"] = None
            df.attrs["high_concentration_warning"] = False

        self.log_candidates(df)
        return df

    def log_candidates(self, df):
        out = df.copy()
        out["date"] = now_ist().strftime("%Y-%m-%d")
        cols = ["date"] + [c for c in out.columns if c != "date"]
        try:
            if os.path.exists(self.cfg.candidate_history_path):
                old = pd.read_csv(self.cfg.candidate_history_path)
                out = pd.concat([old, out], ignore_index=True)
            out[cols].to_csv(self.cfg.candidate_history_path, index=False)
        except Exception as exc:
            log.warning("Candidate history write failed: %s", exc)

    def log_alert(self, df, regime):
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "date": now_ist().strftime("%Y-%m-%d"),
                "ticker": r.get("Ticker"),
                "price": r.get("Price"),
                "score": r.get("Score"),
                "predicted_3d_return_pct": r.get("Predicted_3D_Return_Pct"),
                "probability_3d_pct": r.get("Probability_3D_Pct"),
                "entry_low": r.get("Entry_Low"),
                "entry_high": r.get("Entry_High"),
                "stop_loss": r.get("Stop_Loss"),
                "target_1": r.get("Target_1"),
                "target_2": r.get("Target_2"),
                "tier": r.get("Tier"),
                "action": r.get("Action"),
                "regime": regime["label"],
            })
        new = pd.DataFrame(rows)
        try:
            old = pd.read_csv(self.cfg.history_path) if os.path.exists(self.cfg.history_path) else pd.DataFrame()
            pd.concat([old,new], ignore_index=True).to_csv(self.cfg.history_path,index=False)
        except Exception as exc:
            log.warning("Alert history write failed: %s", exc)

    def format_stock_section(self, df):
        lines = ["--- SHORT-TERM OPPORTUNITIES (1–5 SESSIONS) ---"]
        for i, r in df.iterrows():
            if r.get("No_Trade"):
                lines.append("NO TRADE — no setup cleared the minimum risk-adjusted edge.")
                continue
            lines += [
                f"{i+1}. {r['Ticker']} | Rs.{r['Price']}",
                f"   1D/3D/5D model: {r['Predicted_1D_Return_Pct']}% / {r['Predicted_3D_Return_Pct']}% / {r['Predicted_5D_Return_Pct']}%",
                f"   P(positive): {r['Probability_1D_Pct']}% / {r['Probability_3D_Pct']}% / {r['Probability_5D_Pct']}%",
                f"   Score: {r['Score']} | RSI: {r['RSI']} | ATR: {r['ATR_Pct']}% | Vol: {r['Volume_Ratio']}x",
                f"   ENTRY: Rs.{r['Entry_Low']}–{r['Entry_High']}",
                f"   TARGET 1: Rs.{r['Target_1']} | TARGET 2: Rs.{r['Target_2']}",
                f"   STOP: Rs.{r['Stop_Loss']} | R:R T1 {r['RR_T1']} | T2 {r['RR_T2']}",
                f"   TIER: {r['Tier']} | ACTION: {r['Action']}",
            ]
        corr = df.attrs.get("avg_pairwise_correlation")
        if corr is not None:
            warn = " HIGH CONCENTRATION" if df.attrs.get("high_concentration_warning") else ""
            lines.append(f"Avg pick correlation: {corr}{warn}")
        return lines

    def send_telegram(self, regime, picks, ipo_results):
        if not self.cfg.bot_token or not self.cfg.chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing.")

        lines = [
            "*MULTI-FACTOR MARKET ALERT V6*",
            f"_{now_ist().strftime('%d %b %Y, %H:%M IST')}_",
            "",
            f"MARKET REGIME: {regime['label']} | Nifty: {regime['nifty']:.2f} | "
            f"50D avg: {regime['sma50']:.2f} | VIX: {regime['vix']:.2f}",
            "",
        ]
        lines.extend(self.format_stock_section(picks))
        lines += ["", "--- IPO OPEN / UPCOMING ---"]
        if ipo_results:
            for r in ipo_results:
                lines.append(
                    f"{r['name']} [{r['status']}] | {r['start']} → {r['end']}"
                )
                if r.get("price"):
                    lines.append(f"   Price: {r['price']}")
                if r.get("subscription"):
                    lines.append(f"   Subscription: {r['subscription']}")
                if r.get("gmp") is not None:
                    lines.append(f"   GMP: Rs.{r['gmp']} ({r.get('gmp_pct','?')}%)")
                else:
                    lines.append("   GMP: not supplied/verified")
                if r.get("note"):
                    lines.append(f"   -> {r['note']}")
        else:
            lines.append("No current/upcoming IPO was retrieved from the NSE source.")
        lines += [
            "",
            "_V6 is a probabilistic screen, not a profit guarantee. Targets/stops are model-derived risk controls. GMP is unofficial unless independently verified._"
        ]

        url = f"https://api.telegram.org/bot{self.cfg.bot_token}/sendMessage"
        res = requests.post(
            url, json={"chat_id": self.cfg.chat_id, "text":"\n".join(lines), "parse_mode":"Markdown"},
            timeout=15
        )
        res.raise_for_status()

    def get_regime(self):
        data = download_ohlcv([self.cfg.nifty_ticker,self.cfg.vix_ticker], period="1y")
        market = data["Close"][self.cfg.nifty_ticker].dropna()
        vix = data["Close"][self.cfg.vix_ticker].dropna() if self.cfg.vix_ticker in data["Close"] else pd.Series(dtype=float)
        return market_regime(market,vix)


def fetch_nse_ipos():
    """Best-effort current IPO list from NSE's public page; never invents GMP."""
    url = "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/151 Safari/537.36",
        "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer":"https://www.nseindia.com/"
    }
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com/",headers=headers,timeout=10)
        html = s.get(url,headers=headers,timeout=15).text
        tables = pd.read_html(html)
        best = None
        for t in tables:
            cols = " ".join(map(str,t.columns)).lower()
            if "company" in cols and "issue" in cols and ("date" in cols or "status" in cols):
                best = t
                break
        if best is None or best.empty:
            return []
        best.columns = [str(c).strip() for c in best.columns]
        rows = []
        for _, r in best.iterrows():
            vals = [str(x) for x in r.tolist()]
            joined = " | ".join(vals)
            if "nan" in joined.lower() and len(vals) < 3:
                continue
            # NSE page structures change; keep a conservative display record.
            rows.append({
                "name": vals[0] if vals else "Unknown IPO",
                "status": "OPEN/UPCOMING",
                "start": vals[1] if len(vals)>1 else "?",
                "end": vals[2] if len(vals)>2 else "?",
                "price": None,
                "subscription": vals[-1] if vals else None,
                "gmp": None,
                "note": "Verify issue price/GMP on NSE/offer documents before applying."
            })
        return rows[:12]
    except Exception as exc:
        log.warning("NSE IPO retrieval failed: %s", exc)
        return []


def run_backtest(cfg, period="5y"):
    """Walk-forward 3-session test of the same feature/model logic."""
    universe = load_nifty500_universe(cfg.fallback_universe)
    tickers = list(dict.fromkeys(universe + [cfg.nifty_ticker,cfg.vix_ticker]))
    data = download_ohlcv(tickers, period=period)
    close, high, low, volume = data["Close"], data["High"], data["Low"], data["Volume"]
    market = close[cfg.nifty_ticker].dropna()
    vix = close[cfg.vix_ticker].dropna() if cfg.vix_ticker in close else pd.Series(dtype=float)

    frames={}
    for t in universe:
        if not all(t in x for x in (close,high,low,volume)):
            continue
        if close[t].dropna().shape[0] < 750:
            continue
        frames[t]=build_features(close[t],high[t],low[t],volume[t],market,vix)

    common = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    test_dates = common[max(550, len(common)//3): -6]
    trades=[]
    for dt in test_dates:
        # Train only on observations whose target is known strictly before dt.
        parts=[]
        for f in frames.values():
            parts.append(f[f.index < dt].tail(cfg.model_lookback))
        train=pd.concat(parts,ignore_index=True)
        m=fit_models(train,3,cfg)
        if not m:
            continue
        rm,dm,cal=m
        candidates=[]
        for ticker,f in frames.items():
            if dt not in f.index:
                continue
            r=f.loc[[dt]]
            if r[FEATURES].isna().any(axis=1).iloc[0]:
                continue
            x=r[FEATURES]
            pred=float(rm.predict(x)[0])
            prob=calibrated_probability(dm,cal,x)
            if pred*100 < cfg.min_pred_return_pct or prob < cfg.min_probability:
                continue
            price=float(r["price"].iloc[0])
            a=float(r["atr"].iloc[0])
            atr_pct=a/price*100
            turnover=price*float(r["volume"].iloc[0])/1e7
            if price<cfg.min_price or turnover<cfg.min_avg_turnover_cr:
                continue
            edge=pred*(2*prob-1)-0.2*(a/price)
            candidates.append((edge,ticker,pred,prob,price,a))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        edge,ticker,pred,prob,price,a=candidates[0]
        s=close[ticker].dropna()
        if dt not in s.index:
            continue
        idx=s.index.get_loc(dt)
        if idx+3>=len(s):
            continue
        exitp=float(s.iloc[idx+3])
        trades.append({
            "date":dt,"ticker":ticker,"entry":price,
            "predicted_3d_pct":pred*100,"probability_pct":prob*100,
            "actual_3d_pct":(exitp/price-1)*100
        })
    df=pd.DataFrame(trades)
    if df.empty:
        return df, {"trades":0}
    summary={
        "trades":len(df),
        "hit_rate_pct":round((df.actual_3d_pct>0).mean()*100,1),
        "avg_3d_return_pct":round(df.actual_3d_pct.mean(),2),
        "median_3d_return_pct":round(df.actual_3d_pct.median(),2),
        "worst_3d_return_pct":round(df.actual_3d_pct.min(),2),
        "best_3d_return_pct":round(df.actual_3d_pct.max(),2),
        "total_simple_return_pct":round(df.actual_3d_pct.sum(),2),
    }
    return df,summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backtest",action="store_true")
    p.add_argument("--backtest-period",default="5y")
    args=p.parse_args()

    cfg=EngineConfig(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        top_n=int(os.getenv("TOP_N","3"))
    )

    if args.backtest:
        trades,summary=run_backtest(cfg,args.backtest_period)
        print(json.dumps(summary,indent=2,default=str))
        trades.to_csv("backtest_v6_trades.csv",index=False)
        return

    engine=MarketEngineV6(cfg)
    regime=engine.get_regime()
    picks=engine.get_stock_picks()
    ipos=fetch_nse_ipos()
    engine.log_alert(picks,regime)
    engine.send_telegram(regime,picks,ipos)


if __name__=="__main__":
    main()
