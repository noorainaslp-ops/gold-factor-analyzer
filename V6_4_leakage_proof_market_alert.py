#!/usr/bin/env python3
"""
MULTI-FACTOR MARKET ALERT V6.4
Leakage-proof chronological walk-forward + realistic trading audit.

IMPORTANT:
- Primary target: 5-trading-day forward return.
- Five-session purge is applied before every model fit.
- Validation thresholds are frozen before OOS.
- OOS observations are never used to fit/calibrate/select thresholds.
- Non-overlapping portfolio/trade simulation is included.
- LIVE mode uses the same ML signal logic as the backtest.
- Gemini is intentionally NOT enabled yet. A clean hook is provided for V6.5.

Research only; not financial advice.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import warnings
import math

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

warnings.filterwarnings("ignore")

VERSION = "V6.4"
AUDIT = Path(os.getenv("AUDIT_DIR", "audit"))
AUDIT.mkdir(exist_ok=True)

MODE = os.getenv("MODE", "BACKTEST").upper()
PERIOD = os.getenv("BACKTEST_PERIOD", "6y")

# Model / split settings
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "220"))
MIN_TRAIN = int(os.getenv("MIN_TRAIN", "2000"))
VAL_FRAC = float(os.getenv("VALIDATION_FRACTION", "0.20"))
OOS_FRAC = float(os.getenv("OOS_FRACTION", "0.20"))

# Critical leakage-control setting.
# A 5-day target needs the last 5 observations purged from every training set.
HORIZON = 5
PURGE_DAYS = HORIZON

# Costs are round-trip estimates: entry + exit.
COST_BPS = float(os.getenv("COST_BPS", "10"))
SLIP_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))
ROUND_TRIP_COST = 2.0 * (COST_BPS + SLIP_BPS) / 10000.0

# Validation threshold grid.
PROB_GRID = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
RET_GRID = [0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030]

# Portfolio simulation.
CAPITAL = float(os.getenv("CAPITAL", "100000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
PORTFOLIO_HORIZON = int(os.getenv("PORTFOLIO_HORIZON", "5"))

# Live output.
TOP_TRADE = int(os.getenv("TOP_TRADE", "5"))
TOP_WATCH = int(os.getenv("TOP_WATCH", "5"))

# Telegram credentials are environment variables only.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SYMBOLS = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS",
    "KOTAKBANK.NS","INDUSINDBK.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "SHRIRAMFIN.NS","LT.NS","TMPV.NS","TMCV.NS","EICHERMOT.NS","MARUTI.NS",
    "HEROMOTOCO.NS","M&M.NS","TITAN.NS","ASIANPAINT.NS","HINDUNILVR.NS",
    "ITC.NS","NESTLEIND.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS","DIVISLAB.NS",
    "TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS","BHARTIARTL.NS",
    "NTPC.NS","POWERGRID.NS","ONGC.NS","BPCL.NS","COALINDIA.NS","ADANIENT.NS",
    "ADANIPORTS.NS","BEL.NS","HAL.NS","BHEL.NS","TRENT.NS","PIDILITIND.NS",
    "SIEMENS.NS","ABB.NS","GRASIM.NS","ULTRACEMCO.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "HINDALCO.NS","IOC.NS","VEDL.NS","DLF.NS","LODHA.NS","INDIGO.NS","ETERNAL.NS",
    "NAUKRI.NS","COFORGE.NS","JIOFIN.NS","IRFC.NS","IREDA.NS","POLYCAB.NS"
]

FEATURES = [
    "ret1","ret3","ret5","ret10","ret20",
    "dist20","dist50","dist200","rsi","atr_pct",
    "vol_ratio","vol20","rel20","mkt_ret5","mkt_ret20",
    "mkt_dist50","mkt_vol20","regime","range_pct",
    "close_location","breakout20"
]


def clean(x: pd.DataFrame) -> pd.DataFrame:
    if x is None or x.empty:
        return pd.DataFrame()

    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)

    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in x.columns for c in need):
        return pd.DataFrame()

    x = x[need].copy()
    for c in need:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])

    try:
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
    except Exception:
        pass

    x = x.sort_index()
    return x[~x.index.duplicated(keep="last")]


def download_daily(symbol: str, period: str = PERIOD) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return clean(raw)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(x: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = x["Close"].shift(1)
    tr = pd.concat(
        [
            x["High"] - x["Low"],
            (x["High"] - pc).abs(),
            (x["Low"] - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def features(stock: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Create ONLY information available on each signal date.

    Forward returns are targets and are never included in FEATURES.
    Market data are aligned to the stock's exact dates.
    """
    s = stock.copy()
    m = market.copy()

    try:
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        if getattr(m.index, "tz", None) is not None:
            m.index = m.index.tz_localize(None)
    except Exception:
        pass

    # Only carry market information forward from dates that already existed.
    m = m.reindex(s.index).ffill()

    c = s["Close"]
    v = s["Volume"]
    mc = m["Close"]

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    ms50 = mc.rolling(50).mean()

    ret1 = c.pct_change()
    ret3 = c.pct_change(3)
    ret5 = c.pct_change(5)
    ret10 = c.pct_change(10)
    ret20 = c.pct_change(20)

    mr5 = mc.pct_change(5)
    mr20 = mc.pct_change(20)

    vol20 = ret1.rolling(20).std()
    mvol20 = mr5.rolling(20).std()

    high20 = s["High"].rolling(20).max()

    regime = np.select(
        [
            (mc > ms50 * 1.005),
            (mc < ms50 * 0.995),
        ],
        [1.0, -1.0],
        default=0.0,
    )

    f = pd.DataFrame(
        {
            "ret1": ret1,
            "ret3": ret3,
            "ret5": ret5,
            "ret10": ret10,
            "ret20": ret20,
            "dist20": c / sma20 - 1,
            "dist50": c / sma50 - 1,
            "dist200": c / sma200 - 1,
            "rsi": rsi(c),
            "atr_pct": atr(s) / c,
            "vol_ratio": v / v.rolling(20).mean(),
            "vol20": vol20,
            "rel20": ret20 - mr20,
            "mkt_ret5": mr5,
            "mkt_ret20": mr20,
            "mkt_dist50": mc / ms50 - 1,
            "mkt_vol20": mvol20,
            "regime": regime,
            "range_pct": (s["High"] - s["Low"]) / c,
            "close_location": (
                (c - s["Low"]) /
                (s["High"] - s["Low"]).replace(0, np.nan)
            ),
            "breakout20": c / high20.shift(1) - 1,
            "close": c,
            "market_close": mc,
        },
        index=s.index,
    )

    for h in (1, 3, 5):
        f[f"ret{h}_fwd"] = c.shift(-h) / c - 1

    f["y5"] = (f["ret5_fwd"] > 0).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)


def clf_model():
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.35,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=17,
                ),
            ),
        ]
    )


def ret_model():
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=8.0)),
        ]
    )


def fit(train: pd.DataFrame):
    train = train.dropna(subset=["y5", "ret5"])

    if len(train) < MIN_TRAIN or train.y5.nunique() < 2:
        return None, None

    c = clf_model()
    r = ret_model()

    c.fit(train[FEATURES], train.y5.astype(int))
    r.fit(train[FEATURES], train.ret5)

    return c, r


def predict(c, r, d: pd.DataFrame):
    p = c.predict_proba(d[FEATURES])[:, 1]
    q = r.predict(d[FEATURES])
    return np.clip(p, 0.05, 0.95), q


def calibrate_sigmoid(p, y):
    """Calibration learned only from validation predictions."""
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).reshape(-1, 1)

    cal = LogisticRegression(C=1.0, max_iter=1000, random_state=17)
    cal.fit(z, np.asarray(y).astype(int))
    return cal


def cal_predict(cal, p):
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return np.clip(cal.predict_proba(z)[:, 1], 0.05, 0.95)


def choose_thresholds(v: pd.DataFrame):
    """
    Thresholds are optimized ONLY on validation data.
    OOS is never touched here.
    """
    best = None

    for pm in PROB_GRID:
        for rm in RET_GRID:
            g = v[
                (v.p_cal >= pm)
                & (v.pred_ret >= rm)
                & (v.dist50 >= -0.025)
                & (v.rsi.between(38, 70))
                & (v.vol_ratio >= 0.65)
            ]

            if len(g) < 100:
                continue

            x = g.net5
            w = x[x > 0]
            l = x[x <= 0]

            pf = (
                w.sum() / abs(l.sum())
                if len(l) and l.sum() < 0
                else 0
            )

            score = (
                100 * x.mean()
                + 0.40 * np.log1p(max(pf, 0))
                + 0.10 * ((x > 0).mean() - 0.5)
            )

            cand = (
                score,
                pm,
                rm,
                len(g),
                x.mean(),
                (x > 0).mean(),
                pf,
            )

            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return {
            "pmin": 0.58,
            "rmin": 0.0010,
            "n": 0,
            "avg": np.nan,
            "win": np.nan,
            "pf": np.nan,
        }

    return {
        "pmin": best[1],
        "rmin": best[2],
        "n": best[3],
        "avg": best[4],
        "win": best[5],
        "pf": best[6],
    }


def actions(d: pd.DataFrame, t: dict) -> pd.DataFrame:
    out = []

    for _, x in d.iterrows():
        if not np.isfinite(x.p_cal) or not np.isfinite(x.pred_ret):
            out.append("WAIT")
            continue

        risk_ok = (
            x.rsi >= 38
            and x.rsi <= 70
            and x.vol_ratio >= 0.65
            and x.dist50 >= -0.025
        )

        if (
            x.p_cal >= t["pmin"]
            and x.pred_ret >= t["rmin"]
            and risk_ok
        ):
            out.append("TRADE")
        elif (
            x.p_cal >= max(0.52, t["pmin"] - 0.04)
            and x.pred_ret >= 0
            and x.rsi >= 35
            and x.rsi <= 72
        ):
            out.append("WATCH")
        else:
            out.append("WAIT")

    z = d.copy()
    z["action"] = out
    return z


def performance(d: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if d.empty:
        return pd.DataFrame()

    for a, g in d.groupby("action"):
        for h in (1, 3, 5):
            col = f"net{h}"
            x = g[col].dropna()

            if x.empty:
                continue

            w = x[x > 0]
            l = x[x <= 0]

            pf = (
                w.sum() / abs(l.sum())
                if len(l) and l.sum() < 0
                else np.nan
            )

            rows.append(
                {
                    "selection": a,
                    "horizon": h,
                    "observations": len(x),
                    "win_rate": (x > 0).mean(),
                    "average_net_return": x.mean(),
                    "median_net_return": x.median(),
                    "average_winner": w.mean() if len(w) else np.nan,
                    "average_loser": l.mean() if len(l) else np.nan,
                    "profit_factor": pf,
                    "best": x.max(),
                    "worst": x.min(),
                }
            )

    return pd.DataFrame(rows)


def calibration_table(d: pd.DataFrame) -> pd.DataFrame:
    x = d.dropna(subset=["p_cal", "y5", "net5"]).copy()

    if x.empty:
        return pd.DataFrame()

    bins = [-0.01, .40, .45, .50, .55, .60, .65, .70, .75, 1.01]
    labels = [
        "<40%", "40-45%", "45-50%", "50-55%", "55-60%",
        "60-65%", "65-70%", "70-75%", "75%+"
    ]

    x["bucket"] = pd.cut(
        x.p_cal,
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []

    for b, g in x.groupby("bucket", observed=False):
        if g.empty:
            continue

        rows.append(
            {
                "probability_bucket": str(b),
                "observations": len(g),
                "average_model_probability": g.p_cal.mean(),
                "actual_win_rate": g.y5.mean(),
                "average_net_return": g.net5.mean(),
            }
        )

    return pd.DataFrame(rows)


def prediction_metrics(d: pd.DataFrame) -> pd.DataFrame:
    metrics = []

    for name, x in d:
        q = x.dropna(
            subset=["p_cal", "y5", "pred_ret", "ret5"]
        )

        if q.empty:
            continue

        metrics.append(
            {
                "sample": name,
                "observations": len(q),
                "brier_score": brier_score_loss(
                    q.y5.astype(int),
                    q.p_cal,
                ),
                "log_loss": log_loss(
                    q.y5.astype(int),
                    np.clip(q.p_cal, 1e-6, 1 - 1e-6),
                    labels=[0, 1],
                ),
                "return_mae": mean_absolute_error(
                    q.ret5,
                    q.pred_ret,
                ),
                "directional_accuracy": (
                    (q.pred_ret > 0) == (q.ret5 > 0)
                ).mean(),
                "mean_predicted_return": q.pred_ret.mean(),
                "mean_actual_return": q.ret5.mean(),
            }
        )

    return pd.DataFrame(metrics)


def non_overlapping_trade_test(oos: pd.DataFrame) -> pd.DataFrame:
    """
    Event-based test.

    A new 5-session trade can only be opened after the previous
    selected trade's 5-session holding window has completed.

    This prevents overlapping observations from being counted
    as independent sequential trades for this audit.
    """
    if oos.empty:
        return pd.DataFrame()

    x = oos[oos.action == "TRADE"].copy()
    if x.empty:
        return pd.DataFrame()

    x = x.sort_values(["date", "ticker"]).reset_index(drop=True)

    selected = []
    last_trade_date = None

    for _, row in x.iterrows():
        if last_trade_date is None:
            selected.append(row)
            last_trade_date = row["date"]
            continue

        # Signal dates are trading dates. Require at least HORIZON
        # trading-date observations between selected entries.
        if (row["date"] - last_trade_date).days >= HORIZON:
            selected.append(row)
            last_trade_date = row["date"]

    z = pd.DataFrame(selected)

    if z.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "trades": len(z),
                "win_rate_5d": (z.net5 > 0).mean(),
                "average_5d_net": z.net5.mean(),
                "median_5d_net": z.net5.median(),
                "best_5d": z.net5.max(),
                "worst_5d": z.net5.min(),
                "gross_sum_return": z.ret5.sum(),
                "net_sum_return": z.net5.sum(),
            }
        ]
    )


def portfolio_backtest(oos: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative event-driven portfolio simulation.

    - Starts with CAPITAL.
    - At each signal date, ranks TRADE candidates by calibrated
      probability, then predicted return.
    - Opens up to MAX_POSITIONS.
    - Equal allocation.
    - Holds for exactly PORTFOLIO_HORIZON trading sessions.
    - No overlapping positions beyond MAX_POSITIONS.
    - Round-trip costs are already embedded in net5.
    """
    if oos.empty:
        return pd.DataFrame()

    dates = sorted(oos.date.unique())
    cash = CAPITAL
    positions = []
    equity_curve = []

    for idx, date in enumerate(dates):
        # Mark positions that reach their exit date.
        remaining = []
        for pos in positions:
            if idx >= pos["exit_idx"]:
                cash += pos["allocation"] * (1 + pos["net_return"])
            else:
                remaining.append(pos)
        positions = remaining

        # Open new positions only if capacity exists.
        capacity = MAX_POSITIONS - len(positions)

        if capacity > 0:
            cur = oos[
                (oos.date == date)
                & (oos.action == "TRADE")
            ].copy()

            if not cur.empty:
                cur = cur.sort_values(
                    ["p_cal", "pred_ret"],
                    ascending=False,
                )

                # Equal allocation from currently available cash.
                slots = min(capacity, len(cur))
                allocation = cash / max(slots, 1)

                opened = 0
                for _, row in cur.iterrows():
                    if opened >= slots:
                        break
                    if allocation <= 0:
                        break

                    exit_idx = idx + PORTFOLIO_HORIZON
                    if exit_idx >= len(dates):
                        # Cannot complete a full holding period.
                        continue

                    positions.append(
                        {
                            "ticker": row.ticker,
                            "entry_date": date,
                            "exit_idx": exit_idx,
                            "allocation": allocation,
                            "net_return": float(row.net5),
                        }
                    )

                    cash -= allocation
                    opened += 1

        marked = cash + sum(
            pos["allocation"] for pos in positions
        )
        equity_curve.append(
            {
                "date": date,
                "equity": marked,
                "cash": cash,
                "open_positions": len(positions),
            }
        )

    curve = pd.DataFrame(equity_curve)

    if curve.empty:
        return pd.DataFrame()

    curve["daily_return"] = curve.equity.pct_change()

    running_max = curve.equity.cummax()
    curve["drawdown"] = curve.equity / running_max - 1

    final_equity = float(curve.equity.iloc[-1])
    total_return = final_equity / CAPITAL - 1

    # Annualize using actual number of trading dates.
    years = max(len(curve) / 252.0, 1 / 252.0)
    cagr = (final_equity / CAPITAL) ** (1 / years) - 1

    daily = curve.daily_return.dropna()
    sharpe = (
        np.sqrt(252) * daily.mean() / daily.std()
        if len(daily) > 1 and daily.std() > 0
        else np.nan
    )

    downside = daily[daily < 0]
    sortino = (
        np.sqrt(252) * daily.mean() / downside.std()
        if len(downside) > 1 and downside.std() > 0
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "starting_capital": CAPITAL,
                "ending_equity": final_equity,
                "total_return": total_return,
                "CAGR": cagr,
                "max_drawdown": curve.drawdown.min(),
                "Sharpe": sharpe,
                "Sortino": sortino,
                "trading_dates": len(curve),
            }
        ]
    )


def build_dataset():
    print(f"Starting {VERSION} data build...")
    print(f"Backtest period: {PERIOD}")

    market = download_daily("^NSEI")
    if market.empty:
        raise RuntimeError("NIFTY data unavailable.")

    rows = []
    successful = 0

    for i, sym in enumerate(SYMBOLS, 1):
        print(f"Loading [{i}/{len(SYMBOLS)}] {sym}")

        try:
            stock = download_daily(sym)

            if stock.empty or len(stock) < MIN_HISTORY:
                print(f"WARNING: insufficient history for {sym}; skipping.")
                continue

            f = features(stock, market)
            successful += 1

        except Exception as exc:
            print(f"WARNING: {sym} failed: {exc}")
            continue

        # The final HORIZON observations cannot have a fully known target.
        for j in range(MIN_HISTORY - 1, len(f) - HORIZON):
            x = f.iloc[j]

            if x[FEATURES].isna().any():
                continue

            r = {
                "ticker": sym,
                "date": f.index[j],
                "close": float(x.close),
                "market_close": float(x.market_close),
            }

            for col in FEATURES:
                r[col] = float(x[col])

            for h in (1, 3, 5):
                r[f"ret{h}"] = float(x[f"ret{h}_fwd"])

            r["y5"] = float(x.y5)
            rows.append(r)

    d = pd.DataFrame(rows)

    if d.empty:
        raise RuntimeError("No candidate observations generated.")

    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)

    return d, successful


def purge_training(prior: pd.DataFrame, prediction_date) -> pd.DataFrame:
    """
    CRITICAL FIX.

    For a prediction at T with a 5-session target, the latest usable
    training observation must be at least 5 trading observations before T.

    Using only date arithmetic is unsafe around weekends/holidays, so
    the purge is based on the ordered unique signal dates.
    """
    dates = sorted(prior.date.unique())

    if not dates:
        return prior.iloc[0:0].copy()

    # Remove the last PURGE_DAYS signal dates from prior.
    cutoff_dates = dates[:-PURGE_DAYS] if len(dates) > PURGE_DAYS else []

    if not cutoff_dates:
        return prior.iloc[0:0].copy()

    cutoff = cutoff_dates[-1]
    return prior[prior.date <= cutoff].copy()


def chronological_split(d: pd.DataFrame):
    dates = sorted(d.date.unique())
    n = len(dates)

    val_start = dates[int(n * (1 - OOS_FRAC - VAL_FRAC))]
    oos_start = dates[int(n * (1 - OOS_FRAC))]

    dev = d[d.date < val_start].copy()
    val = d[(d.date >= val_start) & (d.date < oos_start)].copy()
    oos = d[d.date >= oos_start].copy()

    return dev, val, oos, val_start, oos_start


def run_backtest():
    d, successful = build_dataset()

    dev, val, oos, val_start, oos_start = chronological_split(d)

    if len(dev) < MIN_TRAIN:
        raise RuntimeError(
            f"Development sample too small: {len(dev)}"
        )

    # ------------------------------------------------------------
    # DEVELOPMENT -> VALIDATION
    # ------------------------------------------------------------
    # Purge the end of development so validation predictions are
    # never trained on labels that extend into validation.
    dev_fit = purge_training(dev, val.date.min())

    print(
        f"\nDevelopment observations: {len(dev)}"
        f"\nPurged development training observations: {len(dev_fit)}"
    )

    c, r = fit(dev_fit)
    if c is None:
        raise RuntimeError("Unable to fit development model.")

    val["p_raw"], val["pred_ret"] = predict(c, r, val)

    # Calibration uses validation predictions only.
    cal = calibrate_sigmoid(val.p_raw, val.y5)
    val["p_cal"] = cal_predict(cal, val.p_raw)

    for h in (1, 3, 5):
        val[f"net{h}"] = val[f"ret{h}"] - ROUND_TRIP_COST

    thresholds = choose_thresholds(val)

    print("\nVALIDATION-SELECTED THRESHOLDS:")
    print(thresholds)

    # ------------------------------------------------------------
    # FINAL PRE-OOS MODEL
    # ------------------------------------------------------------
    # Purge the final 5 signal dates before OOS.
    pre = d[d.date < oos_start].copy()
    pre_fit = purge_training(pre, oos_start)

    print(
        f"\nPre-OOS observations: {len(pre)}"
        f"\nPurged pre-OOS training observations: {len(pre_fit)}"
    )

    cf, rf = fit(pre_fit)
    if cf is None:
        raise RuntimeError("Unable to fit final pre-OOS model.")

    oos["p_raw"], oos["pred_ret"] = predict(cf, rf, oos)

    # Calibration remains frozen from validation.
    oos["p_cal"] = cal_predict(cal, oos.p_raw)

    for h in (1, 3, 5):
        oos[f"net{h}"] = oos[f"ret{h}"] - ROUND_TRIP_COST

    oos = actions(oos, thresholds)

    # ------------------------------------------------------------
    # TRUE CHRONOLOGICAL WALK-FORWARD DIAGNOSTIC
    # ------------------------------------------------------------
    print("\nRunning PURGED chronological walk-forward diagnostics...")

    dates = sorted(d.date.unique())
    wf_parts = []

    for k, date in enumerate(dates):
        cur = d[d.date == date].copy()
        prior = d[d.date < date].copy()

        if len(prior) < MIN_TRAIN:
            continue

        prior_fit = purge_training(prior, date)

        if len(prior_fit) < MIN_TRAIN:
            continue

        # Keep the historical rolling window, but purge it first.
        prior_fit = prior_fit.tail(12000)

        cm, rm = fit(prior_fit)

        if cm is None:
            continue

        cur["p_raw"], cur["pred_ret"] = predict(cm, rm, cur)

        # Frozen calibration + thresholds from validation only.
        cur["p_cal"] = cal_predict(cal, cur.p_raw)

        for h in (1, 3, 5):
            cur[f"net{h}"] = cur[f"ret{h}"] - ROUND_TRIP_COST

        cur = actions(cur, thresholds)
        wf_parts.append(cur)

        if k == 0 or k % 100 == 0 or k == len(dates) - 1:
            print(
                f"Purged walk-forward date [{k+1}/{len(dates)}]"
            )

    wf = (
        pd.concat(wf_parts, ignore_index=True)
        if wf_parts
        else pd.DataFrame()
    )

    # ------------------------------------------------------------
    # BENCHMARK
    # ------------------------------------------------------------
    benchmark = (
        d.groupby("date")
        .market_close
        .first()
        .sort_index()
        .pct_change(5)
        .dropna()
    )

    # ------------------------------------------------------------
    # REPORTS
    # ------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    outputs = {
        f"walkforward_v6_4_{ts}.csv": wf,
        f"validation_v6_4_{ts}.csv": val,
        f"oos_v6_4_{ts}.csv": oos,
        f"action_group_performance_v6_4_{ts}.csv": performance(wf),
        f"oos_performance_v6_4_{ts}.csv": performance(oos),
        f"probability_calibration_v6_4_{ts}.csv": calibration_table(wf),
        f"oos_probability_calibration_v6_4_{ts}.csv": calibration_table(oos),
        f"threshold_selection_v6_4_{ts}.csv": pd.DataFrame([thresholds]),
        f"nonoverlap_oos_v6_4_{ts}.csv": non_overlapping_trade_test(oos),
        f"portfolio_oos_v6_4_{ts}.csv": portfolio_backtest(oos),
    }

    metrics = prediction_metrics(
        [
            ("WALK_FORWARD", wf),
            ("OOS", oos),
        ]
    )

    outputs[f"prediction_metrics_v6_4_{ts}.csv"] = metrics

    outputs[f"nifty_benchmark_v6_4_{ts}.csv"] = pd.DataFrame(
        [
            {
                "observations": len(benchmark),
                "win_rate": (benchmark > 0).mean(),
                "average_5d_return": benchmark.mean(),
                "median_5d_return": benchmark.median(),
            }
        ]
    )

    for name, frame in outputs.items():
        frame.to_csv(AUDIT / name, index=False)

    print("\n" + "=" * 70)
    print(f"{VERSION} PURGED CHRONOLOGICAL WALK-FORWARD BACKTEST")
    print("=" * 70)

    print(f"Total candidate observations: {len(d)}")
    print(f"Successful symbols: {successful}")
    print(f"Unique symbols: {d.ticker.nunique()}")
    print(f"Unique signal dates: {d.date.nunique()}")
    print(f"Development observations: {len(dev)}")
    print(f"Validation observations: {len(val)}")
    print(f"OOS observations: {len(oos)}")
    print(f"Validation start: {val_start}")
    print(f"OOS start: {oos_start}")
    print(f"Round-trip cost assumption: {ROUND_TRIP_COST*100:.3f}%")

    print("\nFROZEN VALIDATION THRESHOLDS:")
    print(thresholds)

    print("\nOOS ACTION COUNTS:")
    print(
        oos.action.value_counts()
        .rename_axis("action")
        .to_frame("count")
        .to_string()
    )

    print("\nOOS PERFORMANCE:")
    print(
        performance(oos).to_string(index=False)
        if not oos.empty
        else "None"
    )

    print("\nNON-OVERLAPPING OOS TRADE TEST:")
    no = non_overlapping_trade_test(oos)
    print(no.to_string(index=False) if not no.empty else "None")

    print("\nPORTFOLIO OOS TEST:")
    po = portfolio_backtest(oos)
    print(po.to_string(index=False) if not po.empty else "None")

    print("\nOOS PROBABILITY CALIBRATION:")
    cal_oos = calibration_table(oos)
    print(cal_oos.to_string(index=False) if not cal_oos.empty else "None")

    print("\nPREDICTION METRICS:")
    print(metrics.to_string(index=False) if not metrics.empty else "None")

    print("\nNIFTY 5-DAY BENCHMARK:")
    print(
        outputs[
            f"nifty_benchmark_v6_4_{ts}.csv"
        ].to_string(index=False)
    )

    print("\nFILES CREATED:")
    for name in outputs:
        print(AUDIT / name)

    print("\n" + "=" * 70)
    print(f"{VERSION} BACKTEST COMPLETED")
    print("=" * 70)

    print(
        "\nIMPORTANT:\n"
        "1. The last 5 signal dates are purged from every training fit.\n"
        "2. Validation thresholds are selected only on validation data.\n"
        "3. OOS data are not used for fitting, calibration, or threshold selection.\n"
        "4. Non-overlapping and portfolio tests are reported separately.\n"
        "5. This is still a historical simulation; it does not guarantee future profit.\n"
    )


# -----------------------------------------------------------------
# LIVE SIGNAL ENGINE
# -----------------------------------------------------------------

def live_fit_and_score():
    """
    Live mode:
    - downloads current daily data
    - uses only the latest COMPLETED daily bar
    - trains on historical observations
    - purges the last HORIZON dates from training
    - applies fixed thresholds supplied through environment variables
      or the conservative defaults.

    For production deployment, thresholds should be copied from the
    most recent validated V6.4 audit and frozen until a new validation
    run explicitly approves a change.
    """
    market = download_daily("^NSEI")
    if market.empty:
        raise RuntimeError("NIFTY data unavailable.")

    rows = []

    for sym in SYMBOLS:
        try:
            stock = download_daily(sym)
            if stock.empty or len(stock) < MIN_HISTORY:
                continue

            f = features(stock, market)

            # Exclude the last bar if it is the current incomplete day.
            # GitHub Actions should normally run after the market closes,
            # but this makes the live path safer.
            now = pd.Timestamp.now(tz=None).normalize()
            f = f[f.index < now]

            # Need enough history and a fully known historical target.
            for j in range(MIN_HISTORY - 1, len(f) - HORIZON):
                x = f.iloc[j]

                if x[FEATURES].isna().any():
                    continue

                r = {
                    "ticker": sym,
                    "date": f.index[j],
                    "close": float(x.close),
                    "market_close": float(x.market_close),
                    "y5": float(x.y5),
                }

                for col in FEATURES:
                    r[col] = float(x[col])

                r["ret5"] = float(x["ret5_fwd"])
                rows.append(r)

        except Exception as exc:
            print(f"LIVE WARNING {sym}: {exc}")

    d = pd.DataFrame(rows)

    if d.empty:
        raise RuntimeError("No live training observations generated.")

    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)

    signal_date = d.date.max()
    current = d[d.date == signal_date].copy()

    # Purge training labels that are not fully known by signal_date.
    train = d[d.date < signal_date].copy()
    train = purge_training(train, signal_date)

    if len(train) < MIN_TRAIN:
        raise RuntimeError(
            f"Insufficient purged live training observations: {len(train)}"
        )

    c, r = fit(train)
    if c is None:
        raise RuntimeError("Live model fit failed.")

    current["p_raw"], current["pred_ret"] = predict(c, r, current)

    # For live mode, use frozen thresholds from environment.
    # These should be populated from the latest validated audit.
    pmin = float(os.getenv("LIVE_PMIN", "0.50"))
    rmin = float(os.getenv("LIVE_RMIN", "0.0020"))

    frozen = {"pmin": pmin, "rmin": rmin}

    # No new calibration is invented in live mode.
    # Raw model probability is reported as raw model probability.
    current["p_cal"] = current["p_raw"]
    current = actions(current, frozen)

    current = current.sort_values(
        ["action", "p_cal", "pred_ret"],
        ascending=[True, False, False],
    )

    trades = current[current.action == "TRADE"].sort_values(
        ["p_cal", "pred_ret"],
        ascending=False,
    ).head(TOP_TRADE)

    watch = current[current.action == "WATCH"].sort_values(
        ["p_cal", "pred_ret"],
        ascending=False,
    ).head(TOP_WATCH)

    lines = [
        f"MULTI-FACTOR MARKET ALERT {VERSION}",
        datetime.now().astimezone().strftime("%d %b %Y, %H:%M %Z"),
        "",
        f"SIGNAL DATE: {signal_date}",
        "--- TOP SHORT-TERM ML TRADE SETUPS (1–5 sessions) ---",
    ]

    if trades.empty:
        lines.extend(
            [
                "",
                "NO VALID LONG TRADE TODAY",
                "No candidate currently satisfies the frozen V6.4 filters.",
            ]
        )
    else:
        for _, row in trades.iterrows():
            lines.extend(
                [
                    "",
                    f"{row.ticker.replace('.NS','')} — TRADE CANDIDATE",
                    f"Price: ₹{row.close:,.2f}",
                    f"P(UP): {row.p_cal*100:.1f}% (raw model; frozen live calibration not embedded)",
                    f"Predicted 5D return: {row.pred_ret*100:.2f}%",
                    f"RSI: {row.rsi:.1f} | Volume: {row.vol_ratio:.2f}x",
                    "NOTE: Confirm current live price before execution.",
                ]
            )

    lines.extend(["", "--- BEST WATCHLIST SETUPS ---"])

    if watch.empty:
        lines.append("None.")
    else:
        for n, (_, row) in enumerate(watch.iterrows(), 1):
            lines.extend(
                [
                    "",
                    f"{n}. {row.ticker.replace('.NS','')} — WATCH",
                    f"Price: ₹{row.close:,.2f}",
                    f"P(UP): {row.p_cal*100:.1f}% | Predicted 5D return: {row.pred_ret*100:.2f}%",
                    f"RSI: {row.rsi:.1f} | Volume: {row.vol_ratio:.2f}x",
                ]
            )

    lines.extend(
        [
            "",
            "--- V6.4 VALIDATION / SAFETY ---",
            "Five-session training-label purge is enabled.",
            "OOS observations are not used in this live fit.",
            "This is a probabilistic research signal, not a guarantee.",
            "Gemini/news layer is NOT enabled in V6.4.",
        ]
    )

    message = "\n".join(lines)
    print("\n" + message)

    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        url = (
            "https://api.telegram.org/"
            f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        )
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
            },
            timeout=20,
        )
        response.raise_for_status()
        print("Telegram message sent.")
    else:
        print("Telegram credentials not configured; report printed only.")


# -----------------------------------------------------------------
# GEMINI HOOK — intentionally disabled in V6.4
# -----------------------------------------------------------------

def gemini_signal_hook(*args, **kwargs):
    """
    Reserved for V6.5.

    Do NOT let an LLM replace the quantitative model.
    The intended V6.5 design is:

        V6.4 quantitative prediction
                    +
        Gemini-derived news/event features
                    ↓
             independently
             validated ensemble

    Gemini should return structured information features, not a
    free-form "BUY/SELL" opinion.

    This function intentionally raises an error so Gemini cannot
    accidentally become part of the live V6.4 decision path.
    """
    raise RuntimeError(
        "Gemini integration is intentionally disabled in V6.4. "
        "Validate V6.4 first, then add Gemini as a separately tested "
        "information layer in V6.5."
    )


if __name__ == "__main__":
    if MODE == "LIVE":
        live_fit_and_score()
    else:
        run_backtest()
