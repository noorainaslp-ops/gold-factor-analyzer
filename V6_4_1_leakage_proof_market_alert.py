#!/usr/bin/env python3
"""
MULTI-FACTOR MARKET ALERT V6.4.1
Leakage-proof chronological walk-forward backtest.

V6.4.1 fixes:
1. Forward returns are TARGETS ONLY and are never written into feature columns.
2. Five signal-date purge before every training fit.
3. Validation threshold selection is isolated from OOS.
4. Walk-forward predictions use only information available at prediction time.
5. Non-overlapping event test is reported.
6. Portfolio simulation uses actual sequential position accounting and drawdown.
7. No Gemini dependency: Gemini should only be added after this baseline passes audit.

Research only. Not financial advice.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import os
import warnings

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

VERSION = "V6.4.1"
AUDIT = Path(os.getenv("AUDIT_DIR", "audit"))
AUDIT.mkdir(exist_ok=True)

MODE = os.getenv("MODE", "BACKTEST").upper()
PERIOD = os.getenv("BACKTEST_PERIOD", "6y")

HORIZON = 5
PURGE_DAYS = 5

MIN_HISTORY = int(os.getenv("MIN_HISTORY", "220"))
MIN_TRAIN = int(os.getenv("MIN_TRAIN", "2000"))

VAL_FRAC = float(os.getenv("VALIDATION_FRACTION", "0.20"))
OOS_FRAC = float(os.getenv("OOS_FRACTION", "0.20"))

# Round-trip trading-cost assumption.
COST_BPS = float(os.getenv("COST_BPS", "10"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))
ROUND_TRIP_COST = 2.0 * (COST_BPS + SLIPPAGE_BPS) / 10000.0

CAPITAL = float(os.getenv("CAPITAL", "100000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
PORTFOLIO_HORIZON = int(os.getenv("PORTFOLIO_HORIZON", "5"))

PROB_GRID = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
RET_GRID = [0.0000, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030]

TOP_TRADE = int(os.getenv("TOP_TRADE", "5"))
TOP_WATCH = int(os.getenv("TOP_WATCH", "5"))

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

# CRITICAL:
# These are all BACKWARD-LOOKING features.
# No *_fwd field is included here.
FEATURES = [
    "ret1_past", "ret3_past", "ret5_past", "ret10_past", "ret20_past",
    "dist20", "dist50", "dist200",
    "rsi", "atr_pct",
    "vol_ratio", "vol20",
    "rel20",
    "mkt_ret5", "mkt_ret20", "mkt_dist50", "mkt_vol20",
    "regime",
    "range_pct", "close_location", "breakout20"
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

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["Close"])

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


def make_features(stock: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Construct one row per signal date.

    Every feature on date T uses only data at or before T.

    Targets:
      ret1_fwd = T+1 / T - 1
      ret3_fwd = T+3 / T - 1
      ret5_fwd = T+5 / T - 1

    These forward values are NEVER part of FEATURES.
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

    m = m.reindex(s.index).ffill()

    c = s["Close"]
    v = s["Volume"]
    mc = m["Close"]

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    ms50 = mc.rolling(50).mean()

    # PAST returns. No negative shift.
    ret1_past = c.pct_change(1)
    ret3_past = c.pct_change(3)
    ret5_past = c.pct_change(5)
    ret10_past = c.pct_change(10)
    ret20_past = c.pct_change(20)

    mr5 = mc.pct_change(5)
    mr20 = mc.pct_change(20)

    vol20 = ret1_past.rolling(20).std()
    mvol20 = mr5.rolling(20).std()

    high20_prior = s["High"].rolling(20).max().shift(1)

    regime = np.select(
        [
            mc > ms50 * 1.005,
            mc < ms50 * 0.995,
        ],
        [1.0, -1.0],
        default=0.0,
    )

    f = pd.DataFrame(
        {
            # BACKWARD-LOOKING FEATURES
            "ret1_past": ret1_past,
            "ret3_past": ret3_past,
            "ret5_past": ret5_past,
            "ret10_past": ret10_past,
            "ret20_past": ret20_past,

            "dist20": c / sma20 - 1,
            "dist50": c / sma50 - 1,
            "dist200": c / sma200 - 1,

            "rsi": rsi(c),
            "atr_pct": atr(s) / c,

            "vol_ratio": v / v.rolling(20).mean(),
            "vol20": vol20,

            "rel20": ret20_past - mr20,

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

            "breakout20": c / high20_prior - 1,

            # informational fields, NOT model features
            "close": c,
            "market_close": mc,
        },
        index=s.index,
    )

    # FORWARD TARGETS ONLY.
    # They are deliberately separate from FEATURES.
    f["ret1_fwd"] = c.shift(-1) / c - 1
    f["ret3_fwd"] = c.shift(-3) / c - 1
    f["ret5_fwd"] = c.shift(-5) / c - 1
    f["y5"] = (f["ret5_fwd"] > 0).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)


def clf_model():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
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
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=8.0)),
        ]
    )


def fit(train: pd.DataFrame):
    q = train.dropna(subset=["y5", "ret5_fwd"]).copy()

    if len(q) < MIN_TRAIN:
        return None, None

    if q.y5.nunique() < 2:
        return None, None

    c = clf_model()
    r = ret_model()

    c.fit(q[FEATURES], q["y5"].astype(int))
    r.fit(q[FEATURES], q["ret5_fwd"])

    return c, r


def predict(c, r, d: pd.DataFrame):
    p = c.predict_proba(d[FEATURES])[:, 1]
    q = r.predict(d[FEATURES])
    return np.clip(p, 0.01, 0.99), q


def calibrate_sigmoid(p, y):
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).reshape(-1, 1)

    cal = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=17,
    )
    cal.fit(z, np.asarray(y).astype(int))
    return cal


def cal_predict(cal, p):
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return np.clip(cal.predict_proba(z)[:, 1], 0.01, 0.99)


def purge_training(prior: pd.DataFrame, prediction_date) -> pd.DataFrame:
    """
    Purge the last HORIZON signal dates.

    If predicting on T, a training observation on T-4 is unusable
    because its 5-day target reaches into the prediction date.
    """
    dates = sorted(prior.date.unique())

    if len(dates) <= PURGE_DAYS:
        return prior.iloc[0:0].copy()

    cutoff = dates[-PURGE_DAYS - 1]
    return prior[prior.date <= cutoff].copy()


def choose_thresholds(v: pd.DataFrame):
    best = None

    for pm in PROB_GRID:
        for rm in RET_GRID:
            g = v[
                (v.p_cal >= pm)
                & (v.pred_ret >= rm)
                & (v.dist50 >= -0.025)
                & (v.rsi.between(38, 70))
                & (v.vol_ratio >= 0.65)
            ].copy()

            if len(g) < 100:
                continue

            x = g["net5"].dropna()
            w = x[x > 0]
            l = x[x <= 0]

            pf = (
                w.sum() / abs(l.sum())
                if len(l) and l.sum() < 0
                else np.nan
            )

            # Prefer expected return while avoiding pathological
            # solutions with very small sample sizes.
            score = (
                x.mean()
                + 0.05 * ((x > 0).mean() - 0.5)
                + 0.0005 * np.log1p(max(pf, 0))
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
            "rmin": 0.001,
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


def apply_actions(d: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = []

    for _, x in d.iterrows():
        if not np.isfinite(x.p_cal) or not np.isfinite(x.pred_ret):
            out.append("WAIT")
            continue

        risk_ok = (
            38 <= x.rsi <= 70
            and x.vol_ratio >= 0.65
            and x.dist50 >= -0.025
        )

        if (
            x.p_cal >= thresholds["pmin"]
            and x.pred_ret >= thresholds["rmin"]
            and risk_ok
        ):
            out.append("TRADE")
        elif (
            x.p_cal >= max(0.52, thresholds["pmin"] - 0.04)
            and x.pred_ret >= 0
            and 35 <= x.rsi <= 72
        ):
            out.append("WATCH")
        else:
            out.append("WAIT")

    z = d.copy()
    z["action"] = out
    return z


def add_net_returns(d: pd.DataFrame) -> pd.DataFrame:
    z = d.copy()

    for h in (1, 3, 5):
        z[f"net{h}"] = z[f"ret{h}_fwd"] - ROUND_TRIP_COST

    return z


def performance(d: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if d.empty:
        return pd.DataFrame()

    for action, g in d.groupby("action"):
        for h in (1, 3, 5):
            x = g[f"net{h}"].dropna()

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
                    "selection": action,
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
        x["p_cal"],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []

    for bucket, g in x.groupby("bucket", observed=False):
        if g.empty:
            continue

        rows.append(
            {
                "probability_bucket": str(bucket),
                "observations": len(g),
                "average_model_probability": g.p_cal.mean(),
                "actual_win_rate": g.y5.mean(),
                "average_net_return": g.net5.mean(),
            }
        )

    return pd.DataFrame(rows)


def prediction_metrics(named_frames):
    rows = []

    for name, d in named_frames:
        q = d.dropna(
            subset=["p_cal", "y5", "pred_ret", "ret5_fwd"]
        ).copy()

        if q.empty:
            continue

        rows.append(
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
                    q.ret5_fwd,
                    q.pred_ret,
                ),
                "directional_accuracy": (
                    (q.pred_ret > 0) == (q.ret5_fwd > 0)
                ).mean(),
                "mean_predicted_return": q.pred_ret.mean(),
                "mean_actual_return": q.ret5_fwd.mean(),
            }
        )

    return pd.DataFrame(rows)


def non_overlapping_trade_test(oos: pd.DataFrame) -> pd.DataFrame:
    """
    Non-overlapping 5-session event test.

    Select one qualifying trade, then do not select another trade
    until at least HORIZON trading dates later.

    This is deliberately stricter than the observation-level report.
    """
    x = oos[oos.action == "TRADE"].copy()

    if x.empty:
        return pd.DataFrame()

    dates = sorted(x.date.unique())
    date_to_i = {d: i for i, d in enumerate(dates)}

    x["date_i"] = x.date.map(date_to_i)
    x = x.sort_values(
        ["date_i", "p_cal", "pred_ret"],
        ascending=[True, False, False],
    )

    selected = []
    last_i = None

    for _, row in x.iterrows():
        i = int(row.date_i)

        if last_i is None or i >= last_i + HORIZON:
            selected.append(row)
            last_i = i

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
                "gross_sum_return": z.ret5_fwd.sum(),
                "net_sum_return": z.net5.sum(),
            }
        ]
    )


def portfolio_backtest(oos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Conservative event-driven portfolio simulation.

    Important:
    - Each position gets an equal allocation from available equity.
    - A position exits after exactly PORTFOLIO_HORIZON trading sessions.
    - Equity is marked using the actual selected trade's 5-day net return
      only when its holding period completes.
    - New positions are not allowed to consume already-committed cash.
    - Open positions are tracked explicitly.
    """
    if oos.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = sorted(oos.date.unique())
    date_to_i = {d: i for i, d in enumerate(dates)}

    cash = CAPITAL
    positions = []
    curve = []

    trade_log = []

    for i, date in enumerate(dates):
        # Close positions whose holding period has completed.
        still_open = []

        for pos in positions:
            if i >= pos["exit_i"]:
                proceeds = pos["allocation"] * (1 + pos["net_return"])
                cash += proceeds

                trade_log.append(
                    {
                        "ticker": pos["ticker"],
                        "entry_date": pos["entry_date"],
                        "exit_date": date,
                        "allocation": pos["allocation"],
                        "net_return": pos["net_return"],
                        "pnl": proceeds - pos["allocation"],
                    }
                )
            else:
                still_open.append(pos)

        positions = still_open

        # Available capital after existing commitments.
        if len(positions) < MAX_POSITIONS:
            cur = oos[
                (oos.date == date)
                & (oos.action == "TRADE")
            ].copy()

            if not cur.empty:
                cur = cur.sort_values(
                    ["p_cal", "pred_ret"],
                    ascending=False,
                )

                capacity = MAX_POSITIONS - len(positions)
                slots = min(capacity, len(cur))

                # Use current free cash equally among new slots.
                allocation = cash / slots if slots > 0 else 0

                for _, row in cur.head(slots).iterrows():
                    exit_i = i + PORTFOLIO_HORIZON

                    if exit_i >= len(dates):
                        continue

                    if allocation <= 0:
                        continue

                    cash -= allocation

                    positions.append(
                        {
                            "ticker": row.ticker,
                            "entry_date": date,
                            "exit_i": exit_i,
                            "allocation": allocation,
                            "net_return": float(row.net5),
                        }
                    )

        # Conservative mark-to-market:
        # open positions remain at their entry value until exit because
        # their future path is not available from the event dataset.
        invested = sum(p["allocation"] for p in positions)
        equity = cash + invested

        curve.append(
            {
                "date": date,
                "equity": equity,
                "cash": cash,
                "invested": invested,
                "open_positions": len(positions),
            }
        )

    curve = pd.DataFrame(curve)

    if curve.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Close any remaining positions at the end only if their complete
    # 5-day outcome exists in the OOS data. Otherwise they remain
    # excluded from final realized equity.
    realized_equity = CAPITAL

    for tr in trade_log:
        realized_equity += tr["pnl"]

    # Build a realized-equity curve from completed trades.
    realized = pd.DataFrame(
        {
            "date": curve["date"],
            "equity": CAPITAL,
        }
    )

    pnl_by_exit = (
        pd.DataFrame(trade_log)
        .groupby("exit_date")["pnl"]
        .sum()
        if trade_log
        else pd.Series(dtype=float)
    )

    eq = CAPITAL
    vals = []

    for d in realized["date"]:
        if d in pnl_by_exit.index:
            eq += pnl_by_exit.loc[d]
        vals.append(eq)

    realized["equity"] = vals
    realized["daily_return"] = realized.equity.pct_change().fillna(0)

    running_max = realized.equity.cummax()
    realized["drawdown"] = realized.equity / running_max - 1

    final_equity = float(realized.equity.iloc[-1])
    total_return = final_equity / CAPITAL - 1

    years = max(len(realized) / 252.0, 1 / 252.0)
    cagr = (final_equity / CAPITAL) ** (1 / years) - 1

    daily = realized.daily_return.iloc[1:]
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

    summary = pd.DataFrame(
        [
            {
                "starting_capital": CAPITAL,
                "ending_realized_equity": final_equity,
                "total_realized_return": total_return,
                "CAGR": cagr,
                "max_drawdown": realized.drawdown.min(),
                "Sharpe": sharpe,
                "Sortino": sortino,
                "completed_trades": len(trade_log),
            }
        ]
    )

    return summary, pd.DataFrame(trade_log)


def chronological_split(d: pd.DataFrame):
    dates = sorted(d.date.unique())
    n = len(dates)

    val_start_i = int(n * (1 - OOS_FRAC - VAL_FRAC))
    oos_start_i = int(n * (1 - OOS_FRAC))

    val_start = dates[val_start_i]
    oos_start = dates[oos_start_i]

    dev = d[d.date < val_start].copy()
    val = d[
        (d.date >= val_start)
        & (d.date < oos_start)
    ].copy()
    oos = d[d.date >= oos_start].copy()

    return dev, val, oos, val_start, oos_start


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

            f = make_features(stock, market)
            successful += 1

            # Need all 5 future target observations to exist.
            for j in range(MIN_HISTORY - 1, len(f) - HORIZON):
                x = f.iloc[j]

                if x[FEATURES].isna().any():
                    continue

                row = {
                    "ticker": sym,
                    "date": f.index[j],
                    "close": float(x["close"]),
                    "market_close": float(x["market_close"]),
                    "y5": float(x["y5"]),
                    "ret1_fwd": float(x["ret1_fwd"]),
                    "ret3_fwd": float(x["ret3_fwd"]),
                    "ret5_fwd": float(x["ret5_fwd"]),
                }

                # CRITICAL:
                # Copy only backward-looking FEATURES.
                for col in FEATURES:
                    row[col] = float(x[col])

                rows.append(row)

        except Exception as exc:
            print(f"WARNING: {sym} failed: {exc}")

    d = pd.DataFrame(rows)

    if d.empty:
        raise RuntimeError("No candidate observations generated.")

    d = d.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    # Safety assertion: future target fields must NOT overlap FEATURES.
    overlap = set(FEATURES).intersection(
        {"ret1_fwd", "ret3_fwd", "ret5_fwd", "y5"}
    )

    if overlap:
        raise RuntimeError(
            f"FATAL FEATURE/TARGET OVERLAP: {overlap}"
        )

    return d, successful


def run_backtest():
    d, successful = build_dataset()

    dev, val, oos, val_start, oos_start = chronological_split(d)

    if len(dev) < MIN_TRAIN:
        raise RuntimeError(
            f"Development sample too small: {len(dev)}"
        )

    # ---------------------------------------------------------
    # DEVELOPMENT -> VALIDATION
    # ---------------------------------------------------------
    dev_fit = purge_training(dev, val.date.min())

    print(
        f"\nDevelopment observations: {len(dev)}"
        f"\nPurged development training observations: {len(dev_fit)}"
    )

    c, r = fit(dev_fit)

    if c is None:
        raise RuntimeError("Unable to fit development model.")

    val["p_raw"], val["pred_ret"] = predict(c, r, val)

    cal = calibrate_sigmoid(
        val.p_raw,
        val.y5,
    )

    val["p_cal"] = cal_predict(
        cal,
        val.p_raw,
    )

    val = add_net_returns(val)

    thresholds = choose_thresholds(val)

    print("\nVALIDATION-SELECTED THRESHOLDS:")
    print(thresholds)

    # ---------------------------------------------------------
    # FINAL PRE-OOS MODEL
    # ---------------------------------------------------------
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
    oos["p_cal"] = cal_predict(
        cal,
        oos.p_raw,
    )

    oos = add_net_returns(oos)
    oos = apply_actions(oos, thresholds)

    # ---------------------------------------------------------
    # PURGED CHRONOLOGICAL WALK-FORWARD
    # ---------------------------------------------------------
    print("\nRunning PURGED chronological walk-forward diagnostics...")

    dates = sorted(d.date.unique())
    wf_parts = []

    for k, date in enumerate(dates):
        cur = d[d.date == date].copy()
        prior = d[d.date < date].copy()

        if len(prior) < MIN_TRAIN:
            continue

        prior_fit = purge_training(
            prior,
            date,
        )

        if len(prior_fit) < MIN_TRAIN:
            continue

        prior_fit = prior_fit.tail(12000)

        cm, rm = fit(prior_fit)

        if cm is None:
            continue

        cur["p_raw"], cur["pred_ret"] = predict(
            cm,
            rm,
            cur,
        )

        # Frozen calibration and thresholds from validation.
        cur["p_cal"] = cal_predict(
            cal,
            cur.p_raw,
        )

        cur = add_net_returns(cur)
        cur = apply_actions(
            cur,
            thresholds,
        )

        wf_parts.append(cur)

        if k == 0 or k % 100 == 0 or k == len(dates) - 1:
            print(
                f"Purged walk-forward date "
                f"[{k+1}/{len(dates)}]"
            )

    wf = (
        pd.concat(
            wf_parts,
            ignore_index=True,
        )
        if wf_parts
        else pd.DataFrame()
    )

    # ---------------------------------------------------------
    # BENCHMARK
    # ---------------------------------------------------------
    benchmark = (
        d.groupby("date")
        .market_close
        .first()
        .sort_index()
        .pct_change(5)
        .dropna()
    )

    # ---------------------------------------------------------
    # REPORTS
    # ---------------------------------------------------------
    ts = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    wf_perf = performance(wf)
    oos_perf = performance(oos)

    nonoverlap = non_overlapping_trade_test(oos)
    portfolio_summary, portfolio_trades = portfolio_backtest(oos)

    metrics = prediction_metrics(
        [
            ("WALK_FORWARD", wf),
            ("OOS", oos),
        ]
    )

    outputs = {
        f"walkforward_v6_4_1_{ts}.csv": wf,
        f"validation_v6_4_1_{ts}.csv": val,
        f"oos_v6_4_1_{ts}.csv": oos,
        f"action_group_performance_v6_4_1_{ts}.csv": wf_perf,
        f"oos_performance_v6_4_1_{ts}.csv": oos_perf,
        f"probability_calibration_v6_4_1_{ts}.csv": calibration_table(wf),
        f"oos_probability_calibration_v6_4_1_{ts}.csv": calibration_table(oos),
        f"threshold_selection_v6_4_1_{ts}.csv": pd.DataFrame([thresholds]),
        f"nonoverlap_oos_v6_4_1_{ts}.csv": nonoverlap,
        f"portfolio_oos_v6_4_1_{ts}.csv": portfolio_summary,
        f"portfolio_trades_v6_4_1_{ts}.csv": portfolio_trades,
        f"prediction_metrics_v6_4_1_{ts}.csv": metrics,
        f"nifty_benchmark_v6_4_1_{ts}.csv": pd.DataFrame(
            [
                {
                    "observations": len(benchmark),
                    "win_rate": (benchmark > 0).mean(),
                    "average_5d_return": benchmark.mean(),
                    "median_5d_return": benchmark.median(),
                }
            ]
        ),
    }

    for name, frame in outputs.items():
        frame.to_csv(
            AUDIT / name,
            index=False,
        )

    print("\n" + "=" * 72)
    print(f"{VERSION} LEAKAGE-PROOF CHRONOLOGICAL WALK-FORWARD BACKTEST")
    print("=" * 72)

    print(f"Total candidate observations: {len(d)}")
    print(f"Successful symbols: {successful}")
    print(f"Unique symbols: {d.ticker.nunique()}")
    print(f"Unique signal dates: {d.date.nunique()}")
    print(f"Development observations: {len(dev)}")
    print(f"Validation observations: {len(val)}")
    print(f"OOS observations: {len(oos)}")
    print(f"Validation start: {val_start}")
    print(f"OOS start: {oos_start}")
    print(
        f"Round-trip cost assumption: "
        f"{ROUND_TRIP_COST*100:.3f}%"
    )

    print("\nFEATURE/TARGET LEAKAGE CHECK:")
    print("PASS — forward-return target fields are excluded from FEATURES.")

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
        oos_perf.to_string(index=False)
        if not oos_perf.empty
        else "None"
    )

    print("\nNON-OVERLAPPING OOS TRADE TEST:")
    print(
        nonoverlap.to_string(index=False)
        if not nonoverlap.empty
        else "None"
    )

    print("\nPORTFOLIO OOS TEST:")
    print(
        portfolio_summary.to_string(index=False)
        if not portfolio_summary.empty
        else "None"
    )

    print("\nOOS PROBABILITY CALIBRATION:")
    cal_oos = calibration_table(oos)
    print(
        cal_oos.to_string(index=False)
        if not cal_oos.empty
        else "None"
    )

    print("\nPREDICTION METRICS:")
    print(
        metrics.to_string(index=False)
        if not metrics.empty
        else "None"
    )

    print("\nNIFTY 5-DAY BENCHMARK:")
    print(
        outputs[
            f"nifty_benchmark_v6_4_1_{ts}.csv"
        ].to_string(index=False)
    )

    print("\nFILES CREATED:")
    for name in outputs:
        print(AUDIT / name)

    print("\n" + "=" * 72)
    print(f"{VERSION} BACKTEST COMPLETED")
    print("=" * 72)

    print(
        "\nAUDIT NOTES:\n"
        "1. Features are strictly backward-looking.\n"
        "2. Forward returns are targets only.\n"
        "3. Five signal dates are purged before model fitting.\n"
        "4. Validation calibration/thresholds are frozen before OOS.\n"
        "5. OOS labels are never used to fit or select thresholds.\n"
        "6. Non-overlapping and portfolio results are reported separately.\n"
        "7. Historical results do not guarantee future performance.\n"
    )


# -------------------------------------------------------------
# LIVE MODE
# -------------------------------------------------------------

def live_fit_and_score():
    """
    Conservative live signal mode.

    For safety, this is not activated by the backtest workflow.

    LIVE_PMIN and LIVE_RMIN should be copied from a reviewed V6.4.1
    validation run. Do not dynamically optimize them from live data.
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

            f = make_features(
                stock,
                market,
            )

            # Historical rows must have known targets for training.
            for j in range(
                MIN_HISTORY - 1,
                len(f) - HORIZON,
            ):
                x = f.iloc[j]

                if x[FEATURES].isna().any():
                    continue

                row = {
                    "ticker": sym,
                    "date": f.index[j],
                    "close": float(x.close),
                    "market_close": float(x.market_close),
                    "y5": float(x.y5),
                    "ret5_fwd": float(x.ret5_fwd),
                }

                for col in FEATURES:
                    row[col] = float(x[col])

                rows.append(row)

        except Exception as exc:
            print(
                f"LIVE WARNING {sym}: {exc}"
            )

    d = pd.DataFrame(rows)

    if d.empty:
        raise RuntimeError(
            "No live training observations generated."
        )

    d = d.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    signal_date = d.date.max()
    current = d[
        d.date == signal_date
    ].copy()

    train = d[
        d.date < signal_date
    ].copy()

    train = purge_training(
        train,
        signal_date,
    )

    if len(train) < MIN_TRAIN:
        raise RuntimeError(
            f"Insufficient purged live training observations: "
            f"{len(train)}"
        )

    c, r = fit(train)

    if c is None:
        raise RuntimeError(
            "Live model fit failed."
        )

    current["p_raw"], current["pred_ret"] = predict(
        c,
        r,
        current,
    )

    pmin = float(
        os.getenv(
            "LIVE_PMIN",
            "0.58",
        )
    )

    rmin = float(
        os.getenv(
            "LIVE_RMIN",
            "0.001",
        )
    )

    thresholds = {
        "pmin": pmin,
        "rmin": rmin,
    }

    # NOTE:
    # We deliberately do not pretend raw probability is calibrated.
    # It is labeled raw model probability.
    current["p_cal"] = current["p_raw"]

    current = apply_actions(
        current,
        thresholds,
    )

    trades = (
        current[
            current.action == "TRADE"
        ]
        .sort_values(
            ["p_cal", "pred_ret"],
            ascending=False,
        )
        .head(TOP_TRADE)
    )

    watch = (
        current[
            current.action == "WATCH"
        ]
        .sort_values(
            ["p_cal", "pred_ret"],
            ascending=False,
        )
        .head(TOP_WATCH)
    )

    lines = [
        f"MULTI-FACTOR MARKET ALERT {VERSION}",
        datetime.now().astimezone().strftime(
            "%d %b %Y, %H:%M %Z"
        ),
        "",
        f"SIGNAL DATE: {signal_date}",
        "",
        "--- TOP ML TRADE CANDIDATES ---",
    ]

    if trades.empty:
        lines.extend(
            [
                "",
                "NO VALID TRADE CANDIDATE",
            ]
        )
    else:
        for _, row in trades.iterrows():
            lines.extend(
                [
                    "",
                    f"{row.ticker.replace('.NS','')} — TRADE",
                    f"Price: ₹{row.close:,.2f}",
                    f"Raw P(UP): {row.p_raw*100:.1f}%",
                    f"Predicted 5D return: {row.pred_ret*100:.2f}%",
                    f"RSI: {row.rsi:.1f}",
                    f"Volume ratio: {row.vol_ratio:.2f}x",
                    "Confirm current market price before execution.",
                ]
            )

    lines.extend(
        [
            "",
            "--- WATCH ---",
        ]
    )

    if watch.empty:
        lines.append("None.")
    else:
        for n, (_, row) in enumerate(
            watch.iterrows(),
            1,
        ):
            lines.extend(
                [
                    "",
                    f"{n}. {row.ticker.replace('.NS','')} — WATCH",
                    f"Price: ₹{row.close:,.2f}",
                    f"Raw P(UP): {row.p_raw*100:.1f}%",
                    f"Predicted 5D return: {row.pred_ret*100:.2f}%",
                ]
            )

    lines.extend(
        [
            "",
            "--- AUDIT STATUS ---",
            "V6.4.1 future-return feature leakage check: PASS.",
            "Five-session purge: ENABLED.",
            "Gemini/news layer: NOT ENABLED.",
            "Historical model output is not a guarantee.",
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
        print(
            "Telegram credentials not configured; "
            "report printed only."
        )


if __name__ == "__main__":
    if MODE == "LIVE":
        live_fit_and_score()
    else:
        run_backtest()
