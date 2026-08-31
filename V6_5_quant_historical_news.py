#!/usr/bin/env python3
"""
V6.5.1 — LEAKAGE-PROOF QUANT + OPTIONAL HISTORICAL NEWS/GEMINI HYBRID

Important:
- OHLCV is normalized for both pandas Series and yfinance MultiIndex output.
- All market features use information available on/before the signal date.
- Forward returns are targets only.
- Historical news is used only when published_at <= signal date and within the
  configured lookback window.
- Gemini scores are NOT generated retrospectively by this script. They must
  already exist in data/historical_news.csv and must have been produced using
  information available at that historical publication time.
- If historical_news.csv is absent/invalid/empty, the script runs a valid
  QUANT-ONLY baseline instead of fabricating historical news.
- Validation selects the news weight; OOS is untouched for selection.
- Purging removes the last PURGE_DAYS signal dates before every fit.
"""

from pathlib import Path
from datetime import datetime
import os, warnings
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

warnings.filterwarnings("ignore")

VERSION = "V6.5.1"
YF_VERSION = getattr(yf, "__version__", "unknown")

AUDIT = Path(os.getenv("AUDIT_DIR", "audit"))
AUDIT.mkdir(parents=True, exist_ok=True)
NEWS_FILE = Path(os.getenv("NEWS_FILE", "data/historical_news.csv"))
MODE = os.getenv("MODE", "BACKTEST").upper()
PERIOD = os.getenv("BACKTEST_PERIOD", "6y")

HORIZONS = [1, 3, 5, 10]
PRIMARY_HORIZON = 5
PURGE_DAYS = max(HORIZONS)

MIN_HISTORY = int(os.getenv("MIN_HISTORY", "220"))
MIN_TRAIN = int(os.getenv("MIN_TRAIN", "2000"))
VAL_FRAC = float(os.getenv("VALIDATION_FRACTION", "0.20"))
OOS_FRAC = float(os.getenv("OOS_FRACTION", "0.20"))

COST_BPS = float(os.getenv("COST_BPS", "10"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))
ROUND_TRIP_COST = 2.0 * (COST_BPS + SLIPPAGE_BPS) / 10000.0

MAX_TRAIN_OBS = int(os.getenv("MAX_TRAIN_OBS", "12000"))
NEWS_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "5"))
NEWS_DECAY_DAYS = float(os.getenv("NEWS_DECAY_DAYS", "2.0"))
NEWS_MIN_CONFIDENCE = float(os.getenv("NEWS_MIN_CONFIDENCE", "0.50"))

SYMBOLS = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
    "INDUSINDBK.NS","BAJFINANCE.NS","BAJAJFINSV.NS","SHRIRAMFIN.NS","LT.NS","TMPV.NS",
    "TMCV.NS","EICHERMOT.NS","MARUTI.NS","HEROMOTOCO.NS","M&M.NS","TITAN.NS",
    "ASIANPAINT.NS","HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","SUNPHARMA.NS","DRREDDY.NS",
    "CIPLA.NS","DIVISLAB.NS","TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
    "BHARTIARTL.NS","NTPC.NS","POWERGRID.NS","ONGC.NS","BPCL.NS","COALINDIA.NS",
    "ADANIENT.NS","ADANIPORTS.NS","BEL.NS","HAL.NS","BHEL.NS","TRENT.NS","PIDILITIND.NS",
    "SIEMENS.NS","ABB.NS","GRASIM.NS","ULTRACEMCO.NS","JSWSTEEL.NS","TATASTEEL.NS",
    "HINDALCO.NS","IOC.NS","VEDL.NS","DLF.NS","LODHA.NS","INDIGO.NS","ETERNAL.NS",
    "NAUKRI.NS","COFORGE.NS","JIOFIN.NS","IRFC.NS","IREDA.NS","POLYCAB.NS"
]

PRICE = [
    "ret1_past","ret3_past","ret5_past","ret10_past","ret20_past",
    "dist20","dist50","dist200","rsi","atr_pct","range_pct",
    "close_location","breakout20"
]
VOLUME = ["vol_ratio","vol20"]
REL = ["rel5","rel20"]
MARKET = [
    "mkt_ret1","mkt_ret5","mkt_ret20","mkt_dist20","mkt_dist50",
    "mkt_vol20","regime"
]
NEWS = [
    "news_score","news_confidence","news_count","news_positive_share",
    "news_negative_share","news_weighted_score","news_recency"
]
QUANT = PRICE + VOLUME + REL + MARKET
HYBRID = QUANT + NEWS
TARGETS = {"ret1_fwd","ret3_fwd","ret5_fwd","ret10_fwd","y1","y3","y5","y10"}


def clean(raw):
    """Normalize yfinance output to a simple OHLCV DataFrame."""
    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()

    wanted = ["Open", "High", "Low", "Close", "Volume"]
    x = raw.copy()

    if isinstance(x.columns, pd.MultiIndex):
        # Case 1: one level contains the OHLCV names.
        level_found = None
        for level in range(x.columns.nlevels):
            vals = {str(v) for v in x.columns.get_level_values(level)}
            if set(wanted).issubset(vals):
                level_found = level
                break

        if level_found is not None:
            x.columns = x.columns.get_level_values(level_found)
        else:
            # Case 2: OHLCV names are embedded somewhere in each tuple.
            new_cols = []
            for col in x.columns:
                hit = next((str(part) for part in col if str(part) in wanted), None)
                new_cols.append(hit if hit is not None else str(col))
            x.columns = new_cols

    # Keep first occurrence if duplicate labels were created by flattening.
    x = x.loc[:, ~pd.Index(x.columns).duplicated(keep="first")]

    missing = [c for c in wanted if c not in x.columns]
    if missing:
        return pd.DataFrame()

    x = x[wanted].copy()

    for c in wanted:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["Close"])
    x = x.sort_index()

    try:
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
    except Exception:
        pass

    x = x[~x.index.duplicated(keep="last")]
    return x


def download(symbol):
    raw = yf.download(
        symbol,
        period=PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
    )
    return clean(raw)


def calc_rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def calc_atr(x, n=14):
    prev = x["Close"].shift(1)
    tr = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - prev).abs(),
        (x["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def features(stock, market):
    s = stock.copy()
    m = market.reindex(s.index).ffill()

    close = s["Close"]
    volume = s["Volume"]
    mclose = m["Close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    msma20 = mclose.rolling(20).mean()
    msma50 = mclose.rolling(50).mean()

    ret1 = close.pct_change()
    ret3 = close.pct_change(3)
    ret5 = close.pct_change(5)
    ret10 = close.pct_change(10)
    ret20 = close.pct_change(20)

    mret1 = mclose.pct_change()
    mret5 = mclose.pct_change(5)
    mret20 = mclose.pct_change(20)

    prior_high20 = s["High"].rolling(20).max().shift(1)

    f = pd.DataFrame({
        "ret1_past": ret1,
        "ret3_past": ret3,
        "ret5_past": ret5,
        "ret10_past": ret10,
        "ret20_past": ret20,
        "dist20": close / sma20 - 1,
        "dist50": close / sma50 - 1,
        "dist200": close / sma200 - 1,
        "rsi": calc_rsi(close),
        "atr_pct": calc_atr(s) / close,
        "range_pct": (s["High"] - s["Low"]) / close,
        "close_location":
            (close - s["Low"]) /
            (s["High"] - s["Low"]).replace(0, np.nan),
        "breakout20": close / prior_high20 - 1,
        "vol_ratio": volume / volume.rolling(20).mean(),
        "vol20": ret1.rolling(20).std(),
        "rel5": ret5 - mret5,
        "rel20": ret20 - mret20,
        "mkt_ret1": mret1,
        "mkt_ret5": mret5,
        "mkt_ret20": mret20,
        "mkt_dist20": mclose / msma20 - 1,
        "mkt_dist50": mclose / msma50 - 1,
        "mkt_vol20": mret1.rolling(20).std(),
        "regime": np.select(
            [mclose > msma50 * 1.005, mclose < msma50 * 0.995],
            [1.0, -1.0],
            default=0.0,
        ),
    }, index=s.index)

    for h in HORIZONS:
        f[f"ret{h}_fwd"] = close.shift(-h) / close - 1
        f[f"y{h}"] = (f[f"ret{h}_fwd"] > 0).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)


def normalize_ticker(x):
    x = str(x).strip().upper()
    if not x.endswith(".NS"):
        x += ".NS"
    return x


def load_news():
    """
    Load only timestamped historical news with usable pre-existing scores.

    No Gemini API call is made here. This is deliberate: generating a score
    today for a historical article would create look-ahead bias unless the
    original model/version and information set are preserved.
    """
    if not NEWS_FILE.exists():
        print("HISTORICAL NEWS: NOT FOUND — QUANT-ONLY BACKTEST")
        return pd.DataFrame()

    try:
        n = pd.read_csv(NEWS_FILE)
    except Exception as exc:
        print(f"HISTORICAL NEWS: LOAD FAILED ({exc}) — QUANT-ONLY BACKTEST")
        return pd.DataFrame()

    required = {"ticker", "published_at", "title"}
    missing = required - set(n.columns)
    if missing:
        print(
            "HISTORICAL NEWS: INVALID — missing "
            + ", ".join(sorted(missing))
            + " — QUANT-ONLY BACKTEST"
        )
        return pd.DataFrame()

    n = n.copy()
    n["ticker"] = n["ticker"].map(normalize_ticker)
    n["published_at"] = pd.to_datetime(n["published_at"], errors="coerce", utc=True)
    n["published_at"] = n["published_at"].dt.tz_convert(None)

    if "gemini_score" not in n.columns:
        n["gemini_score"] = np.nan
    if "gemini_confidence" not in n.columns:
        n["gemini_confidence"] = np.nan

    n["gemini_score"] = pd.to_numeric(n["gemini_score"], errors="coerce").clip(-1, 1)
    n["gemini_confidence"] = pd.to_numeric(
        n["gemini_confidence"], errors="coerce"
    ).clip(0, 1)

    n["title"] = n["title"].fillna("").astype(str)
    if "summary" not in n.columns:
        n["summary"] = ""
    n["summary"] = n["summary"].fillna("").astype(str)

    # For a hybrid historical backtest, each usable news row needs a score.
    # Rows without a historical Gemini score are not silently scored today.
    n = n.dropna(subset=["published_at"])
    n = n[n["gemini_score"].notna() & n["gemini_confidence"].notna()].copy()
    n = n[n["gemini_confidence"] >= NEWS_MIN_CONFIDENCE].copy()

    if n.empty:
        print(
            "HISTORICAL NEWS: NO USABLE TIMESTAMPED GEMINI SCORES "
            "— QUANT-ONLY BACKTEST"
        )
        return pd.DataFrame()

    n = n.sort_values(["ticker", "published_at"]).reset_index(drop=True)
    print(
        f"HISTORICAL NEWS: LOADED {len(n)} SCORED ARTICLES "
        f"across {n.ticker.nunique()} tickers"
    )
    return n


def news_features_for_group(dates, ticker, news):
    """Create news features for one ticker using only already-published news."""
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(index=idx)

    for c in NEWS:
        out[c] = 0.0

    if news.empty:
        return out

    n = news[news["ticker"] == ticker].copy()
    if n.empty:
        return out

    pub = n["published_at"].to_numpy(dtype="datetime64[ns]")
    scores = n["gemini_score"].to_numpy(dtype=float)
    conf = n["gemini_confidence"].to_numpy(dtype=float)

    for dt in idx:
        lo = np.datetime64(dt - pd.Timedelta(days=NEWS_LOOKBACK_DAYS))
        hi = np.datetime64(dt)

        mask = (pub <= hi) & (pub >= lo)
        if not mask.any():
            continue

        age = (
            pd.Timestamp(dt).to_datetime64() - pub[mask]
        ).astype("timedelta64[s]").astype(float) / 86400.0
        decay = np.exp(-np.maximum(age, 0.0) / max(NEWS_DECAY_DAYS, 0.1))

        sc = scores[mask]
        cf = conf[mask]
        weighted = sc * cf * decay

        pos = sc > 0
        neg = sc < 0

        out.loc[dt, "news_score"] = float(np.mean(sc))
        out.loc[dt, "news_confidence"] = float(np.mean(cf))
        out.loc[dt, "news_count"] = float(mask.sum())
        out.loc[dt, "news_positive_share"] = float(np.mean(pos))
        out.loc[dt, "news_negative_share"] = float(np.mean(neg))
        denom = np.sum(cf * decay)
        out.loc[dt, "news_weighted_score"] = (
            float(np.sum(weighted) / denom) if denom > 0 else 0.0
        )
        out.loc[dt, "news_recency"] = float(np.max(decay))

    return out


def attach_news(data, news):
    data = data.copy()
    for c in NEWS:
        data[c] = 0.0

    if news.empty:
        return data

    pieces = []
    for ticker, g in data.groupby("ticker", sort=False):
        nf = news_features_for_group(g["date"], ticker, news)
        nf.index = g.index
        pieces.append(nf)

    if pieces:
        nf_all = pd.concat(pieces).sort_index()
        for c in NEWS:
            data[c] = nf_all[c].reindex(data.index).fillna(0.0)

    return data


def leakage_check():
    overlap = set(QUANT) & TARGETS
    overlap |= set(HYBRID) & TARGETS
    if overlap:
        raise RuntimeError(f"FATAL FEATURE/TARGET LEAKAGE: {sorted(overlap)}")

    print("FEATURE/TARGET LEAKAGE CHECK: PASS")
    print("Historical news rule: published_at <= signal date and within lookback.")
    print("Forward-return targets are excluded from FEATURES.")


def log_model():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", LogisticRegression(
            C=0.25, max_iter=3000, class_weight="balanced",
            random_state=RANDOM_STATE
        )),
    ])


def ridge():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("m", Ridge(alpha=10.0)),
    ])


def gb_c():
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.04, max_leaf_nodes=15,
        l2_regularization=2.0, random_state=RANDOM_STATE
    )


def gb_r():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.04, max_leaf_nodes=15,
        l2_regularization=2.0, random_state=RANDOM_STATE
    )


def fit(train, feature_cols):
    q = train.dropna(subset=["ret5_fwd", "y5"]).copy()
    if len(q) < MIN_TRAIN or q["y5"].nunique() < 2:
        return None

    a, b, c, e = log_model(), ridge(), gb_c(), gb_r()
    a.fit(q[feature_cols], q["y5"].astype(int))
    b.fit(q[feature_cols], q["ret5_fwd"])
    c.fit(q[feature_cols], q["y5"].astype(int))
    e.fit(q[feature_cols], q["ret5_fwd"])
    return a, b, c, e


def predict(models, d, feature_cols):
    a, b, c, e = models
    pl = a.predict_proba(d[feature_cols])[:, 1]
    pg = c.predict_proba(d[feature_cols])[:, 1]
    rp = (b.predict(d[feature_cols]) + e.predict(d[feature_cols])) / 2.0
    pp = np.clip((pl + pg) / 2.0, 0.01, 0.99)
    return pp, rp


def calibrator(p, y):
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    m.fit(z, np.asarray(y).astype(int))
    return m


def cal(m, p):
    p = np.clip(np.asarray(p), 1e-5, 1 - 1e-5)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    return np.clip(m.predict_proba(z)[:, 1], 0.01, 0.99)


def purge(d, cutoff):
    if d.empty:
        return d.copy()

    dates = sorted(pd.to_datetime(d.loc[d["date"] < cutoff, "date"].unique()))
    if len(dates) <= PURGE_DAYS:
        return d.iloc[0:0].copy()

    allowed_last = dates[-PURGE_DAYS - 1]
    return d[d["date"] <= allowed_last].copy()


def split(d):
    dates = sorted(pd.to_datetime(d["date"].unique()))
    if len(dates) < 100:
        raise RuntimeError("Too few unique dates for chronological split.")

    vi = int(len(dates) * (1 - OOS_FRAC - VAL_FRAC))
    oi = int(len(dates) * (1 - OOS_FRAC))

    vi = max(1, min(vi, len(dates) - 2))
    oi = max(vi + 1, min(oi, len(dates) - 1))

    return (
        d[d["date"] < dates[vi]].copy(),
        d[(d["date"] >= dates[vi]) & (d["date"] < dates[oi])].copy(),
        d[d["date"] >= dates[oi]].copy(),
        dates[vi],
        dates[oi],
    )


def score(d, w):
    d = d.copy()

    q = 0.5 * d["p_ensemble"] + 0.5 * (
        0.5 + np.tanh(d["r_ensemble"] * 20.0) / 2.0
    )
    n = 0.5 + 0.5 * (
        d["news_weighted_score"] * d["news_confidence"]
    )

    has_news = d["news_count"] > 0

    d["quant_score"] = q
    d["news_score_01"] = n
    d["hybrid_score"] = q
    d.loc[has_news, "hybrid_score"] = (
        (1 - w) * q.loc[has_news] + w * n.loc[has_news]
    )
    return d


def actions(d):
    d = d.copy()
    d["rank"] = d.groupby("date")["hybrid_score"].rank(pct=True)

    risk = (
        d["rsi"].between(35, 72)
        & (d["atr_pct"] < 0.06)
        & (d["vol_ratio"] > 0.50)
    )

    d["action"] = np.where(
        (d["rank"] >= 0.90)
        & (d["p_ensemble"] >= 0.53)
        & (d["r_ensemble"] >= 0.001)
        & risk,
        "TRADE",
        np.where(
            (d["rank"] >= 0.70)
            & (d["p_ensemble"] >= 0.50)
            & (d["r_ensemble"] >= 0),
            "WATCH",
            "WAIT",
        ),
    )

    for h in HORIZONS:
        d[f"net{h}"] = d[f"ret{h}_fwd"] - ROUND_TRIP_COST

    return d


def perf(d):
    rows = []
    for action, g in d.groupby("action", sort=True):
        for h in HORIZONS:
            x = g[f"net{h}"].dropna()
            if x.empty:
                continue
            win = x[x > 0]
            loss = x[x <= 0]
            pf = (
                win.sum() / abs(loss.sum())
                if len(loss) and loss.sum() < 0
                else np.nan
            )
            rows.append({
                "selection": action,
                "horizon": h,
                "observations": len(x),
                "win_rate": float((x > 0).mean()),
                "average_net_return": float(x.mean()),
                "median_net_return": float(x.median()),
                "average_winner": float(win.mean()) if len(win) else np.nan,
                "average_loser": float(loss.mean()) if len(loss) else np.nan,
                "profit_factor": float(pf) if pd.notna(pf) else np.nan,
                "best": float(x.max()),
                "worst": float(x.min()),
            })
    return pd.DataFrame(rows)


def topn(d, n):
    rows = []
    for date, g in d.groupby("date"):
        g = g.nlargest(n, "hybrid_score")
        if len(g) < n:
            continue
        r = {
            "date": date,
            "top_n": n,
            "tickers": ",".join(g["ticker"].astype(str)),
        }
        for h in HORIZONS:
            r[f"net{h}"] = float(g[f"ret{h}_fwd"].mean() - ROUND_TRIP_COST)
        rows.append(r)
    return pd.DataFrame(rows)


def topsummary(p):
    if p.empty:
        return pd.DataFrame()

    rows = []
    for h in HORIZONS:
        x = p[f"net{h}"].dropna()
        if x.empty:
            continue
        win = x[x > 0]
        loss = x[x <= 0]
        pf = (
            win.sum() / abs(loss.sum())
            if len(loss) and loss.sum() < 0
            else np.nan
        )
        rows.append({
            "horizon": h,
            "events": len(x),
            "win_rate": float((x > 0).mean()),
            "average_net_return": float(x.mean()),
            "median_net_return": float(x.median()),
            "profit_factor": float(pf) if pd.notna(pf) else np.nan,
            "best": float(x.max()),
            "worst": float(x.min()),
        })
    return pd.DataFrame(rows)


def nonoverlap(d, n, h):
    p = topn(d, n)
    if p.empty:
        return pd.DataFrame()

    dates = sorted(pd.to_datetime(p["date"].unique()))
    idx = {x: i for i, x in enumerate(dates)}

    take = []
    last = -999999
    for _, row in p.sort_values("date").iterrows():
        i = idx[pd.Timestamp(row["date"])]
        if i >= last + h:
            take.append(row)
            last = i

    x = pd.DataFrame(take)[f"net{h}"]
    if x.empty:
        return pd.DataFrame()

    return pd.DataFrame([{
        "top_n": n,
        "horizon": h,
        "trades": len(x),
        "win_rate": float((x > 0).mean()),
        "average_net_return": float(x.mean()),
        "median_net_return": float(x.median()),
        "best": float(x.max()),
        "worst": float(x.min()),
        "sum_net_return": float(x.sum()),
    }])


def prediction_metrics(d):
    rows = []

    y = d["y5"].astype(int).to_numpy()
    p = np.clip(d["p_ensemble"].to_numpy(), 0.001, 0.999)
    rpred = d["r_ensemble"].to_numpy()
    ract = d["ret5_fwd"].to_numpy()

    rows.append({
        "sample": "OOS",
        "observations": len(d),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "return_mae": float(mean_absolute_error(ract, rpred)),
        "directional_accuracy": float((np.sign(rpred) == np.sign(ract)).mean()),
        "mean_predicted_return": float(np.mean(rpred)),
        "mean_actual_return": float(np.mean(ract)),
    })
    return pd.DataFrame(rows)


def portfolio_test(d):
    """
    Conservative non-overlapping portfolio simulation.
    At each selected date, buy the top-ranked stock and hold for PRIMARY_HORIZON
    signal days. One position at a time. Costs are already included in net5.
    """
    candidates = d[d["action"] == "TRADE"].copy()
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    dates = sorted(pd.to_datetime(candidates["date"].unique()))
    last_exit = None
    trades = []

    for date in dates:
        if last_exit is not None and pd.Timestamp(date) < last_exit:
            continue

        g = candidates[candidates["date"] == date].nlargest(1, "hybrid_score")
        if g.empty:
            continue

        row = g.iloc[0]
        net = float(row["net5"])
        exit_date = pd.Timestamp(date) + pd.Timedelta(days=PRIMARY_HORIZON)

        trades.append({
            "entry_date": pd.Timestamp(date),
            "ticker": row["ticker"],
            "signal_price": row["close"],
            "p_ensemble": row["p_ensemble"],
            "r_ensemble": row["r_ensemble"],
            "hybrid_score": row["hybrid_score"],
            "realized_net_5d": net,
            "exit_approx_date": exit_date,
        })
        last_exit = exit_date

    td = pd.DataFrame(trades)
    if td.empty:
        return pd.DataFrame(), td

    equity = 100000.0
    peak = equity
    max_dd = 0.0
    curve = []

    for _, row in td.iterrows():
        equity *= max(0.0, 1.0 + float(row["realized_net_5d"]))
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        max_dd = min(max_dd, dd)
        curve.append(equity)

    total = equity / 100000.0 - 1.0
    years = max(
        (pd.Timestamp(td["entry_date"].iloc[-1]) -
         pd.Timestamp(td["entry_date"].iloc[0])).days / 365.25,
        1 / 365.25
    )
    cagr = (equity / 100000.0) ** (1 / years) - 1.0

    summary = pd.DataFrame([{
        "starting_capital": 100000.0,
        "ending_equity": equity,
        "total_return": total,
        "CAGR": cagr,
        "max_drawdown": max_dd,
        "completed_trades": len(td),
        "win_rate": float((td["realized_net_5d"] > 0).mean()),
        "average_net_5d": float(td["realized_net_5d"].mean()),
    }])

    return summary, td


def run_backtest():
    print("=" * 78)
    print("V6.5.1 LEAKAGE-PROOF QUANT + HISTORICAL NEWS BACKTEST")
    print("=" * 78)
    print("V6.5.1 source revision: 2026-08-31-CLEAN-REBUILD")
    print("yfinance version:", YF_VERSION)
    print("Backtest period:", PERIOD)
    print("Round-trip cost:", f"{ROUND_TRIP_COST*100:.3f}%")

    news = load_news()

    market = download("^NSEI")
    if market.empty:
        raise RuntimeError("Unable to download NIFTY (^NSEI).")

    rows = []
    successful = 0

    for i, sym in enumerate(SYMBOLS, 1):
        print(f"Loading [{i}/{len(SYMBOLS)}] {sym}")
        try:
            stock = download(sym)
            if stock.empty or len(stock) < MIN_HISTORY:
                print(f"WARNING: insufficient history for {sym}; skipping.")
                continue

            feat = features(stock, market)
            market_aligned = market.reindex(feat.index).ffill()

            end = len(feat) - max(HORIZONS)
            for j in range(MIN_HISTORY - 1, end):
                x = feat.iloc[j]

                if x[QUANT].isna().any():
                    continue

                record = {
                    "ticker": sym,
                    "date": pd.Timestamp(feat.index[j]),
                    "close": float(stock["Close"].iloc[j]),
                    "market_close": float(market_aligned["Close"].iloc[j]),
                }

                for c in QUANT:
                    record[c] = float(x[c])

                for h in HORIZONS:
                    record[f"ret{h}_fwd"] = float(x[f"ret{h}_fwd"])
                    record[f"y{h}"] = float(x[f"y{h}"])

                rows.append(record)

            successful += 1

        except Exception as exc:
            print(f"WARNING: {sym} failed: {exc}")

    if not rows:
        raise RuntimeError(
            "No valid observations were built. "
            "Check yfinance output and OHLCV normalization."
        )

    d = pd.DataFrame(rows)
    d = d.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

    d = attach_news(d, news)
    leakage_check()

    dev, val, oos, val_start, oos_start = split(d)

    print("\nDATASET")
    print("Total candidate observations:", len(d))
    print("Successful symbols:", successful)
    print("Unique symbols:", d["ticker"].nunique())
    print("Unique signal dates:", d["date"].nunique())
    print("Development observations:", len(dev))
    print("Validation observations:", len(val))
    print("OOS observations:", len(oos))
    print("Validation start:", val_start)
    print("OOS start:", oos_start)

    train_val = purge(dev, val_start).tail(MAX_TRAIN_OBS)
    print("Purged development training observations:", len(train_val))

    q_val_model = fit(train_val, QUANT)
    if q_val_model is None:
        raise RuntimeError("Quant validation fit failed.")

    p_q, r_q = predict(q_val_model, val, QUANT)
    val["p_quant_raw"] = p_q
    val["r_quant"] = r_q

    cq = calibrator(val["p_quant_raw"], val["y5"])
    val["p_quant"] = cal(cq, val["p_quant_raw"])

    use_news = not news.empty
    if use_news:
        h_val_model = fit(train_val, HYBRID)
        if h_val_model is None:
            print("Historical-news hybrid fit failed — reverting to quant-only.")
            use_news = False

    if use_news:
        p_h, r_h = predict(h_val_model, val, HYBRID)
        val["p_hybrid_raw"] = p_h
        val["r_hybrid"] = r_h
        ch = calibrator(val["p_hybrid_raw"], val["y5"])
        val["p_hybrid"] = cal(ch, val["p_hybrid_raw"])
    else:
        val["p_hybrid"] = val["p_quant"]
        val["r_hybrid"] = val["r_quant"]

    # Select news weight ONLY on validation.
    weight_grid = [0.0, 0.10, 0.20, 0.25, 0.35, 0.50, 0.65]
    weight_rows = []
    best_w = 0.0
    best_score = -np.inf

    for w in (weight_grid if use_news else [0.0]):
        z = val.copy()
        z["p_ensemble"] = z["p_hybrid"]
        z["r_ensemble"] = z["r_hybrid"]
        z = score(z, w)

        rank = z.groupby("date")["hybrid_score"].rank(pct=True)
        top = z.loc[rank >= 0.90, "ret5_fwd"].dropna()
        net = top - ROUND_TRIP_COST

        avg = float(net.mean()) if len(net) else np.nan
        weight_rows.append({
            "news_weight": w,
            "validation_top10pct_avg_net_5d": avg,
            "observations": len(net),
        })

        if len(net) >= 20 and pd.notna(avg) and avg > best_score:
            best_score = avg
            best_w = w

    # Refit once using only data strictly before OOS, with purge.
    pre_oos = d[d["date"] < oos_start].copy()
    train_oos = purge(pre_oos, oos_start).tail(MAX_TRAIN_OBS)

    print("\nPre-OOS observations:", len(pre_oos))
    print("Purged pre-OOS training observations:", len(train_oos))
    print("Selected validation news weight:", best_w)

    q_oos_model = fit(train_oos, QUANT)
    if q_oos_model is None:
        raise RuntimeError("Quant OOS fit failed.")

    pqo, rqo = predict(q_oos_model, oos, QUANT)
    oos["p_quant_raw"] = pqo
    oos["r_quant"] = rqo
    oos["p_quant"] = cal(cq, pqo)

    if use_news:
        h_oos_model = fit(train_oos, HYBRID)
        if h_oos_model is None:
            use_news = False

    if use_news:
        pho, rho = predict(h_oos_model, oos, HYBRID)
        oos["p_hybrid_raw"] = pho
        oos["r_hybrid"] = rho
        oos["p_hybrid"] = cal(ch, pho)
    else:
        oos["p_hybrid"] = oos["p_quant"]
        oos["r_hybrid"] = oos["r_quant"]

    oos["p_ensemble"] = oos["p_hybrid"]
    oos["r_ensemble"] = oos["r_hybrid"]

    hybrid = actions(score(oos.copy(), best_w))

    # Pure quant comparator on exactly the same OOS rows.
    quant_cmp = oos.copy()
    quant_cmp["p_ensemble"] = quant_cmp["p_quant"]
    quant_cmp["r_ensemble"] = quant_cmp["r_quant"]
    quant_cmp["hybrid_score"] = (
        0.5 * quant_cmp["p_quant"]
        + 0.5 * (0.5 + np.tanh(quant_cmp["r_quant"] * 20.0) / 2.0)
    )
    quant_cmp = actions(quant_cmp)

    hybrid_perf = perf(hybrid)
    quant_perf = perf(quant_cmp)

    tops = []
    non = []
    for n in [1, 3, 5]:
        p = topn(hybrid, n)
        s = topsummary(p)
        if not s.empty:
            s.insert(0, "top_n", n)
            tops.append(s)
        for h in [3, 5, 10]:
            x = nonoverlap(hybrid, n, h)
            if not x.empty:
                non.append(x)

    top_table = pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()
    non_table = pd.concat(non, ignore_index=True) if non else pd.DataFrame()

    portfolio_summary, portfolio_trades = portfolio_test(hybrid)

    # NIFTY benchmark.
    benchmark_close = d.groupby("date")["market_close"].first().sort_index()
    bench_rows = []
    for h in HORIZONS:
        rr = (benchmark_close.shift(-h) / benchmark_close - 1).dropna()
        bench_rows.append({
            "horizon": h,
            "observations": len(rr),
            "win_rate": float((rr > 0).mean()),
            "average_return": float(rr.mean()),
            "median_return": float(rr.median()),
        })
    benchmark = pd.DataFrame(bench_rows)

    metrics = prediction_metrics(oos)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = {
        f"dataset_v6_5_1_{ts}.csv": d,
        f"validation_v6_5_1_{ts}.csv": val,
        f"oos_v6_5_1_{ts}.csv": oos,
        f"oos_hybrid_action_performance_v6_5_1_{ts}.csv": hybrid_perf,
        f"oos_quant_action_performance_v6_5_1_{ts}.csv": quant_perf,
        f"topn_oos_v6_5_1_{ts}.csv": top_table,
        f"nonoverlap_oos_v6_5_1_{ts}.csv": non_table,
        f"validation_news_weight_v6_5_1_{ts}.csv": pd.DataFrame(weight_rows),
        f"portfolio_oos_v6_5_1_{ts}.csv": portfolio_summary,
        f"portfolio_trades_v6_5_1_{ts}.csv": portfolio_trades,
        f"prediction_metrics_v6_5_1_{ts}.csv": metrics,
        f"nifty_benchmark_v6_5_1_{ts}.csv": benchmark,
    }

    for fn, frame in outputs.items():
        frame.to_csv(AUDIT / fn, index=False)

    print("\n" + "=" * 78)
    print("V6.5.1 BACKTEST COMPLETED")
    print("=" * 78)
    print("NEWS MODE:", "HISTORICAL NEWS + GEMINI" if use_news else "QUANT-ONLY")
    print("Selected validation news weight:", best_w)

    print("\nOOS HYBRID ACTION COUNTS")
    print(hybrid["action"].value_counts().to_string())

    print("\nOOS HYBRID ACTION PERFORMANCE")
    print(hybrid_perf.to_string(index=False))

    print("\nOOS QUANT-ONLY ACTION PERFORMANCE")
    print(quant_perf.to_string(index=False))

    print("\nTOP-N HYBRID")
    print(top_table.to_string(index=False) if not top_table.empty else "None")

    print("\nNON-OVERLAPPING OOS")
    print(non_table.to_string(index=False) if not non_table.empty else "None")

    print("\nPORTFOLIO OOS")
    print(portfolio_summary.to_string(index=False) if not portfolio_summary.empty else "None")

    print("\nPREDICTION METRICS")
    print(metrics.to_string(index=False))

    print("\nNIFTY BENCHMARK")
    print(benchmark.to_string(index=False))

    print("\nFILES CREATED")
    for fn in outputs:
        print(AUDIT / fn)

    print("\nAUDIT NOTES")
    print("1. Features are backward-looking.")
    print("2. Forward returns are targets only.")
    print("3. Historical news is used only at/after publication time.")
    print("4. Historical Gemini scores must have been produced from the historical information set.")
    print("5. Purging is applied before validation/OOS fits.")
    print("6. News weight is selected on validation only.")
    print("7. OOS labels are never used for fitting or selection.")
    print("8. Historical performance does not guarantee future performance.")


def run_live():
    raise RuntimeError(
        "V6.5.1 LIVE is disabled. Run and review BACKTEST first."
    )


if __name__ == "__main__":
    if MODE == "LIVE":
        run_live()
    else:
        run_backtest()
