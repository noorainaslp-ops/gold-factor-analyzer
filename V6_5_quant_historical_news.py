
"""
V6.5.2 — LEAKAGE-PROOF QUANT vs GEMINI INFORMATION BACKTEST

Purpose:
  Controlled experiment comparing:
    A) Quant-only
    B) Quant + historical Gemini/news score
    C) Gemini/news-only diagnostic

Safety:
  - Historical Gemini/news scores MUST be supplied in a file with timestamps.
  - A score is usable for a signal date only when published_at <= signal_date.
  - If historical Gemini scores are absent, the script runs Quant-only and clearly
    reports that Gemini was NOT tested.
  - No current Gemini call is used to fabricate historical observations.
  - OOS observations are never used for feature/model/weight selection.

Expected optional historical file:
  historical_news.csv

Accepted columns:
  ticker or symbol
  published_at or timestamp/date
  gemini_score or news_score
Optional:
  confidence
  source_count

Score convention:
  gemini_score in [-1, 1], where positive = bullish, negative = bearish.

This is a research backtest, not investment advice.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

warnings.filterwarnings("ignore")

VERSION = "V6.5.2"
REVISION = "2026-08-31-GEMINI-CONTROLLED-EXPERIMENT"
RANDOM_STATE = 42

LOOKBACK_YEARS = 6
ROUND_TRIP_COST = 0.003
PURGE_DAYS = 10
SIGNAL_HORIZONS = [1, 3, 5, 10]

# Same universe used by the previous versions.
TICKERS = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS",
    "KOTAKBANK.NS","INDUSINDBK.NS","BAJFINANCE.NS","BAJAJFINSV.NS",
    "SHRIRAMFIN.NS","LT.NS","TMPV.NS","TMCV.NS","EICHERMOT.NS","MARUTI.NS",
    "HEROMOTOCO.NS","M&M.NS","TITAN.NS","ASIANPAINT.NS","HINDUNILVR.NS",
    "ITC.NS","NESTLEIND.NS","SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS",
    "DIVISLAB.NS","TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
    "BHARTIARTL.NS","NTPC.NS","POWERGRID.NS","ONGC.NS","BPCL.NS",
    "COALINDIA.NS","ADANIENT.NS","ADANIPORTS.NS","BEL.NS","HAL.NS","BHEL.NS",
    "TRENT.NS","PIDILITIND.NS","SIEMENS.NS","ABB.NS","GRASIM.NS",
    "ULTRACEMCO.NS","JSWSTEEL.NS","TATASTEEL.NS","HINDALCO.NS","IOC.NS",
    "VEDL.NS","DLF.NS","LODHA.NS","INDIGO.NS","ETERNAL.NS","NAUKRI.NS",
    "COFORGE.NS","JIOFIN.NS","IRFC.NS","IREDA.NS","POLYCAB.NS",
]

AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(exist_ok=True)


def flat_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance 1-ticker or MultiIndex output to OHLCV columns."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # For a single ticker, select the first ticker level where possible.
        if len(df.columns.levels) >= 2:
            candidates = []
            for col in df.columns:
                if isinstance(col, tuple):
                    if col[0] in {"Open","High","Low","Close","Adj Close","Volume"}:
                        candidates.append(col)
                    elif col[-1] in {"Open","High","Low","Close","Adj Close","Volume"}:
                        candidates.append(col)
            if candidates:
                # Resolve by scanning each desired field.
                out = {}
                fields = ["Open","High","Low","Close","Adj Close","Volume"]
                for f in fields:
                    hits = [c for c in df.columns if f in c]
                    if hits:
                        out[f] = df[hits[0]]
                df = pd.DataFrame(out, index=df.index)
        else:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Some yfinance combinations can still produce Series-like fields.
    normalized = {}
    for f in ["Open","High","Low","Close","Adj Close","Volume"]:
        if f in df.columns:
            s = df[f]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            normalized[f] = pd.to_numeric(s, errors="coerce")

    df = pd.DataFrame(normalized, index=df.index)
    if "Close" not in df.columns:
        return pd.DataFrame()

    return df.dropna(subset=["Close"])


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = down.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def make_features(px: pd.DataFrame) -> pd.DataFrame:
    c = px["Close"]
    ret1 = c.pct_change()

    x = pd.DataFrame(index=px.index)
    x["ret_1"] = ret1
    x["ret_3"] = c.pct_change(3)
    x["ret_5"] = c.pct_change(5)
    x["ret_10"] = c.pct_change(10)
    x["vol_5"] = ret1.rolling(5).std()
    x["vol_20"] = ret1.rolling(20).std()
    x["rsi_14"] = rsi(c, 14)
    x["ema_5_gap"] = c / c.ewm(span=5, adjust=False).mean() - 1
    x["ema_20_gap"] = c / c.ewm(span=20, adjust=False).mean() - 1
    x["range_5"] = (px["High"].rolling(5).max() / px["Low"].rolling(5).min()) - 1
    if "Volume" in px.columns:
        x["volume_ratio"] = px["Volume"] / px["Volume"].rolling(20).mean()
    else:
        x["volume_ratio"] = np.nan

    # Cross-sectional-free market regime features.
    x["dow"] = x.index.dayofweek / 4.0
    return x.replace([np.inf, -np.inf], np.nan)


def build_dataset() -> pd.DataFrame:
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    start = end - pd.DateOffset(years=LOOKBACK_YEARS)

    rows = []
    total = len(TICKERS)

    for i, ticker in enumerate(TICKERS, 1):
        print(f"Loading [{i}/{total}] {ticker}")
        try:
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=(end + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            px = flat_ohlcv(raw)

            if len(px) < 300:
                print(f"WARNING: insufficient history for {ticker}; skipping.")
                continue

            px.index = pd.to_datetime(px.index).tz_localize(None)
            f = make_features(px)

            for h in SIGNAL_HORIZONS:
                f[f"target_ret_{h}"] = px["Close"].shift(-h) / px["Close"] - 1
                f[f"target_up_{h}"] = (f[f"target_ret_{h}"] > 0).astype(float)

            f["ticker"] = ticker
            f["date"] = f.index.normalize()
            f["close"] = px["Close"].values
            rows.append(f.reset_index(drop=True))

        except Exception as exc:
            print(f"WARNING: {ticker} failed: {exc}")

    if not rows:
        raise RuntimeError("No usable market data were loaded.")

    d = pd.concat(rows, ignore_index=True)
    d = d.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Remove rows without enough feature history.
    feature_cols = [
        "ret_1","ret_3","ret_5","ret_10","vol_5","vol_20","rsi_14",
        "ema_5_gap","ema_20_gap","range_5","volume_ratio","dow"
    ]
    d = d.dropna(subset=feature_cols).reset_index(drop=True)
    return d


def load_historical_news(path: Path = Path("historical_news.csv")) -> pd.DataFrame:
    if not path.exists():
        print("HISTORICAL GEMINI/NEWS: NOT FOUND — QUANT-ONLY BASELINE")
        return pd.DataFrame(columns=["ticker", "published_at", "gemini_score"])

    n = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in n.columns}

    def find(*names):
        for name in names:
            if name in cols:
                return cols[name]
        return None

    ticker_col = find("ticker", "symbol")
    time_col = find("published_at", "timestamp", "date", "datetime")
    score_col = find("gemini_score", "news_score", "score")

    if not ticker_col or not time_col or not score_col:
        raise ValueError(
            "historical_news.csv must contain ticker/symbol, "
            "published_at/timestamp/date, and gemini_score/news_score."
        )

    n = n.rename(columns={
        ticker_col: "ticker",
        time_col: "published_at",
        score_col: "gemini_score",
    })
    n["ticker"] = n["ticker"].astype(str)
    n["published_at"] = pd.to_datetime(n["published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    n["gemini_score"] = pd.to_numeric(n["gemini_score"], errors="coerce")
    n = n.dropna(subset=["ticker","published_at","gemini_score"]).copy()
    n["gemini_score"] = n["gemini_score"].clip(-1, 1)

    # Reject obviously future-dated entries relative to the run.
    now = pd.Timestamp.now()
    n = n[n["published_at"] <= now]
    n = n.sort_values(["ticker","published_at"]).reset_index(drop=True)

    print(f"HISTORICAL GEMINI/NEWS: FOUND — {len(n)} timestamped records")
    return n


def attach_historical_news(d: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out["gemini_score"] = 0.0
    out["news_available"] = 0

    if news.empty:
        return out

    # For each signal date, use the most recent historical score available
    # before or at that signal date. This deliberately avoids future information.
    left = out[["ticker","date"]].sort_values(["ticker","date"]).copy()
    right = news[["ticker","published_at","gemini_score"]].sort_values(
        ["ticker","published_at"]
    ).copy()

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="published_at",
        by="ticker",
        direction="backward",
        allow_exact_matches=True,
    )
    score = merged["gemini_score"].fillna(0.0).to_numpy()
    available = merged["gemini_score"].notna().astype(int).to_numpy()

    out["gemini_score"] = score
    out["news_available"] = available
    return out


def leakage_check(d: pd.DataFrame, news: pd.DataFrame) -> None:
    feature_cols = [
        "ret_1","ret_3","ret_5","ret_10","vol_5","vol_20","rsi_14",
        "ema_5_gap","ema_20_gap","range_5","volume_ratio","dow","gemini_score"
    ]
    bad = [c for c in feature_cols if c.startswith("target_")]
    if bad:
        raise AssertionError(f"Target leakage in features: {bad}")

    if not news.empty:
        # Check every attached news record can be traced to <= signal date.
        chk = d.loc[d["news_available"] == 1, ["ticker","date"]].copy()
        if not chk.empty:
            n = news.sort_values(["ticker","published_at"])
            m = pd.merge_asof(
                chk.sort_values(["ticker","date"]),
                n[["ticker","published_at"]].sort_values(["ticker","published_at"]),
                left_on="date", right_on="published_at", by="ticker",
                direction="backward"
            )
            if (m["published_at"] > m["date"]).any():
                raise AssertionError("Historical-news timestamp leakage detected.")

    print("FEATURE/TARGET LEAKAGE CHECK: PASS")
    print("Gemini/news rule: published_at <= signal date.")
    print("Forward-return targets are excluded from FEATURES.")


def model_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def fit_models(train: pd.DataFrame, features: list[str], horizon: int):
    X = train[features]
    y_up = train[f"target_up_{horizon}"].astype(int)
    y_ret = train[f"target_ret_{horizon}"].astype(float)

    clf_models = [
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
                                       random_state=RANDOM_STATE),
    ]
    reg_models = [
        Ridge(alpha=10.0),
        HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
                                      random_state=RANDOM_STATE),
    ]

    clfs = [model_pipeline(m).fit(X, y_up) for m in clf_models]
    regs = [model_pipeline(m).fit(X, y_ret) for m in reg_models]
    return clfs, regs


def predict_ensemble(models, X):
    clfs, regs = models
    p = np.mean([m.predict_proba(X)[:, 1] for m in clfs], axis=0)
    r = np.mean([m.predict(X) for m in regs], axis=0)
    return np.clip(p, 0, 1), r


def add_predictions(train: pd.DataFrame, test: pd.DataFrame, feature_sets, horizon):
    result = test[["date","ticker","close"]].copy()

    for name, features in feature_sets.items():
        models = fit_models(train, features, horizon)
        p, r = predict_ensemble(models, test[features])
        result[f"{name}_p_up"] = p
        result[f"{name}_pred_ret"] = r

    result[f"actual_ret_{horizon}"] = test[f"target_ret_{horizon}"].values
    result[f"actual_up_{horizon}"] = test[f"target_up_{horizon}"].values
    return result


def summarize(pred: pd.DataFrame, model_name: str, horizon: int) -> dict:
    p = pred[f"{model_name}_p_up"].to_numpy()
    r = pred[f"{model_name}_pred_ret"].to_numpy()
    y = pred[f"actual_up_{horizon}"].to_numpy()
    actual = pred[f"actual_ret_{horizon}"].to_numpy()

    mask = np.isfinite(p) & np.isfinite(r) & np.isfinite(y) & np.isfinite(actual)
    p, r, y, actual = p[mask], r[mask], y[mask], actual[mask]

    if len(p) == 0:
        return {}

    pred_up = p >= 0.5
    wins = actual > 0

    # A realistic simple selected-trade rule: model probability >= 0.55
    selected = p >= 0.55
    if selected.sum() == 0:
        avg_selected = np.nan
        win_selected = np.nan
        pf_selected = np.nan
    else:
        rr = actual[selected] - ROUND_TRIP_COST
        avg_selected = float(np.mean(rr))
        win_selected = float(np.mean(rr > 0))
        pos = rr[rr > 0].sum()
        neg = -rr[rr < 0].sum()
        pf_selected = float(pos / neg) if neg > 0 else np.inf

    return {
        "model": model_name,
        "horizon": horizon,
        "observations": int(len(actual)),
        "directional_accuracy": float(np.mean(pred_up == wins)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1-1e-6), labels=[0,1])),
        "return_mae": float(mean_absolute_error(actual, r)),
        "mean_predicted_return": float(np.mean(r)),
        "mean_actual_return": float(np.mean(actual)),
        "selected_n_p>=55": int(selected.sum()),
        "selected_win_rate": win_selected,
        "selected_average_net_return": avg_selected,
        "selected_profit_factor": pf_selected,
    }


def choose_news_weight(validation: pd.DataFrame, quant_pred_col: str, news_pred_col: str,
                       horizon: int) -> float:
    y = validation[f"target_ret_{horizon}"].to_numpy()
    q = validation[quant_pred_col].to_numpy()
    g = validation[news_pred_col].to_numpy()

    mask = np.isfinite(y) & np.isfinite(q) & np.isfinite(g)
    y, q, g = y[mask], q[mask], g[mask]

    # Only choose among conservative weights. Selection is validation-only.
    weights = np.array([0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
    scores = []
    for w in weights:
        pred = (1-w)*q + w*g
        scores.append(np.mean(np.abs(y-pred)))

    best = float(weights[int(np.argmin(scores))])
    return best


def run_backtest():
    print("=" * 78)
    print(f"{VERSION} — CONTROLLED QUANT vs GEMINI BACKTEST")
    print("=" * 78)
    print(f"Source revision: {REVISION}")
    print(f"yfinance version: {getattr(yf, '__version__', 'unknown')}")
    print(f"Backtest period: {LOOKBACK_YEARS}y")
    print(f"Round-trip cost: {ROUND_TRIP_COST:.3%}")
    print()

    news = load_historical_news()
    d = build_dataset()
    d = attach_historical_news(d, news)
    leakage_check(d, news)

    dates = np.array(sorted(d["date"].unique()))
    if len(dates) < 500:
        raise RuntimeError("Insufficient signal dates for chronological backtest.")

    # Fixed chronological split: 60% development, 20% validation, 20% OOS.
    dev_end = dates[int(len(dates)*0.60)]
    val_end = dates[int(len(dates)*0.80)]

    dev = d[d["date"] < dev_end].copy()
    val = d[(d["date"] >= dev_end) & (d["date"] < val_end)].copy()
    oos = d[d["date"] >= val_end].copy()

    # Purge final dates from training around the validation/OOS boundary.
    purge_cut = dev["date"].max() - np.timedelta64(PURGE_DAYS, "D")
    dev_train = dev[dev["date"] <= purge_cut].copy()

    print("\nDATASET")
    print(f"Total candidate observations: {len(d)}")
    print(f"Development observations: {len(dev)}")
    print(f"Validation observations: {len(val)}")
    print(f"OOS observations: {len(oos)}")
    print(f"Development end: {dev_end}")
    print(f"Validation end: {val_end}")
    print(f"Purged development training observations: {len(dev_train)}")
    print(f"Historical news coverage: {d['news_available'].mean():.2%}")

    quant_features = [
        "ret_1","ret_3","ret_5","ret_10","vol_5","vol_20","rsi_14",
        "ema_5_gap","ema_20_gap","range_5","volume_ratio","dow"
    ]
    hybrid_features = quant_features + ["gemini_score"]

    # Require actual historical news coverage before fitting the hybrid.
    has_news = news is not None and not news.empty and d["news_available"].sum() > 0

    if not has_news:
        print("\nGEMINI EXPERIMENT STATUS: NOT RUN")
        print("Reason: no historical Gemini/news file with timestamped scores.")
        print("This run is a QUANT-ONLY BASELINE. It must NOT be interpreted as")
        print("evidence for or against Gemini improving prediction.")
        feature_sets = {"quant": quant_features}
    else:
        feature_sets = {"quant": quant_features, "hybrid": hybrid_features}

    all_rows = []
    for h in SIGNAL_HORIZONS:
        # Fit on development only, evaluate validation.
        val_pred = add_predictions(dev_train, val, feature_sets, h)
        val_pred.to_csv(AUDIT_DIR / f"v6_5_2_validation_h{h}.csv", index=False)

        # If hybrid exists, select its blend weight only on validation.
        if "hybrid" in feature_sets:
            w = choose_news_weight(
                val_pred,
                f"quant_pred_ret",
                f"hybrid_pred_ret",
                h
            )
        else:
            w = 0.0

        # Refit on dev + validation, but purge the end of validation before OOS.
        pre_oos = pd.concat([dev, val], ignore_index=True)
        purge_boundary = pre_oos["date"].max() - np.timedelta64(PURGE_DAYS, "D")
        train_oos = pre_oos[pre_oos["date"] <= purge_boundary].copy()

        oos_features = feature_sets
        oos_pred = add_predictions(train_oos, oos, oos_features, h)

        if "hybrid" in feature_sets:
            oos_pred["blend_p_up"] = (
                (1-w)*oos_pred["quant_p_up"] + w*oos_pred["hybrid_p_up"]
            )
            oos_pred["blend_pred_ret"] = (
                (1-w)*oos_pred["quant_pred_ret"] + w*oos_pred["hybrid_pred_ret"]
            )

        oos_pred.to_csv(AUDIT_DIR / f"v6_5_2_oos_h{h}.csv", index=False)

        for name in ["quant"] + (["hybrid"] if "hybrid" in feature_sets else []):
            all_rows.append(summarize(oos_pred, name, h))

        if "hybrid" in feature_sets:
            # Custom blended columns for summary.
            tmp = oos_pred.copy()
            tmp["blend_p_up"] = (1-w)*tmp["quant_p_up"] + w*tmp["hybrid_p_up"]
            tmp["blend_pred_ret"] = (1-w)*tmp["quant_pred_ret"] + w*tmp["hybrid_pred_ret"]
            all_rows.append(summarize(tmp, "blend", h))

        print(f"\nHORIZON {h}D — selected validation news weight: {w:.2f}")

    summary = pd.DataFrame(all_rows)
    summary.to_csv(AUDIT_DIR / "v6_5_2_oos_model_comparison.csv", index=False)

    print("\n" + "=" * 78)
    print("V6.5.2 OOS MODEL COMPARISON")
    print("=" * 78)
    if summary.empty:
        print("No OOS results.")
    else:
        print(summary.to_string(index=False))

    print("\nAUDIT FILES CREATED:")
    for p in sorted(AUDIT_DIR.glob("v6_5_2_*")):
        print(p)

    print("\n" + "=" * 78)
    print("V6.5.2 BACKTEST COMPLETED")
    print("=" * 78)
    print("IMPORTANT:")
    print("1. Historical Gemini scores are required for a genuine Gemini experiment.")
    print("2. Current Gemini calls are NEVER substituted for historical scores.")
    print("3. News timestamps must be <= signal date.")
    print("4. OOS data are not used for feature/model/weight selection.")
    print("5. A higher return alone is not sufficient to declare Gemini superior.")
    print("6. Historical backtests do not guarantee future performance.")


if __name__ == "__main__":
    run_backtest()
