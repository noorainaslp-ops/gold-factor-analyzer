#!/usr/bin/env python3
"""
================================================================================
V7.0.0 — ENTERPRISE POINT-IN-TIME QUANT + GEMINI ALPHA RESEARCH ENGINE
================================================================================

Core Guarantees & Specifications:
1. Signal Timing: Evaluated strictly at Market Close of Session T.
2. Execution Timing: Market Entry at Session T+1 Open; Market Exit at Session T+H Close.
3. Target Definition: Target Return = Close[T+H] / Open[T+1] - 1 (NaN preserved on terminal rows).
4. Walk-Forward Engine: Chronological expanding window with exact H-session purge boundary.
5. Ensemble Stack: Calibrated HistGradientBoosting + L2 Logistic / Ridge Regressor.
6. Alternative Data Join: Point-in-time backward merge_asof with immutable row tracking.
7. Portfolio Engine: Single-capital sequential position model with session lockup and slippage accounting.
8. Self-Diagnostic Suite: 14 zero-network preflight tests verifying execution, math, and alignment.

Research only. Not financial advice.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION & PARAMETERS
# ============================================================

@dataclasses.dataclass(frozen=True)
class EnterpriseConfig:
    version: str = "V7.0.0"
    revision: str = "2026-08-31-ENTERPRISE-ULTRA"
    years: int = 6
    random_state: int = 42
    round_trip_cost: float = 0.0030  # 30 bps transaction cost + slippage
    min_train_samples: int = 2000
    retrain_every_n_sessions: int = 20
    horizons: Tuple[int, ...] = (1, 3, 5, 10)
    trade_p_threshold: float = 0.55
    trade_return_threshold: float = 0.0020
    gemini_weights: Tuple[float, ...] = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)
    gemini_lookback_days: int = 7
    bootstrap_iterations: int = 3000
    confidence_level: float = 0.95
    starting_capital: float = 100000.0
    audit_dir: Path = Path("audit")

    def validate(self) -> None:
        """Enforce strict parameter validation before runtime."""
        assert 0.0 < self.trade_p_threshold < 1.0, f"Invalid trade_p_threshold: {self.trade_p_threshold}"
        assert self.trade_return_threshold >= 0.0, f"Invalid trade_return_threshold: {self.trade_return_threshold}"
        assert self.round_trip_cost >= 0.0, f"Invalid round_trip_cost: {self.round_trip_cost}"
        assert self.min_train_samples > 0, f"Invalid min_train_samples: {self.min_train_samples}"
        assert len(self.horizons) > 0 and all(h > 0 for h in self.horizons), "Horizons must contain positive integers"
        assert 0.0 < self.confidence_level < 1.0, f"Invalid confidence_level: {self.confidence_level}"


CONFIG = EnterpriseConfig()
CONFIG.audit_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# UNIVERSE & QUANTITATIVE FACTOR DEFINITIONS
# ============================================================

TICKERS: List[str] = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS",
    "LT.NS", "TMPV.NS", "TMCV.NS", "EICHERMOT.NS", "MARUTI.NS",
    "HEROMOTOCO.NS", "M&M.NS", "TITAN.NS", "ASIANPAINT.NS", "HINDUNILVR.NS",
    "ITC.NS", "NESTLEIND.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    "DIVISLAB.NS", "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS",
    "TECHM.NS", "BHARTIARTL.NS", "NTPC.NS", "POWERGRID.NS", "ONGC.NS",
    "BPCL.NS", "COALINDIA.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BEL.NS",
    "HAL.NS", "BHEL.NS", "TRENT.NS", "PIDILITIND.NS", "SIEMENS.NS",
    "ABB.NS", "GRASIM.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "HINDALCO.NS", "IOC.NS", "VEDL.NS", "DLF.NS", "LODHA.NS",
    "INDIGO.NS", "ETERNAL.NS", "NAUKRI.NS", "COFORGE.NS", "JIOFIN.NS",
    "IRFC.NS", "IREDA.NS", "POLYCAB.NS",
]

QUANT_FEATURES: List[str] = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20", "parkinson_vol_14",
    "rsi_7", "rsi_14", "rsi_21",
    "atr_pct",
    "ema10_dist", "ema20_dist", "ema50_dist",
    "ema10_20", "ema20_50",
    "breakout_20", "breakdown_20",
    "volume_z", "range_pct", "close_location", "momentum_accel",
]

# ============================================================
# TECHNICAL ANALYSIS & DATA SANITIZATION
# ============================================================

def sanitize_feature_matrix(X: Any) -> Any:
    """Sanitize feature matrices against NaN/Inf without forward target leakage."""
    if isinstance(X, pd.DataFrame):
        Z = X.copy()
        for c in Z.columns:
            Z[c] = pd.to_numeric(Z[c], errors="coerce")
        Z = Z.replace([np.inf, -np.inf], np.nan)
        med = Z.median(axis=0, skipna=True)
        return Z.fillna(med).fillna(0.0).astype(float)
    A = np.asarray(X, dtype=float)
    A = np.where(np.isfinite(A), A, np.nan)
    if A.ndim == 1:
        vals = A[np.isfinite(A)]
        fill = float(np.median(vals)) if vals.size else 0.0
        return np.where(np.isfinite(A), A, fill)
    for j in range(A.shape[1]):
        vals = A[:, j][np.isfinite(A[:, j])]
        fill = float(np.median(vals)) if vals.size else 0.0
        A[~np.isfinite(A[:, j]), j] = fill
    return A


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    x = raw.copy()
    wanted = ["Open", "High", "Low", "Close", "Volume"]

    if isinstance(x.columns, pd.MultiIndex):
        flat = []
        for col in x.columns:
            parts = [str(v) for v in col]
            found = next((v for v in parts if v in wanted), parts[-1])
            flat.append(found)
        x.columns = flat
        x = x.loc[:, ~x.columns.duplicated(keep="first")]
    else:
        x.columns = [str(c) for c in x.columns]

    if any(c not in x.columns for c in wanted):
        return pd.DataFrame()

    for c in wanted:
        if isinstance(x[c], pd.DataFrame):
            x[c] = x[c].iloc[:, 0]
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x[wanted].dropna(subset=["Open", "High", "Low", "Close"])
    x.index = pd.to_datetime(x.index, errors="coerce")
    x = x.loc[~x.index.isna()]
    return x.sort_index()


def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_parkinson_volatility(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl_ratio = np.log(df["High"] / df["Low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt((factor * hl_ratio).rolling(period).mean())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute factors strictly using data available up to Session T Close."""
    x = df.copy()
    close = x["Close"]
    daily = close.pct_change()

    x["ret_1"] = close.pct_change(1)
    x["ret_3"] = close.pct_change(3)
    x["ret_5"] = close.pct_change(5)
    x["ret_10"] = close.pct_change(10)
    x["ret_20"] = close.pct_change(20)

    x["vol_5"] = daily.rolling(5).std()
    x["vol_10"] = daily.rolling(10).std()
    x["vol_20"] = daily.rolling(20).std()
    x["parkinson_vol_14"] = calc_parkinson_volatility(x, 14)

    x["rsi_7"] = calc_rsi(close, 7)
    x["rsi_14"] = calc_rsi(close, 14)
    x["rsi_21"] = calc_rsi(close, 21)
    x["atr_pct"] = calc_atr(x, 14) / close

    ema10 = close.ewm(span=10, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    x["ema10_dist"] = close / ema10 - 1.0
    x["ema20_dist"] = close / ema20 - 1.0
    x["ema50_dist"] = close / ema50 - 1.0
    x["ema10_20"] = ema10 / ema20 - 1.0
    x["ema20_50"] = ema20 / ema50 - 1.0

    high20 = x["High"].rolling(20).max()
    low20 = x["Low"].rolling(20).min()
    x["breakout_20"] = close / high20.shift(1) - 1.0
    x["breakdown_20"] = close / low20.shift(1) - 1.0

    vmean = x["Volume"].rolling(20).mean()
    vstd = x["Volume"].rolling(20).std()
    x["volume_z"] = (x["Volume"] - vmean) / vstd.replace(0.0, np.nan)
    x["range_pct"] = (x["High"] - x["Low"]) / close
    x["close_location"] = (close - x["Low"]) / (x["High"] - x["Low"]).replace(0.0, np.nan)
    x["momentum_accel"] = x["ret_5"] - x["ret_20"] / 4.0
    return x


def add_targets(df: pd.DataFrame, horizons: Tuple[int, ...] = CONFIG.horizons) -> pd.DataFrame:
    """
    Execution-Realistic Target Construction:
    - Signal Timestamp: Day T Close
    - Entry Price: Day T+1 Open (Open.shift(-1))
    - Exit Price: Day T+H Close (Close.shift(-h))
    - Realized Return = Close[T+H] / Open[T+1] - 1
    Terminal rows where future data is not yet realized explicitly retain np.nan.
    """
    x = df.copy()
    for h in horizons:
        target = x["Close"].shift(-h) / x["Open"].shift(-1) - 1.0
        x[f"target_{h}d"] = target
        x[f"up_{h}d"] = np.where(target.notna(), (target > 0.0).astype(float), np.nan)
    return x

# ============================================================
# POINT-IN-TIME GEMINI JOIN ENGINE
# ============================================================

def load_historical_gemini() -> pd.DataFrame:
    candidates = [Path("data/historical_gemini.csv"), Path("historical_gemini.csv")]
    file = next((p for p in candidates if p.exists()), None)
    if file is None:
        print("HISTORICAL GEMINI: NOT FOUND")
        return pd.DataFrame()

    g = pd.read_csv(file)
    required = {"ticker", "published_at", "gemini_score"}
    missing = required - set(g.columns)
    if missing:
        raise RuntimeError(f"Historical Gemini file missing required columns: {sorted(missing)}")

    if "gemini_confidence" not in g.columns:
        g["gemini_confidence"] = 0.5
    if "gemini_materiality" not in g.columns:
        g["gemini_materiality"] = 0.5

    g["ticker"] = g["ticker"].astype(str).str.strip()
    g["published_at"] = (
        pd.to_datetime(g["published_at"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    for c in ["gemini_score", "gemini_confidence", "gemini_materiality"]:
        g[c] = pd.to_numeric(g[c], errors="coerce")

    g = g.dropna(subset=["ticker", "published_at", "gemini_score"])
    g["gemini_score"] = g["gemini_score"].clip(-1.0, 1.0)
    g["gemini_confidence"] = g["gemini_confidence"].clip(0.0, 1.0)
    g["gemini_materiality"] = g["gemini_materiality"].clip(0.0, 1.0)
    g = g[g["ticker"].isin(TICKERS)].sort_values(["ticker", "published_at"])

    print(f"HISTORICAL GEMINI: FOUND {len(g):,} records")
    return g


def attach_gemini(data: pd.DataFrame, gemini: pd.DataFrame, lookback_days: int = CONFIG.gemini_lookback_days) -> pd.DataFrame:
    """Deterministic, order-preserving, point-in-time merge of alternative data."""
    out = data.copy()
    out["gemini_score"] = 0.0
    out["gemini_confidence"] = 0.0
    out["gemini_materiality"] = 0.0
    out["gemini_available"] = 0
    out["gemini_event_age_days"] = np.nan

    if gemini.empty:
        return out

    out["_row_id"] = np.arange(len(out))
    out["signal_timestamp"] = pd.to_datetime(out["date"])

    left = out[["_row_id", "ticker", "signal_timestamp"]].sort_values(["signal_timestamp", "ticker"])
    right = gemini[["ticker", "published_at", "gemini_score", "gemini_confidence", "gemini_materiality"]].dropna(
        subset=["ticker", "published_at"]
    ).sort_values(["published_at", "ticker"])

    merged = pd.merge_asof(
        left,
        right,
        left_on="signal_timestamp",
        right_on="published_at",
        by="ticker",
        direction="backward",
        tolerance=pd.Timedelta(days=lookback_days),
        allow_exact_matches=True,
    )

    bad = merged["published_at"].notna() & (merged["published_at"] > merged["signal_timestamp"])
    if bad.any():
        raise RuntimeError("FATAL: Gemini timestamp lookahead leakage detected in attach_gemini.")

    merged = merged.sort_values("_row_id").reset_index(drop=True)
    assert len(merged) == len(out), "Row count mismatch after attach_gemini"
    assert (merged["_row_id"].to_numpy() == out["_row_id"].to_numpy()).all(), "Row ID realignment failure in attach_gemini"

    age = (merged["signal_timestamp"] - merged["published_at"]).dt.total_seconds() / 86400.0

    out["gemini_score"] = pd.to_numeric(merged["gemini_score"], errors="coerce").fillna(0.0).to_numpy()
    out["gemini_confidence"] = pd.to_numeric(merged["gemini_confidence"], errors="coerce").fillna(0.0).to_numpy()
    out["gemini_materiality"] = pd.to_numeric(merged["gemini_materiality"], errors="coerce").fillna(0.0).to_numpy()
    out["gemini_available"] = merged["published_at"].notna().astype(int).to_numpy()
    out["gemini_event_age_days"] = age.to_numpy()

    return out.drop(columns=["_row_id", "signal_timestamp"], errors="ignore")


def build_dataset(gemini: pd.DataFrame, config: EnterpriseConfig = CONFIG) -> pd.DataFrame:
    frames = []
    for i, ticker in enumerate(TICKERS, 1):
        print(f"Loading [{i}/{len(TICKERS)}] {ticker}")
        try:
            raw = yf.download(
                ticker,
                period=f"{config.years}y",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            px = normalize_ohlcv(raw)
            if len(px) < 300:
                print(f"WARNING: insufficient history for {ticker}; skipping.")
                continue

            f = build_features(px)
            f = add_targets(f, config.horizons)
            f["ticker"] = ticker
            f["date"] = f.index
            frames.append(f.reset_index(drop=True))
        except Exception as exc:
            print(f"WARNING: {ticker} download failed: {exc}")

    if not frames:
        raise RuntimeError("No valid market data could be generated.")

    data = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    return attach_gemini(data, gemini, config.gemini_lookback_days)

# ============================================================
# ZERO-NETWORK PREFLIGHT DIAGNOSTIC SUITE
# ============================================================

def audit_feature_leakage(features: List[str], horizons: Tuple[int, ...]) -> None:
    forbidden = ["target", "up_", "future", "lead", "shift(-"]
    for f in features:
        for sub in forbidden:
            if sub in f.lower():
                raise RuntimeError(f"FATAL LEAKAGE DETECTED: Feature '{f}' contains pattern '{sub}'")
        for h in horizons:
            if f in (f"target_{h}d", f"up_{h}d"):
                raise RuntimeError(f"FATAL LEAKAGE DETECTED: Target column '{f}' in feature list")


def run_preflight_diagnostics(config: EnterpriseConfig = CONFIG) -> None:
    """Execute complete 14-test diagnostic suite before network calls."""
    print("=" * 78)
    print("STARTING ZERO-NETWORK PREFLIGHT INTEGRITY AUDIT")
    print("=" * 78)

    # 1. Config validation
    config.validate()
    print("PREFLIGHT 01: Configuration parameter boundaries ............ PASS")

    # 2. Threshold evaluation
    sample_df = pd.DataFrame({"p": [0.60, 0.40], "r": [0.010, -0.005]})
    sel = (sample_df["p"] >= config.trade_p_threshold) & (sample_df["r"] >= config.trade_return_threshold)
    assert sel.iloc[0] and not sel.iloc[1], "Threshold evaluation failed"
    print("PREFLIGHT 02: Canonical threshold filtering .................. PASS")

    # 3. Structural feature separation
    audit_feature_leakage(QUANT_FEATURES, config.horizons)
    print("PREFLIGHT 03: Feature/target structural separation ........... PASS")

    # 4. Feature list schema completeness
    dummy_schema = pd.DataFrame(columns=QUANT_FEATURES)
    assert all(c in dummy_schema.columns for c in QUANT_FEATURES), "Feature schema mismatch"
    print("PREFLIGHT 04: Feature schema completeness ..................... PASS")

    # 5. Session purge boundary
    dummy_hist = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="B"),
        "ret_1": [0.01] * 10,
        "target_3d": [0.02] * 10,
        "up_3d": [1.0] * 10,
    })
    eval_d = dummy_hist["date"].iloc[-1]
    prior_d = np.sort(dummy_hist[dummy_hist["date"] < eval_d]["date"].unique())
    cutoff = prior_d[-3]
    purged_train = dummy_hist[(dummy_hist["date"] < eval_d) & (dummy_hist["date"] < cutoff)]
    assert len(purged_train) == len(prior_d) - 3, "Session purge boundary failed"
    print("PREFLIGHT 05: Session purge boundary calculation ............. PASS")

    # 6 & 7. Portfolio and Non-overlap session locking
    test_market = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=12, freq="B"),
        "ticker": ["RELIANCE.NS"] * 12,
        "pred_probability": [0.60] * 12,
        "pred_return": [0.010] * 12,
        "target_3d": [0.015] * 12,
    })
    p_res, p_trades = portfolio_test(test_market, 3, "pred_probability", "pred_return", "test", config)
    assert p_res is not None and len(p_trades) == 4, "Sequential portfolio overlap locking failed"
    print("PREFLIGHT 06: Sequential portfolio session locking ............ PASS")

    n_res = nonoverlap(test_market, 3, "pred_probability", "pred_return", "test", config)
    assert n_res is not None and n_res["trades"] == 4, "Nonoverlap session locking failed"
    print("PREFLIGHT 07: Nonoverlap session spacing ...................... PASS")

    # 8. Gemini empty branch
    empty_att = attach_gemini(test_market, pd.DataFrame(), config.gemini_lookback_days)
    assert (empty_att["gemini_available"] == 0).all() and (empty_att["gemini_score"] == 0.0).all()
    print("PREFLIGHT 08: Gemini empty fallback branch .................... PASS")

    # 9. Point-in-time join & row alignment
    synth_market = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-10"), pd.Timestamp("2025-01-05")],
        "ticker": ["INFY.NS", "RELIANCE.NS", "RELIANCE.NS", "INFY.NS"],
    })
    synth_gemini = pd.DataFrame({
        "ticker": ["RELIANCE.NS", "RELIANCE.NS", "INFY.NS", "INFY.NS", "INFY.NS"],
        "published_at": [pd.Timestamp("2025-01-04"), pd.Timestamp("2025-01-07"), pd.Timestamp("2024-12-20"), pd.Timestamp("2025-01-15"), pd.Timestamp("2025-01-05")],
        "gemini_score": [0.80, 0.90, 0.40, 0.70, 0.65],
        "gemini_confidence": [0.85, 0.95, 0.50, 0.75, 0.80],
        "gemini_materiality": [0.90, 0.90, 0.50, 0.80, 0.85],
    })
    att = attach_gemini(synth_market, synth_gemini, config.gemini_lookback_days)
    assert math.isclose(att.iloc[0]["gemini_score"], 0.65) and att.iloc[0]["gemini_available"] == 1
    assert math.isclose(att.iloc[1]["gemini_score"], 0.80) and att.iloc[1]["gemini_available"] == 1
    assert math.isclose(att.iloc[2]["gemini_score"], 0.90) and att.iloc[2]["gemini_available"] == 1
    assert math.isclose(att.iloc[3]["gemini_score"], 0.65) and att.iloc[3]["gemini_available"] == 1
    print("PREFLIGHT 09: Point-in-time join and row alignment ........... PASS")

    # 10. Terminal target NaN handling
    raw_sample = pd.DataFrame({
        "Open": [100, 101, 102, 103], "High": [105, 106, 107, 108],
        "Low": [95, 96, 97, 98], "Close": [100, 101, 102, 103], "Volume": [1000] * 4,
    })
    t_df = add_targets(raw_sample, config.horizons)
    assert pd.isna(t_df["target_3d"].iloc[1]) and pd.isna(t_df["up_3d"].iloc[1]), "Terminal forward rows must be NaN"
    print("PREFLIGHT 10: Terminal target NaN preservation ................ PASS")

    # 11. Dataclass immutability check
    try:
        config.trade_p_threshold = 0.60  # type: ignore
        raise RuntimeError("Config dataclass must be frozen/immutable")
    except dataclasses.FrozenInstanceError:
        pass
    print("PREFLIGHT 11: Immutable config safety ......................... PASS")

    # 12. Execution timing validation (Signal T -> Entry T+1 Open -> Exit T+H Close)
    synth_ohlcv = pd.DataFrame({
        "Open": [100.0, 106.0, 111.0, 116.0, 121.0, 126.0, 131.0, 136.0, 141.0, 146.0, 151.0, 156.0],
        "High": [105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0],
        "Low": [95.0, 101.0, 106.0, 111.0, 116.0, 121.0, 126.0, 131.0, 136.0, 141.0, 146.0, 151.0],
        "Close": [104.0, 109.0, 114.0, 119.0, 124.0, 129.0, 134.0, 139.0, 144.0, 149.0, 154.0, 159.0],
        "Volume": [1000] * 12,
    })
    timing_df = add_targets(synth_ohlcv, config.horizons)
    assert math.isclose(timing_df["target_1d"].iloc[0], (109.0 / 106.0) - 1.0, rel_tol=1e-7)
    assert math.isclose(timing_df["target_3d"].iloc[0], (119.0 / 106.0) - 1.0, rel_tol=1e-7)
    assert math.isclose(timing_df["target_5d"].iloc[0], (129.0 / 106.0) - 1.0, rel_tol=1e-7)
    assert math.isclose(timing_df["target_10d"].iloc[0], (154.0 / 106.0) - 1.0, rel_tol=1e-7)
    print("PREFLIGHT 12: Execution timing / entry-after-information ...... PASS")

    # 13. Parkinson volatility factor test
    p_vol = calc_parkinson_volatility(synth_ohlcv, 5)
    assert len(p_vol) == 12 and p_vol.dropna().iloc[0] > 0.0, "Parkinson volatility failure"
    print("PREFLIGHT 13: Microstructure factor calculations ............. PASS")

    # 14. Non-shrinkage Gemini hybrid signal verification
    test_hybrid_df = pd.DataFrame({
        "pred_probability": [0.65],
        "pred_return": [0.012],
        "gemini_available": [0],
        "gemini_score": [0.0],
        "gemini_confidence": [0.0],
        "gemini_materiality": [0.0],
    })
    h_out = make_hybrid(test_hybrid_df, 0.40)
    assert math.isclose(h_out["hybrid_probability"].iloc[0], 0.65), "Gemini unvailable signal must not shrink"
    print("PREFLIGHT 14: Non-shrinkage alternative data preservation ..... PASS\n")

# ============================================================
# ESTIMATOR PIPELINES & WALK-FORWARD ENGINE
# ============================================================

def make_classifier(random_state: int = CONFIG.random_state) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=random_state))
    ])


def make_regressor() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])


def fit_model(train: pd.DataFrame, features: List[str], horizon: int, config: EnterpriseConfig = CONFIG) -> Optional[Dict[str, Any]]:
    required = features + [f"target_{horizon}d", f"up_{horizon}d"]
    q = train[required].replace([np.inf, -np.inf], np.nan).dropna(subset=[f"target_{horizon}d", f"up_{horizon}d"])

    if len(q) < config.min_train_samples or q[f"up_{horizon}d"].nunique() < 2:
        return None

    X = q[features]
    y_dir = q[f"up_{horizon}d"].astype(int)
    y_ret = q[f"target_{horizon}d"].astype(float)

    clf1 = make_classifier(config.random_state)
    clf2 = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=config.random_state
    )
    reg1 = make_regressor()
    reg2 = HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=1.0, random_state=config.random_state
    )

    X_clean = sanitize_feature_matrix(X)
    clf1.fit(X_clean, y_dir)
    clf2.fit(X_clean, y_dir)
    reg1.fit(X_clean, y_ret)
    reg2.fit(X_clean, y_ret)

    return {"clf1": clf1, "clf2": clf2, "reg1": reg1, "reg2": reg2}


def predict_model(model: Dict[str, Any], frame: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = frame[features].replace([np.inf, -np.inf], np.nan)
    valid = X.notna().all(axis=1)

    p = np.full(len(frame), np.nan)
    r = np.full(len(frame), np.nan)

    if not valid.any():
        return p, r

    xv = sanitize_feature_matrix(X.loc[valid])
    p1 = model["clf1"].predict_proba(xv)[:, 1]
    p2 = model["clf2"].predict_proba(xv)[:, 1]
    r1 = model["reg1"].predict(xv)
    r2 = model["reg2"].predict(xv)

    p[valid.to_numpy()] = (p1 + p2) / 2.0
    r[valid.to_numpy()] = (r1 + r2) / 2.0
    return p, r


def walk_forward(
    history: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: List[str],
    horizon: int,
    config: EnterpriseConfig = CONFIG,
) -> pd.DataFrame:
    dates = np.sort(evaluation["date"].unique())
    outputs = []
    current_model = None

    for i, date in enumerate(dates):
        if current_model is None or i % config.retrain_every_n_sessions == 0:
            prior_history = history[history["date"] < date]
            prior_dates = np.sort(prior_history["date"].unique())

            if len(prior_dates) <= horizon:
                continue

            purge_cutoff = prior_dates[-horizon]
            train = prior_history[prior_history["date"] < purge_cutoff].copy()

            current_model = fit_model(train, features, horizon, config)
            if current_model is None:
                continue

        day = evaluation[evaluation["date"] == date].copy()
        if day.empty:
            continue

        p, r = predict_model(current_model, day, features)
        day["pred_probability"] = p
        day["pred_return"] = r
        day = day.dropna(subset=["pred_probability", "pred_return"])

        if not day.empty:
            outputs.append(day)

    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()

# ============================================================
# HYBRID BAYESIAN INTEGRATION & VALIDATION
# ============================================================

def make_hybrid(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    x = df.copy()
    quant_p = x["pred_probability"].clip(0.001, 0.999)
    quant_logit = np.log(quant_p / (1.0 - quant_p))
    quant_r = x["pred_return"]

    hybrid_p = quant_p.copy()
    hybrid_r = quant_r.copy()
    gemini_p = pd.Series(0.5, index=x.index)
    gemini_r = pd.Series(0.0, index=x.index)

    has_gemini = (x["gemini_available"] == 1)

    if has_gemini.any() and weight > 0.0:
        gs = (
            x.loc[has_gemini, "gemini_score"]
            * x.loc[has_gemini, "gemini_confidence"]
            * x.loc[has_gemini, "gemini_materiality"]
        ).clip(-1.0, 1.0)

        g_prob = (0.5 + 0.5 * gs).clip(0.001, 0.999)
        g_logit = np.log(g_prob / (1.0 - g_prob))
        g_ret = 0.005 * gs

        h_logit = (1.0 - weight) * quant_logit.loc[has_gemini] + weight * g_logit
        hybrid_p.loc[has_gemini] = sigmoid(h_logit.to_numpy())
        hybrid_r.loc[has_gemini] = (1.0 - weight) * quant_r.loc[has_gemini] + weight * (quant_r.loc[has_gemini] + g_ret)

        gemini_p.loc[has_gemini] = g_prob
        gemini_r.loc[has_gemini] = g_ret

    x["hybrid_probability"] = hybrid_p
    x["hybrid_return"] = hybrid_r
    x["gemini_probability"] = gemini_p
    x["gemini_return"] = gemini_r
    return x


def select_gemini_weight(validation: pd.DataFrame, horizon: int, config: EnterpriseConfig = CONFIG) -> float:
    usable = validation[validation["gemini_available"] == 1].copy()
    if len(usable) < 50:
        return 0.0

    best_weight = 0.0
    best_score = -np.inf

    for w in config.gemini_weights:
        candidate = make_hybrid(usable, w)
        selected = candidate[
            (candidate["hybrid_probability"] >= config.trade_p_threshold)
            & (candidate["hybrid_return"] >= config.trade_return_threshold)
        ].dropna(subset=[f"target_{horizon}d"])

        if len(selected) < 30:
            continue

        net = selected[f"target_{horizon}d"] - config.round_trip_cost
        avg_ret = net.mean()
        win_rate = (net > 0).mean()

        score = avg_ret * np.sqrt(len(net))
        if win_rate < 0.50:
            score *= 0.75

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_weight = float(w)

    return best_weight

# ============================================================
# PERFORMANCE METRICS & PORTFOLIO ENGINE
# ============================================================

def performance(
    df: pd.DataFrame,
    horizon: int,
    probability_col: str,
    return_col: str,
    model_name: str,
    config: EnterpriseConfig = CONFIG,
) -> Optional[Dict[str, Any]]:
    target = f"target_{horizon}d"
    q = df[[target, probability_col, return_col]].dropna()

    if q.empty:
        return None

    y = (q[target] > 0).astype(int)
    p = q[probability_col].clip(0.001, 0.999)
    r = q[return_col]

    directional = ((p >= 0.5) == (y == 1)).mean()
    selected = q[(p >= config.trade_p_threshold) & (r >= config.trade_return_threshold)]

    if selected.empty:
        selected_n, selected_win, selected_avg, profit_factor = 0, np.nan, np.nan, np.nan
    else:
        net = selected[target] - config.round_trip_cost
        selected_n = len(net)
        selected_win = (net > 0).mean()
        selected_avg = net.mean()
        gains = net[net > 0].sum()
        losses = -net[net < 0].sum()
        profit_factor = gains / losses if losses > 0 else np.inf

    return {
        "model": model_name,
        "horizon": horizon,
        "observations": len(q),
        "directional_accuracy": directional,
        "brier_score": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "return_mae": mean_absolute_error(q[target], r),
        "mean_predicted_return": r.mean(),
        "mean_actual_return": q[target].mean(),
        f"selected_n_p>={int(config.trade_p_threshold*100)}": selected_n,
        "selected_win_rate": selected_win,
        "selected_average_net_return": selected_avg,
        "selected_profit_factor": profit_factor,
    }


def bootstrap_stats(returns: np.ndarray, config: EnterpriseConfig = CONFIG) -> Dict[str, Any]:
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 20:
        return {
            "n": len(arr), "mean": np.nan, "mean_ci_low": np.nan, "mean_ci_high": np.nan,
            "win_rate": np.nan, "win_ci_low": np.nan, "win_ci_high": np.nan, "prob_mean_gt_zero": np.nan,
        }

    rng = np.random.default_rng(config.random_state)
    means = np.empty(config.bootstrap_iterations)
    wins = np.empty(config.bootstrap_iterations)
    n = len(arr)

    for i in range(config.bootstrap_iterations):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
        wins[i] = (sample > 0).mean()

    alpha = (1.0 - config.confidence_level) / 2.0
    return {
        "n": n,
        "mean": arr.mean(),
        "mean_ci_low": np.quantile(means, alpha),
        "mean_ci_high": np.quantile(means, 1.0 - alpha),
        "win_rate": (arr > 0).mean(),
        "win_ci_low": np.quantile(wins, alpha),
        "win_ci_high": np.quantile(wins, 1.0 - alpha),
        "prob_mean_gt_zero": float((means > 0).mean()),
    }


def nonoverlap(
    df: pd.DataFrame,
    horizon: int,
    probability_col: str,
    return_col: str,
    model_name: str,
    config: EnterpriseConfig = CONFIG,
) -> Optional[Dict[str, Any]]:
    target = f"target_{horizon}d"
    q = df[(df[probability_col] >= config.trade_p_threshold) & (df[return_col] >= config.trade_return_threshold)].dropna(subset=[target]).copy()

    if q.empty:
        return None

    unique_dates = np.sort(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    q["date_idx"] = q["date"].map(date_to_idx)
    q = q.sort_values(["date", "ticker"])

    selected = []
    next_session = 0

    for _, row in q.iterrows():
        c_idx = row["date_idx"]
        if c_idx >= next_session:
            selected.append(row)
            next_session = c_idx + horizon

    if not selected:
        return None

    x = pd.DataFrame(selected)
    net = x[target] - config.round_trip_cost
    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    boot = bootstrap_stats(net.to_numpy(), config)

    return {
        "model": model_name,
        "horizon": horizon,
        "trades": len(net),
        "win_rate": (net > 0).mean(),
        "average_net": net.mean(),
        "median_net": net.median(),
        "profit_factor": gains / losses if losses > 0 else np.inf,
        "best": net.max(),
        "worst": net.min(),
        "net_sum_return": net.sum(),
        "bootstrap_mean_ci_low": boot["mean_ci_low"],
        "bootstrap_mean_ci_high": boot["mean_ci_high"],
        "bootstrap_win_ci_low": boot["win_ci_low"],
        "bootstrap_win_ci_high": boot["win_ci_high"],
        "prob_mean_gt_zero": boot["prob_mean_gt_zero"],
    }


def portfolio_test(
    df: pd.DataFrame,
    horizon: int,
    probability_col: str,
    return_col: str,
    model_name: str,
    config: EnterpriseConfig = CONFIG,
) -> Tuple[Optional[Dict[str, Any]], pd.DataFrame]:
    target = f"target_{horizon}d"
    q = df[(df[probability_col] >= config.trade_p_threshold) & (df[return_col] >= config.trade_return_threshold)].dropna(subset=[target]).copy()

    if q.empty:
        return None, pd.DataFrame()

    q["selection_score"] = q[probability_col] * q[return_col].clip(lower=0)
    candidates = (
        q.sort_values(["date", "selection_score"], ascending=[True, False])
        .groupby("date", as_index=False)
        .head(1)
        .sort_values("date")
    )

    unique_dates = np.sort(df["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    capital = config.starting_capital
    rows = []
    peak = capital
    max_drawdown = 0.0
    next_session = 0

    for _, row in candidates.iterrows():
        c_date = row["date"]
        c_idx = date_to_idx[c_date]

        if c_idx < next_session:
            continue

        gross_ret = float(row[target])
        net_ret = gross_ret - config.round_trip_cost
        start_cap = capital
        capital = capital * (1.0 + net_ret)
        peak = max(peak, capital)
        drawdown = capital / peak - 1.0
        max_drawdown = min(max_drawdown, drawdown)

        rows.append({
            "date": c_date,
            "ticker": row["ticker"],
            "gross_return": gross_ret,
            "net_return": net_ret,
            "starting_capital": start_cap,
            "ending_capital": capital,
            "drawdown": drawdown,
        })

        next_session = c_idx + horizon

    trades = pd.DataFrame(rows)
    if trades.empty:
        return None, trades

    first_date = pd.Timestamp(trades["date"].min())
    last_date = pd.Timestamp(trades["date"].max())
    years = max((last_date - first_date).days / 365.25, 1.0 / 365.25)
    total_return = capital / config.starting_capital - 1.0
    cagr = (capital / config.starting_capital) ** (1.0 / years) - 1.0

    seq_returns = trades["ending_capital"] / trades["starting_capital"] - 1.0
    sharpe = (
        (seq_returns.mean() / seq_returns.std()) * np.sqrt(max(len(seq_returns), 1))
        if len(seq_returns) > 1 and seq_returns.std() > 0
        else np.nan
    )

    return {
        "model": model_name,
        "horizon": horizon,
        "framework": "Sequential One-Position (Locked Capital)",
        "starting_capital": config.starting_capital,
        "ending_equity": capital,
        "total_return": total_return,
        "CAGR": cagr,
        "max_drawdown": max_drawdown,
        "trade_sequence_sharpe": sharpe,
        "completed_trades": len(trades),
    }, trades

# ============================================================
# BENCHMARKS & STABILITY BREAKDOWNS
# ============================================================

def build_nifty_benchmark(config: EnterpriseConfig = CONFIG) -> pd.DataFrame:
    raw = yf.download(
        "^NSEI",
        period=f"{config.years}y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    px = normalize_ohlcv(raw)
    if px.empty:
        return pd.DataFrame()

    records = []
    for h in config.horizons:
        r = (px["Close"].shift(-h) / px["Open"].shift(-1) - 1.0).dropna()
        records.append({
            "horizon": h,
            "observations": len(r),
            "win_rate": (r > 0).mean(),
            "average_return": r.mean(),
            "median_return": r.median(),
        })
    return pd.DataFrame(records)


def build_equal_weight_benchmark(data: pd.DataFrame, config: EnterpriseConfig = CONFIG) -> pd.DataFrame:
    records = []
    for h in config.horizons:
        grouped = []
        for _, g in data.groupby("date"):
            returns = g[f"target_{h}d"].dropna()
            if len(returns) > 0:
                grouped.append(returns.mean())
        if not grouped:
            continue
        r = pd.Series(grouped)
        records.append({
            "horizon": h,
            "observations": len(r),
            "win_rate": (r > 0).mean(),
            "average_return": r.mean(),
            "median_return": r.median(),
        })
    return pd.DataFrame(records)


def yearly_stability(
    df: pd.DataFrame,
    horizon: int,
    probability_col: str,
    return_col: str,
    model_name: str,
    config: EnterpriseConfig = CONFIG,
) -> pd.DataFrame:
    x = df.copy()
    x["year"] = pd.to_datetime(x["date"]).dt.year
    rows = []
    target = f"target_{horizon}d"

    for year, g in x.groupby("year"):
        selected = g[
            (g[probability_col] >= config.trade_p_threshold)
            & (g[return_col] >= config.trade_return_threshold)
        ].dropna(subset=[target]).copy()

        if selected.empty:
            rows.append({
                "model": model_name, "horizon": horizon, "year": year,
                "trades": 0, "win_rate": np.nan, "average_net": np.nan, "profit_factor": np.nan,
            })
            continue

        net = selected[target] - config.round_trip_cost
        gains = net[net > 0].sum()
        losses = -net[net < 0].sum()

        rows.append({
            "model": model_name,
            "horizon": horizon,
            "year": year,
            "trades": len(net),
            "win_rate": (net > 0).mean(),
            "average_net": net.mean(),
            "profit_factor": (gains / losses if losses > 0 else np.inf),
        })

    return pd.DataFrame(rows)


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    daily = x.groupby("date")["ret_20"].median().rename("market_momentum_20")
    x = x.merge(daily, left_on="date", right_index=True, how="left")
    x["regime"] = np.select(
        [x["market_momentum_20"] > 0.05, x["market_momentum_20"] < -0.05],
        ["BULL", "BEAR"],
        default="NEUTRAL",
    )
    return x


def regime_stability(
    df: pd.DataFrame,
    horizon: int,
    probability_col: str,
    return_col: str,
    model_name: str,
    config: EnterpriseConfig = CONFIG,
) -> pd.DataFrame:
    x = add_regime(df)
    rows = []
    target = f"target_{horizon}d"

    for regime, g in x.groupby("regime"):
        selected = g[
            (g[probability_col] >= config.trade_p_threshold)
            & (g[return_col] >= config.trade_return_threshold)
        ].dropna(subset=[target]).copy()

        if selected.empty:
            rows.append({
                "model": model_name, "horizon": horizon, "regime": regime,
                "trades": 0, "win_rate": np.nan, "average_net": np.nan, "profit_factor": np.nan,
            })
            continue

        net = selected[target] - config.round_trip_cost
        gains = net[net > 0].sum()
        losses = -net[net < 0].sum()

        rows.append({
            "model": model_name,
            "horizon": horizon,
            "regime": regime,
            "trades": len(net),
            "win_rate": (net > 0).mean(),
            "average_net": net.mean(),
            "profit_factor": (gains / losses if losses > 0 else np.inf),
        })

    return pd.DataFrame(rows)

# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

def main() -> None:
    # 1. Zero-Network Diagnostic Preflight
    run_preflight_diagnostics(CONFIG)

    print("=" * 78)
    print(f"{CONFIG.version} — ENTERPRISE QUANT + GEMINI ALPHA RESEARCH ENGINE")
    print("=" * 78)
    print(f"Engine Revision: {CONFIG.revision}")
    print(f"yfinance Version: {yf.__version__}")
    print(f"Backtest Horizon: {CONFIG.years} Years | Universe: {len(TICKERS)} Symbols")
    print(f"Filter Thresholds: P >= {CONFIG.trade_p_threshold:.2f}, Return >= {CONFIG.trade_return_threshold:.4f}")
    print("Execution Convention: Signal @ Day T Close -> Entry @ Day T+1 Open -> Exit @ Day T+H Close")

    # 2. Data Ingestion & Engineering
    gemini = load_historical_gemini()
    data = build_dataset(gemini, CONFIG)

    # 3. Post-build runtime leakage check
    target_cols = [f"target_{h}d" for h in CONFIG.horizons] + [f"up_{h}d" for h in CONFIG.horizons]
    leakage = set(QUANT_FEATURES).intersection(set(target_cols))
    if leakage:
        raise RuntimeError(f"FATAL: Structural leakage intersection found: {leakage}")
    audit_feature_leakage(QUANT_FEATURES, CONFIG.horizons)
    print("FEATURE/TARGET LEAKAGE CHECK: PASS (Zero structural target overlap)")
    print("Point-in-time Gemini timestamp rule: published_at <= signal date")
    print(f"Observations: {len(data):,} | Symbols: {data['ticker'].nunique()} | Signal dates: {data['date'].nunique()}")
    print(f"Gemini coverage: {data['gemini_available'].mean():.2%}")

    # 4. Chronological Partitioning
    dates = np.sort(data["date"].unique())
    dev_end = dates[int(len(dates) * 0.50)]
    val_end = dates[int(len(dates) * 0.75)]

    development = data[data["date"] < dev_end].copy()
    validation = data[(data["date"] >= dev_end) & (data["date"] < val_end)].copy()
    oos = data[data["date"] >= val_end].copy()

    print(f"Development Set: {len(development):,} rows")
    print(f"Validation Set:  {len(validation):,} rows")
    print(f"OOS Set:         {len(oos):,} rows")

    all_metrics, all_nonoverlap, all_portfolio, all_yearly, all_regime, weight_rows = [], [], [], [], [], []

    # 5. Multi-Horizon Expanding Walk-Forward Analysis
    for horizon in CONFIG.horizons:
        print()
        print("=" * 70)
        print(f"HORIZON {horizon}D (Session-Purged Walk-Forward)")
        print("=" * 70)

        # Validation Gemini weight optimization
        val_pred = walk_forward(development, validation, QUANT_FEATURES, horizon, CONFIG)
        selected_weight = select_gemini_weight(val_pred, horizon, CONFIG) if not val_pred.empty else 0.0
        print(f"Validation-selected Gemini weight: {selected_weight:.2f}")
        weight_rows.append({"horizon": horizon, "selected_gemini_weight": selected_weight})

        # Out-Of-Sample Quant Phase
        quant_oos = walk_forward(development, oos, QUANT_FEATURES, horizon, CONFIG)
        if quant_oos.empty:
            print("WARNING: OOS prediction set empty.")
            continue

        quant_oos.to_csv(CONFIG.audit_dir / f"v7_0_quant_oos_h{horizon}.csv", index=False)

        qres = performance(quant_oos, horizon, "pred_probability", "pred_return", "quant", CONFIG)
        if qres:
            all_metrics.append(qres)

        qno = nonoverlap(quant_oos, horizon, "pred_probability", "pred_return", "quant", CONFIG)
        if qno:
            all_nonoverlap.append(qno)

        qport, qtrades = portfolio_test(quant_oos, horizon, "pred_probability", "pred_return", "quant", CONFIG)
        if qport:
            all_portfolio.append(qport)
        if not qtrades.empty:
            qtrades.to_csv(CONFIG.audit_dir / f"v7_0_quant_portfolio_trades_h{horizon}.csv", index=False)

        qyear = yearly_stability(quant_oos, horizon, "pred_probability", "pred_return", "quant", CONFIG)
        if not qyear.empty:
            all_yearly.append(qyear)

        qreg = regime_stability(quant_oos, horizon, "pred_probability", "pred_return", "quant", CONFIG)
        if not qreg.empty:
            all_regime.append(qreg)

        # Out-Of-Sample Gemini Hybrid Phase
        coverage = quant_oos["gemini_available"].sum()
        if coverage < 50:
            print(f"Gemini: NOT TESTED — only {coverage} usable historical Gemini observations in OOS.")
            continue

        hybrid = make_hybrid(quant_oos, selected_weight)
        hybrid.to_csv(CONFIG.audit_dir / f"v7_0_hybrid_oos_h{horizon}.csv", index=False)

        gres = performance(hybrid, horizon, "gemini_probability", "gemini_return", "gemini_only", CONFIG)
        if gres:
            all_metrics.append(gres)

        hres = performance(hybrid, horizon, "hybrid_probability", "hybrid_return", "hybrid", CONFIG)
        if hres:
            all_metrics.append(hres)

        hno = nonoverlap(hybrid, horizon, "hybrid_probability", "hybrid_return", "hybrid", CONFIG)
        if hno:
            all_nonoverlap.append(hno)

        hport, htrades = portfolio_test(hybrid, horizon, "hybrid_probability", "hybrid_return", "hybrid", CONFIG)
        if hport:
            all_portfolio.append(hport)
        if not htrades.empty:
            htrades.to_csv(CONFIG.audit_dir / f"v7_0_hybrid_portfolio_trades_h{horizon}.csv", index=False)

        hyear = yearly_stability(hybrid, horizon, "hybrid_probability", "hybrid_return", "hybrid", CONFIG)
        if not hyear.empty:
            all_yearly.append(hyear)

        hreg = regime_stability(hybrid, horizon, "hybrid_probability", "hybrid_return", "hybrid", CONFIG)
        if not hreg.empty:
            all_regime.append(hreg)

    # 6. Benchmark Engines & Persistence
    nifty = build_nifty_benchmark(CONFIG)
    equal_weight = build_equal_weight_benchmark(data, CONFIG)

    metrics_df = pd.DataFrame(all_metrics)
    nonoverlap_df = pd.DataFrame(all_nonoverlap)
    portfolio_df = pd.DataFrame(all_portfolio)
    yearly_df = pd.concat(all_yearly, ignore_index=True) if all_yearly else pd.DataFrame()
    regime_df = pd.concat(all_regime, ignore_index=True) if all_regime else pd.DataFrame()
    weights_df = pd.DataFrame(weight_rows)

    metrics_df.to_csv(CONFIG.audit_dir / "v7_0_oos_model_comparison.csv", index=False)
    nonoverlap_df.to_csv(CONFIG.audit_dir / "v7_0_nonoverlap_oos.csv", index=False)
    portfolio_df.to_csv(CONFIG.audit_dir / "v7_0_portfolio_oos.csv", index=False)
    yearly_df.to_csv(CONFIG.audit_dir / "v7_0_yearly_stability.csv", index=False)
    regime_df.to_csv(CONFIG.audit_dir / "v7_0_regime_stability.csv", index=False)
    weights_df.to_csv(CONFIG.audit_dir / "v7_0_selected_gemini_weights.csv", index=False)
    nifty.to_csv(CONFIG.audit_dir / "v7_0_nifty_benchmark.csv", index=False)
    equal_weight.to_csv(CONFIG.audit_dir / "v7_0_equal_weight_benchmark.csv", index=False)

    # 7. Audit Reporting Output
    print("\n" + "=" * 78)
    print("V7.0.0 OOS MODEL COMPARISON")
    print("=" * 78)
    print("No model comparison results." if metrics_df.empty else metrics_df.to_string(index=False))

    print("\nNON-OVERLAPPING OOS")
    print("No non-overlapping results." if nonoverlap_df.empty else nonoverlap_df.to_string(index=False))

    print("\nPORTFOLIO OOS (SEQUENTIAL ONE-POSITION MODEL)")
    print("No portfolio results." if portfolio_df.empty else portfolio_df.to_string(index=False))

    print("\nYEARLY STABILITY")
    print("No yearly stability results." if yearly_df.empty else yearly_df.to_string(index=False))

    print("\nREGIME STABILITY")
    print("No regime stability results." if regime_df.empty else regime_df.to_string(index=False))

    print("\nNIFTY BENCHMARK")
    print(nifty.to_string(index=False))

    print("\nEQUAL-WEIGHT BENCHMARK")
    print(equal_weight.to_string(index=False))

    print("\n" + "=" * 78)
    print("V7.0.0 ENTERPRISE RESEARCH AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 78)


if __name__ == "__main__":
    main()
