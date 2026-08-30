#!/usr/bin/env python3
"""
V6.3.17 - Chronological walk-forward / untouched-OOS backtest.

Primary target: 5-trading-day forward return.
No OOS observations are used to choose thresholds.
Market data are aligned by trading date before arithmetic.

Research only; not financial advice.
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
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

warnings.filterwarnings("ignore")

VERSION = "V6.3.17"
AUDIT = Path("audit")
AUDIT.mkdir(exist_ok=True)

PERIOD = os.getenv("BACKTEST_PERIOD", "6y")
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "220"))
MIN_TRAIN = int(os.getenv("MIN_TRAIN", "2000"))
VAL_FRAC = float(os.getenv("VALIDATION_FRACTION", "0.20"))
OOS_FRAC = float(os.getenv("OOS_FRACTION", "0.20"))
COST_BPS = float(os.getenv("COST_BPS", "10"))
SLIP_BPS = float(os.getenv("SLIPPAGE_BPS", "5"))

PROB_GRID = [0.50,0.52,0.54,0.56,0.58,0.60,0.62,0.64]
RET_GRID = [0.0000,0.0005,0.0010,0.0015,0.0020,0.0030]

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
"ret1","ret3","ret5","ret10","ret20","dist20","dist50","dist200",
"rsi","atr_pct","vol_ratio","vol20","rel20","mkt_ret5","mkt_ret20",
"mkt_dist50","mkt_vol20","regime","range_pct","close_location","breakout20"
]

def clean(x):
    if x is None or x.empty: return pd.DataFrame()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = x.columns.get_level_values(0)
    need = ["Open","High","Low","Close","Volume"]
    if any(c not in x.columns for c in need): return pd.DataFrame()
    x = x[need].copy()
    for c in need: x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf,-np.inf],np.nan).dropna(subset=["Close"])
    try:
        if getattr(x.index,"tz",None) is not None: x.index=x.index.tz_localize(None)
    except Exception: pass
    x = x.sort_index()
    return x[~x.index.duplicated(keep="last")]

def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def atr(x,n=14):
    pc=x.Close.shift()
    tr=pd.concat([(x.High-x.Low),(x.High-pc).abs(),(x.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False,min_periods=n).mean()

def features(stock, market):
    # V6.3.16 fix: align market to the stock's exact dates.
    s=stock.copy()
    m=market.copy()
    try:
        if getattr(s.index,"tz",None) is not None: s.index=s.index.tz_localize(None)
        if getattr(m.index,"tz",None) is not None: m.index=m.index.tz_localize(None)
    except Exception: pass

    m=m.reindex(s.index).ffill()

    c=s.Close
    v=s.Volume
    mc=m.Close

    sma20=c.rolling(20).mean()
    sma50=c.rolling(50).mean()
    sma200=c.rolling(200).mean()
    ms50=mc.rolling(50).mean()

    ret1=c.pct_change()
    ret3=c.pct_change(3)
    ret5=c.pct_change(5)
    ret10=c.pct_change(10)
    ret20=c.pct_change(20)

    mr5=mc.pct_change(5)
    mr20=mc.pct_change(20)

    vol20=ret1.rolling(20).std()
    mvol20=mr5.rolling(20).std()

    high20=s.High.rolling(20).max()

    regime=np.select(
        [(mc>ms50*1.005),(mc<ms50*0.995)],
        [1.0,-1.0],
        default=0.0
    )

    f=pd.DataFrame({
        "ret1":ret1,
        "ret3":ret3,
        "ret5":ret5,
        "ret10":ret10,
        "ret20":ret20,
        "dist20":c/sma20-1,
        "dist50":c/sma50-1,
        "dist200":c/sma200-1,
        "rsi":rsi(c),
        "atr_pct":atr(s)/c,
        "vol_ratio":v/v.rolling(20).mean(),
        "vol20":vol20,
        "rel20":ret20-mr20,
        "mkt_ret5":mr5,
        "mkt_ret20":mr20,
        "mkt_dist50":mc/ms50-1,
        "mkt_vol20":mvol20,
        "regime":regime,
        "range_pct":(s.High-s.Low)/c,
        "close_location":(c-s.Low)/(s.High-s.Low).replace(0,np.nan),
        "breakout20":c/high20.shift()-1,
        "close":c,
        "market_close":mc
    },index=s.index)

    for h in (1,3,5):
        f[f"ret{h}_fwd"]=c.shift(-h)/c-1

    f["y5"]=(f.ret5_fwd>0).astype(float)
    return f.replace([np.inf,-np.inf],np.nan)

def clf_model():
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(
            C=.35,max_iter=2000,class_weight="balanced",random_state=17
        ))
    ])

def ret_model():
    return Pipeline([
        ("imp",SimpleImputer(strategy="median")),
        ("scale",StandardScaler()),
        ("reg",Ridge(alpha=8.0))
    ])

def fit(train):
    train=train.dropna(subset=["y5","ret5_fwd"])
    if len(train)<MIN_TRAIN or train.y5.nunique()<2:
        return None,None

    c=clf_model()
    r=ret_model()

    c.fit(train[FEATURES],train.y5.astype(int))
    r.fit(train[FEATURES],train.ret5_fwd)

    return c,r

def predict(c,r,d):
    p=c.predict_proba(d[FEATURES])[:,1]
    q=r.predict(d[FEATURES])
    return np.clip(p,.05,.95),q

def calibrate_sigmoid(p,y):
    """Platt/sigmoid calibration learned from validation predictions only."""
    p=np.clip(np.asarray(p),1e-5,1-1e-5)
    z=np.log(p/(1-p)).reshape(-1,1)

    cal=LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=17
    )
    cal.fit(z,np.asarray(y).astype(int))
    return cal

def cal_predict(cal,p):
    p=np.clip(np.asarray(p),1e-5,1-1e-5)
    z=np.log(p/(1-p)).reshape(-1,1)
    return np.clip(cal.predict_proba(z)[:,1],.05,.95)

def choose_thresholds(v):
    best=None

    for pm in PROB_GRID:
        for rm in RET_GRID:

            g=v[
                (v.p_cal>=pm)&
                (v.pred_ret>=rm)&
                (v.dist50>=-.025)&
                (v.rsi.between(38,70))&
                (v.vol_ratio>=.65)
            ]

            if len(g)<100:
                continue

            x=g.net5
            w=x[x>0]
            l=x[x<=0]

            pf=(
                w.sum()/abs(l.sum())
                if len(l) and l.sum()<0
                else 0
            )

            score=(
                100*x.mean()
                +.40*np.log1p(max(pf,0))
                +.10*((x>0).mean()-.5)
            )

            cand=(
                score,pm,rm,len(g),
                x.mean(),(x>0).mean(),pf
            )

            if best is None or cand[0]>best[0]:
                best=cand

    if best is None:
        return {
            "pmin":.58,
            "rmin":.0010,
            "n":0,
            "avg":np.nan,
            "win":np.nan,
            "pf":np.nan
        }

    return {
        "pmin":best[1],
        "rmin":best[2],
        "n":best[3],
        "avg":best[4],
        "win":best[5],
        "pf":best[6]
    }

def actions(d,t):
    out=[]

    for _,x in d.iterrows():

        if not np.isfinite(x.p_cal) or not np.isfinite(x.pred_ret):
            out.append("WAIT")
            continue

        risk_ok=(
            x.rsi>=38 and
            x.rsi<=70 and
            x.vol_ratio>=.65 and
            x.dist50>=-.025
        )

        if (
            x.p_cal>=t["pmin"] and
            x.pred_ret>=t["rmin"] and
            risk_ok
        ):
            out.append("TRADE")

        elif (
            x.p_cal>=max(.52,t["pmin"]-.04) and
            x.pred_ret>=0 and
            x.rsi>=35 and
            x.rsi<=72
        ):
            out.append("WATCH")

        else:
            out.append("WAIT")

    z=d.copy()
    z["action"]=out
    return z

def perf(d):
    rows=[]

    if d.empty:
        return pd.DataFrame()

    for a,g in d.groupby("action"):

        for h in (1,3,5):

            x=g[f"net{h}"].dropna()

            if x.empty:
                continue

            w=x[x>0]
            l=x[x<=0]

            pf=(
                w.sum()/abs(l.sum())
                if len(l) and l.sum()<0
                else np.nan
            )

            rows.append({
                "selection":a,
                "horizon":h,
                "observations":len(x),
                "win_rate":(x>0).mean(),
                "average_net_return":x.mean(),
                "median_net_return":x.median(),
                "average_winner":w.mean() if len(w) else np.nan,
                "average_loser":l.mean() if len(l) else np.nan,
                "profit_factor":pf,
                "best":x.max(),
                "worst":x.min()
            })

    return pd.DataFrame(rows)

def cal_table(d):
    x=d.dropna(subset=["p_cal","y5","net5"]).copy()

    if x.empty:
        return pd.DataFrame()

    bins=[-.01,.40,.45,.50,.55,.60,.65,.70,.75,1.01]
    labs=[
        "<40%","40-45%","45-50%","50-55%","55-60%",
        "60-65%","65-70%","70-75%","75%+"
    ]

    x["bucket"]=pd.cut(
        x.p_cal,
        bins=bins,
        labels=labs,
        right=False
    )

    rows=[]

    for b,g in x.groupby("bucket",observed=False):

        if g.empty:
            continue

        rows.append({
            "probability_bucket":str(b),
            "observations":len(g),
            "average_model_probability":g.p_cal.mean(),
            "actual_win_rate":g.y5.mean(),
            "average_net_return":g.net5.mean()
        })

    return pd.DataFrame(rows)

def main():

    print(
        f"Starting {VERSION} "
        "chronological walk-forward backtest..."
    )

    print(f"Backtest period: {PERIOD}")

    mr=yf.download(
        "^NSEI",
        period=PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    market=clean(mr)

    if market.empty:
        raise RuntimeError("NIFTY data unavailable.")

    rows=[]
    ok=0

    for i,sym in enumerate(SYMBOLS,1):

        print(
            f"Loading [{i}/{len(SYMBOLS)}] {sym}"
        )

        try:

            raw=yf.download(
                sym,
                period=PERIOD,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False
            )

            stock=clean(raw)

            if stock.empty or len(stock)<MIN_HISTORY:
                print(
                    f"WARNING: insufficient history "
                    f"for {sym}; skipping."
                )
                continue

            f=features(stock,market)
            ok+=1

        except Exception as e:

            print(
                f"WARNING: {sym} failed: {e}"
            )
            continue

        for j in range(
            MIN_HISTORY-1,
            len(f)-5
        ):

            x=f.iloc[j]

            if x[FEATURES].isna().any():
                continue

            r={
                "ticker":sym,
                "date":f.index[j]
            }

            for col in FEATURES:
                r[col]=float(x[col])

            r["close"]=float(x.close)
            r["market_close"]=float(
                x.market_close
            )

            for h in (1,3,5):
                r[f"ret{h}"]=float(
                    x[f"ret{h}_fwd"]
                )

            r["y5"]=float(x.y5)
            rows.append(r)

    d=pd.DataFrame(rows)

    if d.empty:
        raise RuntimeError(
            "No candidate observations generated."
        )

    d=d.sort_values(
        ["date","ticker"]
    ).reset_index(drop=True)

    dates=sorted(
        d.date.unique()
    )

    n=len(dates)

    val_start=dates[
        int(
            n*(1-OOS_FRAC-VAL_FRAC)
        )
    ]

    oos_start=dates[
        int(
            n*(1-OOS_FRAC)
        )
    ]

    dev=d[
        d.date<val_start
    ].copy()

    val=d[
        (d.date>=val_start)&
        (d.date<oos_start)
    ].copy()

    oos=d[
        d.date>=oos_start
    ].copy()

    if len(dev)<MIN_TRAIN:
        raise RuntimeError(
            f"Development sample too small: {len(dev)}"
        )

    # ========================================================
    # DEVELOPMENT -> VALIDATION
    # ========================================================

    c,r=fit(dev)

    if c is None:
        raise RuntimeError(
            "Unable to fit development model."
        )

    val["p_raw"],val["pred_ret"]=predict(
        c,r,val
    )

    # Calibration is learned ONLY from validation
    # predictions, never from OOS observations.
    cal=calibrate_sigmoid(
        val.p_raw,
        val.y5
    )

    val["p_cal"]=cal_predict(
        cal,
        val.p_raw
    )

    cost=2*(COST_BPS+SLIP_BPS)/10000

    for h in (1,3,5):
        val[f"net{h}"]=(
            val[f"ret{h}"]-cost
        )

    thresholds=choose_thresholds(
        val
    )

    print(
        "\nVALIDATION-SELECTED THRESHOLDS:"
    )

    print(thresholds)

    # ========================================================
    # FINAL PRE-OOS MODEL
    # ========================================================

    pre=d[
        d.date<oos_start
    ].copy()

    cf,rf=fit(pre)

    if cf is None:
        raise RuntimeError(
            "Unable to fit final pre-OOS model."
        )

    oos["p_raw"],oos["pred_ret"]=predict(
        cf,rf,oos
    )

    # Calibration object is frozen from validation.
    oos["p_cal"]=cal_predict(
        cal,
        oos.p_raw
    )

    for h in (1,3,5):
        oos[f"net{h}"]=(
            oos[f"ret{h}"]-cost
        )

    # OOS actions use frozen thresholds.
    oos=actions(
        oos,
        thresholds
    )

    # ========================================================
    # CHRONOLOGICAL WALK-FORWARD DIAGNOSTIC
    # ========================================================

    print(
        "\nRunning chronological "
        "walk-forward diagnostics..."
    )

    wf_parts=[]

    for k,date in enumerate(
        dates
    ):

        cur=d[
            d.date==date
        ].copy()

        prior=d[
            d.date<date
        ].copy()

        if len(prior)<MIN_TRAIN:
            continue

        cm,rm=fit(
            prior.tail(12000)
        )

        if cm is None:
            continue

        cur["p_raw"],cur["pred_ret"]=predict(
            cm,
            rm,
            cur
        )

        # Frozen calibration and thresholds.
        cur["p_cal"]=cal_predict(
            cal,
            cur.p_raw
        )

        for h in (1,3,5):
            cur[f"net{h}"]=(
                cur[f"ret{h}"]-cost
            )

        cur=actions(
            cur,
            thresholds
        )

        wf_parts.append(
            cur
        )

        if (
            k==0 or
            k%100==0 or
            k==len(dates)-1
        ):

            print(
                f"Walk-forward date "
                f"[{k+1}/{len(dates)}]"
            )

    wf=(
        pd.concat(
            wf_parts,
            ignore_index=True
        )
        if wf_parts
        else pd.DataFrame()
    )

    # ========================================================
    # SAVE AUDIT FILES
    # ========================================================

    ts=datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    outputs={
        f"walkforward_v6_3_17_{ts}.csv":wf,
        f"validation_v6_3_17_{ts}.csv":val,
        f"oos_v6_3_17_{ts}.csv":oos,
        f"action_group_performance_v6_3_17_{ts}.csv":perf(wf),
        f"oos_performance_v6_3_17_{ts}.csv":perf(oos),
        f"probability_calibration_v6_3_17_{ts}.csv":cal_table(wf),
        f"oos_probability_calibration_v6_3_17_{ts}.csv":cal_table(oos),
        f"threshold_selection_v6_3_17_{ts}.csv":pd.DataFrame([thresholds])
    }

    metrics=[]

    for name,x in [
        ("WALK_FORWARD",wf),
        ("OOS",oos)
    ]:

        q=x.dropna(
            subset=[
                "p_cal",
                "y5",
                "pred_ret",
                "ret5"
            ]
        )

        if q.empty:
            continue

        metrics.append({
            "sample":name,
            "observations":len(q),
            "brier_score":brier_score_loss(
                q.y5.astype(int),
                q.p_cal
            ),
            "log_loss":log_loss(
                q.y5.astype(int),
                np.clip(
                    q.p_cal,
                    1e-6,
                    1-1e-6
                ),
                labels=[0,1]
            ),
            "return_mae":mean_absolute_error(
                q.ret5,
                q.pred_ret
            ),
            "directional_accuracy":(
                (q.pred_ret>0)
                ==
                (q.ret5>0)
            ).mean(),
            "mean_predicted_return":q.pred_ret.mean(),
            "mean_actual_return":q.ret5.mean()
        })

    outputs[
        f"prediction_metrics_v6_3_17_{ts}.csv"
    ]=pd.DataFrame(metrics)

    # ========================================================
    # NIFTY BENCHMARK
    # ========================================================

    benchmark=(
        d.groupby("date")
        .market_close
        .first()
        .sort_index()
        .pct_change(5)
        .dropna()
    )

    outputs[
        f"nifty_benchmark_v6_3_17_{ts}.csv"
    ]=pd.DataFrame([{
        "observations":len(benchmark),
        "win_rate":(benchmark>0).mean(),
        "average_5d_return":benchmark.mean(),
        "median_5d_return":benchmark.median()
    }])

    for name,x in outputs.items():
        x.to_csv(
            AUDIT/name,
            index=False
        )

    # ========================================================
    # REPORT
    # ========================================================

    print("\n"+"="*70)
    print(
        f"{VERSION} CHRONOLOGICAL "
        "WALK-FORWARD BACKTEST"
    )
    print("="*70)

    print(
        f"Total candidate observations: {len(d)}"
    )

    print(
        f"Successful symbols: {ok}"
    )

    print(
        f"Unique symbols: {d.ticker.nunique()}"
    )

    print(
        f"Unique signal dates: {d.date.nunique()}"
    )

    print(
        f"Development observations: {len(dev)}"
    )

    print(
        f"Validation observations: {len(val)}"
    )

    print(
        f"OOS observations: {len(oos)}"
    )

    print(
        f"Validation start: {val_start}"
    )

    print(
        f"OOS start: {oos_start}"
    )

    print(
        "\nFROZEN VALIDATION THRESHOLDS:"
    )

    print(thresholds)

    print(
        "\nWALK-FORWARD ACTION COUNTS:"
    )

    if not wf.empty:
        print(
            wf.action
            .value_counts()
            .rename_axis("action")
            .to_frame("count")
            .to_string()
        )
    else:
        print("None")

    print(
        "\nOOS ACTION COUNTS:"
    )

    print(
        oos.action
        .value_counts()
        .rename_axis("action")
        .to_frame("count")
        .to_string()
    )

    print(
        "\nWALK-FORWARD PERFORMANCE:"
    )

    print(
        perf(wf).to_string(index=False)
        if not wf.empty
        else "None"
    )

    print(
        "\nUNTOUCHED OOS PERFORMANCE:"
    )

    print(
        perf(oos).to_string(index=False)
        if not oos.empty
        else "None"
    )

    print(
        "\nPROBABILITY CALIBRATION:"
    )

    print(
        cal_table(wf).to_string(index=False)
        if not wf.empty
        else "None"
    )

    print(
        "\nOOS PROBABILITY CALIBRATION:"
    )

    print(
        cal_table(oos).to_string(index=False)
        if not oos.empty
        else "None"
    )

    print(
        "\nPREDICTION METRICS:"
    )

    print(
        pd.DataFrame(metrics).to_string(index=False)
        if metrics
        else "None"
    )

    print(
        "\nNIFTY 5-DAY BENCHMARK:"
    )

    benchmark_file=[
        k for k in outputs
        if k.startswith("nifty_benchmark")
    ][0]

    print(
        outputs[
            benchmark_file
        ].to_string(index=False)
    )

    print(
        "\nFILES CREATED:"
    )

    for name in outputs:
        print(
            AUDIT/name
        )

    print(
        "\n"
        + "="*70
    )

    print(
        f"{VERSION} BACKTEST COMPLETED"
    )

    print(
        "="*70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "OOS observations were not used "
        "to choose thresholds."
    )

    print(
        "P(UP) is an empirical model estimate, "
        "not a guaranteed probability."
    )

    print(
        "Do NOT deploy to real-money trading "
        "solely from this backtest."
    )

if __name__=="__main__":
    main()
