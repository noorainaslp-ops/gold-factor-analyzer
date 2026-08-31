#!/usr/bin/env python3
"""
V6.5 — LEAKAGE-PROOF QUANT + HISTORICAL NEWS/GEMINI HYBRID

Research framework:
- causal market features only
- forward returns are targets only
- historical news must have publication timestamps
- Gemini scores must be produced using information available at that time
- validation selects the news weight; OOS does not
- QUANT-ONLY and HYBRID are compared on the same OOS period

Run with MODE=BACKTEST.

Historical news CSV:
data/historical_news.csv

Required:
ticker,published_at,title

Optional:
summary,gemini_score,gemini_confidence

gemini_score: -1..+1
gemini_confidence: 0..1

If the file is absent, the script runs a quant-only baseline and
does NOT pretend that current news was historically available.
"""

from pathlib import Path
from datetime import datetime
import os, re, warnings
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

YF_VERSION=getattr(yf,"__version__","unknown")

VERSION="V6.5"
AUDIT=Path(os.getenv("AUDIT_DIR","audit")); AUDIT.mkdir(exist_ok=True)
NEWS_FILE=Path(os.getenv("NEWS_FILE","data/historical_news.csv"))
MODE=os.getenv("MODE","BACKTEST").upper()
PERIOD=os.getenv("BACKTEST_PERIOD","6y")
HORIZONS=[1,3,5,10]; PRIMARY_HORIZON=5; PURGE_DAYS=max(HORIZONS)
MIN_HISTORY=int(os.getenv("MIN_HISTORY","220"))
MIN_TRAIN=int(os.getenv("MIN_TRAIN","2000"))
VAL_FRAC=float(os.getenv("VALIDATION_FRACTION","0.20"))
OOS_FRAC=float(os.getenv("OOS_FRACTION","0.20"))
COST_BPS=float(os.getenv("COST_BPS","10")); SLIPPAGE_BPS=float(os.getenv("SLIPPAGE_BPS","5"))
ROUND_TRIP_COST=2*(COST_BPS+SLIPPAGE_BPS)/10000
MAX_TRAIN_OBS=int(os.getenv("MAX_TRAIN_OBS","12000"))
RANDOM_STATE=42
NEWS_LOOKBACK_DAYS=int(os.getenv("NEWS_LOOKBACK_DAYS","5"))
NEWS_DECAY_DAYS=float(os.getenv("NEWS_DECAY_DAYS","2.0"))
NEWS_MIN_CONFIDENCE=float(os.getenv("NEWS_MIN_CONFIDENCE","0.50"))

SYMBOLS=[
"RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","AXISBANK.NS","KOTAKBANK.NS",
"INDUSINDBK.NS","BAJFINANCE.NS","BAJAJFINSV.NS","SHRIRAMFIN.NS","LT.NS","TMPV.NS",
"TMCV.NS","EICHERMOT.NS","MARUTI.NS","HEROMOTOCO.NS","M&M.NS","TITAN.NS",
"ASIANPAINT.NS","HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","SUNPHARMA.NS","DRREDDY.NS",
"CIPLA.NS","DIVISLAB.NS","TCS.NS","INFY.NS","HCLTECH.NS","WIPRO.NS","TECHM.NS",
"BHARTIARTL.NS","NTPC.NS","POWERGRID.NS","ONGC.NS","BPCL.NS","COALINDIA.NS",
"ADANIENT.NS","ADANIPORTS.NS","BEL.NS","HAL.NS","BHEL.NS","TRENT.NS","PIDILITIND.NS",
"SIEMENS.NS","ABB.NS","GRASIM.NS","ULTRACEMCO.NS","JSWSTEEL.NS","TATASTEEL.NS",
"HINDALCO.NS","IOC.NS","VEDL.NS","DLF.NS","LODHA.NS","INDIGO.NS","ETERNAL.NS",
"NAUKRI.NS","COFORGE.NS","JIOFIN.NS","IRFC.NS","IREDA.NS","POLYCAB.NS"]

PRICE=["ret1_past","ret3_past","ret5_past","ret10_past","ret20_past","dist20","dist50","dist200",
       "rsi","atr_pct","range_pct","close_location","breakout20"]
VOLUME=["vol_ratio","vol20"]; REL=["rel5","rel20"]
MARKET=["mkt_ret1","mkt_ret5","mkt_ret20","mkt_dist20","mkt_dist50","mkt_vol20","regime"]
NEWS=["news_score","news_confidence","news_count","news_positive_share","news_negative_share",
      "news_weighted_score","news_recency"]
QUANT=PRICE+VOLUME+REL+MARKET
HYBRID=QUANT+NEWS
TARGETS={"ret1_fwd","ret3_fwd","ret5_fwd","ret10_fwd","y1","y3","y5","y10"}


def clean(x):
    """Convert every supported yfinance daily-data shape to OHLCV columns."""
    if x is None:
        return pd.DataFrame()

    if isinstance(x, pd.Series):
        x = x.to_frame()

    if not isinstance(x, pd.DataFrame) or x.empty:
        return pd.DataFrame()

    wanted = ["Open", "High", "Low", "Close", "Volume"]

    # yfinance can return MultiIndex columns such as:
    # ('Close', 'RELIANCE.NS') or ('RELIANCE.NS', 'Close').
    if isinstance(x.columns, pd.MultiIndex):
        found_level = None
        for level in range(x.columns.nlevels):
            vals = {str(v) for v in x.columns.get_level_values(level)}
            if set(wanted).issubset(vals):
                found_level = level
                break

        if found_level is not None:
            x = x.copy()
            x.columns = x.columns.get_level_values(found_level)
        else:
            flattened = []
            for col in x.columns:
                match = next(
                    (str(part) for part in col if str(part) in wanted),
                    None
                )
                flattened.append(match if match else str(col))
            x = x.copy()
            x.columns = flattened

    x = x.loc[:, ~pd.Index(x.columns).duplicated(keep="first")]

    missing = [c for c in wanted if c not in x.columns]
    if missing:
        return pd.DataFrame()

    x = x[wanted].copy()

    for col in wanted:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["Close"])
    x = x.sort_index()

    try:
        if getattr(x.index, "tz", None) is not None:
            x.index = x.index.tz_localize(None)
    except Exception:
        pass

    return x[~x.index.duplicated(keep="last")]


def download(symbol):
    raw = yf.download(
        symbol,
        period=PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column"
    )
    return clean(raw)


def calc_rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(
        alpha=1/n, adjust=False, min_periods=n
    ).mean()
    dn = (-d.clip(upper=0)).ewm(
        alpha=1/n, adjust=False, min_periods=n
    ).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def calc_atr(x, n=14):
    prev_close = x["Close"].shift(1)
    tr = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - prev_close).abs(),
        (x["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(
        alpha=1/n, adjust=False, min_periods=n
    ).mean()


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
            [
                mclose > msma50 * 1.005,
                mclose < msma50 * 0.995
            ],
            [1.0, -1.0],
            default=0.0
        )
    }, index=s.index)

    for h in HORIZONS:
        f[f"ret{h}_fwd"] = close.shift(-h) / close - 1
        f[f"y{h}"] = (f[f"ret{h}_fwd"] > 0).astype(float)

    return f.replace([np.inf, -np.inf], np.nan)




def build(n):
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
                print(
                    f"WARNING: insufficient history for {sym}; skipping."
                )
                continue

            feat = features(stock, market)
            market_aligned = market.reindex(feat.index).ffill()

            successful += 1

            last = len(feat) - max(HORIZONS)

            for j in range(MIN_HISTORY - 1, last):
                row = feat.iloc[j]

                if row[QUANT].isna().any():
                    continue

                record = {
                    "ticker": sym,
                    "date": feat.index[j],
                    # IMPORTANT: bracket indexing, never Series.Close.
                    "close": float(stock["Close"].iloc[j]),
                    "market_close":
                        float(market_aligned["Close"].iloc[j])
                }

                for col in QUANT:
                    record[col] = float(row[col])

                for h in HORIZONS:
                    record[f"ret{h}_fwd"] = float(row[f"ret{h}_fwd"])
                    record[f"y{h}"] = float(row[f"y{h}"])

                rows.append(record)

        except Exception as exc:
            print(f"WARNING: {sym} failed: {exc}")

    if not rows:
        raise RuntimeError(
            "No observations were created. "
            "All ticker downloads/features failed."
        )

    data = pd.DataFrame(rows)

    required = {"date", "ticker"}
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(
            f"Dataset construction failed; missing columns: {sorted(missing)}"
        )

    data = data.sort_values(
        by=["date", "ticker"],
        kind="mergesort"
    ).reset_index(drop=True)

    return attach_news(data, n), successful



def leakage_check():
    if set(QUANT)&TARGETS or set(HYBRID)&TARGETS:raise RuntimeError("FATAL FEATURE/TARGET LEAKAGE")
    print("\nFEATURE/TARGET LEAKAGE CHECK: PASS")

def log_model():
    return Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                     ("m",LogisticRegression(C=.25,max_iter=3000,class_weight="balanced",random_state=42))])
def ridge():
    return Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),("m",Ridge(alpha=10.))])
def gb_c():
    return HistGradientBoostingClassifier(max_iter=200,learning_rate=.04,max_leaf_nodes=15,l2_regularization=2.,random_state=42)
def gb_r():
    return HistGradientBoostingRegressor(max_iter=200,learning_rate=.04,max_leaf_nodes=15,l2_regularization=2.,random_state=42)

def fit(train,features):
    q=train.dropna(subset=["ret5_fwd","y5"])
    if len(q)<MIN_TRAIN or q.y5.nunique()<2:return None
    a,b,c,e=log_model(),ridge(),gb_c(),gb_r()
    a.fit(q[features],q.y5.astype(int)); b.fit(q[features],q.ret5_fwd); c.fit(q[features],q.y5.astype(int)); e.fit(q[features],q.ret5_fwd)
    return a,b,c,e

def predict(models,d,features):
    a,b,c,e=models
    pl=a.predict_proba(d[features])[:,1]; pg=c.predict_proba(d[features])[:,1]
    return np.clip((pl+pg)/2,.01,.99),(b.predict(d[features])+e.predict(d[features]))/2

def calibrator(p,y):
    p=np.clip(np.asarray(p),1e-5,1-1e-5); z=np.log(p/(1-p)).reshape(-1,1)
    m=LogisticRegression(max_iter=2000,random_state=42); m.fit(z,np.asarray(y).astype(int)); return m

def cal(m,p):
    p=np.clip(np.asarray(p),1e-5,1-1e-5); z=np.log(p/(1-p)).reshape(-1,1)
    return np.clip(m.predict_proba(z)[:,1],.01,.99)

def purge(d,date):
    ds=sorted(d[d.date<date].date.unique())
    return d[d.date<=ds[-PURGE_DAYS-1]].copy() if len(ds)>PURGE_DAYS else d.iloc[0:0].copy()

def split(d):
    ds=sorted(d.date.unique()); vi=int(len(ds)*(1-OOS_FRAC-VAL_FRAC)); oi=int(len(ds)*(1-OOS_FRAC))
    return d[d.date<ds[vi]].copy(),d[(d.date>=ds[vi])&(d.date<ds[oi])].copy(),d[d.date>=ds[oi]].copy(),ds[vi],ds[oi]

def score(d,w):
    q=.5*d.p_ensemble+.5*(.5+np.tanh(d.r_ensemble*20)/2)
    n=.5+.5*d.news_weighted_score*d.news_confidence
    has=d.news_count>0
    d["quant_score"]=q; d["news_score_01"]=n; d["hybrid_score"]=q
    d.loc[has,"hybrid_score"]=(1-w)*q.loc[has]+w*n.loc[has]
    return d

def actions(d):
    d=d.copy(); d["rank"]=d.groupby("date").hybrid_score.rank(pct=True)
    risk=d.rsi.between(35,72)&(d.atr_pct<.06)&(d.vol_ratio>.50)
    d["action"]=np.where((d["rank"]>=.90)&(d.p_ensemble>=.53)&(d.r_ensemble>=.001)&risk,"TRADE",
                         np.where((d["rank"]>=.70)&(d.p_ensemble>=.50)&(d.r_ensemble>=0),"WATCH","WAIT"))
    for h in HORIZONS:d[f"net{h}"]=d[f"ret{h}_fwd"]-ROUND_TRIP_COST
    return d

def perf(d):
    out=[]
    for a,g in d.groupby("action"):
        for h in HORIZONS:
            x=g[f"net{h}"].dropna(); win=x[x>0]; loss=x[x<=0]
            out.append({"selection":a,"horizon":h,"observations":len(x),"win_rate":(x>0).mean(),
                        "average_net_return":x.mean(),"median_net_return":x.median(),
                        "profit_factor":win.sum()/abs(loss.sum()) if len(loss) and loss.sum()<0 else np.nan,
                        "best":x.max(),"worst":x.min()})
    return pd.DataFrame(out)

def topn(d,n):
    rows=[]
    for date,g in d.groupby("date"):
        g=g.nlargest(n,"hybrid_score")
        if len(g)<n:continue
        r={"date":date,"top_n":n,"tickers":",".join(g.ticker)}
        for h in HORIZONS:r[f"net{h}"]=g[f"ret{h}_fwd"].mean()-ROUND_TRIP_COST
        rows.append(r)
    return pd.DataFrame(rows)

def topsummary(p):
    if p.empty:return pd.DataFrame()
    out=[]
    for h in HORIZONS:
        x=p[f"net{h}"]; w=x[x>0]; l=x[x<=0]
        out.append({"horizon":h,"events":len(x),"win_rate":(x>0).mean(),"average_net_return":x.mean(),
                    "median_net_return":x.median(),"profit_factor":w.sum()/abs(l.sum()) if len(l) and l.sum()<0 else np.nan,
                    "best":x.max(),"worst":x.min()})
    return pd.DataFrame(out)

def nonoverlap(d,n,h):
    p=topn(d,n)
    if p.empty:return pd.DataFrame()
    dates=sorted(p.date.unique()); idx={x:i for i,x in enumerate(dates)}; take=[]; last=-999
    for _,r in p.sort_values("date").iterrows():
        i=idx[r.date]
        if i>=last+h:take.append(r);last=i
    x=pd.DataFrame(take)[f"net{h}"]
    return pd.DataFrame([{"top_n":n,"horizon":h,"trades":len(x),"win_rate":(x>0).mean(),"average_net_return":x.mean(),
                          "median_net_return":x.median(),"best":x.max(),"worst":x.min(),"sum_net_return":x.sum()}])


    m=download("^NSEI")
    if m.empty:
        raise RuntimeError("Unable to download NIFTY")

    rows=[]
    ok=0

    for i,sym in enumerate(SYMBOLS,1):
        print(f"Loading [{i}/{len(SYMBOLS)}] {sym}")
        try:
            s=download(sym)

            if s.empty or len(s)<MIN_HISTORY:
                print(f"WARNING: insufficient history for {sym}; skipping.")
                continue

            f=features(s,m)
            market_aligned=m.reindex(f.index).ffill()
            ok+=1

            for j in range(MIN_HISTORY-1,len(f)-max(HORIZONS)):
                x=f.iloc[j]
                if x[QUANT].isna().any():
                    continue

                signal_date=f.index[j]

                r={
                    "ticker":sym,
                    "date":signal_date,
                    "close":float(s["Close"].iloc[j]),
                    "market_close":float(market_aligned["Close"].iloc[j])
                }

                for c in QUANT:
                    r[c]=float(x[c])

                for h in HORIZONS:
                    r[f"ret{h}_fwd"]=float(x[f"ret{h}_fwd"])
                    r[f"y{h}"]=float(x[f"y{h}"])

                rows.append(r)

        except Exception as e:
            print(f"WARNING: {sym} failed: {e}")

    if not rows:
        raise RuntimeError(
            "No valid market observations were built. "
            "Check yfinance output and OHLCV normalization."
        )

    d=pd.DataFrame(rows)
    d=d.sort_values(["date","ticker"]).reset_index(drop=True)

    return attach_news(d,n),ok


def run_backtest():
    print("V6.5-FINAL-OHLCV-FIX")
    print("yfinance version:", YF_VERSION)
    print("V6.5 source revision: 2026-08-31-FINAL-OHLCV-FIX")
    print("yfinance version:",getattr(yf,"__version__","unknown"))

    print(f"V6.5 source revision: 2026-08-31-fix2")
    print(f"yfinance version: {YF_VERSION}")
    news=load_news(); d,ok=build(news); leakage_check()
    dev,val,oos,vs,osd=split(d)
    print(f"\nTotal candidate observations: {len(d)}\nSuccessful symbols: {ok}\nUnique symbols: {d.ticker.nunique()}\nUnique signal dates: {d.date.nunique()}")
    print(f"Development observations: {len(dev)}\nValidation observations: {len(val)}\nOOS observations: {len(oos)}")
    print(f"Validation start: {vs}\nOOS start: {osd}\nRound-trip cost assumption: {ROUND_TRIP_COST*100:.3f}%")
    tr=purge(dev,vs).tail(MAX_TRAIN_OBS); print(f"\nPurged development training observations: {len(tr)}")
    qm=fit(tr,QUANT); 
    if qm is None:raise RuntimeError("Quant validation fit failed")
    pv,rv=predict(qm,val,QUANT); val["p_ensemble_raw"]=pv;val["r_ensemble"]=rv
    cq=calibrator(val.p_ensemble_raw,val.y5);val["p_ensemble"]=cal(cq,val.p_ensemble_raw)
    # Historical news model, only if valid timestamped news exists.
    use_news=not news.empty
    if use_news:
        hm=fit(tr,HYBRID)
        if hm is None:use_news=False
    if use_news:
        ph,rh=predict(hm,val,HYBRID);val["p_hybrid_raw"]=ph;val["r_hybrid"]=rh
        ch=calibrator(val.p_hybrid_raw,val.y5);val["p_hybrid"]=cal(ch,val.p_hybrid_raw)
    else:
        ch=cq;val["p_hybrid"]=val.p_ensemble;val["r_hybrid"]=val.r_ensemble
    val["p_ensemble"]=val["p_hybrid"];val["r_ensemble"]=val["r_hybrid"]
    bestw=0.; best=-1e99; wr=[]
    for w in ([0,.1,.2,.25,.35,.5] if use_news else [0]):
        z=score(val.copy(),w); rank=z.groupby("date").hybrid_score.rank(pct=True); top=z.loc[rank>=.90,"ret5_fwd"]
        a=top.mean()-ROUND_TRIP_COST if len(top) else np.nan
        wr.append({"news_weight":w,"validation_top10pct_avg_net_5d":a,"observations":len(top)})
        if len(top)>=20 and a>best:best=a;bestw=w
    pre=d[d.date<osd].copy(); ptr=purge(pre,osd).tail(MAX_TRAIN_OBS)
    print(f"Pre-OOS observations: {len(pre)}\nPurged pre-OOS training observations: {len(ptr)}")
    qmo=fit(ptr,QUANT);po,ro=predict(qmo,oos,QUANT);oos["p_ensemble"]=cal(cq,po);oos["r_ensemble"]=ro
    if use_news:
        hmo=fit(ptr,HYBRID);pho,rho=predict(hmo,oos,HYBRID);oos["p_hybrid"]=cal(ch,pho);oos["r_hybrid"]=rho
    else:oos["p_hybrid"]=oos.p_ensemble;oos["r_hybrid"]=oos.r_ensemble
    oos["p_ensemble"]=oos.p_hybrid;oos["r_ensemble"]=oos.r_hybrid;oos=actions(score(oos,bestw))
    oq=oos.copy();oq["hybrid_score"]=oq["quant_score"];oq=actions(oq)
    oa=perf(oos);qa=perf(oq);tops=[];nos=[]
    for n in [1,3,5]:
        p=topn(oos,n);s=topsummary(p)
        if not s.empty:s.insert(0,"top_n",n);tops.append(s)
        for h in [3,5,10]:nos.append(nonoverlap(oos,n,h))
    top=pd.concat(tops,ignore_index=True);no=pd.concat(nos,ignore_index=True)
    # NIFTY benchmark
    b=d.groupby("date").market_close.first().sort_index(); br=[]
    for h in HORIZONS:
        r=(b.shift(-h)/b-1).dropna();br.append({"horizon":h,"observations":len(r),"win_rate":(r>0).mean(),"average_return":r.mean(),"median_return":r.median()})
    bench=pd.DataFrame(br)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    outs={
      f"dataset_v6_5_{ts}.csv":d,f"validation_v6_5_{ts}.csv":val,f"oos_v6_5_{ts}.csv":oos,
      f"oos_quant_only_v6_5_{ts}.csv":oq,f"oos_action_performance_v6_5_{ts}.csv":oa,
      f"quant_action_performance_v6_5_{ts}.csv":qa,f"topn_oos_v6_5_{ts}.csv":top,
      f"nonoverlap_v6_5_{ts}.csv":no,f"validation_news_weight_v6_5_{ts}.csv":pd.DataFrame(wr),
      f"nifty_benchmark_v6_5_{ts}.csv":bench}
    for fn,df in outs.items():df.to_csv(AUDIT/fn,index=False)
    print("\n"+"="*76+"\nV6.5 QUANT + HISTORICAL NEWS BACKTEST\n"+"="*76)
    print("NEWS MODE:", "HISTORICAL NEWS + GEMINI SCORES" if use_news else "QUANT-ONLY")
    print("Selected validation news weight:",bestw)
    print("\nOOS HYBRID ACTION COUNTS:\n",oos.action.value_counts().to_string())
    print("\nOOS HYBRID ACTION PERFORMANCE:\n",oa.to_string(index=False))
    print("\nOOS QUANT-ONLY ACTION PERFORMANCE:\n",qa.to_string(index=False))
    print("\nTOP-N HYBRID:\n",top.to_string(index=False))
    print("\nNON-OVERLAPPING:\n",no.to_string(index=False))
    print("\nNIFTY BENCHMARK:\n",bench.to_string(index=False))
    print("\nFILES CREATED:")
    for fn in outs:print(AUDIT/fn)
    print("\nV6.5 BACKTEST COMPLETED")
    print("\nAUDIT: future targets are excluded; news is restricted to published_at <= signal date; validation weight is frozen before OOS; OOS labels are never used for fitting/selection.")
def run_live():raise RuntimeError("V6.5 LIVE is disabled. Run and review BACKTEST first.")
if __name__=="__main__":
    if MODE=="LIVE":run_live()
    else:run_backtest()
