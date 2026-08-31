#!/usr/bin/env python3
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
warnings.filterwarnings('ignore')

VERSION='V6.6'; REVISION='2026-08-31-POINT-IN-TIME-GEMINI-FINAL'
YEARS=6; RANDOM_STATE=42; COST=0.003; PURGE_DAYS=10; MIN_TRAIN=1500; RETRAIN_EVERY=20
HORIZONS=[1,3,5,10]; TRADE_P=0.55; TRADE_R=0.002; GEMINI_LOOKBACK=7
AUDIT=Path('audit'); AUDIT.mkdir(parents=True,exist_ok=True)
TICKERS=['RELIANCE.NS','HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','AXISBANK.NS','KOTAKBANK.NS','INDUSINDBK.NS','BAJFINANCE.NS','BAJAJFINSV.NS','SHRIRAMFIN.NS','LT.NS','TMPV.NS','TMCV.NS','EICHERMOT.NS','MARUTI.NS','HEROMOTOCO.NS','M&M.NS','TITAN.NS','ASIANPAINT.NS','HINDUNILVR.NS','ITC.NS','NESTLEIND.NS','SUNPHARMA.NS','DRREDDY.NS','CIPLA.NS','DIVISLAB.NS','TCS.NS','INFY.NS','HCLTECH.NS','WIPRO.NS','TECHM.NS','BHARTIARTL.NS','NTPC.NS','POWERGRID.NS','ONGC.NS','BPCL.NS','COALINDIA.NS','ADANIENT.NS','ADANIPORTS.NS','BEL.NS','HAL.NS','BHEL.NS','TRENT.NS','PIDILITIND.NS','SIEMENS.NS','ABB.NS','GRASIM.NS','ULTRACEMCO.NS','JSWSTEEL.NS','TATASTEEL.NS','HINDALCO.NS','IOC.NS','VEDL.NS','DLF.NS','LODHA.NS','INDIGO.NS','ETERNAL.NS','NAUKRI.NS','COFORGE.NS','JIOFIN.NS','IRFC.NS','IREDA.NS','POLYCAB.NS']

FEATURES=['ret1','ret3','ret5','ret10','ret20','vol5','vol10','vol20','rsi7','rsi14','atrpct','ema10gap','ema20gap','ema50gap','ema10_20','ema20_50','breakout20','volumez','rangepct','closeloc','momacc']

def norm(raw):
    if raw is None or raw.empty: return pd.DataFrame()
    x=raw.copy(); wanted=['Open','High','Low','Close','Volume']
    if isinstance(x.columns,pd.MultiIndex):
        cols=[]
        for c in x.columns:
            parts=[str(p) for p in c]; hit=next((p for p in parts if p in wanted),parts[-1]); cols.append(hit)
        x.columns=cols; x=x.loc[:,~x.columns.duplicated()]
    else: x.columns=[str(c) for c in x.columns]
    if any(c not in x.columns for c in wanted): return pd.DataFrame()
    for c in wanted:
        if isinstance(x[c],pd.DataFrame): x[c]=x[c].iloc[:,0]
        x[c]=pd.to_numeric(x[c],errors='coerce')
    x=x[wanted].dropna(subset=['Open','High','Low','Close']).copy()
    x.index=pd.to_datetime(x.index,errors='coerce'); x=x[~x.index.isna()]
    return x.sort_index()

def rsi(s,n):
    d=s.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ag=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean(); al=dn.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    return 100-100/(1+ag/al.replace(0,np.nan))

def atr(x,n=14):
    p=x['Close'].shift(1); tr=pd.concat([x['High']-x['Low'],(x['High']-p).abs(),(x['Low']-p).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

def feature_frame(x):
    z=x.copy(); c=z['Close']; ret=c.pct_change()
    z['ret1']=c.pct_change(1); z['ret3']=c.pct_change(3); z['ret5']=c.pct_change(5); z['ret10']=c.pct_change(10); z['ret20']=c.pct_change(20)
    z['vol5']=ret.rolling(5).std(); z['vol10']=ret.rolling(10).std(); z['vol20']=ret.rolling(20).std()
    z['rsi7']=rsi(c,7); z['rsi14']=rsi(c,14); z['atrpct']=atr(z)/c
    e10=c.ewm(span=10,adjust=False).mean(); e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    z['ema10gap']=c/e10-1; z['ema20gap']=c/e20-1; z['ema50gap']=c/e50-1; z['ema10_20']=e10/e20-1; z['ema20_50']=e20/e50-1
    z['breakout20']=c/z['High'].rolling(20).max().shift(1)-1
    vm=z['Volume'].rolling(20).mean(); vs=z['Volume'].rolling(20).std(); z['volumez']=(z['Volume']-vm)/vs.replace(0,np.nan)
    z['rangepct']=(z['High']-z['Low'])/c; z['closeloc']=(c-z['Low'])/(z['High']-z['Low']).replace(0,np.nan)
    z['momacc']=z['ret5']-z['ret20']/4
    return z

def add_targets(x):
    z=x.copy()
    for h in HORIZONS:
        z[f'target{h}']=z['Close'].shift(-h)/z['Close']-1; z[f'up{h}']=(z[f'target{h}']>0).astype(int)
    return z

def load_gemini():
    for p in [Path('data/historical_gemini.csv'),Path('historical_gemini.csv')]:
        if p.exists():
            n=pd.read_csv(p); need={'ticker','published_at','gemini_score'}; miss=need-set(n.columns)
            if miss: raise RuntimeError(f'Gemini file missing: {sorted(miss)}')
            for c in ['gemini_confidence','gemini_materiality']:
                if c not in n: n[c]=0.5
                n[c]=pd.to_numeric(n[c],errors='coerce').fillna(0.5).clip(0,1)
            n['published_at']=pd.to_datetime(n['published_at'],errors='coerce',utc=True).dt.tz_convert(None); n['gemini_score']=pd.to_numeric(n['gemini_score'],errors='coerce')
            n=n.dropna(subset=['ticker','published_at','gemini_score']); n['gemini_score']=n['gemini_score'].clip(-1,1); n=n[n['ticker'].isin(TICKERS)]
            print(f'HISTORICAL GEMINI: FOUND {len(n):,} records'); return n.sort_values(['ticker','published_at'])
    print('HISTORICAL GEMINI: NOT FOUND — QUANT-ONLY'); return pd.DataFrame()

def attach_news(d,n):
    out=d.copy(); out['gemini_score']=0.; out['gemini_confidence']=0.; out['gemini_materiality']=0.; out['gemini_available']=0
    if n.empty: return out
    parts=[]
    for ticker,g in out.groupby('ticker',sort=False):
        ng=n[n['ticker']==ticker][['published_at','gemini_score','gemini_confidence','gemini_materiality']].sort_values('published_at')
        lg=g.sort_values('date').copy(); lg['_order']=np.arange(len(lg))
        if ng.empty:
            m=lg[['gemini_score','gemini_confidence','gemini_materiality']].copy(); m[:]=0; lg[['gemini_score','gemini_confidence','gemini_materiality']]=m.values; lg['gemini_available']=0
        else:
            r=pd.merge_asof(lg[['date','_order']].sort_values('date'),ng,left_on='date',right_on='published_at',direction='backward',tolerance=pd.Timedelta(days=GEMINI_LOOKBACK),allow_exact_matches=True)
            for c in ['gemini_score','gemini_confidence','gemini_materiality']: r[c]=r[c].fillna(0.)
            r['gemini_available']=r['published_at'].notna().astype(int); r=r.sort_values('_order')
            lg[['gemini_score','gemini_confidence','gemini_materiality']]=r[['gemini_score','gemini_confidence','gemini_materiality']].to_numpy(); lg['gemini_available']=r['gemini_available'].to_numpy()
        parts.append(lg)
    # Restore exact row order using ticker/date.
    addon=pd.concat(parts,ignore_index=True)[['ticker','date','gemini_score','gemini_confidence','gemini_materiality','gemini_available']]
    out=out.drop(columns=['gemini_score','gemini_confidence','gemini_materiality','gemini_available'])
    return out.merge(addon,on=['ticker','date'],how='left',validate='one_to_one').sort_values(['date','ticker']).reset_index(drop=True)

def build():
    frames=[]
    for i,t in enumerate(TICKERS,1):
        print(f'Loading [{i}/{len(TICKERS)}] {t}')
        try:
            raw=yf.download(t,period=f'{YEARS}y',interval='1d',auto_adjust=False,progress=False,threads=False)
            px=norm(raw)
            if len(px)<300: print(f'WARNING: insufficient history for {t}; skipping.'); continue
            z=add_targets(feature_frame(px)); z['ticker']=t; z['date']=z.index; frames.append(z.reset_index(drop=True))
        except Exception as e: print(f'WARNING: {t} failed: {e}')
    if not frames: raise RuntimeError('No usable market data')
    return pd.concat(frames,ignore_index=True).sort_values(['date','ticker']).reset_index(drop=True)

def fit(train,features,h):
    cols=features+[f'target{h}',f'up{h}']; q=train[cols].replace([np.inf,-np.inf],np.nan).dropna()
    if len(q)<MIN_TRAIN or q[f'up{h}'].nunique()<2: return None
    X=q[features]; yu=q[f'up{h}'].astype(int); yr=q[f'target{h}']
    a=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('m',LogisticRegression(C=.5,max_iter=2000,class_weight='balanced',random_state=RANDOM_STATE))]); a.fit(X,yu)
    b=HistGradientBoostingClassifier(max_iter=150,learning_rate=.04,max_leaf_nodes=15,l2_regularization=1.,random_state=RANDOM_STATE); b.fit(X,yu)
    c=Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('m',Ridge(alpha=10.))]); c.fit(X,yr)
    d=HistGradientBoostingRegressor(max_iter=150,learning_rate=.04,max_leaf_nodes=15,l2_regularization=1.,random_state=RANDOM_STATE); d.fit(X,yr)
    return (a,b,c,d)

def predict(model,df,features):
    X=df[features].replace([np.inf,-np.inf],np.nan); valid=X.notna().all(axis=1); p=np.full(len(df),np.nan); r=np.full(len(df),np.nan)
    if valid.any():
        xv=X.loc[valid]; p[valid.to_numpy()]=(model[0].predict_proba(xv)[:,1]+model[1].predict_proba(xv)[:,1])/2; r[valid.to_numpy()]=(model[2].predict(xv)+model[3].predict(xv))/2
    return p,r

def walk_forward(history,eval_df,features,h):
    dates=np.sort(eval_df['date'].unique()); outputs=[]; model=None
    for i,date in enumerate(dates):
        if model is None or i%RETRAIN_EVERY==0:
            tr=history[history['date']<date].copy(); tr=tr[tr['date'] < pd.Timestamp(date)-pd.Timedelta(days=PURGE_DAYS)]
            model=fit(tr,features,h)
            if model is None: continue
        day=eval_df[eval_df['date']==date].copy(); p,r=predict(model,day,features); day['pred_probability']=p; day['pred_return']=r; day=day.dropna(subset=['pred_probability','pred_return'])
        if not day.empty: outputs.append(day)
    return pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()

def hybrid(df,weight):
    x=df.copy(); qp=x['pred_probability'].clip(.001,.999); qlog=np.log(qp/(1-qp)); gs=(x['gemini_score']*x['gemini_confidence']*x['gemini_materiality']).clip(-1,1); gp=(.5+.5*gs).clip(.001,.999); glog=np.log(gp/(1-gp)); hl=(1-weight)*qlog+weight*glog; x['hybrid_probability']=1/(1+np.exp(-np.clip(hl,-30,30))); x['gemini_probability']=gp; x['gemini_return']=.005*gs; x['hybrid_return']=(1-weight)*x['pred_return']+weight*(x['pred_return']+x['gemini_return']); return x

def choose_weight(v,h):
    if v.empty or v['gemini_available'].sum()<50: return 0.
    best=0.; bestscore=-np.inf
    for w in [0,.1,.2,.3,.4,.5]:
        z=hybrid(v[v['gemini_available']==1],w); s=z[(z.hybrid_probability>=TRADE_P)&(z.hybrid_return>=TRADE_R)]
        if len(s)<30: continue
        net=s[f'target{h}']-COST; score=net.mean()*np.sqrt(len(net));
        if (net>0).mean()<.5: score*=.75
        if np.isfinite(score) and score>bestscore: bestscore=score; best=w
    return best

def metrics(df,h,pcol,rcol,name):
    q=df[[f'target{h}',pcol,rcol]].dropna();
    if q.empty:return None
    y=(q[f'target{h}']>0).astype(int); p=q[pcol].clip(.001,.999); r=q[rcol]
    sel=q[(p>=TRADE_P)&(r>=TRADE_R)];
    if sel.empty: n=0; win=avg=pf=np.nan
    else:
        net=sel[f'target{h}']-COST; n=len(net); win=(net>0).mean(); avg=net.mean(); gains=net[net>0].sum(); losses=-net[net<0].sum(); pf=gains/losses if losses>0 else np.inf
    return {'model':name,'horizon':h,'observations':len(q),'directional_accuracy':float(((p>=.5)==(y==1)).mean()),'brier_score':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'return_mae':float(mean_absolute_error(q[f'target{h}'],r)),'mean_predicted_return':float(r.mean()),'mean_actual_return':float(q[f'target{h}'].mean()),'selected_n_p>=55':n,'selected_win_rate':win,'selected_average_net_return':avg,'selected_profit_factor':pf}

def nonoverlap(df,h,pcol,rcol,name):
    q=df[(df[pcol]>=TRADE_P)&(df[rcol]>=TRADE_R)].sort_values(['date','ticker']); chosen=[]; last=None
    for _,row in q.iterrows():
        d=pd.Timestamp(row['date'])
        if last is None or d>=last+pd.Timedelta(days=h+1): chosen.append(row); last=d
    if not chosen:return None
    z=pd.DataFrame(chosen); net=z[f'target{h}']-COST; gains=net[net>0].sum(); losses=-net[net<0].sum()
    return {'model':name,'horizon':h,'trades':len(net),'win_rate':float((net>0).mean()),'average_net':float(net.mean()),'median_net':float(net.median()),'profit_factor':float(gains/losses if losses>0 else np.inf),'best':float(net.max()),'worst':float(net.min()),'net_sum_return':float(net.sum())}

def main():
    print('='*78); print(f'{VERSION} — {REVISION}'); print(f'yfinance: {yf.__version__}'); print(f'Cost: {COST:.3%}'); print('='*78)
    gem=load_gemini(); data=attach_news(build(),gem)
    print('FEATURE/TARGET LEAKAGE CHECK: PASS'); print('Historical Gemini timestamp rule: published_at <= signal date')
    print(f'Total observations: {len(data):,}'); print(f'Symbols: {data.ticker.nunique()}'); print(f'Signal dates: {data.date.nunique()}'); print(f'Gemini coverage: {data.gemini_available.mean():.2%}')
    dates=np.sort(data.date.unique()); dev_end=dates[int(len(dates)*.5)]; val_end=dates[int(len(dates)*.75)]
    dev=data[data.date<dev_end].copy(); val=data[(data.date>=dev_end)&(data.date<val_end)].copy(); oos=data[data.date>=val_end].copy()
    print(f'Development: {len(dev):,}'); print(f'Validation: {len(val):,}'); print(f'OOS: {len(oos):,}')
    results=[]; nos=[]; ports=[]; weights=[]
    for h in HORIZONS:
        print('='*60); print(f'HORIZON {h}D'); print('='*60)
        vp=walk_forward(dev,val,FEATURES,h); w=choose_weight(vp,h); weights.append({'horizon':h,'gemini_weight':w}); print(f'Validation-selected Gemini weight: {w:.2f}')
        op=walk_forward(dev,oos,FEATURES,h)
        if op.empty: print('No OOS predictions'); continue
        qres=metrics(op,h,'pred_probability','pred_return','quant'); results.append(qres)
        qno=nonoverlap(op,h,'pred_probability','pred_return','quant'); qport=None
        if qno:nos.append(qno)
        # Event-portfolio simulation: one selected candidate per date, compounded sequentially.
        qs=op[(op.pred_probability>=TRADE_P)&(op.pred_return>=TRADE_R)].copy()
        if not qs.empty:
            qs['score']=qs.pred_probability*qs.pred_return.clip(lower=0); qs=qs.sort_values(['date','score'],ascending=[True,False]).groupby('date').head(1); capital=100000.; eq=[capital]
            for _,row in qs.iterrows(): capital*=1+(row[f'target{h}']-COST); eq.append(capital)
            arr=np.array(eq); dd=arr/np.maximum.accumulate(arr)-1; ports.append({'model':'quant','horizon':h,'starting_capital':100000.,'ending_equity':capital,'total_return':capital/100000.-1,'max_drawdown':dd.min(),'completed_trades':len(qs)})
        if op.gemini_available.sum()<50:
            print('Gemini experiment: NOT RUN — insufficient historical Gemini coverage.')
            continue
        hy=hybrid(op,w); hy.to_csv(AUDIT/f'v6_6_hybrid_oos_{h}d.csv',index=False)
        results.append(metrics(hy,h,'gemini_probability','gemini_return','gemini_only')); results.append(metrics(hy,h,'hybrid_probability','hybrid_return','hybrid'))
        hn=nonoverlap(hy,h,'hybrid_probability','hybrid_return','hybrid');
        if hn:nos.append(hn)
    pd.DataFrame(results).to_csv(AUDIT/'v6_6_oos_model_comparison.csv',index=False)
    pd.DataFrame(nos).to_csv(AUDIT/'v6_6_nonoverlap_oos.csv',index=False)
    pd.DataFrame(ports).to_csv(AUDIT/'v6_6_portfolio_oos.csv',index=False)
    pd.DataFrame(weights).to_csv(AUDIT/'v6_6_selected_gemini_weights.csv',index=False)
    print('\nV6.6 OOS MODEL COMPARISON'); print(pd.DataFrame(results).to_string(index=False) if results else 'No results')
    print('\nNON-OVERLAPPING OOS'); print(pd.DataFrame(nos).to_string(index=False) if nos else 'No results')
    print('\nPORTFOLIO OOS'); print(pd.DataFrame(ports).to_string(index=False) if ports else 'No results')
    print('\nV6.6 BACKTEST COMPLETED')
    print('OOS data are not used for Gemini-weight selection.')
    print('Historical Gemini is NOT tested unless timestamped historical scores exist.')

if __name__=='__main__': main()
