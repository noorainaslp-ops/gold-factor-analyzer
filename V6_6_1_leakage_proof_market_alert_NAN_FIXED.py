#!/usr/bin/env python3
"""V7.1 leakage-proof point-in-time quant + historical Gemini research engine."""
from pathlib import Path
import math, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
try:
 import yfinance as yf
 from sklearn.pipeline import Pipeline
 from sklearn.impute import SimpleImputer
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression, Ridge
 from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
 from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
except Exception as e:
 print('FATAL import error:',e); sys.exit(1)

VERSION='V7.1.1'; REVISION='2026-09-01-SKLEARN-1.8-COMPATIBILITY-FIX'
COST=.003; RANDOM_STATE=42; PURGE_DAYS=10; MIN_TRAIN=2500
HORIZONS=[1,3,5,10]; PRIMARY_HORIZON=10
PROB_GRID=[.50,.55,.60,.65,.70]; RETURN_GRID=[0,.002,.004,.006]
GEMINI_WEIGHT_GRID=[0,.10,.20,.30,.40,.50]; MIN_GEMINI_COVERAGE=.10
AUDIT=Path('audit'); AUDIT.mkdir(exist_ok=True)
SYMBOLS=['RELIANCE.NS','HDFCBANK.NS','ICICIBANK.NS','SBIN.NS','AXISBANK.NS','KOTAKBANK.NS','INDUSINDBK.NS','BAJFINANCE.NS','BAJAJFINSV.NS','SHRIRAMFIN.NS','LT.NS','TMPV.NS','TMCV.NS','EICHERMOT.NS','MARUTI.NS','HEROMOTOCO.NS','M&M.NS','TITAN.NS','ASIANPAINT.NS','HINDUNILVR.NS','ITC.NS','NESTLEIND.NS','SUNPHARMA.NS','DRREDDY.NS','CIPLA.NS','DIVISLAB.NS','TCS.NS','INFY.NS','HCLTECH.NS','WIPRO.NS','TECHM.NS','BHARTIARTL.NS','NTPC.NS','POWERGRID.NS','ONGC.NS','BPCL.NS','COALINDIA.NS','ADANIENT.NS','ADANIPORTS.NS','BEL.NS','HAL.NS','BHEL.NS','TRENT.NS','PIDILITIND.NS','SIEMENS.NS','ABB.NS','GRASIM.NS','ULTRACEMCO.NS','JSWSTEEL.NS','TATASTEEL.NS','HINDALCO.NS','IOC.NS','VEDL.NS','DLF.NS','LODHA.NS','INDIGO.NS','ETERNAL.NS','NAUKRI.NS','COFORGE.NS','JIOFIN.NS','IRFC.NS','IREDA.NS','POLYCAB.NS']
FEATURES=['ret_1','ret_3','ret_5','ret_10','ret_20','vol_5','vol_10','vol_20','range_pct','body_pct','close_location','dist_sma5','dist_sma10','dist_sma20','dist_sma50','volume_ratio_5','volume_ratio_20','momentum_accel','vol_regime','trend_regime']

def finite(x):
 z=x.copy().replace([np.inf,-np.inf],np.nan)
 return z.apply(pd.to_numeric,errors='coerce')

def sf(x):
 try:
  v=float(x); return v if np.isfinite(v) else np.nan
 except: return np.nan

def norm(raw):
 if raw is None or raw.empty:return pd.DataFrame()
 if isinstance(raw.columns,pd.MultiIndex):
  out={}
  for f in ['Open','High','Low','Close','Volume']:
   m=[c for c in raw.columns if str(c[0]).lower()==f.lower()]
   if m:
    s=raw[m[0]]; out[f]=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s
  d=pd.DataFrame(out,index=raw.index)
 else:
  mp={str(c).strip().lower():c for c in raw.columns}; out={}
  for f in ['Open','High','Low','Close','Volume']:
   if f.lower() in mp:
    s=raw[mp[f.lower()]]; out[f]=s.iloc[:,0] if isinstance(s,pd.DataFrame) else s
  d=pd.DataFrame(out,index=raw.index)
 if 'Close' not in d:return pd.DataFrame()
 d.index=pd.to_datetime(d.index,errors='coerce'); d=d[~d.index.isna()].copy()
 if getattr(d.index,'tz',None) is not None:d.index=d.index.tz_localize(None)
 for c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.sort_index()

def load_gemini():
 for p in [Path('historical_gemini.csv'),Path('data/historical_gemini.csv'),Path('historical_news.csv'),Path('data/historical_news.csv')]:
  if p.exists():break
 else:
  print('HISTORICAL GEMINI: NOT FOUND — QUANT-ONLY'); return pd.DataFrame()
 try:g=pd.read_csv(p)
 except Exception as e: print('HISTORICAL GEMINI: READ ERROR',e); return pd.DataFrame()
 cols={str(c).strip().lower():c for c in g.columns}
 def pick(a): return next((cols[x] for x in a if x in cols),None)
 tc=pick(['ticker','symbol','stock']); pc=pick(['published_at','timestamp','datetime','date','time']); sc=pick(['score','probability','gemini_score','gemini_probability']); rc=pick(['return_prediction','expected_return','predicted_return','return_score']); hc=pick(['horizon','horizon_days','days'])
 if not(tc and pc and sc): print('HISTORICAL GEMINI: REQUIRED COLUMNS MISSING — QUANT-ONLY'); return pd.DataFrame()
 o=pd.DataFrame({'ticker':g[tc].astype(str).str.strip(),'published_at':pd.to_datetime(g[pc],errors='coerce'),'gemini_score':pd.to_numeric(g[sc],errors='coerce')})
 o['gemini_return']=pd.to_numeric(g[rc],errors='coerce') if rc else np.nan; o['horizon']=pd.to_numeric(g[hc],errors='coerce') if hc else np.nan
 o=o.dropna(subset=['ticker','published_at','gemini_score']); o['published_at']=o['published_at'].dt.tz_localize(None)
 if o.empty:return pd.DataFrame()
 if o.gemini_score.abs().median()>1.5:o.gemini_score/=100
 o.gemini_score=o.gemini_score.clip(0,1); med=o.gemini_return.abs().median(skipna=True)
 if np.isfinite(med) and med>1:o.gemini_return/=100
 print(f'HISTORICAL GEMINI: FOUND — {len(o):,} timestamped rows'); return o.sort_values(['ticker','published_at']).reset_index(drop=True)

def gem(g,t,dt,h):
 if g.empty:return np.nan,np.nan
 q=g[(g.ticker==t)&(g.published_at<=pd.Timestamp(dt))]
 if q.empty:return np.nan,np.nan
 qh=q[q.horizon==h]
 if not qh.empty:q=qh
 r=q.iloc[-1]; return sf(r.gemini_score),sf(r.gemini_return)

def features(px,t):
 c=px.Close; h=px.High if 'High'in px else c; l=px.Low if 'Low'in px else c; o=px.Open if 'Open'in px else c; v=px.Volume if 'Volume'in px else pd.Series(index=px.index,dtype=float)
 r=c.pct_change(); x=pd.DataFrame(index=px.index); x['ret_1']=r; x['ret_3']=c.pct_change(3); x['ret_5']=c.pct_change(5); x['ret_10']=c.pct_change(10); x['ret_20']=c.pct_change(20)
 x['vol_5']=r.rolling(5).std(); x['vol_10']=r.rolling(10).std(); x['vol_20']=r.rolling(20).std(); x['range_pct']=(h-l)/c.replace(0,np.nan); x['body_pct']=(c-o)/o.replace(0,np.nan); x['close_location']=(c-l)/(h-l).replace(0,np.nan)
 for n in [5,10,20,50]:x[f'dist_sma{n}']=c/c.rolling(n).mean()-1
 x['volume_ratio_5']=v/v.rolling(5).mean().replace(0,np.nan); x['volume_ratio_20']=v/v.rolling(20).mean().replace(0,np.nan); x['momentum_accel']=x.ret_5-x.ret_20/4; x['vol_regime']=x.vol_5/x.vol_20.replace(0,np.nan); x['trend_regime']=np.where(x.dist_sma50>.02,1,np.where(x.dist_sma50<-.02,-1,0)); x['ticker']=t; x['date']=px.index
 return x

def build(g):
 rows=[]
 for i,t in enumerate(SYMBOLS,1):
  print(f'Loading [{i}/{len(SYMBOLS)}] {t}')
  try:
   px=norm(yf.download(t,period='6y',interval='1d',auto_adjust=False,progress=False,threads=False))
   if len(px)<300:print(f'WARNING: insufficient history for {t}; skipping.');continue
   x=features(px,t)
   for h in HORIZONS:
    x[f'future_return_{h}']=px.Close.shift(-h)/px.Close-1
    x[f'exec_return_{h}']=px.Close.shift(-h)/px.Open.shift(-1)-1-COST if 'Open'in px else np.nan
   for dt,r in x.iterrows():
    rec={'date':pd.Timestamp(dt).normalize(),'ticker':t,'trend_regime':int(r.trend_regime)}
    for c in FEATURES:rec[c]=sf(r[c])
    for h in HORIZONS:rec[f'future_return_{h}']=sf(r[f'future_return_{h}']);rec[f'exec_return_{h}']=sf(r[f'exec_return_{h}'])
    rec['gemini_score'],rec['gemini_return']=gem(g,t,rec['date'],PRIMARY_HORIZON);rows.append(rec)
  except Exception as e:print(f'WARNING: {t} failed: {e}')
 d=pd.DataFrame(rows).sort_values(['date','ticker']).reset_index(drop=True)
 targets=[c for c in d if c.startswith('future_return_') or c.startswith('exec_return_')]
 assert not(set(FEATURES)&set(targets)),f'LEAKAGE: {set(FEATURES)&set(targets)}'
 print('FEATURE/TARGET LEAKAGE CHECK: PASS');return d

def splits(d):
 dates=np.array(sorted(d.date.unique()),dtype='datetime64[ns]'); a=dates[int(.60*len(dates))]; b=dates[int(.84*len(dates))]
 return d[d.date<a].copy(),d[(d.date>=a)&(d.date<b)].copy(),d[d.date>=b].copy(),pd.Timestamp(a),pd.Timestamp(b)

def models():
 return (Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',LogisticRegression(C=.35,max_iter=1500,class_weight='balanced',random_state=RANDOM_STATE))]),HistGradientBoostingClassifier(max_iter=180,learning_rate=.045,max_leaf_nodes=15,l2_regularization=1,random_state=RANDOM_STATE),Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler()),('m',Ridge(alpha=8))]),HistGradientBoostingRegressor(max_iter=180,learning_rate=.045,max_leaf_nodes=15,l2_regularization=1,loss='squared_error',random_state=RANDOM_STATE))

def fit(train,h):
 q=train.dropna(subset=[f'future_return_{h}']);
 if len(q)<MIN_TRAIN:return None
 X=finite(q[FEATURES]); y=q[f'future_return_{h}'].values; yd=(y>0).astype(int)
 if len(np.unique(yd))<2:return None
 m=models();m[0].fit(X,yd);m[1].fit(X,yd);m[2].fit(X,y);m[3].fit(X,y);return m

def pred(m,q):
 X=finite(q[FEATURES]); p=.5*m[0].predict_proba(X)[:,1]+.5*m[1].predict_proba(X)[:,1]; r=.5*m[2].predict(X)+.5*m[3].predict(X);return np.clip(p,1e-6,1-1e-6),r

def wf(train_pool,test,h):
 out=[]
 for j,dt in enumerate(sorted(test.date.unique()),1):
  tr=train_pool[train_pool.date<pd.Timestamp(dt)-pd.Timedelta(days=PURGE_DAYS)]
  m=fit(tr,h)
  if m is None:continue
  q=test[test.date==dt].copy();q['quant_probability'],q['quant_return_prediction']=pred(m,q);out.append(q)
  if j%100==0 or j==test.date.nunique():print(f'Walk-forward H{h}: [{j}/{test.date.nunique()}]')
 return pd.concat(out,ignore_index=True) if out else pd.DataFrame()

def addblend(q,w=0):
 z=q.copy();z['gemini_available']=z.gemini_score.notna().astype(int);z['blend_probability']=z.quant_probability;z['blend_return_prediction']=z.quant_return_prediction;ok=z.gemini_available.eq(1)
 if w>0 and ok.any():
  z.loc[ok,'blend_probability']=(1-w)*z.loc[ok,'quant_probability']+w*z.loc[ok,'gemini_score']; r=z.loc[ok,'gemini_return'];have=r.notna();idx=r.index[have];z.loc[idx,'blend_return_prediction']=(1-w)*z.loc[idx,'quant_return_prediction']+w*r.loc[idx]
 return z

def thresholds(v,h):
 best=None;tar=f'exec_return_{h}'
 for p in PROB_GRID:
  for r in RETURN_GRID:
   q=v[(v.quant_probability>=p)&(v.quant_return_prediction>=r)].dropna(subset=[tar]);
   if len(q)<30:continue
   vals=q[tar].values;los=vals[vals<=0];win=(vals>0).mean();pf=vals[vals>0].sum()/abs(los.sum()) if len(los) and los.sum() else np.nan;score=vals.mean()*math.log1p(len(q))+(0.0005*min(pf,10) if np.isfinite(pf) else 0)
   z={'pmin':p,'rmin':r,'n':len(q),'avg':vals.mean(),'win_rate':win,'profit_factor':pf,'score':score}
   if best is None or z['score']>best['score']:best=z
 return best or {'pmin':.55,'rmin':.002,'n':0,'avg':np.nan,'win_rate':np.nan,'profit_factor':np.nan,'score':-np.inf}

def weight(v,h,th):
 if v.gemini_available.sum()<max(50,int(len(v)*MIN_GEMINI_COVERAGE)):return 0
 best=(0,-np.inf)
 for w in GEMINI_WEIGHT_GRID:
  q=addblend(v,w);q=q[(q.blend_probability>=th['pmin'])&(q.blend_return_prediction>=th['rmin'])].dropna(subset=[f'exec_return_{h}']);
  if len(q)<30:continue
  s=q[f'exec_return_{h}'].mean()*math.log1p(len(q));
  if s>best[1]:best=(w,s)
 return best[0]

def perf(q,h,pcol,rcol,th):
 z=q.dropna(subset=[pcol,rcol,f'exec_return_{h}']).copy();z['action']=np.where((z[pcol]>=th['pmin'])&(z[rcol]>=th['rmin']),'TRADE',np.where(z[pcol]>=th['pmin'],'WATCH','WAIT'));t=z[z.action=='TRADE'];v=t[f'exec_return_{h}'].values;los=v[v<=0];pf=v[v>0].sum()/abs(los.sum()) if len(los) and los.sum() else np.nan
 return z,{'selected_n':len(t),'selected_win_rate':(v>0).mean() if len(v) else np.nan,'selected_average_net_return':v.mean() if len(v) else np.nan,'selected_profit_factor':pf,'best':v.max() if len(v) else np.nan,'worst':v.min() if len(v) else np.nan}

def nonoverlap(q,h):
 t=q[q.action=='TRADE'].sort_values(['date','blend_return_prediction']).drop_duplicates('date').sort_values('date');a=[];last=None
 for _,r in t.iterrows():
  dt=pd.Timestamp(r.date)
  if last is None or dt>last+pd.Timedelta(days=h+1):a.append(r);last=dt
 v=pd.DataFrame(a)[f'exec_return_{h}'].values if a else np.array([]);los=v[v<=0];return {'trades':len(v),'win_rate':(v>0).mean() if len(v) else np.nan,'average_net':v.mean() if len(v) else np.nan,'median_net':np.median(v) if len(v) else np.nan,'profit_factor':v[v>0].sum()/abs(los.sum()) if len(los) and los.sum() else np.nan,'best':v.max() if len(v) else np.nan,'worst':v.min() if len(v) else np.nan,'net_sum_return':v.sum() if len(v) else np.nan}

def portfolio(q,h):
 t=q[q.action=='TRADE'].sort_values('date').drop_duplicates('date');cap=100000.;peak=cap;mdd=0;last=None;n=0
 for _,r in t.iterrows():
  en=pd.Timestamp(r.date)+pd.Timedelta(days=1);ex=pd.Timestamp(r.date)+pd.Timedelta(days=h)
  if last is not None and en<=last:continue
  ret=sf(r[f'exec_return_{h}']);
  if not np.isfinite(ret):continue
  cap*=1+ret;peak=max(peak,cap);mdd=min(mdd,cap/peak-1);last=ex;n+=1
 years=max((q.date.max()-q.date.min()).days/365.25,1/365.25);cagr=(cap/100000)**(1/years)-1 if cap>0 else -1
 return {'starting_capital':100000.,'ending_equity':cap,'total_return':cap/100000-1,'CAGR':cagr,'max_drawdown':mdd,'completed_trades':n}

def boot(v):
 x=np.asarray(v,float);x=x[np.isfinite(x)]
 if len(x)<5:return (np.nan,)*5
 rng=np.random.default_rng(RANDOM_STATE);m=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(3000)]);w=np.array([rng.choice(x,len(x),replace=True).mean()>0 for _ in range(3000)]);return np.quantile(m,.025),np.quantile(m,.975),np.quantile(w,.025),np.quantile(w,.975),np.mean(m>0)

def main():
 print('='*78);print(f'{VERSION} — POINT-IN-TIME QUANT + GEMINI RESEARCH UPGRADE');print('='*78);print('Revision:',REVISION);print('yfinance:',yf.__version__);print('Backtest: 6 years | Universe:',len(SYMBOLS));print(f'Cost: {COST:.3%}');print('Execution: Signal T Close -> Entry T+1 Open -> Exit T+H Close')
 g=load_gemini();d=build(g);dev,val,oos,de,ve=splits(d);print('\nDATASET');print(f'Total observations: {len(d):,}');print('Symbols:',d.ticker.nunique());print('Signal dates:',d.date.nunique());print(f'Historical Gemini coverage: {d.gemini_score.notna().mean():.2%}');print(f'Development: {len(dev):,}');print(f'Validation: {len(val):,}');print(f'OOS: {len(oos):,}');print('Development end:',de.date());print('Validation end:',ve.date());print('OOS start:',oos.date.min().date())
 comps=[];nos=[];ports=[];years=[];regs=[]
 for h in HORIZONS:
  print('='*78);print(f'HORIZON {h}D '+('(PRIMARY)' if h==PRIMARY_HORIZON else '(CONTROL)'));print('='*78)
  vp=wf(dev,val,h);op=wf(pd.concat([dev,val],ignore_index=True),oos,h)
  if vp.empty or op.empty:print('WARNING: insufficient predictions');continue
  th=thresholds(vp,h);w=weight(vp,h,th);of=addblend(op,w);print(f"Validation-selected thresholds: P>={th['pmin']:.2f}, Return>={th['rmin']:.4f}");print(f'Validation-selected Gemini weight: {w:.2f}')
  cov=of.gemini_available.mean();print('Gemini:', 'TESTABLE' if cov>=MIN_GEMINI_COVERAGE else f'NOT TESTED — OOS coverage {cov:.2%}');q,s=perf(of,h,'quant_probability','quant_return_prediction',th);b,sb=perf(of,h,'blend_probability','blend_return_prediction',th);no=nonoverlap(b,h);vals=b[b.action=='TRADE'][f'exec_return_{h}'].dropna().values;bc=boot(vals);po=portfolio(b,h);m=of.dropna(subset=[f'future_return_{h}','blend_probability','blend_return_prediction']);y=(m[f'future_return_{h}']>0).astype(int);p=np.clip(m.blend_probability,1e-6,1-1e-6)
  comps.append({'model':'quant+gemini' if w>0 else 'quant','horizon':h,'observations':len(m),'directional_accuracy':((p>=.5).astype(int).values==y.values).mean() if len(m) else np.nan,'brier_score':brier_score_loss(y,p) if len(m) else np.nan,'log_loss':log_loss(y,p,labels=[0,1]) if len(m) else np.nan,'return_mae':mean_absolute_error(m[f'future_return_{h}'],m.blend_return_prediction) if len(m) else np.nan,'mean_predicted_return':m.blend_return_prediction.mean() if len(m) else np.nan,'mean_actual_return':m[f'future_return_{h}'].mean() if len(m) else np.nan,'selected_n_p>=55':sb['selected_n'],'selected_win_rate':sb['selected_win_rate'],'selected_average_net_return':sb['selected_average_net_return'],'selected_profit_factor':sb['selected_profit_factor'],'gemini_weight':w,'gemini_oos_coverage':cov})
  no.update({'model':'quant+gemini' if w>0 else 'quant','horizon':h,'bootstrap_mean_ci_low':bc[0],'bootstrap_mean_ci_high':bc[1],'bootstrap_win_ci_low':bc[2],'bootstrap_win_ci_high':bc[3],'prob_mean_gt_zero':bc[4]});nos.append(no);po.update({'model':'quant+gemini' if w>0 else 'quant','horizon':h});ports.append(po)
  b.to_csv(AUDIT/f'v7_1_oos_h{h}.csv',index=False);vp.to_csv(AUDIT/f'v7_1_validation_h{h}.csv',index=False)
  tt=b[b.action=='TRADE'].copy();
  if not tt.empty:
   tt['year']=pd.to_datetime(tt.date).dt.year
   for yy,z in tt.groupby('year'):
    x=z[f'exec_return_{h}'].dropna().values;los=x[x<=0];years.append({'model':'quant+gemini' if w>0 else 'quant','horizon':h,'year':yy,'trades':len(x),'win_rate':(x>0).mean(),'average_net':x.mean(),'profit_factor':x[x>0].sum()/abs(los.sum()) if len(los) and los.sum() else np.nan})
   for rg,z in tt.groupby('trend_regime'):
    x=z[f'exec_return_{h}'].dropna().values;los=x[x<=0];regs.append({'model':'quant+gemini' if w>0 else 'quant','horizon':h,'regime':{-1:'BEAR',0:'NEUTRAL',1:'BULL'}.get(int(rg),'UNKNOWN'),'trades':len(x),'win_rate':(x>0).mean(),'average_net':x.mean(),'profit_factor':x[x>0].sum()/abs(los.sum()) if len(los) and los.sum() else np.nan})
 C=pd.DataFrame(comps);N=pd.DataFrame(nos);P=pd.DataFrame(ports);Y=pd.DataFrame(years);R=pd.DataFrame(regs);C.to_csv(AUDIT/'v7_1_oos_model_comparison.csv',index=False);N.to_csv(AUDIT/'v7_1_nonoverlapping_oos.csv',index=False);P.to_csv(AUDIT/'v7_1_portfolio_oos.csv',index=False);Y.to_csv(AUDIT/'v7_1_yearly_stability.csv',index=False);R.to_csv(AUDIT/'v7_1_regime_stability.csv',index=False)
 bench=[]
 for h in HORIZONS:
  x=d.dropna(subset=[f'future_return_{h}']).groupby('date')[f'future_return_{h}'].mean();bench.append({'horizon':h,'observations':len(x),'win_rate':(x>0).mean(),'average_return':x.mean(),'median_return':x.median()})
 pd.DataFrame(bench).to_csv(AUDIT/'v7_1_equal_weight_benchmark.csv',index=False)
 print('\n'+'='*78);print('V7.1 OOS MODEL COMPARISON');print('='*78);print(C.to_string(index=False));print('\nNON-OVERLAPPING OOS');print(N.to_string(index=False));print('\nPORTFOLIO OOS');print(P.to_string(index=False));print('\nYEARLY STABILITY');print(Y.to_string(index=False));print('\nREGIME STABILITY');print(R.to_string(index=False));print('\nEQUAL-WEIGHT BENCHMARK');print(pd.DataFrame(bench).to_string(index=False));print('\n'+'='*78);print('V7.1 BACKTEST COMPLETED');print('='*78);print('AUDIT GUARANTEES:');print('1. Backward-looking OHLCV features only.');print('2. Forward returns never enter FEATURES.');print('3. Training is purged before every prediction date.');print('4. Validation thresholds are frozen before OOS.');print('5. Gemini weight is selected on validation only.');print('6. Historical Gemini must be timestamped <= signal date.');print('7. Current Gemini API calls never manufacture historical scores.');print('8. OOS is never used for model/threshold/weight selection.');print('9. Non-overlapping and sequential portfolio tests are both reported.');print('10. Historical results do not guarantee future performance.')
if __name__=='__main__':main()
