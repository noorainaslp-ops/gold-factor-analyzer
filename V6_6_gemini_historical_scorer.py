#!/usr/bin/env python3
from __future__ import annotations
import json, os, time
from pathlib import Path
import pandas as pd
import requests

INPUT = Path(os.getenv('GEMINI_RAW_NEWS_FILE','data/historical_news_raw.csv'))
OUTPUT = Path(os.getenv('GEMINI_OUTPUT_FILE','data/historical_gemini.csv'))
API_KEY = os.getenv('GEMINI_API_KEY','')
MODEL = os.getenv('GEMINI_MODEL','gemini-2.5-flash')
SLEEP = float(os.getenv('GEMINI_SLEEP_SECONDS','0.35'))
MAX_ROWS = int(os.getenv('GEMINI_MAX_ROWS','0'))

PROMPT = '''You are scoring one historical financial-news event for a strict point-in-time research dataset.\n\nUse ONLY the supplied historical article text.\nDo NOT browse, search, use future prices, future returns, later news, current market data, or hindsight.\n\nReturn JSON only:\n{\n  "score": number,\n  "confidence": number,\n  "materiality": number,\n  "expected_horizon": "intraday|1-3d|1-2w|1-3m|longer|unclear",\n  "event_type": "earnings|guidance|order|contract|regulation|management|litigation|capital|product|macro|sector|other",\n  "reason": "brief explanation"\n}\n\nscore is in [-1,+1]: negative to positive information.\nconfidence and materiality are in [0,1].\nJudge only what the article communicated at publication time.'''

def load_raw():
    if not INPUT.exists(): raise FileNotFoundError(f'Missing {INPUT}')
    d = pd.read_csv(INPUT)
    need={'ticker','published_at','title'}
    miss=need-set(d.columns)
    if miss: raise ValueError(f'Missing columns: {sorted(miss)}')
    for c in ['summary','body']:
        if c not in d: d[c]=''
    for c in ['ticker','title','summary','body']:
        d[c]=d[c].fillna('').astype(str)
    d['published_at']=pd.to_datetime(d['published_at'],errors='coerce',utc=True).dt.tz_convert(None)
    d=d.dropna(subset=['published_at']).sort_values(['published_at','ticker'])
    if MAX_ROWS>0: d=d.head(MAX_ROWS)
    return d.reset_index(drop=True)

def score_article(row):
    if not API_KEY: raise RuntimeError('GEMINI_API_KEY is not configured')
    text=(f"Ticker: {row['ticker']}\nPublished at: {row['published_at']}\n\n"
          f"TITLE:\n{row['title']}\n\nSUMMARY:\n{row['summary']}\n\nBODY:\n{row['body'][:12000]}")
    payload={'system_instruction':{'parts':[{'text':PROMPT}]},
             'contents':[{'role':'user','parts':[{'text':text}]}],
             'generationConfig':{'temperature':0,'responseMimeType':'application/json'}}
    url=(f'https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}')
    r=requests.post(url,json=payload,timeout=60); r.raise_for_status()
    data=r.json(); raw=data['candidates'][0]['content']['parts'][0]['text']
    obj=json.loads(raw)
    score=float(obj.get('score',0)); conf=float(obj.get('confidence',0)); mat=float(obj.get('materiality',0))
    if not -1<=score<=1: raise ValueError('score outside [-1,1]')
    if not 0<=conf<=1 or not 0<=mat<=1: raise ValueError('confidence/materiality outside [0,1]')
    return {'gemini_score':score,'gemini_confidence':conf,'gemini_materiality':mat,
            'expected_horizon':str(obj.get('expected_horizon','unclear')),
            'event_type':str(obj.get('event_type','other')),
            'gemini_reason':str(obj.get('reason',''))[:500]}

def main():
    print('V6.6 HISTORICAL GEMINI SCORER')
    print(f'Input: {INPUT}')
    print(f'Output: {OUTPUT}')
    print(f'Model: {MODEL}')
    raw=load_raw(); records=[]; done=set()
    if OUTPUT.exists():
        old=pd.read_csv(OUTPUT)
        if not old.empty:
            records=old.to_dict('records')
            done={(str(r.get('ticker','')),str(r.get('published_at','')),str(r.get('title',''))) for r in records}
            print(f'Existing scored rows: {len(records):,}')
    for i,row in enumerate(raw.to_dict('records'),1):
        key=(str(row['ticker']),str(row['published_at']),str(row['title']))
        if key in done: continue
        try:
            result=score_article(row)
            records.append({'ticker':row['ticker'],'published_at':row['published_at'],'title':row['title'],'summary':row['summary'],**result})
            done.add(key)
            OUTPUT.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(records).to_csv(OUTPUT,index=False)
            print(f'Scored {i}/{len(raw)}: {row["ticker"]} {row["published_at"]}')
            time.sleep(SLEEP)
        except Exception as exc:
            print(f'WARNING: failed row {i}: {exc}')
            time.sleep(2)
    out=pd.DataFrame(records)
    if not out.empty: out=out.sort_values(['published_at','ticker'])
    OUTPUT.parent.mkdir(parents=True,exist_ok=True); out.to_csv(OUTPUT,index=False)
    print(f'Completed. Scored rows: {len(out):,}')

if __name__=='__main__': main()
