"""
Indian Gold Market Signal Engine with Mobile Push Notifications
---------------------------------------------------------------
Fetches Indian Gold ETF (GOLDBEES), USD/INR rates, and MCX price proxies.
Dispatches actionable buy/sell signals directly to your smartphone via Telegram.
"""

import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

class IndianGoldMobileNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.tickers = {
            "Gold_ETF_India": "GOLDBEES.NS",
            "USD_INR": "USDINR=X",
            "Intl_Gold": "GC=F"
        }

    def fetch_market_signals(self) -> dict:
        """Fetches daily market movements for Indian Gold and USD/INR."""
        data = yf.download(tickers=list(self.tickers.values()), period="5d", interval="1d", progress=False)['Close']
        
        # Handle yfinance MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        latest_gold_etf = data[self.tickers["Gold_ETF_India"]].dropna().iloc[-1]
        latest_usdinr = data[self.tickers["USD_INR"]].dropna().iloc[-1]
        
        gold_ret = data[self.tickers["Gold_ETF_India"]].dropna().pct_change().iloc[-1] * 100
        usdinr_ret = data[self.tickers["USD_INR"]].dropna().pct_change().iloc[-1] * 100
        
        # Rule-based Signal Matrix
        if gold_ret > 0 and usdinr_ret > 0:
            signal = "🟢 STRONG BUY (Global Gold up & Rupee weakening)"
        elif gold_ret < 0 and usdinr_ret < 0:
            signal = "🔴 STRONG SELL / CORRECTION (Global Gold down & Rupee strengthening)"
        elif gold_ret > 0 and usdinr_ret < 0:
            signal = "🟡 MODERATE BUY (Global gold strength offset by stronger INR)"
        else:
            signal = "⚪ NEUTRAL / CONSOLIDATION"

        return {
            "Gold_ETF": round(latest_gold_etf, 2),
            "USD_INR": round(latest_usdinr, 2),
            "Gold_Change": round(gold_ret, 2),
            "Signal": signal,
            "Timestamp": datetime.now().strftime('%d %b %Y, %I:%M %p')
        }

    def send_telegram_push(self, metrics: dict):
        """Dispatches formatted message directly to your phone."""
        if not self.bot_token or not self.chat_id:
            print("[Error] Missing Telegram credentials.")
            return

        message = (
            f"🇮🇳 *INDIAN GOLD MARKET SIGNAL*\n"
            f"📅 _{metrics['Timestamp']}_\n\n"
            f"💰 *GOLDBEES (NSE):* ₹{metrics['Gold_ETF']} ({metrics['Gold_Change']}%)\n"
            f"💵 *USD/INR:* ₹{metrics['USD_INR']}\n\n"
            f"🎯 *Signal:* {metrics['Signal']}"
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
        
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            print("[Success] Alert pushed to smartphone!")
        else:
            print(f"[Failed] HTTP {res.status_code}: {res.text}")

if __name__ == "__main__":
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    notifier = IndianGoldMobileNotifier(bot_token=BOT_TOKEN, chat_id=CHAT_ID)
    metrics = notifier.fetch_market_signals()
    notifier.send_telegram_push(metrics)
