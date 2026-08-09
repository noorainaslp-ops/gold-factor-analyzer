"""
Indian Market Daily Alert Engine (Gold + Momentum Stock)
-------------------------------------------------------
1. Calculates Gold ETF (GOLDBEES) and USD/INR signals.
2. Evaluates a liquid NSE Nifty stock universe using a short-term momentum model
   (Relative Strength + 20 EMA Breakout) to pick 1 high-probability short-term stock.
3. Dispatches a unified push notification to Telegram.
"""

import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

class IndianMarketSignalEngine:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        
        # Gold & Macro Tickers
        self.gold_tickers = {
            "Gold_ETF": "GOLDBEES.NS",
            "USD_INR": "USDINR=X"
        }
        
        # Liquid Nifty Basket for Short-Term Momentum Screening
        self.stock_universe = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS", "SBIN.NS",
            "BHARTIARTL.NS", "LTIM.NS", "TATAMOTORS.NS", "HAL.NS", "TITAN.NS"
        ]

    def get_gold_signal(self) -> dict:
        """Fetches daily Gold ETF and USD/INR movements."""
        data = yf.download(tickers=list(self.gold_tickers.values()), period="10d", interval="1d", progress=False)['Close']
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        gold_series = data[self.gold_tickers["Gold_ETF"]].dropna()
        usdinr_series = data[self.gold_tickers["USD_INR"]].dropna()

        gold_price = gold_series.iloc[-1]
        usdinr_price = usdinr_series.iloc[-1]
        
        gold_ret = gold_series.pct_change().iloc[-1] * 100
        usdinr_ret = usdinr_series.pct_change().iloc[-1] * 100

        if gold_ret > 0 and usdinr_ret > 0:
            signal = "🟢 STRONG BUY (Global Gold up & Weak INR)"
        elif gold_ret < 0 and usdinr_ret < 0:
            signal = "🔴 STRONG SELL / CORRECTION (Global Gold down & Strong INR)"
        elif gold_ret > 0 and usdinr_ret < 0:
            signal = "🟡 MODERATE BUY (Gold strength offset by stronger INR)"
        else:
            signal = "⚪ NEUTRAL / CONSOLIDATION"

        return {
            "Gold_Price": round(gold_price, 2),
            "Gold_Change": round(gold_ret, 2),
            "USD_INR": round(usdinr_price, 2),
            "Signal": signal
        }

    def get_top_momentum_stock(self) -> dict:
        """
        Screens the stock universe using 20-day EMA and 5-day price momentum
        to identify the stock with highest short-term upward momentum.
        """
        # Fetch 3 months buffer to account for weekends and holidays
        data = yf.download(tickers=self.stock_universe, period="3m", interval="1d", progress=False)['Close']
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        results = []
        for ticker in self.stock_universe:
            if ticker not in data.columns:
                continue
                
            series = data[ticker].dropna()
            if len(series) < 20:
                continue
            
            latest_price = series.iloc[-1]
            ema_20 = series.ewm(span=20, adjust=False).mean().iloc[-1]
            
            # 5 trading days momentum calculation
            return_5d = ((latest_price - series.iloc[-5]) / series.iloc[-5]) * 100
            
            # Percentage distance from 20-day EMA
            ema_diff = ((latest_price - ema_20) / ema_20) * 100
            
            # Momentum Score
            momentum_score = (return_5d * 0.6) + (ema_diff * 0.4)
            
            results.append({
                "Ticker": ticker.replace(".NS", ""),
                "Price": round(latest_price, 2),
                "5D_Return": round(return_5d, 2),
                "EMA_20": round(ema_20, 2),
                "Score": momentum_score
            })

        if not results:
            return {
                "Symbol": "N/A",
                "Price": 0.0,
                "Return_5D": 0.0,
                "EMA_20": 0.0
            }

        df_results = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        best_pick = df_results.iloc[0]
        
        return {
            "Symbol": best_pick["Ticker"],
            "Price": best_pick["Price"],
            "Return_5D": best_pick["5D_Return"],
            "EMA_20": best_pick["EMA_20"]
        }

    def send_combined_push(self, gold_info: dict, stock_info: dict):
        """Sends unified Gold + Short-Term Stock Alert via Telegram."""
        if not self.bot_token or not self.chat_id:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in GitHub Secrets!")

        now_str = datetime.now().strftime('%d %b %Y, 10:00 AM')

        message = (
            f"🇮🇳 *DAILY MARKET ALERT (10:00 AM)*\n"
            f"📅 _{now_str}_\n\n"
            f"--- 🥇 *GOLD & MACRO SIGNAL* ---\n"
            f"💰 *GOLDBEES (NSE):* ₹{gold_info['Gold_Price']} ({gold_info['Gold_Change']}%)\n"
            f"💵 *USD/INR:* ₹{gold_info['USD_INR']}\n"
            f"🎯 *Signal:* {gold_info['Signal']}\n\n"
            f"--- 🚀 *TOP SHORT-TERM STOCK PICK* ---\n"
            f"📊 *Stock:* `{stock_info['Symbol']}`\n"
            f"💵 *Price:* ₹{stock_info['Price']}\n"
            f"📈 *5-Day Momentum:* +{stock_info['Return_5D']}%\n"
            f"🛡️ *20-EMA Support:* ₹{stock_info['EMA_20']}\n"
            f"⚡ *Rationale:* High short-term probability breakout above 20 EMA."
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
        
        res = requests.post(url, json=payload, timeout=10)
        res.raise_for_status()
        print("[Success] Combined alert delivered to mobile!")

if __name__ == "__main__":
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    engine = IndianMarketSignalEngine(bot_token=BOT_TOKEN, chat_id=CHAT_ID)
    gold_data = engine.get_gold_signal()
    stock_data = engine.get_top_momentum_stock()
    engine.send_combined_push(gold_data, stock_data)
