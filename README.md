# Market Alert V6.3.2

## Files

- `market_engine_v6_3_2.py` — main market/Telegram engine
- `requirements.txt` — Python dependencies
- `.github/workflows/market_alert_v6_3_2.yml` — GitHub Actions workflow

## GitHub Secrets

Required:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Optional:

- `ALERT_CAPITAL` (default 100000)
- `ALERT_RISK_PCT` (default 0.01)
- `ALERT_MAX_POSITION_PCT` (default 0.20)

## Important

The model produces research estimates. It does not guarantee profit.

V6.3.2 adds:

- TRADE / WATCH / REJECT classification
- empirical historical-neighbour probabilities
- multi-horizon 1/3/5-session expected returns
- volatility-aware stop loss and targets
- risk-based position sizing
- weekend/non-trading-day status
- explicit IPO retrieval failure
- audit CSV/TXT output
- syntax validation before execution

The NSE IPO endpoint can block automated GitHub runners. When that happens the alert explicitly says `IPO DATA UNAVAILABLE | RETRIEVAL FAILED` rather than falsely reporting that there are no IPOs.
