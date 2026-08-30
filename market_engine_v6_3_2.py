import os
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
NIFTY = "^NSEI"
VIX = "^INDIAVIX"

CAPITAL = float(os.getenv("ALERT_CAPITAL", "100000"))
RISK_PCT = float(os.getenv("ALERT_RISK_PCT", "0.01"))
MAX_POSITION_PCT = float(os.getenv("ALERT_MAX_POSITION_PCT", "0.20"))
MAX_STOCKS = int(os.getenv("ALERT_MAX_STOCKS", "70"))

SYMBOLS = [
    "RELIANCE","HDFCBANK","ICICIBANK","BHARTIARTL","TCS","INFY","ITC",
    "SBIN","LT","AXISBANK","KOTAKBANK","M&M","HINDUNILVR","BAJFINANCE",
    "MARUTI","SUNPHARMA","HCLTECH","NTPC","TITAN","ADANIENT","ULTRACEMCO",
    "ONGC","POWERGRID","TATASTEEL","JSWSTEEL","COALINDIA","ADANIPORTS",
    "BAJAJFINSV","WIPRO","NESTLEIND","TECHM","ASIANPAINT","BEL","TRENT",
    "GRASIM","HINDALCO","EICHERMOT","SHRIRAMFIN","VEDL","TATAMOTORS",
    "CIPLA","DRREDDY","DIVISLAB","APOLLOHOSP","BRITANNIA","HEROMOTOCO",
    "BAJAJ-AUTO","INDUSINDBK","SBILIFE","HDFCLIFE","TATAELXSI",
    "AUROPHARMA","BOSCHLTD","DLF","NAUKRI","SAIL","ABB","GODREJCP",
    "PIDILITIND","SIEMENS","AMBUJACEM","ACC","BPCL","IOC","HAVELLS",
]

TICKERS = [s + ".NS" for s in SYMBOLS]
AUDIT_DIR = Path("audit")
AUDIT_DIR.mkdir(exist_ok=True)


def now_ist():
    return datetime.now(IST)


def finite(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except Exception:
        return np.nan


def money(x):
    x = finite(x)
    return "—" if not np.isfinite(x) else f"₹{x:,.2f}"


def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df, n=14):
    prev = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev).abs(),
            (df["Low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def split_yf_download(raw):
    if raw is None or raw.empty:
        return {}
    if not isinstance(raw.columns, pd.MultiIndex):
        return {"SINGLE": raw}
    level0 = list(raw.columns.get_level_values(0).unique())
    level1 = list(raw.columns.get_level_values(1).unique())

    if "Close" in level0:
        return {
            t: raw.xs(t, axis=1, level=1, drop_level=True)
            for t in level1
        }
    return {
        t: raw.xs(t, axis=1, level=0, drop_level=True)
        for t in level0
    }


def download_history():
    raw = yf.download(
        TICKERS + [NIFTY, VIX],
        period="2y",
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
        threads=True,
    )
    return split_yf_download(raw)


def get_frame(frames, ticker):
    df = frames.get(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return None

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    out = df[needed].copy()
    out = out.dropna(subset=["Close"])
    return out


def make_features(df):
    x = df.copy()
    close = x["Close"]

    x["rsi"] = rsi(close)
    x["sma20"] = close.rolling(20).mean()
    x["sma50"] = close.rolling(50).mean()
    x["sma200"] = close.rolling(200).mean()
    x["ema10"] = close.ewm(span=10, adjust=False).mean()

    x["ret1"] = close.pct_change(1)
    x["ret3"] = close.pct_change(3)
    x["ret5"] = close.pct_change(5)
    x["ret20"] = close.pct_change(20)

    x["vol20"] = close.pct_change().rolling(20).std() * math.sqrt(252)

    vol_mean = x["Volume"].rolling(20).mean()
    x["vol_ratio"] = x["Volume"] / vol_mean.replace(0, np.nan)

    x["atr"] = atr(x)
    x["atr_pct"] = x["atr"] / close
    x["dist_sma50"] = close / x["sma50"] - 1
    x["dist_ema10"] = close / x["ema10"] - 1

    for h in (1, 3, 5):
        x[f"fwd{h}"] = close.shift(-h) / close - 1

    return x


def empirical_analog_model(x):
    """
    Historical-neighbour model.

    It compares the current technical state with earlier states in the
    same stock and estimates forward return/probability from the nearest
    historical observations. The final alert treats these as research
    estimates, not guaranteed probabilities.
    """
    cols = [
        "rsi", "ret3", "ret5", "dist_sma50",
        "dist_ema10", "vol_ratio", "atr_pct"
    ]

    if len(x) < 120:
        return None

    # Never use the last five rows as historical neighbours because their
    # forward outcomes can overlap the current decision point.
    hist = x.iloc[:-5].dropna(
        subset=cols + ["fwd1", "fwd3", "fwd5"]
    ).copy()

    if len(hist) < 80:
        return None

    z_hist = []
    q = []

    for col in cols:
        med = hist[col].median()
        mad = (hist[col] - med).abs().median()
        scale = 1.4826 * mad

        if not np.isfinite(scale) or scale < 1e-8:
            scale = hist[col].std(ddof=0)

        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0

        z_hist.append(((hist[col] - med) / scale).to_numpy())
        q.append((x[col].iloc[-1] - med) / scale)

    z = np.column_stack(z_hist)
    q = np.asarray(q, dtype=float)

    distances = np.sqrt(((z - q) ** 2).mean(axis=1))

    k = min(40, max(20, len(hist) // 20))
    idx = np.argsort(distances)[:k]

    neighbours = hist.iloc[idx]
    weights = 1 / (distances[idx] + 0.35)
    weights = weights / weights.sum()

    result = {}

    for h in (1, 3, 5):
        returns = neighbours[f"fwd{h}"].to_numpy(dtype=float)

        # Weighted probability and expected return.
        result[f"p{h}"] = float(np.sum(weights * (returns > 0)))
        result[f"er{h}"] = float(np.sum(weights * returns))
        result[f"median{h}"] = float(np.median(returns))

    result["analogs"] = int(k)
    result["quality"] = float(
        np.clip(1 - np.mean(distances[idx]) / 4, 0, 1)
    )

    return result


def market_regime(frames):
    nifty_df = get_frame(frames, NIFTY)
    vix_df = get_frame(frames, VIX)

    if nifty_df is None:
        return {
            "label": "UNKNOWN",
            "nifty": np.nan,
            "sma50": np.nan,
            "vix": np.nan,
        }

    nf = make_features(nifty_df)

    nifty = finite(nf["Close"].iloc[-1])
    sma50 = finite(nf["sma50"].iloc[-1])
    sma50_prev = (
        finite(nf["sma50"].iloc[-6])
        if len(nf) >= 6
        else np.nan
    )
    vix = (
        finite(vix_df["Close"].iloc[-1])
        if vix_df is not None
        else np.nan
    )

    above = (
        np.isfinite(nifty)
        and np.isfinite(sma50)
        and nifty > sma50
    )
    rising = (
        np.isfinite(sma50)
        and np.isfinite(sma50_prev)
        and sma50 > sma50_prev
    )

    if above and rising:
        label = "FAVORABLE"
    elif (not above) and (not rising):
        label = "UNFAVORABLE"
    else:
        label = "MIXED"

    return {
        "label": label,
        "nifty": nifty,
        "sma50": sma50,
        "vix": vix,
    }


def score_candidate(row, regime_label):
    score = (
        0.30 * (row["p3"] - 0.50)
        + 0.25 * (row["p5"] - 0.50)
        + 0.20 * np.tanh(row["er3"] / 0.01)
        + 0.15 * np.tanh(row["er5"] / 0.02)
        + 0.10 * np.tanh(row["trend"] / 0.03)
    )

    if regime_label == "FAVORABLE":
        score += 0.02
    elif regime_label == "UNFAVORABLE":
        score -= 0.04

    if row["rsi"] > 75:
        score -= 0.08

    if row["vol_ratio"] < 0.60:
        score -= 0.04

    return float(score)


def build_candidates(frames, regime):
    rows = []

    for symbol in SYMBOLS[:MAX_STOCKS]:
        df = get_frame(frames, symbol + ".NS")

        if df is None or len(df) < 230:
            continue

        x = make_features(df).dropna(
            subset=[
                "rsi", "sma50", "ema10",
                "atr_pct", "ret3", "ret5"
            ]
        )

        if len(x) < 220:
            continue

        model = empirical_analog_model(x)
        if model is None:
            continue

        last = x.iloc[-1]

        price = finite(last["Close"])
        atr_pct = finite(last["atr_pct"])

        if (
            not np.isfinite(price)
            or not np.isfinite(atr_pct)
            or atr_pct <= 0
        ):
            continue

        row = {
            "symbol": symbol,
            "price": price,
            "rsi": finite(last["rsi"]),
            "vol_ratio": finite(last["vol_ratio"]),
            "atr_pct": atr_pct,
            "trend": finite(last["dist_sma50"],),
            **model,
        }

        row["score"] = score_candidate(
            row, regime["label"]
        )

        # Volatility-aware risk levels.
        stop_distance = max(
            0.012,
            min(0.05, 1.5 * atr_pct)
        )
        stop = price * (1 - stop_distance)

        # Targets use positive expected/median forward returns,
        # with minimums only to avoid a zero target.
        expected3 = max(row["er3"], row["median3"], 0.006)
        expected5 = max(row["er5"], row["median5"], 0.012)

        target1 = price * (
            1 + max(0.006, min(0.06, expected3))
        )
        target2 = price * (
            1 + max(0.012, min(0.10, expected5))
        )

        if target2 <= target1:
            target2 = target1 * 1.006

        risk_per_share = price - stop

        rr1 = (
            (target1 - price) / risk_per_share
            if risk_per_share > 0
            else np.nan
        )
        rr2 = (
            (target2 - price) / risk_per_share
            if risk_per_share > 0
            else np.nan
        )

        row.update(
            {
                "stop": stop,
                "target1": target1,
                "target2": target2,
                "rr1": rr1,
                "rr2": rr2,
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def classify_candidates(df, regime):
    if df.empty:
        return df

    d = df.copy()

    # V6.3.2: hard filters are reserved for actual TRADE labels.
    d["valid_return"] = (
        (d["er3"] > 0.002)
        & (d["er5"] > 0.004)
    )

    d["valid_probability"] = (
        (d["p3"] >= 0.54)
        & (d["p5"] >= 0.53)
    )

    d["valid_rr"] = (
        (d["rr1"] >= 1.0)
        & (d["rr2"] >= 1.5)
    )

    d["valid_quality"] = d["quality"] >= 0.35
    d["valid_trend"] = d["trend"] > -0.02
    d["not_extreme"] = d["rsi"] < 76

    d["trade"] = (
        d["valid_return"]
        & d["valid_probability"]
        & d["valid_rr"]
        & d["valid_quality"]
        & d["valid_trend"]
        & d["not_extreme"]
    )

    # In an unfavorable broad market, no long trade is issued.
    if regime["label"] == "UNFAVORABLE":
        d["trade"] = False

    # Broader watch tier: useful when the stock is interesting but
    # not yet strong enough to justify a trade.
    d["watch"] = (
        ~d["trade"]
        & (d["er3"] > 0)
        & (d["p3"] >= 0.51)
        & (d["p5"] >= 0.50)
        & (d["quality"] >= 0.25)
        & (d["rr2"] >= 1.0)
    )

    d["action"] = np.where(
        d["trade"],
        "TRADE",
        np.where(d["watch"], "WATCH", "REJECT"),
    )

    return d.sort_values(
        ["trade", "watch", "score", "er3"],
        ascending=[False, False, False, False],
    )


def position_size(row):
    risk_budget = CAPITAL * RISK_PCT

    risk_per_share = max(
        row["price"] - row["stop"],
        row["price"] * 0.005,
    )

    shares_by_risk = math.floor(
        risk_budget / risk_per_share
    )

    shares_by_capital = math.floor(
        (CAPITAL * MAX_POSITION_PCT)
        / row["price"]
    )

    shares = max(
        0,
        min(shares_by_risk, shares_by_capital),
    )

    value = shares * row["price"]
    max_loss = shares * risk_per_share

    return shares, value, max_loss


def fetch_ipo_data():
    """
    Try NSE's public JSON endpoints.

    If NSE blocks the GitHub Actions runner, return an explicit
    retrieval failure. Never convert a failed request into
    'no IPOs', because those are not equivalent.
    """
    urls = [
        "https://www.nseindia.com/api/all-upcoming-issues",
        "https://www.nseindia.com/api/ipo-current-issue",
    ]

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nseindia.com/",
    }

    session = requests.Session()

    try:
        session.get(
            "https://www.nseindia.com/",
            headers=headers,
            timeout=12,
        )

        records = []

        for url in urls:
            response = session.get(
                url,
                headers=headers,
                timeout=12,
            )

            if not response.ok:
                continue

            payload = response.json()

            if isinstance(payload, list):
                records.extend(payload)

            elif isinstance(payload, dict):
                for key in (
                    "data",
                    "records",
                    "items",
                ):
                    value = payload.get(key)
                    if isinstance(value, list):
                        records.extend(value)

        # Remove obvious duplicates.
        unique = []
        seen = set()

        for record in records:
            if not isinstance(record, dict):
                continue

            key = json_key = str(sorted(record.items()))

            if key not in seen:
                seen.add(key)
                unique.append(record)

        if not unique:
            return [], "NSE returned no IPO records."

        return unique[:20], None

    except Exception as exc:
        return [], str(exc)


def alert_text(regime, df, ipo_records, ipo_error):
    now = now_ist()
    market_day = now.weekday() < 5

    lines = [
        "MULTI-FACTOR MARKET ALERT V6.3.2",
        now.strftime("%d %b %Y, %H:%M IST"),
        "",
        (
            "MARKET STATUS: TRADING DAY"
            if market_day
            else "MARKET STATUS: WEEKEND / NON-TRADING DAY"
        ),
        f"MARKET REGIME: {regime['label']}",
        (
            f"NIFTY: {money(regime['nifty'])} | "
            f"SMA50: {money(regime['sma50'])}"
        ),
        f"INDIA VIX: {money(regime['vix'])}",
        "",
        "--- TOP SHORT-TERM TRADE SETUPS (1–5 SESSIONS) ---",
    ]

    trades = (
        df[df["action"] == "TRADE"].head(3)
        if not df.empty
        else pd.DataFrame()
    )

    watches = (
        df[df["action"] == "WATCH"].head(5)
        if not df.empty
        else pd.DataFrame()
    )

    if trades.empty:
        lines.append("NO VALID LONG TRADE TODAY")
    else:
        for i, (_, row) in enumerate(
            trades.iterrows(), 1
        ):
            shares, value, max_loss = position_size(row)

            lines.extend(
                [
                    "",
                    f"{i}. {row['symbol']} — TRADE",
                    f"Price: {money(row['price'])}",
                    (
                        "P(UP) 1D / 3D / 5D: "
                        f"{row['p1']*100:.1f}% / "
                        f"{row['p3']*100:.1f}% / "
                        f"{row['p5']*100:.1f}%"
                    ),
                    (
                        "Expected return 1D / 3D / 5D: "
                        f"{row['er1']*100:.2f}% / "
                        f"{row['er3']*100:.2f}% / "
                        f"{row['er5']*100:.2f}%"
                    ),
                    (
                        f"Score: {row['score']:.3f} | "
                        f"RSI: {row['rsi']:.1f} | "
                        f"Volume: {row['vol_ratio']:.2f}x"
                    ),
                    f"Entry: {money(row['price'])}",
                    f"Stop Loss: {money(row['stop'])}",
                    (
                        f"Target 1: {money(row['target1'])} | "
                        f"Target 2: {money(row['target2'])}"
                    ),
                    (
                        f"Risk/Reward: {row['rr1']:.2f} / "
                        f"{row['rr2']:.2f}"
                    ),
                    "Expected holding: 1–5 sessions",
                    (
                        f"Suggested position: {shares} shares "
                        f"≈ {money(value)}"
                    ),
                    (
                        f"Maximum planned loss: "
                        f"{money(max_loss)}"
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "--- BEST WATCHLIST SETUPS ---",
        ]
    )

    if watches.empty:
        lines.append("None.")
    else:
        for i, (_, row) in enumerate(
            watches.iterrows(), 1
        ):
            lines.append(
                f"{i}. {row['symbol']} | "
                f"Price {money(row['price'])} | "
                f"P3 {row['p3']*100:.1f}% | "
                f"E3 {row['er3']*100:.2f}% | "
                f"RR2 {row['rr2']:.2f} | "
                f"RSI {row['rsi']:.1f}"
            )

    lines.extend(
        [
            "",
            "--- IPO OPEN / UPCOMING ---",
        ]
    )

    if ipo_records:
        lines.append(
            "IPO records retrieved. Verify issue dates, "
            "price band and subscription status before applying."
        )

        for item in ipo_records[:8]:
            name = (
                item.get("companyName")
                or item.get("issueName")
                or item.get("symbol")
                or item.get("name")
                or "Unnamed issue"
            )
            lines.append(f"• {name}")

    elif ipo_error:
        lines.extend(
            [
                "IPO DATA UNAVAILABLE | RETRIEVAL FAILED",
                "Verify current/upcoming issues directly on NSE.",
                f"Retrieval note: {ipo_error[:160]}",
            ]
        )
    else:
        lines.append(
            "No current/upcoming IPO records were returned "
            "by the configured source."
        )

    lines.extend(
        [
            "",
            "V6.3.2 is a probabilistic research screen; "
            "it does not guarantee profit.",
            "P(UP) is an empirical model estimate, not a "
            "guaranteed probability of profit.",
            "Verify live price, liquidity, corporate news, "
            "market status and order execution before trading.",
        ]
    )

    return "\n".join(lines)


def send_telegram(text):
    token = os.getenv(
        "TELEGRAM_BOT_TOKEN", ""
    ).strip()
    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID", ""
    ).strip()

    if not token or not chat_id:
        raise RuntimeError(
            "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID."
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=20,
    )

    response.raise_for_status()


def save_audit(alert, df):
    stamp = now_ist().strftime(
        "%Y%m%d_%H%M%S"
    )

    Path(
        AUDIT_DIR,
        f"alert_{stamp}.txt",
    ).write_text(
        alert,
        encoding="utf-8",
    )

    if not df.empty:
        df.to_csv(
            Path(
                AUDIT_DIR,
                f"candidates_{stamp}.csv",
            ),
            index=False,
        )


def main():
    frames = download_history()

    if not frames:
        raise RuntimeError(
            "No market data was downloaded."
        )

    regime = market_regime(frames)

    candidates = build_candidates(
        frames,
        regime,
    )

    candidates = classify_candidates(
        candidates,
        regime,
    )

    ipo_records, ipo_error = fetch_ipo_data()

    alert = alert_text(
        regime,
        candidates,
        ipo_records,
        ipo_error,
    )

    print(alert)

    save_audit(
        alert,
        candidates,
    )

    send_telegram(alert)


if __name__ == "__main__":
    main()
