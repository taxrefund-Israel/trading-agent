from __future__ import annotations
import json
from datetime import datetime
import yfinance as yf
import pandas as pd


def get_stock_data(symbol: str, period: str = "3mo", interval: str = "1d") -> str:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return json.dumps({"error": f"No data found for {symbol}"})

        info = ticker.info or {}
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
        change_30d = ((latest["Close"] - df.iloc[0]["Close"]) / df.iloc[0]["Close"]) * 100

        result = {
            "symbol": symbol.upper(),
            "name": info.get("longName", symbol),
            "sector": info.get("sector", "N/A"),
            "market_cap": info.get("marketCap"),
            "last_close": round(float(latest["Close"]), 2),
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "volume": int(latest["Volume"]),
            "avg_volume": int(info.get("averageVolume", 0)),
            "change_1d_pct": round(change, 2),
            "change_period_pct": round(change_30d, 2),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "pe_ratio": info.get("trailingPE"),
            "data_points": len(df),
            "period": period,
            "interval": interval,
            "as_of": df.index[-1].strftime("%Y-%m-%d %H:%M"),
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})


def get_market_news(symbol: str) -> str:
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []

        articles = []
        for item in news[:8]:
            articles.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "published": datetime.fromtimestamp(item.get("providerPublishTime", 0)).strftime("%Y-%m-%d %H:%M"),
                "url": item.get("link", ""),
            })

        sentiment_keywords = {
            "bullish": ["surge", "soar", "rally", "beat", "record", "growth", "profit", "gain", "upgrade", "buy"],
            "bearish": ["drop", "fall", "miss", "loss", "cut", "downgrade", "sell", "crash", "decline", "concern"],
        }

        bull_count = sum(
            1 for a in articles
            for kw in sentiment_keywords["bullish"]
            if kw in a["title"].lower()
        )
        bear_count = sum(
            1 for a in articles
            for kw in sentiment_keywords["bearish"]
            if kw in a["title"].lower()
        )

        if bull_count > bear_count:
            sentiment = "POSITIVE"
        elif bear_count > bull_count:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

        return json.dumps({
            "symbol": symbol.upper(),
            "news_sentiment": sentiment,
            "bullish_signals": bull_count,
            "bearish_signals": bear_count,
            "articles": articles,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})
