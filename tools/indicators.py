from __future__ import annotations
import json
import warnings
import yfinance as yf
import pandas as pd
import ta

warnings.filterwarnings("ignore")


def _safe(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if (f != f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def calculate_indicators(symbol: str, period: str = "6mo") -> str:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if len(df) < 30:
            return json.dumps({"error": f"Insufficient data for {symbol} (need ≥30 bars)"})

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Moving averages
        df["SMA_20"] = ta.trend.sma_indicator(close, window=20)
        df["SMA_50"] = ta.trend.sma_indicator(close, window=50)
        df["SMA_200"] = ta.trend.sma_indicator(close, window=200)
        df["EMA_12"] = ta.trend.ema_indicator(close, window=12)
        df["EMA_26"] = ta.trend.ema_indicator(close, window=26)

        # Momentum
        df["RSI_14"] = ta.momentum.rsi(close, window=14)
        macd_obj = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
        df["MACD"] = macd_obj.macd()
        df["MACD_Signal"] = macd_obj.macd_signal()
        df["MACD_Hist"] = macd_obj.macd_diff()

        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()

        # Volatility
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Mid"] = bb.bollinger_mavg()
        df["BB_Lower"] = bb.bollinger_lband()
        df["ATR"] = ta.volatility.average_true_range(high, low, close, window=14)

        # Volume
        df["OBV"] = ta.volume.on_balance_volume(close, volume)

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        c = float(latest["Close"])
        pc = float(prev["Close"])

        rsi = _safe(latest["RSI_14"])
        sma20 = _safe(latest["SMA_20"])
        sma50 = _safe(latest["SMA_50"])
        sma200 = _safe(latest["SMA_200"])
        ema12 = _safe(latest["EMA_12"])
        ema26 = _safe(latest["EMA_26"])
        macd = _safe(latest["MACD"])
        macd_sig = _safe(latest["MACD_Signal"])
        macd_hist = _safe(latest["MACD_Hist"])
        bb_upper = _safe(latest["BB_Upper"])
        bb_mid = _safe(latest["BB_Mid"])
        bb_lower = _safe(latest["BB_Lower"])
        atr = _safe(latest["ATR"])
        stoch_k = _safe(latest["Stoch_K"])
        stoch_d = _safe(latest["Stoch_D"])

        prev_macd = _safe(prev["MACD"])
        prev_macd_sig = _safe(prev["MACD_Signal"])
        prev_sma50 = _safe(prev["SMA_50"])
        prev_sma200 = _safe(prev["SMA_200"])

        # ── Signal generation ──────────────────────────────────────────
        bullish = []
        bearish = []

        # RSI
        if rsi is not None:
            if rsi < 30:
                bullish.append(f"RSI oversold ({rsi:.1f} < 30)")
            elif rsi < 40:
                bullish.append(f"RSI nearing oversold ({rsi:.1f})")
            elif rsi > 70:
                bearish.append(f"RSI overbought ({rsi:.1f} > 70)")
            elif rsi > 60:
                bearish.append(f"RSI nearing overbought ({rsi:.1f})")

        # MACD
        if all(v is not None for v in [macd, macd_sig, prev_macd, prev_macd_sig]):
            if macd > macd_sig and prev_macd <= prev_macd_sig:
                bullish.append("MACD bullish crossover")
            elif macd < macd_sig and prev_macd >= prev_macd_sig:
                bearish.append("MACD bearish crossover")
            if macd_hist is not None:
                (bullish if macd_hist > 0 else bearish).append(
                    f"MACD histogram {'positive' if macd_hist > 0 else 'negative'} ({macd_hist:.4f})"
                )

        # Price vs SMAs
        if sma20:
            (bullish if c > sma20 else bearish).append(
                f"Price {'above' if c > sma20 else 'below'} SMA20 ({c:.2f} vs {sma20:.2f})"
            )
        if sma50:
            (bullish if c > sma50 else bearish).append(
                f"Price {'above' if c > sma50 else 'below'} SMA50"
            )
        if sma200:
            (bullish if c > sma200 else bearish).append(
                f"Price {'above' if c > sma200 else 'below'} SMA200 — {'long-term uptrend' if c > sma200 else 'long-term downtrend'}"
            )

        # Golden / Death cross
        if all(v is not None for v in [sma50, sma200, prev_sma50, prev_sma200]):
            if sma50 > sma200 and prev_sma50 <= prev_sma200:
                bullish.append("GOLDEN CROSS: SMA50 crossed above SMA200")
            elif sma50 < sma200 and prev_sma50 >= prev_sma200:
                bearish.append("DEATH CROSS: SMA50 crossed below SMA200")

        # Bollinger Bands
        if bb_upper and bb_lower and bb_mid:
            bb_width = bb_upper - bb_lower
            bb_pct = (c - bb_lower) / bb_width * 100 if bb_width > 0 else 50
            if c > bb_upper:
                bearish.append("Price above upper Bollinger Band (overbought/breakout)")
            elif c < bb_lower:
                bullish.append("Price below lower Bollinger Band (oversold/reversal)")
            elif bb_pct < 30:
                bullish.append(f"Price in lower BB zone ({bb_pct:.0f}%)")
            elif bb_pct > 70:
                bearish.append(f"Price in upper BB zone ({bb_pct:.0f}%)")

        # Stochastic
        if stoch_k and stoch_d:
            if stoch_k < 20 and stoch_d < 20:
                bullish.append(f"Stochastic oversold (K={stoch_k:.1f}, D={stoch_d:.1f})")
            elif stoch_k > 80 and stoch_d > 80:
                bearish.append(f"Stochastic overbought (K={stoch_k:.1f}, D={stoch_d:.1f})")

        # EMA trend
        if ema12 and ema26:
            (bullish if ema12 > ema26 else bearish).append(
                f"EMA12 {'above' if ema12 > ema26 else 'below'} EMA26"
            )

        # Overall bias
        nb, nb2 = len(bullish), len(bearish)
        if nb > nb2 + 1:
            bias, recommendation = "BULLISH", "BUY candidate — multiple signals aligned"
        elif nb2 > nb + 1:
            bias, recommendation = "BEARISH", "SELL/AVOID candidate — bearish alignment"
        else:
            bias, recommendation = "NEUTRAL", "MIXED signals — wait for clearer direction"

        result = {
            "symbol": symbol.upper(),
            "last_close": round(c, 2),
            "change_1d_pct": round((c - pc) / pc * 100, 2),
            "indicators": {
                "RSI_14": rsi,
                "SMA_20": sma20,
                "SMA_50": sma50,
                "SMA_200": sma200,
                "EMA_12": ema12,
                "EMA_26": ema26,
                "MACD": macd,
                "MACD_Signal": macd_sig,
                "MACD_Histogram": macd_hist,
                "BB_Upper": bb_upper,
                "BB_Mid": bb_mid,
                "BB_Lower": bb_lower,
                "ATR_14": atr,
                "Stoch_K": stoch_k,
                "Stoch_D": stoch_d,
            },
            "bullish_signals": bullish,
            "bearish_signals": bearish,
            "signal_count": {"bullish": nb, "bearish": nb2},
            "overall_bias": bias,
            "recommendation": recommendation,
            "atr_stop_loss": round(c - (atr * 2), 2) if atr else None,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "symbol": symbol})
