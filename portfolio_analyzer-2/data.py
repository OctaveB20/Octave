"""data.py — fetch and cache price data via yfinance."""

import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=3600)
def get_eur_usd() -> float:
    """Fetch current EUR/USD rate."""
    try:
        hist = yf.Ticker("EURUSD=X").history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 1.10  # fallback


@st.cache_data(ttl=900)
def fetch_ticker_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Return OHLCV DataFrame for a ticker."""
    try:
        t = yf.Ticker(ticker)
        interval = "5m" if period == "1d" else "1h" if period == "5d" else "1d"
        df = t.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # Handle both tz-aware and tz-naive indexes
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        else:
            df.index = pd.to_datetime(df.index)
        result = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        # Ensure Close column has no NaN values
        result = result.dropna(subset=["Close"])
        return result
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900)
def fetch_current_price(ticker: str) -> float | None:
    """Return the latest available closing price."""
    try:
        t = yf.Ticker(ticker)
        price = getattr(t.fast_info, "last_price", None)
        if price and price > 0:
            return round(float(price), 4)
        hist = t.history(period="5d", auto_adjust=True)
        if not hist.empty:
            close_prices = hist["Close"].dropna()
            if not close_prices.empty:
                return round(float(close_prices.iloc[-1]), 4)
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker: str) -> dict:
    """Return key fundamental data from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        keys = [
            "trailingPE", "forwardPE", "priceToBook", "dividendYield",
            "beta", "marketCap", "trailingEps", "sector", "industry",
            "52WeekChange", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "averageVolume", "shortRatio", "returnOnEquity",
        ]
        return {k: info.get(k) for k in keys}
    except Exception:
        return {}


def build_portfolio_snapshot(holdings: list, period: str = "1y") -> pd.DataFrame:
    """Build snapshot DataFrame with P&L for each holding."""
    eur_usd = get_eur_usd()
    rows = []
    for h in holdings:
        ticker = h["ticker"]
        current_price = fetch_current_price(ticker)
        if current_price is None:
            current_price = h["avg_price"]

        # USD tickers have no exchange suffix (.AS, .DE, .PA, .L, .MI...)
        is_usd = "." not in ticker

        # Convert prices to EUR only for calculations
        if is_usd:
            current_price_eur = current_price / eur_usd
            avg_price_eur = h["avg_price"] / eur_usd
        else:
            current_price_eur = current_price
            avg_price_eur = h["avg_price"]

        cost_basis = avg_price_eur * h["shares"]
        market_value = current_price_eur * h["shares"]

        pnl_abs = market_value - cost_basis
        pnl_pct = (pnl_abs / cost_basis) * 100 if cost_basis else 0

        rows.append({
            "Ticker":                 ticker,
            "Name":                   h["name"],
            "Category":               h["category"],
            "Shares":                 h["shares"],
            "Avg Price":              round(h["avg_price"], 4),
            "Current Price":          round(current_price_eur, 4),
            "Current Price (native)": round(current_price, 4),
            "Currency":               "USD" if is_usd else "EUR",
            "Cost Basis":             cost_basis,
            "Market Value":           market_value,
            "P&L (€)":                pnl_abs,
            "P&L (%)":                pnl_pct,
        })

    df = pd.DataFrame(rows)
    df["Weight (%)"] = (df["Market Value"] / df["Market Value"].sum()) * 100
    return df
