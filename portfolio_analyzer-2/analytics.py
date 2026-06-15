"""analytics.py — risk, performance and technical metrics."""

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252


# ── Performance ──────────────────────────────────────────────────────────────

def total_return(prices: pd.Series) -> float:
    """Total % return over the series."""
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100


def annualised_return(prices: pd.Series) -> float:
    """CAGR in %."""
    if len(prices) < 2:
        return 0.0
    years = len(prices) / TRADING_DAYS
    if years == 0:
        return 0.0
    total = prices.iloc[-1] / prices.iloc[0]
    return (total ** (1 / years) - 1) * 100


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


# ── Risk ─────────────────────────────────────────────────────────────────────

def volatility_annualised(prices: pd.Series) -> float:
    """Annualised historical volatility in %."""
    dr = daily_returns(prices)
    if dr.empty:
        return 0.0
    return float(dr.std() * np.sqrt(TRADING_DAYS) * 100)


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown in % (negative value)."""
    if len(prices) < 2:
        return 0.0
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min() * 100)


def drawdown_series(prices: pd.Series) -> pd.Series:
    roll_max = prices.cummax()
    return ((prices - roll_max) / roll_max) * 100


def value_at_risk(prices: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (daily, in %)."""
    dr = daily_returns(prices)
    if dr.empty:
        return 0.0
    return float(np.percentile(dr, (1 - confidence) * 100) * 100)


def sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Annualised Sharpe ratio."""
    dr = daily_returns(prices)
    if dr.empty or dr.std() == 0:
        return 0.0
    excess = dr.mean() - risk_free_rate / TRADING_DAYS
    return float(excess / dr.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(prices: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    dr = daily_returns(prices)
    if dr.empty:
        return 0.0
    downside = dr[dr < 0]
    if downside.empty or downside.std() == 0:
        return 0.0
    excess = dr.mean() - risk_free_rate / TRADING_DAYS
    return float(excess / downside.std() * np.sqrt(TRADING_DAYS))


def calmar_ratio(prices: pd.Series) -> float:
    """Annualised return / |Max Drawdown|."""
    ann = annualised_return(prices)
    mdd = max_drawdown(prices)
    if mdd == 0:
        return 0.0
    return ann / abs(mdd)


def beta_alpha(asset_prices: pd.Series, bench_prices: pd.Series) -> tuple[float, float]:
    """OLS beta and annualised alpha vs benchmark."""
    a = daily_returns(asset_prices)
    b = daily_returns(bench_prices)
    common = a.index.intersection(b.index)
    if len(common) < 20:
        return 0.0, 0.0
    a, b = a.loc[common], b.loc[common]
    slope, intercept, *_ = stats.linregress(b, a)
    alpha_ann = intercept * TRADING_DAYS * 100
    return float(slope), float(alpha_ann)


def correlation_matrix(price_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Return correlation matrix of daily returns."""
    returns = pd.DataFrame({k: daily_returns(v) for k, v in price_dict.items()})
    return returns.corr()


# ── Technicals ────────────────────────────────────────────────────────────────

def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(prices: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    """Rolling Z-score of price vs its own recent history."""
    roll_mean = prices.rolling(window).mean()
    roll_std  = prices.rolling(window).std()
    return (prices - roll_mean) / roll_std.replace(0, np.nan)


def fifty_two_week_position(prices: pd.Series) -> dict:
    """Where is current price in its 52-week range?"""
    if len(prices) < 2:
        return {}
    high = prices.max()
    low  = prices.min()
    cur  = prices.iloc[-1]
    position = (cur - low) / (high - low) * 100 if high != low else 50
    return {
        "52w_high": round(high, 4),
        "52w_low":  round(low, 4),
        "current":  round(cur, 4),
        "position_pct": round(position, 1),
    }


# ── Summary ────────────────────────────────────────────────────────────────

def full_stats(prices: pd.Series, bench_prices: pd.Series | None = None,
               label: str = "") -> dict:
    """Compute all metrics for one ticker and return as dict."""
    b, a = beta_alpha(prices, bench_prices) if bench_prices is not None else (None, None)
    return {
        "Ticker":              label,
        "Total Return (%)":    round(total_return(prices), 2),
        "Ann. Return (%)":     round(annualised_return(prices), 2),
        "Volatility (%)":      round(volatility_annualised(prices), 2),
        "Max Drawdown (%)":    round(max_drawdown(prices), 2),
        "Sharpe":              round(sharpe_ratio(prices), 2),
        "Sortino":             round(sortino_ratio(prices), 2),
        "Calmar":              round(calmar_ratio(prices), 2),
        "VaR 95% (daily %)":   round(value_at_risk(prices), 2),
        "Beta":                round(b, 2) if b is not None else None,
        "Alpha (ann. %)":      round(a, 2) if a is not None else None,
    }
