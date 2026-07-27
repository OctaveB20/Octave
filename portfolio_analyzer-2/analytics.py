"""analytics.py — risk, performance and technical metrics."""

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252


# ── Performance ──────────────────────────────────────────────────────────

def total_return(prices: pd.Series) -> float:
    """Total % return over the series."""
    if prices is None or len(prices) < 2:
        return 0.0
    # Filter out NaN values
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0
    return (prices.iloc[-1] / prices.iloc[0] - 1) * 100


def annualised_return(prices: pd.Series) -> float:
    """CAGR in %."""
    if prices is None or len(prices) < 2:
        return 0.0
    # Filter out NaN values
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0
    years = len(prices) / TRADING_DAYS
    if years == 0 or years < 0.01:  # Need at least some data
        return 0.0
    total = prices.iloc[-1] / prices.iloc[0]
    if total <= 0:
        return 0.0
    return (total ** (1 / years) - 1) * 100


def daily_returns(prices: pd.Series) -> pd.Series:
    """Calculate daily returns, handling NaN values."""
    if prices is None or len(prices) < 2:
        return pd.Series(dtype=float)
    prices = prices.dropna()
    if len(prices) < 2:
        return pd.Series(dtype=float)
    return prices.pct_change().dropna()


# ── Risk ─────────────────────────────────────────────────────────────

def volatility_annualised(prices: pd.Series) -> float:
    """Annualised historical volatility in %."""
    dr = daily_returns(prices)
    if dr is None or dr.empty or len(dr) < 2:
        return 0.0
    std = dr.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(std * np.sqrt(TRADING_DAYS) * 100)


def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown in % (negative value)."""
    if prices is None or len(prices) < 2:
        return 0.0
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return float(drawdown.min() * 100)


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Return drawdown series."""
    if prices is None or len(prices) < 2:
        return pd.Series(dtype=float)
    prices = prices.dropna()
    if len(prices) < 2:
        return pd.Series(dtype=float)
    roll_max = prices.cummax()
    return ((prices - roll_max) / roll_max) * 100


def value_at_risk(prices: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR at given confidence level (daily, in %)."""
    dr = daily_returns(prices)
    if dr is None or dr.empty or len(dr) < 5:
        return 0.0
    return float(np.percentile(dr, (1 - confidence) * 100) * 100)


def sharpe_ratio(prices: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Annualised Sharpe ratio."""
    dr = daily_returns(prices)
    if dr is None or dr.empty or len(dr) < 5:
        return 0.0
    std = dr.std()
    if std == 0 or np.isnan(std):
        return 0.0
    excess = dr.mean() - risk_free_rate / TRADING_DAYS
    return float(excess / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(prices: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    dr = daily_returns(prices)
    if dr is None or dr.empty or len(dr) < 5:
        return 0.0
    downside = dr[dr < 0]
    if downside.empty or len(downside) == 0:
        return 0.0
    std = downside.std()
    if std == 0 or np.isnan(std):
        return 0.0
    excess = dr.mean() - risk_free_rate / TRADING_DAYS
    return float(excess / std * np.sqrt(TRADING_DAYS))


def calmar_ratio(prices: pd.Series) -> float:
    """Annualised return / |Max Drawdown|."""
    ann = annualised_return(prices)
    mdd = max_drawdown(prices)
    if mdd == 0 or mdd >= 0 or np.isnan(ann) or np.isnan(mdd):
        return 0.0
    return float(ann / abs(mdd))


def beta_alpha(asset_prices: pd.Series, bench_prices: pd.Series) -> tuple[float, float]:
    """OLS beta and annualised alpha vs benchmark."""
    if asset_prices is None or bench_prices is None:
        return 0.0, 0.0
    a = daily_returns(asset_prices)
    b = daily_returns(bench_prices)
    if a is None or b is None or a.empty or b.empty:
        return 0.0, 0.0
    common = a.index.intersection(b.index)
    if len(common) < 20:
        return 0.0, 0.0
    a, b = a.loc[common], b.loc[common]
    if len(a) < 20 or len(b) < 20:
        return 0.0, 0.0
    try:
        slope, intercept, *_ = stats.linregress(b, a)
        alpha_ann = intercept * TRADING_DAYS * 100
        return float(slope), float(alpha_ann)
    except Exception:
        return 0.0, 0.0


def correlation_matrix(price_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Return correlation matrix of daily returns."""
    if not price_dict:
        return pd.DataFrame()
    returns = pd.DataFrame({k: daily_returns(v) for k, v in price_dict.items()})
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


# ── Technicals ──────────────────────────────────────────────────────────

def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    if prices is None or len(prices) < window:
        return pd.Series(dtype=float)
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(prices: pd.Series, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    if prices is None or len(prices) < slow:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    """Calculate Bollinger Bands."""
    if prices is None or len(prices) < window:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    mid = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def zscore(prices: pd.Series, window: int = 20) -> pd.Series:
    """Rolling Z-score of price vs its own recent history."""
    if prices is None or len(prices) < window:
        return pd.Series(dtype=float)
    roll_mean = prices.rolling(window).mean()
    roll_std = prices.rolling(window).std()
    return (prices - roll_mean) / roll_std.replace(0, np.nan)


def fifty_two_week_position(prices: pd.Series) -> dict:
    """Where is current price in its 52-week range?"""
    if prices is None or len(prices) < 2:
        return {}
    prices = prices.dropna()
    if len(prices) < 2:
        return {}
    high = prices.max()
    low = prices.min()
    cur = prices.iloc[-1]
    if high == low or high <= 0 or low <= 0:
        return {}
    position = (cur - low) / (high - low) * 100
    return {
        "52w_high": round(high, 4),
        "52w_low": round(low, 4),
        "current": round(cur, 4),
        "position_pct": round(position, 1),
    }


# ── Summary ──────────────────────────────────────────────────────────

def full_stats(prices: pd.Series, bench_prices: pd.Series | None = None,
               label: str = "") -> dict:
    """Compute all metrics for one ticker and return as dict."""
    if prices is None or len(prices) < 2:
        return {
            "Ticker": label,
            "Total Return (%)": 0.0,
            "Ann. Return (%)": 0.0,
            "Volatility (%)": 0.0,
            "Max Drawdown (%)": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Calmar": 0.0,
            "VaR 95% (daily %)": 0.0,
            "Beta": None,
            "Alpha (ann. %)": None,
        }
    
    prices_clean = prices.dropna()
    if len(prices_clean) < 2:
        return {
            "Ticker": label,
            "Total Return (%)": 0.0,
            "Ann. Return (%)": 0.0,
            "Volatility (%)": 0.0,
            "Max Drawdown (%)": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Calmar": 0.0,
            "VaR 95% (daily %)": 0.0,
            "Beta": None,
            "Alpha (ann. %)": None,
        }
    
    b, a = (0.0, 0.0)
    if bench_prices is not None:
        try:
            b, a = beta_alpha(prices_clean, bench_prices)
            b = float(b) if not np.isnan(b) else None
            a = float(a) if not np.isnan(a) else None
        except Exception:
            b, a = None, None
    
    tr = total_return(prices_clean)
    ar = annualised_return(prices_clean)
    vol = volatility_annualised(prices_clean)
    mdd = max_drawdown(prices_clean)
    sharpe = sharpe_ratio(prices_clean)
    sortino = sortino_ratio(prices_clean)
    calmar = calmar_ratio(prices_clean)
    var = value_at_risk(prices_clean)
    
    return {
        "Ticker": label,
        "Total Return (%)": round(tr, 2) if not np.isnan(tr) else 0.0,
        "Ann. Return (%)": round(ar, 2) if not np.isnan(ar) else 0.0,
        "Volatility (%)": round(vol, 2) if not np.isnan(vol) else 0.0,
        "Max Drawdown (%)": round(mdd, 2) if not np.isnan(mdd) else 0.0,
        "Sharpe": round(sharpe, 2) if not np.isnan(sharpe) else 0.0,
        "Sortino": round(sortino, 2) if not np.isnan(sortino) else 0.0,
        "Calmar": round(calmar, 2) if not np.isnan(calmar) else 0.0,
        "VaR 95% (daily %)": round(var, 2) if not np.isnan(var) else 0.0,
        "Beta": round(b, 2) if b is not None and not np.isnan(b) else None,
        "Alpha (ann. %)": round(a, 2) if a is not None and not np.isnan(a) else None,
    }
