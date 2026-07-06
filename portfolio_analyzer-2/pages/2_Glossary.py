"""pages/2_Glossary.py — Metrics Glossary: plain-language definitions + live examples."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    total_return, annualised_return, volatility_annualised, max_drawdown,
    drawdown_series, value_at_risk, sharpe_ratio, sortino_ratio, calmar_ratio,
    beta_alpha, rsi, macd, bollinger_bands, zscore, fifty_two_week_position,
    daily_returns,
)

st.set_page_config(page_title="Glossary", page_icon="📖", layout="wide")

PLOTLY_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono, monospace", color="#c9d1d9", size=11),
    margin=dict(l=10, r=10, t=25, b=20),
    xaxis=dict(gridcolor="#21262d", showgrid=True),
    yaxis=dict(gridcolor="#21262d", showgrid=True),
)
GREEN, RED, BLUE, AMBER = "#3fb950", "#f85149", "#58a6ff", "#e3b341"

st.markdown("# 📖 Metrics Glossary")
st.caption("Plain-language definitions with a worked example for every stat used in this dashboard.")
st.markdown("---")

# ── Synthetic demo data (fixed seed → reproducible examples) ─────────────────
rng = np.random.default_rng(7)
n = 252
dates = pd.bdate_range("2025-01-01", periods=n)
price = pd.Series(100 * (1 + rng.normal(0.0006, 0.018, n)).cumprod(), index=dates)
bench = pd.Series(100 * (1 + rng.normal(0.0004, 0.012, n)).cumprod(), index=dates)
dr = daily_returns(price)

# ── Small chart helpers ───────────────────────────────────────────────────────

def line_fig(series, name, color=BLUE, hlines=None, height=170, fill=None):
    fig = go.Figure(go.Scatter(x=series.index, y=series, name=name,
                                line=dict(color=color, width=1.6), fill=fill,
                                fillcolor="rgba(248,81,73,0.15)" if fill else None))
    for yv, c in (hlines or []):
        fig.add_hline(y=yv, line=dict(color=c, dash="dot", width=1))
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=False)
    return fig


def hist_fig(data, vline=None, height=170):
    fig = go.Figure(go.Histogram(x=data * 100, marker_color=BLUE, opacity=0.75, nbinsx=30))
    if vline is not None:
        fig.add_vline(x=vline, line=dict(color=RED, width=1.5, dash="dash"))
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=False, xaxis_title="Daily return %")
    return fig


def scatter_fig(x, y, slope, intercept, height=170):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x * 100, y=y * 100, mode="markers",
                              marker=dict(color=BLUE, size=4, opacity=0.5)))
    xs = np.linspace(x.min(), x.max(), 20)
    fig.add_trace(go.Scatter(x=xs * 100, y=(slope * xs + intercept) * 100,
                              line=dict(color=AMBER, width=2)))
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=False,
                       xaxis_title="Benchmark %", yaxis_title="Asset %")
    return fig


def macd_fig(height=170):
    macd_line, signal_line, hist = macd(price)
    colors = [GREEN if v >= 0 else RED for v in hist]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist.index, y=hist, marker_color=colors, opacity=0.6))
    fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, line=dict(color=BLUE, width=1.4)))
    fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, line=dict(color=AMBER, width=1.4)))
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=False)
    return fig


def bollinger_fig(height=170):
    upper, mid, lower = bollinger_bands(price)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=upper, line=dict(color=RED, width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=price.index, y=lower, line=dict(color=GREEN, width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(63,185,80,0.06)"))
    fig.add_trace(go.Scatter(x=price.index, y=price, line=dict(color=BLUE, width=1.8)))
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=False)
    return fig


def range_fig(low, high, cur, height=110):
    pct = (cur - low) / (high - low) * 100 if high != low else 50
    fig = go.Figure(go.Bar(x=[pct], y=["52w"], orientation="h", marker_color=BLUE, width=0.5))
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=False,
                       xaxis=dict(range=[0, 100], gridcolor="#21262d"), yaxis=dict(showticklabels=False))
    return fig


def corr_fig(height=170):
    both = pd.DataFrame({"Asset": daily_returns(price), "Benchmark": daily_returns(bench)})
    c = both.corr()
    fig = go.Figure(go.Heatmap(z=c.values, x=c.columns, y=c.index, zmid=0, zmin=-1, zmax=1,
                                colorscale=[[0, RED], [0.5, "#0d1117"], [1, GREEN]],
                                text=c.round(2).values, texttemplate="%{text}"))
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig


# ── Precompute example values ────────────────────────────────────────────────
b, a = beta_alpha(price, bench)
fw = fifty_two_week_position(price)

GLOSSARY = [
    ("Total Return",
     "The simple % change in price from the first to the last data point, ignoring the "
     "path in between. It's the headline number investors check first — buying at €100 and "
     "ending at €128 is a +28% total return. It says nothing about the volatility or drawdowns "
     "experienced along the way, so read it alongside risk metrics.",
     f"Example series → **{total_return(price):+.1f}%** total return",
     lambda: line_fig(price, "Price")),

    ("Annualised Return (CAGR)",
     "The Compound Annual Growth Rate: what the total return looks like if it had grown at a "
     "steady, compounding rate every year. It lets you fairly compare a position held 3 months "
     "against one held 3 years by expressing both 'per year'. A CAGR of 12% means capital "
     "compounded at 12% annually on average.",
     f"Example series → **{annualised_return(price):+.1f}%** CAGR",
     lambda: line_fig(price, "Price", color=AMBER)),

    ("Volatility",
     "Annualised standard deviation of daily returns — how much the price swings around its "
     "average, expressed as a yearly %. Higher volatility means bigger swings in both directions, "
     "so more uncertainty (not necessarily lower return). ±1% daily moves compound to roughly "
     "16% annualised; ±3% daily is closer to 48%.",
     f"Example series → **{volatility_annualised(price):.1f}%** annualised volatility",
     lambda: hist_fig(dr)),

    ("Max Drawdown",
     "The largest peak-to-trough loss suffered before recovering — the worst case if you bought "
     "at the top and sold at the bottom. A -35% max drawdown means the position lost over a third "
     "of its value from its prior high at some point. Useful for gauging how much pain a strategy "
     "actually requires you to tolerate.",
     f"Example series → **{max_drawdown(price):.1f}%** max drawdown",
     lambda: line_fig(drawdown_series(price), "Drawdown %", color=RED, fill="tozeroy")),

    ("Sharpe Ratio",
     "Risk-adjusted return: excess return over the risk-free rate, divided by volatility. It "
     "answers 'how much return per unit of risk taken?'. Above 1 is considered good, above 2 very "
     "good, below 0 means you underperformed a risk-free asset while still taking on risk. Compares "
     "smoothness of returns, not just size.",
     f"Example series → **{sharpe_ratio(price):.2f}** Sharpe",
     lambda: line_fig(price, "Price")),

    ("Sortino Ratio",
     "A refinement of Sharpe that only penalises downside volatility — the swings investors "
     "actually dislike — while ignoring beneficial upside swings. Two assets with identical Sharpe "
     "ratios can have very different Sortino ratios if one's volatility comes mostly from sharp "
     "gains rather than losses. Generally a fairer measure for asymmetric returns.",
     f"Example series → **{sortino_ratio(price):.2f}** Sortino",
     lambda: hist_fig(dr)),

    ("Calmar Ratio",
     "Annualised return divided by the absolute value of max drawdown. It measures return earned "
     "per unit of the worst drawdown suffered, rather than per unit of everyday volatility like "
     "Sharpe. A Calmar of 2 means annual return was twice the deepest peak-to-trough loss — useful "
     "for judging recovery-adjusted performance.",
     f"Example series → **{calmar_ratio(price):.2f}** Calmar",
     lambda: line_fig(drawdown_series(price), "Drawdown %", color=RED, fill="tozeroy")),

    ("Value at Risk (VaR 95%)",
     "Historical VaR estimates the worst daily loss you can statistically expect at a given "
     "confidence level (typically 95%). A daily VaR of -2.5% means that on 95% of trading days, "
     "losses should not exceed 2.5% — the remaining 5% of days can be worse, sometimes much worse.",
     f"Example series → **{value_at_risk(price):.2f}%** daily VaR (95%)",
     lambda: hist_fig(dr, vline=value_at_risk(price))),

    ("Beta",
     "Measures how sensitive an asset's returns are to its benchmark's returns — its market risk. "
     "A beta of 1 means it moves in lockstep with the market; 1.5 means amplified moves (50% more "
     "volatile); below 1 means dampened moves. Beta only captures the linear, market-driven part "
     "of returns.",
     f"Example series → **{b:.2f}** beta vs benchmark",
     lambda: scatter_fig(daily_returns(bench), daily_returns(price), b, a / 100 / 252)),

    ("Alpha (annualised)",
     "The annualised return left over after accounting for what beta and the benchmark's moves "
     "would already explain — the 'skill' or edge component. Positive alpha means the asset "
     "outperformed what its market exposure alone would predict; negative alpha means it "
     "underperformed even after adjusting for market risk carried.",
     f"Example series → **{a:+.1f}%** annualised alpha",
     lambda: scatter_fig(daily_returns(bench), daily_returns(price), b, a / 100 / 252)),

    ("RSI (Relative Strength Index)",
     "A momentum oscillator (0–100) comparing the size of recent gains to recent losses over 14 "
     "days. Readings above 70 typically signal 'overbought' (possible pullback ahead), below 30 "
     "signal 'oversold' (possible bounce). A mean-reversion signal, not a guarantee — strong "
     "trends can stay overbought a long time.",
     f"Example series → RSI now **{rsi(price).iloc[-1]:.0f}**",
     lambda: line_fig(rsi(price), "RSI", hlines=[(70, RED), (30, GREEN)])),

    ("MACD",
     "Moving Average Convergence Divergence tracks the gap between a fast (12-day) and slow "
     "(26-day) exponential moving average. When the MACD line crosses above its 9-day signal "
     "line, it's often read as bullish momentum building; crossing below is bearish. The histogram "
     "visualises the gap, i.e. momentum shifts.",
     "Example series → MACD line, signal line & histogram",
     lambda: macd_fig()),

    ("Bollinger Bands",
     "A moving average (20-day) with bands plotted 2 standard deviations above and below it, "
     "forming a volatility envelope. Prices near the upper band suggest the asset is statistically "
     "'expensive' relative to its recent range; near the lower band, statistically 'cheap'. Bands "
     "widen in volatile periods, narrow when calm.",
     "Example series → price with 20-day Bollinger envelope",
     lambda: bollinger_fig()),

    ("Z-score",
     "How many standard deviations the current price sits from its own rolling 20-day average — "
     "a mean-reversion signal. A Z-score above +2 means the price is unusually high relative to "
     "its recent range; below -2, unusually low. Traders sometimes watch for reversion back toward "
     "zero after such extremes.",
     f"Example series → Z-score now **{zscore(price).iloc[-1]:.2f}**",
     lambda: line_fig(zscore(price), "Z-score", hlines=[(2, RED), (-2, GREEN)])),

    ("52-Week Position",
     "Shows where the current price sits within its own 52-week high-low range, as a percentile. "
     "A position of 90% means the price is near its yearly high; 10% means near its yearly low. "
     "A quick, intuitive gauge of where you are in the longer-term cycle.",
     f"Example series → **{fw['position_pct']:.0f}%ile** of 52-week range",
     lambda: range_fig(fw["52w_low"], fw["52w_high"], fw["current"])),

    ("Correlation Matrix",
     "Pairwise correlation of daily returns between assets, ranging from -1 (move in exact "
     "opposite directions) to +1 (perfect lockstep). Low or negative correlations between holdings "
     "are valuable for diversification, since losses in one position are more likely offset by "
     "gains in another during downturns.",
     "Example series → asset vs. benchmark correlation",
     lambda: corr_fig()),
]

for title, desc, example_line, fig_fn in GLOSSARY:
    with st.expander(f"**{title}**", expanded=False):
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown(desc)
            st.caption(example_line)
        with c2:
            st.plotly_chart(fig_fn(), width="stretch", key=f"chart_{title}")
