# portfolio_analyzer

A local Streamlit dashboard for deep portfolio analysis — built around your actual holdings, PRU, and categories.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit your holdings
#    Open portfolio.json and update tickers, shares, avg_price (PRU), category

# 3. Run the dashboard
streamlit run app.py
```

Opens at http://localhost:8501

## Tabs

| Tab | What you get |
|-----|-------------|
| **Overview** | Snapshot table, allocation donut, P&L bars, cumulative return vs benchmark |
| **Deep Dive** | Per-ticker: Bollinger Bands + PRU line, RSI, MACD, drawdown, Z-score |
| **Risk** | Volatility, Sharpe, Sortino, Calmar, VaR 95%, Beta, Alpha — all positions |
| **Correlations** | Full heatmap of daily return correlations |
| **Fundamentals** | P/E, P/B, dividend yield, beta, ROE, sector, market cap |

## Ticker format

| Exchange | Format example |
|----------|---------------|
| Euronext Amsterdam | `VWCE.AS` |
| XETRA (Frankfurt) | `DBXJ.DE` |
| London | `IGLD.L` |
| US | `GOOGL`, `CRWD` |
| OTC / micro-cap | `OTLK`, `NVNO` |

## Updating holdings

Edit `portfolio.json` directly, or use the sidebar in the app (sidebar → expanders → Save changes).

## Data

- Price data: Yahoo Finance via `yfinance` (~15 min delay, free)
- Cached 15 min for prices, 1 hour for fundamentals (Streamlit cache)
- No API key required
