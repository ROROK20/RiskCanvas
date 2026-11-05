# RiskCanvas Frontend

Interactive portfolio risk dashboard built with Next.js 15 + Tailwind + Recharts.

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3004

## Features

- Live portfolio builder with 9 assets
- 4 quick presets (60/40, All Weather, Aggressive, Diversified)
- 12 risk metrics (VaR at 95/99%, CVaR at 95%, 10-day VaR, max drawdown, Sharpe,
  Sortino, Calmar, CAGR, volatility, skewness, excess kurtosis)
- Return distribution histogram with VaR threshold highlighted
- Monte Carlo fan chart: the engine's 5th-95th / 25th-75th percentile bands
  and median (5,000 zero-drift GBM paths over 252 days; 1,000 in-browser
  paths when the engine is offline)
- 6 crisis stress scenarios (calibrated synthetic paths) scaled by the
  engine-estimated stress beta, with portfolio impact + recovery estimate

All inputs are **synthetic**: seeded per-asset daily returns from a one-factor
model generated in the browser (`lib/mockRisk.ts`), not market data. The page
says so in every state.
