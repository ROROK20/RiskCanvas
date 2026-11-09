# Architecture

RiskCanvas is a small, layered quantitative library with an optional REST
service and a standalone web demo. The core is pure Python (NumPy / pandas /
SciPy) with no framework lock-in — the FastAPI layer and the Next.js frontend
are thin presentation layers on top.

## Component overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         Next.js 15 frontend                        │
│     (frontend/ — dashboard; SYNTHETIC inputs from mockRisk.ts)     │
└───────────────────────────────┬──────────────────────────────────┘
                                 │ POST returns + market_returns (JSON)
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI service (api/app.py)                  │
│   /portfolio/analyze  /portfolio/var  /portfolio/stress            │
│   /portfolio/montecarlo  /scenarios  /health                       │
└───────────────────────────────┬──────────────────────────────────┘
                                 │ calls into
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                         riskcanvas core                            │
│                                                                    │
│  portfolio/loader.py     Portfolio + PortfolioLoader               │
│      from_tickers · from_returns_df · from_csv                     │
│                                                                    │
│  metrics/risk.py         RiskMetrics → RiskReport                  │
│      VaR (hist/parametric) · CVaR · drawdown · tail · Omega        │
│                                                                    │
│  metrics/performance.py  PerformanceMetrics → PerformanceReport    │
│      CAGR · Sharpe · Sortino · Calmar · factor exposure            │
│                                                                    │
│  simulation/monte_carlo.py  MonteCarloSimulator → SimulationResult │
│      simulate_gbm · simulate_t_distribution · simulate_bootstrap   │
│                                                                    │
│  scenarios/stress_test.py   StressTester → ScenarioResult          │
│      run_scenario · run_all · custom_shock · estimate_beta         │
│      beta_from_returns                                             │
└───────────────────────────────┬──────────────────────────────────┘
                                 │ live-ticker calls only
                                 ▼
                         yfinance (live market prices; installed with the
                         package, imported only when a live call runs)
```

## The central data structure: `Portfolio`

Everything flows through the `Portfolio` dataclass
(`riskcanvas/portfolio/loader.py`). It bundles:

- `symbols` — list of asset tickers
- `weights` — `{symbol: weight}` (normalized to sum 1.0)
- `prices` — `pd.DataFrame`, DatetimeIndex × symbols
- `returns` — daily returns DataFrame

and exposes computed properties used by every downstream module:

- `portfolio_returns` — weighted daily return `pd.Series`
- `equity_curve` — cumulative value (base 1.0)
- `correlation_matrix` / `covariance_matrix` (annualized)

`PortfolioLoader` builds a `Portfolio` from one of three sources:

| Method             | Source                                   | Network |
| ------------------ | ---------------------------------------- | ------- |
| `from_tickers`     | live prices via `yfinance` (common sessions only; never filled, never drops a ticker; must cover `start`/`end` and have no gap over 12 calendar days unless `max_gap_days` accepts it) | yes     |
| `from_returns_df`  | a pre-computed returns DataFrame (missing values rejected by column and date, or dropped by explicit `missing="drop"`; never a 0% day) | no      |
| `from_csv`         | a CSV of dated asset returns (parsed, then the same rules) | no      |

All three validate and normalise weights with one shared function,
`check_weights`, which the API also uses for its 422s; a weighted ticker with
no data is always an error. `Portfolio` re-checks at construction, and
`portfolio_returns` raises rather than skip a missing value.

This is what makes the test suite hermetic: tests build portfolios from
synthetic returns with `from_returns_df`, so no network is required.

## Analytics modules

Each analytics module takes a returns `Series` (or a `Portfolio`) and returns a
plain dataclass. `RiskReport` and `PerformanceReport` serialize with `.to_dict()`;
`SimulationResult` and `ScenarioResult` have no `to_dict`, and the API maps their
fields itself (`riskcanvas/api/app.py`).

- **`RiskMetrics`** (`metrics/risk.py`) → `RiskReport`. Historical and
  parametric VaR, historical and parametric (closed-form normal) CVaR /
  Expected Shortfall (one formula for every series, constant ones included;
  square-root-of-time 10-day VaR and the normal tail capped at a total
  loss), drawdown series + max
  drawdown duration (drawdowns run from the starting capital, so a first-day
  loss counts; `drawdown_series` is shared with `PerformanceMetrics`), volatility (daily/annual/downside), skew, excess kurtosis,
  and a VaR stress multiplier (actual vs. normal VaR). Also exposes
  `information_ratio`, `tail_ratio`, `omega_ratio`, and
  `conditional_drawdown_at_risk`.

- **`PerformanceMetrics`** (`metrics/performance.py`) → `PerformanceReport`.
  Total return, CAGR, Sharpe, Sortino, Calmar, win-rate, profit factor, plus
  `rolling_sharpe`, `rolling_vol`, and `factor_exposure` (OLS betas via
  `np.linalg.lstsq`).

- **`MonteCarloSimulator`** (`simulation/monte_carlo.py`) → `SimulationResult`.
  Three forward-simulation models:
  - `simulate_gbm` — Gaussian (log-normal) steps, correlated via an exact
    (Cholesky, or eigen square root when singular; no jitter)
    factor of the covariance matrix
  - `simulate_t_distribution` — multivariate Student-t steps: a correlated
    Gaussian vector times one shared `sqrt((df - 2) / chi2_df)` mixing factor
    per path-day (unit variance, same covariance as GBM; every portfolio of
    the assets is t-distributed, so the tails survive diversification)
  - `simulate_bootstrap` — circular block bootstrap (starts uniform over
    every observation, blocks wrap to the start) of the portfolio's own
    historical returns (non-parametric; no covariance model — co-movement is
    carried by resampling whole historical days); needs at least one block of
    history and always spans the full horizon

  Drift is zero by default (`drift="historical"` / `"risk_free"` optional).
  `SimulationResult` carries terminal-value VaR/CVaR, probability of loss,
  expected terminal value, a `percentile_table`, `percentile_bands`
  (percentiles of simulated value at checkpoint days, for fan charts), the
  Monte Carlo `standard_errors` of each estimate and the `seed`. All three
  models rebalance to the fixed weights daily.

- **`StressTester`** (`scenarios/stress_test.py`) → `ScenarioResult`. Equity-
  market shocks only (no rate, credit or FX factor; `scenario_fit_warnings`
  flags portfolios they cannot represent). Applies
  six deterministic, calibrated synthetic return paths for the S&P 500 (the
  `SCENARIOS` dict): a hand-written stylized shape times one fitted factor, so
  the path's max drawdown from pre-crisis capital equals the crisis's S&P 500
  close-to-close peak-to-trough figure. One step per trading day of the real
  window, ending at its trough, so at beta 1 `total_loss` equals
  `portfolio_drawdown` equals the table. Daily values are stylized: Black
  Monday's and COVID's crash days match the real ones, the other four paths
  are far smoother than the index (README has the numbers). Peak/trough dates
  and closes are recorded per scenario, verified against ^GSPC 2026-10-30.
  Each path is scaled by the beta it is given, day by day (exact against
  the table at beta 1 only; at other betas an approximation whose
  volatility drag depends on the path's roughness, quantified in the
  README) —
  `StressTester` never estimates beta itself; `estimate_beta_detailed` (SPY via
  yfinance) and `beta_from_returns` (from series in hand) do, and the API
  reports which one it used. `run_all` returns results sorted by impact;
  `custom_shock` applies a user-defined S&P 500 decline (exact geometric,
  over `duration_days` trading days), beta-scaled like the built-ins.

## Data flow for a typical analysis

```
weights ──▶ PortfolioLoader.from_tickers ──▶ Portfolio
                                               │
                 ┌─────────────────────────────┼─────────────────────────────┐
                 ▼                             ▼                              ▼
        RiskMetrics(returns)        MonteCarloSimulator(...)        StressTester(beta)
        .compute_all()              .simulate_gbm(portfolio)        .run_all(portfolio)
                 │                             │                              │
                 ▼                             ▼                              ▼
           RiskReport                  SimulationResult              list[ScenarioResult]
                 │                             │                              │
                 └────── JSON (.to_dict(), or the API's field mapping) ◀─────┘
                                          │
                                          ▼
                              FastAPI response  /  console output
```

## Design principles

- **Pure-Python core, thin edges.** No web framework or UI assumptions leak
  into the analytics. The FastAPI app and the frontend are replaceable shells.
- **Dataclass reports.** Every computation returns a typed dataclass, keeping
  testing simple; the two report types add `to_dict()`, and the API maps the
  simulation and scenario results field by field.
- **Deterministic where it matters.** Simulators accept a `seed` so results are
  reproducible in tests and CI.
- **Offline-testable.** The synthetic-returns loaders (`from_returns_df`,
  `from_csv`) let the entire test suite run without network access.

## Frontend

The `frontend/` Next.js app is a dashboard over the FastAPI service. Its
inputs are **synthetic**: `frontend/lib/mockRisk.ts` generates seeded
per-asset daily returns from a one-factor model (one fat-tailed market shock
shared by all assets, plausible betas and volatilities) — not market data —
and `frontend/lib/liveRisk.ts` POSTs them, with the market series, to the
engine, which computes every displayed number, once the inputs have been still for
300 ms (a newer change aborts a request in flight). With no API configured, or if it
fails, the page falls back to approximate in-browser estimates from the same series
and says which of those happened.
A banner states that the inputs are synthetic in every state; the data-source
badge, footer and banner copy come from one typed map over the five states
(`loading` / `live` / `demo` / `empty` / `error`, with `demo` by reason, in
`frontend/lib/engineRequest.mjs`), so no state can inherit another's label.
