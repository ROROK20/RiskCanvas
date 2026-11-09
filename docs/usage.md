# Usage Guide

Concrete, copy-paste examples for the RiskCanvas Python API and REST service.
All snippets are grounded in the actual public API.

> **Network note.** `PortfolioLoader.from_tickers(...)` fetches live prices via
> `yfinance` and needs internet access. To run fully offline, build portfolios
> from a returns DataFrame with `from_returns_df(...)` or from a CSV with
> `from_csv(...)` — see [Offline / synthetic data](#offline--synthetic-data).

## Installation

Python 3.11+ (3.12+ for the pinned `-c requirements.txt` install).

```bash
pip install -e ".[dev]" -c requirements.txt   # from a clone, with test extras, pinned
# or runtime only:
pip install -r requirements.txt
```

## 1. Load a portfolio

### From live tickers (yfinance)

```python
from riskcanvas.portfolio.loader import PortfolioLoader

loader = PortfolioLoader()
portfolio = loader.from_tickers(
    {"AAPL": 0.30, "MSFT": 0.30, "GOOGL": 0.20, "AMZN": 0.20},
    start="2020-01-01",
    end="2023-12-31",
)
print(portfolio)              # Portfolio(name='Portfolio', assets=4, period=...)
returns = portfolio.portfolio_returns   # weighted daily returns (pd.Series)
```

`from_tickers` keeps every ticker and never invents a price. A ticker with no
data raises `ValueError` naming it; so does one whose history starts more
than 7 calendar days after `start` (listed later) or ends more than 7 before
`end` (delisted; with `end=None`, before today), with its first or last
available date, even when it is the only ticker. A
gap of more than 12 calendar days inside a history, or between sessions common to
every ticker, raises `ValueError` naming the ticker and dates instead of becoming
one multi-month "daily" return. Regular closures such as Golden Week and Lunar New
Year (up to 11 days between sessions) fit under that. For a longer closure, pass
`max_gap_days=` (the error names the value): it sets the limit inside the window
and at both ends, and the return across a gap spans all of it. Prices are aligned
on the sessions every ticker traded, never back- or forward-filled.

**No FX conversion.** Each ticker's prices are used in its own listing currency,
so `{"7203.T": 0.5, "AAPL": 0.5}` blends yen and dollar returns without the
USD/JPY move, and the figures hold only for a single-currency portfolio. To
analyse a cross-currency one, convert the prices first and use `from_returns_df`.

### Offline / synthetic data

```python
import numpy as np
import pandas as pd
from riskcanvas.portfolio.loader import PortfolioLoader

# Build a synthetic returns DataFrame (no network)
idx = pd.bdate_range("2021-01-01", periods=750)
rng = np.random.default_rng(0)
rets = pd.DataFrame(
    rng.normal(0.0004, 0.012, size=(len(idx), 3)),
    index=idx,
    columns=["A", "B", "C"],
)

loader = PortfolioLoader()
portfolio = loader.from_returns_df(rets, weights={"A": 0.5, "B": 0.3, "C": 0.2})
```

### From a CSV of dated returns

```python
# CSV with a "Date" column + one column of daily returns per asset
portfolio = loader.from_csv(
    "returns.csv",
    weights={"SPY": 0.6, "AGG": 0.4},
    date_col="Date",
)
```

Both offline loaders enforce the REST API's contract, so nothing is invented
or rescaled behind your back:

- **Weights** are long-only (finite, `>= 0`, positive sum) and normalised by
  their sum, the only transformation: `{"SPY": 60, "AGG": 40}` is
  `{"SPY": 0.6, "AGG": 0.4}`. A weight for a ticker with no column is a
  `ValueError` naming it, never dropped. The portfolio holds exactly the
  weighted tickers; other columns are ignored. With no `weights`, every column
  is held at equal weight. `normalize_weights=False` keeps them as given, as
  in `from_tickers`: a partially invested portfolio whose remainder
  `1 - sum` is cash (zero return in every model; the risk-free rate under
  Monte Carlo `drift="risk_free"`). A sum above 1 (leverage) is refused:
  no financing is modelled.
- **Values** are finite fractions in [-1, 10] (`-0.02` for -2%), on a
  `DatetimeIndex` that is strictly increasing. A CSV without its date column
  is an error (pass `date_col=`).
- **A file in percent** is rejected in two ways, and one case passes. A day
  below -1 (-1%) is out of range. A quiet asset never gets there, so a column
  with 20 or more observations, up and down days, and a median absolute return
  of 0.10 to 1.0 is refused as percent too: about 15% daily volatility, 235% a
  year, at a normal shape, so ordinary high-volatility assets are untouched.
  `allow_large_returns=True` accepts a window that really is that violent.
  **Not caught:** a percent column quieter than that (cash-like, typically
  under 0.1% a day) cannot be told from fractions, so check `annual_vol`
  against what you loaded.
- **Levels** are refused too: gross returns (`1.01, 0.99`) and prices under 10
  stay inside the range, so 5 or more observations, at least 95% of them above
  zero, with a median of 0.10 or more, are a `ValueError` saying to subtract 1
  or take the percentage change. `allow_large_returns=True` accepts them as well.
  **Not caught:** prices under 0.10; over HTTP their figures carry the warning
  `implausible_statistics`.
- **A series that is not daily** gets larger floors for both checks, by the square
  root of the time one observation spans: 0.22 weekly, 0.46 monthly and, for levels,
  at most 0.75, which still catches gross returns. The dates set the rate;
  `from_returns_df(..., periods_per_year=12)` states it.
- **Missing values** (NaN, an empty CSV cell), leading, interior or trailing,
  are a `ValueError` naming the column and dates. A missing return is never
  counted as a 0% day, which would understate volatility and VaR. To exclude
  those dates instead, say so: `from_csv(..., missing="drop")` drops every
  date on which any held asset has no value (nothing spans the gap for
  returns, so those days are absent from every statistic; a month left with
  none is not counted in `positive_months`). Dropping does not compound, so do
  not use it to mix calendars: a 7-day asset with blanks at weekends beside a
  5-day one loses its weekend moves, and its Monday covers one day where the
  other's covers three. Use `from_tickers`, which aligns prices, or compound the
  returns onto common dates first.

The shared weight rule is public as
`riskcanvas.portfolio.loader.check_weights(weights, normalize=True)`.

## 2. Risk metrics

```python
from riskcanvas.metrics.risk import RiskMetrics

risk = RiskMetrics(portfolio.portfolio_returns).compute_all()

print(f"VaR 95% (1-day): {risk.var_95_1d:.2%}")
print(f"VaR 99% (1-day): {risk.var_99_1d:.2%}")
print(f"CVaR 95%:        {risk.cvar_95:.2%}")
print(f"Max Drawdown:    {risk.max_drawdown:.2%}")
print(f"Annual Vol:      {risk.annual_vol:.2%}")
print(f"Skew / Kurtosis: {risk.skewness:.2f} / {risk.excess_kurtosis:.2f}")

# Serialize the whole report
print(risk.to_dict())
```

Drawdowns are measured from the starting capital (1.0), so a loss on the
first day counts: `[-0.30, 0.01, 0.01]` has a max drawdown of -30%, and a
total loss is exactly -100%.

Individual estimators are also available directly:

```python
rm = RiskMetrics(portfolio.portfolio_returns)
rm.var_historical(0.99)        # historical VaR at 99%
rm.var_parametric(0.95)        # normal-distribution 1-day VaR
rm.cvar_parametric(0.95)       # normal 1-day Expected Shortfall (closed form)
rm.cvar_historical(0.95)       # Expected Shortfall
rm.var_historical_scaled(0.95, 10)  # 1-day VaR x sqrt(10), capped at 1.0
rm.omega_ratio()               # Omega ratio (+inf without losses)
rm.tail_ratio()                # 95th gain / 5th loss (NaN when the 5th is 0)

# A benchmark on the same dates (synthetic here; e.g. SPY's daily returns)
bench = pd.Series(np.random.default_rng(1).normal(0.0003, 0.01, len(rm.returns)),
                  index=rm.returns.index)
rm.information_ratio(bench)    # needs a benchmark value on every date (else ValueError)
```

No helper invents a value: an undefined statistic is NaN (`null` over the
wire), a ratio without a denominator is `+inf`, and a benchmark or factor
series that misses portfolio dates is an error, never filled with 0%.
`PerformanceMetrics.rolling_sharpe` leaves its first `window - 1` days NaN.

Historical VaR is minus the linearly interpolated `1 - c` quantile (numpy's
default); CVaR is the discrete Acerbi-Tasche expected shortfall, the mean of
the worst `n * (1 - c)` days with the boundary day fractionally weighted.

Annual figures use the observations per year the dates imply:
`riskcanvas.metrics.risk.periods_per_year(index)` reads 12 for monthly and 52 for
weekly spacing, 365 for daily on all seven weekdays, 252 for daily on five (any
two days closed), and the **measured** rate for anything else it can measure
(Monday-Thursday about 209, every other day about 183, a 6-day week about 313,
quarterly 4, and two exchanges' common sessions, 227-239 a year over a year or more
of dates). It raises `ValueError` only for intraday bars and fewer than one
observation a year, for which you pass `RiskMetrics(returns, periods=...)` or
`PerformanceMetrics(returns, periods=...)`. `compute_all().periods_per_year` says
which rate was used; over HTTP a measured one carries the warning
`non_standard_calendar`. One exchange's holidays and holes keep a standard
calendar's rate (each observation counts as one session); `periods=` overrides it if
your returns span the holes. CAGR's years are `n / periods` on a calendar holding its
rate, the dates' span otherwise, and always `n / periods` when you pass `periods=`. On a weekly or monthly series the `1d` / `10d` figures and a
simulation's `horizon_days` count weeks or months.

VaR and CVaR are loss fractions (positive = loss) from one formula for every
series: minus the tail quantile, so a series whose worst days are still gains
(a constant +1%) has a negative VaR, historical and parametric alike.
Returns are >= -100%, so model extrapolations past a total loss
(square-root-of-time scaling, the unbounded normal tail) are capped at 1.0.

## 3. Performance metrics

```python
from riskcanvas.metrics.performance import PerformanceMetrics

perf = PerformanceMetrics(portfolio.portfolio_returns).compute_all()
print(f"CAGR:    {perf.cagr:.2%}")
print(f"Sharpe:  {perf.sharpe:.2f}")
print(f"Sortino: {perf.sortino:.2f}")
print(f"Calmar:  {perf.calmar:.2f}")

# Rolling series
pm = PerformanceMetrics(portfolio.portfolio_returns)
rolling_sharpe = pm.rolling_sharpe(window=63)
rolling_vol = pm.rolling_vol(window=21)
```

The risk-free rate is 2% a year unless you pass `risk_free_rate=`; over HTTP it is
fixed, and `/portfolio/analyze` states it as `risk_free_rate`. `avg_win` and
`avg_loss` average the gaining and the losing days; a flat day counts as neither.

### Factor exposure (OLS betas)

One OLS regression of the portfolio's daily returns on factor series you
supply: point estimates of `alpha` (per day) and one loading per factor, no
standard errors or R². The factors must cover every portfolio date.

```python
import pandas as pd

# factors: factor return series on the portfolio's dates (synthetic here;
# e.g. Fama-French daily factors)
idx = portfolio.portfolio_returns.index
rng = np.random.default_rng(2)
market_returns = rng.normal(0.0004, 0.01, len(idx))
smb_returns = rng.normal(0.0, 0.005, len(idx))
factors = pd.DataFrame({"MKT": market_returns, "SMB": smb_returns}, index=idx)
betas = PerformanceMetrics(portfolio.portfolio_returns).factor_exposure(factors)
print(betas)   # index: ['alpha', 'MKT', 'SMB']
```

## 4. Monte Carlo simulation

```python
from riskcanvas.simulation.monte_carlo import MonteCarloSimulator

sim = MonteCarloSimulator(n_paths=50_000, seed=42)   # drift="zero" by default
# drift="historical" projects the in-sample mean; drift="risk_free" 2% a year

# Standard GBM
gbm = sim.simulate_gbm(portfolio, horizon_days=252)
print(f"P(Loss):      {gbm.prob_loss:.1%}")
print(f"E[terminal]:  {gbm.expected_terminal:.3f}x")
print(gbm.percentile_table.to_string(index=False))

# Multivariate Student-t (df=5 by default; df must be > 2): one shared
# chi-square mixing variable per day, so the tails survive diversification
t_dist = sim.simulate_t_distribution(portfolio, horizon_days=252, df_t=5)

# Non-parametric circular block bootstrap: every historical day equally likely
boot = sim.simulate_bootstrap(portfolio, horizon_days=252, block_size=20)

print(f"GBM       VaR 99%: {gbm.var_99:.2%}")
print(f"Student-t VaR 99%: {t_dist.var_99:.2%}")
print(f"Bootstrap VaR 99%: {boot.var_99:.2%}")

# Monte Carlo sampling error of each estimate
print(f"GBM VaR 95%: {gbm.var_95:.2%} ± {gbm.standard_errors['var_95']:.2%} (1 s.e.)")
```

Drift is zero by default (no expected return projected; `historical` and
`risk_free` are explicit choices, and `result.drift` records it). All three
models rebalance to the fixed weights daily. Over HTTP,
`/portfolio/montecarlo` accepts `n_paths` from 1,000 and an optional `seed`
(default 42, echoed back), and returns the same `standard_errors`.

## 5. Stress testing

```python
from riskcanvas.scenarios.stress_test import StressTester

tester = StressTester(beta_to_spy=0.8)  # the portfolio's beta to the S&P 500
results = tester.run_all(portfolio)     # sorted worst-first
for r in results:
    print(f"{r.scenario_name[:34]:34s} portfolio={r.portfolio_drawdown:.1%}  "
          f"market={r.scenario_drawdown:.1%}")

# A single named scenario
covid = tester.run_scenario(portfolio, "covid_crash_2020")

# A custom shock: an S&P 500 decline of 40% spread geometrically over 45
# trading days (duration_days=1 is a one-day shock), scaled by the beta (0.8)
custom = tester.custom_shock(portfolio, equity_shock=-0.40,
                             duration_days=45, name="Custom Crash")
print(custom.portfolio_drawdown)   # beta-scaled; custom.scenario_drawdown is -0.40
```

`portfolio_drawdown` is the scaled path's max drawdown and `total_loss` its
end-of-window return; every built-in path ends at its trough, so at beta 1
both equal the documented drawdown. `duration_days` is in trading days.

Available scenario keys: `black_monday_1987`, `dotcom_crash`, `gfc_2008`,
`covid_crash_2020`, `rate_hike_2022`, `european_debt_crisis`.

The scenarios are **equity-market shocks only**: no interest-rate, credit or
FX factor, so a bond or other low- or negative-beta portfolio barely moves
(or gains) in all of them, "Fed Rate Hike Bear Market (2022)" included.
`beta_fit(portfolio_returns, market_returns)` returns `(beta, r_squared)`,
and `scenario_fit_warnings(beta, r_squared)` lists the reasons a result does
not represent the portfolio (`low_equity_beta` below beta 0.3,
`weak_equity_fit` below R² 0.3); `/portfolio/stress` returns them as
`warnings`, with `beta_r_squared`.

Each scenario is an S&P 500 path; the portfolio's path is `beta × path`,
compounded daily. `StressTester()` with no argument applies the market-like
ASSUMPTION `beta = 1.0`, the only beta that reproduces the documented S&P 500
drawdowns exactly; `beta_to_spy=0` means zero exposure. At any other beta
the result is an approximation: compounding beta-scaled daily moves carries a
volatility drag that depends on the path's roughness, and the synthetic paths
are smoother than the index, so below beta 1 the engine is somewhat more
severe than the same position on the real `^GSPC` window and above beta 1
less severe (by up to 3.2 points at beta 2; README "Stress Scenarios" has the
table, measured 2026-11-01). Estimate beta rather than assume it:

```python
# From SPY prices fetched via yfinance (estimated=False means the 1.0 fallback)
beta, estimated = StressTester().estimate_beta_detailed(portfolio)
```

Both legs are compounded over the sessions both traded, so an asset on another
calendar (7-day, Sunday-Thursday) is measured over the same intervals as SPY.
`information_ratio` and `factor_exposure` do the same with a benchmark or factor
on a denser calendar than the portfolio. `beta_from_returns` takes two series you
have already aligned: pass it returns over the same intervals.

```python
# Or from a market series you already have, aligned with the portfolio
# (synthetic here; e.g. SPY's daily returns on the same dates)
from riskcanvas.scenarios.stress_test import beta_from_returns
spy_daily_returns = np.random.default_rng(3).normal(0.0004, 0.01, len(portfolio.portfolio_returns))
beta = beta_from_returns(portfolio.portfolio_returns.values, spy_daily_returns)

tester = StressTester(beta_to_spy=beta)
```

## 6. REST API

Start the service:

```bash
uvicorn riskcanvas.api.app:app --reload --port 8004
# interactive OpenAPI docs at http://localhost:8004/docs
```

### Health & scenarios

```bash
curl http://localhost:8004/health
curl http://localhost:8004/scenarios
```

### Full analysis

```bash
curl -X POST http://localhost:8004/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{"weights": {"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}, "start": "2020-01-01"}'
```

Statistics that are undefined for the input are JSON `null`: a ratio with a
zero denominator, and any correlation involving a zero-variance series (a cash
leg, a constant series).

### Value at Risk (with a portfolio value)

```bash
curl -X POST "http://localhost:8004/portfolio/var?portfolio_value=1000000" \
  -H "Content-Type: application/json" \
  -d '{"weights": {"SPY": 0.6, "AGG": 0.4}, "start": "2018-01-01"}'
```

Every portfolio endpoint also returns `period`, `n_observations` and
`warnings`: `short_history_tail` when a historical VaR/CVaR level has fewer
than 1/(1 - c) observations (10 / 20 / 100 for 90 / 95 / 99%), `short_history`
below 20 observations, the calendar warnings (`calendar_day_horizon`,
`non_daily_horizon`, `non_standard_calendar`, `sparse_calendar`; README
"Annualisation follows the series' calendar"), `non_synchronous_closes` when the
tickers close in different regions (read from their suffixes, a heuristic; beta,
volatility, VaR and stress losses then read low), and on `/portfolio/analyze` and
`/portfolio/var` `implausible_statistics` for a VaR outside 0 to 100%, a CAGR
above +1,000% a year or annual volatility above 100%. `returns` are read as daily
unless `periods_per_year` says otherwise: send 12 for monthly returns or 52 for
weekly, and the annual figures and the observations' dates follow it. Left unstated,
`/portfolio/analyze` warns `assumed_periods_per_year` (and `/portfolio/montecarlo`
under `drift=risk_free`): its annual figures assumed 252, which reads a crypto series'
volatility 17% low. Every portfolio endpoint warns `ignored_fields` for a body field
or query parameter it does not take, naming it.

The report has one entry per confidence level (`90%`, `95%`, `99%`): `var_1d`,
`cvar_1d` (historical, 1-day), `var_10d` (`var_1d × √10`), and
`var_parametric_1d` / `cvar_parametric_1d` (normal, 1-day), in currency units
at full floating-point precision (not rounded); `portfolio_value=1` gives
fractions of the portfolio.

### Stress test

```bash
curl -X POST http://localhost:8004/portfolio/stress \
  -H "Content-Type: application/json" \
  -d '{"weights": {"SPY": 0.6, "AGG": 0.4}, "start": "2018-01-01"}'
```

The response reports the beta that scaled the scenarios: `beta_used`,
`beta_source` (`explicit`, `market_returns`, `spy_leg`, `yfinance_spy`, or
`assumed` when no market series was available and 1.0 was assumed), a
`beta_note`, `beta_r_squared` (estimated betas only) and `warnings`
(`low_equity_beta`, `weak_equity_fit`, `assumed_beta`, `non_synchronous_closes`): the
scenarios are equity-market shocks only. Offline, pass `"beta": 0.8` or a `"market_returns": [...]` series
aligned with `returns`.

### Monte Carlo (method = gbm | t | bootstrap)

```bash
curl -X POST "http://localhost:8004/portfolio/montecarlo?n_paths=10000&horizon_days=252&method=t" \
  -H "Content-Type: application/json" \
  -d '{"weights": {"AAPL": 0.5, "NVDA": 0.5}, "start": "2021-01-01"}'
```

Request body schema (`PortfolioRequest`):

- `weights` (required, `{ticker: weight}`) — long-only: finite, `>= 0`,
  positive and finite sum; normalised to sum to 1. Negative weights, and
  weights whose sum overflows floating point, are rejected (422), by the same
  `check_weights` the offline loaders use.
- `returns` (optional, `{ticker: [daily returns]}`) — exactly one series per
  weighted ticker, equal lengths, 2–20,000 finite values in [-1, 10]; runs
  offline, observations dated as business days from `start`, the last on or
  before today (a 422 otherwise); `positive_months` uses those assumed
  dates.
- `periods_per_year` (optional, 1 to 365, only with `returns`) — how many observations
  make a year, for a series that is not daily (12 for monthly, 52 for weekly). The
  annual figures, the observations' dates, CAGR's years and the percent and levels
  checks use it. Default: read as daily, on business days, so 252.
- `allow_large_returns` (optional, default `false`) — accept a `returns` or
  `market_returns` series that reads as percent or as levels (see
  [above](#from-a-csv-of-dated-returns)), a 422 otherwise.
- `start` (ISO date string, default `"2020-01-01"`), `end` (optional ISO date
  string after `start`, checked even though `returns` does not use it). Without
  `returns`: at most 20 tickers.
- `max_gap_days` (optional, 1 to 366) — without `returns`, the longest gap in
  prices to accept, as above; one over 12 days is flagged as `long_gap_spanned`.
- Types are strict: `true`, `"0.5"`, or a number given as a date are a 422,
  never coerced.
- A field not listed here, or a query parameter the endpoint does not take, is
  ignored, not refused, and named in `warnings` as `ignored_fields`.
- A body that is not JSON is a 422 (`json_invalid`) too. One that cannot even be decoded,
  because it is not UTF-8 or is nested past the parser's limit, is FastAPI's 400.
- `beta` (optional, `|beta| <= 4`), `market_returns` (optional, aligned with
  `returns`) — stress-test beta inputs.
- `name` (default `"Portfolio"`, at most 200 characters).

Query parameters:

- `/portfolio/var`: `portfolio_value` (default 1,000,000; above 0, at most 1e15).
- `/portfolio/montecarlo`: `n_paths` (default 10,000; 1,000 to 100,000),
  `horizon_days` (default 252; 1 to 2,520), `method` (`gbm` by default, `t` or
  `bootstrap`, lower case), `drift` (`zero` by default, `historical` or
  `risk_free`) and `seed` (default 42; 0 to 2^32 - 1).

`/portfolio/montecarlo` also rejects `n_paths × horizon_days × n_assets`
above 25,000,000 (`gbm`) or 12,500,000 (`t`), so the defaults (10,000 paths over 252
days) fit 9 assets under `gbm` and 4 under `t`, and the `422` names the largest `n_paths`
that fits yours; the bootstrap needs at least 20
observations. Full list: [SECURITY.md](../SECURITY.md).

## 7. Web demo

```bash
cd frontend
npm install
npm run dev      # http://localhost:3004
```

The dashboard's inputs are synthetic return series (seeded one-factor model,
not market data). With the API running, the real engine computes every number
from them; without it, the page shows in-browser estimates. No market-data
keys are needed either way.
