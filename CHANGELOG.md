# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `warnings` code `ignored_fields` on every portfolio endpoint. It names a body field or
  query parameter the endpoint does not take. Such names are still ignored, not refused,
  but a misspelt `periods_per_yeer` no longer passes in silence.
- `warnings` code `assumed_periods_per_year` on `/portfolio/analyze`, and on
  `/portfolio/montecarlo` under `drift=risk_free`. It fires when undated `returns` are
  read at 252 a year only because `periods_per_year` is not stated. Read that way, a
  crypto series shows 17% less annual volatility than it has.
- The dashboard's social-preview tags include `og:url` when `NEXT_PUBLIC_SITE_URL` is set,
  so a shared link's card resolves to the demo itself.

### Changed

- CI runs the test suite on the exact pins in `requirements.txt` (Python 3.12) on every push,
  beside the matrix on the newest versions. The pins were last tested by hand.

### Removed

- `lucide-react`, a frontend dependency nothing imported.

### Fixed

- The dashboard's Portfolio Value field is 16px on phones and wherever the main pointer is
  a finger (an iPhone held sideways, an iPad), so iOS Safari no longer zooms the page
  when it is tapped. It is 14px only from 640px wide with a fine pointer, such as a mouse.
- The Return Distribution's axis ticks are round returns one step apart (1, 2 or 5 times a
  power of ten, in percent), evenly spaced, and no label hangs past either end of the
  plot. They were bucket midpoints, and the last was kept however close it fell: at 375px
  the axis read `-7.5% -5.0% -2.5% 0.0% 4.6%`.
- The Monte Carlo Fan's day axis steps by a trading month, quarter or half year (21, 63
  or 126 days) from day 0, the smallest step whose labels stay 16px apart, so its ticks
  are evenly spaced. It labelled the engine's checkpoint days, keeping the last however
  close it fell: `0 20 45 71 96 121 146 171 197 222 252` at 1440px. The axis is numeric
  now, so each band point sits at its own day rather than at an even index (the engine's
  checkpoints are 5 or 6 days apart).
- The Portfolio Allocation panel's controls take a finger on phones and wherever the main
  pointer is one: each weight slider is 44px tall, each remove button 44px square, and the
  add and preset buttons 44px tall. They were 24px and 28px, so a press 16px above or
  below a slider's centre line landed on nothing. The sliders keep their length, and from
  640px wide with a fine pointer the panel is as before.

- `returns` with `periods_per_year` 12 (or any rate dividing 12) are dated whole months
  from `start`. A series starting on the 29th to 31st used to stick on the 28th after the
  first short month, so its `period` ended up to three days early.
- The README and `docs/usage.md` said any body that is not UTF-8 is a `400`. UTF-16 and
  UTF-32 bodies are decoded like UTF-8, and only bytes none of them decode are a `400`.

## [0.1.0] — first public release

2026-11-04

### What ships

- **Python library.** Historical and parametric VaR and CVaR at 90/95/99%, square-root-of-time
  10-day VaR, drawdown, Sharpe, Sortino, Calmar, Omega, CAGR and an OLS `factor_exposure`
  helper. Portfolios load from yfinance, a returns DataFrame or a CSV.
- **Monte Carlo.** GBM, multivariate Student-t and a circular block bootstrap, each with
  percentile bands and the sampling standard error of every estimate.
- **Stress tests.** Six crisis scenarios, S&P 500 paths calibrated to drawdowns derived from
  cited index closes and scaled by the portfolio's beta, plus custom shocks.
- **REST API.** FastAPI endpoints `/portfolio/analyze`, `/var`, `/stress`, `/montecarlo`,
  `/scenarios` and `/health`, with bounded inputs (see `SECURITY.md`).
- **Web demo.** A Next.js 15 dashboard that sends synthetic return series to the engine, or
  shows in-browser estimates when no engine is connected, and says which.
- **Proof.** A pytest suite and `make smoke`, a driver that runs the real API and a real
  browser and asserts the documented drawdowns over HTTP and on the page.

### Behaviour to know

- Monte Carlo drift is **zero by default**; `historical` and `risk_free` are explicit choices,
  echoed in every response.
- Expected shortfall is the discrete Acerbi-Tasche estimator, so ties at the VaR cannot dilute it.
- Annual figures use the observations per year the dates imply. `returns` over HTTP are read as
  daily unless `periods_per_year` says otherwise.
- Weights are long-only and normalised to sum to 1. A missing value, an unknown ticker, or a
  series that reads as percent or as levels is an error, never filled, dropped or rescaled.
- `warnings` flags thin histories, non-standard calendars, implausible statistics, weak equity
  fit and markets that close at different times; the figures are still returned.
- An undefined statistic is JSON `null`, never a `500`.

### Known limits

- **Equity-market shocks only.** No interest-rate, credit, FX or commodity factor, so a bond or
  other low-beta portfolio barely moves in any scenario, the 2022 one included.
- **Stress paths are calibrated and synthetic, not replays.** Only beta 1 reproduces the table
  exactly; other betas are approximations, understating a high-beta loss by up to 3.2 points
  at beta 2. Daily values are stylized, so they do not serve daily VaR or liquidity analysis.
- **Long-only, one currency.** No shorts or leverage, and no FX conversion.
- **Rebalanced daily.** Every model holds fixed weights; buy-and-hold drift is not modelled.
- **No volatility clustering in GBM or Student-t.** The block bootstrap resamples real days.
- **Standard errors are simulation noise only**, not the estimation error of a finite history.
- **The heuristics can miss.** A cash-like series in percent, prices under 0.10, or monthly
  returns read as daily below the warning lines are not caught.
- **The API has no authentication, rate limit or body-size limit.** Read `SECURITY.md` before
  exposing it.
- **Not financial advice.** Research and demo software, with no warranty.
