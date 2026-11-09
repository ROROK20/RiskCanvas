# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.** Report privately to
**rohan.s.khiani.vora@gmail.com** or through GitHub's
[private vulnerability reporting](https://github.com/ROROK20/RiskCanvas/security/advisories/new),
with the impact and steps to reproduce. You get an acknowledgement within **48 hours** and a
triage within **7 days**. A fix and its disclosure are coordinated with you, with credit in the
release notes unless you prefer otherwise.

## Scope

RiskCanvas is a quantitative analytics library and reference web demo: stateless, storing no
user data. It is **not** investment advice and ships with no warranty (see `LICENSE`). Live
market data comes through `yfinance`; review that project's terms before relying on it. `.env`
is git-ignored and `.env.example` holds only placeholders.

## Dependency audits

Measured 2026-11-04 with `pip-audit` 2.10.1, Node 23.9 and npm 10.9. Advisory databases move
faster than this file, so re-run these rather than trust it.

| Reproduce | Result |
|---|---|
| `pip-audit -r requirements.txt --no-deps --disable-pip` | No known vulnerabilities |
| `cd frontend && npm audit` | 0 critical, 0 high, 6 moderate |
| `cd frontend && npm audit --omit=dev` | 0 vulnerabilities |

- **Python.** The command audits the 12 exact pins, not what they pull in; drop both flags to
  resolve the full tree. `setup.py` keeps `starlette>=1.3.1`, which excludes PYSEC-2026-248 and
  PYSEC-2026-249. A floor binds only at install, so check `pip show starlette` in a long-lived
  environment.
- **The 6 moderates are lint tooling.** One advisory, GHSA-g3hv-75wg-m249 (`minimatch` denial of
  service), through `@typescript-eslint/*`. It runs only on this repository's own files and is
  not in the production install. `npm audit` lists a fix; it is not applied.
- **`postcss`.** Next vendors its own copy, so `frontend/package.json` overrides it with the
  patched 8.5.32. Keep that override when upgrading Next.

## Demo API surface (FastAPI)

**Input bounds.** A violation is a `422` before any computation or download. The constants are
at the top of `riskcanvas/api/app.py`, and `tests/test_api.py` covers them.

- `weights`: 1–100 tickers, each 1–32 characters; finite, `>= 0`, positive and finite sum.
- `returns`: at most 100 series, one per weighted ticker, equal lengths of 2–20,000 finite values
  in [-1, 10], none reading as percent or as levels unless `allow_large_returns` is set.
  A valid body carries at most 2,020,000 return values: 100 series × 20,000, plus 20,000 in `market_returns`.
- `market_returns`: only with `returns`, same length, finite, in [-1, 10]. `beta`: finite,
  `|beta| <= 4`. `name`: at most 200 characters.
- `max_gap_days`: a whole number, 1 to 366. `periods_per_year`: a whole number, 1 to 365, only
  with `returns`; the dates it implies must fit between 1900-01-01 and today.
- `start` / `end`: `YYYY-MM-DD`, `start` from 1900-01-01 and not in the future, `end` after
  `start`; with `returns`, the observations' dates must end by today.
- Types are strict: `true`, `"0.5"` and `start: 0` are a `422`, not coerced.
- An unknown body field or query parameter is ignored, not a `422`, and named in `warnings`
  (`ignored_fields`): at most 5 names, each cut to 40 characters.
- Live-ticker mode (no `returns`): at most **20** tickers, each a sequential yfinance download
  (plus SPY for stress), network-bound and **outside** the budget below. A public deployment
  should prefer `returns` or rate-limit this.
- `/portfolio/var`: `portfolio_value` finite, `0 < value <= 1e15`.
- `/portfolio/montecarlo`: `method` one of `gbm` / `t` / `bootstrap`, `n_paths` 1,000–100,000,
  `seed` 0 to 2^32 − 1, `horizon_days` 1–2,520, **plus a total budget**: `n_paths × horizon_days ×
  n_assets` at most 25,000,000 for `gbm` and 12,500,000 for `t`. The defaults (10,000 paths, 252
  days) fit 9 assets under `gbm` and 4 under `t`; past that the `422` names what fits. The per-parameter
  caps bound the bootstrap, which resamples one portfolio series.
- `422` responses carry each error's `type` / `loc` / `msg` only, never the input.

**Cost of the largest request each limit admits.** Measured 2026-11-03, in-process (`TestClient`,
with JSON encoding and parsing), the cheaper of two runs, on an 8-core laptop at load average ~16.

| Request | CPU | Wall |
|---|---|---|
| `/portfolio/analyze`, 100 series × 20,000 observations | 2.1 s | 2.7 s |
| `/portfolio/stress`, 100 series × 20,000 observations | 1.8 s | 2.4 s |
| `montecarlo` `gbm`, 1 asset × 100,000 paths × 250 days | 0.8 s | 1.3 s |
| `montecarlo` `gbm`, 100 assets × 1,000 paths × 250 days | 0.5 s | 0.5 s |
| `montecarlo` `t`, 1 asset × 100,000 paths × 125 days | 0.7 s | 0.8 s |
| `montecarlo` `t`, 100 assets × 5,000 paths × 25 days | 0.4 s | 0.5 s |
| `montecarlo` `bootstrap`, 100 × 20,000 history, 100,000 paths × 2,520 days | 1.7 s | 2.1 s |
| the dashboard's own `gbm` request, 9 assets × 5,000 paths × 252 days | 0.2 s | 0.2 s |

**What a deployer must add.**

- **A body-size limit.** JSON is parsed before the field caps apply; set one in the reverse proxy.
- **Rate and concurrency limits.** The budget bounds one request, not how many arrive.
- **The one-click deploy has none of those limits.** `render.yaml` serves the `Dockerfile` straight
  to the internet on Render's free plan, with no proxy or authentication and a fraction of that
  laptop's CPU, so a few concurrent requests can saturate it. Before publishing the URL, front it
  with such a proxy, or keep it private and ship the frontend in demo-data mode.
- **Narrower CORS and an API key.** CORS is `*` and the compute endpoints have no authentication,
  for a public, read-only, stateless demo. Restrict `allow_origins` and add a key for anything else.
