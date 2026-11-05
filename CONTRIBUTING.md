# Contributing to RiskCanvas

Thanks for your interest in contributing! RiskCanvas is a Python library, REST
API and demo dashboard for portfolio risk. This guide covers everything you need to get a working dev environment
and submit a high-quality change.

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Prerequisites

- **Python 3.11+** for the analytics core, tests, and the FastAPI service;
  **3.12+** to install the exact pins in `requirements.txt` (numpy 2.5.1 and
  scipy 1.18.0 publish no Python 3.11 wheels)
- **Node.js 18+** and npm (only if you work on the `frontend/` demo)
- **git**

## Getting Started (Backend)

```bash
# 1. Fork and clone
git clone https://github.com/ROROK20/RiskCanvas.git
cd RiskCanvas

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install the package in editable mode with dev extras (pytest + coverage),
#    using the pinned, reproducible dependency set (Python 3.12+; on 3.11,
#    drop `-c requirements.txt` to resolve compatible versions instead)
pip install -e ".[dev]" -c requirements.txt
```

This installs the runtime dependencies (`pandas`, `numpy`, `scipy`, `fastapi`,
`uvicorn`, `pydantic`, `yfinance`) plus `pytest` and `pytest-cov`.

> Tip: the test suite runs on synthetic and stubbed data and needs no network
> access. Only live `from_tickers(...)` calls hit yfinance.

### Running the API locally

```bash
uvicorn riskcanvas.api.app:app --reload --port 8004
# Interactive docs: http://localhost:8004/docs
```

## Running the Test Suite

The full suite must pass before a PR is merged:

```bash
pytest tests/
```

With coverage (matches CI):

```bash
pytest tests/ --cov=riskcanvas --cov-report=term-missing -v
```

## Linting & Type Checking

No Python linter config is enforced, so keep changes clean
and consistent with the existing style. If you have the tools installed locally,
these are the recommended checks:

```bash
# optional, if installed
python -m pyflakes riskcanvas    # catch unused imports / undefined names
python -m compileall riskcanvas  # byte-compile sanity check (always available)
```

For the frontend, ESLint is wired up:

```bash
cd frontend
npm install
npm run lint
```

## Frontend Demo

```bash
cd frontend
npm install
npm run dev
# open http://localhost:3004
```

What CI runs (lint, unit test, production bundle):

```bash
cd frontend
npm run lint && npm test && npm run build
```

The demo's inputs are synthetic return series (`frontend/lib/mockRisk.ts`,
not market data): with the API running the real engine computes everything
from them; without it the dashboard falls back to in-browser estimates. No
market-data keys are required either way.

## Code Style

**Python**
- Target Python 3.11+; use modern type hints (`dict[str, float]`, `X | None`).
- 4-space indentation, descriptive names, module/class/function docstrings.
- Keep numerical code vectorized with NumPy/pandas where practical.
- New metrics or estimators should ship with a unit test asserting against a
  hand-computed or known-property value (see `tests/` for the pattern).

**TypeScript / React**
- Follow the existing functional-component + hooks style.
- `npm run lint` must pass (`next/core-web-vitals`).

## Branch & PR Workflow

1. Create a topic branch off `main`: `git checkout -b feat/my-change`.
2. Make your change with focused, logically-grouped commits.
3. Ensure `pytest tests/` passes (and `npm run lint && npm test && npm run build` if you touched the frontend).
4. Push and open a Pull Request against `main`.
5. Fill out the PR template and link any related issue.

A maintainer will review. Keep PRs scoped — smaller PRs get merged faster.

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(optional scope): <short summary>
```

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.

Examples:

```
feat(metrics): add Omega ratio to RiskMetrics
fix(montecarlo): correct Student-t variance scaling
docs(readme): clarify quick-start install command
test(stress): add COVID scenario drawdown assertion
```

## Reporting Bugs & Requesting Features

Use the GitHub issue templates (Bug Report / Feature Request). For security
issues, **do not** open a public issue — see [SECURITY.md](SECURITY.md).

Thanks for contributing! 🎯
