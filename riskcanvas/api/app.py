"""
RiskCanvas FastAPI application.

Endpoints:
  POST /portfolio/analyze   — Full risk analysis for a portfolio
  POST /portfolio/var       — VaR report
  POST /portfolio/stress    — Stress test all scenarios
  POST /portfolio/montecarlo — Monte Carlo simulation
  GET  /scenarios           — List available scenarios
  GET  /health
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import date, datetime, timedelta
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, PrivateAttr, StrictBool, StringConstraints, field_validator, model_validator

from riskcanvas.portfolio.loader import (
    AMERICAS,
    MAX_CLOSURE_GAP_DAYS,
    MAX_DAILY_RETURN,
    MIN_DAILY_RETURN,
    MIN_RETURNS_LEN,
    PortfolioLoader,
    _refuse_non_returns,
    check_weights,
    close_region,
)
from riskcanvas.metrics.risk import (
    CALENDAR_DAYS,
    DAYS_PER_YEAR,
    MAX_LOSS,
    RISK_FREE,
    SPARSE_CALENDAR_MIN_DAYS,
    SPARSE_CALENDAR_SHARE,
    STANDARD_PERIODS,
    TRADING_DAYS,
    RiskMetrics,
    observations_per_year,
    periods_per_year,
)
from riskcanvas.metrics.performance import PerformanceMetrics
from riskcanvas.simulation.monte_carlo import MonteCarloSimulator
from riskcanvas.scenarios.stress_test import (
    MAX_ABS_BETA,
    MIN_BETA_OBSERVATIONS,
    SCENARIOS,
    StressTester,
    beta_fit,
    scenario_fit_warnings,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RiskCanvas API",
    description="Portfolio risk analytics and stress testing",
    version="0.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with type/loc/msg only. FastAPI's default also echoes each error's
    `input`, which for a body-level check is the entire request body."""
    errors = [
        {"type": e.get("type"), "loc": e.get("loc"), "msg": e.get("msg")}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


# ---- Input bounds -----------------------------------------------------------
# Every limit below is enforced (422 on violation) and listed in SECURITY.md.
# The data rules shared with the offline loaders (MIN_RETURNS_LEN,
# MIN_DAILY_RETURN, MAX_DAILY_RETURN, and check_weights) are defined once in
# riskcanvas.portfolio.loader and imported above.
MAX_ASSETS = 100                 # tickers per request
MAX_LIVE_TICKERS = 20            # tickers when prices are fetched via yfinance
MAX_RETURNS_LEN = 20_000         # observations per series
MIN_START_DATE = date(1900, 1, 1)
MAX_TICKER_LEN = 32
MAX_NAME_LEN = 200
MAX_PORTFOLIO_VALUE = 1e15
MAX_GAP_DAYS_LIMIT = 366         # the largest `max_gap_days` a request may ask for
MAX_PERIODS_PER_YEAR = 365       # the largest `periods_per_year`: one observation a day

# Monte Carlo: per-parameter caps plus a total compute budget, so no single
# request can pin a worker for minutes. GBM / Student-t cost scales with
# n_paths * horizon_days * n_assets ("asset-path-days"); Student-t draws cost
# about twice a Gaussian's, hence its smaller budget. The bootstrap resamples
# the portfolio's own series (one value per path-day, vectorised per block),
# so the per-parameter caps alone bound it.
DEFAULT_PATHS = 10_000
DEFAULT_HORIZON_DAYS = 252
MAX_PATHS = 100_000
# Fewest paths per request: the 99% tail then holds at least 10 paths, and
# every estimate is reported with its Monte Carlo standard error.
MIN_PATHS = 1_000
MAX_HORIZON_DAYS = 2_520
MAX_SEED = 2**32 - 1
MC_WORK_BUDGET = {"gbm": 25_000_000, "t": 12_500_000}
MCMethod = Literal["gbm", "t", "bootstrap"]
MCDrift = Literal["zero", "historical", "risk_free"]
BOOTSTRAP_BLOCK_SIZE = 20

# The scenarios are S&P 500 paths, so stress beta is measured against SPY.
MARKET_TICKER = "SPY"

Ticker = Annotated[str, StringConstraints(min_length=1, max_length=MAX_TICKER_LEN)]
# A JSON number and nothing else: pydantic's default also turns `true` into 1.0
# and "0.5" into 0.5, which is a silent rewrite.
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _observation_dates(start: date, n: int, periods: int | None = None) -> pd.DatetimeIndex:
    """The dates ``returns`` mode gives its ``n`` observations: consecutive business
    days, or with a stated rate that many a year: whole months where it divides 12,
    so monthly statistics stay monthly, and evenly spaced days otherwise."""
    if periods is None or periods == TRADING_DAYS:
        return pd.bdate_range(start, periods=n)
    if 12 % periods == 0:
        return pd.date_range(start, periods=n, freq=pd.DateOffset(months=12 // periods))
    return pd.Timestamp(start) + pd.to_timedelta(np.floor(np.arange(n) * DAYS_PER_YEAR / periods), unit="D")


def _latest_start(today: date, n: int, periods: int | None = None) -> date:
    """The last ``start`` whose ``n`` observations end by ``today``."""
    if periods is None or periods == TRADING_DAYS:
        return pd.bdate_range(end=today, periods=n)[0].date()
    if 12 % periods == 0:
        return (pd.Timestamp(today) - pd.DateOffset(months=(n - 1) * (12 // periods))).date()
    return today - timedelta(days=int((n - 1) * DAYS_PER_YEAR / periods))


def _dated_as(periods: int | None) -> str:
    return "business days" if periods is None or periods == TRADING_DAYS else f"{periods} dates a year"


class PortfolioRequest(BaseModel):
    # Set when returns mode dated the series back from today (no `start`).
    _start_derived: bool = PrivateAttr(default=False)
    weights: dict[Ticker, Number] = Field(
        ...,
        description=(
            "{ticker: weight}. Long-only: every weight must be finite and >= 0 "
            "and their sum positive and finite; weights are normalised to sum "
            "to 1. Negative (short) weights, and weights whose sum overflows "
            "floating point, are rejected with 422."
        ),
    )
    start: date = Field(
        date(2020, 1, 1),
        description=(
            "ISO date (YYYY-MM-DD). With `returns`, the date of the first "
            "observation (observations are consecutive business days, or "
            "`periods_per_year` dates a year, and "
            "the last must fall on or before today; calendar statistics such "
            "as `positive_months` use these assumed dates); "
            "without, the start of the yfinance price history. Live mode "
            "never fills prices or drops a ticker: a ticker with no data, "
            "listed more than 7 calendar days after `start`, delisted more "
            "than 7 before `end` (or today, if `end` is omitted), or with a gap of more than "
            f"{MAX_CLOSURE_GAP_DAYS} calendar days inside the window is a 422 naming it (see "
            "`max_gap_days`). Every endpoint reports the window it used as `period` and "
            "`n_observations`."
        ),
    )
    end: date | None = Field(
        None,
        description=(
            "ISO date, after `start` (a 422 otherwise, with or without `returns`). "
            "Price-history end; unused when `returns` is supplied."
        ),
    )
    name: str = Field("Portfolio", max_length=MAX_NAME_LEN)
    returns: dict[Ticker, list[Number]] | None = Field(
        default=None,
        description=(
            "Optional pre-computed returns per asset, daily unless `periods_per_year` "
            "says otherwise, {ticker: [r0, r1, ...]}: "
            "one series per ticker in `weights`, all the same length (aligned "
            "observations). When provided, the engine runs FULLY OFFLINE from "
            "these returns and never touches the network (no yfinance). When "
            "omitted, prices are fetched via yfinance using `start`/`end`."
        ),
    )
    market_returns: list[Number] | None = Field(
        default=None,
        description=(
            "Optional S&P 500 (e.g. SPY) daily returns aligned with `returns`, "
            "used to estimate the portfolio's beta for /portfolio/stress."
        ),
    )
    beta: Number | None = Field(
        default=None,
        ge=-MAX_ABS_BETA,
        le=MAX_ABS_BETA,
        description=(
            "Optional explicit portfolio beta to the S&P 500 for "
            "/portfolio/stress. 0 means no market exposure."
        ),
    )
    max_gap_days: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=MAX_GAP_DAYS_LIMIT,
        description=(
            "Without `returns`, the longest gap in prices to accept, in calendar days, "
            "for an exceptional market closure: inside a history, between common "
            "sessions and at either end of the window. The return across a gap spans "
            f"all of it, and one over {MAX_CLOSURE_GAP_DAYS} days is flagged in `warnings` as "
            f"`long_gap_spanned`. Default: {MAX_CLOSURE_GAP_DAYS} inside the window, which "
            "fits regular closures such as Golden Week and Lunar New Year, and 7 at its "
            "ends. Unused with `returns`."
        ),
    )
    periods_per_year: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=MAX_PERIODS_PER_YEAR,
        description=(
            "Only with `returns`: how many observations make a year, for a series that "
            "is not daily (12 for monthly, 52 for weekly). The annual figures use it, the "
            "observations are dated at that rate, and CAGR's years are n / periods_per_year. "
            "Default: the series is read as daily, on business days, so 252."
        ),
    )
    allow_large_returns: StrictBool = Field(
        default=False,
        description=(
            "With `returns`, accept a series that reads as percent, not fractions "
            "(a typical day of 10-100% over 20 or more observations, both up and "
            "down days), or as levels, gross returns or prices (a median of 10% or "
            "more, at least 95% of 5 or more values above zero); otherwise a 422. "
            "With `periods_per_year` under 252 those floors are larger, by the square "
            "root of the rate. Figures that come out implausible are then flagged in "
            "`warnings`."
        ),
    )

    @field_validator("start", "end", mode="before")
    @classmethod
    def _iso_date_only(cls, v):
        """A ``YYYY-MM-DD`` string: pydantic's default also reads 0 as
        1970-01-01 and a timestamp as a date."""
        if v is None or (isinstance(v, date) and not isinstance(v, datetime)):
            return v
        if isinstance(v, str) and _ISO_DATE.fullmatch(v):
            return v
        raise ValueError(f"must be an ISO date string, YYYY-MM-DD (got {v!r})")

    @field_validator("weights")
    @classmethod
    def _check_weights(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError(
                '`weights` must contain at least one asset, e.g. {"AAPL": 0.6, "MSFT": 0.4}.'
            )
        if len(v) > MAX_ASSETS:
            raise ValueError(
                f"`weights` may contain at most {MAX_ASSETS} assets (received {len(v)})."
            )
        # The library's rule, verbatim: >= 0, positive and finite sum (the API
        # always normalises, so any positive scale such as percentages is fine).
        check_weights(v)
        return v

    @field_validator("returns")
    @classmethod
    def _check_returns(
        cls, v: dict[str, list[float]] | None
    ) -> dict[str, list[float]] | None:
        if v is None:
            return v
        if not v:
            raise ValueError(
                "`returns` is empty: it needs one series per ticker in `weights`. "
                "Omit it to fetch prices instead."
            )
        if len(v) > MAX_ASSETS:
            raise ValueError(
                f"`returns` may contain at most {MAX_ASSETS} assets (received {len(v)})."
            )
        lengths = {ticker: len(series) for ticker, series in v.items()}
        for ticker, n in lengths.items():
            if n > MAX_RETURNS_LEN:
                raise ValueError(
                    f"Each `returns` series may contain at most {MAX_RETURNS_LEN} "
                    f"observations (series {ticker!r} has {n})."
                )
            if n < MIN_RETURNS_LEN:
                raise ValueError(
                    f"Each `returns` series must contain at least {MIN_RETURNS_LEN} "
                    f"observations (series {ticker!r} has {n})."
                )
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"All `returns` series must have the same length (aligned daily "
                f"observations); got {lengths}."
            )
        for ticker, series in v.items():
            bad = [r for r in series if not MIN_DAILY_RETURN <= r <= MAX_DAILY_RETURN]
            if bad:
                raise ValueError(
                    f"Daily returns must lie in [{MIN_DAILY_RETURN}, {MAX_DAILY_RETURN}] "
                    f"(fractions, e.g. -0.02 for -2%); series {ticker!r} has {bad[0]!r}."
                )
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> "PortfolioRequest":
        if self.start < MIN_START_DATE:
            raise ValueError(f"`start` must be on or after {MIN_START_DATE.isoformat()}.")
        if self.end is not None and self.end <= self.start:
            raise ValueError("`end` must be after `start`.")
        if self.returns is None:
            if self.periods_per_year is not None:
                raise ValueError(
                    "`periods_per_year` describes `returns`; without them the dates of "
                    "the prices decide. Supply both or neither."
                )
            if self.start > date.today():
                raise ValueError("`start` must not be in the future.")
            if len(self.weights) > MAX_LIVE_TICKERS:
                raise ValueError(
                    f"Without `returns`, prices are downloaded per ticker, so at most "
                    f"{MAX_LIVE_TICKERS} tickers are accepted (received "
                    f"{len(self.weights)}). Supply `returns` for larger portfolios."
                )
            if self.market_returns is not None:
                raise ValueError(
                    "`market_returns` must be aligned with `returns`; supply both or neither."
                )
            return self
        today = date.today()
        if self.start > today:
            raise ValueError(
                "`start` must not be in the future: with `returns` it is the date of "
                "the first observation."
            )
        n_obs = len(next(iter(self.returns.values())))
        rate = self.periods_per_year
        if rate is not None and (n_obs - 1) / rate > today.year - MIN_START_DATE.year:
            raise ValueError(
                f"{n_obs} observations at {rate} a year span {(n_obs - 1) / rate:,.0f} years, "
                f"more than lie between {MIN_START_DATE.isoformat()} and today."
            )
        last_day = _observation_dates(self.start, n_obs, rate)[-1].date()
        if last_day > today and "start" not in self.model_fields_set:
            # No `start` given and the default would date the series into the
            # future: date it backwards so it ends today (reported in
            # `warnings` as start_derived_from_today).
            self.start = _latest_start(today, n_obs, rate)
            self._start_derived = True
            last_day = _observation_dates(self.start, n_obs, rate)[-1].date()
        if last_day > today:
            latest_start = _latest_start(today, n_obs, rate)
            raise ValueError(
                f"With `returns`, the {n_obs} observations are dated as consecutive "
                f"{_dated_as(rate)} from `start` ({self.start.isoformat()}), ending "
                f"{last_day.isoformat()}, after today ({today.isoformat()}): returns "
                f"cannot be in the future. Pass `start` on or before "
                f"{latest_start.isoformat()}."
            )
        if set(self.returns) != set(self.weights):
            missing = sorted(set(self.weights) - set(self.returns))
            extra = sorted(set(self.returns) - set(self.weights))
            raise ValueError(
                "`returns` must have exactly one series per ticker in `weights`"
                + (f"; missing series for {missing}" if missing else "")
                + (f"; no weight for {extra}" if extra else "")
                + "."
            )
        if self.market_returns is not None:
            n = len(next(iter(self.returns.values())))
            if len(self.market_returns) != n:
                raise ValueError(
                    f"`market_returns` must have the same length as the `returns` "
                    f"series ({n}); got {len(self.market_returns)}."
                )
            bad = [r for r in self.market_returns
                   if not MIN_DAILY_RETURN <= r <= MAX_DAILY_RETURN]
            if bad:
                raise ValueError(
                    f"`market_returns` values must lie in [{MIN_DAILY_RETURN}, "
                    f"{MAX_DAILY_RETURN}]; got {bad[0]!r}."
                )
            if not self.allow_large_returns:
                _refuse_non_returns(
                    "market_returns", np.asarray(self.market_returns, dtype=float), rate or TRADING_DAYS
                )
        return self


def _build_portfolio(request: PortfolioRequest, loader: PortfolioLoader):
    """
    Build a Portfolio from a validated request.

    With ``request.returns``, run fully offline via ``from_returns_df`` (no
    network): the series are taken as consecutive business days starting at
    ``request.start``, or dated at ``request.periods_per_year``. Otherwise fetch historical prices from yfinance via
    ``from_tickers``, which keeps every ticker and never fills a price (a
    ticker it cannot cover is a ``ValueError``, so a 422). Either way the
    loader normalises the long-only weights to sum to 1 with ``check_weights``,
    the only transformation applied to them (see PortfolioRequest).
    """
    weights = dict(request.weights)

    if request.returns is not None:
        tickers = list(request.weights)
        n = len(request.returns[tickers[0]])
        idx = _observation_dates(request.start, n, request.periods_per_year)
        returns_df = pd.DataFrame(
            {t: request.returns[t] for t in tickers}, index=idx
        ).astype(float)
        return loader.from_returns_df(
            returns_df, weights=weights, name=request.name,
            allow_large_returns=request.allow_large_returns,
            periods_per_year=request.periods_per_year,
        )

    return loader.from_tickers(
        weights,
        request.start.isoformat(),
        request.end.isoformat() if request.end else None,
        request.name,
        max_gap_days=request.max_gap_days,
    )


def _window(portfolio) -> dict:
    """The data window every endpoint actually used: ``period`` (first →
    last price date) and ``n_observations`` (daily returns). In live mode it
    can be shorter than ``start``/``end`` asked for only by the calendar
    tolerance (weekends, holidays); anything more is a 422."""
    idx = portfolio.prices.index
    return {"period": f"{idx[0].date()} → {idx[-1].date()}", "n_observations": len(portfolio.returns)}


# A historical tail at confidence c holds at least one observation only from
# 1 / (1 - c) observations on. Mean / volatility / covariance estimates
# (parametric VaR, Monte Carlo, the stress recovery drift) are flagged below
# the 20 observations a beta estimate requires (MIN_BETA_OBSERVATIONS).
TAIL_MIN_OBSERVATIONS = {0.90: 10, 0.95: 20, 0.99: 100}


def _history_warnings(n: int, levels: tuple[float, ...] = ()) -> list[dict[str, str]]:
    """``{code, message}`` warnings for a history too short for a statistic:
    ``short_history_tail`` (a historical VaR/CVaR level in ``levels`` with
    fewer than ``TAIL_MIN_OBSERVATIONS`` observations) and ``short_history``
    (fewer than ``MIN_BETA_OBSERVATIONS`` for any estimate). The figures are
    still returned; the warning says how little they rest on."""
    out = []
    short = [c for c in levels if n < TAIL_MIN_OBSERVATIONS[c]]
    if short:
        needs = ", ".join(f"{c:.0%} needs {TAIL_MIN_OBSERVATIONS[c]}" for c in short)
        out.append({
            "code": "short_history_tail",
            "message": (
                f"Only {n} observations: the historical VaR/CVaR at "
                f"{', '.join(f'{c:.0%}' for c in short)} rest on fewer than one "
                f"observation in the tail ({needs}); treat them as indicative."
            ),
        })
    if n < MIN_BETA_OBSERVATIONS:
        out.append({
            "code": "short_history",
            "message": (
                f"Only {n} observations: the mean, volatility and covariance "
                f"estimates behind this result rest on fewer than "
                f"{MIN_BETA_OBSERVATIONS} and are unreliable."
            ),
        })
    return out


def _start_warnings(request: PortfolioRequest) -> list[dict[str, str]]:
    """Every date adjustment returns mode made to ``start``, see
    :func:`_moved_start_warnings`, plus ``start_derived_from_today`` when no
    ``start`` was given and the default (2020-01-01) would have dated the
    series past today, so it was dated backwards to end today instead."""
    out = []
    if request._start_derived:
        n = len(next(iter(request.returns.values())))
        out.append({
            "code": "start_derived_from_today",
            "message": (
                f"No `start` given: the {n} observations are dated as {_dated_as(request.periods_per_year)} "
                f"ending today, from {request.start.isoformat()} (the default "
                "2020-01-01 would have run past today)."
            ),
        })
    return out + _moved_start_warnings(request)


def _moved_start_warnings(request: PortfolioRequest) -> list[dict[str, str]]:
    """``start_moved_to_business_day`` when, with ``returns``, ``start`` is a
    Saturday or Sunday: the observations are dated as business days, so the
    first falls on the next Monday."""
    if request.returns is None or request.start.weekday() < 5:
        return []
    if request.periods_per_year not in (None, TRADING_DAYS):   # not dated as business days
        return []
    first = pd.bdate_range(request.start, periods=1)[0].date()
    return [{
        "code": "start_moved_to_business_day",
        "message": (
            f"`start` {request.start.isoformat()} is a {request.start.strftime('%A')}; "
            f"with `returns` the observations are dated as business days, so the "
            f"first is {first.isoformat()} ({first.strftime('%A')})."
        ),
    }]


def _cap_warnings(rm: RiskMetrics, levels: tuple[float, ...]) -> list[dict[str, str]]:
    """``var_10d_capped`` when the sqrt(10)-scaled VaR hit its total-loss cap
    at any of ``levels``: the scaling has broken down for this series."""
    capped = [c for c in levels if rm.var_historical_scaled(c, 10) >= MAX_LOSS]
    if not capped:
        return []
    return [{
        "code": "var_10d_capped",
        "message": (
            f"The 10-day VaR at {', '.join(f'{c:.0%}' for c in capped)} is capped at a "
            "total loss: sqrt(10) scaling of the 1-day VaR exceeds 100% for this "
            "series, so the figure is the cap, not an estimate. Simulate the horizon "
            "with /portfolio/montecarlo instead."
        ),
    }]


def _gap_warnings(portfolio) -> list[dict[str, str]]:
    """``long_gap_spanned`` when one return spans a gap longer than regular
    closures, which only ``max_gap_days`` lets through: it holds the whole gap's
    move in one observation."""
    idx = portfolio.prices.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return []
    days = (idx[1:] - idx[:-1]).days
    long_gaps = np.flatnonzero(days > MAX_CLOSURE_GAP_DAYS)
    if not len(long_gaps):
        return []
    i = long_gaps[int(np.argmax(days[long_gaps]))]
    return [{
        "code": "long_gap_spanned",
        "message": (
            f"{len(long_gaps)} return(s) span more than {MAX_CLOSURE_GAP_DAYS} calendar days, the "
            f"longest {days[i]} ({idx[i].date()} to {idx[i + 1].date()}): each counts as one "
            "observation and holds the whole gap's move."
        ),
    }]


def _close_warnings(request: PortfolioRequest, beta_source: str | None = None) -> list[dict[str, str]]:
    """``non_synchronous_closes`` when the held tickers, or a beta's S&P 500 series
    and the tickers measured against it, close in different regions, judged from the
    ticker suffixes (:func:`~riskcanvas.portfolio.loader.close_region`). Same-date
    closes are then hours apart. An explicit or assumed beta measures nothing against
    the S&P 500, so only a mix among the tickers themselves counts there."""
    regions = {close_region(t) for t, w in request.weights.items() if w > 0}
    if beta_source in ("market_returns", "spy_leg", "yfinance_spy"):
        regions.add(AMERICAS)
    if len(regions) < 2:
        return []
    return [{
        "code": "non_synchronous_closes",
        "message": (
            f"These series close in different regions ({'; '.join(sorted(regions))}; read from "
            "the ticker suffixes, a heuristic), and each date pairs closes struck hours apart: "
            "what one market does after another has closed reaches it the next day. "
            "Correlations and betas read low, so diversification reads high and volatility, "
            "VaR and stress losses read low. Compound the series over several days first and "
            "send them with `periods_per_year`, or state a `beta`."
        ),
    }]


def _assumed_daily(request: PortfolioRequest) -> bool:
    """Are these returns read as daily only because nothing said otherwise?"""
    return request.returns is not None and request.periods_per_year is None


def _series_warnings(
    request: PortfolioRequest, portfolio, horizon_days: int | None = None, beta_source: str | None = None
) -> list[dict[str, str]]:
    """What the series' dates and listings do to the figures, for every portfolio
    endpoint."""
    stated = request.periods_per_year
    return (
        _calendar_warnings(portfolio, horizon_days, stated)
        + ([] if stated is not None else _gap_warnings(portfolio))
        + _close_warnings(request, beta_source)
    )


# Above this CAGR (+1,000% a year) or annual volatility (100%) the figure is flagged.
# Stated heuristics: a real asset can pass either in a short window, which is why
# this is a warning. Volatility above 100% is what a quiet series in percent reads
# as (a short bond fund: 194%); the loader cannot refuse that one.
IMPLAUSIBLE_CAGR = 10.0
IMPLAUSIBLE_ANNUAL_VOL = 1.0
# Series read as daily because nothing dates them. Together these are what monthly
# returns can read as (true volatility 12% reads 55%, a CAGR of 4% reads +136%), and a
# volatile real asset rarely reaches both over a long window.
NON_DAILY_ANNUAL_VOL = 0.50
NON_DAILY_CAGR = 1.0


def _plausibility_warnings(
    rm: RiskMetrics, levels: tuple[float, ...], cagr: float | None = None, *, assumed_daily: bool = False
) -> list[dict[str, str]]:
    """``implausible_statistics`` when a VaR falls outside [0, 1], the CAGR is
    above ``IMPLAUSIBLE_CAGR`` or the annual volatility above
    ``IMPLAUSIBLE_ANNUAL_VOL``: figures a return series rarely produces, and that
    prices or gross returns the loader could not recognise (prices under 0.10, a
    series under 5 observations), a quiet series in percent, or
    ``allow_large_returns`` let through do. With ``assumed_daily``, returns the
    request did not date, also when volatility is above ``NON_DAILY_ANNUAL_VOL``
    and CAGR above ``NON_DAILY_CAGR`` together. The figures are still returned,
    under the VaR convention (negative: a gain)."""
    odd = []
    vol = float(rm.returns.std() * math.sqrt(rm.periods))
    gains = [f"{c:.0%}" for c in levels if not 0.0 <= rm.var_historical(c) <= MAX_LOSS]
    if gains:
        odd.append(f"historical VaR at {', '.join(gains)} is outside 0 to 100% of the portfolio "
                   "(negative: even that tail of the days is a gain)")
    if cagr is not None and not cagr <= IMPLAUSIBLE_CAGR:
        shown = f"{cagr:.3g}" if math.isfinite(cagr) else "unbounded"
        odd.append(f"CAGR is {shown} (above {IMPLAUSIBLE_CAGR:g}, +{IMPLAUSIBLE_CAGR:.0%} a year)")
    if vol > IMPLAUSIBLE_ANNUAL_VOL:
        odd.append(f"annual volatility is {vol:.0%} (above {IMPLAUSIBLE_ANNUAL_VOL:.0%})")
    elif assumed_daily and cagr is not None and vol > NON_DAILY_ANNUAL_VOL and cagr > NON_DAILY_CAGR:
        odd.append(
            f"annual volatility {vol:.0%} with CAGR {cagr:.0%} (above {NON_DAILY_ANNUAL_VOL:.0%} and "
            f"+{NON_DAILY_CAGR:.0%} together), with the series read as daily"
        )
    if not odd:
        return []
    text = "; ".join(odd)
    return [{
        "code": "implausible_statistics",
        "message": (
            f"{text[0].upper()}{text[1:]}. That fits a cash-like, steadily rising or very "
            "volatile series or a very short history. Otherwise check that the input is daily "
            "fractional returns, not percent, gross returns (1 + r) or prices; for weekly or "
            "monthly returns, state `periods_per_year`."
        ),
    }]


def _calendar_warnings(portfolio, horizon_days: int | None = None, stated: int | None = None) -> list[dict[str, str]]:
    """What the series' calendar does to the figures, for every portfolio
    endpoint. ``horizon_days`` is Monte Carlo's; the others count ``1d`` /
    ``10d`` and recovery days, which are observations of the series too.
    ``stated`` is the request's ``periods_per_year``: nothing about the rate is
    then measured, so only the horizon warnings can apply.

    * ``calendar_day_horizon``: it trades every calendar day (a 7-day asset
      alone), so those count calendar days and the annual figures use 365.
    * ``non_daily_horizon``: fewer than 252 observations a year, so they count
      weeks, months or whatever one observation is.
    * ``non_standard_calendar``: the annual figures use the measured rate
      (see :func:`periods_per_year`).
    * ``sparse_calendar``: a standard calendar with far fewer observations than
      its rate, which stays standard.
    """
    idx = portfolio.returns.index
    periods = periods_per_year(idx) if stated is None else stated
    out = []
    if stated is not None:
        pass
    elif periods not in STANDARD_PERIODS:
        out.append({
            "code": "non_standard_calendar",
            "message": (
                f"These {len(idx)} dates, {idx[0].date()} to {idx[-1].date()}, hold about "
                f"{periods} observations a year, not a standard 252, 365, 52 or 12: annual "
                f"figures use the measured {periods}."
            ),
        })
    elif (
        periods in (TRADING_DAYS, CALENDAR_DAYS)
        and (idx[-1] - idx[0]).days >= SPARSE_CALENDAR_MIN_DAYS
        and observations_per_year(idx) < SPARSE_CALENDAR_SHARE * periods
    ):
        out.append({
            "code": "sparse_calendar",
            "message": (
                f"These dates fall on a {periods}-a-year calendar but hold only about "
                f"{observations_per_year(idx):.0f} observations a year. Annual volatility uses "
                f"{periods}, which counts each observation as one session: right for dates "
                "excluded as missing, and high if the returns span the gaps (an illiquid asset, "
                "or tickers with few common sessions). CAGR uses the dates: right if the returns "
                "span the gaps, and low for dates excluded as missing."
            ),
        })
    if periods == CALENDAR_DAYS:
        count = (
            "the 1d and 10d figures and recovery estimates count calendar days"
            if horizon_days is None else
            f"horizon_days={horizon_days} is {horizon_days} calendar days "
            f"({horizon_days / CALENDAR_DAYS:.2f} years)"
        )
        out.append({
            "code": "calendar_day_horizon",
            "message": (
                f"The series trades every calendar day: {count}, and annual figures use "
                "365 periods a year."
            ),
        })
    elif periods < TRADING_DAYS:
        count = (
            "the 1d and 10d figures and recovery estimates count observations, not days"
            if horizon_days is None else
            f"horizon_days={horizon_days} is {horizon_days} observations "
            f"({horizon_days / periods:.2f} years), not days"
        )
        out.append({
            "code": "non_daily_horizon",
            "message": f"The series has {periods} observations a year: {count}.",
        })
    return out


def _drift_warnings(result, port_returns=None) -> list[dict[str, str]]:
    """``historical_drift_gain`` when the in-sample mean, used as drift,
    makes the 95% VaR negative: the simulated tail is a gain because the
    sample's average return dominates its volatility over the horizon.
    ``drift_removes_extreme_trend`` when the zero / risk-free drift re-centres
    away an in-sample mean larger than half a daily standard deviation (or
    any nonzero mean of a constant series): e.g. a -100%-every-day series
    then simulates zero risk."""
    if result.drift != "historical":
        if port_returns is None:
            return []
        mean, sd = float(np.mean(port_returns)), float(np.std(port_returns, ddof=1))
        if abs(mean) < 1e-12 or (sd > 1e-12 and abs(mean) <= 0.5 * sd):
            return []
        return [{
            "code": "drift_removes_extreme_trend",
            "message": (
                f"The history's mean daily return ({mean:.2%}) is extreme next to its "
                f"daily volatility ({sd:.2%}); drift={result.drift} removes it, so this "
                "simulation does not reflect that trend. Check the input, or pass "
                "drift=historical to project it."
            ),
        }]
    if result.var_95 >= 0:
        return []
    return [{
        "code": "historical_drift_gain",
        "message": (
            f"With the in-sample mean return as drift, the 95% worst outcome over "
            f"{result.horizon_days} days is a gain (VaR95 {result.var_95:.2%}, VaR99 "
            f"{result.var_99:.2%}): the sample's average return dominates its "
            "volatility. The default drift=zero does not project it forward."
        ),
    }]


def _bootstrap_warnings(n_obs: int, horizon_days: int) -> list[dict[str, str]]:
    """``few_bootstrap_windows`` when the history holds fewer than
    ``TAIL_MIN_OBSERVATIONS[0.99]`` (100) independent, non-overlapping
    horizon-length windows (``n_obs // horizon_days``). The bootstrap's
    horizon outcomes are recombinations of overlapping historical windows,
    so its 99% tail then rests on less than one independent observation,
    however many paths are drawn (300 observations at 10 days hold 30 such
    windows)."""
    windows = n_obs // horizon_days
    limit = TAIL_MIN_OBSERVATIONS[0.99]
    if windows >= limit:
        return []
    return [{
        "code": "few_bootstrap_windows",
        "message": (
            f"{n_obs} observations hold only {windows} independent {horizon_days}-day "
            f"window(s): the bootstrap's {horizon_days}-day outcomes recombine that "
            f"little history, so its 99% VaR/CVaR rest on fewer than {limit} "
            "independent observations and can understate the tail. The standard "
            "errors measure simulation noise only, not this estimation error."
        ),
    }]


def _correlation_to_wire(corr: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Correlation matrix as ``{column: {row: value}}``, rounded to 4 dp.

    A pair involving a zero-variance series (a cash leg, any constant series,
    two equal observations) has no defined correlation; pandas returns NaN,
    which is not JSON. Such entries, including the series' own diagonal, are
    serialised as ``null``.
    """
    return {
        str(col): {
            str(row): (round(float(v), 4) if math.isfinite(v) else None)
            for row, v in corr[col].items()
        }
        for col in corr.columns
    }


def _resolve_beta(request: PortfolioRequest, portfolio) -> tuple[float, str, str, float | None]:
    """Decide the beta that scales the stress scenarios, and say where it came from.

    Returns ``(beta, source, note, r_squared)``; ``r_squared`` is the beta
    fit's R^2 for the estimated sources and ``None`` otherwise. ``source`` one of:

    * ``explicit``       — the request's ``beta`` (0 means 0).
    * ``market_returns`` — OLS beta on the request's ``market_returns``.
    * ``spy_leg``        — OLS beta on the portfolio's own SPY series.
    * ``yfinance_spy``   — OLS beta on SPY downloaded for the price window.
    * ``assumed``        — no usable market series: beta 1.0 is ASSUMED, and
      the note says so; the results are then the S&P 500's own drawdowns.
    """
    if request.beta is not None:
        return float(request.beta), "explicit", "Beta supplied by the caller (`beta`).", None

    port = portfolio.portfolio_returns
    if request.market_returns is not None:
        beta, r2 = beta_fit(port.values, np.asarray(request.market_returns, dtype=float))
        return beta, "market_returns", (
            f"OLS beta of the portfolio's daily returns on the supplied "
            f"`market_returns` ({len(port)} observations)."
        ), r2

    why_not = (
        "The request carries no market series (`market_returns`, or a "
        f"`{MARKET_TICKER}` series among the assets)."
    )
    if MARKET_TICKER in portfolio.returns.columns:
        try:
            beta, r2 = beta_fit(port.values, portfolio.returns[MARKET_TICKER].values)
            return beta, "spy_leg", (
                f"OLS beta of the portfolio's daily returns on its own "
                f"{MARKET_TICKER} series ({len(port)} observations)."
            ), r2
        except ValueError as exc:
            why_not = f"The {MARKET_TICKER} series could not support a beta estimate ({exc})."
    elif request.returns is None:
        beta, r2, estimated = StressTester().estimate_beta_fit(portfolio)
        if estimated:
            return beta, "yfinance_spy", (
                f"OLS beta vs {MARKET_TICKER} returns downloaded via yfinance over "
                "the portfolio's price window, both legs compounded over the "
                "sessions both traded."
            ), r2
        why_not = f"{MARKET_TICKER} history could not be downloaded to estimate beta."

    return 1.0, "assumed", (
        f"{why_not} Scenarios are scaled by an ASSUMED beta of 1.0 "
        "(market-like exposure), so these are the S&P 500's own scenario "
        "drawdowns, not a portfolio-specific estimate. Pass `beta` or "
        "`market_returns` for a portfolio-specific result."
    ), None


@app.get("/health")
def health():
    return {"status": "ok", "service": "RiskCanvas", "version": "0.1.0"}


@app.get("/scenarios")
def list_scenarios():
    """The six calibrated synthetic scenarios and the S&P 500 window each is
    calibrated to: peak and trough close dates, the peak-to-trough drawdown,
    the window's length in trading days (= the path length, the stress
    response's ``duration_days``) and in calendar days."""
    return {
        "scenarios": [
            {
                "key": k,
                "name": v["name"],
                "description": v["description"],
                "market_drawdown": v["s&p500_drawdown"],
                "peak_date": v["start_date"],
                "trough_date": v["end_date"],
                "trading_days": v["trading_days"],
                "calendar_days": v["calendar_days"],
            }
            for k, v in SCENARIOS.items()
        ]
    }


@app.post("/portfolio/analyze")
def analyze_portfolio(request: PortfolioRequest):
    """Full risk + performance analysis.

    Every statistic that is undefined for the input — a ratio with a zero
    denominator, or a correlation involving a zero-variance series such as a
    cash leg — is JSON ``null``, never a 500. Sharpe, Sortino and downside
    volatility use the annual ``risk_free_rate`` the response states.
    """
    try:
        loader = PortfolioLoader()
        portfolio = _build_portfolio(request, loader)
        port_returns = portfolio.portfolio_returns
        stated = request.periods_per_year

        rm = RiskMetrics(port_returns, periods=stated)
        risk = rm.compute_all()
        perf = PerformanceMetrics(port_returns, periods=stated).compute_all()

        return {
            "portfolio": request.name,
            "n_assets": portfolio.n_assets,
            **_window(portfolio),
            "warnings": _start_warnings(request) + _history_warnings(len(portfolio.returns), (0.95, 0.99))
            + _cap_warnings(rm, (0.95, 0.99))
            + _series_warnings(request, portfolio)
            + _plausibility_warnings(rm, (0.95, 0.99), perf.cagr, assumed_daily=_assumed_daily(request)),
            "risk_free_rate": RISK_FREE,
            "risk": risk.to_dict(),
            "performance": perf.to_dict(),
            "correlation_matrix": _correlation_to_wire(portfolio.correlation_matrix),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error while analyzing portfolio.")


@app.post("/portfolio/var")
def compute_var(
    request: PortfolioRequest,
    portfolio_value: float = Query(
        1_000_000, gt=0, le=MAX_PORTFOLIO_VALUE, allow_inf_nan=False
    ),
):
    """VaR / CVaR report at 90, 95 and 99% confidence, in currency units of
    ``portfolio_value`` (positive = loss), at full floating-point precision:
    nothing is rounded, so ``portfolio_value=1`` returns fractions of the
    portfolio. Every key names its horizon:

    * ``var_1d`` / ``cvar_1d`` — historical 1-day VaR and CVaR (expected
      shortfall) from the portfolio's daily returns.
    * ``var_10d`` — ``var_1d * sqrt(10)``: square-root-of-time scaling, which
      assumes i.i.d. daily returns with zero mean (a drifting series is
      mis-scaled), capped at ``portfolio_value`` (a total loss; returns are
      >= -100%).
    * ``var_parametric_1d`` / ``cvar_parametric_1d`` — normal (closed-form)
      1-day VaR and CVaR from the sample mean and standard deviation, capped
      at ``portfolio_value``.

    A negative figure is a gain: even the tail quantile is positive (a
    constant +1% series). Historical and parametric follow one formula for
    every series, constant ones included.
    """
    try:
        loader = PortfolioLoader()
        portfolio = _build_portfolio(request, loader)
        port_returns = portfolio.portfolio_returns
        stated = request.periods_per_year
        rm = RiskMetrics(port_returns, periods=stated)

        report = {}
        for conf in [0.90, 0.95, 0.99]:
            report[f"{int(conf*100)}%"] = {
                "var_1d": rm.var_historical(conf) * portfolio_value,
                "var_10d": rm.var_historical_scaled(conf, 10) * portfolio_value,
                "cvar_1d": rm.cvar_historical(conf) * portfolio_value,
                "var_parametric_1d": rm.var_parametric(conf) * portfolio_value,
                "cvar_parametric_1d": rm.cvar_parametric(conf) * portfolio_value,
            }

        return {
            "portfolio_value": portfolio_value,
            **_window(portfolio),
            "warnings": _start_warnings(request)
            + _history_warnings(len(portfolio.returns), (0.90, 0.95, 0.99))
            + _cap_warnings(rm, (0.90, 0.95, 0.99))
            + _series_warnings(request, portfolio)
            + _plausibility_warnings(
                rm, (0.90, 0.95, 0.99), PerformanceMetrics(port_returns, periods=stated).cagr(),
                assumed_daily=_assumed_daily(request),
            ),
            "var_report": report,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"VaR computation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error while computing VaR.")


@app.post("/portfolio/stress")
def stress_test(request: PortfolioRequest):
    """Run all crisis stress scenarios (drawdown-calibrated deterministic paths).

    The scenarios are EQUITY-MARKET shocks only: each scenario's S&P 500 path
    is scaled by the portfolio's beta, with no interest-rate, credit or FX
    factor, so a bond or other low-beta portfolio barely moves (or gains)
    even in "Fed Rate Hike Bear Market (2022)". The beta is resolved by
    ``_resolve_beta`` and reported as ``beta_used`` / ``beta_source`` /
    ``beta_note`` (including when 1.0 is only assumed), with
    ``beta_r_squared`` for an estimated beta. ``warnings`` lists every
    reason the result may not represent the portfolio (``low_equity_beta``,
    ``weak_equity_fit``, ``assumed_beta``, ``non_synchronous_closes``; see
    ``scenario_fit_warnings`` and ``_close_warnings``).

    Per scenario: ``portfolio_drawdown`` is the scaled path's max drawdown
    from pre-crisis capital and ``total_loss`` its end-of-window return (equal
    at beta 1, where both are the README table's ``market_drawdown``);
    ``duration_days`` is the path length in trading days. Only beta 1
    reproduces the table exactly; any other beta is a daily-rebalanced
    beta-scaled approximation whose volatility drag depends on the synthetic
    path's roughness (README "Stress Scenarios" quantifies it).

    `recovery_days_estimate` is the trading days to regain the pre-crisis
    level from the end of the window, compounding at the portfolio's historical
    geometric mean daily return (its CAGR), a rough guide that assumes that
    growth resumes at once. It is `null` when no recovery time is estimable:
    notably when that growth is <= 0, which ordinary sampling noise produces
    often, or when the window ends at or above the pre-crisis level. Never a
    sentinel.
    """
    try:
        loader = PortfolioLoader()
        portfolio = _build_portfolio(request, loader)
        beta, source, note, r_squared = _resolve_beta(request, portfolio)
        try:
            tester = StressTester(beta_to_spy=beta)
        except ValueError:
            raise ValueError(
                f"The estimated beta {beta:.3f} ({source}) is outside the supported "
                f"range |beta| <= {MAX_ABS_BETA}; pass an explicit `beta`."
            ) from None
        results = tester.run_all(portfolio)
        stress_warnings = scenario_fit_warnings(tester.beta, r_squared)
        if source == "assumed":
            stress_warnings.append({
                "code": "assumed_beta",
                "message": (
                    "No market series: beta 1.0 is ASSUMED, so these are the S&P "
                    "500's own drawdowns, not this portfolio's (see beta_note)."
                ),
            })

        return {
            "portfolio": request.name,
            **_window(portfolio),
            "beta_used": round(tester.beta, 6),
            "beta_source": source,
            "beta_note": note,
            "beta_r_squared": None if r_squared is None else round(r_squared, 6),
            "warnings": _start_warnings(request) + stress_warnings
            + _history_warnings(len(portfolio.returns))
            + _series_warnings(request, portfolio, beta_source=source),
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "portfolio_drawdown": round(r.portfolio_drawdown, 4),
                    "market_drawdown": round(r.scenario_drawdown, 4),
                    "duration_days": r.duration_days,
                    "total_loss": round(r.total_loss, 4),
                    "recovery_days_estimate": r.recovery_estimate_days,
                }
                for r in results
            ],
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error while running stress test.")


@app.post("/portfolio/montecarlo")
def monte_carlo(
    request: PortfolioRequest,
    n_paths: int = Query(DEFAULT_PATHS, ge=MIN_PATHS, le=MAX_PATHS),
    horizon_days: int = Query(DEFAULT_HORIZON_DAYS, ge=1, le=MAX_HORIZON_DAYS),
    method: MCMethod = Query(
        "gbm",
        description="`gbm`, `t` (multivariate Student-t, df 5) or `bootstrap`, lower case; "
                    "anything else is a 422.",
    ),
    drift: MCDrift = Query(
        "zero",
        description="Mean daily return simulated: `zero` (default, the risk-desk "
                    "convention), `historical` (the sample mean) or `risk_free` "
                    "(2% a year). Echoed in the response.",
    ),
    seed: int = Query(
        42, ge=0, le=MAX_SEED,
        description="RNG seed. The default makes identical requests return identical "
                    "results; pass another value for an independent sample.",
    ),
):
    """Run Monte Carlo simulation.

    Rejected with 422 before any work is done when GBM / Student-t requests
    exceed ``MC_WORK_BUDGET`` (n_paths * horizon_days * n_assets), which the defaults
    do from 10 assets (GBM) or 5 (Student-t): the message says what fits.

    ``standard_errors`` gives the Monte Carlo sampling error of every
    estimate (``null`` where undefined); ``seed`` echoes the seed used. All
    three models rebalance to the fixed weights daily.
    """
    if method in MC_WORK_BUDGET:
        n_assets = len(request.weights)
        budget = MC_WORK_BUDGET[method]
        work = n_paths * horizon_days * n_assets
        if work > budget:
            fits = budget // (horizon_days * n_assets)
            advice = (
                f"at horizon_days={horizon_days} this portfolio fits n_paths up to {fits:,}"
                if fits >= MIN_PATHS else
                f"at n_paths={MIN_PATHS:,}, the fewest allowed, this portfolio fits horizon_days "
                f"up to {budget // (MIN_PATHS * n_assets):,}"
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Monte Carlo request too large: n_paths * horizon_days * n_assets "
                    f"= {n_paths} * {horizon_days} * {n_assets} = {work:,}, "
                    f"over the {method!r} budget of {budget:,}, which keeps one request from "
                    f"pinning a worker (the defaults, {DEFAULT_PATHS:,} paths over "
                    f"{DEFAULT_HORIZON_DAYS} days, fit {budget // (DEFAULT_PATHS * DEFAULT_HORIZON_DAYS)} "
                    f"assets). Reduce n_paths or horizon_days: {advice}."
                ),
            )
    try:
        loader = PortfolioLoader()
        portfolio = _build_portfolio(request, loader)
        sim = MonteCarloSimulator(n_paths=n_paths, seed=seed, drift=drift, periods=request.periods_per_year)

        if method == "gbm":
            result = sim.simulate_gbm(portfolio, horizon_days)
        elif method == "t":
            result = sim.simulate_t_distribution(portfolio, horizon_days)
        else:
            result = sim.simulate_bootstrap(
                portfolio, horizon_days, block_size=BOOTSTRAP_BLOCK_SIZE
            )

        return {
            **_window(portfolio),
            "n_paths": result.n_paths,
            "horizon_days": result.horizon_days,
            "method": result.method,
            "seed": result.seed,
            "drift": result.drift,
            "var_95": round(result.var_95, 4),
            "var_99": round(result.var_99, 4),
            "cvar_95": round(result.cvar_95, 4),
            "cvar_99": round(result.cvar_99, 4),
            "prob_loss": round(result.prob_loss, 4),
            "expected_terminal": round(result.expected_terminal, 4),
            "standard_errors_note": (
                "Monte Carlo sampling error only: how far another seed would move "
                "each estimate. It excludes estimation error from the finite "
                "history, which is often larger."
            ),
            "standard_errors": {
                k: (round(v, 6) if math.isfinite(v) else None)
                for k, v in result.standard_errors.items()
            },
            "percentile_paths": result.percentile_paths,
            "percentile_bands": [
                {k: (v if k == "day" else round(v, 6)) for k, v in band.items()}
                for band in result.percentile_bands
            ],
            "percentile_table": result.percentile_table.to_dict(orient="records"),
            "warnings": _start_warnings(request) + _history_warnings(len(portfolio.returns))
            + (_bootstrap_warnings(len(portfolio.returns), horizon_days)
               if method == "bootstrap" else [])
            + _drift_warnings(result, portfolio.portfolio_returns.to_numpy())
            + _series_warnings(request, portfolio, horizon_days),
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal error while running Monte Carlo simulation.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "riskcanvas.api.app:app",
        host=os.environ.get("RISKCANVAS_HOST", "127.0.0.1"),
        port=int(os.environ.get("RISKCANVAS_PORT", "8004")),
    )
