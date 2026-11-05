"""
Crisis stress-test scenarios.

Six DETERMINISTIC, calibrated SYNTHETIC daily-return paths for the S&P 500,
one per documented market crisis. They are not replays of historical prices.

**Equity-market shocks only.** Every scenario is an S&P 500 path scaled by
the portfolio's equity beta. There is no interest-rate, credit-spread, FX or
commodity factor: a portfolio moves in a scenario only through its beta to
the S&P 500. A bond, gold or other low- or negative-beta portfolio is barely
moved, or gains, in every scenario, including "Fed Rate Hike Bear Market
(2022)", whose bond losses are not modelled. :func:`scenario_fit_warnings`
names the portfolios this cannot represent (the API returns them as
``warnings``).

What each path is
-----------------
* **Calibrated statistic.** The path's maximum drawdown, measured from
  pre-crisis capital (the 1.0 baseline is the first peak), equals the
  crisis's S&P 500 close-to-close peak-to-trough drawdown
  (``s&p500_drawdown``, the README table) to ~1e-6.
* **Length.** One step per trading session after the peak close, through the
  trough close (``trading_days``).
* **Ends at its trough**, as the real window does (it runs from the peak
  close to the trough close). So at beta 1 the end-of-window return
  (``total_loss``) and the max drawdown (``portfolio_drawdown``) are the same
  number, and both equal the table.
* **Shape vs magnitude.** Each path is a hand-written stylized SHAPE (phases
  of drift, crash days, relief rallies) multiplied by ONE factor ``k`` that
  ``_calibrated`` fits by bisection to hit the drawdown. Scaling moves every
  day by the same factor, so daily values are not the index's daily moves.
  Two shapes are written so that ``k`` is close to 1 and their crash day
  matches the real one-day drop (tests enforce both): Black Monday's
  1987-10-19 (-20.47%) and COVID's 2020-03-16 (-11.98%). The dot-com, GFC,
  2011 and 2022 paths are smooth grinds whose daily volatility is a fraction
  of the real index's in those windows (README table, measured 2026-11-01).
  Use the scenarios for drawdown magnitude and duration, not for daily
  volatility, daily VaR or liquidity analysis. The paths never depend on the
  analyzed portfolio; only the beta scaling applied to them does.
* **Beta scaling is exact only at beta 1.** The portfolio's path is ``beta *``
  the market path, compounded daily (a daily-rebalanced beta position).
  Compounding carries a volatility drag of about ``(beta - beta**2) *
  sum(r**2) / 2`` in log terms, and the calibration fixes the beta-1 drawdown,
  not the path's roughness. The smooth paths therefore make beta < 1 slightly
  more severe and beta > 1 less severe than the same position on the real
  index path: up to 3.2 points at beta 2 (dot-com, GFC; README table and
  ``tests/test_stress.py``, measured 2026-11-01).

Provenance: every ``peak_close`` / ``trough_close`` / date / day count below
was re-verified on 2026-10-30 against Yahoo Finance ``^GSPC`` daily closes
(via ``yfinance``); ``s&p500_drawdown`` is ``trough_close / peak_close - 1``
rounded to 0.1 point, and a unit test enforces that relationship. The two
pinned crash days were re-verified the same way on 2026-11-01.

Scenarios included:
- Black Monday 1987
- Dot-com crash 2000-2002
- Global Financial Crisis 2007-2009
- European Debt Crisis 2011
- COVID Crash 2020
- 2022 Rate Hike Bear Market
- Custom user-defined shocks (an exact geometric market decline, beta-scaled)

Beta
----
A scenario is a path for the S&P 500. A portfolio's path is ``beta * r`` per
day, where ``beta`` is the portfolio's sensitivity to the S&P 500. Pass it to
:class:`StressTester` explicitly (``0`` means ``0``), or estimate it from data
with :func:`beta_from_returns` / :meth:`StressTester.estimate_beta_detailed`.
``None`` means the stated market-like assumption ``beta = 1.0``:
``StressTester.beta_assumed`` is then ``True`` and running a scenario logs a
warning, and callers that surface results (the API) must say when that
assumption is in force.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from riskcanvas.metrics.risk import compound_onto

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# |beta| above this would push Black Monday's crash day (about -20.5%) past a
# -100% one-day loss under linear scaling, which is meaningless for an
# unlevered-loss model. tests/test_stress.py checks every path against it.
MAX_ABS_BETA = 4.0

# Fewest overlapping daily observations a beta estimate is computed from.
MIN_BETA_OBSERVATIONS = 20

# Below either threshold an equity-beta scenario cannot represent the
# portfolio (scenario_fit_warnings). Heuristics, stated as such: under 0.3 the
# S&P 500 path moves the portfolio by less than a third of the market's move
# (negative: it gains); under R^2 0.3 the S&P 500 explains less than 30% of
# its daily variance, so most of its risk is outside the scenario.
LOW_EQUITY_BETA = 0.3
LOW_R_SQUARED = 0.3


def beta_from_returns(portfolio_returns, market_returns) -> float:
    """OLS beta of a portfolio's daily returns on the market's.

    ``cov(p, m) / var(m)``, both with the sample (``ddof=1``) normaliser so a
    series regressed on itself is exactly 1.0. No clamping: the estimate is
    returned as computed. :func:`beta_fit` also returns the fit's R^2.

    Raises
    ------
    ValueError
        Different lengths, fewer than ``MIN_BETA_OBSERVATIONS`` observations,
        non-finite values, or a zero-variance market series (beta undefined).
    """
    return beta_fit(portfolio_returns, market_returns)[0]


def beta_fit(portfolio_returns, market_returns) -> tuple[float, float]:
    """``(beta, r_squared)`` of the OLS fit of portfolio on market daily
    returns: the :func:`beta_from_returns` estimate, and the share of the
    portfolio's daily variance the market explains (the squared correlation,
    0 for a constant portfolio series). Same validation and errors as
    :func:`beta_from_returns`."""
    p = np.asarray(portfolio_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    if p.shape != m.shape or p.ndim != 1:
        raise ValueError(
            "portfolio and market return series must be 1-D and the same "
            f"length (got {p.shape} and {m.shape})"
        )
    if p.size < MIN_BETA_OBSERVATIONS:
        raise ValueError(
            f"beta needs at least {MIN_BETA_OBSERVATIONS} observations "
            f"(got {p.size})"
        )
    if not (np.isfinite(p).all() and np.isfinite(m).all()):
        raise ValueError("return series contain non-finite values")
    var_m = float(np.var(m, ddof=1))
    # Same degeneracy threshold as RiskMetrics: a constant series' float
    # variance is ~1e-36, not exactly 0, and would yield an absurd beta.
    if math.sqrt(var_m) < 1e-12:
        raise ValueError("market return series has (near-)zero variance; beta is undefined")
    cov_pm = float(np.cov(p, m, ddof=1)[0, 1])
    var_p = float(np.var(p, ddof=1))
    r_squared = 0.0 if math.sqrt(var_p) < 1e-12 else min(1.0, cov_pm * cov_pm / (var_m * var_p))
    return cov_pm / var_m, r_squared


def scenario_fit_warnings(beta: float, r_squared: float | None = None) -> list[dict[str, str]]:
    """Warnings for a portfolio the equity-beta scenarios cannot represent.

    Each warning is ``{"code": ..., "message": ...}``:

    * ``low_equity_beta`` — ``beta < LOW_EQUITY_BETA`` (0.3): the S&P 500
      path barely moves the portfolio, or (negative beta) makes it gain.
    * ``weak_equity_fit`` — ``r_squared < LOW_R_SQUARED`` (0.3), when the
      beta was estimated: the S&P 500 explains little of the portfolio's
      daily variance.

    Empty for an equity-like portfolio. The scenarios carry no rate, credit
    or FX factor, so for a flagged portfolio the result says nothing about
    its rate, credit, FX or idiosyncratic risk.
    """
    out = []
    if beta < LOW_EQUITY_BETA:
        moves = "gains" if beta < 0 else "barely moves"
        out.append({
            "code": "low_equity_beta",
            "message": (
                f"Beta to the S&P 500 is {beta:.2f}, so the portfolio {moves} in "
                "every scenario. The scenarios are equity-market shocks only (no "
                "interest-rate, credit or FX factor) and do not capture this "
                "portfolio's rate or credit risk, e.g. 2022's bond losses."
            ),
        })
    if r_squared is not None and r_squared < LOW_R_SQUARED:
        out.append({
            "code": "weak_equity_fit",
            "message": (
                f"The S&P 500 explains {r_squared:.0%} of this portfolio's daily "
                f"variance (R^2 {r_squared:.2f}): an equity-beta scenario captures "
                "only that part; its rate, credit, FX and idiosyncratic risk are "
                "not modelled."
            ),
        })
    return out


def _phase(days: int, drift: float, wobble: float = 0.0, period: int = 5) -> list[float]:
    """Deterministic phase of a stylized path: constant daily drift plus a
    fixed-period wobble day (relief rally or selloff cluster) so the path
    has market-like texture instead of a straight line."""
    return [drift + (wobble if i % period == period - 1 else 0.0) for i in range(days)]


def _max_drawdown(returns: list[float]) -> float:
    """Max drawdown of a return path, measured from pre-crisis capital
    (the 1.0 baseline counts as the first peak)."""
    equity = np.cumprod([1.0] + [1.0 + r for r in returns])
    peak = np.maximum.accumulate(equity)
    return float(((equity - peak) / peak).min())


def _calibrated(shape: list[float], target_dd: float) -> list[float]:
    """Scale a stylized shape by a single factor k (bisection) so the
    compounded path's max drawdown equals ``target_dd`` (negative)."""
    lo, hi = 0.0, 1.0
    while _max_drawdown([hi * r for r in shape]) > target_dd:
        hi *= 2.0
        if hi > 16 or any(hi * r <= -0.99 for r in shape):
            raise ValueError("shape cannot reach target drawdown")
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _max_drawdown([mid * r for r in shape]) > target_dd:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2.0
    return [k * r for r in shape]


# Verified crisis parameters (Yahoo Finance ^GSPC daily closes, 2026-10-30).
# start_date/end_date are the S&P 500 peak and trough CLOSE dates;
# calendar_days is the calendar span between them and trading_days the number
# of sessions after the peak up to and including the trough — the length of
# each ``shock_returns`` path, and what ``ScenarioResult.duration_days`` (and
# the API's ``duration_days``) reports.
SCENARIOS: dict[str, dict[str, Any]] = {
    "black_monday_1987": {
        "name": "Black Monday (Oct 1987)",
        "description": (
            "S&P 500 fell 20.5% on Oct 19, 1987, its largest one-day drop, "
            "inside a 33.5% peak-to-trough decline (Aug 25 to Dec 4)."
        ),
        "start_date": "1987-08-25",
        "end_date": "1987-12-04",
        "peak_close": 336.77,
        "trough_close": 223.92,
        "s&p500_drawdown": -0.335,
        "calendar_days": 101,
        "trading_days": 71,
        # 37-session slide (about -16%, as Aug 26-Oct 16 was), the crash day
        # (about -20.5%), the Oct 20-26 whipsaw, a relief rally of about +15%
        # and a fade into the Dec 4 retest low, which is the trough.
        "shock_returns": _calibrated(
            _phase(37, -0.0060, 0.008, 6)
            + [-0.2047]
            + [0.053, 0.091, -0.039, 0.0, -0.083]
            + _phase(14, 0.0065, -0.010, 5)
            + _phase(14, -0.0070, 0.006, 6),
            -0.335,
        ),
    },
    "dotcom_crash": {
        "name": "Dot-com Crash (2000-2002)",
        "description": (
            "Technology bubble burst. S&P 500 fell 49.1% peak to trough; "
            "the NASDAQ Composite fell 77.9%."
        ),
        "start_date": "2000-03-24",
        "end_date": "2002-10-09",
        "peak_close": 1527.46,
        "trough_close": 776.76,
        "s&p500_drawdown": -0.491,
        "calendar_days": 929,
        "trading_days": 637,
        # Slow bleed, persistent grind with failed rallies, capitulation.
        "shock_returns": _calibrated(
            _phase(200, -0.0008, 0.003, 9)
            + _phase(300, -0.0013, 0.004, 11)
            + _phase(137, -0.0016, -0.006, 17),
            -0.491,
        ),
    },
    "gfc_2008": {
        "name": "Global Financial Crisis (2008-2009)",
        "description": "Subprime mortgage crisis. S&P 500 fell 56.8% peak to trough.",
        "start_date": "2007-10-09",
        "end_date": "2009-03-09",
        "peak_close": 1565.15,
        "trough_close": 676.53,
        "s&p500_drawdown": -0.568,
        "calendar_days": 517,
        "trading_days": 355,
        # Drift down, a Lehman-style panic week, volatile grind, then the
        # early-2009 capitulation leg.
        "shock_returns": _calibrated(
            _phase(120, -0.0009, 0.003, 8)
            + [-0.030, -0.047, -0.020, -0.062, -0.045]
            + _phase(150, -0.0022, 0.005, 9)
            + _phase(80, -0.0030, -0.008, 13),
            -0.568,
        ),
    },
    "covid_crash_2020": {
        "name": "COVID-19 Crash (Feb-Mar 2020)",
        "description": (
            "One of the fastest 30%+ declines on record: S&P 500 -33.9% in "
            "23 trading days (33 calendar days)."
        ),
        "start_date": "2020-02-19",
        "end_date": "2020-03-23",
        "peak_close": 3386.15,
        "trough_close": 2237.40,
        "s&p500_drawdown": -0.339,
        "calendar_days": 33,
        "trading_days": 23,
        # Round-number stylized days, sized like the crisis's moves (the
        # real window's daily sigma was 5.0%): calm, a first leg down, failed
        # relief rallies, a panic of crash days and violent rebounds, and a
        # final leg into the trough on the last day. Written so the fitted
        # k is ~0.99: the -12% crash day stays within 0.5 point of the real
        # 2020-03-16 close (-11.98%; a test enforces it).
        "shock_returns": _calibrated(
            [-0.002] * 3
            + [-0.030, -0.030, -0.010, -0.040, -0.010]
            + [0.045, -0.030, 0.040, -0.035, -0.020, -0.075]
            + [0.050, -0.050, -0.095, 0.090, -0.120, 0.060]
            + [-0.050, -0.040, -0.030],
            -0.339,
        ),
    },
    "rate_hike_2022": {
        "name": "Fed Rate Hike Bear Market (2022)",
        "description": (
            "Fed tightening bear market. S&P 500 fell 25.4% peak to trough. "
            "An equity-market path only: the scenario carries no interest-rate "
            "shock, so bonds are moved only through their equity beta."
        ),
        "start_date": "2022-01-03",
        "end_date": "2022-10-12",
        "peak_close": 4796.56,
        "trough_close": 3577.03,
        "s&p500_drawdown": -0.254,
        "calendar_days": 282,
        "trading_days": 195,
        # The real window's three legs, as smooth stylized phases: a
        # 113-session stair-step decline (to the June low), a 42-session
        # bear-market rally, and a 40-session final leg into the trough on
        # the last day.
        "shock_returns": _calibrated(
            _phase(113, -0.0030, 0.008, 8)
            + _phase(42, 0.0040, -0.009, 7)
            + _phase(40, -0.0055, 0.010, 9),
            -0.254,
        ),
    },
    "european_debt_crisis": {
        "name": "European Debt Crisis (2011)",
        "description": (
            "Greek/Euro sovereign debt crisis and the US downgrade. S&P 500 "
            "fell 19.4% in five months (Apr 29 to Oct 3, 2011)."
        ),
        "start_date": "2011-04-29",
        "end_date": "2011-10-03",
        "peak_close": 1363.61,
        "trough_close": 1099.23,
        "s&p500_drawdown": -0.194,
        "calendar_days": 157,
        "trading_days": 108,
        # Drift, an August-2011-style panic cluster, choppy base.
        "shock_returns": _calibrated(
            _phase(60, -0.0015, 0.004, 7)
            + [-0.045, -0.020, -0.047, 0.043, -0.066]
            + _phase(43, -0.0018, 0.005, 9),
            -0.194,
        ),
    },
}


def _recovery_days(portfolio, end_return: float) -> int | None:
    """Trading days to regain the pre-crisis level from where the scenario ENDS
    (``end_return``, its ``total_loss``), compounding at the portfolio's own
    historical geometric mean daily return ``g`` (its CAGR, per day):
    ``log(1 / (1 + end_return)) / log(1 + g)``.

    A linear rule, ``|dd| / mean daily return``, understates deep drawdowns badly
    (a -50% drawdown needs +100%): for the GFC it gives about 3,300 days where
    compounding at the portfolio's growth gives about 7,500. It starts from the end
    of the window, not the max drawdown, so at beta -0.5 a scenario ending at +19.6%
    has nothing to recover. ``None``
    when not estimable: non-positive geometric growth (or a total-loss day in
    the history), a window that ends at or above the pre-crisis level (nothing
    to recover), a total loss, or a growth rate so small the quotient overflows."""
    with np.errstate(divide="ignore"):
        log_growth = float(np.log1p(portfolio.portfolio_returns.to_numpy()).mean())
    if not (log_growth > 0 and -1.0 < end_return < 0):
        return None
    days = -math.log1p(end_return) / log_growth
    return int(days) if math.isfinite(days) else None


@dataclass
class ScenarioResult:
    """Result of applying a stress scenario to a portfolio.

    ``recovery_estimate_days`` is ``None`` — never a sentinel number — whenever
    a recovery time is not estimable; see :meth:`StressTester.run_scenario`.
    """

    scenario_name: str
    portfolio_drawdown: float      # max drawdown of the beta-scaled path, from pre-crisis capital
    scenario_drawdown: float       # documented S&P 500 peak-to-trough drawdown
    duration_days: int             # trading days: the path's length
    daily_losses: list[float]      # the beta-scaled daily returns
    total_loss: float              # end-of-window return of the beta-scaled path
    # Trading days to regain the pre-crisis level from the end of the window,
    # compounding at the portfolio's historical geometric mean daily return; None
    # when that is not estimable (non-positive growth, or the window ends at or
    # above the pre-crisis level).
    recovery_estimate_days: int | None

    def __repr__(self) -> str:
        return (
            f"ScenarioResult({self.scenario_name!r} | "
            f"portfolio={self.portfolio_drawdown:.1%} | "
            f"market={self.scenario_drawdown:.1%})"
        )


class StressTester:
    """
    Applies the calibrated synthetic crisis scenarios to a portfolio.

    Each scenario's S&P 500 path is scaled by the portfolio's beta to the
    S&P 500 (``beta * r`` per day). The tester does not estimate beta on its
    own: supply it, or estimate it first with :meth:`estimate_beta_detailed`
    (downloads SPY) or :func:`beta_from_returns` (from series you already
    have).

    Parameters
    ----------
    beta_to_spy : float | None
        Portfolio beta to the S&P 500. ``0`` means zero market sensitivity
        (every scenario drawdown is 0). ``None`` applies the market-like
        ASSUMPTION ``beta = 1.0``, which reproduces each scenario's documented
        S&P 500 drawdown: ``beta_assumed`` is then ``True``, the first scenario
        run logs a warning, and anything presenting results must say so. Must be
        finite with ``|beta| <= MAX_ABS_BETA``.

    Examples
    --------
    >>> beta, estimated = StressTester().estimate_beta_detailed(portfolio)
    >>> tester = StressTester(beta_to_spy=beta)
    >>> for r in tester.run_all(portfolio):
    ...     print(r)
    """

    def __init__(self, beta_to_spy: float | None = None) -> None:
        self.beta_assumed = beta_to_spy is None
        self._assumption_logged = False
        if beta_to_spy is None:
            self.beta = 1.0
            return
        beta = float(beta_to_spy)
        if not math.isfinite(beta) or abs(beta) > MAX_ABS_BETA:
            raise ValueError(
                f"beta must be a finite number with |beta| <= {MAX_ABS_BETA} "
                f"(got {beta_to_spy!r})"
            )
        self.beta = beta

    def _log_assumed_beta(self) -> None:
        """Say once, when a result is first computed, that its beta is no estimate."""
        if self.beta_assumed and not self._assumption_logged:
            self._assumption_logged = True
            logger.warning(
                "StressTester: no beta given, so beta 1.0 is ASSUMED; these are the S&P 500's "
                "own drawdowns, not this portfolio's. Pass beta_to_spy."
            )

    def estimate_beta(self, portfolio) -> float:
        """Estimate portfolio beta vs SPY from historical returns.

        Returns the estimated beta, or the neutral fallback of ``1.0`` when it
        cannot be computed (yfinance unavailable/offline, empty download, too
        little overlapping history, or a degenerate return series). Use
        :meth:`estimate_beta_detailed` when the caller needs to know *whether*
        the value is a real estimate or the fallback.
        """
        return self.estimate_beta_detailed(portfolio)[0]

    def estimate_beta_detailed(self, portfolio) -> tuple[float, bool]:
        """Estimate portfolio beta vs SPY: ``(beta, estimated)``, i.e.
        :meth:`estimate_beta_fit` without the R^2."""
        beta, _r_squared, estimated = self.estimate_beta_fit(portfolio)
        return beta, estimated

    def estimate_beta_fit(self, portfolio) -> tuple[float, float | None, bool]:
        """Estimate portfolio beta vs SPY, reporting whether it is genuine.

        Returns ``(beta, r_squared, estimated)``; ``r_squared`` is the fit's
        (see :func:`beta_fit`), ``None`` with the 1.0 fallback.

        Downloads SPY over the portfolio's date range via yfinance and
        computes :func:`beta_from_returns` over the sessions both traded, each
        leg compounded from one such session to the next, so both span the
        same intervals whatever the portfolio's calendar. (Paired date by date,
        a 7-day asset's 1-day Monday return against SPY's 3-day one would read a
        true beta of 1.00 as 0.72.) The estimate is returned unclamped.

        Returns
        -------
        (beta, estimated) : tuple[float, bool]
            ``beta`` is the estimate, or ``1.0`` on any expected failure.
            ``estimated`` is ``True`` only when ``beta`` was computed from
            data; ``False`` marks the neutral fallback so an API/consumer can
            surface it instead of silently presenting a market-like
            assumption as a measurement.

        Notes
        -----
        Only *expected* failure modes are caught and folded into the fallback:
        yfinance missing (``ImportError``), the download failing or returning
        an unusable frame (network/HTTP errors, ``KeyError``), and
        empty/misaligned/degenerate return data (``IndexError``,
        ``ValueError``, ``ZeroDivisionError``, ``FloatingPointError``). Any
        other exception propagates rather than masquerading as beta=1.0.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.warning(
                "estimate_beta: yfinance not installed; falling back to "
                "beta=1.0 (market-like assumption, not a measurement)."
            )
            return 1.0, None, False

        try:
            # yfinance's `end` is exclusive: ask for the day after the last
            # price, or the last day drops out of the fit.
            spy = yf.download(
                "SPY",
                start=portfolio.prices.index[0],
                end=portfolio.prices.index[-1] + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:  # yfinance surfaces network/HTTP/parse errors
            # as bare Exceptions/URLError subclasses; treat any download failure
            # as an expected offline/rate-limited condition, not a code bug.
            logger.warning(
                "estimate_beta: SPY download failed (%s: %s); falling back to "
                "beta=1.0.", type(exc).__name__, exc
            )
            return 1.0, None, False

        try:
            if spy.empty:
                logger.warning(
                    "estimate_beta: SPY download returned no rows; falling back "
                    "to beta=1.0."
                )
                return 1.0, None, False
            # Not .squeeze(): one row would come back as a bare float.
            spy_close = spy["Close"]
            if isinstance(spy_close, pd.DataFrame):  # yfinance's (field, ticker) columns
                spy_close = spy_close.iloc[:, 0]
            spy_close = spy_close.dropna()
            port_returns = portfolio.portfolio_returns
            # The sessions both have a level on; the portfolio's first price
            # date is the level its first return starts from.
            port_dates = port_returns.index.union(portfolio.prices.index[:1])
            common = port_dates.intersection(spy_close.index)
            intervals = max(len(common) - 1, 0)
            if intervals < MIN_BETA_OBSERVATIONS:
                logger.warning(
                    "estimate_beta: only %d overlapping days with SPY (<%d "
                    "required); falling back to beta=1.0.",
                    intervals, MIN_BETA_OBSERVATIONS,
                )
                return 1.0, None, False
            market = spy_close.loc[common].pct_change().iloc[1:]
            port = compound_onto(port_returns, common).loc[common[1]:]
            beta, r_squared = beta_fit(port.values, market.values)
            return beta, r_squared, True
        except (KeyError, IndexError, ValueError, ZeroDivisionError,
                FloatingPointError) as exc:
            logger.warning(
                "estimate_beta: could not compute beta from returns (%s: %s); "
                "falling back to beta=1.0.", type(exc).__name__, exc
            )
            return 1.0, None, False

    def run_scenario(
        self,
        portfolio,
        scenario_key: str,
    ) -> ScenarioResult:
        """
        Apply a single stress scenario.

        Parameters
        ----------
        portfolio : Portfolio
        scenario_key : str
            Key from SCENARIOS dict.

        Returns
        -------
        ScenarioResult

        Notes
        -----
        ``portfolio_drawdown`` is the beta-scaled path's max drawdown from
        pre-crisis capital (the statistic the scenarios are calibrated on) and
        ``total_loss`` its end-of-window return. Every path ends at its
        trough, so at beta 1 both equal the documented drawdown; at other
        betas compounding can make them differ slightly.

        ``recovery_estimate_days`` is a deliberately rough figure: the trading
        days the portfolio would take to regain its pre-crisis level from the
        end of the window, compounding at its own historical geometric mean
        daily return (``log(1 / (1 + total_loss)) / mean(log(1 + r))``). It
        assumes that growth resumes at once, which no crisis guarantees.

        It is ``None``, never a sentinel number, when no honest estimate
        exists:

        * **geometric mean daily return <= 0** — a portfolio with
          non-positive historical growth has no finite recovery time, and roughly half of
          realistic return samples land here from ordinary sampling noise
          alone. Reporting a number here (or a bare ``-1``, which reads as a
          negative day count) would be fabricating one.
        * **the window ends at or above the pre-crisis level**
          (``total_loss >= 0``) — nothing to recover, even if it dipped on the
          way, as a negative-beta portfolio does.
        * **a subnormal drift** — the quotient overflows to infinity.

        Over the wire this serializes to JSON ``null``; the dashboard renders
        it as "N/A".
        """
        if scenario_key not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_key!r}. "
                             f"Available: {list(SCENARIOS.keys())}")

        self._log_assumed_beta()
        s = SCENARIOS[scenario_key]
        market_dd = s["s&p500_drawdown"]

        # Every scenario carries a deterministic, drawdown-calibrated path.
        market_returns = list(s["shock_returns"])

        # Scale the S&P 500 path by the portfolio's beta (0 means no exposure),
        # day by day: exact against the table at beta 1 only (module notes).
        port_returns = [r * self.beta for r in market_returns]

        # Compute drawdown
        portfolio_dd = _max_drawdown(port_returns)
        total_loss = float(np.prod([1 + r for r in port_returns]) - 1.0)

        # Recovery estimate: time to recover lost capital at the portfolio's own
        # mean daily return. None (not a sentinel) when that is not estimable —
        # see the Notes in this method's docstring.
        return ScenarioResult(
            scenario_name=s["name"],
            portfolio_drawdown=portfolio_dd,
            scenario_drawdown=market_dd,
            duration_days=len(port_returns),
            daily_losses=port_returns,
            total_loss=total_loss,
            recovery_estimate_days=_recovery_days(portfolio, total_loss),
        )

    def run_all(self, portfolio) -> list[ScenarioResult]:
        """Run all built-in scenarios and return sorted by portfolio impact."""
        # No per-scenario catch-all: a failing scenario is a bug and must
        # surface, not silently shrink the result list.
        results = [self.run_scenario(portfolio, key) for key in SCENARIOS]
        return sorted(results, key=lambda x: x.portfolio_drawdown)

    def custom_shock(
        self,
        portfolio,
        equity_shock: float,
        duration_days: int = 30,
        name: str = "Custom Shock",
    ) -> ScenarioResult:
        """
        Apply a user-defined S&P 500 decline, scaled by this tester's beta.

        The market path is an exact geometric decline: ``duration_days``
        equal daily returns that compound to precisely ``equity_shock`` (no
        noise). ``duration_days=1`` is a one-day (instantaneous) shock. As in
        the built-in scenarios, the portfolio's path is ``beta *`` the market
        path, so ``portfolio_drawdown`` equals ``equity_shock`` only at
        beta 1; ``scenario_drawdown`` is the market's ``equity_shock``.

        Parameters
        ----------
        portfolio : Portfolio
            Supplies the drift for ``recovery_estimate_days`` (same rule as
            :meth:`run_scenario`).
        equity_shock : float
            Fractional S&P 500 move over the shock, > -1.0 (e.g. -0.20).
        duration_days : int
            Trading days the decline is spread over, >= 1.
        name : str
            Scenario label.

        Raises
        ------
        ValueError
            ``equity_shock <= -1``, ``duration_days < 1``, or a beta-scaled
            daily return of -100% or worse (more than a total loss).
        """
        if not math.isfinite(equity_shock) or equity_shock <= -1.0:
            raise ValueError("equity_shock must be finite and > -1.0 (cannot lose more than 100%)")
        if int(duration_days) != duration_days or duration_days < 1:
            raise ValueError(f"duration_days must be a whole number >= 1 (got {duration_days!r})")
        duration_days = int(duration_days)
        market_daily = (1.0 + equity_shock) ** (1.0 / duration_days) - 1.0
        daily = self.beta * market_daily
        if 1.0 + daily <= 0.0:
            raise ValueError(
                f"beta {self.beta} x a {market_daily:.2%} daily market move is a loss of "
                "100% or more in a day; spread the shock over more days or lower beta"
            )
        self._log_assumed_beta()
        returns = [daily] * duration_days

        dd = _max_drawdown(returns)
        total = float(np.prod([1 + r for r in returns]) - 1.0)

        return ScenarioResult(
            scenario_name=name,
            portfolio_drawdown=dd,
            scenario_drawdown=equity_shock,
            duration_days=duration_days,
            daily_losses=returns,
            total_loss=total,
            recovery_estimate_days=_recovery_days(portfolio, total),
        )
