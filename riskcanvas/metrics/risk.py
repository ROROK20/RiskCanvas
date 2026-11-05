"""
Portfolio risk metrics.

Historical and parametric VaR / CVaR, drawdown, downside deviation and the
Sortino ratio are tested against hand-computed values
(``tests/test_risk_metrics.py``); the other statistics against their defining
formulas or invariants.

Sign convention: VaR and CVaR are losses as a fraction of portfolio value
(positive = loss). One formula serves every series, constant ones included:
VaR is minus the tail quantile, so it is negative when even that quantile is a
gain, the same convention as Monte Carlo VaR. Returns are >= -100% (limited
liability), so no horizon can lose more than everything: extrapolations past
a total loss (square-root-of-time scaling, the unbounded normal tail) are
reported as ``MAX_LOSS`` (1.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252
RISK_FREE = 0.02
MAX_LOSS = 1.0  # a total loss: the most a long-only portfolio can lose
CALENDAR_DAYS = 365  # periods per year of a series that trades every day
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
STANDARD_PERIODS = (MONTHS_PER_YEAR, WEEKS_PER_YEAR, TRADING_DAYS, CALENDAR_DAYS)
DAYS_PER_YEAR = 365.25   # for the measured rate: a span of dates, in years

# Under this median gap (days) the dates are intraday bars; a daylight-saving
# change leaves a daily series at 23 hours (0.96).
_MIN_DAILY_GAP = 0.9
# A weekday counts as traded when it holds at least this share of the busiest
# weekday's observations, so a few special weekend sessions do not turn a
# 5-day market into a 7-day one.
_TRADED_WEEKDAY_SHARE = 0.20
# A daily series with holes (`missing="drop"`, an illiquid asset) still has many
# one-day gaps; every-other-day or twice-weekly data has none.
_MIN_ONE_DAY_GAP_SHARE = 0.25
# Fewer observations than these cannot show a weekly rhythm or an irregular
# spacing: a short daily series is taken as daily, not refused.
_MIN_OBS_FOR_SPACING = 10
_MIN_OBS_FOR_WEEK_COUNT = 14
# How much of its rate a standard calendar must hold, over at least a year of dates
# (a shorter window can sit inside one long closure). A single exchange's holidays
# leave at least FULL_CALENDAR_SHARE of 252, about 240; two exchanges' common sessions
# leave less (New York with Tokyo about 235), and from SPARSE_CALENDAR_SHARE down it
# is holes, not a calendar. Stated heuristics: see periods_per_year.
FULL_CALENDAR_SHARE = 0.95
SPARSE_CALENDAR_SHARE = 0.90
SPARSE_CALENDAR_MIN_DAYS = 365
# Annual dates across a leap year measure just under one a year (366 days between
# two of them); this is how far under still counts as one.
_MIN_ANNUAL_RATE = 0.95
# The most a calendar-month series can measure: three business month-ends across a short
# February, 12.82 a year. Four-weekly dates measure 13.04.
_MAX_MONTHLY_RATE = 12.9


def observations_per_year(index: pd.DatetimeIndex) -> float:
    """The measured rate: intervals between the dates over the years they span."""
    return (len(index) - 1) / ((index.max() - index.min()) / pd.Timedelta(days=DAYS_PER_YEAR))


def _spans_a_year(dates: pd.DatetimeIndex) -> bool:
    return (dates.max() - dates.min()).days >= SPARSE_CALENDAR_MIN_DAYS


def years_spanned(index: pd.DatetimeIndex, periods: int) -> float:
    """The years ``len(index)`` observations cover, the ``n`` in CAGR's
    ``growth ** (1 / n)``. A calendar that holds its standard rate counts one
    session as ``1 / periods`` of a year, so 252 weekday returns are exactly one
    year, as are 252 sessions of a real exchange. Anything else counts the dates:
    ``n`` observations at the measured rate, so one more interval than the span
    between the first and the last, whose return began one interval earlier.
    Fewer than ``_MIN_OBS_FOR_WEEK_COUNT`` dates cannot show a rate."""
    n = len(index)
    if n < _MIN_OBS_FOR_WEEK_COUNT:
        return n / periods
    rate = observations_per_year(index)
    holds_its_rate = rate >= SPARSE_CALENDAR_SHARE * periods or not _spans_a_year(index)
    return n / periods if periods in STANDARD_PERIODS and holds_its_rate else n / rate


def _unmeasurable(dates: pd.DatetimeIndex, gap: float, why: str) -> ValueError:
    return ValueError(
        f"Cannot infer periods per year from these {len(dates)} dates (median gap {gap:g} days, "
        f"from {dates[0].date()} to {dates[-1].date()}): {why} Annual figures need the true "
        "number: pass `periods` to RiskMetrics / PerformanceMetrics, or resample the series."
    )


def periods_per_year(index) -> int:
    """Observations per year, read from the dates' spacing and weekdays. One
    rule, the first line that fits:

    * intraday spacing (median gap under ``_MIN_DAILY_GAP`` days): refused.
      Bars show neither how many sessions a year there are nor what the
      overnight gaps hold, so nothing here can be measured. So is a series of
      fewer than one observation a year (annual dates count as one, leap years
      or not).
    * monthly spacing (median gap 27-32 days, at most ``_MAX_MONTHLY_RATE`` a year):
      ``MONTHS_PER_YEAR`` (12), so that every 28 days reads 13;
    * weekly spacing (6-8 days): ``WEEKS_PER_YEAR`` (52);
    * daily spacing on all seven weekdays: ``CALENDAR_DAYS`` (365);
    * daily spacing on five: ``TRADING_DAYS`` (252), whichever two days the
      market closes (Sunday-Thursday as much as Monday-Friday), except over a
      year or more of dates holding 90-95% of 252 (227-239 a year): what two
      exchanges' common sessions leave, which is measured;
    * fewer than ``_MIN_OBS_FOR_WEEK_COUNT`` daily dates cannot show a week,
      and an undated index or fewer than 3 dates no pattern at all: 252;
    * anything else: the MEASURED rate, :func:`observations_per_year` rounded.
      Four weekdays (Sunday-Thursday and Monday-Friday tickers align on
      Monday-Thursday: about 209), two a week, every other day, a 6-day week,
      quarterly. The API then warns ``non_standard_calendar``.

    A single exchange's holidays do not change a standard calendar's rate, and
    nor do holes below 90% of it: each observation counts as one session, as
    ``missing="drop"`` leaves them. If the returns instead SPAN the holes (an
    illiquid asset) that overstates volatility; the API then warns
    ``sparse_calendar``, and ``periods`` overrides it in the library. CAGR counts
    the dates there (:func:`years_spanned`).

    Raises
    ------
    ValueError
        What cannot be measured (above).

    Why measure: at a flat 252, Monday-Thursday sessions (about 209 a year) would
    report volatility 20.9% for 19.1% and CAGR 21.7% for 17.6%, and a seeded 235
    sessions a year, about what New York with Tokyo leave, volatility 17.01% for
    16.42% and CAGR 6.22% for 5.79%.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return TRADING_DAYS
    dates = index.dropna().sort_values()
    n = len(dates)
    if n < 3:
        return TRADING_DAYS
    gaps = np.asarray((dates[1:] - dates[:-1]) / pd.Timedelta(days=1), dtype=float)
    gap = float(np.median(gaps))
    if gap < _MIN_DAILY_GAP:
        raise _unmeasurable(dates, gap, "intraday bars do not show how many sessions a year there are.")
    if 27 <= gap <= 32 and observations_per_year(dates) <= _MAX_MONTHLY_RATE:
        return MONTHS_PER_YEAR
    if 6 <= gap <= 8:
        return WEEKS_PER_YEAR
    one_day_share = float(((gaps >= 0.9) & (gaps <= 1.1)).mean())
    if one_day_share >= _MIN_ONE_DAY_GAP_SHARE or (gap < 6 and n < _MIN_OBS_FOR_SPACING):
        per_weekday = np.bincount(dates.dayofweek, minlength=7)
        traded = int((per_weekday >= _TRADED_WEEKDAY_SHARE * per_weekday.max()).sum())
        if traded == 7:
            return CALENDAR_DAYS
        if traded == 5 or n < _MIN_OBS_FOR_WEEK_COUNT:
            if n >= _MIN_OBS_FOR_WEEK_COUNT and _spans_a_year(dates) and (
                SPARSE_CALENDAR_SHARE * TRADING_DAYS <= observations_per_year(dates)
                < FULL_CALENDAR_SHARE * TRADING_DAYS
            ):
                return round(observations_per_year(dates))
            return TRADING_DAYS
        if traded == 6:
            # One market that moved its weekend (still five sessions a week),
            # or a real 6-day week, which is measured below.
            per_week = dates.to_series().resample("W").count()
            if float(per_week[per_week > 0].median()) <= 5.5:
                return TRADING_DAYS
    rate = observations_per_year(dates)
    if rate < _MIN_ANNUAL_RATE:
        raise _unmeasurable(dates, gap, "that is fewer than one observation a year.")
    return round(rate)


def compound_onto(returns, dates: pd.DatetimeIndex):
    """Simple returns (a Series or DataFrame) compounded onto ``dates``, sorted
    dates of the same or a sparser calendar, so that series on different
    calendars are measured over the SAME intervals: the value at ``dates[k]``
    is the growth over ``(dates[k-1], dates[k]]``. The value at ``dates[0]`` is
    the return dated there, if any; earlier and later returns are left out.

    A Monday return of a 7-day asset covers one day and a Monday return of a
    Monday-Friday asset three, so pairing them date by date regresses one on
    the other across different intervals: a beta of 1.00 read 0.72.

    On the same calendar the returns come back untouched, not round-tripped
    through ``1 + r``. A NaN inside an interval makes that interval NaN.
    """
    ordered = returns.sort_index()
    within = ordered[(ordered.index >= dates[0]) & (ordered.index <= dates[-1])]
    ends = dates[np.searchsorted(dates, within.index, side="left")]
    if ends.equals(within.index):
        return within
    first = np.flatnonzero(np.r_[True, ends[1:] != ends[:-1]])
    out = within.iloc[first].copy()
    out.iloc[:] = np.multiply.reduceat(1.0 + within.to_numpy(dtype=float), first, axis=0) - 1.0
    out.index = ends[first]
    return out


def _cap_at_total_loss(loss: float) -> float:
    """``min(loss, MAX_LOSS)``; NaN passes through unchanged."""
    return MAX_LOSS if loss > MAX_LOSS else loss


def _tail_percentile(confidence: float) -> float:
    """The percentile (0-100) of the loss tail, ``(1 - confidence) * 100``,
    free of binary rounding: computed raw, 1 - 0.90 gives 9.999999999999998,
    which put the historical CVaR threshold just below the observation that
    IS the 90% VaR and dropped it from the tail (95% rounded the other way and
    kept it)."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1 (got {confidence!r})")
    return round((1.0 - confidence) * 100.0, 10)


def _no_missing(returns: pd.Series) -> pd.Series:
    """The series unchanged, or ``ValueError`` naming its missing values:
    a NaN is never silently dropped (shared by RiskMetrics and
    PerformanceMetrics)."""
    missing = returns.index[returns.isna().to_numpy()]
    if len(missing):
        raise ValueError(
            f"returns has {len(missing)} missing value(s) (first at {missing[0]}); "
            "they are never dropped silently. Remove or fill them yourself first."
        )
    return returns


def expected_shortfall(values, confidence: float) -> float:
    """Expected shortfall of a discrete sample (Acerbi & Tasche, 2002), as a
    positive loss: minus the average of the worst ``k = (1 - confidence) * n``
    outcomes, the boundary outcome weighted by the fractional part of ``k``::

        ES = -(x_(1) + ... + x_(m) + (k - m) * x_(m+1)) / k,   m = floor(k)

    for ``x`` sorted ascending. Unlike "the mean of every value at or below
    VaR", it does not dilute the tail when many values tie at the VaR
    (``[0] * 97 + [-0.10] * 3`` has ES95 0.06, not 0.003), and it is
    continuous in ``confidence``. Shared by historical CVaR, CDaR and every
    Monte Carlo ES."""
    x = np.sort(np.asarray(values, dtype=float))
    k = len(x) * _tail_percentile(confidence) / 100.0
    m = int(math.floor(k))
    total = x[:m].sum() + ((k - m) * x[m] if m < len(x) else 0.0)
    return float(-total / k) + 0.0   # + 0.0: never -0.0 on the wire


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown after each observation, as a fraction (0 or negative).

    Measured against the running peak of an equity curve that STARTS at 1.0,
    the capital held before the first return: a loss on day one is a
    drawdown from that starting capital, exactly as in the stress scenarios.
    (Peaking at the first day's close instead made ``[-30%, +1%, +1%]`` report
    no drawdown at all.)

    Computed in log space, ``expm1(log_equity - running_peak)``, so a -100%
    day gives exactly -1.0 for the rest of the series and long compounding
    never overflows to ``inf`` / ``nan``.

    Raises
    ------
    ValueError
        A return below -1.0 (a loss of more than 100%) is not a return.
    """
    r = np.asarray(returns, dtype=float)
    if (r < -1.0).any():
        raise ValueError("returns below -1.0 (a loss of more than 100%) are not valid returns")
    with np.errstate(divide="ignore"):  # log1p(-1) = -inf: a total loss
        log_equity = np.cumsum(np.log1p(r))
    # 0.0 = log(1.0): the starting capital is the first peak.
    peak = np.maximum(np.maximum.accumulate(log_equity), 0.0)
    return pd.Series(np.expm1(log_equity - peak), index=returns.index)


@dataclass
class RiskReport:
    """Complete risk metrics for a portfolio."""

    # VaR metrics (positive = loss amount as fraction of portfolio)
    var_95_1d: float
    var_99_1d: float
    var_95_10d: float
    var_99_10d: float
    cvar_95: float
    cvar_99: float

    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    current_drawdown: float

    # Volatility
    daily_vol: float
    annual_vol: float
    downside_vol: float            # annualised downside deviation below the rf target

    # Tail risk
    skewness: float
    excess_kurtosis: float
    var_stress_multiplier: float   # ratio of actual to normal VaR
    periods_per_year: int = TRADING_DAYS   # annualisation: see periods_per_year()

    def to_dict(self) -> dict:
        # Non-finite values become None (JSON null), never a 500 on the wire.
        return {
            k: (round(v, 6) if math.isfinite(v) else None)
            for k, v in self.__dict__.items()
        }

    def __repr__(self) -> str:
        return (
            f"RiskReport("
            f"VaR95={self.var_95_1d:.2%}, "
            f"CVaR95={self.cvar_95:.2%}, "
            f"MaxDD={self.max_drawdown:.2%}, "
            f"AnnualVol={self.annual_vol:.2%})"
        )


class RiskMetrics:
    """
    Computes comprehensive risk metrics from a returns series.

    Parameters
    ----------
    returns : pd.Series
        Portfolio returns (fractional), normally daily. The ``1d`` / ``10d``
        figures are 1 and 10 observations of the series: on a weekly or
        monthly series they are 1 and 10 weeks or months.
    risk_free_rate : float
        Annual risk-free rate (default 0.02).
    periods : int | None
        Observations per year for the annual figures. Default: read from the
        dates by :func:`periods_per_year`, which measures a non-standard
        calendar and refuses only what it cannot measure; pass the true number
        for such a series. ``compute_all().periods_per_year`` reports the rate.

    Examples
    --------
    >>> rm = RiskMetrics(portfolio_returns)
    >>> report = rm.compute_all()
    >>> print(report)
    """

    # Below this daily volatility the series is treated as constant /
    # degenerate: dispersion-based metrics (VaR, CVaR, skew, kurtosis) are
    # numerically meaningless and would otherwise emit negative "VaR" or NaN.
    _VOL_EPS = 1e-12

    def __init__(
        self,
        returns: pd.Series,
        risk_free_rate: float = RISK_FREE,
        periods: int | None = None,
    ) -> None:
        self.returns = _no_missing(returns)
        self.rfr = risk_free_rate
        if periods is not None and periods <= 0:
            raise ValueError(f"periods must be positive (got {periods!r})")
        self.periods = periods_per_year(self.returns.index) if periods is None else periods
        self.rfr_daily = risk_free_rate / self.periods

    def compute_all(self) -> RiskReport:
        """Compute all risk metrics.

        Degenerate inputs are handled explicitly so the report is always
        JSON-serialisable (no NaN/inf leaks):

        - An empty returns series raises ``ValueError`` rather than throwing an
          opaque ``IndexError`` from downstream percentile/iloc access.
        - A constant / zero-variance series (daily vol < ``_VOL_EPS``) has
          NaN skew, kurtosis and stress multiplier (undefined; JSON null). Its VaR/CVaR come from the same formulas as any other
          series (a constant ``c`` gives ``-c`` for historical and parametric
          alike), so nothing jumps at the cutoff.
        - ``var_95_10d`` / ``var_99_10d`` are capped at a total loss (see
          :meth:`var_historical_scaled`).
        """
        r = self.returns

        if len(r) == 0:
            raise ValueError("returns series is empty")

        dd_series, max_dd = self.max_drawdown()
        avg_dd = float(dd_series.mean())
        current_dd = float(dd_series.iloc[-1])
        max_dd_dur = self._max_drawdown_duration(dd_series)

        daily_vol = float(r.std())
        annual_vol = float(daily_vol * np.sqrt(self.periods))
        degenerate = daily_vol < self._VOL_EPS

        # Annualised downside deviation below the daily risk-free target over
        # ALL observations — sqrt(mean(min(r - T, 0)^2)) — the same quantity
        # PerformanceMetrics' Sortino divides by. 0.0 when nothing falls short.
        shortfall = np.minimum(r.values - self.rfr_daily, 0.0)
        downside_vol = float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(self.periods))

        var95_1d = self.var_historical(0.95)
        var99_1d = self.var_historical(0.99)
        cvar95 = self.cvar_historical(0.95)
        cvar99 = self.cvar_historical(0.99)

        if degenerate:
            # No dispersion: the standardised moments and the VaR ratio are
            # undefined: NaN (JSON null).
            skewness = float("nan")
            excess_kurtosis = float("nan")
            stress_mult = float("nan")
        else:
            skewness = float(stats.skew(r))
            excess_kurtosis = float(stats.kurtosis(r))
            # Stress multiplier = historical VaR / normal (parametric) VaR.
            # Only defined when both VaRs are positive losses; otherwise the
            # ratio is sign-ambiguous: NaN (it read a "neutral" 1.0).
            var95_normal = self.var_parametric(0.95)
            if abs(var95_normal) > 1e-10 and var95_normal > 0 and var95_1d > 0:
                stress_mult = float(var95_1d / var95_normal)
            else:
                stress_mult = float("nan")

        return RiskReport(
            var_95_1d=var95_1d,
            var_99_1d=var99_1d,
            var_95_10d=self.var_historical_scaled(0.95, 10),
            var_99_10d=self.var_historical_scaled(0.99, 10),
            cvar_95=cvar95,
            cvar_99=cvar99,
            max_drawdown=max_dd,
            max_drawdown_duration_days=max_dd_dur,
            avg_drawdown=avg_dd,
            current_drawdown=current_dd,
            daily_vol=daily_vol,
            annual_vol=annual_vol,
            downside_vol=downside_vol,
            skewness=skewness,
            excess_kurtosis=excess_kurtosis,
            var_stress_multiplier=stress_mult,
            periods_per_year=self.periods,
        )

    def var_historical(self, confidence: float = 0.95) -> float:
        """Historical simulation 1-day VaR: minus the ``1 - confidence``
        quantile of the daily returns (numpy's linear interpolation).
        Positive = loss; negative when even that quantile is a gain (a
        constant +1% series has VaR -1%), as for parametric and Monte Carlo
        VaR."""
        # + 0.0 turns -0.0 (minus a zero quantile) into 0.0.
        return float(-np.percentile(self.returns, _tail_percentile(confidence))) + 0.0

    def var_historical_scaled(self, confidence: float = 0.95, horizon_days: int = 10) -> float:
        """Historical VaR scaled to ``horizon_days`` by the square root of
        time, ``var_historical * sqrt(horizon_days)`` (assumes i.i.d. daily
        returns with ZERO mean: the mean compounds linearly, not by sqrt(t),
        so a series with drift is mis-scaled; a constant +1% a day scales to
        -3.16% where 10 days actually return +10.46%), capped at ``MAX_LOSS``: a long-only portfolio cannot lose
        more than everything, while the scaling alone put a 10-day VaR at 316%
        for a series of near-total daily losses. When the cap binds the
        sqrt-time rule has broken down for that series; simulate the horizon
        instead (``MonteCarloSimulator``)."""
        if int(horizon_days) != horizon_days or horizon_days < 1:
            raise ValueError(f"horizon_days must be a whole number >= 1 (got {horizon_days!r})")
        return _cap_at_total_loss(self.var_historical(confidence) * math.sqrt(horizon_days)) + 0.0

    def var_parametric(self, confidence: float = 0.95) -> float:
        """Parametric (normal) 1-day VaR (positive = loss): ``-(mu + z * sigma)``
        with ``z = Phi^-1(1 - confidence)``, from the sample mean and standard
        deviation of the daily returns. Capped at ``MAX_LOSS``: the normal
        tail is unbounded, a long-only portfolio's loss is not. ``confidence``
        is a fraction: 95 raises, as in :meth:`var_historical`."""
        _tail_percentile(confidence)
        mu = self.returns.mean()
        sigma = self.returns.std()
        z = stats.norm.ppf(1 - confidence)
        return _cap_at_total_loss(float(-(mu + z * sigma))) + 0.0

    def cvar_parametric(self, confidence: float = 0.95) -> float:
        """Parametric (normal) 1-day CVaR / Expected Shortfall (positive = loss).

        Closed form for returns ~ N(mu, sigma^2), tail probability
        ``a = 1 - confidence`` and ``z = Phi^-1(a)``::

            ES = -mu + sigma * phi(z) / a

        (``phi`` is the standard normal density): the mean loss on the days
        beyond the parametric VaR, so always ``>= var_parametric``. Capped at
        ``MAX_LOSS`` like :meth:`var_parametric`.
        """
        _tail_percentile(confidence)
        a = 1.0 - confidence
        mu = float(self.returns.mean())
        sigma = float(self.returns.std())
        z = stats.norm.ppf(a)
        return _cap_at_total_loss(float(-mu + sigma * stats.norm.pdf(z) / a)) + 0.0

    def cvar_historical(self, confidence: float = 0.95) -> float:
        """Historical CVaR / Expected Shortfall of the daily returns: the
        discrete Acerbi-Tasche estimator, :func:`expected_shortfall`. The mean
        of every return at or below VaR would not do: ties at the VaR dilute it
        up to 20x."""
        return expected_shortfall(self.returns.to_numpy(), confidence)

    def max_drawdown(self) -> tuple[pd.Series, float]:
        """
        Compute drawdown series and max drawdown.

        Drawdowns are measured from a running peak that starts at the initial
        capital (1.0), so a first-day loss counts; see :func:`drawdown_series`.

        Returns
        -------
        tuple[pd.Series, float]
            (drawdown series, maximum drawdown as negative fraction; -1.0 for
            a total loss)
        """
        dd = drawdown_series(self.returns)
        return dd, float(dd.min())

    def conditional_drawdown_at_risk(self, confidence: float = 0.95) -> float:
        """Average drawdown in the worst (1-confidence) fraction of periods
        (negative), by the same :func:`expected_shortfall` estimator."""
        dd, _ = self.max_drawdown()
        return -expected_shortfall(dd.to_numpy(), confidence)

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Information Ratio = annualised mean excess return over the benchmark /
        annualised tracking error.

        ``benchmark_returns`` must have a finite value on every date of the
        portfolio's returns (``ValueError`` otherwise; a missing benchmark day
        never counts as 0%). A benchmark on a denser calendar (a 7-day index
        under a Monday-Friday portfolio) is compounded onto the portfolio's
        dates, so both legs span the same intervals (:func:`compound_onto`).
        NaN when the tracking error is zero (the ratio is undefined).
        """
        on_date = benchmark_returns.reindex(self.returns.index)
        bench = compound_onto(benchmark_returns, self.returns.index).reindex(self.returns.index)
        finite = np.isfinite(on_date.to_numpy(dtype=float)) & np.isfinite(bench.to_numpy(dtype=float))
        missing = bench.index[~finite]
        if len(missing):
            raise ValueError(
                f"benchmark_returns has no value on {len(missing)} of the portfolio's "
                f"{len(self.returns)} dates (first {missing[0]}); align the two series "
                "first. A missing day is never counted as 0%."
            )
        excess = self.returns - bench
        te = float(excess.std() * np.sqrt(self.periods))
        if te < 1e-10:
            return float("nan")
        return float(excess.mean() * self.periods / te)

    def tail_ratio(self) -> float:
        """
        Tail ratio = 95th percentile gain / abs(5th percentile loss).
        Ratio > 1 = right-skewed, > 2 = strong positive asymmetry. NaN when the
        5th percentile is zero (undefined).
        """
        p95 = np.percentile(self.returns, 95)
        p05 = abs(np.percentile(self.returns, 5))
        return float(p95 / p05) if p05 > 1e-10 else float("nan")

    def omega_ratio(self, threshold: float | None = None) -> float:
        """
        Omega ratio: ratio of gains above threshold to losses below.
        """
        if threshold is None:
            threshold = self.rfr_daily
        gains = self.returns[self.returns > threshold] - threshold
        losses = threshold - self.returns[self.returns <= threshold]
        total_losses = losses.sum()
        if total_losses > 1e-10:
            return float(gains.sum() / total_losses)
        # No losses: unbounded with gains, undefined (0 / 0) without.
        return float("inf") if gains.sum() > 0 else float("nan")

    def _max_drawdown_duration(self, dd_series: pd.Series) -> int:
        """Length of the longest drawdown period in trading days."""
        max_dur = 0
        cur = 0
        for v in dd_series:
            if v < 0:
                cur += 1
                max_dur = max(max_dur, cur)
            else:
                cur = 0
        return max_dur
