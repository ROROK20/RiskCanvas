"""
Portfolio performance metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskcanvas.metrics.risk import _no_missing, compound_onto, drawdown_series, periods_per_year, years_spanned

TRADING_DAYS = 252
RISK_FREE = 0.02


@dataclass
class PerformanceReport:
    """Full performance metrics.

    ``sortino``, ``calmar`` and ``profit_factor`` are ``+inf`` when their
    denominator is zero with a positive numerator (no downside, no drawdown,
    no losing days). ``to_dict()`` maps every non-finite value to ``None`` so
    the report is always JSON-serialisable (``null`` on the wire).
    ``positive_months`` is the share of months holding at least one
    observation whose compounded return is positive. ``avg_win`` and
    ``avg_loss`` average the gaining and the losing days; a flat day is neither.
    """
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    annual_vol: float
    max_drawdown: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    best_day: float
    worst_day: float
    positive_months: float

    def to_dict(self) -> dict:
        return {
            k: (round(v, 6) if math.isfinite(v) else None)
            for k, v in self.__dict__.items()
        }

    def __repr__(self) -> str:
        return (
            f"PerformanceReport("
            f"CAGR={self.cagr:.2%}, "
            f"Sharpe={self.sharpe:.2f}, "
            f"MaxDD={self.max_drawdown:.2%}, "
            f"Calmar={self.calmar:.2f})"
        )


class PerformanceMetrics:
    """
    Portfolio performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Returns (fractional), normally daily, on a ``DatetimeIndex``: the
        monthly statistics (``positive_months``) need dates.
    risk_free_rate : float
        Annual risk-free rate.
    periods : int | None
        Observations per year for the annual figures. Default: read from the
        dates by :func:`~riskcanvas.metrics.risk.periods_per_year`, which
        measures a non-standard calendar and refuses only what it cannot
        measure; pass the true number for such a series. CAGR's years come from
        the dates unless the calendar holds a standard rate
        (:func:`~riskcanvas.metrics.risk.years_spanned`); ``periods`` overrides
        both, as ``n / periods`` years.

    Raises
    ------
    ValueError
        A missing value, an index that is not a ``DatetimeIndex`` (the monthly
        statistics need dates), or dates whose periods per year cannot be
        measured with ``periods`` left out.
    """

    def __init__(
        self,
        returns: pd.Series,
        risk_free_rate: float = RISK_FREE,
        periods: int | None = None,
    ) -> None:
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError(
                "PerformanceMetrics needs daily returns on a DatetimeIndex (one row per "
                f"trading day; monthly statistics use the dates); got {type(returns.index).__name__}."
            )
        self.returns = _no_missing(returns)
        self.rfr = risk_free_rate
        if periods is not None and periods <= 0:
            raise ValueError(f"periods must be positive (got {periods!r})")
        self.periods = periods_per_year(self.returns.index) if periods is None else periods
        self.n_years = len(self.returns) / periods if periods is not None else years_spanned(self.returns.index, self.periods)
        self.rfr_daily = risk_free_rate / self.periods

    def cagr(self) -> float:
        """Compound annual growth rate over :attr:`n_years`; ``+inf`` when a finite
        total is too large to annualise."""
        total = float((1 + self.returns).prod())
        try:
            return float(total ** (1 / max(self.n_years, 1e-9)) - 1)
        except OverflowError:
            return float("inf")

    def compute_all(self) -> PerformanceReport:
        r = self.returns

        total_ret = float((1 + r).prod() - 1)
        cagr = self.cagr()
        annual_vol = float(r.std() * np.sqrt(self.periods))

        # Sharpe
        excess = r - self.rfr_daily
        # Undefined (NaN, null over the wire) at zero volatility.
        sharpe = float(excess.mean() / excess.std() * np.sqrt(self.periods)) if excess.std() > 1e-10 else float("nan")

        # Sortino: annualised mean excess over the daily risk-free target T,
        # divided by the annualised downside deviation
        #     sqrt( (1/N) * sum_t min(r_t - T, 0)^2 )
        # taken over ALL N observations (Sortino & Price, 1994) — not the
        # standard deviation of only the below-target days, which ignores how
        # rarely the shortfalls occur and inflates the ratio. With no
        # shortfall at all the ratio is +inf (positive excess) or NaN (every
        # day exactly at the target: 0 / 0).
        shortfall = np.minimum(r.values - self.rfr_daily, 0.0)
        down_dev = float(np.sqrt(np.mean(shortfall ** 2)) * np.sqrt(self.periods))
        annual_excess = float(excess.mean() * self.periods)
        if down_dev > 0.0:
            sortino = annual_excess / down_dev
        elif annual_excess > 0:
            sortino = float("inf")
        else:
            sortino = float("nan")

        # Max drawdown (from the starting capital, so a first-day loss counts;
        # the same series RiskMetrics reports) and Calmar = CAGR / |max DD|.
        max_dd = float(drawdown_series(r).min())
        if abs(max_dd) > 1e-10:
            calmar = float(cagr / abs(max_dd))
        else:
            # No drawdown: Calmar is unbounded for a gain, undefined (0 / 0)
            # for a flat series.
            calmar = float("inf") if cagr > 0 else float("nan")

        # Win/loss stats
        wins = r[r > 0]
        losses = r[r < 0]   # a flat day is not a loss, and would dilute avg_loss
        win_rate = float(len(wins) / max(len(r), 1))
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        if abs(losses.sum()) > 1e-10:
            profit_factor = float(wins.sum() / abs(losses.sum()))
        else:   # no losses: unbounded with gains, undefined (0 / 0) without
            profit_factor = float("inf") if wins.sum() > 0 else float("nan")

        # Monthly win rate, over the months that hold an observation: a month
        # with none (dropped by missing="drop") is not a flat month.
        by_month = (1 + r).resample("ME")
        monthly = by_month.prod()[by_month.count() > 0] - 1
        pos_months = float((monthly > 0).mean())

        return PerformanceReport(
            total_return=total_ret,
            cagr=cagr,
            sharpe=sharpe,
            sortino=sortino,
            calmar=calmar,
            annual_vol=annual_vol,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            best_day=float(r.max()),
            worst_day=float(r.min()),
            positive_months=pos_months,
        )

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Rolling annualised Sharpe ratio over ``window`` days. The first
        ``window - 1`` values are NaN (no full window yet), as is any window
        with zero volatility."""
        excess = self.returns - self.rfr_daily
        roll = excess.rolling(window)
        std = roll.std()
        return roll.mean() / std.where(std > 0) * np.sqrt(self.periods)

    def rolling_vol(self, window: int = 21) -> pd.Series:
        """Rolling annualized volatility."""
        return self.returns.rolling(window).std() * np.sqrt(self.periods)

    def factor_exposure(self, factors: pd.DataFrame) -> pd.Series:
        """
        OLS regression of the portfolio's daily returns on factor returns:
        ``r = alpha + sum_k beta_k * f_k + e``. Returns the point estimates
        only (``alpha`` per day, then one loading per factor column): no
        standard errors, t-statistics or R^2. One helper, not a factor
        model; supply the factor series (e.g. Fama-French) yourself.

        ``factors`` must have a finite value for every portfolio date
        (``ValueError`` otherwise, never silently dropped), and there must be
        more observations than parameters. Factors on a denser calendar are
        compounded onto the portfolio's dates, so both sides of the regression
        span the same intervals (:func:`~riskcanvas.metrics.risk.compound_onto`).
        """
        on_date = factors.reindex(self.returns.index)
        aligned = compound_onto(factors, self.returns.index).reindex(self.returns.index)
        bad = ~(np.isfinite(on_date.to_numpy(dtype=float)) & np.isfinite(aligned.to_numpy(dtype=float))).all(axis=1)
        if bad.any():
            raise ValueError(
                f"factors have no value on {int(bad.sum())} of the portfolio's "
                f"{len(self.returns)} dates (first {aligned.index[bad][0]}); align them first."
            )
        n_params = aligned.shape[1] + 1
        if len(aligned) <= n_params:
            raise ValueError(
                f"factor_exposure needs more than {n_params} observations (got {len(aligned)})."
            )
        X = np.column_stack([np.ones(len(aligned)), aligned.to_numpy(dtype=float)])
        betas = np.linalg.lstsq(X, self.returns.to_numpy(dtype=float), rcond=None)[0]
        labels = ["alpha"] + list(factors.columns)
        return pd.Series(betas, index=labels)
