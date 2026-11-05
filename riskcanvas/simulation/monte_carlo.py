"""
Monte Carlo portfolio simulation.

Simulates multi-asset portfolio paths using:
1. GBM (Geometric Brownian Motion) — standard
2. Student-t returns — fat tails
3. Historical block bootstrap — non-parametric

GBM and Student-t correlate their per-asset shocks with an exact factor of
the historical covariance matrix (Cholesky, or an eigen square root when the
matrix is singular, e.g. with a cash leg; no jitter). The bootstrap uses no
covariance model: it
resamples blocks of the portfolio's own historical daily returns, and every
resampled day is a real day for all assets at once, so cross-asset
co-movement is carried over empirically.

The bootstrap is circular (blocks wrap from the last observation to the first),
so every historical day is equally likely at every day of a path.

Every result reports terminal-value statistics plus ``percentile_bands``: the
5/25/50/75/95th percentiles of simulated portfolio value at checkpoint days
across the horizon (a genuine fan chart, not an interpolation), and
``standard_errors``: the Monte Carlo sampling error of each estimate, so a
reader can tell signal from simulation noise (see ``_standard_errors``).

Drift: what each asset (GBM, Student-t) or the portfolio series (bootstrap)
is assumed to earn per day, chosen by ``MonteCarloSimulator(drift=...)``:

* ``"zero"`` (default, the risk-desk convention): no expected return. GBM and
  Student-t use a zero mean; the bootstrap re-centres the historical series
  to a zero arithmetic mean. A short, strongly trending sample cannot turn
  the tail into a gain.
* ``"historical"``: the in-sample mean daily return. Two years at +0.12% a day (0.8% daily vol) project a one-year
  95% VaR that is a gain (VaR95 -8.8%; VaR99 stays a +0.2% loss); the API
  warns when this drift makes VaR95 negative.
* ``"risk_free"``: the library's ``RISK_FREE`` rate (2% a year), divided by the
  series' own periods per year (252 for a trading-day series, 365 for a 7-day
  one).

``horizon_days`` counts observations of the series: calendar days on a 7-day
series, and weeks or months on a weekly or monthly one (``periods_per_year``
says which).

Weights that sum to less than 1 (``normalize_weights=False``) leave the
remainder in cash, at zero return (the risk-free rate under
``drift="risk_free"``); the loaders refuse a sum above 1 (leverage).

Rebalancing assumption: all three models hold the weights FIXED, i.e. the
portfolio is rebalanced back to its target weights every day. GBM and
Student-t blend the per-asset daily multipliers with the weights each step;
the bootstrap resamples the daily-rebalanced portfolio return series. A
buy-and-hold portfolio, whose weights drift with prices, is not modelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from riskcanvas.metrics.risk import RISK_FREE, expected_shortfall, periods_per_year
from riskcanvas.portfolio.loader import floor_at_total_loss

DRIFTS = ("zero", "historical", "risk_free")

BAND_PERCENTILES = (5, 25, 50, 75, 95)
MAX_BAND_POINTS = 50
_Z95 = 1.959963984540054  # two-sided 95% normal quantile
# The standard errors square simulated values and sum over the paths, so past
# about sqrt(float max / MAX n_paths) a variance overflows even though the
# values themselves are finite. A portfolio up 1e150-fold has no meaningful
# risk statistics anyway.
MAX_SIMULATED_VALUE = 1e150


def _quantile_se(sorted_values: np.ndarray, p: float) -> float:
    """Standard error of the empirical ``p``-quantile, distribution-free.

    The order statistics at ranks ``n*p -/+ 1.96*sqrt(n*p*(1-p))`` bound a
    95% confidence interval for the quantile (binomial argument, no density
    estimate needed); its half-width / 1.96 is the standard error. On a
    discrete simulated distribution (a bootstrap over a horizon of about one
    block has only as many outcomes as block starts) it errs conservative.
    """
    n = len(sorted_values)
    half = _Z95 * math.sqrt(n * p * (1.0 - p))
    lo = max(0, math.floor(n * p - half))
    hi = min(n - 1, math.ceil(n * p + half))
    return float((sorted_values[hi] - sorted_values[lo]) / (2.0 * _Z95))


def _expected_shortfall_se(losses: np.ndarray, var: float, es: float, a: float) -> float:
    """Asymptotic standard error of the expected shortfall ``es`` at tail
    probability ``a``: ``sqrt((Var(L | tail) + (1 - a) * (ES - VaR)^2) / (n * a))``,
    the tail being the worst ``ceil(n * a)`` losses, the outcomes the
    Acerbi-Tasche ES averages (with ties at the VaR, "every loss >= VaR"
    over-counts them). NaN when the tail holds fewer than two paths."""
    k = int(math.ceil(len(losses) * a))
    if k < 2:
        return float("nan")
    tail = np.sort(losses)[-k:]
    return float(math.sqrt((float(tail.var(ddof=1)) + (1.0 - a) * (es - var) ** 2) / (len(losses) * a)))


def _held(portfolio) -> tuple[pd.DataFrame, np.ndarray]:
    """The weighted tickers' returns and their weights, aligned BY NAME, in sorted
    order: which asset takes which draw depends on the order, so without sorting the
    same portfolio and seed with its JSON keys reordered would move a VaR99 from 9.40%
    to 9.57%.

    Every model reads both from here, so a ``Portfolio`` whose ``returns``
    columns are in a different order from ``symbols`` cannot pair a weight
    with another asset's returns."""
    tickers = sorted(portfolio.weights)
    return portfolio.returns[tickers], np.array([portfolio.weights[t] for t in tickers], dtype=float)


def _covariance_factor(cov: np.ndarray) -> np.ndarray:
    """A matrix ``F`` with ``F @ F.T == cov`` (to rounding), so draws
    ``z @ F.T`` with ``z ~ N(0, I)`` have covariance exactly ``cov``.

    Cholesky when ``cov`` is positive definite. A singular but valid
    covariance (a riskless cash leg or other constant series, perfectly
    collinear assets) has no Cholesky factor; its symmetric eigen square root
    ``V * sqrt(max(eigenvalues, 0))`` is used instead, which keeps every
    correlation. Nothing is added to the diagonal: a ``1e-8`` daily-variance
    jitter would give a riskless cash leg about 0.16% annual simulated
    volatility.
    """
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        return vecs * np.sqrt(np.clip(vals, 0.0, None))


def _checkpoint_days(horizon_days: int) -> list[int]:
    """Days (0..horizon) at which percentile bands are recorded: at most
    ``MAX_BAND_POINTS + 1`` evenly spaced days, always including 0 and the
    horizon."""
    n = min(horizon_days, MAX_BAND_POINTS)
    return sorted({int(round(x)) for x in np.linspace(0, horizon_days, n + 1)})


def _band(day: int, values: np.ndarray) -> dict[str, float]:
    pct = np.percentile(values, BAND_PERCENTILES)
    return {"day": day, **{f"p{p}": float(v) for p, v in zip(BAND_PERCENTILES, pct)}}


@dataclass
class SimulationResult:
    """Output from Monte Carlo simulation."""

    n_paths: int
    horizon_days: int
    terminal_values: np.ndarray        # shape: (n_paths,)
    # Percentiles of the TERMINAL value only (p5..p95). Kept under this name
    # for wire compatibility; it is not a path.
    percentile_paths: dict[str, float]
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    prob_loss: float                   # fraction of paths ending below 1.0
    expected_terminal: float           # mean terminal value
    method: str
    # Percentiles of portfolio value at checkpoint days 0..horizon:
    # [{"day": d, "p5": ..., "p25": ..., "p50": ..., "p75": ..., "p95": ...}]
    percentile_bands: list[dict[str, float]]
    # Monte Carlo standard error of each estimate: var_95, var_99, cvar_95,
    # cvar_99, prob_loss, expected_terminal (NaN where undefined).
    standard_errors: dict[str, float]
    seed: int | None = None
    drift: str = "zero"                # see the module notes: zero | historical | risk_free

    @property
    def percentile_table(self) -> pd.DataFrame:
        """Table of key percentiles of terminal values."""
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        return pd.DataFrame({
            "percentile": percentiles,
            "terminal_value": [np.percentile(self.terminal_values, p) for p in percentiles],
            "return": [np.percentile(self.terminal_values, p) - 1.0 for p in percentiles],
        })

    def __repr__(self) -> str:
        return (
            f"SimulationResult(paths={self.n_paths}, "
            f"horizon={self.horizon_days}d, "
            f"VaR95={self.var_95:.2%}, "
            f"E[terminal]={self.expected_terminal:.3f})"
        )


def _whole_horizon(horizon_days) -> int:
    """The horizon as an int, or ``ValueError``: zero steps of a simulation report a
    VaR of 0.0, which reads as a riskless portfolio."""
    if int(horizon_days) != horizon_days or horizon_days < 1:
        raise ValueError(f"horizon_days must be a whole number >= 1 (got {horizon_days!r})")
    return int(horizon_days)


class MonteCarloSimulator:
    """
    Monte Carlo portfolio simulation with multiple return models.

    Parameters
    ----------
    n_paths : int
        Number of simulation paths (default 10_000).
    seed : int | None
        Random seed for reproducibility.
    periods : int | None
        Observations per year, for the rate cash earns under ``drift="risk_free"``.
        Default: read from the dates (:func:`~riskcanvas.metrics.risk.periods_per_year`).

    Examples
    --------
    >>> sim = MonteCarloSimulator(n_paths=50_000)
    >>> result = sim.simulate_gbm(portfolio, horizon_days=252)
    >>> print(result)
    """

    def __init__(
        self, n_paths: int = 10_000, seed: int | None = 42, drift: str = "zero", periods: int | None = None
    ) -> None:
        if int(n_paths) != n_paths or n_paths < 1:
            raise ValueError(f"n_paths must be a whole number >= 1 (got {n_paths!r})")
        if drift not in DRIFTS:
            raise ValueError(f"drift must be one of {DRIFTS} (got {drift!r})")
        if periods is not None and periods <= 0:
            raise ValueError(f"periods must be positive (got {periods!r})")
        self.periods = periods
        self.n_paths = int(n_paths)
        self.seed = seed
        self.drift = drift

    def _rate_per_period(self, index) -> float:
        """What cash earns per observation: 0, or under ``drift="risk_free"``
        ``RISK_FREE`` over the series' own periods per year: dividing by 252 on
        every series would earn a riskless leg of a 7-day series 2.94% over 365
        days, not the stated 2%."""
        if self.drift != "risk_free":
            return 0.0
        return RISK_FREE / (periods_per_year(index) if self.periods is None else self.periods)

    def _cash(self, weights: np.ndarray, index) -> tuple[float, float]:
        """``(cash_weight, daily_growth)`` of the uninvested remainder
        ``1 - sum(weights)`` (a partially invested portfolio,
        ``normalize_weights=False``): it earns :meth:`_rate_per_period` and stays
        in the blend, where leaving it out would lose the uninvested share 100%
        every day."""
        cash = max(0.0, 1.0 - float(weights.sum()))
        return cash, 1.0 + self._rate_per_period(index)

    def _daily_drift(self, sample_mean, index):
        """The mean return per observation to simulate, per the ``drift``
        setting: the sample mean (``historical``), or :meth:`_rate_per_period`."""
        if self.drift == "historical":
            return sample_mean
        return np.full_like(np.asarray(sample_mean, dtype=float), self._rate_per_period(index))

    def simulate_gbm(
        self,
        portfolio,
        horizon_days: int = 252,
    ) -> SimulationResult:
        """
        Geometric Brownian Motion simulation with correlation.

        Each asset price evolves as a true log-normal GBM. Per daily step the
        asset log-return is

            d(log S_i) = (mu_i - 0.5 * sigma_i^2) * dt + correlated_gaussian_i

        with dt = 1 day, sigma_i^2 = cov[i, i], and the correlated Gaussian
        shocks produced from an exact factor of the daily covariance matrix
        (``_covariance_factor``).
        The per-asset gross multiplier is exp(log-return), so asset prices are
        genuinely log-normal (strictly positive, right-skewed). The portfolio
        value is the weight-blended basket of these multipliers, compounded
        across the horizon — so ``terminal_values`` are products of log-normals
        and can never go negative.

        The -0.5 * sigma^2 Itô drift correction makes the expected per-asset
        multiplier equal exp(mu_i) per day, so E[terminal] matches the intended
        arithmetic drift rather than being inflated by convexity.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio with returns data.
        horizon_days : int
            Simulation horizon, in observations of the series (trading days
            for a trading-day series; see the module notes).

        Returns
        -------
        SimulationResult

        Raises
        ------
        ValueError
            ``horizon_days`` is not a whole number >= 1.
        """
        horizon_days = _whole_horizon(horizon_days)
        rng = np.random.default_rng(self.seed)
        returns, weights = _held(portfolio)
        cash_weight, cash_growth = self._cash(weights, returns.index)
        mu = self._daily_drift(returns.mean().values, returns.index)
        cov = returns.cov().values

        # Exact covariance factor for correlated draws (no jitter).
        L = _covariance_factor(cov)

        n_assets = len(mu)
        # Itô drift correction: log-drift = mu - 0.5 * sigma^2 (per day, dt=1).
        var = np.diag(cov)
        log_drift = mu - 0.5 * var
        terminal_values = np.ones(self.n_paths)
        checkpoints = set(_checkpoint_days(horizon_days))
        bands = [_band(0, terminal_values)]

        for day in range(1, horizon_days + 1):
            # Draw correlated standard normals -> correlated diffusion shocks.
            z = rng.standard_normal((self.n_paths, n_assets))
            correlated = z @ L.T
            # Asset log-returns for this step (drift-corrected).
            asset_log_returns = log_drift + correlated  # (n_paths, n_assets)
            # Log-normal gross multipliers, blended into a portfolio basket.
            asset_multipliers = np.exp(asset_log_returns)
            port_multiplier = asset_multipliers @ weights + cash_weight * cash_growth
            terminal_values *= port_multiplier
            if day in checkpoints:
                bands.append(_band(day, terminal_values))

        return self._make_result(terminal_values, horizon_days, "GBM", bands)

    def simulate_t_distribution(
        self,
        portfolio,
        horizon_days: int = 252,
        df_t: float = 5.0,
    ) -> SimulationResult:
        """
        Fat-tailed simulation with multivariate Student-t innovations.

        Each path-day draws ONE chi-square mixing variable shared by every
        asset, ``W ~ chi2(df)``, and a correlated Gaussian vector
        ``z ~ N(0, cov)`` (the same covariance factor as GBM). The shock is

            z * sqrt((df - 2) / W)

        a multivariate Student-t with ``df`` degrees of freedom, scaled to
        unit variance so its covariance is exactly ``cov`` (the raw t has
        variance ``df / (df - 2)``). Because the mixing variable is shared,
        every portfolio of the assets is itself Student-t with the same
        ``df``: a fat-tailed day is fat-tailed for all assets at once, so the
        tails survive diversification (1-day portfolio excess kurtosis about
        ``6 / (df - 4)`` for ``df > 4``, whatever the number of assets).

        An independent t per asset, mixed by a Cholesky factor, would not do:
        by the central limit theorem the portfolio shock tends to a Gaussian as
        assets are added (1-day excess kurtosis 4.17 with one asset, 0.42 with
        twenty), so "Student-t" would barely differ from GBM.

        Compounding is log-normal and shares the GBM structure: per step the
        asset log-return is (mu_i - 0.5 * sigma_i^2) + correlated_t_shock_i, the
        gross multiplier is exp(log-return), and the portfolio value is the
        weight-blended basket. This keeps terminal values strictly positive
        (never negative on extreme tail draws) while the fat tails come from the
        Student-t innovations rather than Gaussian ones.

        Caveat on the mean: the -0.5 * sigma^2 term is the Gaussian Itô
        correction and does NOT make E[terminal] match the drift here. The
        Student-t has no moment-generating function, so E[exp(shock)] is not
        finite in theory; ``expected_terminal`` under this model is a sample
        statistic driven by the largest draws. Read the percentiles (and
        VaR/CVaR), not the mean.

        Parameters
        ----------
        portfolio : Portfolio
        horizon_days : int
        df_t : float
            Degrees of freedom for Student-t (lower = fatter tails). Default 5.
            Must be finite and > 2 (the variance is infinite at or below 2, so
            no scaling can match the covariance).

        Raises
        ------
        ValueError
            ``df_t`` not finite or ``<= 2``; ``horizon_days`` not a whole number >= 1.
        """
        horizon_days = _whole_horizon(horizon_days)
        if not (math.isfinite(df_t) and df_t > 2):
            raise ValueError(f"df_t must be finite and > 2 (got {df_t!r})")
        rng = np.random.default_rng(self.seed)
        returns, weights = _held(portfolio)
        cash_weight, cash_growth = self._cash(weights, returns.index)
        mu = self._daily_drift(returns.mean().values, returns.index)
        cov = returns.cov().values

        L = _covariance_factor(cov)

        n_assets = len(mu)
        # Gaussian Itô term (see the caveat in the docstring: it does not make
        # E[terminal] track the drift under Student-t shocks).
        var = np.diag(cov)
        log_drift = mu - 0.5 * var
        terminal_values = np.ones(self.n_paths)
        checkpoints = set(_checkpoint_days(horizon_days))
        bands = [_band(0, terminal_values)]

        for day in range(1, horizon_days + 1):
            # Correlated Gaussian vector, then ONE chi-square mixing variable
            # per path-day shared by all assets: a multivariate t, scaled to
            # unit variance so the covariance is exactly cov.
            z = rng.standard_normal((self.n_paths, n_assets)) @ L.T
            w = rng.chisquare(df_t, size=self.n_paths)
            correlated = z * np.sqrt((df_t - 2.0) / w)[:, None]
            # Asset log-returns with the fat-tailed shock, drift-corrected.
            asset_log_returns = log_drift + correlated
            asset_multipliers = np.exp(asset_log_returns)
            port_multiplier = asset_multipliers @ weights + cash_weight * cash_growth
            terminal_values *= port_multiplier
            if day in checkpoints:
                bands.append(_band(day, terminal_values))

        return self._make_result(terminal_values, horizon_days, "Student-t", bands)

    def simulate_bootstrap(
        self,
        portfolio,
        horizon_days: int = 252,
        block_size: int = 20,
    ) -> SimulationResult:
        """
        Historical circular block bootstrap (Politis & Romano, 1992).

        Each path is built from consecutive blocks of ``block_size`` days of
        the portfolio's historical daily returns. Each block starts at a
        uniformly drawn observation ``0 .. n_obs - 1`` and wraps around the
        end of the history to its start, so every observation is equally
        likely at every day of every path, whatever the horizon. Blocks
        preserve short-range autocorrelation and volatility clustering (the
        one block that wraps joins the last day to the first). Non-parametric,
        no distribution assumption. Every path is exactly ``horizon_days``
        long: the final block is truncated to fit. With ``horizon_days <=
        block_size`` there are only ``n_obs`` distinct outcomes (one per
        start), i.e. the 1-day result IS the empirical distribution.

        Wrapping is what makes that hold: block starts limited to ``0 .. n_obs -
        block_size`` would never draw the last ``block_size - horizon_days``
        observations at a horizon shorter than a block (20 observations at 1 day:
        a single outcome), and would under-weight recent days at longer horizons.

        Parameters
        ----------
        portfolio : Portfolio
        horizon_days : int
        block_size : int
            Length of return blocks to resample (default 20 = ~1 month).

        Raises
        ------
        ValueError
            If ``block_size`` is < 1, ``horizon_days`` is not a whole number >= 1, or the history is
            shorter than one block (a shorter history cannot supply a full
            block, and silently shrinking the horizon or the block would
            misreport what was simulated).
        """
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1 (got {block_size})")
        horizon_days = _whole_horizon(horizon_days)

        rng = np.random.default_rng(self.seed)
        returns, weights = _held(portfolio)
        # Cash (1 - sum(weights)) returns 0 here; re-centring below sets the
        # whole portfolio's mean, cash included, to the chosen drift.
        port_returns_hist = floor_at_total_loss(returns.values @ weights)
        if self.drift != "historical":
            port_returns_hist = port_returns_hist - port_returns_hist.mean() + float(
                self._daily_drift(port_returns_hist.mean(), returns.index))
            if (port_returns_hist < -1.0).any():
                raise ValueError(
                    f"drift={self.drift!r} re-centres the history, which turns its "
                    "worst day into a loss of more than 100%; use drift='historical'."
                )
        n_obs = len(port_returns_hist)
        if n_obs < block_size:
            raise ValueError(
                f"bootstrap needs at least block_size={block_size} historical "
                f"observations (got {n_obs})"
            )

        # within_block[s, j] = growth over days s .. s+j (mod n_obs) of the
        # history, for every block start s in 0 .. n_obs - 1 (circular: the
        # series is extended by its first block_size - 1 days). Each path
        # multiplies one row prefix per block, so the simulation is
        # O(n_paths * n_blocks) vectorised.
        n_starts = n_obs
        wrapped = np.concatenate([port_returns_hist, port_returns_hist[: block_size - 1]])
        windows = np.lib.stride_tricks.sliding_window_view(1.0 + wrapped, block_size)
        within_block = np.cumprod(windows, axis=1)

        checkpoints = _checkpoint_days(horizon_days)[1:]
        bands = [_band(0, np.ones(self.n_paths))]
        values = np.ones(self.n_paths)
        day0 = 0
        cp = 0
        while day0 < horizon_days:
            length = min(block_size, horizon_days - day0)
            starts = rng.integers(0, n_starts, size=self.n_paths)
            while cp < len(checkpoints) and checkpoints[cp] <= day0 + length:
                offset = checkpoints[cp] - day0 - 1
                bands.append(_band(checkpoints[cp], values * within_block[starts, offset]))
                cp += 1
            values = values * within_block[starts, length - 1]
            day0 += length

        return self._make_result(values, horizon_days, "Block Bootstrap", bands)

    def _make_result(
        self,
        terminal_values: np.ndarray,
        horizon_days: int,
        method: str,
        percentile_bands: list[dict[str, float]],
    ) -> SimulationResult:
        # A path that overflows float64 stays inf (or becomes nan) to the end;
        # a bootstrap checkpoint inside a block can overflow while the block's
        # end does not, so the bands are checked too. Values below that but
        # past MAX_SIMULATED_VALUE overflow the statistics computed from them.
        # None of this has a meaningful VaR or percentile: refuse it rather
        # than report inf/nan statistics (or crash computing them).
        finite = np.isfinite(terminal_values).all() and all(
            math.isfinite(v) for band in percentile_bands for v in band.values()
        )
        if not finite or terminal_values.max() > MAX_SIMULATED_VALUE:
            raise ValueError(
                f"Simulated portfolio values overflowed floating-point range over "
                f"{horizon_days} days ({method}): the input's daily returns are too "
                "extreme to compound over this horizon. Shorten horizon_days or "
                "check the returns."
            )
        returns_from_base = terminal_values - 1.0

        var_95 = float(-np.percentile(returns_from_base, 5)) + 0.0   # never -0.0
        var_99 = float(-np.percentile(returns_from_base, 1)) + 0.0

        # Discrete (Acerbi-Tasche) ES: ties at the VaR cannot dilute it.
        cvar_95 = expected_shortfall(returns_from_base, 0.95)
        cvar_99 = expected_shortfall(returns_from_base, 0.99)

        n = len(terminal_values)
        losses = -returns_from_base
        sorted_returns = np.sort(returns_from_base)
        prob_loss = float((terminal_values < 1.0).mean())
        standard_errors = {
            "var_95": _quantile_se(sorted_returns, 0.05),
            "var_99": _quantile_se(sorted_returns, 0.01),
            "cvar_95": _expected_shortfall_se(losses, var_95, cvar_95, 0.05),
            "cvar_99": _expected_shortfall_se(losses, var_99, cvar_99, 0.01),
            "prob_loss": math.sqrt(prob_loss * (1.0 - prob_loss) / n),
            "expected_terminal": float(terminal_values.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        }

        # Terminal-value percentiles (see ``percentile_bands`` for the fan).
        pct_paths = {
            "p5": float(np.percentile(terminal_values, 5)),
            "p25": float(np.percentile(terminal_values, 25)),
            "p50": float(np.percentile(terminal_values, 50)),
            "p75": float(np.percentile(terminal_values, 75)),
            "p95": float(np.percentile(terminal_values, 95)),
        }

        return SimulationResult(
            n_paths=len(terminal_values),
            horizon_days=horizon_days,
            terminal_values=terminal_values,
            percentile_paths=pct_paths,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            prob_loss=prob_loss,
            expected_terminal=float(terminal_values.mean()),
            method=method,
            drift=self.drift,
            percentile_bands=percentile_bands,
            standard_errors=standard_errors,
            seed=self.seed,
        )
