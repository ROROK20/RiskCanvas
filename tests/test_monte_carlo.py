"""Tests for Monte Carlo simulation."""

import numpy as np
import pandas as pd
import pytest

from riskcanvas.portfolio.loader import PortfolioLoader
from riskcanvas.simulation.monte_carlo import MonteCarloSimulator, SimulationResult


def make_mock_portfolio():
    """Create a simple 2-asset portfolio from synthetic returns."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.DataFrame({
        "A": rng.normal(0.0005, 0.012, n),
        "B": rng.normal(0.0003, 0.010, n),
    }, index=dates)
    loader = PortfolioLoader()
    return loader.from_returns_df(returns, weights={"A": 0.60, "B": 0.40})


@pytest.fixture(scope="module")
def portfolio():
    return make_mock_portfolio()


@pytest.fixture(scope="module")
def simulator():
    return MonteCarloSimulator(n_paths=1000, seed=42)


class TestMonteCarloSimulator:
    def test_gbm_returns_result(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, horizon_days=63)
        assert isinstance(result, SimulationResult)

    def test_terminal_values_length(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, horizon_days=63)
        assert len(result.terminal_values) == 1000

    def test_var_positive(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, horizon_days=252)
        assert result.var_95 > 0
        assert result.var_99 > 0

    def test_cvar_ge_var(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, horizon_days=252)
        assert result.cvar_95 >= result.var_95 - 1e-9
        assert result.cvar_99 >= result.var_99 - 1e-9

    def test_prob_loss_between_zero_one(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, horizon_days=252)
        assert 0 <= result.prob_loss <= 1

    def test_expected_terminal_positive(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, horizon_days=252)
        assert result.expected_terminal > 0

    def test_var99_ge_var95(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, horizon_days=252)
        assert result.var_99 >= result.var_95

    def test_percentile_paths_ordered(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, horizon_days=252)
        pp = result.percentile_paths
        assert pp["p5"] <= pp["p25"] <= pp["p50"] <= pp["p75"] <= pp["p95"]

    def test_t_one_day_var99_exceeds_gbm(self):
        """docs/usage.md's portfolio, 1 day, 50,000 paths: a unit-variance t5 tail quantile
        is 2.606 sigma against the normal's 2.326 (x1.12); independent per-asset t's
        give only 1.033 here."""
        sim = MonteCarloSimulator(n_paths=50_000, seed=42)
        p = _usage_example_portfolio()
        ratio = sim.simulate_t_distribution(p, 1).var_99 / sim.simulate_gbm(p, 1).var_99
        assert 1.09 < ratio < 1.16, ratio

    def test_bootstrap_returns_result(self, simulator, portfolio):
        result = simulator.simulate_bootstrap(portfolio, horizon_days=63)
        assert isinstance(result, SimulationResult)
        assert len(result.terminal_values) == 1000

    def test_percentile_table_shape(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, 63)
        table = result.percentile_table
        assert len(table) == 9  # 9 percentile rows
        assert "terminal_value" in table.columns

    def test_method_label(self, simulator, portfolio):
        result = simulator.simulate_gbm(portfolio, 63)
        assert result.method == "GBM"
        result2 = simulator.simulate_t_distribution(portfolio, 63)
        assert "Student" in result2.method


def _usage_example_portfolio():
    idx = pd.bdate_range("2021-01-01", periods=750)
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, size=(750, 3)), index=idx, columns=list("ABC"))
    return PortfolioLoader().from_returns_df(rets, weights={"A": 0.5, "B": 0.3, "C": 0.2})


def _factor_portfolio(n_assets, seed=0, n_obs=1500):
    """n assets loading 0.5 on one market factor plus idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n_obs)
    m = rng.standard_normal(n_obs) * 0.008
    return PortfolioLoader().from_returns_df(
        pd.DataFrame({f"A{i}": 0.5 * m + rng.standard_normal(n_obs) * 0.012 for i in range(n_assets)}, index=idx)
    )


class TestMultivariateStudentT:
    """Each asset drew an independent t and a
    Cholesky factor mixed them, so the fat tails averaged away under
    diversification (1-day portfolio excess kurtosis 4.17 with one asset,
    0.42 with twenty). One chi-square mixing variable per path-day, shared by
    every asset, makes every portfolio of them Student-t with the same df."""

    @pytest.mark.parametrize("n_assets", [1, 20])
    def test_portfolio_excess_kurtosis_is_the_t_value_for_any_asset_count(self, n_assets):
        """df 10: theoretical excess kurtosis 6 / (10 - 4) = 1.0 (finite 8th moment, so the
        sample estimate converges). Independent t's give 0.05-0.09 with twenty assets."""
        from scipy import stats

        r = MonteCarloSimulator(n_paths=200_000, seed=3).simulate_t_distribution(
            _factor_portfolio(n_assets), 1, df_t=10).terminal_values - 1.0
        assert abs(stats.kurtosis(r) - 1.0) < 0.25

    def test_standardised_tail_quantile_is_the_t_quantile_for_twenty_assets(self):
        """df 5, unit variance: the 1% quantile is sqrt(3/5) * t5.ppf(0.01) = -2.606
        standard deviations (normal: -2.326). Independent t's give about -2.36."""
        r = MonteCarloSimulator(n_paths=200_000, seed=4).simulate_t_distribution(
            _factor_portfolio(20), 1, df_t=5).terminal_values - 1.0
        q = np.percentile((r - r.mean()) / r.std(), 1)
        assert -2.70 < q < -2.50, q

    def test_covariance_is_preserved(self):
        """Unit-variance scaling: the simulated 1-day portfolio variance
        matches GBM's (both have covariance cov)."""
        p = _factor_portfolio(5)
        sim = MonteCarloSimulator(n_paths=200_000, seed=5)
        t = sim.simulate_t_distribution(p, 1).terminal_values
        g = sim.simulate_gbm(p, 1).terminal_values
        assert t.std() == pytest.approx(g.std(), rel=0.03)

    def test_seeded(self):
        p = _factor_portfolio(3)
        a = MonteCarloSimulator(n_paths=2000, seed=9).simulate_t_distribution(p, 5)
        b = MonteCarloSimulator(n_paths=2000, seed=9).simulate_t_distribution(p, 5)
        assert np.array_equal(a.terminal_values, b.terminal_values)

    @pytest.mark.parametrize("df_t", [2.0, 1.5, float("inf"), float("nan")])
    def test_df_without_a_finite_variance_is_rejected(self, df_t):
        """df <= 2 has no finite variance to scale to one, so the simulated covariance
        would not be the portfolio's: refused."""
        with pytest.raises(ValueError, match="df_t"):
            MonteCarloSimulator(n_paths=1000, seed=1).simulate_t_distribution(_factor_portfolio(2), 5, df_t=df_t)


def _portfolio_from(series) -> object:
    series = np.asarray(series, dtype=float)
    dates = pd.date_range("2020-01-01", periods=len(series), freq="B")
    return PortfolioLoader().from_returns_df(pd.DataFrame({"A": series}, index=dates), {"A": 1.0})


class TestBootstrapHorizon:
    """With fewer observations than one block, every block would be the same short slice, a
    252-day horizon silently 130 days and all paths identical: refused."""

    def test_rejects_history_shorter_than_one_block(self):
        p = _portfolio_from(np.random.default_rng(0).normal(0, 0.01, 10))
        with pytest.raises(ValueError, match="block_size"):
            MonteCarloSimulator(n_paths=50, seed=1).simulate_bootstrap(p, horizon_days=252)

    def test_every_path_spans_the_full_horizon(self):
        p = _portfolio_from(np.full(25, 0.001))
        r = MonteCarloSimulator(n_paths=200, seed=1, drift="historical").simulate_bootstrap(p, horizon_days=252)
        assert np.allclose(r.terminal_values, 1.001 ** 252, rtol=1e-12)

    def test_most_recent_block_can_be_drawn(self):
        """Every block start can be drawn, so the latest days (here, the only ones holding
        the +10% day) can be sampled."""
        hist = np.zeros(25)
        hist[-1] = 0.10
        p = _portfolio_from(hist)
        r = MonteCarloSimulator(n_paths=600, seed=2, drift="historical").simulate_bootstrap(p, horizon_days=20, block_size=20)
        assert np.isclose(r.terminal_values.max(), 1.10)

    @pytest.mark.parametrize("kwargs", [{"block_size": 0}, {"horizon_days": 0}])
    def test_rejects_degenerate_parameters(self, portfolio, kwargs):
        args = {"horizon_days": 20, "block_size": 20, **kwargs}
        with pytest.raises(ValueError):
            MonteCarloSimulator(n_paths=10).simulate_bootstrap(portfolio, **args)


class TestPercentileBands:
    """The engine reports real percentiles of simulated value at checkpoint days, not a
    straight line from 1.0 to the terminal percentiles (width proportional to t)."""

    @pytest.mark.parametrize("method", ["simulate_gbm", "simulate_t_distribution", "simulate_bootstrap"])
    def test_bands_are_well_formed(self, portfolio, method):
        r = getattr(MonteCarloSimulator(n_paths=2000, seed=3), method)(portfolio, 126)
        bands = r.percentile_bands
        assert bands[0] == {"day": 0, "p5": 1.0, "p25": 1.0, "p50": 1.0, "p75": 1.0, "p95": 1.0}
        assert bands[-1]["day"] == 126
        assert [b["day"] for b in bands] == sorted({b["day"] for b in bands})
        for b in bands:
            assert b["p5"] <= b["p25"] <= b["p50"] <= b["p75"] <= b["p95"]
        for k in ("p5", "p25", "p50", "p75", "p95"):
            assert bands[-1][k] == pytest.approx(r.percentile_paths[k], rel=1e-12)

    def test_gbm_band_width_grows_like_sqrt_t(self):
        """Single-asset GBM: the log-spread between p95 and p5 grows like
        sqrt(t) — about 2x from a quarter to a year, not the 4x a straight-line
        interpolation of terminal percentiles draws."""
        p = _portfolio_from(np.random.default_rng(9).normal(0.0, 0.01, 500))
        r = MonteCarloSimulator(n_paths=20_000, seed=4).simulate_gbm(p, 252)
        by_day = {b["day"]: b for b in r.percentile_bands}
        d = min(day for day in by_day if day >= 63)
        q, t = by_day[d], by_day[252]
        ratio = np.log(t["p95"] / t["p5"]) / np.log(q["p95"] / q["p5"])
        assert ratio == pytest.approx(np.sqrt(252 / d), rel=0.07)


def test_overflowing_simulation_raises_instead_of_returning_inf():
    """A simulation whose values overflow float64 used
    to return inf/nan VaR and percentiles (a raw 500 over HTTP)."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    p = PortfolioLoader().from_returns_df(
        pd.DataFrame({"A": [0.5, 0.6] * 100}, index=idx), {"A": 1.0}, allow_large_returns=True
    )
    sim = MonteCarloSimulator(n_paths=1000, seed=5, drift="historical")
    for run in (sim.simulate_gbm, sim.simulate_t_distribution, sim.simulate_bootstrap):
        with pytest.raises(ValueError, match="overflow"):
            run(p, 2520)



def test_a_checkpoint_that_overflows_inside_a_block_is_refused():
    """A path can overflow at a checkpoint and fall back under the limit by
    the block's end; the band then held inf (a raw 500 over HTTP)."""
    from riskcanvas.simulation.monte_carlo import _band

    sim = MonteCarloSimulator(n_paths=10, seed=1)
    bands = [_band(0, np.ones(10)), {"day": 5, "p5": 1.0, "p25": 1.0, "p50": 1.0, "p75": 1.0, "p95": float("inf")}]
    with pytest.raises(ValueError, match="overflow"):
        sim._make_result(np.ones(10), 5, "Block Bootstrap", bands)


def test_values_below_float_max_that_would_overflow_the_statistics_are_refused():
    """1e200 is a finite float, but its square (the standard errors) is not: `(es - var) **
    2` overflows, so the input is refused."""
    sim = MonteCarloSimulator(n_paths=1000, seed=1)
    values = np.geomspace(1e180, 1e200, 1000)
    with pytest.raises(ValueError, match="overflow"):
        sim._make_result(values, 252, "GBM", [])
    assert sim._make_result(np.geomspace(1.0, 1e100, 1000), 252, "GBM", []).standard_errors["cvar_95"] > 0


class TestMonteCarloError:
    """Each estimate carries its sampling standard error; this checks it against the spread
    of the same estimate across independent seeds."""

    KEYS = ("var_95", "var_99", "cvar_95", "cvar_99", "prob_loss", "expected_terminal")

    def test_reported_error_matches_the_spread_across_seeds(self, portfolio):
        runs = [MonteCarloSimulator(n_paths=2000, seed=s).simulate_gbm(portfolio, 20) for s in range(150)]
        for k in self.KEYS:
            empirical = np.std([getattr(r, k) for r in runs], ddof=1)
            reported = np.mean([r.standard_errors[k] for r in runs])
            assert 0.8 <= empirical / reported <= 1.25, (k, empirical, reported)

    def test_error_shrinks_like_one_over_root_n(self, portfolio):
        small = MonteCarloSimulator(n_paths=4000, seed=1).simulate_gbm(portfolio, 20)
        large = MonteCarloSimulator(n_paths=16000, seed=1).simulate_gbm(portfolio, 20)
        for k in ("var_95", "cvar_95", "expected_terminal"):
            assert 1.6 <= small.standard_errors[k] / large.standard_errors[k] <= 2.5, k

    def test_seed_is_recorded_on_the_result(self, portfolio):
        assert MonteCarloSimulator(n_paths=1000, seed=9).simulate_gbm(portfolio, 5).seed == 9

    @pytest.mark.parametrize("n", [0, -5, 2.5])
    def test_invalid_path_count_rejected(self, n):
        with pytest.raises(ValueError):
            MonteCarloSimulator(n_paths=n)


class TestCovarianceFactorHasNoJitter:
    """Nothing is added to the daily covariance diagonal: 1e-8 would give a riskless cash
    leg about 0.16% annual volatility (a 252-day terminal std of 0.0016) and a positive
    1-day VaR99."""

    @staticmethod
    def _cash(n=500):
        idx = pd.bdate_range("2021-01-01", periods=n)
        return PortfolioLoader().from_returns_df(pd.DataFrame({"CASH": [0.0001] * n}, index=idx))

    @pytest.mark.parametrize("method", ["simulate_gbm", "simulate_t_distribution"])
    def test_riskless_cash_has_no_simulated_dispersion(self, method):
        res = getattr(MonteCarloSimulator(n_paths=5000, seed=1, drift="historical"), method)(self._cash(), 252)
        assert np.ptp(res.terminal_values) < 1e-12
        assert res.terminal_values[0] == pytest.approx(np.exp(0.0001 * 252), rel=1e-12)
        assert res.var_99 < 0   # a sure gain is not a loss

    def test_singular_covariance_keeps_its_correlations(self):
        """A cash leg makes cov singular; the factor must still reproduce it exactly,
        correlations included."""
        from riskcanvas.simulation.monte_carlo import _covariance_factor

        rng = np.random.default_rng(2)
        a = rng.normal(0, 0.01, 300)
        cov = np.cov(np.vstack([a, 0.5 * a + rng.normal(0, 0.005, 300), np.zeros(300)]))
        f = _covariance_factor(cov)
        assert np.allclose(f @ f.T, cov, atol=1e-18, rtol=0)
        assert np.allclose(f[2], 0.0)


class TestCircularBootstrap:
    """The circular block bootstrap draws starts over all n observations and wraps, so
    every observation is equally likely at every day of every path. Starts limited to
    0..n-block would never draw the last block-h observations at a horizon shorter than
    a block (20 observations at h=1: one outcome, VaR95 = VaR99 = -r[0], standard errors
    0), and reordering the same returns would move 1-day VaR99 2.3% <-> 4.9%."""

    def test_one_day_draws_every_observation_equally(self):
        r = (np.arange(40) - 20) * 1e-3                    # 40 distinct, identifiable returns
        res = MonteCarloSimulator(n_paths=40_000, seed=1, drift="historical").simulate_bootstrap(_portfolio_from(r), 1)
        idx = np.rint((res.terminal_values - 1.0) / 1e-3).astype(int) + 20
        counts = np.bincount(idx, minlength=40)
        assert counts.min() > 850 and counts.max() < 1150, counts   # ~1000 each
        assert counts[-19:].min() > 850                               # the last block-1 days included

    def test_every_day_of_a_multi_block_path_is_uniform_over_history(self):
        """E[log terminal] = h * mean(log(1 + r)) when every path-day is a
        uniform draw. The last 19 of these 40 days are the gains, which the
        old start range under-sampled."""
        r = np.r_[np.full(21, -0.002), np.full(19, 0.02)]
        h = 25
        res = MonteCarloSimulator(n_paths=20_000, seed=2, drift="historical").simulate_bootstrap(_portfolio_from(r), h)
        assert np.log(res.terminal_values).mean() == pytest.approx(h * np.log1p(r).mean(), rel=0.03)

    def test_one_day_var_does_not_depend_on_the_order_of_history(self):
        r = np.random.default_rng(2).standard_t(4, 260) * 0.012
        sim = MonteCarloSimulator(n_paths=50_000, seed=3)
        a = sim.simulate_bootstrap(_portfolio_from(np.sort(r)), 1)
        b = sim.simulate_bootstrap(_portfolio_from(np.sort(r)[::-1]), 1)
        se = np.hypot(a.standard_errors["var_99"], b.standard_errors["var_99"])
        assert abs(a.var_99 - b.var_99) <= 4 * se + 1e-12, (a.var_99, b.var_99, se)
        assert a.var_99 == pytest.approx(-np.percentile(r, 1), rel=0.1)

    def test_twenty_observations_one_day_is_the_whole_history(self):
        r = np.random.default_rng(1).normal(0.0005, 0.01, 20)
        res = MonteCarloSimulator(n_paths=10_000, seed=42).simulate_bootstrap(_portfolio_from(r), 1)
        assert np.unique(res.terminal_values).size == 20
        assert res.var_99 >= res.var_95 and res.standard_errors["var_95"] > 0


class TestWeightsAlignByName:
    """Means, covariance and weights align by ticker. Paired by position, a Portfolio whose
    `returns` columns and `symbols` differ in order would give each weight another
    asset's returns: 99% cash would simulate a 10-day VaR99 of ~20% (vs ~0.2%)."""

    @staticmethod
    def _pair(order):
        from riskcanvas.portfolio.loader import Portfolio

        idx = pd.bdate_range("2020-01-01", periods=500)
        rng = np.random.default_rng(6)
        rets = pd.DataFrame({"CASH": [0.0001] * 500, "EQ": rng.normal(0.0004, 0.03, 500)}, index=idx)[order]
        return Portfolio(symbols=["EQ", "CASH"], weights={"EQ": 0.01, "CASH": 0.99},
                         prices=100 * (1 + rets).cumprod(), returns=rets)

    @pytest.mark.parametrize("method", ["simulate_gbm", "simulate_t_distribution", "simulate_bootstrap"])
    def test_column_order_does_not_change_the_simulation(self, method):
        shuffled = getattr(MonteCarloSimulator(n_paths=5000, seed=7), method)(self._pair(["CASH", "EQ"]), 10)
        aligned = getattr(MonteCarloSimulator(n_paths=5000, seed=7), method)(self._pair(["EQ", "CASH"]), 10)
        assert np.array_equal(shuffled.terminal_values, aligned.terminal_values)
        assert shuffled.var_99 < 0.01   # 1% in a 3%-vol asset: about 0.2% over 10 days


class TestAssetOrderDoesNotMoveTheResult:
    """Which asset takes which draw must not follow the order of the weights: the same
    portfolio and seed give the same VaR with the keys reordered."""

    IDX = pd.bdate_range("2021-01-04", periods=300)
    RNG = np.random.default_rng(11)
    RETURNS = pd.DataFrame({"A": RNG.normal(0, 0.03, 300), "B": RNG.normal(0, 0.01, 300), "C": RNG.normal(0, 0.02, 300)}, index=IDX)

    def _result(self, method, order):
        weights = {k: {"A": 0.5, "B": 0.25, "C": 0.25}[k] for k in order}
        p = PortfolioLoader().from_returns_df(self.RETURNS[list(order)], weights=weights)
        return getattr(MonteCarloSimulator(n_paths=2000, seed=42), method)(p, 20)

    @pytest.mark.parametrize("method", ["simulate_gbm", "simulate_t_distribution", "simulate_bootstrap"])
    @pytest.mark.parametrize("order", ["BAC", "CBA", "CAB"])
    def test_every_model_gives_identical_paths(self, method, order):
        assert np.array_equal(self._result(method, order).terminal_values, self._result(method, "ABC").terminal_values)


def _trending(n=504, mean=0.0012, vol=0.008, seed=8):
    """Two years at exactly +0.12% a day, 0.8% daily volatility."""
    z = np.random.default_rng(seed).standard_normal(n)
    return _portfolio_from(mean + vol * (z - z.mean()) / z.std(ddof=1))


class TestDrift:
    """Zero drift, the risk-desk convention, is the default; `historical` and `risk_free`
    are explicit choices. In-sample drift makes two strong years project a one-year tail
    that is a GAIN (here: GBM VaR95 -8.8%, VaR99 +0.2%, P(loss) 1.0%)."""

    @pytest.mark.parametrize("method", ["simulate_gbm", "simulate_t_distribution", "simulate_bootstrap"])
    def test_default_drift_is_zero_and_the_tail_stays_a_loss(self, method):
        p = _trending()
        default = getattr(MonteCarloSimulator(n_paths=20_000, seed=5), method)(p, 252)
        hist = getattr(MonteCarloSimulator(n_paths=20_000, seed=5, drift="historical"), method)(p, 252)
        assert default.drift == "zero" and hist.drift == "historical"
        assert default.var_99 > 0.15 and default.prob_loss > 0.4
        assert hist.var_95 < -0.05 and hist.prob_loss < 0.03   # the 95% tail is a gain

    @pytest.mark.parametrize("method", ["simulate_gbm", "simulate_bootstrap"])
    def test_zero_drift_has_unit_expected_terminal(self, method):
        res = getattr(MonteCarloSimulator(n_paths=40_000, seed=6), method)(_trending(), 63)
        assert abs(res.expected_terminal - 1.0) < 4 * res.standard_errors["expected_terminal"]

    def test_risk_free_drift_grows_at_two_percent_a_year(self):
        res = MonteCarloSimulator(n_paths=40_000, seed=7, drift="risk_free").simulate_bootstrap(_trending(), 252)
        assert abs(res.expected_terminal - (1 + 0.02 / 252) ** 252) < 4 * res.standard_errors["expected_terminal"]

    def test_unknown_drift_is_rejected(self):
        with pytest.raises(ValueError, match="drift"):
            MonteCarloSimulator(drift="sample")

    def test_recentring_never_creates_a_loss_beyond_total(self):
        p = _portfolio_from(np.r_[-1.0, np.full(39, 0.05)])
        with pytest.raises(ValueError, match="more than 100%"):
            MonteCarloSimulator(n_paths=1000, seed=1).simulate_bootstrap(p, 5)


def test_readme_drift_example_is_the_engine():
    """README "Drift is zero by default": the one-year GBM figures quoted for
    the two-year +0.12%/day sample (20,000 paths, seed 5)."""
    from pathlib import Path

    readme = " ".join((Path(__file__).resolve().parents[1] / "README.md").read_text().split())
    zero = MonteCarloSimulator(n_paths=20_000, seed=5).simulate_gbm(_trending(), 252)
    hist = MonteCarloSimulator(n_paths=20_000, seed=5, drift="historical").simulate_gbm(_trending(), 252)
    quoted = (f"{zero.var_99:.1%}", f"{hist.var_99:+.1%}", f"{hist.var_95:.1%}", f"{hist.prob_loss:.1%}")
    assert quoted == ("26.3%", "+0.2%", "-8.8%", "1.0%"), quoted
    assert ("one-year GBM VaR 99% is 26.3% at zero drift and +0.2% with historical drift "
            "(VaR 95% -8.8%, a gain; P(loss) 1.0%)") in readme


def test_es_standard_error_matches_the_seed_spread_with_ties():
    """With ties at the VaR the ES standard error uses the worst ceil(n * a) losses, the
    outcomes the Acerbi-Tasche ES averages. Using every loss >= VaR around a diluted ES
    reads 0.0012 against an empirical spread of 0.0057 (measured 2026-11-02)."""
    idx = pd.bdate_range("2020-01-01", periods=100)
    p = PortfolioLoader().from_returns_df(pd.DataFrame({"A": [0.0] * 97 + [-0.10] * 3}, index=idx))
    runs = [MonteCarloSimulator(n_paths=4000, seed=s, drift="historical").simulate_bootstrap(p, 1) for s in range(60)]
    spread = np.std([r.cvar_95 for r in runs], ddof=1)
    reported = np.mean([r.standard_errors["cvar_95"] for r in runs])
    assert reported == pytest.approx(spread, rel=0.3), (reported, spread)


def test_readme_t5_ratios_at_95_and_99_percent():
    """README "How much fatter, by horizon": at 1 day the unit-variance t5's
    95% VaR is below GBM's (0.95) while its 99% VaR is above (1.12)."""
    from pathlib import Path

    readme = " ".join((Path(__file__).resolve().parents[1] / "README.md").read_text().split())
    sim = MonteCarloSimulator(n_paths=50_000, seed=42)
    p = _usage_example_portfolio()
    t, g = sim.simulate_t_distribution(p, 1), sim.simulate_gbm(p, 1)
    assert (f"{t.var_95 / g.var_95:.2f}", f"{t.var_99 / g.var_99:.2f}") == ("0.95", "1.12")
    assert "**0.95 at 1 day**" in readme and "**1.12 at 1 day," in readme


class TestPartiallyInvested:
    """With normalize_weights=False (a partially invested portfolio, as the loader
    documents) the residual is cash: left out of the GBM and Student-t blend, the
    uninvested half would lose 100% every day (10-day VaR99 0.9991 against the
    bootstrap's 0.0231). Leverage (weights above 100%) is refused."""

    @staticmethod
    def _frame():
        idx = pd.bdate_range("2020-01-01", periods=500)
        rng = np.random.default_rng(1)
        return pd.DataFrame({"A": rng.normal(0.0004, 0.012, 500), "B": rng.normal(0.0003, 0.01, 500)}, index=idx)

    def test_half_invested_is_half_the_risk_in_every_model(self):
        half = PortfolioLoader().from_returns_df(self._frame(), {"A": 0.25, "B": 0.25}, normalize_weights=False)
        full = PortfolioLoader().from_returns_df(self._frame(), {"A": 0.5, "B": 0.5})
        var = {}
        for m in ("simulate_gbm", "simulate_t_distribution", "simulate_bootstrap"):
            h = getattr(MonteCarloSimulator(n_paths=20_000, seed=1), m)(half, 10).var_99
            f = getattr(MonteCarloSimulator(n_paths=20_000, seed=1), m)(full, 10).var_99
            assert h == pytest.approx(f / 2, rel=0.1), (m, h, f)
            var[m] = h
        assert max(var.values()) < 1.3 * min(var.values()), var   # the models agree

    def test_cash_earns_the_risk_free_rate_under_risk_free_drift(self):
        cash_only_ish = PortfolioLoader().from_returns_df(self._frame(), {"A": 1e-9, "B": 0.0}, normalize_weights=False)
        res = MonteCarloSimulator(n_paths=2000, seed=2, drift="risk_free").simulate_gbm(cash_only_ish, 252)
        assert res.expected_terminal == pytest.approx((1 + 0.02 / 252) ** 252, rel=1e-6)

    def test_leverage_is_refused(self):
        with pytest.raises(ValueError, match="leverage"):
            PortfolioLoader().from_returns_df(self._frame(), {"A": 0.75, "B": 0.75}, normalize_weights=False)


class TestRiskFreeUsesTheSeriesOwnPeriods:
    """drift="risk_free" divided 2% by 252 on every series, so a riskless leg
    of a 7-day series earned 2.94% over 365 days instead of the stated 2%."""

    CAL = pd.date_range("2022-01-01", periods=730, freq="D")

    def _riskless(self, idx):
        return PortfolioLoader().from_returns_df(pd.DataFrame({"C": [0.0] * len(idx)}, index=idx))

    @pytest.mark.parametrize("run", ["simulate_gbm", "simulate_t_distribution", "simulate_bootstrap"])
    def test_a_riskless_leg_earns_two_percent_a_year_on_a_7_day_series(self, run):
        r = getattr(MonteCarloSimulator(1000, seed=1, drift="risk_free"), run)(self._riskless(self.CAL), 365)
        assert r.expected_terminal == pytest.approx((1 + 0.02 / 365) ** 365, rel=1e-5)

    def test_uninvested_cash_earns_it_too(self):
        idx = self.CAL
        p = PortfolioLoader().from_returns_df(
            pd.DataFrame({"C": [0.0] * len(idx)}, index=idx), weights={"C": 0.5}, normalize_weights=False)
        r = MonteCarloSimulator(1000, seed=1, drift="risk_free").simulate_gbm(p, 365)
        assert r.expected_terminal == pytest.approx((1 + 0.02 / 365) ** 365, rel=1e-5)

    def test_a_trading_day_series_keeps_252(self):
        idx = pd.bdate_range("2021-01-01", periods=730)
        r = MonteCarloSimulator(1000, seed=1, drift="risk_free").simulate_gbm(self._riskless(idx), 252)
        assert r.expected_terminal == pytest.approx((1 + 0.02 / 252) ** 252, rel=1e-5)

    def test_a_measured_calendar_earns_two_percent_a_year_too(self):
        idx = pd.date_range("2020-01-01", periods=730, freq="2D")   # about 183 a year
        r = MonteCarloSimulator(1000, seed=1, drift="risk_free").simulate_gbm(self._riskless(idx), 183)
        assert r.expected_terminal == pytest.approx((1 + 0.02 / 183) ** 183, rel=1e-5)

    def test_only_risk_free_needs_the_periods(self):
        """A calendar that cannot be measured is refused where it matters, and
        only there."""
        idx = pd.date_range("2020-01-01", periods=200, freq="h")
        p = PortfolioLoader().from_returns_df(
            pd.DataFrame({"A": np.random.default_rng(1).normal(0, 0.01, 200)}, index=idx))
        assert MonteCarloSimulator(1000, seed=1).simulate_gbm(p, 10).var_95 > 0
        with pytest.raises(ValueError, match="Cannot infer"):
            MonteCarloSimulator(1000, seed=1, drift="risk_free").simulate_gbm(p, 10)


@pytest.mark.parametrize("method", ["simulate_gbm", "simulate_t_distribution", "simulate_bootstrap"])
class TestHorizonIsAWholeNumberOfSteps:
    """simulate_gbm(p, 0) has no steps, and VaR 0.0 with P(loss) 0 would read as no risk:
    refused."""

    @pytest.mark.parametrize("horizon", [0, -1, 1.5])
    def test_refused(self, simulator, portfolio, method, horizon):
        with pytest.raises(ValueError, match="horizon_days must be a whole number"):
            getattr(simulator, method)(portfolio, horizon)

    def test_a_whole_float_is_a_horizon(self, simulator, portfolio, method):
        result = getattr(simulator, method)(portfolio, 5.0)
        assert result.horizon_days == 5 and result.var_95 > 0
