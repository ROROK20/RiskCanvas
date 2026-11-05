"""Tests for risk metrics — numerical accuracy validation."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from riskcanvas.metrics.risk import RiskMetrics, RiskReport, observations_per_year, periods_per_year
from riskcanvas.metrics.performance import PerformanceMetrics, PerformanceReport
from riskcanvas.portfolio.loader import PortfolioLoader


def make_returns(n: int = 500, seed: int = 42, annual_return: float = 0.10) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily = (1 + annual_return) ** (1 / 252) - 1
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(daily + rng.normal(0, 0.01, n), index=dates)


def make_trending_up(n: int = 252) -> pd.Series:
    """Monotonically increasing — zero drawdown."""
    return pd.Series([0.001] * n, index=pd.date_range("2020-01-01", periods=n, freq="B"))


def make_all_loss(n: int = 100) -> pd.Series:
    return pd.Series([-0.005] * n, index=pd.date_range("2020-01-01", periods=n, freq="B"))


class TestRiskMetrics:
    def test_var_historical_positive(self):
        rm = RiskMetrics(make_returns())
        assert rm.var_historical(0.95) > 0

    def test_var_higher_at_higher_confidence(self):
        rm = RiskMetrics(make_returns())
        assert rm.var_historical(0.99) >= rm.var_historical(0.95)

    def test_cvar_ge_var(self):
        """CVaR must be >= VaR by definition."""
        rm = RiskMetrics(make_returns())
        assert rm.cvar_historical(0.95) >= rm.var_historical(0.95) - 1e-9

    def test_var_parametric_formula(self):
        """Manual calculation: VaR = -(mu + z*sigma) for normal."""
        from scipy.stats import norm
        returns = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 500))
        rm = RiskMetrics(returns)
        mu = returns.mean()
        sigma = returns.std()
        z = norm.ppf(0.05)
        expected = -(mu + z * sigma)
        assert rm.var_parametric(0.95) == pytest.approx(expected, rel=1e-5)

    def test_max_drawdown_zero_for_monotone_up(self):
        rm = RiskMetrics(make_trending_up())
        _, mdd = rm.max_drawdown()
        assert mdd == pytest.approx(0.0, abs=1e-9)

    def test_max_drawdown_negative(self):
        rm = RiskMetrics(make_returns())
        _, mdd = rm.max_drawdown()
        assert mdd <= 0

    def test_known_drawdown(self):
        """Manual: returns of [+10%, -20%, +5%] → dd at idx 1 = (0.88 - 1.10) / 1.10 ≈ -0.2"""
        returns = pd.Series([0.10, -0.20, 0.05],
                            index=pd.date_range("2020-01-01", periods=3, freq="B"))
        rm = RiskMetrics(returns)
        dd_series, _ = rm.max_drawdown()
        # After 10% gain then 20% loss: equity = 1.1 * 0.8 = 0.88, rolling max = 1.1
        # dd = (0.88 - 1.10) / 1.10 ≈ -0.2
        assert dd_series.iloc[1] == pytest.approx((1.10 * 0.80 - 1.10) / 1.10, rel=1e-6)

    def test_compute_all_returns_report(self):
        rm = RiskMetrics(make_returns())
        report = rm.compute_all()
        assert isinstance(report, RiskReport)
        assert report.annual_vol > 0
        assert report.cvar_95 >= report.var_95_1d - 1e-9

    def test_tail_ratio_positive(self):
        rm = RiskMetrics(make_returns())
        assert rm.tail_ratio() > 0

    def test_omega_ratio_positive_for_positive_returns(self):
        rm = RiskMetrics(make_returns(annual_return=0.15))
        assert rm.omega_ratio() > 1.0


class TestPerformanceMetrics:
    def test_sharpe_formula(self):
        """Manual Sharpe: (mean_excess / std_excess) * sqrt(252)."""
        returns = make_returns()
        pm = PerformanceMetrics(returns, risk_free_rate=0.02)
        rfr_d = 0.02 / 252
        excess = returns - rfr_d
        expected = float(excess.mean() / excess.std() * np.sqrt(252))
        report = pm.compute_all()
        assert report.sharpe == pytest.approx(expected, rel=1e-4)

    def test_calmar_ratio_formula(self):
        """Calmar = CAGR / |max_drawdown|."""
        returns = make_returns()
        pm = PerformanceMetrics(returns)
        report = pm.compute_all()
        assert abs(report.calmar - report.cagr / abs(report.max_drawdown)) < 1e-6

    def test_positive_cagr_for_positive_drift(self):
        returns = make_returns(annual_return=0.12)
        pm = PerformanceMetrics(returns)
        report = pm.compute_all()
        assert report.cagr > 0

    def test_win_rate_between_zero_one(self):
        pm = PerformanceMetrics(make_returns())
        report = pm.compute_all()
        assert 0 <= report.win_rate <= 1

    def test_rolling_sharpe_length(self):
        pm = PerformanceMetrics(make_returns())
        rs = pm.rolling_sharpe(window=63)
        assert len(rs) == 500

    def test_factor_exposure_alpha(self):
        returns = make_returns()
        factors = pd.DataFrame(
            {"market": make_returns(seed=99).values},
            index=returns.index
        )
        pm = PerformanceMetrics(returns)
        betas = pm.factor_exposure(factors)
        assert "alpha" in betas.index
        assert "market" in betas.index


def _dated(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="B"))


class TestDownsideDeviation:
    """The downside deviation is the standard sqrt(mean(min(r - T, 0)^2)) over ALL
    observations. The std of only the below-target days would inflate Sortino ~15% on
    realistic data."""

    RETURNS = [0.02, -0.01, 0.03, -0.02, 0.01]

    def test_sortino_hand_computed(self):
        # T = 0: shortfalls 0, -0.01, 0, -0.02, 0 -> mean square 1e-4 -> DD 0.01.
        # Sortino = 0.006 * 252 / (0.01 * sqrt(252)) = 0.6 * sqrt(252).
        report = PerformanceMetrics(_dated(self.RETURNS), risk_free_rate=0.0).compute_all()
        assert report.sortino == pytest.approx(0.6 * np.sqrt(252), rel=1e-12)

    def test_downside_vol_hand_computed(self):
        report = RiskMetrics(_dated(self.RETURNS), risk_free_rate=0.0).compute_all()
        assert report.downside_vol == pytest.approx(0.01 * np.sqrt(252), rel=1e-12)

    def test_downside_vol_is_zero_without_shortfall(self):
        """With fewer than two days below target it does not fall back to the TOTAL annual
        vol, a different quantity under the same name."""
        report = RiskMetrics(make_trending_up(), risk_free_rate=0.0).compute_all()
        assert report.downside_vol == 0.0


class TestNonFiniteRatios:
    """All-positive returns made Sortino / profit factor
    +inf, which is not JSON and turned /portfolio/analyze into a raw 500."""

    def test_all_positive_report_serialises_with_nulls(self):
        import json

        r = _dated([0.001 + 0.0001 * (i % 3) for i in range(300)])
        report = PerformanceMetrics(r).compute_all()
        assert report.sortino == float("inf")
        assert report.profit_factor == float("inf")
        assert report.calmar == float("inf")
        d = report.to_dict()
        json.dumps(d, allow_nan=False)
        assert d["sortino"] is None and d["profit_factor"] is None and d["calmar"] is None
        json.dumps(RiskMetrics(r).compute_all().to_dict(), allow_nan=False)

    def test_avg_loss_is_zero_not_a_sentinel_without_losses(self):
        report = PerformanceMetrics(make_trending_up()).compute_all()
        assert report.avg_loss == 0.0


class TestDrawdownFromStartingCapital:
    """The running peak starts at the starting capital, so a loss on day one counts: [-30%,
    +1%, +1%] has max_drawdown -30%, next to total_return -28.6%, and a -100% first day
    reads -100%. Calmar and CDaR inherit it. test_known_drawdown above starts with a
    +10% day and cannot see this."""

    def test_day_one_loss_is_the_max_drawdown(self):
        r = _dated([-0.30, 0.01, 0.01])
        dd, mdd = RiskMetrics(r).max_drawdown()
        assert mdd == pytest.approx(-0.30, abs=1e-12)
        # Hand-computed from capital 1.0: 0.70, 0.707, 0.71407 against peak 1.0.
        assert dd.tolist() == pytest.approx([-0.30, 0.707 - 1.0, 0.71407 - 1.0], abs=1e-12)
        report = RiskMetrics(r).compute_all()
        assert report.current_drawdown == pytest.approx(0.71407 - 1.0, abs=1e-12)
        assert report.max_drawdown_duration_days == 3

    def test_performance_report_agrees_and_calmar_is_negative(self):
        r = _dated([-0.30, 0.01, 0.01])
        perf = PerformanceMetrics(r).compute_all()
        assert perf.max_drawdown == pytest.approx(-0.30, abs=1e-12)
        assert perf.total_return == pytest.approx(0.71407 - 1.0, abs=1e-12)
        assert perf.calmar == pytest.approx(perf.cagr / 0.30, rel=1e-12)
        assert perf.calmar < 0

    def test_drawdown_never_shallower_than_the_total_loss(self):
        """Running from capital 1.0, the deepest point is at least as deep as
        the final loss whenever the path ends below its starting capital."""
        for path in ([-0.30, 0.01, 0.01], [-0.05, -0.05, 0.02, -0.01], [-0.2, 0.1, 0.05]):
            r = _dated(path)
            total = PerformanceMetrics(r).compute_all().total_return
            _, mdd = RiskMetrics(r).max_drawdown()
            assert mdd <= min(total, 0.0) + 1e-12, path

    def test_total_loss_path_is_minus_one_not_null(self):
        r = _dated([-1.0, 0.01, 0.02])
        risk = RiskMetrics(r).compute_all()
        perf = PerformanceMetrics(r).compute_all()
        assert risk.max_drawdown == -1.0 and risk.current_drawdown == -1.0
        assert perf.max_drawdown == -1.0 and perf.calmar == pytest.approx(-1.0)
        assert risk.to_dict()["max_drawdown"] == -1.0

    def test_mid_path_total_loss(self):
        r = _dated([0.10, -1.0, 0.05])
        dd, mdd = RiskMetrics(r).max_drawdown()
        assert mdd == -1.0
        assert dd.tolist() == [0.0, -1.0, -1.0]

    def test_conditional_drawdown_at_risk_sees_the_day_one_loss(self):
        rm = RiskMetrics(_dated([-0.30] + [0.0] * 19))
        assert rm.conditional_drawdown_at_risk(0.95) == pytest.approx(-0.30, abs=1e-12)

    def test_long_compounding_does_not_overflow_to_null(self):
        """Equity of 11^5000 overflows a cumprod; the log-space drawdown stays
        exactly 0 for an always-rising path."""
        r = _dated([10.0] * 5000)
        assert RiskMetrics(r).compute_all().max_drawdown == 0.0
        assert PerformanceMetrics(r).compute_all().max_drawdown == 0.0

    def test_return_below_minus_one_is_rejected(self):
        with pytest.raises(ValueError):
            RiskMetrics(_dated([0.01, -1.5, 0.02])).max_drawdown()



class TestParametricCVaR:
    """Closed-form normal expected shortfall, as the README states."""

    def test_standard_normal_value(self):
        """mu 0, sigma 0.01, 95%: ES = phi(1.6449) / 0.05 * 0.01 = 0.020627128.
        (TestLossCappedAtTotalLoss covers sigma 1, a 206% loss that the total-loss cap
        reports as 1.0.)"""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(1000)
        x = 0.01 * (x - x.mean()) / x.std(ddof=1)  # sample mean 0, sample std 0.01 exactly
        assert RiskMetrics(pd.Series(x)).cvar_parametric(0.95) == pytest.approx(0.020627128, abs=1e-8)

    @pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
    def test_equals_the_conditional_normal_tail_mean(self, conf):
        from scipy.stats import norm

        r = make_returns()
        mu, sigma = r.mean(), r.std()
        q = norm.ppf(1 - conf, loc=mu, scale=sigma)
        expected = -norm.expect(lambda v: v, loc=mu, scale=sigma, ub=q, conditional=True)
        assert RiskMetrics(r).cvar_parametric(conf) == pytest.approx(expected, rel=1e-8)

    @pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
    def test_at_least_the_parametric_var(self, conf):
        rm = RiskMetrics(make_returns())
        assert rm.cvar_parametric(conf) > rm.var_parametric(conf)


class TestHandComputedVaR:
    """Historical VaR / CVaR against values worked out by hand. 21 observations, so numpy's
    linear-interpolation percentile at p lands on sorted index p/100 * 20: 5% -> index
    1.0, 10% -> 2.0, 1% -> 0.2."""

    R = [-0.05, -0.03, -0.01] + [0.0] * 9 + [0.01] * 9

    def test_historical_var(self):
        rm = RiskMetrics(_dated(self.R))
        assert rm.var_historical(0.95) == pytest.approx(0.03, abs=1e-15)    # sorted[1]
        assert rm.var_historical(0.90) == pytest.approx(0.01, abs=1e-15)    # sorted[2]
        # sorted[0] + 0.2 * (sorted[1] - sorted[0]) = -0.05 + 0.2 * 0.02 = -0.046
        assert rm.var_historical(0.99) == pytest.approx(0.046, abs=1e-15)

    def test_historical_cvar(self):
        rm = RiskMetrics(_dated(self.R))
        # Acerbi-Tasche, worst k = 21 * a:
        assert rm.cvar_historical(0.95) == pytest.approx(0.0515 / 1.05, abs=1e-15)  # k 1.05
        assert rm.cvar_historical(0.90) == pytest.approx(0.081 / 2.1, abs=1e-15)    # k 2.1
        assert rm.cvar_historical(0.99) == pytest.approx(0.05, abs=1e-15)           # k 0.21

    def test_parametric_var_and_cvar_at_one_percent_vol(self):
        """Sample mean 0, sample std 0.01: VaR95 = 1.6448536 * 0.01 and
        ES95 = phi(1.6448536) / 0.05 * 0.01 = 2.0627128 * 0.01."""
        x = np.random.default_rng(0).standard_normal(1000)
        x = 0.01 * (x - x.mean()) / x.std(ddof=1)
        rm = RiskMetrics(pd.Series(x))
        assert rm.var_parametric(0.95) == pytest.approx(0.016448536, abs=1e-9)
        assert rm.cvar_parametric(0.95) == pytest.approx(0.020627128, abs=1e-9)

    def test_ten_day_is_root_ten_times_one_day(self):
        rm = RiskMetrics(_dated(self.R))
        assert rm.var_historical_scaled(0.95, 10) == pytest.approx(0.03 * np.sqrt(10), abs=1e-15)
        assert rm.compute_all().var_95_10d == pytest.approx(0.03 * np.sqrt(10), abs=1e-15)


class TestZeroVolatilityIsContinuous:
    """One formula applies to every series: VaR = minus the quantile (negative when even
    the tail quantile is a gain), the same convention as Monte Carlo VaR. So a constant
    +1% series reads -1% historical and parametric alike (-10,000 per 1,000,000 over
    HTTP), and VaR does not jump as vol crosses the 1e-12 cutoff."""

    @staticmethod
    def _wobble(level, k, n=100):
        return _dated([level + (k if i % 2 else -k) for i in range(n)])

    def test_constant_series_historical_and_parametric_agree(self):
        rm = RiskMetrics(_dated([0.01] * 50))
        for conf in (0.90, 0.95, 0.99):
            assert rm.var_historical(conf) == pytest.approx(-0.01, abs=1e-15)
            assert rm.cvar_historical(conf) == pytest.approx(-0.01, abs=1e-15)
            assert rm.var_parametric(conf) == pytest.approx(-0.01, abs=1e-15)
            assert rm.cvar_parametric(conf) == pytest.approx(-0.01, abs=1e-15)

    def test_no_jump_at_the_degenerate_cutoff(self):
        below = RiskMetrics(self._wobble(0.01, 1e-14))   # daily vol < 1e-12
        above = RiskMetrics(self._wobble(0.01, 1e-11))   # daily vol > 1e-12
        assert float(below.returns.std()) < RiskMetrics._VOL_EPS < float(above.returns.std())
        for f in ("var_historical", "cvar_historical", "var_parametric", "cvar_parametric"):
            assert getattr(below, f)(0.95) == pytest.approx(getattr(above, f)(0.95), abs=1e-9), f

    def test_report_for_a_constant_series(self):
        report = RiskMetrics(_dated([0.01] * 50)).compute_all()
        assert report.var_95_1d == pytest.approx(-0.01, abs=1e-15)
        assert report.cvar_99 == pytest.approx(-0.01, abs=1e-15)
        # Undefined for a constant series.
        assert np.isnan(report.skewness) and np.isnan(report.excess_kurtosis)
        assert np.isnan(report.var_stress_multiplier)

    def test_losing_constant_series_is_unchanged(self):
        rm = RiskMetrics(_dated([-0.005] * 50))
        assert rm.var_historical(0.95) == pytest.approx(0.005, abs=1e-15)
        assert rm.var_parametric(0.95) == pytest.approx(0.005, abs=1e-15)


class TestLossCappedAtTotalLoss:
    """Sqrt-of-time scaling put the 10-day VaR of a
    long-only portfolio at 316% (var_95_10d 3.16). Returns are >= -100%, so no
    horizon can lose more than everything; extrapolations past a total loss
    (sqrt-time, the unbounded normal tail) are capped at 1.0."""

    WILD = _dated([10.0, -0.9999] * 50)

    def test_ten_day_var_is_capped(self):
        rm = RiskMetrics(self.WILD)
        assert rm.var_historical(0.95) == pytest.approx(0.9999)
        assert rm.var_historical_scaled(0.95, 10) == 1.0
        report = rm.compute_all()
        assert report.var_95_10d == 1.0 and report.var_99_10d == 1.0

    def test_parametric_var_and_cvar_are_capped(self):
        rm = RiskMetrics(self.WILD)
        for conf in (0.90, 0.95, 0.99):
            assert rm.var_parametric(conf) <= 1.0
            assert rm.cvar_parametric(conf) == 1.0
        assert rm.var_parametric(0.99) == 1.0

    def test_unit_daily_vol_parametric_cvar_hits_the_cap(self):
        """The closed form gives 2.0627 (a 206% loss) at a daily vol of 100%."""
        x = np.random.default_rng(0).standard_normal(1000)
        rm = RiskMetrics(pd.Series((x - x.mean()) / x.std(ddof=1)))
        assert rm.cvar_parametric(0.95) == 1.0 and rm.var_parametric(0.95) == 1.0

    def test_horizon_must_be_a_positive_day_count(self):
        with pytest.raises(ValueError):
            RiskMetrics(self.WILD).var_historical_scaled(0.95, 0)


class TestNoInventedValues:
    """No helper invents a value. Undefined is NaN (JSON null over the wire); misaligned
    input is an error."""

    def test_information_ratio_rejects_a_benchmark_missing_dates(self):
        r = make_returns(100)
        bench = make_returns(100, seed=7).iloc[:-5]   # last 5 days missing
        with pytest.raises(ValueError, match="5 of the portfolio's 100 dates"):
            RiskMetrics(r).information_ratio(bench)

    def test_information_ratio_hand_computed(self):
        r = _dated([0.02, -0.01, 0.03, -0.02, 0.01])
        b = _dated([0.01, -0.01, 0.01, -0.01, 0.00])
        excess = np.array([0.01, 0.0, 0.02, -0.01, 0.01])
        expected = excess.mean() * 252 / (excess.std(ddof=1) * np.sqrt(252))
        assert RiskMetrics(r).information_ratio(b) == pytest.approx(expected, rel=1e-12)

    def test_information_ratio_is_undefined_without_tracking_error(self):
        r = make_returns(100)
        assert np.isnan(RiskMetrics(r).information_ratio(r))

    def test_rolling_sharpe_warm_up_is_undefined_not_zero(self):
        rs = PerformanceMetrics(make_returns(500)).rolling_sharpe(window=63)
        assert rs.iloc[:62].isna().all() and rs.iloc[62:].notna().all()

    def test_tail_ratio_is_undefined_when_the_5th_percentile_is_zero(self):
        r = _dated([0.0] * 20 + [0.01, 0.02] * 10)
        assert np.isnan(RiskMetrics(r).tail_ratio())

    def test_conditional_drawdown_at_risk_uses_the_exact_tail_percentile(self):
        """Drawdowns -5%, -3%, -1% then 18 days at the peak: CDaR90 is the mean of the
        worst 10% of 21 days, sorted index 0..2 = (-0.05 - 0.03 - 0.01) / 3. The raw (1
        - 0.90) * 100 would drop -1%."""
        equity = np.r_[0.95, 0.97, 0.99, np.ones(18)]
        r = _dated(np.diff(np.r_[1.0, equity]) / np.r_[1.0, equity][:-1])
        dd, _ = RiskMetrics(r).max_drawdown()
        assert dd.iloc[:3].tolist() == pytest.approx([-0.05, -0.03, -0.01], abs=1e-12)
        # Acerbi-Tasche: worst 2.1 of 21 days.
        assert RiskMetrics(r).conditional_drawdown_at_risk(0.90) == pytest.approx(-0.081 / 2.1, abs=1e-12)


class TestFactorExposure:
    """`factor_exposure` is one OLS regression on factor series the caller supplies."""

    def _factors(self, n=250, seed=3):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame({"MKT": rng.normal(0, 0.01, n), "SMB": rng.normal(0, 0.005, n)}, index=idx)

    def test_recovers_an_exact_linear_model(self):
        f = self._factors()
        r = 0.0002 + 1.3 * f["MKT"] - 0.4 * f["SMB"]
        b = PerformanceMetrics(r).factor_exposure(f)
        assert list(b.index) == ["alpha", "MKT", "SMB"]
        assert b.to_numpy() == pytest.approx([0.0002, 1.3, -0.4], abs=1e-12)

    def test_matches_the_normal_equations_with_noise(self):
        f = self._factors()
        r = 0.0001 + 0.9 * f["MKT"] + pd.Series(np.random.default_rng(9).normal(0, 0.004, 250), index=f.index)
        X = np.column_stack([np.ones(250), f.to_numpy()])
        expected = np.linalg.solve(X.T @ X, X.T @ r.to_numpy())
        assert PerformanceMetrics(r).factor_exposure(f).to_numpy() == pytest.approx(expected, rel=1e-9)

    def test_factors_missing_portfolio_dates_are_an_error(self):
        f = self._factors()
        r = 0.5 * f["MKT"]
        with pytest.raises(ValueError, match="3 of the portfolio's 250 dates"):
            PerformanceMetrics(r).factor_exposure(f.iloc[3:])

    def test_too_few_observations_is_an_error(self):
        f = self._factors(n=3)
        with pytest.raises(ValueError, match="more than 3 observations"):
            PerformanceMetrics(f["MKT"]).factor_exposure(f)


class TestExpectedShortfallWithTies:
    """ES is not the mean of every observation <= VaR, where ties at the VaR dilute the
    tail: [0] * 97 + [-0.10] * 3 would give CVaR95 = CVaR90 = 0.003 instead of 0.06 /
    0.03."""

    TIES = _dated([0.0] * 97 + [-0.10] * 3)

    def test_ties_at_the_var_do_not_dilute_the_tail(self):
        rm = RiskMetrics(self.TIES)
        assert rm.cvar_historical(0.95) == pytest.approx(0.06, abs=1e-15)
        assert rm.cvar_historical(0.90) == pytest.approx(0.03, abs=1e-15)
        assert rm.cvar_historical(0.99) == pytest.approx(0.10, abs=1e-15)

    def test_continuous_sample_is_unchanged(self):
        """No ties: Acerbi-Tasche and the plain mean of the tail agree to within a fraction
        of a percent."""
        r = make_returns(5000)
        x = np.sort(r.to_numpy())
        plain_tail_mean = -x[x <= np.percentile(x, 5)].mean()
        assert RiskMetrics(r).cvar_historical(0.95) == pytest.approx(plain_tail_mean, rel=0.005)

    def test_monte_carlo_es_with_ties(self):
        from riskcanvas.simulation.monte_carlo import MonteCarloSimulator

        p = PortfolioLoader().from_returns_df(pd.DataFrame({"A": self.TIES.to_numpy()}, index=self.TIES.index))
        res = MonteCarloSimulator(n_paths=100_000, seed=1, drift="historical").simulate_bootstrap(p, 1)
        assert res.cvar_95 == pytest.approx(0.06, abs=0.003)   # a plain mean below the VaR: ~0.003


class TestUndefinedRatiosAreNull:
    """Every ratio with a 0 / 0 case is NaN at zero volatility, as /analyze's "undefined ->
    null" states (a cash series has no Sharpe, beside a defined Sortino); +inf stays
    where the ratio is genuinely unbounded."""

    CASH = _dated([0.0001] * 300)
    FLAT = _dated([0.0] * 300)

    def test_sharpe_of_a_cash_series_is_undefined(self):
        assert np.isnan(PerformanceMetrics(self.CASH).compute_all().sharpe)

    def test_flat_series_has_no_defined_ratio(self):
        rf0 = PerformanceMetrics(self.FLAT, risk_free_rate=0.0).compute_all()
        for name in ("sharpe", "sortino", "calmar", "profit_factor"):
            assert np.isnan(getattr(rf0, name)), name
        assert np.isnan(RiskMetrics(self.FLAT).omega_ratio(threshold=0.0))

    def test_unbounded_ratios_stay_infinite(self):
        up = PerformanceMetrics(_dated([0.001, 0.002] * 150), risk_free_rate=0.0).compute_all()
        assert up.sortino == float("inf") and up.profit_factor == float("inf") and up.calmar == float("inf")

    def test_over_the_wire_they_are_null(self):
        from fastapi.testclient import TestClient

        from riskcanvas.api.app import app

        out = TestClient(app).post("/portfolio/analyze", json={"weights": {"C": 1}, "returns": {"C": [0.0001] * 300}}).json()
        assert out["performance"]["sharpe"] is None
        assert out["risk"]["skewness"] is None and out["risk"]["var_stress_multiplier"] is None


class TestMissingValuesAndEdges:
    """Missing values, and helpers on degenerate input."""

    def test_missing_values_are_an_error_not_dropped(self):
        r = make_returns(50)
        r.iloc[7] = np.nan
        for cls in (RiskMetrics, PerformanceMetrics):
            with pytest.raises(ValueError, match="1 missing value"):
                cls(r)

    def test_zero_var_is_never_negative_zero(self):
        """var_1d is never serialised as -0.0 (the dashboard would show "-0.00%")."""
        import math

        from fastapi.testclient import TestClient

        from riskcanvas.api.app import app

        cell = TestClient(app).post("/portfolio/var", params={"portfolio_value": 1},
                                    json={"weights": {"A": 1}, "returns": {"A": [0.0] * 30}}).json()["var_report"]["95%"]
        for key, v in cell.items():
            assert v == 0 and math.copysign(1.0, v) == 1.0, (key, v)

    def test_sqrt_time_scaling_documents_its_zero_mean_assumption(self):
        assert "ZERO mean" in RiskMetrics.var_historical_scaled.__doc__
        r = RiskMetrics(_dated([0.01] * 50))
        assert r.var_historical_scaled(0.95, 10) == pytest.approx(-0.01 * np.sqrt(10))   # what the doc warns about


def test_performance_metrics_needs_dates_and_says_so():
    """A plain Series is a clear error up front, not a bare TypeError from resample("ME")
    inside compute_all."""
    with pytest.raises(ValueError, match="DatetimeIndex"):
        PerformanceMetrics(pd.Series([0.01, -0.02, 0.005]))


class TestPeriodsPerYear:
    """Annualisation follows the series, not a fixed 252: for a 7-day asset (BTC-USD, 730
    calendar-daily observations) 252 would read CAGR 29.6% vs the 45.6%/yr it grew, with
    volatility understated by sqrt(252/365)."""

    CAL = pd.Series([0.001] * 730, index=pd.date_range("2023-01-01", periods=730, freq="D"))

    def test_calendar_daily_series_annualises_with_365(self):
        perf = PerformanceMetrics(self.CAL, risk_free_rate=0.0).compute_all()
        assert perf.cagr == pytest.approx(1.001 ** 365 - 1, rel=1e-9)
        noisy = pd.Series(np.random.default_rng(5).normal(0, 0.03, 730), index=self.CAL.index)
        rm = RiskMetrics(noisy)
        assert rm.periods == 365
        assert rm.compute_all().annual_vol == pytest.approx(noisy.std() * np.sqrt(365), rel=1e-12)
        assert rm.compute_all().periods_per_year == 365

    def test_business_day_and_undated_series_keep_252(self):
        assert RiskMetrics(make_returns(300)).periods == 252
        assert RiskMetrics(pd.Series([0.01, -0.01, 0.0])).periods == 252
        assert PerformanceMetrics(make_returns(300)).compute_all().cagr == pytest.approx(
            (1 + make_returns(300)).prod() ** (252 / 300) - 1, rel=1e-12)


class TestCalendarInference:
    """The annualisation factor comes from the series' spacing and weekdays. A
    weekend-share rule would read a Sunday-Thursday market and a month-end CSV as 365."""

    DAYS = pd.date_range("2022-01-02", periods=900, freq="D")

    @pytest.mark.parametrize("closed", [(4, 5), (5, 6), (3, 4), (0, 1)], ids=["sun-thu", "mon-fri", "sat-wed", "wed-sun"])
    def test_any_five_day_week_is_252(self, closed):
        idx = self.DAYS[~self.DAYS.dayofweek.isin(closed)]
        assert periods_per_year(idx) == 252

    def test_sun_thu_volatility_is_annualised_with_252(self):
        idx = self.DAYS[~self.DAYS.dayofweek.isin((4, 5))]
        s = pd.Series(np.random.default_rng(3).normal(0.0004, 0.01, len(idx)), index=idx)
        assert RiskMetrics(s).compute_all().annual_vol == pytest.approx(s.std() * np.sqrt(252), rel=1e-12)

    @pytest.mark.parametrize("n", [3, 14, 40, 130])
    def test_every_28_days_is_13_a_year_not_12(self, n):
        assert periods_per_year(pd.date_range("2020-01-01", periods=n, freq="28D")) == 13

    @pytest.mark.parametrize("freq", ["ME", "MS", "BME", "BMS"])
    @pytest.mark.parametrize("n", [3, 4, 5, 13])
    def test_every_window_of_calendar_months_stays_12(self, freq, n):
        months = pd.date_range("1994-01-01", "2029-01-01", freq=freq)
        assert {periods_per_year(months[i:i + n]) for i in range(len(months) - n)} == {12}

    @pytest.mark.parametrize("freq", ["ME", "MS", "BME"])
    def test_monthly_series_is_12(self, freq):
        idx = pd.date_range("2015-01-31", periods=120, freq=freq)
        assert periods_per_year(idx) == 12
        s = pd.Series(np.random.default_rng(3).normal(0.008, 0.04, 120), index=idx)
        rm = RiskMetrics(s).compute_all()
        assert rm.annual_vol == pytest.approx(s.std() * np.sqrt(12), rel=1e-12) and rm.periods_per_year == 12
        perf = PerformanceMetrics(s, risk_free_rate=0.0).compute_all()
        assert perf.cagr == pytest.approx((1 + s).prod() ** (12 / 120) - 1, rel=1e-9)

    def test_weekly_series_is_52(self):
        idx = pd.date_range("2015-01-02", periods=200, freq="W-FRI")
        assert periods_per_year(idx) == 52
        # A holiday moves one close a day: still weekly.
        assert periods_per_year(idx[:50].append(pd.DatetimeIndex(["2016-01-07"])).append(idx[55:])) == 52

    def test_daily_weekday_numbers_do_not_move(self):
        """Holidays, special weekend sessions and holes keep a 5-day series at 252."""
        idx = pd.bdate_range("2015-01-01", "2024-12-31")
        rng = np.random.default_rng(9)
        holidays = idx[rng.choice(len(idx), 90, replace=False)]
        assert periods_per_year(idx.difference(holidays)) == 252
        assert periods_per_year(idx.union(pd.date_range("2015-01-03", "2024-12-31", freq="W-SAT")[::26])) == 252
        assert periods_per_year(idx[~((idx >= "2019-04-01") & (idx <= "2019-08-31"))]) == 252
        assert periods_per_year(idx[rng.random(len(idx)) > 0.4]) == 252     # 40% of days dropped
        s = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
        assert RiskMetrics(s).compute_all().annual_vol == pytest.approx(s.std() * np.sqrt(252), rel=1e-12)

    def test_a_market_that_moved_its_weekend_is_still_252(self):
        """Sat-Wed, then Sun-Thu: six weekdays seen, five sessions a week."""
        before = self.DAYS[self.DAYS.dayofweek.isin((5, 6, 0, 1, 2))][:300]
        after = self.DAYS[self.DAYS.dayofweek.isin((6, 0, 1, 2, 3))]
        assert periods_per_year(before.append(after[after > before[-1]])) == 252

    def test_seven_day_series_with_holes_stays_365(self):
        idx = self.DAYS[np.random.default_rng(3).random(len(self.DAYS)) > 0.4]
        assert periods_per_year(idx) == 365

    @pytest.mark.parametrize("idx", [
        pd.date_range("2020-01-01", periods=100, freq="h"),
        pd.date_range("2024-01-02 09:30", periods=9, freq="h"),
        pd.date_range("2024-01-02 09:30", periods=390, freq="min"),
        pd.date_range("1990-12-31", periods=10, freq="2YE"),
    ], ids=["hourly", "nine-hourly-bars", "minutes", "every-two-years"])
    def test_what_cannot_be_measured_is_refused_with_a_way_out(self, idx):
        """Nine hourly bars are refused, not read as a "short daily series" at 252."""
        with pytest.raises(ValueError, match="periods"):
            periods_per_year(idx)
        s = pd.Series(np.random.default_rng(2).normal(0, 0.01, len(idx)), index=idx)
        with pytest.raises(ValueError, match="Cannot infer"):
            RiskMetrics(s)
        with pytest.raises(ValueError, match="Cannot infer"):
            PerformanceMetrics(s)
        assert RiskMetrics(s, periods=4).compute_all().annual_vol == pytest.approx(s.std() * 2, rel=1e-12)
        assert PerformanceMetrics(s, periods=4).compute_all().annual_vol == pytest.approx(s.std() * 2, rel=1e-12)

    SIX_YEARS = pd.bdate_range("2020-01-01", "2025-12-31")
    EVERY_DAY = pd.date_range("2020-01-01", "2025-12-31", freq="D")

    @pytest.mark.parametrize("idx, rate", [
        (SIX_YEARS[SIX_YEARS.dayofweek < 4], 209),                    # Monday-Thursday
        (SIX_YEARS[SIX_YEARS.dayofweek < 3], 157),                    # Monday-Wednesday
        (SIX_YEARS[SIX_YEARS.dayofweek < 2], 104),                    # Monday, Tuesday
        (SIX_YEARS[SIX_YEARS.dayofweek.isin((1, 3))], 104),           # Tuesday, Thursday
        (EVERY_DAY[EVERY_DAY.dayofweek != 6], 313),                   # a 6-day week
        (pd.date_range("2020-01-01", "2025-12-31", freq="2D"), 183),  # every other day
        (pd.date_range("2015-03-31", periods=40, freq="QE"), 4),      # quarterly
    ], ids=["mon-thu", "mon-wed", "mon-tue", "tue-thu", "six-day-week", "every-other-day", "quarterly"])
    def test_anything_else_that_can_be_measured_uses_the_measured_rate(self, idx, rate):
        """Monday-Thursday, 6-day, every-other-day and quarterly series are measured: their
        rates can simply be read off the dates."""
        assert periods_per_year(idx) == rate
        s = pd.Series(np.random.default_rng(3).normal(0.0004, 0.01, len(idx)), index=idx)
        rm = RiskMetrics(s).compute_all()
        assert rm.periods_per_year == rate
        assert rm.annual_vol == pytest.approx(s.std() * np.sqrt(rate), rel=1e-12)

    def test_short_common_session_week_reads_true_vol_and_cagr(self):
        """The latent market's 7-day compounded level read on Monday-Thursday:
        the annual figures must match that market's, not a 252-day week's."""
        rng = np.random.default_rng(11)
        days = pd.date_range(end="2026-11-03", periods=365 * 6, freq="D")
        latent = pd.Series(rng.normal(0.0004, 0.01, len(days)), index=days)
        level = (1 + latent).cumprod()
        sessions = level[level.index.dayofweek < 4].pct_change().dropna()
        assert periods_per_year(sessions.index) == 209
        true_cagr = float((1 + latent).prod() ** (1 / 6) - 1)
        true_vol = float(latent.std() * np.sqrt(365))
        perf = PerformanceMetrics(sessions).compute_all()
        assert perf.cagr == pytest.approx(true_cagr, abs=0.005)
        assert perf.annual_vol == pytest.approx(true_vol, abs=0.005)
        wrong = PerformanceMetrics(sessions, periods=252).compute_all()
        assert wrong.cagr > true_cagr + 0.03 and wrong.annual_vol > true_vol + 0.015

    def test_exchange_calendars_stay_on_the_standard_rate(self):
        """The other side: real weekday calendars, holidays and a long closure
        included, are 252 and not "measured"."""
        idx = pd.bdate_range("2023-01-03", "2026-11-03")
        golden_week = (idx >= "2023-09-29") & (idx <= "2023-10-08")
        spring_festival = (idx >= "2024-02-09") & (idx <= "2024-02-18")
        assert periods_per_year(idx[~(golden_week | spring_festival)]) == 252
        rng = np.random.default_rng(4)
        assert periods_per_year(idx[rng.random(len(idx)) > 0.07]) == 252     # about 242 a year

    @staticmethod
    def _holding(per_year, first="2016-01-04", last="2025-12-31"):
        """Weekdays thinned to exactly ``per_year`` sessions each calendar year."""
        days = pd.bdate_range(first, last)
        rng = np.random.default_rng(per_year)
        keep = np.ones(len(days), dtype=bool)
        for year in days.year.unique():
            where = np.flatnonzero(days.year == year)
            keep[rng.choice(where, max(len(where) - per_year, 0), replace=False)] = False
        return days[keep]

    # New York with Tokyo: Monday-Friday, minus each market's holidays.
    COMMON = _holding.__func__(235)
    COMMON_RETURNS = pd.Series(np.random.default_rng(8).normal(0.0006, 0.0105, len(COMMON)), index=COMMON)

    def test_two_exchanges_common_sessions_read_as_measured(self):
        """235 common sessions a year read as measured, not as 252; the figures are the
        ones the README quotes."""
        rate = periods_per_year(self.COMMON)
        assert rate == pytest.approx(235, abs=1)
        s = self.COMMON_RETURNS
        assert RiskMetrics(s).compute_all().annual_vol == pytest.approx(s.std() * np.sqrt(rate), rel=1e-12)
        years = (self.COMMON[-1] - self.COMMON[0]) / pd.Timedelta(days=365.25) * len(s) / (len(s) - 1)
        perf = PerformanceMetrics(s).compute_all()
        assert perf.cagr == pytest.approx((1 + s).prod() ** (1 / years) - 1, rel=1e-9)
        at_252 = PerformanceMetrics(s, periods=252).compute_all()
        assert (perf.annual_vol, at_252.annual_vol) == (pytest.approx(0.1642, abs=1e-4), pytest.approx(0.1701, abs=1e-4))
        assert (perf.cagr, at_252.cagr) == (pytest.approx(0.0579, abs=1e-4), pytest.approx(0.0622, abs=1e-4))
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        assert "volatility 17.01% for 16.42% and CAGR 6.22% for 5.79%" in " ".join(readme.split())

    @pytest.mark.parametrize("per_year, expected", [
        (261, 252), (252, 252), (243, 252), (240, 252),   # one exchange, however few holidays
        (239, 239), (235, 235), (227, 227),               # two exchanges' common sessions
        (225, 252),                                       # holes: 252, and sparse_calendar over HTTP
    ])
    def test_the_line_between_one_calendar_and_two(self, per_year, expected):
        idx = self._holding(per_year)
        assert observations_per_year(idx) == pytest.approx(per_year, abs=0.5)
        assert periods_per_year(idx) == expected

    def test_full_year_weekday_cagr_is_still_n_over_252(self):
        """bdate_range holds 261 a year and a real exchange 252: 252 of either are a year."""
        for idx in (pd.bdate_range("2021-01-04", periods=756), self._holding(252)):
            s = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, len(idx)), index=idx)
            assert PerformanceMetrics(s).compute_all().cagr == pytest.approx(
                (1 + s).prod() ** (252 / len(s)) - 1, rel=1e-12)

    def test_under_a_year_of_dates_stays_252(self):
        """A short window can sit inside one closure: its rate proves nothing."""
        idx = self.COMMON[self.COMMON >= "2024-01-01"][:150]
        assert periods_per_year(idx) == 252
        s = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, len(idx)), index=idx)
        assert PerformanceMetrics(s).compute_all().cagr == pytest.approx(
            (1 + s).prod() ** (252 / len(s)) - 1, rel=1e-12)

    def test_cagr_counts_the_dates_on_a_sparse_calendar(self):
        """40% of days gone over ten years: 252 is kept for volatility, and CAGR reads ten
        years, not the six that n / 252 would make of them."""
        idx = pd.bdate_range("2015-01-01", "2024-12-31")
        idx = idx[np.random.default_rng(9).random(len(idx)) > 0.4]
        s = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, len(idx)), index=idx)
        perf = PerformanceMetrics(s).compute_all()
        assert perf.cagr == pytest.approx((1 + s).prod() ** (1 / 10) - 1, abs=1e-3)
        assert perf.annual_vol == pytest.approx(s.std() * np.sqrt(252), rel=1e-12)

    def test_stated_periods_override_the_dates_for_cagr(self):
        s = self.COMMON_RETURNS
        assert PerformanceMetrics(s, periods=100).compute_all().cagr == pytest.approx(
            (1 + s).prod() ** (100 / len(s)) - 1, rel=1e-12)

    @pytest.mark.parametrize("idx", [
        pd.date_range("2020-01-01", periods=3, freq="YS"),
        pd.date_range("2019-12-31", periods=6, freq="YE"),
    ], ids=["year-starts", "year-ends"])
    def test_annual_dates_across_a_leap_year_are_one_a_year(self, idx):
        """366 days between two of them measure 0.9993 a year, which must not be refused as
        fewer than one."""
        assert periods_per_year(idx) == 1
        s = pd.Series(np.random.default_rng(3).normal(0.05, 0.1, len(idx)), index=idx)
        assert RiskMetrics(s).compute_all().annual_vol == pytest.approx(s.std(), rel=1e-12)

    def test_dates_further_apart_than_a_year_stay_refused(self):
        with pytest.raises(ValueError, match="fewer than one observation a year"):
            periods_per_year(pd.date_range("2000-01-01", periods=5, freq="400D"))

    def test_too_few_dates_have_no_pattern_to_read(self):
        assert periods_per_year(pd.DatetimeIndex(["2023-01-06", "2023-01-09"])) == 252
        assert periods_per_year(pd.DatetimeIndex(["2023-01-06", "2023-01-09", "2023-01-10"])) == 252
        assert periods_per_year(pd.date_range("2023-01-01", periods=5, freq="D")) == 252

    def test_periods_must_be_positive(self):
        for cls in (RiskMetrics, PerformanceMetrics):
            with pytest.raises(ValueError, match="positive"):
                cls(make_returns(30), periods=0)


class TestPositiveMonths:
    """A month with no observation is not a non-positive month: missing="drop" says dropped
    days are absent from every statistic."""

    IDX = pd.bdate_range("2023-01-02", "2023-12-29")

    def _portfolio(self, frame, **kw):
        return PortfolioLoader().from_returns_df(frame, **kw).portfolio_returns

    def test_dropped_months_are_absent_not_flat(self):
        full = pd.DataFrame({"A": 0.002}, index=self.IDX)
        holed = full.astype(float).copy()
        holed.loc["2023-04-01":"2023-08-31", "A"] = np.nan
        assert PerformanceMetrics(self._portfolio(full)).compute_all().positive_months == 1.0
        assert PerformanceMetrics(self._portfolio(holed, missing="drop")).compute_all().positive_months == 1.0

    def test_denominator_is_the_months_with_data(self):
        """Seven months remain; three of them lose."""
        s = pd.Series(0.002, index=self.IDX)
        s.loc["2023-01-01":"2023-03-31"] = -0.002
        s = s[(s.index < "2023-04-01") | (s.index > "2023-08-31")]
        assert PerformanceMetrics(s).compute_all().positive_months == pytest.approx(4 / 7)

    def test_a_real_flat_month_still_counts_as_not_positive(self):
        s = pd.Series(0.002, index=self.IDX)
        s.loc["2023-06-01":"2023-06-30"] = 0.0
        assert PerformanceMetrics(s).compute_all().positive_months == pytest.approx(11 / 12)


def test_a_total_too_large_to_annualise_is_unbounded_not_an_exception():
    """+1000% a day for 100 days is a finite total, 1.4e104, and on a 7-day
    series its annualised power is not: OverflowError (a raw 500 over HTTP)."""
    s = pd.Series([10.0] * 100, index=pd.date_range("2023-01-01", periods=100, freq="D"))
    report = PerformanceMetrics(s).compute_all()
    assert report.cagr == float("inf") and report.to_dict()["cagr"] is None


class TestDenserBenchmarkCalendar:
    """A benchmark or factor on a denser calendar than the portfolio: Monday's
    portfolio return covers three days, the benchmark's Monday return one."""

    DAYS = pd.date_range("2022-01-03", periods=700, freq="D")
    LATENT = pd.Series(np.random.default_rng(7).normal(0.0005, 0.012, 700), index=DAYS)

    def _weekday_portfolio(self):
        level = (1 + self.LATENT).cumprod()
        return level[level.index.dayofweek < 5].pct_change().iloc[1:]

    def test_information_ratio_measures_both_legs_over_the_same_intervals(self):
        """The portfolio IS the benchmark on a sparser calendar: no tracking error."""
        assert np.isnan(RiskMetrics(self._weekday_portfolio()).information_ratio(self.LATENT))

    def test_factor_exposure_measures_both_sides_over_the_same_intervals(self):
        factors = pd.DataFrame({"MKT": self.LATENT})
        b = PerformanceMetrics(self._weekday_portfolio()).factor_exposure(factors)
        assert b["MKT"] == pytest.approx(1.0, abs=1e-9) and b["alpha"] == pytest.approx(0.0, abs=1e-9)

    def test_same_calendar_series_pass_through_untouched(self):
        from riskcanvas.metrics.risk import compound_onto

        r = self._weekday_portfolio()
        assert compound_onto(r, r.index) is not None and compound_onto(r, r.index).equals(r)

    def test_a_missing_value_inside_an_interval_is_not_a_flat_day(self):
        bench = self.LATENT.copy()
        saturday = bench.index[bench.index.dayofweek == 5][10]
        bench.loc[saturday] = np.nan
        with pytest.raises(ValueError, match="no value"):
            RiskMetrics(self._weekday_portfolio()).information_ratio(bench)
        with pytest.raises(ValueError, match="no value"):
            PerformanceMetrics(self._weekday_portfolio()).factor_exposure(pd.DataFrame({"MKT": bench}))


class TestSmallerTruths:
    IDX = pd.bdate_range("2023-01-02", periods=8)

    def test_a_flat_day_is_not_a_loss(self):
        """Seven flat days and one -3%: avg_loss is 3%, not 0.375%."""
        r = pd.Series([0.0] * 7 + [-0.03], index=self.IDX)
        report = PerformanceMetrics(r).compute_all()
        assert report.avg_loss == pytest.approx(0.03, rel=1e-12)
        assert report.win_rate == 0.0

    def test_profit_factor_and_win_rate_do_not_move(self):
        r = pd.Series([0.02, 0.0, -0.01, 0.0, 0.01, 0.0, -0.01, 0.0], index=self.IDX)
        report = PerformanceMetrics(r).compute_all()
        assert report.profit_factor == pytest.approx(0.03 / 0.02, rel=1e-12)
        assert report.win_rate == pytest.approx(2 / 8)
        assert report.avg_win == pytest.approx(0.015, rel=1e-12)
        assert report.avg_loss == pytest.approx(0.01, rel=1e-12)

    @pytest.mark.parametrize("method", ["var_historical", "var_parametric", "cvar_historical", "cvar_parametric"])
    @pytest.mark.parametrize("confidence", [95, 0.0, 1.0, -0.5])
    def test_every_var_refuses_a_confidence_outside_0_1(self, method, confidence):
        """var_parametric(95) raises as var_historical(95) does, rather than return NaN."""
        with pytest.raises(ValueError, match="confidence"):
            getattr(RiskMetrics(make_returns(300)), method)(confidence)

    def test_the_risk_free_rate_is_stated_over_http(self):
        from fastapi.testclient import TestClient

        from riskcanvas.api.app import app
        from riskcanvas.metrics.risk import RISK_FREE

        out = TestClient(app).post("/portfolio/analyze", json={
            "weights": {"A": 1}, "returns": {"A": np.random.default_rng(0).normal(0.0004, 0.01, 300).tolist()},
            "start": "2020-01-01"}).json()
        assert out["risk_free_rate"] == RISK_FREE == 0.02
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        assert "**The risk-free rate is 2% a year.**" in readme and "`risk_free_rate`" in readme

    def test_docs_do_not_promise_what_does_not_exist(self):
        from riskcanvas.scenarios.stress_test import ScenarioResult
        from riskcanvas.simulation.monte_carlo import SimulationResult

        assert not hasattr(SimulationResult, "to_dict") and not hasattr(ScenarioResult, "to_dict")
        root = Path(__file__).resolve().parents[1]
        assert "Every computation returns a typed dataclass with a" not in (root / "docs" / "architecture.md").read_text()
        assert "endpoints for all analytics" not in (root / "README.md").read_text()


class TestTailRatioAndRollingSharpeValues:
    """Values, not only the sign, the length and the warm-up: inverting the ratio or
    dropping the annualisation must fail."""

    TAILS = pd.Series(np.linspace(-0.01, 0.03, 101), index=pd.date_range("2020-01-01", periods=101, freq="B"))

    def test_tail_ratio_is_the_gain_tail_over_the_loss_tail(self):
        """Evenly spaced, so the 95th percentile is 0.028 and the 5th -0.008."""
        assert RiskMetrics(self.TAILS).tail_ratio() == pytest.approx(0.028 / 0.008, rel=1e-12)

    def test_a_left_skewed_series_reads_below_one(self):
        assert RiskMetrics(-self.TAILS).tail_ratio() == pytest.approx(0.008 / 0.028, rel=1e-12)

    def test_rolling_sharpe_is_annualised(self):
        s = make_returns(100)
        window = s.iloc[-20:] - 0.02 / 252
        expected = window.mean() / window.std(ddof=1) * np.sqrt(252)
        assert PerformanceMetrics(s).rolling_sharpe(window=20).iloc[-1] == pytest.approx(expected, rel=1e-9)

    def test_rolling_sharpe_uses_the_series_own_rate(self):
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        s = pd.Series(np.random.default_rng(3).normal(0.008, 0.04, 60), index=idx)
        window = s.iloc[-12:] - 0.02 / 12
        expected = window.mean() / window.std(ddof=1) * np.sqrt(12)
        assert PerformanceMetrics(s).rolling_sharpe(window=12).iloc[-1] == pytest.approx(expected, rel=1e-9)
