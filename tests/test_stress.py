"""Tests for stress testing scenarios."""

import pytest
from riskcanvas.scenarios.stress_test import StressTester, SCENARIOS, ScenarioResult
from riskcanvas.portfolio.loader import PortfolioLoader
import numpy as np
import pandas as pd


def make_mock_portfolio(vol_scale: float = 1.0):
    rng = np.random.default_rng(1)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = pd.DataFrame({
        "A": rng.normal(0.0005, 0.012 * vol_scale, n),
        "B": rng.normal(0.0003, 0.010 * vol_scale, n),
    }, index=dates)
    return PortfolioLoader().from_returns_df(returns, {"A": 0.6, "B": 0.4})


@pytest.fixture(scope="module")
def portfolio():
    return make_mock_portfolio()


class TestStressTester:
    def test_scenarios_dict_non_empty(self):
        assert len(SCENARIOS) >= 5

    def test_each_scenario_has_required_keys(self):
        """`duration_days` means trading days (the path length) in every ScenarioResult and
        API response, so a scenario names its two spans `trading_days` and
        `calendar_days`."""
        for key, scenario in SCENARIOS.items():
            assert "name" in scenario, f"{key} missing 'name'"
            assert "s&p500_drawdown" in scenario, f"{key} missing 's&p500_drawdown'"
            assert "trading_days" in scenario and "calendar_days" in scenario, key
            assert "duration_days" not in scenario, f"{key}: ambiguous 'duration_days' is back"

    def test_run_scenario_returns_result(self, portfolio):
        tester = StressTester(beta_to_spy=1.0)
        result = tester.run_scenario(portfolio, "covid_crash_2020")
        assert isinstance(result, ScenarioResult)

    def test_portfolio_drawdown_nonpositive(self, portfolio):
        tester = StressTester(beta_to_spy=1.0)
        result = tester.run_scenario(portfolio, "gfc_2008")
        assert result.portfolio_drawdown <= 0

    def test_high_beta_worse_drawdown(self, portfolio):
        """Portfolio with beta=2 should have worse drawdown than beta=1."""
        t1 = StressTester(beta_to_spy=1.0)
        t2 = StressTester(beta_to_spy=2.0)
        r1 = t1.run_scenario(portfolio, "covid_crash_2020")
        r2 = t2.run_scenario(portfolio, "covid_crash_2020")
        assert r2.portfolio_drawdown <= r1.portfolio_drawdown

    def test_run_all_returns_list(self, portfolio):
        tester = StressTester()
        results = tester.run_all(portfolio)
        assert len(results) >= 5
        assert all(isinstance(r, ScenarioResult) for r in results)

    def test_run_all_sorted_worst_first(self, portfolio):
        tester = StressTester()
        results = tester.run_all(portfolio)
        for i in range(len(results) - 1):
            assert results[i].portfolio_drawdown <= results[i + 1].portfolio_drawdown

    def test_custom_shock(self, portfolio):
        tester = StressTester()
        result = tester.custom_shock(portfolio, equity_shock=-0.30, duration_days=30)
        assert result.portfolio_drawdown <= 0
        assert "Custom" in result.scenario_name

    def test_unknown_scenario_raises(self, portfolio):
        tester = StressTester()
        with pytest.raises(ValueError):
            tester.run_scenario(portfolio, "nonexistent_crisis")

    def test_scenario_duration_realistic(self, portfolio):
        tester = StressTester()
        result = tester.run_scenario(portfolio, "gfc_2008")
        assert result.duration_days >= 30


# S&P 500 (^GSPC) peak and trough closes for each scenario, re-verified on
# 2026-10-30 against Yahoo Finance daily closes via yfinance:
# (peak date, peak close, trough date, trough close, drawdown to 0.1pt,
#  trading days after the peak through the trough, calendar days).
VERIFIED_GSPC = {
    "black_monday_1987": ("1987-08-25", 336.77, "1987-12-04", 223.92, -0.335, 71, 101),
    "dotcom_crash": ("2000-03-24", 1527.46, "2002-10-09", 776.76, -0.491, 637, 929),
    "gfc_2008": ("2007-10-09", 1565.15, "2009-03-09", 676.53, -0.568, 355, 517),
    "european_debt_crisis": ("2011-04-29", 1363.61, "2011-10-03", 1099.23, -0.194, 108, 157),
    "covid_crash_2020": ("2020-02-19", 3386.15, "2020-03-23", 2237.40, -0.339, 23, 33),
    "rate_hike_2022": ("2022-01-03", 4796.56, "2022-10-12", 3577.03, -0.254, 195, 282),
}


class TestScenarioProvenance:
    """Every documented number is tied to verified S&P 500 closes and the
    scenario's own fields agree with each other (a number and its source are
    one claim)."""

    def test_scenarios_match_verified_closes(self):
        assert set(VERIFIED_GSPC) == set(SCENARIOS)
        for key, (pk_d, pk, tr_d, tr, dd, td, cd) in VERIFIED_GSPC.items():
            s = SCENARIOS[key]
            assert (s["start_date"], s["peak_close"]) == (pk_d, pk), key
            assert (s["end_date"], s["trough_close"]) == (tr_d, tr), key
            assert s["s&p500_drawdown"] == dd, key
            assert s["trading_days"] == td, key
            assert s["calendar_days"] == cd, key

    def test_documented_drawdown_is_trough_over_peak(self):
        for key, s in SCENARIOS.items():
            assert round(s["trough_close"] / s["peak_close"] - 1.0, 3) == s["s&p500_drawdown"], key

    def test_european_debt_crisis_is_19_4_not_21_9(self):
        """1363.61 -> 1099.23 is -19.39%, not -21.9%."""
        assert SCENARIOS["european_debt_crisis"]["s&p500_drawdown"] == -0.194

    def test_path_length_is_peak_to_trough_trading_days(self):
        for key, s in SCENARIOS.items():
            assert len(s["shock_returns"]) == s["trading_days"], key

    def test_calendar_days_is_calendar_span(self):
        from datetime import date

        for key, s in SCENARIOS.items():
            span = date.fromisoformat(s["end_date"]) - date.fromisoformat(s["start_date"])
            assert span.days == s["calendar_days"], key

    def test_description_day_counts_agree_with_fields(self):
        """COVID's window is 23 trading days (33 calendar)."""
        import re

        for key, s in SCENARIOS.items():
            for n in re.findall(r"(\d+) trading days", s["description"]):
                assert int(n) == s["trading_days"], f"{key}: {s['description']}"
            for n in re.findall(r"(\d+) calendar days", s["description"]):
                assert int(n) == s["calendar_days"], f"{key}: {s['description']}"

    def test_black_monday_crash_day_tracks_the_real_one_day_drop(self):
        """Oct 19, 1987 closed -20.47% on the S&P 500 (verified); the stylized
        crash day must stay within half a point of it."""
        worst = min(SCENARIOS["black_monday_1987"]["shock_returns"])
        assert abs(worst - (-0.2047)) < 0.005


class TestScenarioCalibration:
    """Every scenario's path is deterministic, portfolio-independent, and compounds to its
    documented drawdown."""

    def test_every_scenario_has_deterministic_path(self):
        for key, s in SCENARIOS.items():
            assert "shock_returns" in s, f"{key} has no shock_returns path"
            assert len(s["shock_returns"]) >= 20

    # Per-scenario max-drawdown tolerance (fraction). Every scenario,
    # Black Monday included, is bisection-scaled by _calibrated() to hit its
    # documented S&P 500 peak-to-trough drawdown to machine precision, so all
    # get the same tight 0.1pt bound, which is what lets the README say a beta-1
    # portfolio gets exactly the table.
    MAGNITUDE_TOL = {
        "black_monday_1987": 0.001,
        "dotcom_crash": 0.001,
        "gfc_2008": 0.001,
        "covid_crash_2020": 0.001,
        "rate_hike_2022": 0.001,
        "european_debt_crisis": 0.001,
    }

    def test_magnitude_matches_documented_drawdown(self, portfolio):
        """Beta-1 portfolio drawdown == the scenario's own advertised S&P 500
        drawdown, within the per-scenario tolerance above (earlier paths did
        NOT match their documented magnitude)."""
        tester = StressTester(beta_to_spy=1.0)
        # Guard against a scenario being added without a stated tolerance.
        assert set(self.MAGNITUDE_TOL) == set(SCENARIOS), (
            "MAGNITUDE_TOL and SCENARIOS have drifted; add the new scenario's "
            "documented-drawdown tolerance."
        )
        for key, s in SCENARIOS.items():
            r = tester.run_scenario(portfolio, key)
            tol = self.MAGNITUDE_TOL[key]
            err = abs(r.portfolio_drawdown - s["s&p500_drawdown"])
            assert err <= tol, (
                f"{key}: portfolio dd {r.portfolio_drawdown:.4f} vs documented "
                f"{s['s&p500_drawdown']:.4f} (err {err:.4f} > tol {tol:.4f})"
            )

    def test_calibrated_scenarios_hit_documented_to_machine_precision(self):
        """Every scenario is built via _calibrated() and must reproduce its
        documented drawdown essentially exactly — the bisection target."""
        tester = StressTester(beta_to_spy=1.0)
        p = make_mock_portfolio()
        for key, s in SCENARIOS.items():
            r = tester.run_scenario(p, key)
            assert abs(r.portfolio_drawdown - s["s&p500_drawdown"]) <= 1e-4, (
                f"{key}: calibrated dd {r.portfolio_drawdown:.6f} drifted from "
                f"documented {s['s&p500_drawdown']:.6f}"
            )

    def test_path_independent_of_portfolio_vol(self):
        """Scenario paths must not be derived from the analyzed portfolio, such as by
        scaling 'GFC' by the user's own vol."""
        p_low = make_mock_portfolio(vol_scale=1.0)
        p_high = make_mock_portfolio(vol_scale=4.0)
        tester = StressTester(beta_to_spy=1.0)
        for key in SCENARIOS:
            r_low = tester.run_scenario(p_low, key)
            r_high = tester.run_scenario(p_high, key)
            assert r_low.daily_losses == r_high.daily_losses, f"{key} path varies with portfolio"
            assert r_low.portfolio_drawdown == r_high.portfolio_drawdown

    def test_runs_are_deterministic(self, portfolio):
        tester = StressTester(beta_to_spy=1.0)
        for key in SCENARIOS:
            a = tester.run_scenario(portfolio, key)
            b = tester.run_scenario(portfolio, key)
            assert a.daily_losses == b.daily_losses
            assert a.portfolio_drawdown == b.portfolio_drawdown

    def test_custom_shock_compounds_exactly(self, portfolio):
        tester = StressTester()
        r = tester.custom_shock(portfolio, equity_shock=-0.25, duration_days=20)
        assert abs(r.portfolio_drawdown - (-0.25)) < 1e-9
        assert abs(r.total_loss - (-0.25)) < 1e-9

    def test_custom_shock_rejects_total_loss(self, portfolio):
        tester = StressTester()
        with pytest.raises(ValueError):
            tester.custom_shock(portfolio, equity_shock=-1.5)


def make_drift_portfolio(mean_daily: float, seed: int = 11, n: int = 400):
    """Portfolio whose mean daily return has a KNOWN sign.

    The noise is de-meaned before the target drift is added, so the realized
    sample mean matches ``mean_daily`` and the sign is not left to sampling
    luck (the very noise that makes the negative-drift branch so common).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    cols = {}
    for name, vol in (("A", 0.012), ("B", 0.010)):
        noise = rng.normal(0.0, vol, n)
        cols[name] = noise - noise.mean() + mean_daily
    returns = pd.DataFrame(cols, index=dates)
    return PortfolioLoader().from_returns_df(returns, {"A": 0.6, "B": 0.4})


class TestRecoveryEstimate:
    """`recovery_estimate_days` must be an honest int or None — never a `-1` sentinel,
    which reads over the wire as a negative day count. The non-positive-drift branch is
    hit by roughly half of realistic portfolios from ordinary sampling noise, so it is a
    mainline path, not an edge case."""

    def test_positive_drift_gives_positive_int(self):
        p = make_drift_portfolio(0.0005)
        assert p.portfolio_returns.mean() > 0
        r = StressTester(beta_to_spy=1.0).run_scenario(p, "gfc_2008")
        assert isinstance(r.recovery_estimate_days, int)
        assert r.recovery_estimate_days > 0

    def test_negative_drift_gives_none_not_sentinel(self):
        p = make_drift_portfolio(-0.0004)
        assert p.portfolio_returns.mean() < 0
        r = StressTester(beta_to_spy=1.0).run_scenario(p, "gfc_2008")
        assert r.recovery_estimate_days is None, (
            "non-positive drift has no finite recovery time; it must be None, "
            f"got {r.recovery_estimate_days!r}"
        )

    def test_zero_drift_gives_none(self):
        p = make_drift_portfolio(0.0)
        r = StressTester(beta_to_spy=1.0).run_scenario(p, "covid_crash_2020")
        assert r.recovery_estimate_days is None

    @pytest.mark.parametrize("mean_daily", [-0.001, -0.0004, 0.0, 0.0003, 0.0009])
    def test_never_negative_across_every_scenario(self, mean_daily):
        """A bare -1 would appear for every scenario on half of all inputs."""
        p = make_drift_portfolio(mean_daily)
        for r in StressTester(beta_to_spy=1.0).run_all(p):
            v = r.recovery_estimate_days
            assert v is None or v > 0, f"{r.scenario_name}: got {v!r}"
            assert v != -1, f"{r.scenario_name}: -1 sentinel is back"

    def test_custom_shock_recovery_follows_the_scenario_rule(self):
        """custom_shock applies the same rule as the built-in scenarios: the portfolio's
        own drift, None when that is not estimable."""
        up = make_drift_portfolio(0.0005)
        r = StressTester(beta_to_spy=1.0).custom_shock(up, equity_shock=-0.3, duration_days=30)
        # Compounding, not the linear int(0.3 / mean), which understates it.
        growth = np.log1p(up.portfolio_returns.to_numpy()).mean()
        assert r.recovery_estimate_days == int(-np.log(0.7) / growth)
        down = make_drift_portfolio(-0.0004)
        r = StressTester(beta_to_spy=1.0).custom_shock(down, equity_shock=-0.3, duration_days=30)
        assert r.recovery_estimate_days is None

    def test_estimate_scales_inversely_with_drift(self):
        """Sanity on the surviving positive branch: more drift recovers sooner."""
        slow = StressTester(beta_to_spy=1.0).run_scenario(
            make_drift_portfolio(0.0002), "gfc_2008"
        )
        fast = StressTester(beta_to_spy=1.0).run_scenario(
            make_drift_portfolio(0.0008), "gfc_2008"
        )
        assert slow.recovery_estimate_days > fast.recovery_estimate_days


class TestEstimateBetaFallback:
    """estimate_beta must NOT swallow every exception into a silent beta=1.0. Expected
    failure modes fall back with a warning and an honest ``estimated=False`` marker;
    genuine estimates report ``estimated=True``; unexpected errors are allowed to
    propagate."""

    def _install_fake_yfinance(self, monkeypatch, download):
        """Register a fake ``yfinance`` module whose ``download`` is ``download``
        so ``import yfinance`` inside estimate_beta picks it up."""
        import sys
        import types

        fake = types.ModuleType("yfinance")
        fake.download = download
        monkeypatch.setitem(sys.modules, "yfinance", fake)

    def test_download_failure_falls_back_not_estimated(self, portfolio, monkeypatch, caplog):
        def boom(*_a, **_k):
            raise RuntimeError("simulated network failure")

        self._install_fake_yfinance(monkeypatch, boom)
        tester = StressTester()
        with caplog.at_level("WARNING"):
            beta, estimated = tester.estimate_beta_detailed(portfolio)
        assert beta == 1.0
        assert estimated is False
        assert any("SPY download failed" in r.message for r in caplog.records)

    def test_empty_download_falls_back(self, portfolio, monkeypatch):
        def empty(*_a, **_k):
            return pd.DataFrame()

        self._install_fake_yfinance(monkeypatch, empty)
        tester = StressTester()
        beta, estimated = tester.estimate_beta_detailed(portfolio)
        assert beta == 1.0
        assert estimated is False

    def test_insufficient_overlap_falls_back(self, portfolio, monkeypatch):
        # SPY history that shares fewer than 20 days with the portfolio.
        def short(*_a, **_k):
            idx = portfolio.prices.index[:5]
            return pd.DataFrame({"Close": np.linspace(400, 410, len(idx))}, index=idx)

        self._install_fake_yfinance(monkeypatch, short)
        tester = StressTester()
        beta, estimated = tester.estimate_beta_detailed(portfolio)
        assert beta == 1.0
        assert estimated is False

    def test_missing_yfinance_falls_back(self, portfolio, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_yf(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("yfinance not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_yf)
        tester = StressTester()
        beta, estimated = tester.estimate_beta_detailed(portfolio)
        assert beta == 1.0
        assert estimated is False

    def test_real_estimate_reports_estimated_true(self, portfolio, monkeypatch):
        # SPY series correlated with the portfolio over the full overlap so a
        # genuine beta is computed and flagged estimated=True.
        def spy_like(*_a, **_k):
            idx = portfolio.prices.index
            rng = np.random.default_rng(7)
            # Build a price series from returns loosely tracking the portfolio.
            port = portfolio.portfolio_returns.reindex(idx).fillna(0.0).values
            spy_ret = 0.8 * port + rng.normal(0, 0.004, len(idx))
            prices = 400 * np.cumprod(1 + spy_ret)
            return pd.DataFrame({"Close": prices}, index=idx)

        self._install_fake_yfinance(monkeypatch, spy_like)
        tester = StressTester()
        beta, estimated = tester.estimate_beta_detailed(portfolio)
        assert estimated is True
        assert 0.0 <= beta <= 3.0
        # The float wrapper must stay backward-compatible.
        assert tester.estimate_beta(portfolio) == beta

    def test_unexpected_error_propagates(self, portfolio, monkeypatch):
        """A non-expected error type in the compute path must NOT be masked as beta=1.0."""
        def weird(*_a, **_k):
            idx = portfolio.prices.index
            df = pd.DataFrame({"Close": np.linspace(400, 410, len(idx))}, index=idx)
            return df

        self._install_fake_yfinance(monkeypatch, weird)
        tester = StressTester()

        # Force an unexpected error (KeyboardInterrupt is not in the caught set)
        # from inside the compute block via a patched np.cov.
        def interrupt(*_a, **_k):
            raise KeyboardInterrupt("unexpected")

        monkeypatch.setattr(np, "cov", interrupt)
        with pytest.raises(KeyboardInterrupt):
            tester.estimate_beta_detailed(portfolio)


def _orthogonal(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``x`` with its sample projection on ``m`` removed (and de-meaned), so its
    sample covariance with ``m`` is exactly 0."""
    x = x - x.mean()
    mc = m - m.mean()
    return x - (x @ mc) / (mc @ mc) * mc


class TestBeta:
    """The portfolio's market sensitivity must drive the stress result: an explicit beta of
    0 means 0, and the API estimates beta when it can."""

    def test_explicit_zero_beta_means_zero(self, portfolio):
        tester = StressTester(beta_to_spy=0.0)
        assert tester.beta == 0.0
        for r in tester.run_all(portfolio):
            assert r.portfolio_drawdown == 0.0, r.scenario_name
            assert r.recovery_estimate_days is None

    def test_none_is_the_market_like_assumption(self):
        assert StressTester().beta == 1.0
        assert StressTester(beta_to_spy=None).beta == 1.0

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 4.5, -4.01])
    def test_invalid_beta_rejected(self, bad):
        with pytest.raises(ValueError):
            StressTester(beta_to_spy=bad)

    def test_low_beta_scales_every_scenario_down(self, portfolio):
        low = {r.scenario_name: r for r in StressTester(beta_to_spy=0.3).run_all(portfolio)}
        for r in StressTester(beta_to_spy=1.0).run_all(portfolio):
            assert r.portfolio_drawdown < low[r.scenario_name].portfolio_drawdown < 0, r.scenario_name

    def test_max_abs_beta_keeps_every_day_above_total_loss(self):
        from riskcanvas.scenarios.stress_test import MAX_ABS_BETA

        for key, s in SCENARIOS.items():
            for b in (MAX_ABS_BETA, -MAX_ABS_BETA):
                assert min(1.0 + b * r for r in s["shock_returns"]) > 0.0, (key, b)

    def test_run_all_surfaces_a_failing_scenario(self, portfolio, monkeypatch):
        """A broken scenario must surface from run_all, not leave a shorter list with only
        a log line."""
        tester = StressTester(beta_to_spy=1.0)
        real = tester.run_scenario

        def flaky(p, key):
            if key == "gfc_2008":
                raise RuntimeError("boom")
            return real(p, key)

        monkeypatch.setattr(tester, "run_scenario", flaky)
        with pytest.raises(RuntimeError):
            tester.run_all(portfolio)


class TestBetaFromReturns:
    def test_series_on_itself_is_exactly_one(self):
        """cov and var must share the ddof=1 normaliser; mixing ddof=1 cov with
        ddof=0 var made a market portfolio's beta n/(n-1), not 1."""
        from riskcanvas.scenarios.stress_test import beta_from_returns

        m = np.random.default_rng(3).normal(0.0004, 0.011, 300)
        assert beta_from_returns(m, m) == pytest.approx(1.0, abs=1e-12)

    def test_recovers_a_known_loading(self):
        from riskcanvas.scenarios.stress_test import beta_from_returns

        rng = np.random.default_rng(4)
        m = rng.normal(0.0004, 0.011, 400)
        p = 0.35 * m + _orthogonal(rng.normal(0, 0.004, 400), m)
        assert beta_from_returns(p, m) == pytest.approx(0.35, abs=1e-12)

    def test_negative_beta_is_not_clamped(self):
        from riskcanvas.scenarios.stress_test import beta_from_returns

        m = np.random.default_rng(5).normal(0.0, 0.01, 100)
        assert beta_from_returns(-0.5 * m, m) == pytest.approx(-0.5, abs=1e-12)

    @pytest.mark.parametrize(
        "p, m",
        [
            (np.zeros(10), np.ones(10)),            # too few observations
            (np.ones(30), np.full(30, 0.01)),       # zero-variance market
            (np.ones(30), np.ones(31)),             # length mismatch
            (np.r_[np.nan, np.ones(29)], np.linspace(0, 1, 30)),  # non-finite
        ],
    )
    def test_rejects_unusable_input(self, p, m):
        from riskcanvas.scenarios.stress_test import beta_from_returns

        with pytest.raises(ValueError):
            beta_from_returns(p, m)


class TestEstimateBetaUnclamped:
    def test_estimate_is_returned_as_computed(self, portfolio, monkeypatch):
        """The SPY estimate is not clamped: a negative-beta portfolio reads as negative,
        not as an estimated 0.0."""
        import sys
        import types

        def spy_inverse(*_a, **_k):
            idx = portfolio.prices.index
            port = portfolio.portfolio_returns.reindex(idx).fillna(0.0).values
            prices = 400 * np.cumprod(1 - 2.0 * port)
            return pd.DataFrame({"Close": prices}, index=idx)

        fake = types.ModuleType("yfinance")
        fake.download = spy_inverse
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        beta, estimated = StressTester().estimate_beta_detailed(portfolio)
        assert estimated is True
        assert beta == pytest.approx(-0.5, abs=1e-9)


# RECORDED values, not recomputed here (the suite runs offline, so it cannot
# download ^GSPC): daily close-to-close returns of ^GSPC over each scenario's
# peak-to-trough window (peak close excluded, trough close included), measured
# once on 2026-11-01 via yfinance: (worst day, its date, sample std of daily
# returns). The
# README's path-vs-real table quotes these; COVID's worst day is
# 2711.02 (2020-03-13) -> 2386.13 (2020-03-16).
REAL_GSPC_DAILY = {
    "black_monday_1987": (-0.2047, "1987-10-19", 0.0338),
    "dotcom_crash": (-0.0583, "2000-04-14", 0.0145),
    "gfc_2008": (-0.0903, "2008-10-15", 0.0239),
    "european_debt_crisis": (-0.0666, "2011-08-08", 0.0172),
    "covid_crash_2020": (-0.1198, "2020-03-16", 0.0503),
    "rate_hike_2022": (-0.0432, "2022-09-13", 0.0152),
}

# RECORDED values, not recomputed here (offline suite): max drawdown, from
# pre-crisis capital, of prod(1 + beta * r) over the same ^GSPC windows (a
# daily-rebalanced beta position on the real path), measured once on
# 2026-11-01 via yfinance (closes and day counts asserted against SCENARIOS
# in the same run). Columns: beta 0.5, beta 2, beta 0.618 (the README's 60/40
# example).
REAL_GSPC_BETA_DRAWDOWN = {
    "black_monday_1987": (-0.176642, -0.603528, -0.215217),
    "dotcom_crash": (-0.275021, -0.773775, -0.331119),
    "gfc_2008": (-0.325426, -0.848290, -0.389858),
    "european_debt_crisis": (-0.098505, -0.371317, -0.121338),
    "covid_crash_2020": (-0.180533, -0.592265, -0.219967),
    "rate_hike_2022": (-0.131511, -0.468783, -0.161324),
}

README_NAMES = {
    "Black Monday": "black_monday_1987",
    "Dot-com Crash": "dotcom_crash",
    "Global Financial Crisis": "gfc_2008",
    "European Debt Crisis": "european_debt_crisis",
    "COVID-19 Crash": "covid_crash_2020",
    "Fed Rate Hike Bear": "rate_hike_2022",
}


def _equity(path):
    return np.cumprod(np.r_[1.0, 1.0 + np.asarray(path)])


class TestScenarioShape:
    """Crash days and troughs match the crisis: COVID's worst day is about the real -12.0%,
    and every path ends at its trough, like the real peak-close-to-trough-close window."""

    def test_every_path_ends_at_its_trough(self):
        for key, s in SCENARIOS.items():
            eq = _equity(s["shock_returns"])
            assert int(np.argmin(eq)) == len(eq) - 1, f"{key}: trough on day {int(np.argmin(eq))} of {len(eq) - 1}"

    def test_beta_one_total_loss_equals_drawdown_equals_table(self, portfolio):
        for r in StressTester(beta_to_spy=1.0).run_all(portfolio):
            assert r.total_loss == pytest.approx(r.portfolio_drawdown, abs=1e-12), r.scenario_name
            assert r.total_loss == pytest.approx(r.scenario_drawdown, abs=1e-6), r.scenario_name

    def test_covid_crash_day_tracks_the_real_one_day_drop(self):
        """2020-03-16 closed 2386.13 from 2711.02 (-11.98%, verified
        2026-11-01); the stylized crash day stays within half a point."""
        assert round(2386.13 / 2711.02 - 1.0, 4) == REAL_GSPC_DAILY["covid_crash_2020"][0]
        worst = min(SCENARIOS["covid_crash_2020"]["shock_returns"])
        assert abs(worst - (-0.1198)) < 0.005

    def test_readme_tables_match_the_engine(self):
        """The README's scenario table is the engine's SCENARIOS, and its
        path-vs-real table is the engine's paths plus the measured windows."""
        import re
        from pathlib import Path

        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        rows = re.findall(
            r"^\| ([A-Za-z0-9 -]+?) \| (\d{4}-\d\d-\d\d) ([\d.]+) → (\d{4}-\d\d-\d\d) ([\d.]+) "
            r"\| (-[\d.]+)% \| (\d+) \|$",
            readme, re.M,
        )
        assert {README_NAMES[r[0]] for r in rows} == set(SCENARIOS), rows
        for name, pk_d, pk, tr_d, tr, dd, td in rows:
            s = SCENARIOS[README_NAMES[name]]
            assert (s["start_date"], s["peak_close"], s["end_date"], s["trough_close"]) == (
                pk_d, float(pk), tr_d, float(tr)), name
            assert round(float(dd) / 100, 3) == s["s&p500_drawdown"], name
            assert int(td) == s["trading_days"], name

        shape_rows = re.findall(
            r"^\| ([A-Za-z0-9 -]+?) \| (-[\d.]+)% / ([\d.]+)% \| (-[\d.]+)% \((\d{4}-\d\d-\d\d)\) / ([\d.]+)% \|$",
            readme, re.M,
        )
        assert {README_NAMES[r[0]] for r in shape_rows} == set(SCENARIOS), shape_rows
        for name, p_worst, p_std, r_worst, r_date, r_std in shape_rows:
            key = README_NAMES[name]
            path = np.asarray(SCENARIOS[key]["shock_returns"])
            assert float(p_worst) == round(path.min() * 100, 2), name
            assert float(p_std) == round(path.std(ddof=1) * 100, 2), name
            assert (round(float(r_worst) / 100, 4), r_date, round(float(r_std) / 100, 4)) == REAL_GSPC_DAILY[key], name


class TestBetaApproximation:
    """Stress at beta != 1 depends on the synthetic path's roughness (beta-scaled daily
    compounding carries a volatility drag the smooth paths understate). The README says
    only beta 1 reproduces the table and quantifies the gap against the real path; these
    tests keep that table and its claims true."""

    @staticmethod
    def _engine(portfolio, key, beta):
        return StressTester(beta_to_spy=beta).run_scenario(portfolio, key).portfolio_drawdown

    def test_readme_beta_table_matches_engine_and_measurement(self, portfolio):
        import re
        from pathlib import Path

        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        rows = re.findall(
            r"^\| ([A-Za-z0-9 -]+?) \| (-[\d.]+)% / (-[\d.]+)% \| (-[\d.]+)% / (-[\d.]+)% \|$",
            readme, re.M,
        )
        assert {README_NAMES[r[0]] for r in rows} == set(SCENARIOS), rows
        for name, e05, r05, e2, r2 in rows:
            key = README_NAMES[name]
            real = REAL_GSPC_BETA_DRAWDOWN[key]
            assert float(e05) == round(self._engine(portfolio, key, 0.5) * 100, 1), name
            assert float(e2) == round(self._engine(portfolio, key, 2.0) * 100, 1), name
            assert (float(r05), float(r2)) == (round(real[0] * 100, 1), round(real[1] * 100, 1)), name

    def test_engine_is_more_severe_below_beta_one_and_less_above(self, portfolio):
        for key, (real05, real2, _) in REAL_GSPC_BETA_DRAWDOWN.items():
            assert self._engine(portfolio, key, 0.5) < real05, key
            assert self._engine(portfolio, key, 2.0) > real2, key

    def test_largest_gap_quoted_in_the_readme(self, portfolio):
        gaps = {k: self._engine(portfolio, k, 2.0) - v[1] for k, v in REAL_GSPC_BETA_DRAWDOWN.items()}
        assert round(max(gaps.values()) * 100, 1) == 3.2
        assert {k for k, g in gaps.items() if round(g * 100, 1) == 3.2} == {"dotcom_crash", "gfc_2008"}

    def test_readme_60_40_example(self, portfolio):
        """README REST example: beta 0.618, GFC engine -40.4% vs -39.0% on the real path."""
        from pathlib import Path

        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        assert round(self._engine(portfolio, "gfc_2008", 0.618) * 100, 1) == -40.4
        assert round(REAL_GSPC_BETA_DRAWDOWN["gfc_2008"][2] * 100, 1) == -39.0
        assert "-40.4% for this 60/40" in readme and "bottoms at -39.0%" in readme


class TestCustomShock:
    """custom_shock is a geometric S&P 500 decline over duration_days, scaled by the
    tester's beta, not an instantaneous shock."""

    def test_beta_scales_the_market_shock(self, portfolio):
        r = StressTester(beta_to_spy=0.5).custom_shock(portfolio, equity_shock=-0.20, duration_days=10)
        market_daily = 0.80 ** (1 / 10) - 1.0
        assert r.daily_losses == pytest.approx([0.5 * market_daily] * 10, abs=1e-15)
        assert r.portfolio_drawdown == pytest.approx((1 + 0.5 * market_daily) ** 10 - 1.0, abs=1e-12)
        assert r.scenario_drawdown == -0.20
        assert -0.20 < r.portfolio_drawdown < -0.10

    def test_zero_beta_custom_shock_is_zero(self, portfolio):
        r = StressTester(beta_to_spy=0.0).custom_shock(portfolio, equity_shock=-0.40, duration_days=5)
        assert r.portfolio_drawdown == 0.0 and r.total_loss == 0.0

    def test_one_day_shock_is_instantaneous(self, portfolio):
        r = StressTester(beta_to_spy=1.0).custom_shock(portfolio, equity_shock=-0.25, duration_days=1)
        assert r.daily_losses == [pytest.approx(-0.25, abs=1e-15)]
        assert r.duration_days == 1

    @pytest.mark.parametrize("days", [0, -3, 2.5])
    def test_duration_must_be_a_positive_whole_number(self, portfolio, days):
        with pytest.raises(ValueError):
            StressTester().custom_shock(portfolio, equity_shock=-0.2, duration_days=days)

    def test_scaled_daily_loss_beyond_total_is_rejected(self, portfolio):
        with pytest.raises(ValueError, match="100%"):
            StressTester(beta_to_spy=4.0).custom_shock(portfolio, equity_shock=-0.5, duration_days=1)


class TestScenarioFit:
    """The scenarios are equity-beta shocks only;
    beta_fit's R^2 and scenario_fit_warnings flag portfolios they cannot
    represent."""

    def test_beta_fit_hand_computed(self):
        # p = 2m + e, e orthogonal to m, both mean 0 (20 observations):
        # sum m^2 = 4 * 10 = 40, sum e^2 = 4 * 4 = 16, sum m*e = 0, so
        # beta = 2 and R^2 = 4 * 40 / (4 * 40 + 16) = 160 / 176 = 10 / 11.
        from riskcanvas.scenarios.stress_test import beta_fit

        m = np.array([-2.0, -1.0, 0.0, 1.0, 2.0] * 4)
        e = np.array([1.0, -1.0, 0.0, -1.0, 1.0] * 4)
        beta, r2 = beta_fit(2 * m + e, m)
        assert beta == pytest.approx(2.0, abs=1e-12)
        assert r2 == pytest.approx(10 / 11, abs=1e-12)
        assert beta_fit(m, m) == pytest.approx((1.0, 1.0), abs=1e-12)
        assert beta_fit(np.zeros(20), m) == (0.0, 0.0)

    def test_beta_from_returns_is_beta_fit_beta(self):
        from riskcanvas.scenarios.stress_test import beta_fit, beta_from_returns

        rng = np.random.default_rng(4)
        m = rng.normal(0, 0.01, 200)
        p = 0.7 * m + rng.normal(0, 0.01, 200)
        assert beta_from_returns(p, m) == beta_fit(p, m)[0]

    @pytest.mark.parametrize(
        "beta, r2, codes",
        [
            (1.0, 0.9, []),
            (0.6, None, []),
            (0.05, 0.02, ["low_equity_beta", "weak_equity_fit"]),
            (-0.3, 0.2, ["low_equity_beta", "weak_equity_fit"]),
            (0.0, None, ["low_equity_beta"]),
            (1.2, 0.1, ["weak_equity_fit"]),
        ],
    )
    def test_warnings(self, beta, r2, codes):
        from riskcanvas.scenarios.stress_test import scenario_fit_warnings

        assert [w["code"] for w in scenario_fit_warnings(beta, r2)] == codes

    def test_readme_bond_example_is_the_engine(self, portfolio):
        from pathlib import Path

        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        rate = StressTester(beta_to_spy=-0.29).run_scenario(portfolio, "rate_hike_2022")
        gfc = StressTester(beta_to_spy=-0.29).run_scenario(portfolio, "gfc_2008")
        assert (round(rate.total_loss * 100, 1), round(rate.portfolio_drawdown * 100, 1)) == (8.8, -3.4)
        assert round(gfc.total_loss * 100, 1) == 27.3
        assert "beta -0.29 ends that scenario up +8.8%" in readme
        assert "drawdown -3.4%) and the GFC up +27.3%" in " ".join(readme.split())

    def test_module_and_rate_hike_description_say_equity_only(self):
        import riskcanvas.scenarios.stress_test as st

        assert "Equity-market shocks only" in st.__doc__
        assert "no interest-rate" in st.__doc__
        assert "AGG" not in SCENARIOS["rate_hike_2022"]["description"]


class TestRecoveryCompounds:
    """Recovery compounds: regaining a -50% drawdown at +0.1%/day needs log(2) / log(1.001)
    = 693.5 days, where a linear |drawdown| / mean daily return reads 500."""

    def test_hand_computed(self):
        p = PortfolioLoader().from_returns_df(
            pd.DataFrame({"A": [0.001] * 300}, index=pd.date_range("2020-01-01", periods=300, freq="B")))
        r = StressTester(beta_to_spy=1.0).custom_shock(p, equity_shock=-0.5, duration_days=1)
        assert r.recovery_estimate_days == 693

    def test_volatility_drag_counts(self):
        """+5.2% / -5% alternating: arithmetic mean +0.1%/day, geometric growth negative
        (1.052 * 0.95 = 0.9994 < 1): no recovery, where a linear rule says 500 days."""
        p = PortfolioLoader().from_returns_df(
            pd.DataFrame({"A": [0.052, -0.05] * 150}, index=pd.date_range("2020-01-01", periods=300, freq="B")))
        assert p.portfolio_returns.mean() > 0
        r = StressTester(beta_to_spy=1.0).custom_shock(p, equity_shock=-0.5, duration_days=1)
        assert r.recovery_estimate_days is None


def test_demo_scenario_paths_are_the_engines():
    """The dashboard's in-browser demo compounds
    frontend/lib/scenarioPaths.mjs; it must be the engine's SCENARIOS
    exactly (paths, drawdowns, day counts), or demo and live disagree."""
    import json
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "frontend" / "lib" / "scenarioPaths.mjs").read_text()
    data = json.loads(text.split("export const SCENARIO_PATHS = ", 1)[1].rstrip().rstrip(";"))
    assert set(data) == set(SCENARIOS)
    for key, s in SCENARIOS.items():
        assert data[key]["returns"] == list(s["shock_returns"]), key
        assert data[key]["marketDrawdown"] == s["s&p500_drawdown"], key
        assert data[key]["tradingDays"] == s["trading_days"], key


def test_beta_fit_includes_the_last_day(monkeypatch):
    """yfinance's `end` is exclusive, so the SPY download asks for the day after the
    portfolio's last price. Here the last day is the one where the portfolio moves 10x
    the market, so dropping it would change the beta."""
    import sys
    import types

    from riskcanvas.scenarios.stress_test import beta_fit

    idx = pd.bdate_range("2023-01-02", periods=60)
    m = np.random.default_rng(3).normal(0.0, 0.01, 60)
    p = m.copy()
    p[-1] = 10 * m[-1] if m[-1] != 0 else 0.05
    spy_px = 400 * np.cumprod(np.r_[1.0, 1 + m])
    spy = pd.DataFrame({"Close": spy_px}, index=idx.insert(0, idx[0] - pd.Timedelta(days=3)))

    def download(sym, start=None, end=None, **_k):   # end exclusive, like yfinance
        return spy[(spy.index >= pd.Timestamp(start)) & (spy.index < pd.Timestamp(end))]

    fake = types.ModuleType("yfinance")
    fake.download = download
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    port = PortfolioLoader().from_returns_df(pd.DataFrame({"A": p}, index=idx))
    beta, _r2, estimated = StressTester().estimate_beta_fit(port)
    assert estimated
    # SPY is downloaded from the first price date, so its first return is on
    # day 2: the fit covers days 2..60, the last (10x) day included.
    expected, _ = beta_fit(p[1:], m[1:])
    assert beta == pytest.approx(expected, rel=1e-12)


def test_the_demos_weak_fit_warning_is_the_engines():
    """The in-browser demo repeats the engine's sentence and threshold
    (frontend/lib/stress.mjs); its unit test pins the JS output to this text."""
    from pathlib import Path

    from riskcanvas.scenarios.stress_test import LOW_R_SQUARED, scenario_fit_warnings

    engine = [w for w in scenario_fit_warnings(1.08, 0.08) if w["code"] == "weak_equity_fit"]
    assert engine[0]["message"] == (
        "The S&P 500 explains 8% of this portfolio's daily variance (R^2 0.08): an equity-beta "
        "scenario captures only that part; its rate, credit, FX and idiosyncratic risk are not modelled."
    )
    lib = Path(__file__).resolve().parents[1] / "frontend" / "lib"
    assert engine[0]["message"] in (lib / "stress.test.mjs").read_text()
    assert f"LOW_R_SQUARED = {LOW_R_SQUARED};" in (lib / "stress.mjs").read_text()


class TestBetaAcrossCalendars:
    """Stress beta vs SPY for an asset that does not trade SPY's calendar. Paired date by
    date, a 7-day asset's 1-day Monday return would be regressed on SPY's 3-day Monday
    return: a true beta of 1.00 would read 0.72, and the GFC -45.1% for -56.8%. The same
    data through a `spy_leg` reads 1.00."""

    DAYS = pd.date_range("2019-01-01", periods=365 * 6, freq="D")
    MARKET = pd.Series(np.random.default_rng(11).normal(0.0004, 0.01, 365 * 6), index=DAYS)
    NOISE = pd.Series(np.random.default_rng(12).normal(0.0, 0.006, 365 * 6), index=DAYS)

    @pytest.fixture
    def spy_on_weekdays(self, monkeypatch):
        """yfinance whose SPY is the latent 7-day market read on weekdays."""
        import sys
        import types

        level = 100 * (1 + self.MARKET).cumprod()
        weekdays = level[level.index.dayofweek < 5]

        def download(_sym, start=None, end=None, **_k):
            return pd.DataFrame({"Close": weekdays}).loc[pd.Timestamp(start):pd.Timestamp(end) - pd.Timedelta(days=1)]

        fake = types.ModuleType("yfinance")
        fake.download = download
        monkeypatch.setitem(sys.modules, "yfinance", fake)

    def _beta(self, series):
        portfolio = PortfolioLoader().from_returns_df(pd.DataFrame({"X": series}))
        return StressTester().estimate_beta_fit(portfolio)

    @pytest.mark.parametrize("true_beta", [0.5, 1.0, 1.5])
    def test_seven_day_asset_recovers_its_true_beta(self, spy_on_weekdays, true_beta):
        """Within 0.05 of the truth (about 3 standard errors: noise / market vol
        0.6 over some 1,560 intervals) over six years with idiosyncratic noise."""
        r = true_beta * self.MARKET + self.NOISE
        beta, r_squared, estimated = self._beta(r)
        assert estimated
        assert beta == pytest.approx(true_beta, abs=0.05)
        assert r_squared > 0.3

    def test_noiseless_seven_day_asset_is_exactly_beta_one(self, spy_on_weekdays):
        beta, r_squared, estimated = self._beta(self.MARKET)
        assert estimated
        assert beta == pytest.approx(1.0, abs=1e-9)
        assert r_squared == pytest.approx(1.0, abs=1e-9)

    def test_sunday_thursday_asset_recovers_its_true_beta(self, spy_on_weekdays):
        """Overlapping but different weeks: Friday's market move lands in the
        interval that ends on the common Monday, for both legs."""
        level = (1 + self.MARKET + self.NOISE).cumprod()
        sun_thu = level[level.index.dayofweek.isin((6, 0, 1, 2, 3))].pct_change().dropna()
        beta, _r2, estimated = self._beta(sun_thu)
        assert estimated
        assert beta == pytest.approx(1.0, abs=0.05)

    def test_spy_calendar_asset_is_the_plain_date_by_date_fit(self, spy_on_weekdays):
        """Numbers for an asset on SPY's own calendar do not move at all."""
        from riskcanvas.scenarios.stress_test import beta_fit

        level = 100 * (1 + self.MARKET).cumprod()
        spy = level[level.index.dayofweek < 5].pct_change().dropna()
        asset = 0.7 * spy + self.NOISE.reindex(spy.index)
        beta, r_squared, estimated = self._beta(asset)
        assert estimated
        # SPY's first return in the fit is on the asset's second date.
        assert (beta, r_squared) == beta_fit(asset.iloc[1:].to_numpy(), spy.iloc[1:].to_numpy())

    def test_a_total_loss_day_does_not_end_the_fit(self, spy_on_weekdays):
        """Compounding levels would turn every return after -100% into 0 / 0."""
        r = self.MARKET.copy()
        r.iloc[500] = -1.0
        beta, _r2, estimated = self._beta(r)
        assert estimated and np.isfinite(beta)

    def test_too_few_common_sessions_still_falls_back(self, spy_on_weekdays):
        beta, r_squared, estimated = self._beta(self.MARKET.iloc[:20])
        assert (beta, r_squared, estimated) == (1.0, None, False)


class TestBetaAcrossCalendarsOverHttp:
    """The endpoint, end to end, with the probe's market: a 7-day asset on the
    latent market has beta 1, so every scenario is the S&P 500's own figure."""

    DAYS = pd.date_range(end=pd.Timestamp.today().normalize(), periods=365 * 6, freq="D")

    @pytest.fixture
    def stub(self, monkeypatch):
        import sys
        import types

        market = pd.Series(np.random.default_rng(11).normal(0.0004, 0.01, len(self.DAYS)), index=self.DAYS)
        level = (1 + market).cumprod() * 100
        universe = {"SPY": level[level.index.dayofweek < 5], "BTC-USD": level}

        def download(sym, start=None, end=None, **_k):
            s = universe[sym]
            s = s[s.index >= pd.Timestamp(start)]
            return pd.DataFrame({"Close": s if end is None else s[s.index < pd.Timestamp(end)]})

        fake = types.ModuleType("yfinance")
        fake.download = download
        monkeypatch.setitem(sys.modules, "yfinance", fake)

    def test_seven_day_asset_alone_matches_the_table_and_its_spy_leg(self, stub):
        from fastapi.testclient import TestClient

        from riskcanvas.api.app import app

        client = TestClient(app)
        start = str(self.DAYS[0].date())
        alone = client.post("/portfolio/stress", json={"weights": {"BTC-USD": 1}, "start": start}).json()
        assert alone["beta_source"] == "yfinance_spy"
        assert alone["beta_used"] == pytest.approx(1.0, abs=1e-6)
        gfc = [s for s in alone["scenarios"] if "Global Financial" in s["name"]][0]
        assert gfc["portfolio_drawdown"] == -0.568

        mixed = client.post("/portfolio/stress", json={"weights": {"BTC-USD": 0.5, "SPY": 0.5}, "start": start}).json()
        assert mixed["beta_source"] == "spy_leg"
        assert mixed["beta_used"] == pytest.approx(alone["beta_used"], abs=1e-6)


class TestRecoveryStartsFromTheEndOfTheWindow:
    """At beta -0.5 a scenario ending at +19.6% has nothing to recover."""

    PORTFOLIO = PortfolioLoader().from_returns_df(pd.DataFrame(
        {"A": 0.001 + np.random.default_rng(2).normal(0, 0.004, 500)},
        index=pd.bdate_range("2020-01-01", periods=500),
    ))

    def _growth(self):
        return np.log1p(self.PORTFOLIO.portfolio_returns.to_numpy()).mean()

    def test_a_window_ending_above_the_start_has_nothing_to_recover(self):
        for r in StressTester(beta_to_spy=-0.5).run_all(self.PORTFOLIO):
            assert r.total_loss > 0 and r.portfolio_drawdown < 0, r.scenario_name
            assert r.recovery_estimate_days is None, r.scenario_name

    def test_beta_one_numbers_do_not_move(self):
        for r in StressTester(beta_to_spy=1.0).run_all(self.PORTFOLIO):
            assert r.recovery_estimate_days == int(-np.log1p(r.portfolio_drawdown) / self._growth()), r.scenario_name

    def test_it_counts_from_where_the_window_ends_not_its_low(self):
        r = StressTester(beta_to_spy=0.3).run_scenario(self.PORTFOLIO, "black_monday_1987")
        assert r.portfolio_drawdown < r.total_loss < 0
        assert r.recovery_estimate_days == int(-np.log1p(r.total_loss) / self._growth())
        assert r.recovery_estimate_days < int(-np.log1p(r.portfolio_drawdown) / self._growth())

    def test_over_http(self):
        from fastapi.testclient import TestClient

        from riskcanvas.api.app import app

        series = self.PORTFOLIO.portfolio_returns.tolist()
        out = TestClient(app).post("/portfolio/stress", json={
            "weights": {"A": 1}, "returns": {"A": series}, "start": "2020-01-01", "beta": -0.5}).json()
        assert all(s["total_loss"] > 0 and s["recovery_days_estimate"] is None for s in out["scenarios"])


class TestAssumedBetaIsNotSilent:
    """StressTester() with no beta applies an assumed 1.0, and says so."""

    def test_the_assumption_is_stated(self, portfolio, caplog):
        tester = StressTester()
        assert tester.beta_assumed is True
        with caplog.at_level("WARNING"):
            tester.run_all(portfolio)
            tester.custom_shock(portfolio, equity_shock=-0.2)
        assert sum("ASSUMED" in r.message for r in caplog.records) == 1

    def test_a_custom_shock_alone_states_it_too(self, portfolio, caplog):
        with caplog.at_level("WARNING"):
            StressTester().custom_shock(portfolio, equity_shock=-0.2)
        assert any("ASSUMED" in r.message for r in caplog.records)

    @pytest.mark.parametrize("beta", [1.0, 0, 0.5])
    def test_a_stated_beta_is_not_an_assumption(self, portfolio, caplog, beta):
        tester = StressTester(beta_to_spy=beta)
        assert tester.beta_assumed is False
        with caplog.at_level("WARNING"):
            tester.run_all(portfolio)
        assert not any("ASSUMED" in r.message for r in caplog.records)

    def test_estimating_a_beta_assumes_nothing_yet(self, portfolio, monkeypatch, caplog):
        def boom(*_a, **_k):
            raise OSError("offline")

        monkeypatch.setattr("yfinance.download", boom)
        with caplog.at_level("WARNING"):
            StressTester().estimate_beta_detailed(portfolio)
        assert not any("ASSUMED" in r.message for r in caplog.records)

    def test_the_readme_says_so(self):
        from pathlib import Path

        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        assert "never guesses" not in readme and "`tester.beta_assumed`" in readme
