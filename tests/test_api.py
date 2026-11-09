"""
API endpoint tests (FastAPI TestClient).

These exercise the REAL engine fully offline: each request supplies a `returns`
payload, so no network (yfinance) is touched. We assert 200s and genuine,
correctly-shaped numeric results from numpy/scipy.
"""

import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from riskcanvas.api.app import app
from riskcanvas.metrics.risk import RiskMetrics

client = TestClient(app)


def _returns(n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    return {
        "SPY": (0.0004 + rng.normal(0, 0.011, n)).tolist(),
        "QQQ": (0.0006 + rng.normal(0, 0.014, n)).tolist(),
        "AGG": (0.0001 + rng.normal(0, 0.004, n)).tolist(),
    }


def _body():
    return {
        "weights": {"SPY": 0.5, "QQQ": 0.3, "AGG": 0.2},
        "name": "Test Portfolio",
        "returns": _returns(),
    }


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "RiskCanvas"


def test_scenarios():
    r = client.get("/scenarios")
    assert r.status_code == 200
    scenarios = r.json()["scenarios"]
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 6
    first = scenarios[0]
    assert {"key", "name", "description", "market_drawdown"} <= set(first.keys())
    assert isinstance(first["market_drawdown"], (int, float))


def test_analyze_real_numbers():
    r = client.post("/portfolio/analyze", json=_body())
    assert r.status_code == 200
    body = r.json()
    assert body["n_assets"] == 3
    risk = body["risk"]
    perf = body["performance"]
    # Real VaR is a positive loss fraction.
    assert risk["var_95_1d"] > 0
    # CVaR >= VaR by definition.
    assert risk["cvar_95"] >= risk["var_95_1d"] - 1e-9
    # Higher confidence => higher VaR.
    assert risk["var_99_1d"] >= risk["var_95_1d"] - 1e-9
    assert risk["annual_vol"] > 0
    assert "sharpe" in perf and isinstance(perf["sharpe"], (int, float))
    # Correlation matrix present and shaped.
    cm = body["correlation_matrix"]
    assert set(cm.keys()) == {"SPY", "QQQ", "AGG"}
    assert cm["SPY"]["SPY"] == pytest.approx(1.0, abs=1e-6)


def test_var_report():
    r = client.post("/portfolio/var", json=_body())
    assert r.status_code == 200
    body = r.json()
    assert body["portfolio_value"] == 1_000_000
    report = body["var_report"]
    assert {"90%", "95%", "99%"} <= set(report.keys())
    cell = report["95%"]
    # Every key names its horizon, and parametric CVaR exists.
    assert set(cell) == {"var_1d", "var_10d", "cvar_1d", "var_parametric_1d", "cvar_parametric_1d"}
    # 10-day VaR scales up from 1-day VaR.
    assert cell["var_10d"] > cell["var_1d"] > 0
    for level in ("90%", "95%", "99%"):
        c = report[level]
        assert c["cvar_parametric_1d"] >= c["var_parametric_1d"] > 0, level
        assert c["cvar_1d"] >= c["var_1d"], level


def test_stress():
    r = client.post("/portfolio/stress", json=_body())
    assert r.status_code == 200
    scenarios = r.json()["scenarios"]
    assert len(scenarios) >= 6
    s = scenarios[0]
    assert {"name", "portfolio_drawdown", "market_drawdown", "duration_days"} <= set(s.keys())
    # Drawdowns are losses (<= 0) for crisis scenarios.
    assert s["portfolio_drawdown"] <= 0
    assert s["market_drawdown"] <= 0


def _drift_body(mean_daily: float, n: int = 400, seed: int = 13):
    """Request body whose portfolio mean daily return has a KNOWN sign (noise
    de-meaned first, so the sign is not left to sampling luck)."""
    rng = np.random.default_rng(seed)
    returns = {}
    for ticker, vol in (("SPY", 0.011), ("AGG", 0.004)):
        noise = rng.normal(0.0, vol, n)
        returns[ticker] = (noise - noise.mean() + mean_daily).tolist()
    return {
        "weights": {"SPY": 0.6, "AGG": 0.4},
        "name": f"drift {mean_daily}",
        "returns": returns,
    }


def test_stress_recovery_days_estimate_is_null_on_nonpositive_drift():
    """The recovery estimate is JSON `null`, never a negative day count, whenever the
    supplied portfolio's mean daily return is <= 0, which ordinary sampling noise
    produces about half the time."""
    r = client.post("/portfolio/stress", json=_drift_body(-0.0004))
    assert r.status_code == 200
    scenarios = r.json()["scenarios"]
    assert len(scenarios) >= 6
    for s in scenarios:
        assert "recovery_days_estimate" in s
        assert s["recovery_days_estimate"] is None, (
            f'{s["name"]}: expected null, got {s["recovery_days_estimate"]!r}'
        )


def test_stress_recovery_days_estimate_is_positive_int_on_positive_drift():
    r = client.post("/portfolio/stress", json=_drift_body(0.0006))
    assert r.status_code == 200
    for s in r.json()["scenarios"]:
        v = s["recovery_days_estimate"]
        assert isinstance(v, int) and v > 0, f'{s["name"]}: got {v!r}'


def test_stress_recovery_days_estimate_never_negative():
    """No input may produce a negative day count on the wire."""
    for mean_daily in (-0.002, -0.0004, 0.0, 0.0005):
        r = client.post("/portfolio/stress", json=_drift_body(mean_daily))
        assert r.status_code == 200
        for s in r.json()["scenarios"]:
            v = s["recovery_days_estimate"]
            assert v is None or v > 0, f"{mean_daily} / {s['name']}: got {v!r}"


def test_montecarlo():
    r = client.post("/portfolio/montecarlo?n_paths=2000&horizon_days=126", json=_body())
    assert r.status_code == 200
    body = r.json()
    assert body["n_paths"] == 2000
    assert body["horizon_days"] == 126
    assert body["var_95"] > 0
    assert 0.0 <= body["prob_loss"] <= 1.0
    assert body["expected_terminal"] > 0
    assert len(body["percentile_table"]) == 9


def test_offline_no_network_required():
    """A request with `returns` must succeed without any network access."""
    r = client.post("/portfolio/analyze", json=_body())
    assert r.status_code == 200
    assert r.json()["risk"]["var_95_1d"] > 0


@pytest.mark.parametrize(
    "endpoint",
    [
        "/portfolio/analyze",
        "/portfolio/var",
        "/portfolio/stress",
        "/portfolio/montecarlo",
    ],
)
def test_empty_weights_returns_4xx_not_500(endpoint):
    """
    Schema-valid but empty weights ({}) must yield a clean 4xx with a friendly
    message — never a 500 that leaks a raw exception string, and never a
    network call to yfinance.
    """
    r = client.post(endpoint, json={"weights": {}})
    assert r.status_code == 422, f"{endpoint} returned {r.status_code}"
    detail = str(r.json().get("detail", ""))
    assert "weights" in detail.lower()
    # No leaked internal exception text.
    assert "No price data" not in detail
    assert "Traceback" not in detail


# --- DoS-hardening bounds on unauthenticated inputs --------------------------


def test_montecarlo_n_paths_over_cap_returns_422():
    """n_paths above the 100_000 cap must be rejected cleanly, not run."""
    r = client.post("/portfolio/montecarlo?n_paths=100001&horizon_days=126", json=_body())
    assert r.status_code == 422, f"got {r.status_code}: {r.text}"


def test_montecarlo_n_paths_zero_returns_422():
    """Non-positive n_paths (< ge=1) must be rejected."""
    r = client.post("/portfolio/montecarlo?n_paths=0&horizon_days=126", json=_body())
    assert r.status_code == 422, f"got {r.status_code}: {r.text}"


def test_montecarlo_horizon_days_non_positive_returns_422():
    """Non-positive horizon_days (< ge=1) must be rejected."""
    r = client.post("/portfolio/montecarlo?n_paths=2000&horizon_days=0", json=_body())
    assert r.status_code == 422, f"got {r.status_code}: {r.text}"


def test_montecarlo_horizon_days_over_cap_returns_422():
    """horizon_days above the 2520 cap must be rejected."""
    r = client.post("/portfolio/montecarlo?n_paths=2000&horizon_days=2521", json=_body())
    assert r.status_code == 422, f"got {r.status_code}: {r.text}"


def test_montecarlo_n_paths_at_cap_ok():
    """The cap boundary (100_000) is accepted."""
    r = client.post(
        "/portfolio/montecarlo?n_paths=100000&horizon_days=5",
        json=_body(),
    )
    assert r.status_code == 200, f"got {r.status_code}: {r.text}"


@pytest.mark.parametrize(
    "endpoint",
    [
        "/portfolio/analyze",
        "/portfolio/var",
        "/portfolio/stress",
        "/portfolio/montecarlo",
    ],
)
def test_too_many_assets_returns_422(endpoint):
    """More than MAX_ASSETS (100) weights must be rejected as a validation error."""
    weights = {f"T{i}": 1.0 for i in range(101)}
    r = client.post(endpoint, json={"weights": weights})
    assert r.status_code == 422, f"{endpoint} returned {r.status_code}: {r.text}"


def test_returns_series_too_long_returns_422():
    """A returns series longer than MAX_RETURNS_LEN (20_000) must be rejected."""
    body = {
        "weights": {"SPY": 1.0},
        "name": "Big",
        "returns": {"SPY": [0.0] * 20_001},
    }
    r = client.post("/portfolio/analyze", json=body)
    assert r.status_code == 422, f"got {r.status_code}: {r.text}"


# --- Stress beta: the portfolio's market sensitivity drives the result -------

# The README table: S&P 500 peak-to-trough drawdowns
# (re-verified against ^GSPC closes 2026-10-30), keyed by wire scenario name.
DOCUMENTED_DRAWDOWNS = {
    "Black Monday (Oct 1987)": -0.335,
    "Dot-com Crash (2000-2002)": -0.491,
    "Global Financial Crisis (2008-2009)": -0.568,
    "European Debt Crisis (2011)": -0.194,
    "COVID-19 Crash (Feb-Mar 2020)": -0.339,
    "Fed Rate Hike Bear Market (2022)": -0.254,
}


def _market(n: int = 300, seed: int = 21) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0004, 0.011, n)


def _orthogonal(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    mc = m - m.mean()
    return x - (x @ mc) / (mc @ mc) * mc


def _stress(body: dict) -> dict:
    r = client.post("/portfolio/stress", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_stress_market_portfolio_reproduces_documented_table():
    """A 100% SPY portfolio has beta exactly 1 on its own SPY leg and must get
    exactly the documented drawdowns over HTTP."""
    m = _market()
    out = _stress({"weights": {"SPY": 1.0}, "returns": {"SPY": m.tolist()}})
    assert out["beta_source"] == "spy_leg"
    assert out["beta_used"] == pytest.approx(1.0, abs=1e-9)
    got = {s["name"]: s for s in out["scenarios"]}
    assert set(got) == set(DOCUMENTED_DRAWDOWNS)
    for name, dd in DOCUMENTED_DRAWDOWNS.items():
        assert got[name]["market_drawdown"] == pytest.approx(dd, abs=1e-9), name
        assert got[name]["portfolio_drawdown"] == pytest.approx(dd, abs=1e-4), name


def test_stress_explicit_beta_one_reproduces_documented_table():
    out = _stress({**_body(), "beta": 1.0})
    assert (out["beta_used"], out["beta_source"]) == (1.0, "explicit")
    for s in out["scenarios"]:
        assert s["portfolio_drawdown"] == pytest.approx(DOCUMENTED_DRAWDOWNS[s["name"]], abs=1e-4)


def test_stress_explicit_beta_zero_means_zero():
    """An explicit beta of 0 means 0; it must not fall through to the 1.0 default."""
    out = _stress({**_body(), "beta": 0})
    assert (out["beta_used"], out["beta_source"]) == (0.0, "explicit")
    for s in out["scenarios"]:
        assert s["portfolio_drawdown"] == 0.0, s["name"]


def test_stress_low_beta_portfolio_scales_down():
    """30% SPY + 70% of a series uncorrelated with SPY has beta exactly 0.3."""
    from riskcanvas.scenarios.stress_test import StressTester

    m = _market()
    agg = _orthogonal(np.random.default_rng(22).normal(0, 0.004, len(m)), m) + 0.0001
    out = _stress({"weights": {"SPY": 0.3, "AGG": 0.7},
                   "returns": {"SPY": m.tolist(), "AGG": agg.tolist()}})
    assert out["beta_source"] == "spy_leg"
    assert out["beta_used"] == pytest.approx(0.3, abs=1e-9)
    engine = {r.scenario_name: r.portfolio_drawdown for r in StressTester(0.3).run_all(_portfolio_stub(m))}
    for s in out["scenarios"]:
        assert s["market_drawdown"] < s["portfolio_drawdown"] < 0, s["name"]
        assert s["portfolio_drawdown"] == pytest.approx(engine[s["name"]], abs=1e-4), s["name"]


def _portfolio_stub(m: np.ndarray):
    import pandas as pd

    from riskcanvas.portfolio.loader import PortfolioLoader

    idx = pd.bdate_range("2020-01-01", periods=len(m))
    return PortfolioLoader().from_returns_df(pd.DataFrame({"SPY": m}, index=idx), {"SPY": 1.0})


def test_stress_portfolio_volatility_and_market_exposure_matter():
    """With `market_returns`, a near-cash 0.01%-vol portfolio and a 2%-vol equity portfolio
    get different drawdowns."""
    m = _market()
    cash = _orthogonal(np.random.default_rng(23).normal(0, 0.0001, len(m)), m) + 0.0001
    equity = 1.8 * m
    near_cash = _stress({"weights": {"BIL": 1.0}, "returns": {"BIL": cash.tolist()},
                         "market_returns": m.tolist()})
    levered = _stress({"weights": {"EQ": 1.0}, "returns": {"EQ": equity.tolist()},
                       "market_returns": m.tolist()})
    assert near_cash["beta_source"] == levered["beta_source"] == "market_returns"
    assert near_cash["beta_used"] == pytest.approx(0.0, abs=1e-9)
    assert levered["beta_used"] == pytest.approx(1.8, abs=1e-9)
    for s in near_cash["scenarios"]:
        assert s["portfolio_drawdown"] == 0.0
    gfc = {s["name"]: s["portfolio_drawdown"] for s in levered["scenarios"]}
    assert gfc["Global Financial Crisis (2008-2009)"] < -0.568


def test_stress_without_market_data_says_beta_is_assumed():
    m = _market()
    out = _stress({"weights": {"QQQ": 0.6, "AGG": 0.4},
                   "returns": {"QQQ": (1.2 * m).tolist(), "AGG": (0.1 * m).tolist()}})
    assert out["beta_source"] == "assumed"
    assert out["beta_used"] == 1.0
    assert "ASSUMED" in out["beta_note"]


def test_stress_spy_leg_too_short_falls_back_to_assumed_with_reason():
    out = _stress({"weights": {"SPY": 1.0}, "returns": {"SPY": _market(10).tolist()}})
    assert out["beta_source"] == "assumed"
    assert "could not support" in out["beta_note"] and "ASSUMED" in out["beta_note"]


@pytest.mark.parametrize("beta", [4.5, -4.5])
def test_stress_explicit_beta_out_of_range_returns_422(beta):
    r = client.post("/portfolio/stress", json={**_body(), "beta": beta})
    assert r.status_code == 422, r.text


def test_stress_estimated_beta_out_of_range_returns_422():
    m = _market()
    r = client.post("/portfolio/stress", json={
        "weights": {"X": 1.0}, "returns": {"X": (5 * m).tolist()}, "market_returns": m.tolist()})
    assert r.status_code == 422, r.text
    assert "explicit `beta`" in r.json()["detail"]


def test_market_returns_must_align_with_returns():
    body = {**_body(), "market_returns": _market(399).tolist()}
    assert client.post("/portfolio/stress", json=body).status_code == 422
    live = {"weights": {"SPY": 1.0}, "market_returns": _market(30).tolist()}
    assert client.post("/portfolio/stress", json=live).status_code == 422


# --- Weights: long-only, validated, never silently rewritten -----------------


@pytest.mark.parametrize("weights", [{"A": -1, "B": -1}, {"A": 1, "B": -1}, {"A": 0, "B": 0}])
def test_short_or_zero_weights_are_rejected_not_rewritten(weights):
    """Short or all-zero weights are a 422, never replaced by an equal-weight LONG
    portfolio."""
    rng = np.random.default_rng(31)
    body = {"weights": weights,
            "returns": {"A": rng.normal(0, 0.01, 200).tolist(), "B": rng.normal(0, 0.02, 200).tolist()}}
    r = client.post("/portfolio/var", json=body)
    assert r.status_code == 422, r.text
    assert "short" in r.text or "positive" in r.text


def test_weights_are_normalised_to_sum_one():
    rng = np.random.default_rng(32)
    rets = {"A": rng.normal(0, 0.01, 200).tolist(), "B": rng.normal(0, 0.02, 200).tolist()}

    def var95(w):
        r = client.post("/portfolio/var", json={"weights": w, "returns": rets})
        assert r.status_code == 200, r.text
        return r.json()["var_report"]["95%"]["var_1d"]

    assert var95({"A": 2, "B": 2}) == var95({"A": 0.5, "B": 0.5})
    assert var95({"A": 0.9, "B": 0.1}) != var95({"A": 0.5, "B": 0.5})
    # A zero weight is allowed (the asset is held at 0%), not rejected.
    zero_b = var95({"A": 1, "B": 0})
    only_a = client.post("/portfolio/var", json={"weights": {"A": 1}, "returns": {"A": rets["A"]}})
    assert zero_b == only_a.json()["var_report"]["95%"]["var_1d"]


@pytest.mark.parametrize("endpoint", [
    "/portfolio/analyze",
    "/portfolio/var",
    "/portfolio/stress",
    "/portfolio/montecarlo?n_paths=1000&horizon_days=10",
])
def test_weights_whose_sum_overflows_are_422_not_zeroed(endpoint):
    """{"A": 1e308, "B": 1e308} passes the per-weight finite check, but the sum overflows
    to inf and would normalise every weight to 0: /var VaR 0, /montecarlo prob_loss 1.0
    and expected_terminal 0, all HTTP 200. Rejected by the shared check_weights."""
    rng = np.random.default_rng(33)
    body = {"weights": {"A": 1e308, "B": 1e308},
            "returns": {"A": rng.normal(0, 0.01, 300).tolist(), "B": rng.normal(0, 0.012, 300).tolist()}}
    r = lenient.post(endpoint, json=body)
    assert r.status_code == 422, r.text[:300]
    assert "overflow" in r.text


def test_api_and_library_share_one_weight_rule():
    """The validator and the loaders call the same function, so a weight
    set is accepted (and normalised) identically by both."""
    import riskcanvas.api.app as api
    from riskcanvas.portfolio import loader

    assert api.check_weights is loader.check_weights
    assert (api.MIN_DAILY_RETURN, api.MAX_DAILY_RETURN, api.MIN_RETURNS_LEN) == (
        loader.MIN_DAILY_RETURN, loader.MAX_DAILY_RETURN, loader.MIN_RETURNS_LEN)


@pytest.mark.parametrize(
    "weights, returns",
    [
        ({"A": 0.5, "B": 0.5}, {"A": [0.01, -0.01, 0.0]}),              # B has no series
        ({"A": 1.0}, {"A": [0.01, -0.01, 0.0], "B": [0.0, 0.01, 0.0]}),  # B has no weight
        ({"A": 0.5, "B": 0.5}, {"A": [0.01, -0.01, 0.0], "B": [0.0, 0.01]}),  # unequal lengths
        ({"A": 1.0}, {"A": [0.01]}),                                     # too short
        ({"A": 1.0}, {"A": [0.01, -1.5, 0.0]}),                          # below -100%
    ],
)
def test_inconsistent_returns_are_rejected(weights, returns):
    r = client.post("/portfolio/analyze", json={"weights": weights, "returns": returns})
    assert r.status_code == 422, r.text


@pytest.mark.parametrize(
    "raw",
    [
        '{"weights": {"A": NaN}, "returns": {"A": [0.01, -0.01, 0.02]}}',
        '{"weights": {"A": 1}, "returns": {"A": [0.01, NaN, -0.01, 0.02]}}',
        '{"weights": {"A": 1}, "returns": {"A": [0.01, Infinity, -0.01]}}',
        '{"weights": {"A": 1}, "returns": {"A": [0.01, -0.01, 0.02]}, "market_returns": [0.0, NaN, 0.01]}',
        '{"weights": {"A": 1}, "returns": {"A": [0.01, -0.01, 0.02]}, "beta": NaN}',
    ],
)
def test_non_finite_numbers_are_rejected(raw):
    """NaN and infinite weights or returns are a 422, never a 500 and never treated as 0%."""
    for endpoint in ("/portfolio/analyze", "/portfolio/stress"):
        r = client.post(endpoint, content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422, f"{endpoint}: {r.status_code} {r.text[:200]}"


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0"])
def test_var_portfolio_value_must_be_finite_and_positive(value):
    r = client.post(f"/portfolio/var?portfolio_value={value}", json=_body())
    assert r.status_code == 422, r.text


def test_all_positive_returns_analyze_is_200_with_null_ratios():
    """Sortino / profit factor / Calmar are unbounded here, and JSON has no +inf: they are
    `null`, so encoding cannot fail into a raw 500."""
    body = {"weights": {"A": 1}, "returns": {"A": [0.001 + 0.0001 * (i % 3) for i in range(300)]}}
    r = client.post("/portfolio/analyze", json=body)
    assert r.status_code == 200, r.text
    perf = r.json()["performance"]
    assert perf["sortino"] is None and perf["profit_factor"] is None and perf["calmar"] is None


# --- Dates and request-size bounds -------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"start": "not-a-date"},
        {"start": "2020-13-45"},
        {"start": "1850-01-01"},
        {"start": "2021-01-01", "end": "2020-01-01"},
        {"start": "2999-01-01"},
    ],
)
def test_invalid_dates_return_422_without_a_download(fields, monkeypatch):
    import riskcanvas.portfolio.loader as loader_mod

    def no_network(*_a, **_k):
        raise AssertionError("validation must reject before any yfinance call")

    monkeypatch.setattr(loader_mod.PortfolioLoader, "from_tickers", no_network)
    r = client.post("/portfolio/analyze", json={"weights": {"SPY": 1.0}, **fields})
    assert r.status_code == 422, r.text


def test_live_ticker_mode_is_capped():
    weights = {f"T{i}": 1.0 for i in range(21)}
    r = client.post("/portfolio/analyze", json={"weights": weights})
    assert r.status_code == 422, r.text
    assert "at most 20 tickers" in r.text


def test_offline_observations_start_at_request_start():
    """The offline index starts at `start`, not at a fixed 2020-01-01."""
    body = {**_body(), "start": "2015-06-01"}
    r = client.post("/portfolio/analyze", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["period"].startswith("2015-06-01")


def test_montecarlo_over_compute_budget_is_rejected_before_running():
    """The parameters' product is budgeted as well as each one: 100 assets x 100k paths x
    25 days (Student-t) would run for minutes. Here 10 assets x 100k paths x 25 days =
    25M asset-path-days, over the 't' budget."""
    rng = np.random.default_rng(41)
    rets = {f"T{i}": rng.normal(0, 0.01, 100).tolist() for i in range(10)}
    body = {"weights": {t: 1.0 for t in rets}, "returns": rets}
    r = client.post("/portfolio/montecarlo?n_paths=100000&horizon_days=25&method=t", json=body)
    assert r.status_code == 422, r.text
    assert "budget" in r.json()["detail"]


def test_montecarlo_budget_boundary(monkeypatch):
    import riskcanvas.api.app as app_mod

    # n_paths 1000/1001: 1,000 is the minimum path count.
    monkeypatch.setitem(app_mod.MC_WORK_BUDGET, "gbm", 3 * 1000 * 10)
    ok = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=10&method=gbm", json=_body())
    over = client.post("/portfolio/montecarlo?n_paths=1001&horizon_days=10&method=gbm", json=_body())
    assert ok.status_code == 200, ok.text
    assert over.status_code == 422, over.text


def test_expected_terminal_is_the_mean_of_the_paths_not_their_median():
    """Zero-drift GBM is a martingale, so its mean terminal value is 1 within sampling
    error, while volatility drag pulls the median well below: exp(-sigma^2 T / 2),
    about 0.89 here. Nothing checked this field over HTTP."""
    z = np.random.default_rng(3).normal(0, 1, 300)
    series = (0.03 * (z - z.mean()) / z.std(ddof=1)).tolist()
    r = client.post("/portfolio/montecarlo?n_paths=20000&horizon_days=252", json={"weights": {"A": 1}, "returns": {"A": series}})
    out = r.json()
    se = out["standard_errors"]["expected_terminal"]
    assert 0 < se < 0.01
    assert abs(out["expected_terminal"] - 1.0) < 4 * se
    assert out["percentile_paths"]["p50"] == pytest.approx(np.exp(-0.03**2 * 252 / 2), abs=0.02)
    assert out["expected_terminal"] - out["percentile_paths"]["p50"] > 10 * se


@pytest.mark.parametrize("method", ["gbm", "t", "bootstrap"])
def test_reordering_the_json_keys_changes_nothing(method):
    """Without sorting the assets, this request and seed would read a VaR99 of 9.40% one
    way and 9.57% the other."""
    rng = np.random.default_rng(7)
    rets = {"A": rng.normal(0, 0.02, 300).tolist(), "B": rng.normal(0, 0.01, 300).tolist()}
    url = f"/portfolio/montecarlo?n_paths=2000&horizon_days=20&method={method}"
    one = client.post(url, json={"weights": {"A": 0.5, "B": 0.5}, "returns": rets})
    other = client.post(url, json={"weights": {"B": 0.5, "A": 0.5}, "returns": {"B": rets["B"], "A": rets["A"]}})
    assert one.status_code == other.status_code == 200
    assert one.json() == other.json()


class TestDefaultsAgainstTheBudget:
    """The default parameters are over budget at 10 tickers (GBM) and 5 (Student-t), so the
    422 says why they fail and what would fit."""

    @staticmethod
    def _body(n):
        rng = np.random.default_rng(42)
        rets = {f"T{i}": rng.normal(0, 0.01, 40).tolist() for i in range(n)}
        return {"weights": {t: 1.0 for t in rets}, "returns": rets}

    @pytest.mark.parametrize("method, assets, default_fit, fits", [("gbm", 10, 9, 9920), ("t", 5, 4, 9920)])
    def test_the_message_says_why_and_what_fits(self, method, assets, default_fit, fits):
        r = client.post(f"/portfolio/montecarlo?method={method}", json=self._body(assets))
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert f"the defaults, 10,000 paths over 252 days, fit {default_fit} assets" in detail
        assert f"at horizon_days=252 this portfolio fits n_paths up to {fits:,}" in detail

    def test_what_it_names_fits_and_one_path_more_does_not(self, monkeypatch):
        """Exact, on a small budget so no simulation has to run long."""
        import riskcanvas.api.app as app_mod

        monkeypatch.setitem(app_mod.MC_WORK_BUDGET, "gbm", 5 * 10 * 1000)
        over = client.post("/portfolio/montecarlo?n_paths=2000&horizon_days=10&method=gbm", json=self._body(5))
        assert "fits n_paths up to 1,000" in over.json()["detail"]
        assert client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=10&method=gbm", json=self._body(5)).status_code == 200
        assert client.post("/portfolio/montecarlo?n_paths=1001&horizon_days=10&method=gbm", json=self._body(5)).status_code == 422

    def test_when_no_path_count_fits_it_names_the_horizon(self):
        body = self._body(100)
        r = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=2520&method=t", json=body)
        assert r.status_code == 422
        assert "at n_paths=1,000, the fewest allowed, this portfolio fits horizon_days up to 125" in r.json()["detail"]

    @pytest.mark.parametrize("method, assets", [("gbm", 9), ("t", 4)])
    def test_the_defaults_run_at_the_asset_count_it_names(self, method, assets):
        r = client.post(f"/portfolio/montecarlo?method={method}", json=self._body(assets))
        assert r.status_code == 200, r.text[:200]
        assert r.json()["n_paths"] == 10_000 and r.json()["horizon_days"] == 252


def test_dashboard_monte_carlo_request_fits_the_budget():
    """The dashboard asks for 5,000 GBM paths over 252 days for up to its 9
    assets; that must stay inside the budget it is subject to."""
    from riskcanvas.api.app import MC_WORK_BUDGET

    assert 9 * 5_000 * 252 <= MC_WORK_BUDGET["gbm"]


def test_montecarlo_bootstrap_short_history_returns_422():
    body = {"weights": {"A": 1.0}, "returns": {"A": [0.01, -0.01] * 5}}
    r = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=252&method=bootstrap", json=body)
    assert r.status_code == 422, r.text
    assert "block_size" in r.json()["detail"]


def test_montecarlo_percentile_bands_on_the_wire():
    r = client.post("/portfolio/montecarlo?n_paths=2000&horizon_days=126", json=_body())
    assert r.status_code == 200, r.text
    body = r.json()
    bands = body["percentile_bands"]
    assert bands[0]["day"] == 0 and bands[0]["p50"] == 1.0
    assert bands[-1]["day"] == 126
    assert bands[-1]["p50"] == pytest.approx(body["percentile_paths"]["p50"], abs=1e-6)
    for b in bands:
        assert b["p5"] <= b["p25"] <= b["p50"] <= b["p75"] <= b["p95"]


def test_validation_errors_do_not_echo_the_request_body():
    """FastAPI's default 422 echoes the offending input; for a body-level check
    that is the whole (up to ~40 MB) request."""
    body = {**_body(), "weights": {"SPY": 1.0}}
    r = client.post("/portfolio/analyze", json=body)
    assert r.status_code == 422
    assert len(r.content) < 2_000, len(r.content)
    for err in r.json()["detail"]:
        assert set(err) == {"type", "loc", "msg"}


@pytest.mark.parametrize(
    "url, body",
    [
        ("/portfolio/analyze", {"weights": {"T" * 33: 1.0}, "returns": {"T" * 33: [0.01, -0.01, 0.0]}}),
        ("/portfolio/analyze", {**_body(), "name": "x" * 201}),
        ("/portfolio/var?portfolio_value=2e15", _body()),
        ("/portfolio/analyze", {"weights": {"A": 1.0}, "returns": {"A": [0.01, 10.5, 0.0]}}),
        ("/portfolio/stress", {**_body(), "market_returns": [11.0] + [0.0] * 399}),
    ],
)
def test_remaining_documented_bounds_return_422(url, body):
    """SECURITY.md lists every enforced bound; each one has a test."""
    r = client.post(url, json=body)
    assert r.status_code == 422, r.text


# --- Undefined statistics are null, never a raw 500 ------

# A client that returns 500s instead of raising them, so a test can assert on
# the status code a real caller would see.
lenient = TestClient(app, raise_server_exceptions=False)


def _strict_json(text: str):
    """Parse like a strict JSON client: NaN / Infinity tokens are an error."""
    import json

    def reject(token):
        raise ValueError(f"non-standard JSON token {token!r}")

    return json.loads(text, parse_constant=reject)


def test_cash_leg_correlations_serialise_as_null():
    """Any constant-return leg (cash) has an undefined
    correlation; pandas gives NaN, which crashed JSON encoding into a raw 500
    after the endpoint's own error handling."""
    body = {"weights": {"A": 0.9, "C": 0.1},
            "returns": {"A": [0.01, -0.02, 0.03], "C": [0, 0, 0]}}
    r = lenient.post("/portfolio/analyze", json=body)
    assert r.status_code == 200, r.text
    cm = _strict_json(r.text)["correlation_matrix"]
    assert cm["A"]["A"] == pytest.approx(1.0)
    assert cm["A"]["C"] is None and cm["C"]["A"] is None and cm["C"]["C"] is None


def test_spy_cash_90_10_analyze_is_200_with_defined_correlations_kept():
    rng = np.random.default_rng(51)
    body = {"weights": {"SPY": 0.9, "CASH": 0.1},
            "returns": {"SPY": rng.normal(0.0004, 0.011, 300).tolist(), "CASH": [0.0001] * 300}}
    r = lenient.post("/portfolio/analyze", json=body)
    assert r.status_code == 200, r.text
    out = _strict_json(r.text)
    assert out["correlation_matrix"]["SPY"]["SPY"] == pytest.approx(1.0)
    assert out["correlation_matrix"]["SPY"]["CASH"] is None
    assert out["risk"]["var_95_1d"] > 0


DEGENERATE_BODIES = {
    "cash leg": {"weights": {"A": 0.9, "C": 0.1},
                 "returns": {"A": [0.01, -0.02, 0.03], "C": [0, 0, 0]}},
    "constant series": {"weights": {"A": 1}, "returns": {"A": [0.001] * 50}},
    "two equal observations": {"weights": {"A": 1}, "returns": {"A": [0.01, 0.01]}},
    "all zero": {"weights": {"A": 0.5, "B": 0.5}, "returns": {"A": [0.0] * 30, "B": [0.0] * 30}},
    "total loss on day one": {"weights": {"A": 1}, "returns": {"A": [-1.0, 0.01, 0.02]}},
    "+1000% every day": {"weights": {"A": 1}, "returns": {"A": [10.0] * 300}},
    "+1000% every day, accepted": {"weights": {"A": 1}, "returns": {"A": [10.0] * 300}, "allow_large_returns": True},
    "subnormal drift": {"weights": {"A": 1}, "returns": {"A": [1e-310, 1e-310]}},
}


@pytest.mark.parametrize("endpoint", [
    "/portfolio/analyze",
    "/portfolio/var",
    "/portfolio/stress",
    "/portfolio/montecarlo?n_paths=1000&horizon_days=10",
])
@pytest.mark.parametrize("case", list(DEGENERATE_BODIES))
def test_degenerate_inputs_never_500_and_always_strict_json(endpoint, case):
    """Sweep: every endpoint answers every degenerate-but-valid body with a 200
    or a clean 422, and the body is standard JSON (no NaN / Infinity)."""
    r = lenient.post(endpoint, json=DEGENERATE_BODIES[case])
    assert r.status_code in (200, 422), f"{endpoint} / {case}: {r.status_code} {r.text[:200]}"
    _strict_json(r.text)


@pytest.mark.parametrize("method", ["gbm", "t", "bootstrap"])
def test_montecarlo_overflow_is_a_clear_422_not_a_500(method):
    """+50-60% a day compounded over 2,520 days overflows float64: a clear 422, never
    inf/nan statistics or a raw 500."""
    body = {"weights": {"A": 1}, "returns": {"A": [0.5, 0.6] * 100}, "allow_large_returns": True}
    r = lenient.post(f"/portfolio/montecarlo?n_paths=1000&horizon_days=2520&method={method}&drift=historical", json=body)
    assert r.status_code == 422, r.text[:200]
    assert "overflow" in r.json()["detail"]


_ALTERNATING = [3.0, 5.0, 2.0, 7.0] * 200


@pytest.mark.parametrize("method", ["gbm", "t", "bootstrap"])
@pytest.mark.parametrize("drift", ["zero", "historical", "risk_free"])
@pytest.mark.parametrize("horizon", [1, 252, 2520])
@pytest.mark.parametrize("series", [
    _ALTERNATING,
    [10.0] * 300,
    [10.0, 9.0] * 150,
    [10.0, -1.0] * 50,
    [0.15, 0.16, 0.14, 0.15] * 200,
    [1e-300, -1e-300] * 50,
], ids=["alternating", "+1000%", "+1000/900%", "+1000/-100%", "+15%", "subnormal"])
def test_montecarlo_extreme_returns_never_500(method, drift, horizon, series):
    """`(es - var) ** 2` overflows on values below the float64 limit, including with
    `drift=historical`, which the sweep above does not cover: README promises a 200 or a
    422 for every method and drift."""
    url = f"/portfolio/montecarlo?n_paths=1000&horizon_days={horizon}&method={method}&drift={drift}"
    r = lenient.post(url, json={"weights": {"A": 1}, "returns": {"A": series}, "allow_large_returns": True})
    assert r.status_code in (200, 422), f"{url}: {r.status_code} {r.text[:200]}"
    _strict_json(r.text)
    if r.status_code == 422:
        assert "overflow" in r.json()["detail"] or "re-centres" in r.json()["detail"], r.text[:200]


def test_montecarlo_values_too_large_to_summarise_are_the_documented_422():
    """Both routes: returns of +200-700% a day, and +15% a day over 2,520 days."""
    for series, q in (
        (_ALTERNATING, "n_paths=1000&horizon_days=252&method=bootstrap&drift=historical"),
        ([0.15, 0.16, 0.14, 0.15] * 200, "n_paths=1000&horizon_days=2520&method=gbm&drift=historical"),
    ):
        r = lenient.post(f"/portfolio/montecarlo?{q}", json={"weights": {"A": 1}, "returns": {"A": series}, "allow_large_returns": True})
        assert r.status_code == 422, f"{q}: {r.status_code} {r.text[:200]}"
        assert "overflow" in r.json()["detail"]


def test_stress_subnormal_drift_recovery_is_null_not_a_500():
    """A positive but subnormal mean return overflows drawdown / drift to inf, and int(inf)
    raises: the estimate is `null`, not a 500."""
    r = lenient.post("/portfolio/stress", json=DEGENERATE_BODIES["subnormal drift"])
    assert r.status_code == 200, r.text
    for s in r.json()["scenarios"]:
        assert s["recovery_days_estimate"] is None, s["name"]


def test_analyze_day_one_loss_drawdown_over_http():
    """Over the wire: risk and performance both report the -30% first day as the max
    drawdown, and Calmar is negative."""
    body = {"weights": {"A": 1}, "returns": {"A": [-0.30, 0.01, 0.01]}}
    r = client.post("/portfolio/analyze", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["risk"]["max_drawdown"] == -0.3
    assert out["performance"]["max_drawdown"] == -0.3
    assert out["performance"]["calmar"] < 0


# --- Live mode never fabricates prices or drops tickers -


@pytest.fixture
def stub_yfinance(monkeypatch):
    """The no-network price stub of tests/test_loader.py."""
    import sys
    import types

    from tests.test_loader import _download

    fake = types.ModuleType("yfinance")
    fake.download = _download
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def test_live_late_listed_ticker_is_a_422_naming_it(stub_yfinance):
    """RIVN listed 2021-11-10, requested from 2020-01-01: a 422, never a 200 with a
    back-filled RIVN (fabricated 0% days)."""
    body = {"weights": {"SPY": 0.5, "RIVN": 0.5}, "start": "2020-01-01", "end": "2024-03-01"}
    for endpoint in ("/portfolio/var", "/portfolio/analyze", "/portfolio/stress", "/portfolio/montecarlo"):
        r = client.post(endpoint, json=body)
        assert r.status_code == 422, f"{endpoint}: {r.status_code} {r.text[:200]}"
        assert "RIVN (first available 2021-11-10)" in r.json()["detail"]


def test_live_unknown_ticker_is_a_422_not_a_one_asset_200(stub_yfinance):
    body = {"weights": {"SPY": 0.5, "RIVNN": 0.5}, "start": "2020-01-01", "end": "2024-03-01"}
    r = client.post("/portfolio/analyze", json=body)
    assert r.status_code == 422, r.text
    assert "RIVNN" in r.json()["detail"]


def test_live_common_window_is_200_with_every_ticker(stub_yfinance):
    body = {"weights": {"SPY": 0.5, "RIVN": 0.5}, "start": "2021-11-10", "end": "2024-03-01"}
    r = client.post("/portfolio/analyze", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["n_assets"] == 2
    assert out["period"].startswith("2021-11-10")



def test_stress_beta_one_total_loss_and_duration_match_the_scenario_window():
    """Over the wire, for a beta-1 portfolio: the end-of-window loss equals
    the max drawdown equals the README table (every path ends at its trough),
    and duration_days is the window's trading days as /scenarios lists them."""
    m = _market()
    out = _stress({"weights": {"SPY": 1.0}, "returns": {"SPY": m.tolist()}})
    windows = {s["name"]: s for s in client.get("/scenarios").json()["scenarios"]}
    for s in out["scenarios"]:
        dd = DOCUMENTED_DRAWDOWNS[s["name"]]
        assert s["total_loss"] == pytest.approx(dd, abs=1e-4), s["name"]
        assert s["portfolio_drawdown"] == pytest.approx(dd, abs=1e-4), s["name"]
        assert s["duration_days"] == windows[s["name"]]["trading_days"], s["name"]


def test_scenarios_endpoint_reports_each_window():
    from riskcanvas.scenarios.stress_test import SCENARIOS

    got = {s["key"]: s for s in client.get("/scenarios").json()["scenarios"]}
    for key, s in SCENARIOS.items():
        g = got[key]
        assert (g["peak_date"], g["trough_date"]) == (s["start_date"], s["end_date"])
        assert (g["trading_days"], g["calendar_days"]) == (s["trading_days"], s["calendar_days"])



def test_var_parametric_cvar_matches_the_engine_closed_form():
    """Parametric CVaR is served at 90/95/99%, as the README states."""
    import pandas as pd

    from riskcanvas.metrics.risk import RiskMetrics

    body = _body()
    r = client.post("/portfolio/var?portfolio_value=1000000", json=body)
    assert r.status_code == 200, r.text
    rets = pd.DataFrame(body["returns"])
    w = pd.Series(body["weights"]) / sum(body["weights"].values())
    rm = RiskMetrics((rets * w).sum(axis=1))
    for level, conf in (("90%", 0.90), ("95%", 0.95), ("99%", 0.99)):
        # Unrounded (see the next test).
        assert r.json()["var_report"][level]["cvar_parametric_1d"] == pytest.approx(
            rm.cvar_parametric(conf) * 1e6, rel=1e-12)


def test_var_is_not_rounded_away_for_a_small_portfolio_value():
    """/portfolio/var does not round: whole currency units would zero every VaR and CVaR of
    this ~1%-vol portfolio at portfolio_value=1. The figures are the engine's fractions
    there, and they scale linearly."""
    import pandas as pd

    from riskcanvas.metrics.risk import RiskMetrics

    body = _body()
    one = client.post("/portfolio/var?portfolio_value=1", json=body)
    big = client.post("/portfolio/var?portfolio_value=1000000", json=body)
    assert one.status_code == big.status_code == 200
    rets = pd.DataFrame(body["returns"])
    rm = RiskMetrics((rets * pd.Series(body["weights"])).sum(axis=1))
    for level, conf in (("90%", 0.90), ("95%", 0.95), ("99%", 0.99)):
        cell = one.json()["var_report"][level]
        assert cell["var_1d"] == pytest.approx(rm.var_historical(conf), rel=1e-12)
        assert cell["cvar_1d"] == pytest.approx(rm.cvar_historical(conf), rel=1e-12)
        assert cell["var_parametric_1d"] == pytest.approx(rm.var_parametric(conf), rel=1e-12)
        for key, value in cell.items():
            assert 0 < value < 0.2, (level, key, value)
            assert big.json()["var_report"][level][key] == pytest.approx(value * 1e6, rel=1e-12)



# --- Monte Carlo: seed, sampling error, minimum paths ---


@pytest.mark.parametrize("n_paths", [1, 999])
def test_montecarlo_below_minimum_paths_is_422(n_paths):
    """A '99% VaR' from one path means nothing, so n_paths has a minimum."""
    r = client.post(f"/portfolio/montecarlo?n_paths={n_paths}&horizon_days=20", json=_body())
    assert r.status_code == 422, r.text


def test_montecarlo_seed_is_exposed_and_echoed():
    """The default seed repeats a request's sample, and another seed draws an independent
    one."""
    url = "/portfolio/montecarlo?n_paths=2000&horizon_days=60"
    default, again = client.post(url, json=_body()).json(), client.post(url, json=_body()).json()
    other = client.post(url + "&seed=7", json=_body()).json()
    assert default["seed"] == 42 and other["seed"] == 7
    assert default["var_95"] == again["var_95"]
    assert other["var_95"] != default["var_95"]
    assert client.post(url + "&seed=-1", json=_body()).status_code == 422


def test_montecarlo_reports_standard_errors_and_cvar_99():
    r = client.post("/portfolio/montecarlo?n_paths=5000&horizon_days=60", json=_body())
    assert r.status_code == 200, r.text
    out = r.json()
    se = out["standard_errors"]
    assert set(se) == {"var_95", "var_99", "cvar_95", "cvar_99", "prob_loss", "expected_terminal"}
    assert all(v is not None and v > 0 for v in se.values()), se
    # Sampling noise is small next to the estimates at 5,000 paths.
    assert se["var_95"] < 0.2 * out["var_95"]
    assert out["cvar_99"] >= out["var_99"] >= out["var_95"] > 0


def test_var_constant_series_historical_and_parametric_agree_over_http():
    """A constant +1% series reads -10,000 (per 1,000,000) historical and parametric alike:
    even the worst day is a 1% gain."""
    body = {"weights": {"A": 1}, "returns": {"A": [0.01] * 50}}
    r = client.post("/portfolio/var", json=body)
    assert r.status_code == 200, r.text
    for level, cell in r.json()["var_report"].items():
        for key in ("var_1d", "cvar_1d", "var_parametric_1d", "cvar_parametric_1d"):
            assert cell[key] == pytest.approx(-10_000, abs=1e-6), (level, key)


def test_no_var_figure_exceeds_the_portfolio_value():
    """Uncapped sqrt-of-time scaling would report a 10-day VaR of 316% of a long-only
    portfolio (var_95_10d 3.16)."""
    body = {"weights": {"A": 1}, "returns": {"A": [10.0, -0.9999] * 50}}
    r = client.post("/portfolio/var?portfolio_value=1000", json=body)
    assert r.status_code == 200, r.text
    for level, cell in r.json()["var_report"].items():
        assert cell["var_10d"] == pytest.approx(1000), level
        assert all(v <= 1000 + 1e-9 for v in cell.values()), (level, cell)
    risk = client.post("/portfolio/analyze", json=body).json()["risk"]
    assert risk["var_95_10d"] == 1.0 and risk["var_99_10d"] == 1.0


# --- Stress scenarios are equity-market shocks only ------


def _bond_and_equity(seed=3, n=750):
    """p3's long-duration bond (negative equity beta, as 2018-21) and an
    equity-like series, both against one synthetic market."""
    rng = np.random.default_rng(seed)
    mkt = rng.standard_normal(n) * 0.01
    bond = -0.3 * mkt + rng.standard_normal(n) * 0.006
    equity = 1.1 * mkt + rng.standard_normal(n) * 0.005
    return mkt, bond, equity


def test_stress_warns_that_a_bond_portfolio_is_not_represented():
    """A negative-beta bond GAINED in every scenario
    (+8.9% in "Fed Rate Hike Bear Market (2022)", +27.5% in the GFC) with
    no flag that an equity-beta scenario cannot describe it."""
    mkt, bond, _ = _bond_and_equity()
    out = _stress({"weights": {"TLT": 1.0}, "returns": {"TLT": bond.tolist()},
                   "market_returns": mkt.tolist()})
    assert out["beta_used"] < 0
    assert 0 <= out["beta_r_squared"] < 0.3
    codes = {w["code"] for w in out["warnings"]}
    assert codes == {"low_equity_beta", "weak_equity_fit"}
    low = next(w for w in out["warnings"] if w["code"] == "low_equity_beta")
    assert "gains" in low["message"] and "rate" in low["message"]


def test_stress_has_no_warning_for_an_equity_like_portfolio():
    mkt, _, equity = _bond_and_equity()
    out = _stress({"weights": {"QQQ": 1.0}, "returns": {"QQQ": equity.tolist()},
                   "market_returns": mkt.tolist()})
    assert out["beta_used"] == pytest.approx(1.1, abs=0.05)
    assert out["beta_r_squared"] > 0.8
    assert out["warnings"] == []


def test_stress_warns_for_an_explicit_near_zero_beta_and_an_assumed_one():
    body = {"weights": {"AGG": 1.0}, "returns": {"AGG": _market(300).tolist()}}
    explicit = _stress({**body, "beta": 0.05})
    assert [w["code"] for w in explicit["warnings"]] == ["low_equity_beta"]
    assert explicit["beta_r_squared"] is None
    assumed = _stress(body)
    assert assumed["beta_source"] == "assumed"
    assert [w["code"] for w in assumed["warnings"]] == ["assumed_beta"]


def test_rate_hike_scenario_no_longer_cites_a_bond_return_it_does_not_model():
    """The description cites no bond return it does not model: a 0.05-beta AGG portfolio
    shows -1.45% in the scenario."""
    s = {x["key"]: x for x in client.get("/scenarios").json()["scenarios"]}["rate_hike_2022"]
    assert "AGG" not in s["description"] and "-13.0%" not in s["description"]
    assert "no interest-rate shock" in s["description"]


def test_live_single_late_listed_ticker_is_a_422(stub_yfinance):
    """{"COIN": 1} from 2015 is a 422: the late-listing check compares each ticker with
    `start`, not with the other tickers."""
    body = {"weights": {"RIVN": 1.0}, "start": "2020-01-01", "end": "2024-03-01"}
    for endpoint in ("/portfolio/analyze", "/portfolio/var", "/portfolio/stress", "/portfolio/montecarlo"):
        r = client.post(endpoint, json=body)
        assert r.status_code == 422, f"{endpoint}: {r.status_code} {r.text[:200]}"
        assert "RIVN (first available 2021-11-10)" in r.json()["detail"]


def test_live_sole_delisted_ticker_without_end_is_a_422(stub_yfinance):
    """README: "even when it is the only ticker". Never a 200 with `warnings: []` and a
    window ending at the ticker's own last price."""
    r = client.post("/portfolio/analyze", json={"weights": {"GONE": 1.0}, "start": "2020-01-01"})
    assert r.status_code == 422, r.text[:200]
    assert "GONE (last available 2022-06-30)" in r.json()["detail"]


def test_live_long_interior_gap_is_a_422(stub_yfinance):
    """A 126-day hole must not become one 'daily' return."""
    body = {"weights": {"SPY": 0.5, "HOLE": 0.5}, "start": "2020-01-01", "end": "2024-03-01"}
    r = client.post("/portfolio/var", json=body)
    assert r.status_code == 422, r.text[:200]
    assert "HOLE has no prices between 2022-02-28 and 2022-07-04" in r.json()["detail"]


@pytest.mark.parametrize("endpoint", ["/portfolio/analyze", "/portfolio/var", "/portfolio/stress", "/portfolio/montecarlo"])
def test_every_endpoint_reports_the_window_it_used(stub_yfinance, endpoint):
    body = {"weights": {"SPY": 0.5, "RIVN": 0.5}, "start": "2021-11-10", "end": "2024-03-01"}
    out = client.post(endpoint, json=body).json()
    assert out["period"] == "2021-11-10 → 2024-02-29"
    assert out["n_observations"] == len(pd.bdate_range("2021-11-10", "2024-02-29")) - 1
    offline = client.post(endpoint, json=_body()).json()
    assert offline["n_observations"] == 400 and offline["period"].startswith("2020-01-01")


@pytest.mark.parametrize("method", ["GBM", "Bootstrap", "normal", ""])
def test_unknown_montecarlo_method_is_a_422_like_every_other_bad_input(method):
    """`method=GBM` is the documented 422, like every other invalid input."""
    r = client.post(f"/portfolio/montecarlo?n_paths=1000&horizon_days=5&method={method}", json=_body())
    assert r.status_code == 422, r.text
    assert r.json()["detail"][0]["loc"] == ["query", "method"]


def _bdays_ending_today(n, today):
    return pd.bdate_range(end=today, periods=n)


def _default_start() -> date:
    """`start` when a request omits it (2020-01-01)."""
    from riskcanvas.api.app import PortfolioRequest

    return PortfolioRequest.model_fields["start"].default


# Returns mode dates its observations and refuses any after today, so every test of that
# rule runs at today's date and again at later ones: a fixed date or length that only fits
# today's calendar (2,000 business days from the default start end 2027-08-31, 120 months
# 2029-12-01) fails now, not on the day it runs out.
LATER_TODAYS = ("2029-12-03", "2071-03-16")


@pytest.fixture(params=[None, *LATER_TODAYS], ids=["today", *LATER_TODAYS])
def today(request, monkeypatch):
    """The date the API reads as today: the real one, then each of ``LATER_TODAYS``.
    ``date.today()`` is ``fromtimestamp(time.time())`` and calls ``time.time``, so
    shifting it moves the API's clock. Pandas' ``Timestamp.today()`` reads the C clock
    and does not move, so these tests take today from here, never from pandas."""
    if request.param is None:
        return date.today()
    later = date.fromisoformat(request.param)
    real = time.time
    shift = datetime(later.year, later.month, later.day, 12).timestamp() - real()
    monkeypatch.setattr(time, "time", lambda: real() + shift)
    assert date.today() == later, "the clock did not move, so this run would prove nothing"
    return later


@pytest.mark.parametrize("start", ["9999-01-01", "2262-04-01"])
def test_returns_mode_start_in_the_future_is_a_422(start):
    """With `returns`, the observations' dates must end by today: "9999-01-01" is a 422,
    not a 200 with a period in 9999 and a positive_months share counted over invented
    months."""
    r = client.post("/portfolio/analyze",
                    json={"weights": {"A": 1}, "returns": {"A": [0.01, -0.02, 0.0]}, "start": start})
    assert r.status_code == 422, r.text[:200]
    assert "future" in r.text


def test_returns_mode_dates_running_past_today_are_a_422(today):
    idx = _bdays_ending_today(30, today)
    body = {"weights": {"A": 1}, "returns": {"A": [0.001] * 30}, "start": idx[-5].date().isoformat()}
    r = client.post("/portfolio/var", json=body)
    assert r.status_code == 422, r.text[:200]
    assert f"on or before {idx[0].date().isoformat()}" in r.text


def test_returns_mode_window_ending_today_is_accepted(today):
    idx = _bdays_ending_today(30, today)
    body = {"weights": {"A": 1}, "returns": {"A": [0.001, -0.002] * 15}, "start": idx[0].date().isoformat()}
    r = client.post("/portfolio/analyze", json=body)
    assert r.status_code == 200, r.text[:200]
    assert r.json()["period"].endswith(idx[-1].date().isoformat())


# --- Thin histories are flagged ---------------------------


def _short(n, seed=61):
    rng = np.random.default_rng(seed)
    return {"weights": {"A": 1}, "returns": {"A": rng.normal(0.0005, 0.01, n).tolist()}}


def _codes(out):
    return [w["code"] for w in out["warnings"]]


def test_var_from_three_observations_is_flagged():
    """A 99% VaR/CVaR from 2-3 observations is flagged; so is anything under the 20 a
    stress beta needs."""
    out = client.post("/portfolio/var", json=_short(3)).json()
    assert _codes(out) == ["short_history_tail", "short_history"]
    assert "90%, 95%, 99%" in out["warnings"][0]["message"]
    assert out["var_report"]["99%"]["var_1d"] > 0   # still returned


def test_tail_rule_is_per_confidence_level():
    out = client.post("/portfolio/var", json=_short(50)).json()
    assert _codes(out) == ["short_history_tail"]
    msg = out["warnings"][0]["message"]
    assert "at 99% rest" in msg and "99% needs 100" in msg and "95%" not in msg


def test_long_history_has_no_warnings_on_any_endpoint():
    """With its rate stated, a long history has nothing to flag anywhere. Left unstated,
    only /analyze, whose annual figures use the rate, says it assumed 252."""
    endpoints = ("/portfolio/analyze", "/portfolio/var", "/portfolio/montecarlo?n_paths=1000&horizon_days=5")
    for endpoint in endpoints:
        assert client.post(endpoint, json={**_body(), "periods_per_year": 252}).json()["warnings"] == [], endpoint
    for endpoint in endpoints[1:]:
        assert client.post(endpoint, json=_body()).json()["warnings"] == [], endpoint
    assert _codes(client.post(endpoints[0], json=_body()).json()) == ["assumed_periods_per_year"]


def test_short_history_flags_analyze_montecarlo_and_stress():
    body = _short(15)
    assert _codes(client.post("/portfolio/analyze", json=body).json()) == [
        "short_history_tail", "short_history", "assumed_periods_per_year"]
    assert _codes(client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=5", json=body).json()) == ["short_history"]
    assert _codes(client.post("/portfolio/stress", json={**body, "beta": 1.0}).json()) == ["short_history"]


def test_bootstrap_with_few_independent_windows_is_flagged():
    """The bootstrap's horizon outcomes recombine overlapping historical windows; under 100
    independent horizon-length windows (n_obs // horizon) its 99% tail rests on less
    than one independent observation."""
    url = "/portfolio/montecarlo?n_paths=1000&method=bootstrap&horizon_days="
    few = client.post(url + "1", json=_short(40)).json()
    assert "few_bootstrap_windows" in _codes(few)
    assert "40 observations hold only 40 independent 1-day" in next(
        w for w in few["warnings"] if w["code"] == "few_bootstrap_windows")["message"]
    assert _codes(client.post(url + "1", json=_body()).json()) == []        # 400 windows
    assert "few_bootstrap_windows" in _codes(client.post(url + "60", json=_short(40)).json())
    assert "few_bootstrap_windows" in _codes(client.post(url + "10", json=_short(300)).json())   # 30 windows
    assert "few_bootstrap_windows" not in _codes(client.post(url + "2", json=_short(300)).json())  # 150


def test_montecarlo_states_that_standard_errors_are_simulation_error_only():
    out = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=5", json=_body()).json()
    assert "Monte Carlo sampling error only" in out["standard_errors_note"]
    assert "estimation error" in out["standard_errors_note"]


def _trending_body(n=504):
    z = np.random.default_rng(8).standard_normal(n)
    return {"weights": {"A": 1}, "returns": {"A": (0.0012 + 0.008 * (z - z.mean()) / z.std(ddof=1)).tolist()}}


def test_montecarlo_default_drift_is_zero_and_echoed():
    """Two strong years projected a one-year tail
    that is a gain, from an undisclosed in-sample drift."""
    out = client.post("/portfolio/montecarlo?n_paths=5000&horizon_days=252", json=_trending_body()).json()
    assert out["drift"] == "zero" and out["var_99"] > 0.15 and out["warnings"] == []


def test_historical_drift_that_turns_var_into_a_gain_is_flagged():
    out = client.post("/portfolio/montecarlo?n_paths=5000&horizon_days=252&drift=historical",
                      json=_trending_body()).json()
    assert out["drift"] == "historical" and out["var_95"] < 0
    assert _codes(out) == ["historical_drift_gain"]
    r = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=5&drift=mean", json=_body())
    assert r.status_code == 422


@pytest.mark.parametrize("endpoint", ["/portfolio/analyze", "/portfolio/var", "/portfolio/stress",
                                      "/portfolio/montecarlo?n_paths=1000&horizon_days=5"])
def test_weekend_start_in_returns_mode_is_reported(endpoint):
    """A weekend `start` moves to the next Monday, and the response says so."""
    body = {**_body(), "start": "2024-01-06"}                    # a Saturday
    out = client.post(endpoint, json=body).json()
    moved = [w for w in out["warnings"] if w["code"] == "start_moved_to_business_day"]
    assert len(moved) == 1 and "2024-01-06 is a Saturday" in moved[0]["message"]
    assert "first is 2024-01-08 (Monday)" in moved[0]["message"]
    assert out["period"].startswith("2024-01-08")
    weekday = client.post(endpoint, json={**_body(), "start": "2024-01-08"}).json()
    assert all(w["code"] != "start_moved_to_business_day" for w in weekday["warnings"])


def test_calendar_daily_series_warns_that_the_horizon_is_calendar_days(stub_yfinance):
    """A 7-day asset alone: horizon_days counts calendar days."""
    body = {"weights": {"COIN7": 1.0}, "start": "2020-01-01", "end": "2024-03-01"}
    out = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=252", json=body).json()
    assert "calendar_day_horizon" in _codes(out)
    analyze = client.post("/portfolio/analyze", json=body).json()
    assert analyze["risk"]["periods_per_year"] == 365
    weekday = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=10", json=_body()).json()
    assert "calendar_day_horizon" not in _codes(weekday)


@pytest.mark.parametrize("endpoint", ["/portfolio/analyze", "/portfolio/var", "/portfolio/stress"])
def test_every_endpoint_says_a_seven_day_series_counts_calendar_days(stub_yfinance, endpoint):
    """`calendar_day_horizon` covers the `1d` / `10d` figures as the README states: /var
    and /analyze warn on a 7-day series, as Monte Carlo does."""
    body = {"weights": {"COIN7": 1.0}, "start": "2020-01-01", "end": "2024-03-01"}
    out = client.post(endpoint, json=body).json()
    message = [w["message"] for w in out["warnings"] if w["code"] == "calendar_day_horizon"]
    assert len(message) == 1 and "the 1d and 10d figures" in message[0]
    weekday = client.post(endpoint, json=_body()).json()
    assert "calendar_day_horizon" not in _codes(weekday)


class TestAnnualisationOverHttp:
    """Sunday-Thursday plus Monday-Friday tickers leave Monday-Thursday, 208 sessions a
    year. Read as 252 they would show volatility 20.9% for 19.1% and CAGR 21.7% for
    17.6%."""

    # Fixed dates, passed as `end` too: the measured rate depends on the window.
    DAYS = pd.date_range(end="2026-11-02", periods=365 * 6, freq="D")
    WINDOW = {"start": str(DAYS[0].date()), "end": "2026-11-03"}

    @pytest.fixture
    def market(self, monkeypatch):
        import sys
        import types

        latent = pd.Series(np.random.default_rng(11).normal(0.0004, 0.01, len(self.DAYS)), index=self.DAYS)
        level = (1 + latent).cumprod() * 100
        universe = {
            "MONFRI": level[level.index.dayofweek < 5],
            "SUNTHU": level[level.index.dayofweek.isin((6, 0, 1, 2, 3))],
        }

        def download(sym, start=None, end=None, **_k):
            s = universe[sym]
            return pd.DataFrame({"Close": s[(s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end))]})

        fake = types.ModuleType("yfinance")
        fake.download = download
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        return latent

    def test_common_sessions_use_the_measured_rate_and_say_so(self, market):
        body = {"weights": {"SUNTHU": 0.5, "MONFRI": 0.5}, **self.WINDOW}
        out = client.post("/portfolio/analyze", json=body).json()
        assert out["risk"]["periods_per_year"] == 208
        assert out["performance"]["cagr"] == pytest.approx(float((1 + market).prod() ** (1 / 6) - 1), abs=0.005)
        assert out["risk"]["annual_vol"] == pytest.approx(float(market.std() * np.sqrt(365)), abs=0.005)
        message = [w["message"] for w in out["warnings"] if w["code"] == "non_standard_calendar"]
        assert len(message) == 1 and "about 208 observations a year" in message[0]
        var = client.post("/portfolio/var", json=body).json()
        assert "non_standard_calendar" in _codes(var)

    def test_a_single_exchange_calendar_is_252_without_that_warning(self, market):
        body = {"weights": {"MONFRI": 1}, **self.WINDOW}
        out = client.post("/portfolio/analyze", json=body).json()
        assert out["risk"]["periods_per_year"] == 252
        assert not {"non_standard_calendar", "sparse_calendar", "non_daily_horizon"} & set(_codes(out))


class TestMixedExchangeCalendar:
    """AAPL with 7203.T leaves about 235 common sessions a year, which read as measured,
    not as if on 252, with a warning."""

    DAYS = pd.bdate_range("2016-01-04", "2025-12-31")

    @pytest.fixture
    def market(self, monkeypatch):
        import sys
        import types

        rng = np.random.default_rng(21)
        level = pd.Series((1 + rng.normal(0.0006, 0.0105, len(self.DAYS))).cumprod() * 100, index=self.DAYS)
        universe = {
            "AAPL": level[rng.random(len(level)) > 9 / 261],
            "7203.T": level[rng.random(len(level)) > 17 / 261],
        }

        def download(sym, start=None, end=None, **_k):
            s = universe[sym]
            return pd.DataFrame({"Close": s[(s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end))]})

        fake = types.ModuleType("yfinance")
        fake.download = download
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        return universe

    def test_the_mixed_example_says_it_is_not_on_252(self, market):
        body = {"weights": {"AAPL": 0.5, "7203.T": 0.5}, "start": "2016-01-04", "end": "2026-01-01"}
        out = client.post("/portfolio/analyze", json=body).json()
        assert out["risk"]["periods_per_year"] == pytest.approx(235, abs=2)
        assert "non_standard_calendar" in _codes(out)
        common = pd.concat(market.values(), axis=1, join="inner").iloc[:, 0]
        years = (common.index[-1] - common.index[0]) / pd.Timedelta(days=365.25)
        assert out["performance"]["cagr"] == pytest.approx((common.iloc[-1] / common.iloc[0]) ** (1 / years) - 1, abs=1e-3)

    def test_one_exchange_of_that_pair_is_252_with_no_warning(self, market):
        body = {"weights": {"AAPL": 1}, "start": "2016-01-04", "end": "2026-01-01"}
        out = client.post("/portfolio/analyze", json=body).json()
        assert out["risk"]["periods_per_year"] == 252 and "non_standard_calendar" not in _codes(out)


class TestNonSynchronousCloses:
    """Tokyo closes before New York opens, so New York's move reaches Tokyo the next day.
    Same-date pairing reads a true beta of 1.1 as 0.63 and the GFC at -41% for about
    -60%, so the response warns."""

    ENDPOINTS = ("/portfolio/analyze", "/portfolio/var", "/portfolio/stress", "/portfolio/montecarlo")
    _rng = np.random.default_rng(42)
    _market = _rng.normal(0.0005, 0.011, 1501)
    _noise = _rng.normal(0, 0.008, 1501)
    SPY = _market[1:].tolist()
    LAGGED = (1.25 * _market[:-1] + _noise[1:]).tolist()    # one session behind
    SYNCED = (1.25 * _market[1:] + _noise[1:]).tolist()     # the same exposure, same day

    def _body(self, tokyo):
        return {"weights": {"SPY": 0.6, "7203.T": 0.4}, "returns": {"SPY": self.SPY, "7203.T": tokyo}}

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_every_endpoint_says_so_on_a_mix(self, endpoint):
        out = client.post(endpoint, json=self._body(self.LAGGED)).json()
        message = [w["message"] for w in out["warnings"] if w["code"] == "non_synchronous_closes"]
        assert len(message) == 1
        assert "Asia-Pacific" in message[0] and "the Americas" in message[0] and "heuristic" in message[0]

    def test_the_documented_bias_is_what_the_engine_does(self):
        lagged = client.post("/portfolio/stress", json=self._body(self.LAGGED)).json()
        synced = client.post("/portfolio/stress", json=self._body(self.SYNCED)).json()
        gfc = lambda out: next(s for s in out["scenarios"] if "Financial" in s["name"])["portfolio_drawdown"]
        assert lagged["beta_used"] == pytest.approx(0.63, abs=0.005)
        assert synced["beta_used"] == pytest.approx(1.10, abs=0.01)
        assert gfc(lagged) == pytest.approx(-0.41, abs=0.005) and gfc(synced) == pytest.approx(-0.60, abs=0.005)
        vol = lambda tokyo: client.post("/portfolio/analyze", json=self._body(tokyo)).json()
        read, true = vol(self.LAGGED), vol(self.SYNCED)
        assert read["correlation_matrix"]["SPY"]["7203.T"] == pytest.approx(0.05, abs=0.01)
        assert read["risk"]["annual_vol"] == pytest.approx(0.151, abs=0.001)
        assert true["risk"]["annual_vol"] == pytest.approx(0.198, abs=0.001)
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        for figure in ("reads 0.63", "-41%", "-60%", "15.1%", "19.8%", "correlation 0.05"):
            assert figure in readme, figure

    def test_a_tokyo_ticker_against_a_new_york_market_series_says_so(self):
        series = self.LAGGED
        body = {"weights": {"7203.T": 1}, "returns": {"7203.T": series}}
        assert "non_synchronous_closes" in _codes(client.post("/portfolio/stress", json={**body, "market_returns": self.SPY}).json())
        for quiet in ({"beta": 1.1}, {}):      # nothing is measured against New York
            assert "non_synchronous_closes" not in _codes(client.post("/portfolio/stress", json={**body, **quiet}).json())
        assert "non_synchronous_closes" not in _codes(client.post("/portfolio/analyze", json=body).json())

    @pytest.mark.parametrize("weights", [
        {"AAPL": 0.5, "MSFT": 0.5},
        {"SPY": 0.5, "BRK.B": 0.5},             # a share class, not an exchange
        {"SPY": 0.5, "RY.TO": 0.5},             # Toronto closes with New York
        {"VOD.L": 0.5, "SAP.DE": 0.5},          # one close
        {"7203.T": 0.5, "0700.HK": 0.5},        # one region: a stated limit
        {"SPY": 1, "7203.T": 0},                # a zero weight holds nothing
    ], ids=["us", "share-class", "toronto", "europe", "asia", "zero-weight"])
    def test_one_region_is_not_flagged(self, weights):
        series = {t: self.SPY for t in weights}
        for endpoint in ("/portfolio/analyze", "/portfolio/var"):
            assert "non_synchronous_closes" not in _codes(client.post(endpoint, json={"weights": weights, "returns": series}).json())

    @pytest.mark.parametrize("weights", [
        {"AAPL": 0.5, "7203.T": 0.5}, {"AAPL": 0.5, "VOD.L": 0.5}, {"VOD.L": 0.5, "005930.KS": 0.5}, {"AAPL": 0.5, "sap.de": 0.5},
    ], ids=["new-york-tokyo", "new-york-london", "london-seoul", "lower-case"])
    def test_two_regions_are_flagged(self, weights):
        series = {t: self.SPY for t in weights}
        assert "non_synchronous_closes" in _codes(client.post("/portfolio/var", json={"weights": weights, "returns": series}).json())

    def test_live_prices_are_flagged_too(self, monkeypatch):
        import sys
        import types

        days = pd.bdate_range("2023-01-02", "2024-06-28")
        level = pd.Series((1 + np.random.default_rng(1).normal(0.0004, 0.01, len(days))).cumprod() * 100, index=days)

        def download(_sym, start=None, end=None, **_k):
            return pd.DataFrame({"Close": level[(level.index >= pd.Timestamp(start)) & (level.index < pd.Timestamp(end))]})

        fake = types.ModuleType("yfinance")
        fake.download = download
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        body = {"weights": {"AAPL": 0.5, "7203.T": 0.5}, "start": "2023-01-02", "end": "2024-06-29"}
        assert "non_synchronous_closes" in _codes(client.post("/portfolio/stress", json=body).json())


@pytest.mark.parametrize("shape", ["flat", "field-ticker"])
def test_a_one_row_spy_download_is_an_assumed_beta_not_a_500(monkeypatch, shape):
    """A weekend-spanning window: SPY has one session. The single row is an assumed beta,
    not an error raised outside the expected failures."""
    import sys
    import types

    def download(sym, start=None, end=None, **_k):
        if sym != "SPY":
            days = pd.date_range("2024-01-05", "2024-01-07")
            return pd.DataFrame({"Close": [44000.0, 44100.0, 43900.0]}, index=days)
        frame = pd.DataFrame({"Close": [470.0]}, index=pd.DatetimeIndex(["2024-01-05"]))
        if shape == "field-ticker":
            frame.columns = pd.MultiIndex.from_tuples([("Close", "SPY")])
        return frame

    fake = types.ModuleType("yfinance")
    fake.download = download
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    body = {"weights": {"BTC-USD": 1}, "start": "2024-01-05", "end": "2024-01-08"}
    r = lenient.post("/portfolio/stress", json=body)
    assert r.status_code == 200, r.text[:200]
    assert r.json()["beta_source"] == "assumed" and "assumed_beta" in _codes(r.json())


def test_a_standard_calendar_with_many_holes_is_flagged_sparse():
    """40% of days gone: still 252, one session per observation, which is right
    for dates excluded as missing and wrong if returns span the holes. Say so."""
    from riskcanvas.api.app import _calendar_warnings

    idx = pd.bdate_range("2015-01-01", "2024-12-31")
    holes = idx[np.random.default_rng(9).random(len(idx)) > 0.4]
    assert [w["code"] for w in _calendar_warnings(_Dated(holes))] == ["sparse_calendar"]
    assert _calendar_warnings(_Dated(idx)) == []
    # Under a year, one long closure is not "sparse".
    closure = pd.bdate_range("2024-01-02", "2024-03-29")
    closure = closure[~((closure >= "2024-02-05") & (closure <= "2024-02-23"))]
    assert _calendar_warnings(_Dated(closure)) == []


# --- Long series, capped VaR, erased trends ---------------------------------


def test_long_returns_without_start_are_dated_back_from_today(today):
    """A series with more observations than there are business days from the 2020-01-01
    default to today is dated back from today: from the default it would end in the
    future."""
    from riskcanvas.api.app import MAX_RETURNS_LEN

    n = len(pd.bdate_range(_default_start(), today)) + 1    # the last lands after today
    assert n <= MAX_RETURNS_LEN, "no series this long is accepted: the default start needs moving"
    body = {"weights": {"A": 1}, "returns": {"A": np.random.default_rng(71).normal(0, 0.01, n).tolist()}}
    out = client.post("/portfolio/var", json=body)
    assert out.status_code == 200, out.text[:200]
    j = out.json()
    assert j["period"].endswith(pd.bdate_range(end=today, periods=1)[0].date().isoformat())
    assert "start_derived_from_today" in _codes(j)
    explicit = client.post("/portfolio/var", json={**body, "start": _default_start().isoformat()})
    assert explicit.status_code == 422 and "after today" in explicit.text   # an explicit start is not moved
    assert "start_derived_from_today" not in _codes(client.post("/portfolio/var", json=_body()).json())


def test_capped_10_day_var_is_flagged():
    body = {"weights": {"A": 1}, "returns": {"A": [10.0, -0.9999] * 50}}
    for endpoint in ("/portfolio/var", "/portfolio/analyze"):
        assert "var_10d_capped" in _codes(client.post(endpoint, json=body).json()), endpoint
    assert "var_10d_capped" not in _codes(client.post("/portfolio/var", json=_body()).json())


def test_zero_drift_erasing_an_extreme_trend_is_flagged():
    """A -100%-every-day series simulated VaR99 0.0 under the default zero
    drift, silently."""
    body = {"weights": {"A": 1}, "returns": {"A": [-1.0] * 30}}
    out = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=5&method=gbm", json=body).json()
    assert out["var_99"] == 0.0 and "drift_removes_extreme_trend" in _codes(out)
    normal = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=5", json=_body()).json()
    assert "drift_removes_extreme_trend" not in _codes(normal)


class _Dated:
    """The part of a Portfolio the calendar warning reads."""

    def __init__(self, idx):
        self.returns = pd.DataFrame({"A": 0.0}, index=idx)


def test_calendar_warning_names_the_real_calendar():
    """A Sunday-Thursday market is not told it "trades every calendar day"."""
    from riskcanvas.api.app import _calendar_warnings

    days = pd.date_range("2022-01-02", periods=900, freq="D")
    assert _calendar_warnings(_Dated(days[~days.dayofweek.isin((4, 5))]), 252) == []
    assert _calendar_warnings(_Dated(pd.bdate_range("2022-01-03", periods=900)), 252) == []
    assert [w["code"] for w in _calendar_warnings(_Dated(days), 365)] == ["calendar_day_horizon"]
    monthly = _calendar_warnings(_Dated(pd.date_range("2015-01-31", periods=120, freq="ME")), 24)
    assert [w["code"] for w in monthly] == ["non_daily_horizon"] and "2.00 years" in monthly[0]["message"]


class TestMarketClosuresOverHttp:
    BODY = {"weights": {"HOLE": 1.0}, "start": "2020-01-01", "end": "2024-03-01"}

    def test_a_long_gap_is_a_422_that_says_how_to_accept_it_then_is_flagged(self, stub_yfinance):
        refused = client.post("/portfolio/var", json=self.BODY)
        assert refused.status_code == 422 and "`max_gap_days=" in refused.json()["detail"]
        for endpoint in ("/portfolio/analyze", "/portfolio/var", "/portfolio/stress",
                         "/portfolio/montecarlo?n_paths=1000&horizon_days=10"):
            out = client.post(endpoint, json={**self.BODY, "max_gap_days": 130})
            assert out.status_code == 200, f"{endpoint}: {out.text[:200]}"
            assert "long_gap_spanned" in _codes(out.json()), endpoint

    def test_no_flag_without_a_long_gap(self, stub_yfinance):
        out = client.post("/portfolio/var", json={"weights": {"SPY": 1}, "start": "2020-01-01",
                                                  "end": "2024-03-01", "max_gap_days": 130}).json()
        assert "long_gap_spanned" not in _codes(out)

    @pytest.mark.parametrize("bad", [0, -1, 367, 1.5, "12", True])
    def test_max_gap_days_is_validated(self, bad):
        assert client.post("/portfolio/var", json={**self.BODY, "max_gap_days": bad}).status_code == 422


class TestLevelsAreNotReturns:
    """A column of `1.01, 0.99, ...` would read VaR95 -98.4% and CAGR 6.75e75, and prices
    near $5 VaR95 -4.15 and CAGR 1.7e191: such a series is not returns."""

    _R = np.random.default_rng(1).normal(0.0005, 0.01, 300)
    GROSS = (1 + _R).tolist()
    PRICES = (5 * np.cumprod(1 + _R)).tolist()
    ENDPOINTS = ("/portfolio/analyze", "/portfolio/var", "/portfolio/stress",
                 "/portfolio/montecarlo?n_paths=1000&horizon_days=10")

    def _body(self, series, **kw):
        return {"weights": {"A": 1}, "returns": {"A": series}, "start": "2020-01-01", **kw}

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    @pytest.mark.parametrize("series", [GROSS, PRICES], ids=["gross", "prices"])
    def test_refused_by_every_endpoint_with_the_way_out(self, endpoint, series):
        r = client.post(endpoint, json=self._body(series))
        assert r.status_code == 422, r.text[:200]
        detail = r.json()["detail"]
        assert "looks like levels" in detail and "'A'" in detail and "allow_large_returns" in detail

    def test_one_stray_value_does_not_hide_a_level_column(self):
        series = list(self.GROSS)
        series[150] = -0.5
        assert client.post("/portfolio/analyze", json=self._body(series)).status_code == 422

    @pytest.mark.parametrize("series", [
        np.random.default_rng(2).normal(0.002, 0.08, 300).tolist(),         # 127% annual vol
        np.abs(np.random.default_rng(3).normal(0.001, 0.005, 300)).tolist(),  # always up, quietly
        [0.0001] * 300,                                                     # cash
    ], ids=["high-vol", "always-up", "cash"])
    def test_real_return_series_pass(self, series):
        assert client.post("/portfolio/analyze", json=self._body(series)).status_code == 200

    def test_a_market_series_of_prices_is_refused_too(self):
        body = {**self._body(np.random.default_rng(2).normal(0, 0.01, 300).tolist()), "market_returns": self.PRICES}
        r = client.post("/portfolio/stress", json=body)
        assert r.status_code == 422 and "market_returns" in r.text
        assert client.post("/portfolio/stress", json={**body, "allow_large_returns": True}).status_code == 200

    @pytest.mark.parametrize("series", [
        GROSS,
        PRICES,
        (0.05 * np.cumprod(1 + _R)).tolist(),     # prices under 0.10: the loader cannot see these
        [1.01, 0.99, 1.02],                       # too short to refuse
        [0.01] * 300,                             # a steady +1% a day
    ], ids=["gross", "prices", "penny-prices", "short-gross", "steady"])
    def test_no_var_outside_0_1_or_absurd_cagr_ships_with_empty_warnings(self, series):
        body = self._body(series, allow_large_returns=True)
        out = client.post("/portfolio/analyze", json=body)
        assert out.status_code == 200, out.text[:200]
        j = out.json()
        cagr, var95 = j["performance"]["cagr"], j["risk"]["var_95_1d"]
        assert cagr is None or cagr > 10 or not 0 <= var95 <= 1, "the fixture is not implausible"
        assert "implausible_statistics" in _codes(j)
        assert "implausible_statistics" in _codes(client.post("/portfolio/var", json=body).json())

    def test_an_ordinary_series_is_not_flagged(self):
        series = np.random.default_rng(4).normal(0.0004, 0.012, 500).tolist()
        for endpoint in ("/portfolio/analyze", "/portfolio/var"):
            assert "implausible_statistics" not in _codes(client.post(endpoint, json=self._body(series)).json())


class TestPeriodsPerYear:
    """`periods_per_year` says what a series is: 120 monthly returns read as daily show
    volatility 55% for a true 12% and CAGR 136% for 4.2%."""

    ENDPOINTS = ("/portfolio/analyze", "/portfolio/var", "/portfolio/stress", "/portfolio/montecarlo")
    _z = np.random.default_rng(5).normal(0, 1, 120)
    MONTHLY = (0.004 + (_z - _z.mean()) / _z.std(ddof=1) * 0.12 / np.sqrt(12)).tolist()

    def _body(self, series=None, **kw):
        return {"weights": {"A": 1}, "returns": {"A": self.MONTHLY if series is None else series}, **kw}

    def test_unstated_monthly_is_flagged(self):
        out = client.post("/portfolio/analyze", json=self._body()).json()
        assert out["risk"]["annual_vol"] == pytest.approx(0.12 * np.sqrt(21), abs=1e-4)
        assert out["performance"]["cagr"] == pytest.approx(1.3588, abs=1e-3)
        assert "implausible_statistics" in _codes(out)
        readme = " ".join((Path(__file__).resolve().parents[1] / "README.md").read_text().split())
        assert "read 55% (× √21), and a seeded CAGR of 4.2% read +136%" in readme

    def test_stated_monthly_reads_as_monthly(self):
        out = client.post("/portfolio/analyze", json=self._body(periods_per_year=12, start="2010-01-31")).json()
        s = pd.Series(self.MONTHLY)
        assert out["risk"]["periods_per_year"] == 12
        assert out["risk"]["annual_vol"] == pytest.approx(0.12, abs=1e-3)
        assert out["performance"]["cagr"] == pytest.approx(float((1 + s).prod() ** (1 / 10) - 1), abs=1e-5)
        assert out["performance"]["cagr"] == pytest.approx(0.0417, abs=1e-3)
        assert out["n_observations"] == 120
        assert out["period"] == "2010-01-31 → 2019-12-31"    # 119 whole months on
        first, last = out["period"].split(" → ")
        assert (pd.Timestamp(last) - pd.Timestamp(first)).days == pytest.approx(365.25 * 119 / 12, abs=3)
        assert 0 < out["performance"]["positive_months"] < 1
        assert "implausible_statistics" not in _codes(out)
        assert "non_daily_horizon" in _codes(out) and "long_gap_spanned" not in _codes(out)
        assert "non_standard_calendar" not in _codes(out) and "sparse_calendar" not in _codes(out)

    @pytest.mark.parametrize("rate, start, dates", [
        (12, "2010-01-31", ["2010-01-31", "2010-02-28", "2010-03-31", "2010-04-30", "2010-05-31"]),
        (4, "2011-11-30", ["2011-11-30", "2012-02-29", "2012-05-30", "2012-08-30", "2012-11-30"]),
        (1, "2012-02-29", ["2012-02-29", "2013-02-28", "2014-02-28", "2015-02-28", "2016-02-29"]),
    ], ids=["monthly", "quarterly", "annual"])
    def test_whole_months_count_from_start(self, rate, start, dates):
        """Stepping from each previous date stayed on the 28th after the first short month:
        monthly from Jan 31 read Mar 28, Apr 28, so a series ended up to three days early."""
        from riskcanvas.api.app import _observation_dates

        assert [d.isoformat() for d in _observation_dates(date.fromisoformat(start), 5, rate).date] == dates
        body = self._body([0.01, -0.02, 0.03, -0.01, 0.02], periods_per_year=rate, start=start)
        out = client.post("/portfolio/analyze", json=body)
        assert out.status_code == 200, out.text[:200]
        assert out.json()["period"] == f"{dates[0]} → {dates[-1]}"

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_every_endpoint_takes_it(self, endpoint):
        r = client.post(endpoint, json=self._body(periods_per_year=12))
        assert r.status_code == 200, r.text[:200]
        assert "non_daily_horizon" in _codes(r.json())

    def test_the_default_is_unchanged(self):
        series = np.random.default_rng(1).normal(0.0004, 0.01, 300).tolist()
        plain = client.post("/portfolio/analyze", json=self._body(series, start="2020-01-01")).json()
        stated = client.post("/portfolio/analyze", json=self._body(series, start="2020-01-01", periods_per_year=252)).json()
        assert plain["risk"] == stated["risk"] and plain["performance"] == stated["performance"]
        assert plain["period"] == stated["period"]
        # The only difference: stating 252 confirms what the default assumed.
        assert plain["warnings"][:-1] == stated["warnings"]
        assert _codes(plain)[-1:] == ["assumed_periods_per_year"]

    @pytest.mark.parametrize("rate", [1, 4, 12, 26, 52, 100, 200, 261, 365])
    def test_every_rate_annualises_and_dates_in_order(self, rate):
        series = np.random.default_rng(rate).normal(0.001, 0.01, 40).tolist()
        out = client.post("/portfolio/analyze", json=self._body(series, periods_per_year=rate, start="1980-01-01"))
        assert out.status_code == 200, out.text[:200]
        j = out.json()
        assert j["risk"]["periods_per_year"] == rate
        assert j["risk"]["annual_vol"] == pytest.approx(pd.Series(series).std() * np.sqrt(rate), abs=1e-5)

    @staticmethod
    def _months_past(today):
        """Monthly returns that, dated from the 2020-01-01 default, end after ``today``:
        the last falls on the first of the following month."""
        start = _default_start()
        n = (today.year - start.year) * 12 + today.month - start.month + 2
        z = np.random.default_rng(6).normal(0, 1, n)
        return (0.004 + z * 0.12 / np.sqrt(12)).tolist()

    def test_no_start_dates_it_back_from_today_at_that_rate(self, today):
        out = client.post("/portfolio/analyze", json=self._body(self._months_past(today), periods_per_year=12))
        assert out.status_code == 200, out.text[:200]
        j = out.json()
        assert "start_derived_from_today" in _codes(j)
        message = next(w["message"] for w in j["warnings"] if w["code"] == "start_derived_from_today")
        assert "12 dates a year" in message
        last = pd.Timestamp(j["period"].split(" → ")[1]).date()
        assert today - timedelta(days=31) < last <= today

    def test_a_late_start_names_the_latest_one_that_fits(self, today):
        series = self._months_past(today)
        r = client.post("/portfolio/analyze", json=self._body(series, periods_per_year=12,
                                                               start=_default_start().isoformat()))
        assert r.status_code == 422 and "12 dates a year" in r.text
        latest = re.search(r"on or before (\d{4}-\d{2}-\d{2})", r.text).group(1)
        fits = client.post("/portfolio/analyze", json=self._body(series, periods_per_year=12, start=latest))
        assert fits.status_code == 200, fits.text[:200]

    def test_a_weekend_start_is_not_moved_when_not_on_business_days(self):
        out = client.post("/portfolio/analyze", json=self._body(periods_per_year=12, start="2010-01-02")).json()
        assert "start_moved_to_business_day" not in _codes(out)
        assert out["period"].startswith("2010-01-02")

    @pytest.mark.parametrize("value", [0, -1, 366, 12.5, "12", True])
    def test_out_of_range_or_wrong_type_is_a_422(self, value):
        assert client.post("/portfolio/analyze", json=self._body(periods_per_year=value)).status_code == 422

    def test_too_long_a_span_is_a_422(self):
        r = lenient.post("/portfolio/analyze", json=self._body([0.001, -0.001] * 500, periods_per_year=1))
        assert r.status_code == 422 and "years" in r.text

    def test_it_describes_returns_only(self):
        r = client.post("/portfolio/analyze", json={"weights": {"A": 1}, "periods_per_year": 12})
        assert r.status_code == 422 and "`periods_per_year` describes `returns`" in r.text

    def test_risk_free_drift_uses_the_stated_rate(self):
        flat = [0.0] * 100
        body = {"weights": {"A": 1}, "returns": {"A": flat}, "periods_per_year": 12}
        out = client.post("/portfolio/montecarlo?n_paths=1000&horizon_days=12&drift=risk_free&method=gbm", json=body).json()
        assert out["expected_terminal"] == pytest.approx(1.02, abs=2e-3)   # (1 + 2% / 12) ** 12

    def test_weekly_compounding_reduces_the_close_bias_as_the_warning_says(self):
        rng = np.random.default_rng(42)
        market, noise = rng.normal(0.0005, 0.011, 1501), rng.normal(0, 0.008, 1501)
        spy, tokyo = pd.Series(market[1:]), pd.Series(1.25 * market[:-1] + noise[1:])
        weekly = lambda s: ((1 + s).groupby(s.index // 5).prod() - 1).tolist()
        daily = {"weights": {"SPY": 0.6, "7203.T": 0.4}, "returns": {"SPY": spy.tolist(), "7203.T": tokyo.tolist()}}
        compounded = {**daily, "returns": {"SPY": weekly(spy), "7203.T": weekly(tokyo)}, "periods_per_year": 52}
        low = client.post("/portfolio/stress", json=daily).json()["beta_used"]
        better = client.post("/portfolio/stress", json=compounded).json()["beta_used"]
        assert low < 0.65 and better == pytest.approx(1.01, abs=0.005)   # the true beta is 1.1


class TestAssumedRateIsStated:
    """`returns` carry no dates, so with no `periods_per_year` they are read at 252 a year.
    A 7-day series' annual volatility then read sqrt(252 / 365) of its value, 17% low,
    with nothing in `warnings`."""

    SERIES = np.random.default_rng(31).normal(0.001, 0.03, 730).tolist()

    def _body(self, **kw):
        return {"weights": {"BTC-USD": 1}, "returns": {"BTC-USD": self.SERIES}, "start": "2022-01-03", **kw}

    def test_analyze_says_its_annual_figures_assumed_252(self):
        undated = client.post("/portfolio/analyze", json=self._body()).json()
        assert undated["risk"]["periods_per_year"] == 252
        message = next(w["message"] for w in undated["warnings"] if w["code"] == "assumed_periods_per_year")
        stated = client.post("/portfolio/analyze", json=self._body(periods_per_year=365)).json()
        ratio = undated["risk"]["annual_vol"] / stated["risk"]["annual_vol"]
        assert ratio == pytest.approx(np.sqrt(252 / 365), rel=1e-3)
        assert f"annual volatility {1 - ratio:.0%} low" in message and "`periods_per_year`" in message
        assert "assumed_periods_per_year" not in _codes(stated)
        confirmed = client.post("/portfolio/analyze", json=self._body(periods_per_year=252)).json()
        assert "assumed_periods_per_year" not in _codes(confirmed)
        assert confirmed["risk"] == undated["risk"]
        readme = " ".join((Path(__file__).resolve().parents[1] / "README.md").read_text().split())
        assert f"of its annual volatility, {1 - ratio:.0%} low" in readme

    def test_only_endpoints_whose_figures_use_the_rate_say_so(self):
        """/var and /stress report nothing annual; Monte Carlo uses the rate only to
        spread the risk-free drift."""
        body = self._body()
        assert "assumed_periods_per_year" not in _codes(client.post("/portfolio/var", json=body).json())
        stress = client.post("/portfolio/stress", json={**body, "beta": 1.0}).json()
        assert "assumed_periods_per_year" not in _codes(stress)
        mc = "/portfolio/montecarlo?n_paths=1000&horizon_days=5"
        for drift in ("zero", "historical"):
            assert "assumed_periods_per_year" not in _codes(client.post(f"{mc}&drift={drift}", json=body).json())
        rf = client.post(f"{mc}&drift=risk_free", json=body).json()
        message = next(w["message"] for w in rf["warnings"] if w["code"] == "assumed_periods_per_year")
        assert "drift=risk_free spreads its 2% a year" in message
        stated = client.post(f"{mc}&drift=risk_free", json=self._body(periods_per_year=365)).json()
        assert "assumed_periods_per_year" not in _codes(stated)

    def test_dated_prices_are_measured_not_assumed(self, stub_yfinance):
        body = {"weights": {"COIN7": 1.0}, "start": "2020-01-01", "end": "2024-03-01"}
        out = client.post("/portfolio/analyze", json=body).json()
        assert out["risk"]["periods_per_year"] == 365 and "assumed_periods_per_year" not in _codes(out)


class TestIgnoredFieldsAreNamed:
    """A key the API does not define was dropped without a word: `periods_per_yeer: 12`
    read 120 monthly returns of 12% volatility as daily, 55%, in a 200 with no warning.
    Such a key is still ignored, since a client may send more than this API reads, but
    every endpoint now names it."""

    BODY = {"weights": {"A": 1}, "returns": {"A": TestPeriodsPerYear.MONTHLY}, "start": "2010-01-29"}
    ENDPOINTS = ("/portfolio/analyze", "/portfolio/var", "/portfolio/stress",
                 "/portfolio/montecarlo?n_paths=1000&horizon_days=5")

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_a_misspelt_field_is_named_and_changes_nothing(self, endpoint):
        plain = client.post(endpoint, json=self.BODY).json()
        typo = client.post(endpoint, json={**self.BODY, "periods_per_yeer": 12})
        assert typo.status_code == 200, typo.text[:200]
        out = typo.json()
        assert out["warnings"][0]["code"] == "ignored_fields"
        assert "body field(s) `periods_per_yeer`" in out["warnings"][0]["message"]
        assert "periods_per_year" in out["warnings"][0]["message"]   # names what the body takes
        assert {**out, "warnings": out["warnings"][1:]} == plain

    def test_an_unknown_query_parameter_is_named_too(self):
        out = client.post("/portfolio/montecarlo?n_path=1000&horizon_days=5", json=_body()).json()
        assert out["warnings"][0]["code"] == "ignored_fields"
        message = out["warnings"][0]["message"]
        assert "query parameter(s) `n_path`" in message and "n_paths" in message
        assert out["n_paths"] == 10_000   # the default ran: the misspelling changed nothing

    def test_no_documented_name_is_flagged(self):
        body = {**self.BODY, "end": "2020-01-01", "name": "P", "market_returns": TestPeriodsPerYear.MONTHLY,
                "beta": 1.0, "max_gap_days": 12, "periods_per_year": 12, "allow_large_returns": False}
        for endpoint in ("/portfolio/analyze", "/portfolio/var?portfolio_value=5", "/portfolio/stress",
                         "/portfolio/montecarlo?n_paths=1000&horizon_days=5&method=gbm&drift=zero&seed=1"):
            out = client.post(endpoint, json=body)
            assert out.status_code == 200, f"{endpoint}: {out.text[:200]}"
            assert "ignored_fields" not in _codes(out.json()), endpoint

    def test_the_message_stays_short_however_many_keys_arrive(self):
        extra = {f"{'x' * 100}{i:02d}": i for i in range(50)}
        message = client.post("/portfolio/var", json={**_body(), **extra}).json()["warnings"][0]["message"]
        assert "and 45 more" in message and "x" * 41 not in message and len(message) < 1000
        from riskcanvas.api.app import MAX_IGNORED_NAME_LEN, MAX_IGNORED_NAMES

        security = " ".join((Path(__file__).resolve().parents[1] / "SECURITY.md").read_text().split())
        assert f"at most {MAX_IGNORED_NAMES} names, each cut to {MAX_IGNORED_NAME_LEN} characters" in security

    def test_a_private_attribute_name_sets_nothing(self):
        """`_start_derived` is the model's private state, not a field: sent as a key it is
        ignored like any other, and no start is reported as derived."""
        out = client.post("/portfolio/var", json={**_body(), "_start_derived": True}).json()
        assert _codes(out) == ["ignored_fields"] and "`_start_derived`" in out["warnings"][0]["message"]


class TestImplausibleVolatility:
    """A short bond fund in percent passes the percent check, whose median is under
    0.10: annual_vol 194%, VaR95 19.6%, empty `warnings`."""

    @staticmethod
    def _scaled(annual_vol, n=500):
        z = np.random.default_rng(1).normal(0, 1, n)
        return (z / z.std(ddof=1) * annual_vol / np.sqrt(252)).tolist()

    def test_a_quiet_series_in_percent_is_flagged(self):
        series = (0.01 + np.array(self._scaled(0.1225 * np.sqrt(252)))).tolist()
        out = client.post("/portfolio/analyze", json={"weights": {"SHY": 1}, "returns": {"SHY": series}}).json()
        assert out["risk"]["annual_vol"] == pytest.approx(1.945, abs=1e-3)
        assert out["risk"]["var_95_1d"] == pytest.approx(0.1965, abs=1e-3)
        for endpoint in ("/portfolio/analyze", "/portfolio/var"):
            r = client.post(endpoint, json={"weights": {"SHY": 1}, "returns": {"SHY": series}})
            assert r.status_code == 200, r.text[:200]
            message = next(w["message"] for w in r.json()["warnings"] if w["code"] == "implausible_statistics")
            assert "Annual volatility" in message and "above 100%" in message and "percent" in message

    @pytest.mark.parametrize("endpoint", ["/portfolio/analyze", "/portfolio/var"])
    @pytest.mark.parametrize("vol, flagged", [(0.99, False), (1.01, True)], ids=["below", "above"])
    def test_both_sides_of_the_bound(self, endpoint, vol, flagged):
        series = self._scaled(vol)
        out = client.post(endpoint, json={"weights": {"A": 1}, "returns": {"A": series}}).json()
        assert ("implausible_statistics" in _codes(out)) is flagged

    def test_the_bound_is_documented(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        assert "`IMPLAUSIBLE_ANNUAL_VOL`" in readme and "annual volatility passes 100%" in readme

    def _monthly_like(self, vol, mean):
        z = np.random.default_rng(3).normal(0, 1, 500)
        return (mean + (z - z.mean()) / z.std(ddof=1) * vol / np.sqrt(252)).tolist()

    @pytest.mark.parametrize("vol, mean, flagged", [
        (0.56, 0.004, True),      # monthly returns read as daily: above both lines
        (0.49, 0.004, False),     # under the volatility line
        (0.56, 0.002, False),     # under the CAGR line: a volatile real asset
    ], ids=["both-above", "vol-below", "cagr-below"])
    def test_series_read_as_daily_are_flagged_only_on_both(self, vol, mean, flagged):
        series = self._monthly_like(vol, mean)
        for endpoint in ("/portfolio/analyze", "/portfolio/var"):
            out = client.post(endpoint, json={"weights": {"A": 1}, "returns": {"A": series}}).json()
            assert ("implausible_statistics" in _codes(out)) is flagged, endpoint

    def test_dates_or_a_stated_rate_remove_the_daily_assumption(self):
        from riskcanvas.api.app import _plausibility_warnings

        idx = pd.bdate_range("2020-01-01", periods=500)
        rm = RiskMetrics(pd.Series(self._monthly_like(0.56, 0.004), index=idx))
        assert _plausibility_warnings(rm, (0.95,), 1.7, assumed_daily=True)
        assert not _plausibility_warnings(rm, (0.95,), 1.7, assumed_daily=False)
        series = self._monthly_like(0.56, 0.004)
        out = client.post("/portfolio/analyze", json={
            "weights": {"A": 1}, "returns": {"A": series}, "periods_per_year": 252}).json()
        assert "implausible_statistics" not in _codes(out)


def test_the_stated_largest_body_counts_market_returns():
    from riskcanvas.api.app import MAX_ASSETS, MAX_RETURNS_LEN

    security = (Path(__file__).resolve().parents[1] / "SECURITY.md").read_text()
    assert f"{MAX_ASSETS * MAX_RETURNS_LEN + MAX_RETURNS_LEN:,} return values" in security


class TestBodiesThatAreNotJson:
    """What README "Request rules" and docs/usage.md state."""

    @pytest.mark.parametrize("body", [b'{"weights":', b"hello", b""], ids=["truncated", "text", "empty"])
    def test_invalid_json_is_a_422_with_no_echo(self, body):
        r = client.post("/portfolio/analyze", content=body, headers={"content-type": "application/json"})
        assert r.status_code == 422
        assert "input" not in r.json()["detail"][0]

    @pytest.mark.parametrize("body", [b"\xff\xfe{", b"[" * 200_000], ids=["not-utf8", "nested-too-deep"])
    def test_a_body_that_cannot_be_decoded_is_a_400(self, body):
        r = client.post("/portfolio/analyze", content=body, headers={"content-type": "application/json"})
        assert r.status_code == 400


class TestEntrypoint:
    """`python -m riskcanvas.api.app` bound every interface, with auto-reload on."""

    @staticmethod
    def _run(monkeypatch):
        import runpy
        import sys

        import uvicorn

        calls = []
        monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
        monkeypatch.delitem(sys.modules, "riskcanvas.api.app")
        runpy.run_module("riskcanvas.api.app", run_name="__main__")
        return calls[0]

    def test_it_stays_local_with_no_reload(self, monkeypatch):
        monkeypatch.delenv("RISKCANVAS_HOST", raising=False)
        monkeypatch.delenv("RISKCANVAS_PORT", raising=False)
        args, kwargs = self._run(monkeypatch)
        assert args == ("riskcanvas.api.app:app",)
        assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8004
        assert not kwargs.get("reload")

    def test_the_environment_overrides_it(self, monkeypatch):
        monkeypatch.setenv("RISKCANVAS_HOST", "0.0.0.0")
        monkeypatch.setenv("RISKCANVAS_PORT", "9000")
        _, kwargs = self._run(monkeypatch)
        assert kwargs["host"] == "0.0.0.0" and kwargs["port"] == 9000


class TestTotalLossWithWeightsSummingOverOne:
    """6/30 + 23/30 + 1/30 sums a rounding error above 1, so a day on which every holding
    loses everything must read -100%, not below it: no 500 from /portfolio/var and no
    422 from /portfolio/analyze on valid input."""

    SERIES = [0.01, -0.01, -1.0] + [0.01] * 8
    BODY = {"weights": {"A": 6, "B": 23, "C": 1}, "returns": dict.fromkeys("ABC", SERIES)}

    @pytest.mark.parametrize("endpoint", ["/portfolio/analyze", "/portfolio/var", "/portfolio/stress", "/portfolio/montecarlo"])
    def test_every_endpoint_answers(self, endpoint):
        r = TestClient(app, raise_server_exceptions=False).post(endpoint, json=self.BODY)
        assert r.status_code == 200, r.text[:200]

    def test_analyze_reports_the_total_loss(self):
        out = client.post("/portfolio/analyze", json=self.BODY).json()
        assert out["risk"]["max_drawdown"] == -1.0
        assert out["performance"]["cagr"] == -1.0

    def test_var_is_at_most_the_portfolio(self):
        out = client.post("/portfolio/var?portfolio_value=1", json=self.BODY).json()
        assert 0 < out["var_report"]["99%"]["var_1d"] <= 1.0
        assert out["var_report"]["99%"]["cvar_1d"] == 1.0


class TestChecksFollowTheStatedRate:
    """`periods_per_year` drives the percent and levels checks as well as the annual
    figures: annual returns are not a 422 telling the caller to divide them by 100."""

    ENDPOINTS = ("/portfolio/analyze", "/portfolio/var", "/portfolio/stress", "/portfolio/montecarlo")
    ANNUAL = [-0.03, 0.30, 0.08, 0.10, 0.01, 0.38, 0.23, 0.33, 0.29, 0.21, -0.09, -0.12, -0.22, 0.29, 0.11, 0.05, 0.16,
              0.05, -0.37, 0.26, 0.15, 0.02, 0.16, 0.32, 0.14, 0.01, 0.12, 0.22, -0.04, 0.31, 0.18, 0.29, -0.18, 0.26]
    MONTHLY = np.random.default_rng(8).normal(0.01, 0.18, 120).tolist()

    def _post(self, endpoint, series, **kw):
        return client.post(endpoint, json={"weights": {"A": 1}, "returns": {"A": series}, **kw})

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_34_annual_returns_are_analysed_at_every_endpoint(self, endpoint):
        r = self._post(endpoint, self.ANNUAL, periods_per_year=1)
        assert r.status_code == 200, r.text[:200]

    def test_they_read_as_annual_and_keep_their_size(self):
        out = self._post("/portfolio/analyze", self.ANNUAL, periods_per_year=1).json()
        assert out["risk"]["annual_vol"] == pytest.approx(float(np.std(self.ANNUAL, ddof=1)), abs=1e-3)
        assert out["performance"]["cagr"] == pytest.approx(float(np.prod(1 + np.array(self.ANNUAL)) ** (1 / 34) - 1), abs=1e-5)

    def test_unstated_they_are_a_422_that_names_the_rate(self):
        r = self._post("/portfolio/analyze", self.ANNUAL)
        assert r.status_code == 422 and "periods_per_year" in r.json()["detail"]

    def test_monthly_at_18_percent_and_six_up_years(self):
        assert self._post("/portfolio/analyze", self.MONTHLY, periods_per_year=12).status_code == 200
        assert self._post("/portfolio/analyze", [0.10, 0.18, 0.12, 0.20, 0.15, 0.11], periods_per_year=1).status_code == 200

    def test_a_quiet_monthly_fund_in_percent_stays_a_422(self):
        series = np.random.default_rng(9).normal(0.05, 0.9, 120).clip(-0.99, 3).tolist()
        r = self._post("/portfolio/analyze", series, periods_per_year=12)
        assert r.status_code == 422 and "looks like percent" in r.json()["detail"]

    def test_gross_annual_returns_stay_a_422(self):
        r = self._post("/portfolio/analyze", [1 + x for x in self.ANNUAL], periods_per_year=1)
        assert r.status_code == 422 and "looks like levels" in r.json()["detail"]

    def test_a_percent_file_at_one_a_year_is_flagged_where_the_rule_cannot_fire(self):
        series = [3.2, 5.1, -0.5, 4.2, 2.8, 1.9, -0.2, 3.3, 4.8, 2.2, 1.1, 3.9, -0.7, 2.5, 4.4, 3.1, 1.7, 2.9, -0.4, 3.6, 2.1, 4.0]
        out = self._post("/portfolio/analyze", series, periods_per_year=1).json()
        assert "implausible_statistics" in _codes(out)

    def test_market_returns_follow_the_rate_too(self):
        r = self._post("/portfolio/stress", self.ANNUAL, periods_per_year=1, market_returns=self.ANNUAL)
        assert r.status_code == 200, r.text[:200]
        r = self._post("/portfolio/stress", self.ANNUAL, periods_per_year=1, market_returns=[1 + x for x in self.ANNUAL])
        assert r.status_code == 422 and "'market_returns' looks like levels" in r.text


class TestPercentSeriesOverHttp:
    def _body(self, **kw):
        series = np.clip(np.random.default_rng(3).normal(0.01, 0.25, 300), -0.95, 3).tolist()
        return {"weights": {"AGG": 1}, "returns": {"AGG": series}, **kw}

    def test_a_bond_fund_in_percent_is_a_422(self):
        """As a 200 it would read VaR95 42%, annual volatility 402%."""
        for endpoint in ("/portfolio/analyze", "/portfolio/var", "/portfolio/stress", "/portfolio/montecarlo"):
            r = client.post(endpoint, json=self._body())
            assert r.status_code == 422, f"{endpoint}: {r.text[:200]}"
            assert "percent" in r.json()["detail"] and "'AGG'" in r.json()["detail"]

    def test_the_caller_can_say_it_is_real(self):
        assert client.post("/portfolio/analyze", json=self._body(allow_large_returns=True)).status_code == 200


@pytest.mark.parametrize("endpoint", ["/portfolio/analyze", "/portfolio/var", "/portfolio/stress", "/portfolio/montecarlo"])
def test_empty_returns_is_a_422_that_says_so(endpoint):
    """An empty `returns` mapping is a validation error: the check for future dates must
    not read the first series of an empty mapping and raise StopIteration."""
    quiet = TestClient(app, raise_server_exceptions=False)
    r = quiet.post(endpoint, json={"weights": {"A": 1}, "returns": {}})
    assert r.status_code == 422, r.text[:200]
    assert "`returns` is empty" in r.text


class TestStrictTypes:
    """README: violations return 422, "never a silent rewrite". pydantic's default would
    read `true` as 1.0, "0.5" as 0.5 and 0 as 1970-01-01, and `end` before `start` is
    refused with `returns` too."""

    SERIES = np.random.default_rng(5).normal(0, 0.01, 30).tolist()

    @pytest.mark.parametrize("patch", [
        {"weights": {"A": True}},
        {"weights": {"A": "0.5"}},
        {"returns": {"A": ["0.01"] + SERIES[1:]}},
        {"returns": {"A": [True] + SERIES[1:]}},
        {"market_returns": ["0.01"] + SERIES[1:]},
        {"market_returns": [False] + SERIES[1:]},
        {"beta": "1"},
        {"beta": True},
        {"start": 0},
        {"start": 1700000000},
        {"start": True},
        {"start": 1.5},
        {"start": "20200101"},
        {"start": "2020-01-01T00:00:00"},
        {"end": 0},
        {"start": "2024-01-01", "end": "2020-01-01"},
        {"start": "2024-01-01", "end": "2024-01-01"},
        {"allow_large_returns": "yes"},
        {"allow_large_returns": 1},
        {"weights": {"A": 10 ** 400}},
    ], ids=str)
    @pytest.mark.parametrize("endpoint", ["/portfolio/analyze", "/portfolio/montecarlo?n_paths=1000&horizon_days=2"])
    def test_coercible_junk_is_a_422(self, endpoint, patch):
        body = {"weights": {"A": 1}, "returns": {"A": self.SERIES}, **patch}
        r = lenient.post(endpoint, json=body)
        assert r.status_code == 422, f"{patch}: {r.status_code} {r.text[:200]}"

    @pytest.mark.parametrize("patch", [
        {},
        {"weights": {"A": 60}},
        {"weights": {"A": 0.5}},
        {"returns": {"A": [0, 1, -1] * 10}},
        {"beta": 1},
        {"start": "2024-01-01"},
        {"start": "2024-01-01", "end": "2024-06-01"},
        {"allow_large_returns": True},
    ], ids=str)
    def test_real_json_numbers_and_iso_dates_still_pass(self, patch):
        body = {"weights": {"A": 1}, "returns": {"A": self.SERIES}, **patch}
        r = lenient.post("/portfolio/analyze", json=body)
        assert r.status_code == 200, f"{patch}: {r.status_code} {r.text[:200]}"
