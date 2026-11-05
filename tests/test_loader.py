"""Live-ticker loading (PortfolioLoader.from_tickers) against a STUBBED price source: no
network. The stub honours start / end the way yfinance does (end exclusive) and returns
the same frame shape (a "Close" column).

The loader never back-fills a later-listed ticker's first price over the whole window
(on this stub, 630 fabricated 0% days of 1,042, VaR 24% and volatility 32% too low) and
never drops a typo'd or delisted ticker, rescaling the weights."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from riskcanvas.metrics.risk import RiskMetrics
from riskcanvas.portfolio.loader import PortfolioLoader


def _prices(start, end, seed, vol, freq="B", drop=()):
    idx = pd.date_range(start, end, freq=freq)
    idx = idx[~idx.isin(pd.to_datetime(list(drop)))]
    r = np.random.default_rng(seed).normal(0.0004, vol, len(idx))
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + r)}, index=idx)


# SPY from the first 2020 session; RIVN listed 2021-11-10 (as the real one).
UNIVERSE = {
    "SPY": _prices("2020-01-02", "2024-02-29", 1, 0.012),
    "AGG": _prices("2020-01-02", "2024-02-29", 3, 0.004),
    "RIVN": _prices("2021-11-10", "2024-02-29", 2, 0.045),
    "RIVN2": _prices("2021-11-10", "2024-02-29", 8, 0.04),
    # Delisted mid-window.
    "GONE": _prices("2020-01-02", "2022-06-30", 4, 0.02),
    # Trades every calendar day (a crypto-like asset).
    "COIN7": _prices("2020-01-01", "2024-03-03", 5, 0.03, freq="D"),
    # Misses three sessions in the middle (a halt).
    "HALT": _prices("2020-01-02", "2024-02-29", 6, 0.015, drop=("2021-03-02", "2021-03-03", "2021-03-04")),
    # A four-month hole (2022-03-01 .. 2022-07-01).
    "HOLE": _prices("2020-01-02", "2024-02-29", 7, 0.015,
                    drop=tuple(pd.bdate_range("2022-03-01", "2022-07-01").strftime("%Y-%m-%d"))),
}


def _download(sym, start=None, end=None, **_kwargs):
    if sym == "BOOM":
        raise ConnectionError("simulated network failure")
    df = UNIVERSE.get(sym)
    if df is None:
        return pd.DataFrame()
    lo = pd.Timestamp(start) if start is not None else df.index[0]
    hi = pd.Timestamp(end) if end is not None else df.index[-1] + pd.Timedelta(days=1)
    return df[(df.index >= lo) & (df.index < hi)]


@pytest.fixture
def stub_prices(monkeypatch):
    fake = types.ModuleType("yfinance")
    fake.download = _download
    monkeypatch.setitem(sys.modules, "yfinance", fake)


def _load(weights, start="2020-01-01", end="2024-03-01", **kw):
    return PortfolioLoader().from_tickers(weights, start, end, **kw)


class TestNeverFabricatePrices:
    def test_late_listed_ticker_is_rejected_with_its_first_date(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            _load({"SPY": 0.5, "RIVN": 0.5})
        msg = str(exc.value)
        assert "RIVN" in msg and "2021-11-10" in msg
        assert "set `start` on or after 2021-11-10" in msg

    def test_starting_at_the_listing_date_gives_no_fabricated_zero_days(self, stub_prices):
        p = _load({"SPY": 0.5, "RIVN": 0.5}, start="2021-11-10")
        assert p.n_assets == 2
        assert p.prices.index[0] == pd.Timestamp("2021-11-10")
        assert not (p.returns == 0.0).any().any()
        assert not p.prices.isna().any().any()

    def test_backfilled_window_would_have_understated_var(self, stub_prices):
        """Why it matters, on this stub: back-filling RIVN over 2020-01-02..2021-11-09
        would add hundreds of exact-0% days and pull VaR well below the honest window's."""
        honest = _load({"SPY": 0.5, "RIVN": 0.5}, start="2021-11-10")
        both = pd.DataFrame({s: UNIVERSE[s]["Close"] for s in ("SPY", "RIVN")})
        filled = both.loc["2020-01-02":"2024-02-29"].ffill().bfill().pct_change().dropna()
        assert (filled["RIVN"] == 0.0).sum() > 400
        filled_var = RiskMetrics((filled * 0.5).sum(axis=1)).var_historical(0.95)
        honest_var = RiskMetrics(honest.portfolio_returns).var_historical(0.95)
        assert filled_var < 0.85 * honest_var

    def test_ticker_delisted_before_the_window_ends_is_rejected(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            _load({"SPY": 0.5, "GONE": 0.5})
        msg = str(exc.value)
        assert "GONE" in msg and "2022-06-30" in msg and "set `end` on or before 2022-07-01" in msg

    def test_interior_gap_is_an_inner_join_not_a_fill(self, stub_prices):
        p = _load({"SPY": 0.5, "HALT": 0.5})
        halted = pd.to_datetime(["2021-03-02", "2021-03-03", "2021-03-04"])
        assert not p.prices.index.isin(halted).any()
        # The first return after the gap spans it, for every ticker.
        after = p.prices.index[p.prices.index.get_loc(pd.Timestamp("2021-03-01")) + 1]
        assert after == pd.Timestamp("2021-03-05")
        for s in ("SPY", "HALT"):
            raw = UNIVERSE[s]["Close"]
            assert p.returns.loc[after, s] == pytest.approx(
                raw.loc["2021-03-05"] / raw.loc["2021-03-01"] - 1.0, rel=1e-12
            )
        assert not (p.returns == 0.0).any().any()

    def test_weekend_trading_and_holiday_offsets_are_aligned_not_rejected(self, stub_prices):
        """COIN7 starts on the 2020-01-01 holiday and trades at weekends: a
        calendar offset within MAX_EDGE_GAP_DAYS, absorbed by the join."""
        from riskcanvas.portfolio.loader import MAX_EDGE_GAP_DAYS

        assert MAX_EDGE_GAP_DAYS == 7
        p = _load({"SPY": 0.6, "COIN7": 0.4})
        assert p.n_assets == 2
        assert p.prices.index.equals(UNIVERSE["SPY"].loc["2020-01-02":"2024-02-29"].index)
        assert not (p.returns == 0.0).any().any()


class TestNeverDropATicker:
    def test_unknown_ticker_is_rejected_by_name(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            _load({"SPY": 0.5, "RIVNN": 0.5})
        assert "RIVNN" in str(exc.value) and "never dropped silently" in str(exc.value)

    def test_failed_download_is_rejected_by_name(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            _load({"SPY": 0.5, "BOOM": 0.5})
        assert "BOOM (download failed (ConnectionError))" in str(exc.value)

    def test_every_missing_ticker_is_listed(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            _load({"NOPE1": 0.5, "NOPE2": 0.5})
        assert "NOPE1" in str(exc.value) and "NOPE2" in str(exc.value)

    def test_weights_are_kept_as_requested(self, stub_prices):
        p = _load({"SPY": 3, "AGG": 1})
        assert p.symbols == ["SPY", "AGG"]
        assert p.weights == {"SPY": 0.75, "AGG": 0.25}

    def test_normalize_weights_false_is_respected(self, stub_prices):
        """A missing ticker must not rescale the 'available' tickers, which would also
        override normalize_weights=False."""
        p = _load({"SPY": 0.3, "AGG": 0.3}, normalize_weights=False)
        assert p.weights == {"SPY": 0.3, "AGG": 0.3}

    def test_too_few_common_sessions_is_rejected(self, stub_prices):
        with pytest.raises(ValueError, match="at least 3"):
            _load({"SPY": 1.0}, start="2024-02-28", end="2024-03-01")


class TestWindowIsTheRequestedOne:
    """The late-listing check compares each ticker with `start`, not with the other
    tickers, so {"COIN": 1} from 2015 is a 422: "listed after `start` ... is a 422"."""

    def test_single_late_listed_ticker_is_rejected(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            _load({"RIVN": 1.0})
        msg = str(exc.value)
        assert "RIVN (first available 2021-11-10)" in msg and "`start` (2020-01-01)" in msg
        assert "set `start` on or after 2021-11-10" in msg

    def test_every_late_ticker_is_named_even_when_they_agree(self, stub_prices):
        """Two tickers listed together after `start` are refused, though neither is late
        relative to the other."""
        with pytest.raises(ValueError) as exc:
            _load({"RIVN": 0.5, "RIVN2": 0.5})
        assert "RIVN (first available 2021-11-10)" in str(exc.value)
        assert "RIVN2 (first available 2021-11-10)" in str(exc.value)

    def test_single_ticker_delisted_before_an_explicit_end_is_rejected(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            _load({"GONE": 1.0})
        assert "GONE (last available 2022-06-30)" in str(exc.value)

    def test_without_end_a_sole_delisted_ticker_is_rejected(self, stub_prices):
        """`end` omitted means "to today", so GONE alone is compared with today, not with
        its own last date."""
        with pytest.raises(ValueError) as exc:
            PortfolioLoader().from_tickers({"GONE": 1.0}, start="2020-01-01", end=None)
        msg = str(exc.value)
        assert "GONE (last available 2022-06-30)" in msg and "today" in msg
        assert "set `end` on or before 2022-07-01" in msg

    def test_without_end_every_delisted_ticker_is_named_even_when_they_agree(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            PortfolioLoader().from_tickers({"SPY": 0.5, "AGG": 0.5}, start="2020-01-01", end=None)
        assert "SPY (last available 2024-02-29)" in str(exc.value)
        assert "AGG (last available 2024-02-29)" in str(exc.value)

    def test_without_end_a_live_ticker_runs_to_its_latest_price(self, monkeypatch, stub_prices):
        today = pd.Timestamp.today().normalize()
        monkeypatch.setitem(UNIVERSE, "LIVE", _prices(today - pd.Timedelta(days=400), today - pd.Timedelta(days=3), 9, 0.01))
        start = (today - pd.Timedelta(days=398)).date().isoformat()
        p = PortfolioLoader().from_tickers({"LIVE": 1.0}, start=start, end=None)
        assert p.prices.index[-1] == UNIVERSE["LIVE"].index[-1]

    def test_holiday_start_is_within_tolerance(self, stub_prices):
        p = _load({"SPY": 1.0}, start="2019-12-28")   # first session 2020-01-02: 5 days
        assert p.prices.index[0] == pd.Timestamp("2020-01-02")


class TestInteriorGaps:
    """An interior hole must not be spanned by one "daily" return (a 126-day halt as a -62%
    1-day return), nor the other tickers' sessions inside it dropped silently."""

    def test_long_gap_in_one_ticker_is_rejected_by_name_and_dates(self, stub_prices):
        with pytest.raises(ValueError) as exc:
            _load({"SPY": 0.5, "HOLE": 0.5})
        msg = str(exc.value)
        assert "HOLE has no prices between 2022-02-28 and 2022-07-04" in msg
        assert "126 calendar days" in msg

    def test_gap_alone_is_rejected_too(self, stub_prices):
        with pytest.raises(ValueError, match="HOLE"):
            _load({"HOLE": 1.0})

    def test_short_halt_within_tolerance_still_spans(self, stub_prices):
        """HALT misses 3 sessions (4 calendar days): aligned, not rejected."""
        assert _load({"SPY": 0.5, "HALT": 0.5}).n_assets == 2


def test_a_7_day_asset_alone_is_365_a_year_but_aligned_with_spy_is_252(stub_prices):
    """Mixed portfolios align to common sessions."""
    from riskcanvas.metrics.risk import periods_per_year

    assert periods_per_year(_load({"COIN7": 1.0}).returns.index) == 365
    assert periods_per_year(_load({"SPY": 0.6, "COIN7": 0.4}).returns.index) == 252


def test_fx_limit_is_disclosed():
    """from_tickers does no FX conversion, so yen and dollar returns blend: the package
    says so, and does not claim "any equity portfolio"."""
    import riskcanvas
    from pathlib import Path

    assert "any equity portfolio" not in riskcanvas.__doc__
    assert "one currency" in " ".join(riskcanvas.__doc__.split())
    assert "no FX conversion" in PortfolioLoader.from_tickers.__doc__
    root = Path(__file__).resolve().parents[1]
    for doc in ("README.md", "docs/usage.md"):
        assert "No FX conversion" in (root / doc).read_text(), doc


class TestMarketClosures:
    """A window across a regular long closure passes, and a longer one gets a way to accept
    it (`max_gap_days`), not the blame of "a halt, missing data". Calendars recalled,
    not checked against the exchanges; only the length of the gap matters here."""

    SHANGHAI_CLOSED = (
        ("2023-01-21", "2023-01-29"), ("2023-09-29", "2023-10-08"), ("2024-02-09", "2024-02-18"),
        ("2024-10-01", "2024-10-07"), ("2025-01-28", "2025-02-04"), ("2025-10-01", "2025-10-08"),
    )
    TOKYO_CLOSED = (("2019-04-27", "2019-05-06"),)   # 2019-04-26 to 2019-05-07: 11 days

    @staticmethod
    def _closed(start, end, closures, seed=1):
        idx = pd.bdate_range(start, end)
        for a, b in closures:
            idx = idx[~((idx >= a) & (idx <= b))]
        r = np.random.default_rng(seed).normal(0.0003, 0.012, len(idx))
        return pd.DataFrame({"Close": 100 * np.cumprod(1 + r)}, index=idx)

    @pytest.fixture
    def exchanges(self, monkeypatch):
        universe = {
            "600519.SS": self._closed("2023-01-03", "2025-12-30", self.SHANGHAI_CLOSED),
            "7203.T": self._closed("2019-01-04", "2019-12-30", self.TOKYO_CLOSED),
            # Closed 13 calendar days between sessions: longer than any regular closure.
            "LONG": self._closed("2023-01-03", "2023-12-29", (("2023-06-10", "2023-06-22"),)),
            "ODD": self._closed("2023-01-03", "2023-12-29", (("2023-06-10", "2023-06-18"),), seed=2),
            "EVEN": self._closed("2023-01-03", "2023-12-29", (("2023-06-17", "2023-06-25"),), seed=3),
        }

        def download(sym, start=None, end=None, **_k):
            df = universe[sym]
            return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]

        fake = types.ModuleType("yfinance")
        fake.download = download
        monkeypatch.setitem(sys.modules, "yfinance", fake)

    def test_a_share_window_across_golden_week_and_lunar_new_year_loads(self, exchanges):
        p = _load({"600519.SS": 1.0}, start="2023-01-03", end="2025-12-31")
        gaps = (p.prices.index[1:] - p.prices.index[:-1]).days
        assert gaps.max() == 11

    def test_tokyo_across_golden_week_2019_loads(self, exchanges):
        p = _load({"7203.T": 1.0}, start="2019-01-04", end="2019-12-31")
        assert pd.Timestamp("2019-05-07") in p.returns.index

    def test_a_longer_gap_is_refused_and_says_how_to_accept_it(self, exchanges):
        with pytest.raises(ValueError) as exc:
            _load({"LONG": 1.0}, start="2023-01-03", end="2023-12-30")
        msg = str(exc.value)
        assert "LONG has no prices between 2023-06-09 and 2023-06-23 (14 calendar days)" in msg
        assert "`max_gap_days=14`" in msg and "market closure" in msg and "halt or missing data" in msg

    def test_max_gap_days_accepts_exactly_what_it_says(self, exchanges):
        kw = {"start": "2023-01-03", "end": "2023-12-30"}
        with pytest.raises(ValueError, match="max_gap_days=14"):
            _load({"LONG": 1.0}, max_gap_days=13, **kw)
        p = _load({"LONG": 1.0}, max_gap_days=14, **kw)
        raw = p.prices["LONG"]
        assert p.returns.loc["2023-06-23", "LONG"] == pytest.approx(raw.loc["2023-06-23"] / raw.loc["2023-06-09"] - 1, rel=1e-12)

    def test_the_gap_between_common_sessions_has_the_same_rule(self, exchanges):
        kw = {"start": "2023-01-03", "end": "2023-12-30"}
        with pytest.raises(ValueError, match=r"No session common to every ticker.*`max_gap_days=\d+`"):
            _load({"ODD": 0.5, "EVEN": 0.5}, **kw)
        assert _load({"ODD": 0.5, "EVEN": 0.5}, max_gap_days=30, **kw).n_assets == 2

    def test_a_late_listing_is_not_mistaken_for_a_closure(self, stub_prices):
        """The window's ends keep the 7-day tolerance unless the caller widens it."""
        with pytest.raises(ValueError, match="first available 2021-11-10"):
            _load({"RIVN": 1.0}, start="2021-11-01")      # 9 days before the listing
        assert _load({"RIVN": 1.0}, start="2021-11-01", max_gap_days=9).n_assets == 1

    def test_a_halt_beyond_any_closure_is_still_refused_by_default(self, stub_prices):
        with pytest.raises(ValueError, match="HOLE"):
            _load({"HOLE": 1.0})
        assert _load({"HOLE": 1.0}, max_gap_days=130).n_assets == 1

    @pytest.mark.parametrize("bad", [0, -1, 1.5])
    def test_max_gap_days_is_a_whole_number_of_at_least_one(self, stub_prices, bad):
        with pytest.raises(ValueError, match="max_gap_days"):
            _load({"SPY": 1.0}, max_gap_days=bad)
