"""Offline loading (PortfolioLoader.from_returns_df / from_csv) enforces the same contract
as the HTTP API: no invented data, no unchecked weights.

A CSV gap is an error, never summed as a 0% day, which would understate risk. Weights
are checked and normalised: {"A": 60, "B": 40} taken as given would give a 1-day
historical VaR of 123.7%. A weight for a ticker absent from the data is an error, never
dropped. A CSV without the date column is an error, never dated from 1970 by casting its
row numbers to timestamps."""

import math

import numpy as np
import pandas as pd
import pytest

from riskcanvas.metrics.risk import RiskMetrics
from riskcanvas.portfolio import loader as loader_module
from riskcanvas.portfolio.loader import PERCENT_MIN_TYPICAL_MOVE, Portfolio, PortfolioLoader
from riskcanvas.simulation.monte_carlo import MonteCarloSimulator

# A file where B has no value on three of five days.
GAP_CSV = """Date,A,B
2024-01-01,0.01,-0.05
2024-01-02,-0.02,
2024-01-03,0.015,
2024-01-04,-0.01,0.03
2024-01-05,0.005,
"""

CLEAN_CSV = """Date,A,B
2024-01-01,0.01,-0.05
2024-01-02,-0.02,0.01
2024-01-03,0.015,-0.004
2024-01-04,-0.01,0.03
2024-01-05,0.005,0.002
"""


def _csv(tmp_path, text, name="returns.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def _frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"A": rng.normal(0.0005, 0.01, n), "B": rng.normal(0.0003, 0.012, n)}, index=idx
    )


L = PortfolioLoader()


class TestFromCsv:
    def test_gap_is_rejected_naming_the_column_and_every_date(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            L.from_csv(_csv(tmp_path, GAP_CSV), weights={"A": 0.5, "B": 0.5})
        msg = str(exc.value)
        assert "B" in msg
        for day in ("2024-01-02", "2024-01-03", "2024-01-05"):
            assert day in msg
        assert "2024-01-04" not in msg and "never treated as a 0% day" in msg

    def test_gap_is_never_a_zero_return_day(self, tmp_path):
        """Summing past the gap would answer 2024-01-02 with 0.5 * -2% + 0.5 * 0% = -1%."""
        with pytest.raises(ValueError):
            L.from_csv(_csv(tmp_path, GAP_CSV), weights={"A": 0.5, "B": 0.5}).portfolio_returns

    def test_drop_is_an_explicit_opt_in_that_excludes_whole_dates(self, tmp_path):
        p = L.from_csv(_csv(tmp_path, GAP_CSV), weights={"A": 0.5, "B": 0.5}, missing="drop")
        assert list(p.returns.index) == list(pd.to_datetime(["2024-01-01", "2024-01-04"]))
        assert p.portfolio_returns.tolist() == pytest.approx(
            [0.5 * 0.01 + 0.5 * -0.05, 0.5 * -0.01 + 0.5 * 0.03], abs=1e-15
        )

    def test_unknown_missing_policy_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            L.from_csv(_csv(tmp_path, GAP_CSV), missing="zero")

    def test_percent_weights_are_normalised_like_the_api(self, tmp_path):
        path = _csv(tmp_path, CLEAN_CSV)
        pct = L.from_csv(path, weights={"A": 60, "B": 40})
        frac = L.from_csv(path, weights={"A": 0.6, "B": 0.4})
        assert pct.weights == pytest.approx({"A": 0.6, "B": 0.4}, abs=1e-15)
        assert math.fsum(pct.weights.values()) == pytest.approx(1.0, abs=1e-15)
        assert pct.portfolio_returns.tolist() == pytest.approx(frac.portfolio_returns.tolist(), abs=1e-15)

    def test_weight_for_a_ticker_absent_from_the_file_is_an_error(self, tmp_path):
        with pytest.raises(ValueError) as exc:
            L.from_csv(_csv(tmp_path, CLEAN_CSV), weights={"A": 0.5, "C": 0.5})
        assert "C" in str(exc.value) and "never dropped silently" in str(exc.value)

    def test_missing_date_column_is_an_error_not_1970_dates(self, tmp_path):
        """Without the date column the row numbers must not be cast to timestamps
        (1970-01-01 00:00:00.000000000, ...)."""
        path = _csv(tmp_path, CLEAN_CSV.replace("Date,", "Day,"))
        with pytest.raises(ValueError) as exc:
            L.from_csv(path)
        assert "'Date'" in str(exc.value) and "Day" in str(exc.value)
        assert L.from_csv(path, date_col="Day").returns.index[0] == pd.Timestamp("2024-01-01")

    def test_dates_out_of_order_are_an_error(self, tmp_path):
        lines = CLEAN_CSV.strip().splitlines()
        reversed_csv = "\n".join([lines[0]] + lines[:0:-1]) + "\n"
        with pytest.raises(ValueError, match="increasing"):
            L.from_csv(_csv(tmp_path, reversed_csv))

    def test_percent_valued_returns_are_an_error(self, tmp_path):
        """-2.5 meaning -2.5% would otherwise be read as a -250% day."""
        with pytest.raises(ValueError, match="fractions"):
            L.from_csv(_csv(tmp_path, CLEAN_CSV.replace("-0.02,", "-2.5,")))

    def test_return_cols_selects_and_rejects_an_absent_column(self, tmp_path):
        path = _csv(tmp_path, CLEAN_CSV)
        p = L.from_csv(path, return_cols=["B"])
        assert p.symbols == ["B"] and p.weights == {"B": 1.0}
        with pytest.raises(ValueError, match="Z"):
            L.from_csv(path, return_cols=["B", "Z"])

    def test_clean_file_loads_exactly_with_equal_default_weights(self, tmp_path):
        p = L.from_csv(_csv(tmp_path, CLEAN_CSV))
        assert p.symbols == ["A", "B"] and p.weights == {"A": 0.5, "B": 0.5}
        assert p.returns.index[0] == pd.Timestamp("2024-01-01") and len(p.returns) == 5
        assert p.returns.loc["2024-01-03", "B"] == -0.004
        assert p.portfolio_returns.iloc[1] == pytest.approx(0.5 * -0.02 + 0.5 * 0.01, abs=1e-15)


class TestFromReturnsDf:
    def test_percent_weights_no_longer_give_a_123_percent_var(self):
        df = _frame()
        pct = RiskMetrics(L.from_returns_df(df, {"A": 60, "B": 40}).portfolio_returns)
        frac = RiskMetrics(L.from_returns_df(df, {"A": 0.6, "B": 0.4}).portfolio_returns)
        assert pct.var_historical(0.95) == pytest.approx(frac.var_historical(0.95), rel=1e-12)
        assert 0 < pct.var_historical(0.95) < 0.05

    def test_monte_carlo_sees_normalised_weights(self):
        """Percent weights left unnormalised would put the GBM terminal at ~1e40."""
        df = _frame()
        sim = MonteCarloSimulator(n_paths=1000, seed=3)
        pct = sim.simulate_gbm(L.from_returns_df(df, {"A": 60, "B": 40}), 20)
        frac = sim.simulate_gbm(L.from_returns_df(df, {"A": 0.6, "B": 0.4}), 20)
        assert pct.var_95 == pytest.approx(frac.var_95, rel=1e-9)
        assert pct.expected_terminal == pytest.approx(frac.expected_terminal, rel=1e-9)

    @pytest.mark.parametrize("where", ["interior", "leading", "trailing"])
    def test_missing_value_anywhere_in_a_held_series_is_an_error(self, where):
        df = _frame(n=30)
        row = {"interior": 10, "leading": 0, "trailing": 29}[where]
        df.iloc[row, 1] = np.nan
        with pytest.raises(ValueError) as exc:
            L.from_returns_df(df, {"A": 0.5, "B": 0.5})
        assert "B on " + df.index[row].date().isoformat() in str(exc.value)

    def test_long_gap_lists_the_first_dates_and_counts_the_rest(self):
        df = _frame(n=30)
        df.iloc[3:20, 0] = np.nan
        with pytest.raises(ValueError) as exc:
            L.from_returns_df(df)
        assert "A on " + df.index[3].date().isoformat() in str(exc.value)
        assert "(+12 more)" in str(exc.value)

    @pytest.mark.parametrize("bad", [np.inf, -np.inf, -1.5, 10.5])
    def test_non_finite_or_out_of_range_value_is_an_error(self, bad):
        df = _frame(n=30)
        df.iloc[7, 0] = bad
        with pytest.raises(ValueError) as exc:
            L.from_returns_df(df)
        assert "'A'" in str(exc.value) and df.index[7].date().isoformat() in str(exc.value)

    def test_unweighted_columns_are_not_held_and_cannot_block(self):
        df = _frame(n=30)
        df["C"] = np.nan
        p = L.from_returns_df(df, {"A": 1.0})
        assert p.symbols == ["A"] and list(p.returns.columns) == ["A"]
        assert p.portfolio_returns.tolist() == df["A"].tolist()

    @pytest.mark.parametrize(
        "weights, fragment",
        [
            ({"A": -0.2, "B": 1.2}, "short"),
            ({"A": float("nan"), "B": 1.0}, "finite"),
            ({"A": float("inf"), "B": 1.0}, "finite"),
            ({"A": 0.0, "B": 0.0}, "positive"),
            ({"A": 1e308, "B": 1e308}, "overflow"),
            ({}, "at least one"),
        ],
    )
    def test_invalid_weights_are_an_error(self, weights, fragment):
        with pytest.raises(ValueError, match=fragment):
            L.from_returns_df(_frame(n=30), weights)

    def test_normalize_weights_false_is_respected(self):
        """Same switch as from_tickers: a partially invested portfolio."""
        p = L.from_returns_df(_frame(n=30), {"A": 0.3, "B": 0.3}, normalize_weights=False)
        assert p.weights == {"A": 0.3, "B": 0.3}

    def test_index_must_be_dates(self):
        with pytest.raises(ValueError, match="DatetimeIndex"):
            L.from_returns_df(_frame(n=30).reset_index(drop=True))

    def test_duplicate_dates_are_an_error(self):
        df = _frame(n=30)
        df.index = df.index[:5].append(df.index[4:29])
        with pytest.raises(ValueError, match="increasing"):
            L.from_returns_df(df)

    def test_fewer_than_two_observations_is_an_error(self):
        with pytest.raises(ValueError, match="at least 2"):
            L.from_returns_df(_frame(n=1))


class TestBooleanColumnsAreRefused:
    """A column of true/false is refused: its True days would read as +100%."""

    DATES = pd.bdate_range("2023-01-02", periods=4)

    @pytest.mark.parametrize("values", [
        [True, False, True, False],
        np.array([True, False, True, False], dtype=object),
        np.array([0.01, True, -0.01, 0.02], dtype=object),
        pd.array([True, False, True, False], dtype="boolean"),
    ], ids=["bool", "object", "one-in-an-object-column", "nullable"])
    def test_from_returns_df(self, values):
        frame = pd.DataFrame({"A": values}, index=self.DATES)
        with pytest.raises(ValueError, match="'A' is not numeric: it holds true/false"):
            PortfolioLoader().from_returns_df(frame)

    def test_from_csv(self, tmp_path):
        path = _csv(tmp_path, "Date,A\n2023-01-02,True\n2023-01-03,False\n2023-01-04,True\n")
        with pytest.raises(ValueError, match="true/false"):
            PortfolioLoader().from_csv(path)

    def test_a_bool_column_nobody_holds_is_ignored(self):
        frame = pd.DataFrame({"A": [0.01, -0.01, 0.0, 0.02], "open": [True, True, False, True]}, index=self.DATES)
        assert PortfolioLoader().from_returns_df(frame, weights={"A": 1}).symbols == ["A"]

    def test_numbers_stored_as_objects_still_load(self):
        frame = pd.DataFrame({"A": np.array([0.01, -0.01, 0.0, 0.02], dtype=object)}, index=self.DATES)
        assert PortfolioLoader().from_returns_df(frame).returns["A"].tolist() == [0.01, -0.01, 0.0, 0.02]


class TestCheckWeights:
    """One weight rule, shared by every loader and the HTTP API."""

    def test_normalises_by_the_sum_only(self):
        check_weights = loader_module.check_weights
        assert check_weights({"A": 3, "B": 1}) == {"A": 0.75, "B": 0.25}
        # normalize=False returns the exposures unchanged; a sum above 1
        # (leverage) is refused.
        assert check_weights({"A": 0.3, "B": 0.1}, normalize=False) == {"A": 0.3, "B": 0.1}
        with pytest.raises(ValueError, match="leverage"):
            check_weights({"A": 3, "B": 1}, normalize=False)

    def test_zero_weight_is_kept(self):
        assert loader_module.check_weights({"A": 1, "B": 0}) == {"A": 1.0, "B": 0.0}

    def test_overflowing_sum_is_an_error_not_all_zero_weights(self):
        with pytest.raises(ValueError, match="overflow"):
            loader_module.check_weights({"A": 1e308, "B": 1e308})


class TestPortfolioIsValidated:
    """A Portfolio built by hand gets the same guarantees: its weighted
    return never skips a missing value or a weighted asset."""

    def _parts(self):
        df = _frame(n=30)
        return df, 100 * (1 + df).cumprod()

    def test_nan_in_a_held_series_is_rejected_at_construction(self):
        df, prices = self._parts()
        df.iloc[4, 1] = np.nan
        with pytest.raises(ValueError, match="B on"):
            Portfolio(symbols=["A", "B"], weights={"A": 0.5, "B": 0.5}, prices=prices, returns=df)

    def test_weight_without_a_returns_column_is_rejected(self):
        df, prices = self._parts()
        with pytest.raises(ValueError, match="never dropped silently"):
            Portfolio(symbols=["A", "C"], weights={"A": 0.5, "C": 0.5}, prices=prices, returns=df)

    def test_nan_introduced_after_construction_raises_instead_of_counting_zero(self):
        df, prices = self._parts()
        p = Portfolio(symbols=["A", "B"], weights={"A": 0.5, "B": 0.5}, prices=prices, returns=df.copy())
        p.returns.iloc[4, 1] = np.nan
        with pytest.raises(ValueError, match="B on"):
            p.portfolio_returns


class TestPercentColumns:
    """A file in percent can pass the [-1, 10] range when its worst day is above -1 (%): a
    bond fund would read VaR95 42%, volatility 402%. It is refused."""

    IDX = pd.bdate_range("2021-01-04", periods=300)

    def _series(self, sd, seed=3, mean=0.0):
        return np.random.default_rng(seed).normal(mean, sd, 300)

    def _load(self, values, **kw):
        return PortfolioLoader().from_returns_df(pd.DataFrame({"AGG": values}, index=self.IDX), **kw)

    @pytest.mark.parametrize("sd", [0.25, 0.30, 0.20])
    def test_quiet_asset_in_percent_is_refused_naming_the_column_and_the_cure(self, sd):
        values = np.clip(self._series(sd, mean=0.01), -0.95, 3)
        with pytest.raises(ValueError, match="looks like percent") as exc:
            self._load(values)
        assert "'AGG'" in str(exc.value) and "divide the column by 100" in str(exc.value)
        assert "allow_large_returns" in str(exc.value)

    def test_a_csv_in_percent_is_refused_too(self, tmp_path):
        path = tmp_path / "pct.csv"
        pd.DataFrame({"Date": self.IDX, "AGG": np.clip(self._series(0.25), -0.95, 3)}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="looks like percent"):
            PortfolioLoader().from_csv(path)
        assert PortfolioLoader().from_csv(path, allow_large_returns=True).n_assets == 1

    def test_the_same_data_as_fractions_is_accepted(self):
        values = np.clip(self._series(0.25), -0.95, 3) / 100
        assert self._load(values).n_assets == 1

    @pytest.mark.parametrize("sd", [0.01, 0.04, 0.08, 0.10])
    def test_legitimate_high_volatility_is_not_refused(self, sd):
        """Up to 10% a day (159% a year), which no equity or major coin reaches."""
        p = self._load(self._series(sd))
        assert p.n_assets == 1

    def test_a_frenzy_of_wild_days_on_a_normal_base_is_not_refused(self):
        """The median ignores outliers: a meme-stock window with +130% and -60%
        days around a 4% a day base."""
        values = self._series(0.04)
        values[[20, 21, 50, 51, 52, 80, 120, 200]] = [1.3, -0.6, 0.7, -0.4, 0.9, -0.5, 0.6, -0.3]
        assert self._load(values).n_assets == 1

    def test_fat_tails_at_20_percent_daily_volatility_are_not_refused(self):
        raw = np.random.default_rng(4).standard_t(3, 300) * (0.20 / np.sqrt(3))
        assert self._load(np.clip(raw, -0.99, 9)).n_assets == 1

    def test_one_way_and_short_series_are_not_percent_shaped(self):
        # +50% every day has no down day, so it is not percent-shaped; it reads as
        # levels instead (see TestLevelsAreRefused).
        with pytest.raises(ValueError, match="looks like levels"):
            self._load(np.full(300, 0.5))
        assert self._load(np.full(300, -0.5)).n_assets == 1      # no up day
        short = pd.DataFrame({"A": [0.2, -0.3, 0.15, -0.25, 0.4]}, index=self.IDX[:5])
        assert PortfolioLoader().from_returns_df(short).n_assets == 1

    def test_the_stated_limit_a_cash_like_fund_in_percent_passes(self):
        """Documented, not hidden: typical moves under 0.10 read as fractions."""
        assert self._load(self._series(0.02)).n_assets == 1

    def test_the_threshold_on_both_sides(self):
        base = np.resize([1.0, -1.0], 300)
        assert self._load(base * 0.0999).n_assets == 1
        with pytest.raises(ValueError, match="looks like percent"):
            self._load(base * 0.10)
        with pytest.raises(ValueError, match="looks like percent"):
            self._load(base * 0.9999)
        assert self._load(base * 1.0).n_assets == 1    # every day +-100%: not percent-shaped

    def test_only_the_percent_column_is_named(self):
        frame = pd.DataFrame({"OK": self._series(0.01), "PCT": np.clip(self._series(0.25), -0.95, 3)}, index=self.IDX)
        with pytest.raises(ValueError, match="'PCT'"):
            PortfolioLoader().from_returns_df(frame)


class TestATotalLossIsMinusOne:
    """Weights can sum a rounding error above 1; a day on which every holding loses
    everything still reads -100%, not below it."""

    IDX = pd.bdate_range("2021-01-04", periods=40)
    SERIES = np.array([0.01, -0.01, -1.0] + [0.01] * 37)

    def _portfolio(self):
        frame = pd.DataFrame({k: self.SERIES for k in "ABC"}, index=self.IDX)
        return PortfolioLoader().from_returns_df(frame, weights={"A": 6, "B": 23, "C": 1})

    def test_the_weights_do_sum_over_one(self):
        # Added in order, as the weighted sum does: Python 3.12's sum() compensates.
        assert np.cumsum(list(self._portfolio().weights.values()))[-1] > 1

    def test_the_portfolio_return_is_exactly_minus_one(self):
        assert self._portfolio().portfolio_returns.iloc[2] == -1.0

    def test_the_other_days_are_untouched(self):
        p = self._portfolio()
        expected = (p.returns * pd.Series(p.weights)).sum(axis=1)
        assert p.portfolio_returns.drop(self.IDX[2]).equals(expected.drop(self.IDX[2]))

    def test_a_return_further_below_is_not_rounding(self):
        frame = pd.DataFrame({"A": [0.01, -1.5, 0.01]}, index=self.IDX[:3])
        p = Portfolio(symbols=["A"], weights={"A": 1.0}, prices=frame, returns=frame)
        assert p.portfolio_returns.iloc[1] == -1.5
        with pytest.raises(ValueError, match="below -1.0"):
            RiskMetrics(p.portfolio_returns).compute_all()

    def test_a_bootstrap_path_never_goes_below_nothing(self):
        result = MonteCarloSimulator(n_paths=2000, seed=1, drift="historical").simulate_bootstrap(self._portfolio(), 40)
        assert result.terminal_values.min() >= 0.0


class TestChecksFollowTheRate:
    """The percent and levels floors follow the series' rate. At a daily floor 34 annual
    returns of an equity index (median move 17%) would be refused as percent, and
    following the advice to divide by 100 would shrink them 100-fold; six up years at
    10-20% would be refused as levels."""

    # An equity index's calendar years, recalled and rounded: the shape, not a dataset.
    ANNUAL = np.array([
        -0.03, 0.30, 0.08, 0.10, 0.01, 0.38, 0.23, 0.33, 0.29, 0.21, -0.09, -0.12, -0.22, 0.29, 0.11, 0.05, 0.16,
        0.05, -0.37, 0.26, 0.15, 0.02, 0.16, 0.32, 0.14, 0.01, 0.12, 0.22, -0.04, 0.31, 0.18, 0.29, -0.18, 0.26,
    ])
    UP_YEARS = np.array([0.10, 0.18, 0.12, 0.20, 0.15, 0.11])
    MONTHLY = np.random.default_rng(8).normal(0.01, 0.18, 120)
    ALTERNATING = np.resize([1.0, -1.0], 120)

    def _load(self, values, rate=None, freq=None, **kw):
        idx = pd.date_range("1990-01-31", periods=len(values), freq=freq or pd.DateOffset(months=12))
        return PortfolioLoader().from_returns_df(pd.DataFrame({"A": values}, index=idx), periods_per_year=rate, **kw)

    def test_annual_returns_are_returns(self):
        assert float(np.median(np.abs(self.ANNUAL))) > PERCENT_MIN_TYPICAL_MOVE
        assert self._load(self.ANNUAL, rate=1).n_assets == 1

    def test_read_as_daily_they_are_still_refused_and_the_message_names_the_rate(self):
        with pytest.raises(ValueError, match="looks like percent") as exc:
            self._load(self.ANNUAL, rate=252)
        assert "periods_per_year" in str(exc.value) and "typical day" in str(exc.value)

    def test_six_up_years_are_not_levels(self):
        assert self._load(self.UP_YEARS, rate=1).n_assets == 1
        with pytest.raises(ValueError, match="looks like levels"):
            self._load(self.UP_YEARS, rate=252)

    def test_monthly_returns_at_18_percent_volatility_are_returns(self):
        assert self._load(self.MONTHLY, rate=12, freq="ME").n_assets == 1
        with pytest.raises(ValueError, match="looks like percent"):
            self._load(self.MONTHLY, rate=252, freq="ME")

    @pytest.mark.parametrize("rate, floor", [(52, 0.10 * np.sqrt(252 / 52)), (12, 0.10 * np.sqrt(21)), (4, 0.10 * np.sqrt(63))])
    def test_a_percent_series_is_refused_from_the_scaled_floor_up(self, rate, floor):
        assert self._load(self.ALTERNATING * floor * 0.999, rate=rate, freq="ME").n_assets == 1
        with pytest.raises(ValueError, match="looks like percent") as exc:
            self._load(self.ALTERNATING * floor * 1.001, rate=rate, freq="ME")
        assert "typical observation" in str(exc.value) and "periods_per_year" not in str(exc.value)

    def test_a_quiet_monthly_fund_in_percent_is_still_refused(self):
        values = np.random.default_rng(9).normal(0.05, 0.9, 120).clip(-0.99, 3)
        with pytest.raises(ValueError, match="looks like percent"):
            self._load(values, rate=12, freq="ME")
        assert self._load(values / 100, rate=12, freq="ME").n_assets == 1

    @pytest.mark.parametrize("rate", [1, 2, 4, 12, 52])
    def test_gross_returns_are_refused_at_every_rate(self, rate):
        with pytest.raises(ValueError, match="looks like levels"):
            self._load(1 + self.ANNUAL, rate=rate)

    def test_the_levels_floor_on_both_sides_of_its_cap(self):
        cap = loader_module.LEVELS_MAX_TYPICAL_MOVE
        assert self._load(np.full(10, cap * 0.999), rate=1).n_assets == 1
        with pytest.raises(ValueError, match="looks like levels"):
            self._load(np.full(10, cap), rate=1)

    def test_prices_are_refused_at_every_rate(self):
        prices = 5 * np.cumprod(1 + self.ANNUAL / 10)
        with pytest.raises(ValueError, match="looks like levels"):
            self._load(prices, rate=12)

    def test_a_rate_faster_than_daily_keeps_the_daily_floor(self):
        base = np.resize([1.0, -1.0], 300)
        idx = pd.date_range("2020-01-01", periods=300, freq="D")
        frame = pd.DataFrame({"A": base * 0.10}, index=idx)
        with pytest.raises(ValueError, match="looks like percent"):
            PortfolioLoader().from_returns_df(frame, periods_per_year=365)

    @pytest.mark.parametrize("freq, values", [("ME", MONTHLY), (pd.DateOffset(months=12), ANNUAL)], ids=["monthly", "annual"])
    def test_the_dates_set_the_rate_when_none_is_stated(self, freq, values, tmp_path):
        assert self._load(values, freq=freq).n_assets == 1
        path = tmp_path / "dated.csv"
        idx = pd.date_range("1990-01-31", periods=len(values), freq=freq)
        pd.DataFrame({"A": values}, index=idx).rename_axis("Date").to_csv(path)
        assert PortfolioLoader().from_csv(path).n_assets == 1

    def test_dates_the_metrics_refuse_keep_the_daily_floor(self):
        """Hourly bars show no rate; the metrics refuse them, the loader as before."""
        frame = pd.DataFrame({"A": self.ALTERNATING * 0.10}, index=pd.date_range("2021-01-04", periods=120, freq="h"))
        with pytest.raises(ValueError, match="looks like percent"):
            PortfolioLoader().from_returns_df(frame)

    @pytest.mark.parametrize("rate", [0, -12])
    def test_a_rate_must_be_positive(self, rate):
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            self._load(self.ANNUAL, rate=rate)


class TestLevelsAreRefused:
    IDX = pd.bdate_range("2021-01-04", periods=300)
    R = np.random.default_rng(5).normal(0.0005, 0.01, 300)

    @pytest.mark.parametrize("values", [1 + R, 5 * np.cumprod(1 + R)], ids=["gross", "prices"])
    def test_from_returns_df_and_from_csv_refuse_them(self, values, tmp_path):
        frame = pd.DataFrame({"A": values}, index=self.IDX)
        with pytest.raises(ValueError, match="looks like levels"):
            PortfolioLoader().from_returns_df(frame)
        path = tmp_path / "levels.csv"
        frame.rename_axis("Date").to_csv(path)
        with pytest.raises(ValueError, match="allow_large_returns"):
            PortfolioLoader().from_csv(path)
        assert PortfolioLoader().from_returns_df(frame, allow_large_returns=True).n_assets == 1

    def test_only_a_held_column_is_checked(self):
        frame = pd.DataFrame({"A": self.R, "PX": 5 * np.cumprod(1 + self.R)}, index=self.IDX)
        assert PortfolioLoader().from_returns_df(frame, weights={"A": 1}).n_assets == 1

    def test_a_fat_tailed_high_vol_asset_passes(self):
        """About 8% daily volatility, 127% a year, with both up and down days."""
        frame = pd.DataFrame({"A": np.random.default_rng(6).normal(0.002, 0.08, 300)}, index=self.IDX)
        assert PortfolioLoader().from_returns_df(frame).n_assets == 1

    def test_four_observations_are_too_few_to_call(self):
        frame = pd.DataFrame({"A": [1.01, 0.99, 1.02, 1.0]}, index=self.IDX[:4])
        assert PortfolioLoader().from_returns_df(frame).n_assets == 1
