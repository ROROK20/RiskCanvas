"""
Portfolio data loader.

Builds a :class:`Portfolio` from one of three sources:

1. ``from_tickers`` — {symbol: weight}, prices fetched from yfinance
2. ``from_returns_df`` — a DataFrame of daily returns (offline)
3. ``from_csv`` — a CSV of dated daily returns (offline)

All three enforce the contract the HTTP API enforces: weights pass
:func:`check_weights` (long-only, finite, normalised by their sum and nothing
else), every weighted ticker has data, and no value is ever invented — a
missing price or return is an error naming it, never a 0% day.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# A daily return is a finite fraction in [MIN_DAILY_RETURN, MAX_DAILY_RETURN].
# Shared with the HTTP API (riskcanvas.api.app imports these), so the library
# and the service accept exactly the same data.
MIN_DAILY_RETURN = -1.0          # a -100% day; below that is not a return
MAX_DAILY_RETURN = 10.0          # +1000% in a day: anything larger is bad data
MIN_RETURNS_LEN = 2              # observations per series

# A column in percent, not fractions, passes the [-1, 10] range when its worst
# day is better than -1 (%): a quiet asset, such as a bond fund, whose typical
# day then reads as a move of 10-100%. See _refuse_percent_column.
PERCENT_MIN_TYPICAL_MOVE = 0.10  # median |return|; about 15% daily vol, 235% a year, if normal
PERCENT_MAX_TYPICAL_MOVE = 1.0   # from here on, a percent file has days below -1: range error
PERCENT_MIN_OBSERVATIONS = 20

# A column of levels, not returns: gross returns (1.01, 0.99, ...) or prices under
# 10 pass the range, and read as a gain of 100% or more every day. Almost every
# day up, by a typical 10% or more, is no return series. See _refuse_levels_column.
LEVELS_MIN_UP_SHARE = 0.95
LEVELS_MIN_OBSERVATIONS = 5
# Gross returns have a median near 1 at any rate, so the levels floor stops here
# however long one observation is. A real series with a typical gain of 75% a year
# and hardly a down year is as rare as the daily one the floor describes.
LEVELS_MAX_TYPICAL_MOVE = 0.75

# How far above 100% an exposure may sum: what normalising by a sum can leave behind.
WEIGHT_SUM_TOLERANCE = 1e-9

MISSING_POLICIES = ("raise", "drop")
_MAX_DATES_LISTED = 5

# Live price histories may start or end a few calendar days from the requested
# window for calendar reasons alone: exchange holidays, weekends, an asset that
# also trades at weekends. Anything longer is a ticker listed after `start` or
# delisted before `end`, and is rejected.
MAX_EDGE_GAP_DAYS = 7
# Inside the window, exchanges close for longer: up to 11 calendar days between
# sessions (Tokyo's Golden Week 2019, Shanghai's Lunar New Year and National Day
# weeks; dates recalled from the exchange calendars, not re-checked against them),
# and a neighbouring holiday on a mixed portfolio can add a day. A longer gap is a
# halt, missing data or an exceptional closure, and is rejected unless the caller
# accepts it with `max_gap_days`. A gap is never filled in.
MAX_CLOSURE_GAP_DAYS = 12

# Yahoo Finance listing suffixes, by the region whose exchanges close within hours of
# one another (recalled from Yahoo's list, not re-checked against it). Anything else,
# no suffix included, reads as the Americas. See close_region.
AMERICAS = "the Americas"
_CLOSE_REGION_SUFFIXES = {
    "Europe, the Middle East and Africa": frozenset({
        "L", "IL", "DE", "F", "BE", "DU", "HM", "HA", "MU", "SG", "PA", "AS", "BR", "LS", "MI", "MC", "SW",
        "VI", "ST", "OL", "HE", "CO", "IC", "IR", "WA", "PR", "BD", "AT", "IS", "JO", "TA", "SR", "QA",
    }),
    "Asia-Pacific": frozenset({
        "T", "KS", "KQ", "HK", "SS", "SZ", "TW", "TWO", "SI", "AX", "NZ", "NS", "BO", "JK", "KL", "BK",
    }),
}


def close_region(ticker: str) -> str:
    """The region a ticker's closing price is struck in, judged from its Yahoo
    Finance suffix: a heuristic. ``7203.T`` is Asia-Pacific, ``VOD.L`` Europe, and a
    ticker with no known suffix (``AAPL``, ``BRK.B``, ``BTC-USD``) reads as the
    Americas. Exchanges inside one region share hours of trading, so Tokyo with Hong
    Kong counts as one region although their closes differ by about 90 minutes."""
    suffix = ticker.rpartition(".")[2].upper() if "." in ticker else ""
    for region, suffixes in _CLOSE_REGION_SUFFIXES.items():
        if suffix in suffixes:
            return region
    return AMERICAS


def check_weights(weights: Mapping[str, float], *, normalize: bool = True) -> dict[str, float]:
    """Validate long-only portfolio weights and, by default, scale them to sum to 1.

    The one weight rule for every entry point: ``from_tickers``,
    ``from_returns_df``, ``from_csv`` and the HTTP API (which answers 422 with
    this message), so the library and the service cannot disagree. Dividing
    by the sum is the only transformation ever applied: ``{"A": 60, "B": 40}``
    becomes ``{"A": 0.6, "B": 0.4}``.

    Parameters
    ----------
    weights : Mapping[str, float]
        {ticker: weight}.
    normalize : bool
        Divide by the sum (default). ``False`` validates and returns the
        weights unchanged as the portfolio's exposures: a partially invested
        portfolio, whose remainder ``1 - sum`` is held as cash (zero return;
        the risk-free rate under Monte Carlo ``drift="risk_free"``). An
        exposure above 100% (leverage) is refused: no financing is modelled.

    Raises
    ------
    ValueError
        No weights; a weight that is not a finite number ``>= 0`` (short
        positions are not modelled); a sum that is not positive; a sum that
        overflows floating point (``{"A": 1e308, "B": 1e308}`` would
        normalise every weight to 0); with ``normalize=False``, a sum above 1
        (leverage).
    """
    if not weights:
        raise ValueError("weights must contain at least one asset.")
    bad = {}
    for ticker, w in weights.items():
        try:
            ok = math.isfinite(w) and w >= 0
        except TypeError:
            ok = False
        if not ok:
            bad[ticker] = w
    if bad:
        raise ValueError(
            "weights must be finite numbers >= 0: short positions are not "
            f"supported (RiskCanvas models long-only portfolios). Invalid: {bad}."
        )
    total = sum(float(w) for w in weights.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number (all weights are 0).")
    if not math.isfinite(total):
        raise ValueError(
            f"weights sum to {total} (floating-point overflow). Scale them down, "
            "e.g. to fractions that sum to 1 or percentages that sum to 100."
        )
    if not normalize:
        if total > 1.0 + WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"weights sum to {total:g} with normalize_weights=False: an exposure "
                "above 100% (leverage) is not modelled (no financing cost). Normalise "
                "the weights or keep their sum at or below 1 (the rest is cash)."
            )
        return {ticker: float(w) for ticker, w in weights.items()}
    return {ticker: float(w) / total for ticker, w in weights.items()}


def floor_at_total_loss(returns: np.ndarray) -> np.ndarray:
    """Portfolio returns with rounding below a total loss put back at -1. Weights can
    sum a rounding error above 1 (6/30 + 23/30 + 1/30 does), so a day on which every
    holding loses everything can sum to -1.0000000000000002, which the metrics would
    refuse on valid input. Anything further below is not rounding: it stays, for the
    metrics to refuse."""
    rounding = (returns < MIN_DAILY_RETURN) & (returns >= MIN_DAILY_RETURN - 2 * WEIGHT_SUM_TOLERANCE)
    return np.where(rounding, MIN_DAILY_RETURN, returns)


def _missing_values_error(frame: pd.DataFrame) -> ValueError:
    """Name every column with a missing value and its first dates."""
    parts = []
    for col in frame.columns:
        dates = frame.index[frame[col].isna().to_numpy()]
        if len(dates):
            shown = ", ".join(_day(d) for d in dates[:_MAX_DATES_LISTED])
            more = len(dates) - _MAX_DATES_LISTED
            parts.append(f"{col} on {shown}" + (f" (+{more} more)" if more > 0 else ""))
    return ValueError(
        "Missing returns (NaN / empty cells): " + "; ".join(parts) + ". A missing "
        "return is never treated as a 0% day: that would add fabricated flat days "
        "and understate volatility and VaR. Supply the real values, remove the "
        'column, or pass missing="drop" to exclude every date on which a held '
        "asset has no value."
    )


def _typical_move_floor(periods: int) -> float:
    """The median absolute return from which a column at ``periods`` observations a
    year stops reading as returns: ``PERCENT_MIN_TYPICAL_MOVE`` for a daily series,
    and larger with the square root of the time one observation spans, as volatility
    is: 0.22 weekly, 0.46 monthly, 1.59 annually. A faster series keeps the daily
    floor."""
    return PERCENT_MIN_TYPICAL_MOVE * math.sqrt(max(TRADING_DAYS / periods, 1.0))


def _way_past(periods: int) -> str:
    """How to say these really are returns. Only a series read as daily can be
    refused for being at another rate."""
    rate = "state their rate with `periods_per_year` if they are not daily, or " if periods == TRADING_DAYS else ""
    return f"If these really are fractions, {rate}pass allow_large_returns=True."


def _refuse_percent_column(col, values: np.ndarray, periods: int = TRADING_DAYS) -> None:
    """``ValueError`` when a held column reads as percent, not fractions: at
    least ``PERCENT_MIN_OBSERVATIONS`` values, both up and down moves, and a
    median absolute return in [:func:`_typical_move_floor`,
    ``PERCENT_MAX_TYPICAL_MOVE``). The median ignores a few wild days, so a
    genuinely volatile asset passes (a normal 10% daily volatility has a
    median move of 6.7%); only every day being wild does not. A window that
    really is that violent passes with ``allow_large_returns=True``.

    The floor follows the series' rate: at a daily floor, 34 annual returns of an
    equity index would read as percent, and the advice to divide by 100 would shrink
    real returns 100-fold.

    The limits: a percent column whose typical move is below the floor (a cash-like
    fund moving under 0.1% a day) is indistinguishable from a quiet fraction
    column and is not caught. At one or two observations a year the floor passes
    ``PERCENT_MAX_TYPICAL_MOVE``, so this never refuses: a percent column there almost
    always has a move below -1, a range error, or reads above 100% volatility, which
    the API flags (``implausible_statistics``)."""
    if len(values) < PERCENT_MIN_OBSERVATIONS or not values.min() < 0 < values.max():
        return
    typical = float(np.median(np.abs(values)))
    if _typical_move_floor(periods) <= typical < PERCENT_MAX_TYPICAL_MOVE:
        one = "day" if periods >= TRADING_DAYS else "observation"
        raise ValueError(
            f"Column {col!r} looks like percent, not fractions: its typical {one} moves "
            f"{typical:.3g} (median absolute return), i.e. {typical:.1%} as a fraction, "
            f"from {len(values)} observations. Returns are fractions (-0.02 for -2%), "
            f"so divide the column by 100. {_way_past(periods)}"
        )


def _refuse_levels_column(col, values: np.ndarray, periods: int = TRADING_DAYS) -> None:
    """``ValueError`` when a held column reads as levels, not returns: at least
    ``LEVELS_MIN_OBSERVATIONS`` values, at least ``LEVELS_MIN_UP_SHARE`` of them
    above zero, and a median of :func:`_typical_move_floor` or more, capped at
    ``LEVELS_MAX_TYPICAL_MOVE``. A gross return series (``1.01, 0.99``) or prices
    under 10 pass the range check and then compound to a CAGR of 1e75 or more. An
    asset up 10% a day, nearly every day, is not one anyone holds, while six up years
    of 10-20% are ordinary; a window that really is that violent passes with
    ``allow_large_returns=True``.

    The limit: prices under 0.10 are not caught; the API flags what they produce
    (``implausible_statistics``)."""
    if len(values) < LEVELS_MIN_OBSERVATIONS:
        return
    up_share = float((values > 0).mean())
    typical = float(np.median(np.abs(values)))
    if up_share >= LEVELS_MIN_UP_SHARE and typical >= min(_typical_move_floor(periods), LEVELS_MAX_TYPICAL_MOVE):
        one = "day" if periods >= TRADING_DAYS else "observation"
        raise ValueError(
            f"Column {col!r} looks like levels, not returns: {up_share:.0%} of its "
            f"{len(values)} values are above zero and the median is {typical:.3g}, a gain of "
            f"{typical:.0%} nearly every {one}. Returns are fractions (-0.02 for -2%): for "
            f"gross returns (1 + r) subtract 1, and for prices take the percentage change. "
            f"{_way_past(periods)}"
        )


def _refuse_non_returns(col, values: np.ndarray, periods: int = TRADING_DAYS) -> None:
    """Refuse a column that reads as percent or as levels at ``periods`` observations
    a year. The checks are heuristics; ``allow_large_returns=True`` is the way past
    both."""
    _refuse_percent_column(col, values, periods)
    _refuse_levels_column(col, values, periods)


def _rate_of(index: pd.DatetimeIndex) -> int:
    """The rate the dates show, or daily when they show none the metrics could use
    (those refuse such dates themselves)."""
    from riskcanvas.metrics.risk import periods_per_year

    try:
        return periods_per_year(index)
    except ValueError:
        return TRADING_DAYS


def _holds_booleans(column: pd.Series) -> bool:
    """A bool column, or an object column with a bool in it: ``pd.to_numeric`` reads
    ``True`` as 1.0, a +100% day."""
    if pd.api.types.is_bool_dtype(column):
        return True
    return column.dtype == object and column.map(lambda v: isinstance(v, (bool, np.bool_))).any()


def _day(ts) -> str:
    return ts.date().isoformat() if hasattr(ts, "date") else str(ts)


def _gap_advice(days: int) -> str:
    return (
        "A gap is never filled in. If it is a market closure, pass "
        f"`max_gap_days={days}` to accept it: the return across it then spans the whole "
        "closure, and API responses flag it. If it is a halt or missing data, choose a "
        "window without it."
    )


def _require_columns(weights: Mapping[str, float], columns: pd.Index) -> None:
    absent = [t for t in weights if t not in columns]
    if absent:
        raise ValueError(
            f"weights name {absent}, which have no returns column (columns: "
            f"{list(columns)}). A weighted ticker is never dropped silently."
        )


def _check_held_returns(returns: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    """The returns of the weighted tickers, refusing anything that would make
    the weighted sum skip data: a weighted ticker with no column, or a missing
    value in a held column."""
    _require_columns(weights, returns.columns)
    held = returns[list(weights)]
    if held.isna().to_numpy().any():
        raise _missing_values_error(held)
    return held


@dataclass
class Portfolio:
    """
    A portfolio: historical prices, weights, returns, and metadata.

    Parameters
    ----------
    symbols : list[str]
        Asset tickers.
    weights : dict[str, float]
        Asset weights, normally summing to 1.0 (the loaders normalise them;
        ``normalize_weights=False`` keeps a partially invested portfolio).
    prices : pd.DataFrame
        OHLCV or close prices, DatetimeIndex, columns = symbols.
    returns : pd.DataFrame
        Daily simple returns.
    name : str
        Portfolio name label.

    Raises
    ------
    ValueError
        At construction: invalid weights (see :func:`check_weights`), a
        weighted ticker that is not in ``symbols`` or has no ``returns``
        column, or a missing value in a weighted ``returns`` column.
    """

    symbols: list[str]
    weights: dict[str, float]
    prices: pd.DataFrame
    returns: pd.DataFrame
    name: str = "Portfolio"

    def __post_init__(self) -> None:
        check_weights(self.weights, normalize=False)
        unlisted = [t for t in self.weights if t not in self.symbols]
        if unlisted:
            raise ValueError(f"weights name {unlisted}, which are not in symbols {self.symbols}.")
        _check_held_returns(self.returns, self.weights)

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    @property
    def portfolio_returns(self) -> pd.Series:
        """Weighted portfolio daily returns, ``sum_i w_i * r_i`` each day: the
        portfolio is rebalanced to its fixed weights daily (buy-and-hold
        weight drift is not modelled).

        Raises ``ValueError`` rather than skip data: a missing value in a
        held series is never summed as a 0% day."""
        held = _check_held_returns(self.returns, self.weights)
        w = pd.Series(self.weights, dtype=float)
        summed = (held * w.values).sum(axis=1)
        return pd.Series(floor_at_total_loss(summed.to_numpy()), index=summed.index)

    @property
    def equity_curve(self) -> pd.Series:
        """Cumulative portfolio value (base=1.0)."""
        return (1 + self.portfolio_returns).cumprod()

    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Pearson correlation of the asset returns. NaN wherever a series has
        zero variance (cash, any constant series): the correlation is
        undefined there, including on that series' own diagonal."""
        return self.returns.corr()

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        # Annualised with the series' own periods per year (see periods_per_year).
        from riskcanvas.metrics.risk import periods_per_year

        return self.returns.cov() * periods_per_year(self.returns.index)

    def __repr__(self) -> str:
        return (
            f"Portfolio(name={self.name!r}, assets={self.n_assets}, "
            f"period={self.prices.index[0].date()} → {self.prices.index[-1].date()})"
        )


class PortfolioLoader:
    """
    Loads portfolio data from various sources.

    Examples
    --------
    >>> loader = PortfolioLoader()
    >>> portfolio = loader.from_tickers(
    ...     {"AAPL": 0.30, "MSFT": 0.30, "GOOGL": 0.20, "AMZN": 0.20},
    ...     start="2020-01-01", end="2023-12-31"
    ... )
    """

    def from_tickers(
        self,
        weights: dict[str, float],
        start: str = "2020-01-01",
        end: str | None = None,
        name: str = "Portfolio",
        normalize_weights: bool = True,
        max_gap_days: int | None = None,
    ) -> Portfolio:
        """
        Build portfolio by fetching prices from yfinance.

        Every requested ticker is kept, and no price is ever invented:

        * A ticker with no price history in the window (a typo, a delisted
          or not-yet-listed symbol, a failed download) raises ``ValueError``
          naming it. It is never dropped with the weights rescaled.
        * A ticker whose history starts more than ``MAX_EDGE_GAP_DAYS``
          calendar days after ``start`` (listed after it), or ends more than
          that before ``end`` (before today, if ``end`` is later or omitted),
          i.e. delisted inside the window, raises ``ValueError`` naming it
          and its first / last available date, even when it is the only
          ticker. Back- or forward-filling would add fabricated 0% days,
          which understate volatility and VaR.
        * A gap of more than ``MAX_CLOSURE_GAP_DAYS`` calendar days between
          consecutive prices of a ticker, or between consecutive sessions
          common to every ticker, raises ``ValueError`` naming the ticker and
          dates and the ``max_gap_days`` that accepts it: one "daily" return
          across it would hide months of moves in a single day. Exchange
          closures up to that length (Golden Week, Lunar New Year) pass.
        * Prices are then aligned on the sessions every ticker traded (an
          inner join; nothing is filled), so a return across a non-common
          session spans both days. Weekend-trading assets and exchange
          holidays are absorbed this way.
        * Same-date closes are paired as if simultaneous, though Tokyo's close
          comes about 14 hours before New York's: what New York does after it
          reaches Tokyo the next day. Correlations and betas across regions read
          low, so diversification reads high and volatility, VaR and stress losses
          read low. The API warns ``non_synchronous_closes`` (:func:`close_region`).

        There is no FX conversion: each ticker's prices are used in its own
        listing currency, so a yen and a dollar ticker blend returns without
        the USD/JPY move. Convert first and use :meth:`from_returns_df` for a
        cross-currency portfolio.

        Parameters
        ----------
        weights : dict[str, float]
            {symbol: weight}, validated by :func:`check_weights` (long-only:
            finite, >= 0, positive finite sum; ``ValueError`` otherwise).
            Normalized to sum=1 if normalize_weights=True.
        start, end : str
            Date range for price history (``end`` exclusive, as in yfinance).
        name : str
            Portfolio label.
        normalize_weights : bool
            Normalize weights to sum=1 (default True).
        max_gap_days : int | None
            The longest gap to accept, in calendar days, for an exceptional
            closure: inside a history, between common sessions, and at either
            end of the window. The return across a gap spans all of it. Default:
            ``MAX_CLOSURE_GAP_DAYS`` inside, ``MAX_EDGE_GAP_DAYS`` at the ends,
            so that a late listing does not read as a closure.

        Returns
        -------
        Portfolio

        Raises
        ------
        ValueError
            Invalid weights; a ticker with no data; histories that do not
            cover a common window; fewer than two common returns; a
            non-positive price in the source data; ``max_gap_days`` that is not
            a whole number >= 1.
        """
        import yfinance as yf

        weights = check_weights(weights, normalize=normalize_weights)
        if max_gap_days is not None and (int(max_gap_days) != max_gap_days or max_gap_days < 1):
            raise ValueError(f"max_gap_days must be a whole number >= 1 (got {max_gap_days!r}).")
        edge_limit = MAX_EDGE_GAP_DAYS if max_gap_days is None else int(max_gap_days)
        gap_limit = MAX_CLOSURE_GAP_DAYS if max_gap_days is None else int(max_gap_days)

        symbols = list(weights.keys())
        window = f"{start} to {end or 'today'}"
        logger.info(f"Fetching prices for {symbols} from {window}")

        closes: dict[str, pd.Series] = {}
        missing: dict[str, str] = {}
        for sym in symbols:
            try:
                df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            except Exception as e:  # yfinance raises bare network/HTTP/parse errors
                missing[sym] = f"download failed ({type(e).__name__})"
                continue
            close = df["Close"] if "Close" in df else pd.Series(dtype=float)
            if isinstance(close, pd.DataFrame):  # yfinance's (field, ticker) columns
                close = close.iloc[:, 0]
            close = close.dropna()
            if close.empty:
                missing[sym] = "no prices returned"
            else:
                closes[sym] = close

        if missing:
            raise ValueError(
                f"No usable price history in {window} for "
                + ", ".join(f"{s} ({why})" for s, why in missing.items())
                + ". Check the symbol: unknown, delisted, or not trading in that "
                "window (or the price source is unreachable). A ticker is never "
                "dropped silently; remove it from `weights` or change the dates."
            )

        first = {s: c.index[0] for s, c in closes.items()}
        last = {s: c.index[-1] for s, c in closes.items()}
        start_ts = pd.Timestamp(start)
        # The last session the window can hold: the day before the exclusive
        # `end`, or today when `end` lies in the future or is omitted. The latest
        # price among the tickers would not do: a sole delisted ticker always
        # satisfies it, with a window ending years ago.
        today = pd.Timestamp.today().normalize()
        if end is not None:
            last_needed = min(pd.Timestamp(end) - pd.Timedelta(days=1), today)
            end_rule = "`end`"
        else:
            last_needed = today
            end_rule = "today (`end` was omitted)"
        late = {s: d for s, d in first.items() if (d - start_ts).days > edge_limit}
        ends_early = {s: d for s, d in last.items() if (last_needed - d).days > edge_limit}
        if late or ends_early:
            problems = []
            if late:
                problems.append(
                    f"history starts more than {edge_limit} calendar days after "
                    f"`start` ({start_ts.date()}) for "
                    + ", ".join(f"{s} (first available {d.date()})" for s, d in late.items())
                    + f"; set `start` on or after {max(late.values()).date()}"
                )
            if ends_early:
                cut = min(ends_early.values()) + pd.Timedelta(days=1)
                problems.append(
                    f"history ends more than {edge_limit} calendar days before "
                    f"{end_rule} ({last_needed.date()}) for "
                    + ", ".join(f"{s} (last available {d.date()})" for s, d in ends_early.items())
                    + f"; set `end` on or before {cut.date()} (`end` is exclusive)"
                )
            raise ValueError(
                f"Price histories in {window} do not cover a common window: "
                + "; ".join(problems)
                + ". Missing prices are never filled in: a filled price adds "
                "fabricated 0% days, which understate volatility and VaR. If the "
                "window starts or ends inside a market closure, move it or pass "
                "`max_gap_days`."
            )

        holes = []
        longest = 0
        for sym in symbols:
            idx = closes[sym].index
            for i in np.flatnonzero((idx[1:] - idx[:-1]).days > gap_limit):
                a, b = idx[i], idx[i + 1]
                longest = max(longest, (b - a).days)
                holes.append(f"{sym} has no prices between {a.date()} and {b.date()} "
                             f"({(b - a).days} calendar days)")
        if holes:
            raise ValueError(
                f"Price histories in {window} have gaps longer than {gap_limit} "
                "calendar days: " + "; ".join(holes) + ". " + _gap_advice(longest)
            )

        # Inner join on the sessions every ticker traded; nothing is filled.
        prices = pd.DataFrame(closes)[symbols].dropna(how="any")
        common = prices.index
        joint = np.flatnonzero((common[1:] - common[:-1]).days > gap_limit)
        if len(joint):
            a, b = common[joint[0]], common[joint[0] + 1]
            longest = int((common[joint + 1] - common[joint]).days.max())
            raise ValueError(
                f"No session common to every ticker between {a.date()} and {b.date()} "
                f"({(b - a).days} calendar days; the tickers trade on different days). "
                + _gap_advice(longest)
            )
        if (prices <= 0).any().any():
            bad_syms = [s for s in symbols if (prices[s] <= 0).any()]
            raise ValueError(f"The price source returned non-positive prices for {bad_syms}.")
        returns = prices.pct_change().iloc[1:]
        if len(returns) < 2:
            raise ValueError(
                f"Only {len(prices)} sessions in {window} have a price for "
                f"every ticker; at least 3 (2 daily returns) are needed."
            )

        return Portfolio(
            symbols=symbols,
            weights=weights,
            prices=prices,
            returns=returns,
            name=name,
        )

    def from_returns_df(
        self,
        returns: pd.DataFrame,
        weights: dict[str, float] | None = None,
        name: str = "Portfolio",
        normalize_weights: bool = True,
        missing: str = "raise",
        allow_large_returns: bool = False,
        periods_per_year: int | None = None,
    ) -> Portfolio:
        """
        Build a portfolio from a pre-computed daily returns DataFrame (offline).

        The same contract as the HTTP API's ``returns`` mode; nothing is
        filled, dropped or rescaled behind your back:

        * ``weights`` pass :func:`check_weights` (long-only, finite, positive
          finite sum) and are normalised by their sum, the only
          transformation. Every weighted ticker must be a column (a missing
          one is a ``ValueError`` naming it, never dropped). The portfolio
          holds exactly the weighted tickers; other columns are ignored.
        * The index is a ``DatetimeIndex``, strictly increasing (no
          duplicate dates).
        * Every held value is a finite fraction in [-1, 10] (``-0.02`` for
          -2%), never a true/false. A missing value (NaN) anywhere in a held series, leading,
          interior or trailing, is a ``ValueError`` naming the column and
          dates: summing it as a 0% day would understate volatility and VaR.
          ``missing="drop"`` is the explicit alternative: every date on which
          any held asset has no value is excluded (for returns nothing spans
          the gap, so those days are simply absent from every statistic). It
          does not compound, so it cannot align assets on different calendars:
          use :meth:`from_tickers`, or compound onto common dates first.
        * At least 2 observations remain.
        * A column that reads as percent is refused (see
          :func:`_refuse_percent_column`; a typical day of 10-100% over 20 or
          more observations with both up and down days), and so is one that
          reads as levels, gross returns or prices (see
          :func:`_refuse_levels_column`; a median of 10% or more with at least
          95% of 5 or more observations above zero). A series that is not daily
          has larger floors, by the square root of its rate (46% for monthly).

        Parameters
        ----------
        returns : pd.DataFrame
            Daily simple returns, DatetimeIndex, one column per asset.
        weights : dict[str, float] | None
            {ticker: weight}. Defaults to equal weights over every column.
        name : str
            Portfolio label.
        normalize_weights : bool
            Normalize weights to sum=1 (default True), as ``from_tickers``.
        missing : {"raise", "drop"}
            Missing-value policy (default ``"raise"``).
        allow_large_returns : bool
            Accept a column that reads as percent or as levels, for a window
            that really is that violent (default False).
        periods_per_year : int | None
            Observations per year, for those two checks. Default: read from the
            dates, as the metrics do.

        Raises
        ------
        ValueError
            Any violation above, with a message naming the column / dates.
        """
        if missing not in MISSING_POLICIES:
            raise ValueError(f"missing must be one of {MISSING_POLICIES}; got {missing!r}.")
        if periods_per_year is not None and periods_per_year <= 0:
            raise ValueError(f"periods_per_year must be positive (got {periods_per_year!r}).")
        if not isinstance(returns, pd.DataFrame):
            raise ValueError(f"returns must be a pandas DataFrame; got {type(returns).__name__}.")
        if not isinstance(returns.index, pd.DatetimeIndex):
            raise ValueError(
                "returns must have a DatetimeIndex (one row per trading day); got "
                f"{type(returns.index).__name__}."
            )
        if not (returns.index.is_monotonic_increasing and returns.index.is_unique):
            pos = next(
                i for i in range(1, len(returns.index))
                if not returns.index[i] > returns.index[i - 1]
            )
            raise ValueError(
                "returns dates must be strictly increasing (sorted, no duplicates); "
                f"{_day(returns.index[pos])} follows {_day(returns.index[pos - 1])}."
            )
        if not returns.columns.is_unique:
            dupes = sorted({str(c) for c in returns.columns[returns.columns.duplicated()]})
            raise ValueError(f"returns has duplicate columns: {dupes}.")

        if weights is None:
            weights = {s: 1.0 for s in returns.columns}
        weights = check_weights(weights, normalize=normalize_weights)
        _require_columns(weights, returns.columns)
        symbols = list(weights)

        held = {}
        for col in symbols:
            if _holds_booleans(returns[col]):
                raise ValueError(
                    f"returns column {col!r} is not numeric: it holds true/false values, "
                    "which would count as +100% and 0% days."
                )
            try:
                held[col] = pd.to_numeric(returns[col], errors="raise").astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"returns column {col!r} is not numeric: {exc}") from None
        frame = pd.DataFrame(held, index=returns.index)

        if missing == "drop":
            keep = frame.notna().all(axis=1)
            if not keep.all():
                logger.info("missing='drop': excluded %d of %d dates", int((~keep).sum()), len(frame))
            frame = frame[keep]
        elif frame.isna().to_numpy().any():
            raise _missing_values_error(frame)

        rate = _rate_of(frame.index) if periods_per_year is None else periods_per_year
        for col in symbols:
            values = frame[col].to_numpy()
            bad = ~((values >= MIN_DAILY_RETURN) & (values <= MAX_DAILY_RETURN))
            if bad.any():
                i = int(bad.argmax())
                raise ValueError(
                    f"Daily returns must be finite and lie in [{MIN_DAILY_RETURN}, "
                    f"{MAX_DAILY_RETURN}] (fractions, e.g. -0.02 for -2%); column "
                    f"{col!r} has {values[i]!r} on {_day(frame.index[i])}."
                )
            if not allow_large_returns:
                _refuse_non_returns(col, values, rate)
        if len(frame) < MIN_RETURNS_LEN:
            raise ValueError(
                f"returns must contain at least {MIN_RETURNS_LEN} observations "
                f"(got {len(frame)})."
            )

        # Synthetic prices from returns (base=100)
        prices = 100 * (1 + frame).cumprod()

        return Portfolio(
            symbols=symbols,
            weights=weights,
            prices=prices,
            returns=frame,
            name=name,
        )

    def from_csv(
        self,
        path: str | Path,
        weights: dict[str, float] | None = None,
        name: str = "Portfolio",
        date_col: str = "Date",
        return_cols: list[str] | None = None,
        normalize_weights: bool = True,
        missing: str = "raise",
        allow_large_returns: bool = False,
    ) -> Portfolio:
        """
        Load portfolio returns from a CSV file (offline).

        Parsed, then built with :meth:`from_returns_df`, so the same rules
        apply: an empty cell is a missing return (a ``ValueError`` naming the
        column and dates unless ``missing="drop"``), weights are validated
        and normalised, every weighted ticker must be a column, dates must be
        strictly increasing.

        Parameters
        ----------
        path : str | Path
            Path to CSV: a date column plus one column of daily returns
            (fractions) per asset.
        weights : dict[str, float] | None
            Asset weights. Defaults to equal weight over the return columns.
        name : str
            Portfolio label.
        date_col : str
            Name of the date column (required; a CSV without it is an error,
            never dated from row numbers).
        return_cols : list[str] | None
            Columns to use. If None, uses all non-date columns.
        normalize_weights : bool
            Normalize weights to sum=1 (default True).
        missing : {"raise", "drop"}
            Missing-value policy, as in :meth:`from_returns_df`.
        allow_large_returns : bool
            Accept a column that reads as percent or as levels, as in
            :meth:`from_returns_df`; the dates set the rate those checks use.
        """
        df = pd.read_csv(path)
        if date_col not in df.columns:
            raise ValueError(
                f"{path}: no date column {date_col!r} (columns: {list(df.columns)}). "
                "Pass date_col= to name it."
            )
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        if return_cols is not None:
            absent = [c for c in return_cols if c not in df.columns]
            if absent:
                raise ValueError(f"{path}: return_cols {absent} are not columns of the file.")
            df = df[list(return_cols)]

        return self.from_returns_df(
            df, weights=weights, name=name,
            normalize_weights=normalize_weights, missing=missing,
            allow_large_returns=allow_large_returns,
        )
