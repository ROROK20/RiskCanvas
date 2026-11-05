"""
RiskCanvas: portfolio risk analytics, built so you can check the numbers by hand.

Computes VaR, CVaR, Sharpe, Sortino, Calmar, Max Drawdown, Monte Carlo
simulations, and crisis stress scenarios for a long-only portfolio held in one
currency (prices and returns are never converted between currencies).

Stress scenarios: what they are, exactly
----------------------------------------
The six built-in crisis scenarios (Black Monday 1987, Dot-com 2000-2002, GFC
2008-2009, European Debt Crisis 2011, COVID-19 2020, Fed Rate Hike Bear 2022)
are **deterministic, calibrated SYNTHETIC daily-return paths** — each scaled by
a single factor so that its compounded maximum drawdown matches that crisis's
documented S&P 500 peak-to-trough drawdown.

They are **not replays of historical prices**, and no market data is fetched to
build them. Daily values are stylized to each crisis's character (crash days,
relief rallies, grinds); only the calibrated drawdown is a real, documented
figure. The paths never depend on the portfolio being analyzed — only the beta
scaling applied to them does. This is deliberate: it makes every stress number
reproducible rather than dependent on which exact dates a run happens to
sample. See :mod:`riskcanvas.scenarios.stress_test` for the paths and the
calibration routine.
"""

__version__ = "0.1.0"
__author__ = "Rohan Khiani"
__license__ = "MIT"
