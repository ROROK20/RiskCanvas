/**
 * Live analytics: drive the REAL Python engine and map its responses into the
 * existing `PortfolioAnalysis` shape that page.tsx already renders.
 *
 * Strategy: the frontend generates SYNTHETIC per-asset return series plus the
 * synthetic market series they load on (lib/mockRisk.ts — not market data)
 * and POSTs them (`returns`, `market_returns`). The engine runs fully offline
 * (no yfinance) and returns VaR/CVaR/stress/Monte-Carlo computed by
 * numpy/scipy — real math on synthetic inputs, including a stress beta
 * estimated from the market series. On ANY error the caller falls back to the
 * in-browser mock, so `npm run build` and a no-backend deploy still work.
 */

import {
  analyzePortfolio,
  generateAssetReturns,
  portfolioSeries,
  type PortfolioInput,
  type PortfolioAnalysis,
  type MonteCarloPath,
  type ReturnDistribution,
  type StressResult,
} from "./mockRisk";
import { api, type PortfolioRequestBody } from "./api";
import { histogram } from "./histogram.mjs";

const DESCRIPTIONS: Record<string, string> = {
  "Black Monday (Oct 1987)": "S&P 500 -20.5% on Oct 19, 1987; -33.5% peak to trough",
  "Dot-com Crash (2000-2002)": "Technology bubble burst",
  "Global Financial Crisis (2008-2009)": "Subprime mortgage crisis",
  "COVID-19 Crash (Feb-Mar 2020)": "-33.9% in 23 trading days",
  "Fed Rate Hike Bear Market (2022)": "2022 Fed tightening bear market (equity drawdown only)",
  "European Debt Crisis (2011)": "Greek/Euro sovereign crisis, US downgrade",
};

// What the dashboard asks the engine to simulate (the label shows it).
const MC_PARAMS = { n_paths: 5000, horizon_days: 252, method: "gbm" } as const;

/**
 * Run the full analysis against the live engine. Returns the same shape as
 * `analyzePortfolio`. Throws on any network/engine failure so the caller can
 * fall back to the mock; `signal` aborts every request of the call.
 */
export async function liveAnalyzePortfolio(
  input: PortfolioInput,
  signal?: AbortSignal,
): Promise<PortfolioAnalysis> {
  // Only held assets are sent: the engine requires exactly one series per
  // weighted ticker and rejects (never rewrites) anything else.
  const weights = Object.fromEntries(
    Object.entries(input.weights).filter(([, w]) => w > 0),
  );
  const symbols = Object.keys(weights);
  if (symbols.length === 0) throw new Error("No assets selected");

  const { assets: returns, market } = generateAssetReturns(symbols);
  const body: PortfolioRequestBody = {
    weights,
    name: "RiskCanvas Portfolio",
    returns,
    market_returns: market,
  };

  // Hit the real engine. Any rejection bubbles up to the caller's catch.
  // /portfolio/var is not called: every number the page shows comes from
  // /analyze, /stress and /montecarlo.
  const [analyze, stress, mc] = await Promise.all([
    api.analyze(body, signal),
    api.stress(body, signal),
    api.montecarlo(body, MC_PARAMS, signal),
  ]);

  // --- metrics (real risk + performance from the engine) ---
  const risk = analyze.risk;
  const perf = analyze.performance;
  const metrics = {
    var95_1d: risk.var_95_1d,
    var99_1d: risk.var_99_1d,
    var95_10d: risk.var_95_10d,
    cvar95: risk.cvar_95,
    cvar99: risk.cvar_99,
    maxDrawdown: risk.max_drawdown,
    annualVol: risk.annual_vol,
    sharpe: perf.sharpe,
    sortino: perf.sortino,
    calmar: perf.calmar,
    cagr: perf.cagr,
    skewness: risk.skewness,
    excessKurtosis: risk.excess_kurtosis,
  };

  // --- return distribution: rebuild a histogram from the local series but
  //     anchor the tail threshold to the engine's real VaR95. ---
  // portfolioSeries normalises the weights exactly as the engine does, so
  // this is the engine's input series.
  const flatSeries = portfolioSeries(weights, returns);
  const varDistribution: ReturnDistribution[] = histogram(flatSeries, 30, -risk.var_95_1d);

  // --- Monte Carlo fan: the engine's own percentiles of simulated value at
  //     checkpoint days (not an interpolation of terminal values). ---
  const monteCarloFan: MonteCarloPath[] = mc.percentile_bands.map((b) => ({
    day: b.day,
    p5: b.p5,
    p25: b.p25,
    p50: b.p50,
    p75: b.p75,
    p95: b.p95,
  }));

  // --- stress results straight from the engine ---
  const stressResults: StressResult[] = stress.scenarios.map((s) => ({
    name: s.name,
    description: DESCRIPTIONS[s.name] ?? "Crisis scenario (calibrated to documented drawdown)",
    marketDrawdown: s.market_drawdown,
    portfolioDrawdown: s.portfolio_drawdown,
    durationDays: s.duration_days,
    recoveryDays: s.recovery_days_estimate,
  }));

  // --- correlation matrix from the engine ---
  const correlationMatrix: { row: string; col: string; value: number | null }[] = [];
  const cm = analyze.correlation_matrix ?? {};
  for (const row of Object.keys(cm)) {
    for (const col of Object.keys(cm[row] ?? {})) {
      const v = cm[row][col];
      correlationMatrix.push({ row, col, value: v === null ? null : Number(v.toFixed(2)) });
    }
  }

  // --- equity curve: reconstruct from the portfolio return series (the engine
  //     uses the same series). Keeps the chart data real and consistent. ---
  const equityCurve: { day: number; equity: number; drawdown: number }[] = [];
  let eq = input.initialValue;
  let peak = eq;
  flatSeries.forEach((r, i) => {
    eq *= 1 + r;
    peak = Math.max(peak, eq);
    equityCurve.push({
      day: i,
      equity: Math.round(eq),
      drawdown: Number(((eq - peak) / peak).toFixed(4)),
    });
  });


  return {
    metrics,
    varDistribution,
    monteCarloFan,
    monteCarlo: { paths: mc.n_paths, horizonDays: mc.horizon_days, method: mc.method, drift: mc.drift },
    stressResults,
    stressBeta: {
      value: stress.beta_used,
      source: stress.beta_source,
      note: stress.beta_note,
      warnings: stress.warnings.map((w) => w.message),
    },
    correlationMatrix,
    equityCurve,
  };
}

/** Mock fallback re-export for symmetry. */
export { analyzePortfolio };
