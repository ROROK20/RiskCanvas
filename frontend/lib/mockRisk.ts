/**
 * Synthetic inputs + in-browser fallback analytics.
 *
 * `generateAssetReturns` produces the SYNTHETIC return series the dashboard
 * sends to the real engine. `analyzePortfolio` is the offline fallback that
 * computes approximate metrics from the same series in the browser when the
 * engine is unreachable. Nothing in this file is market data.
 */

import { histogram } from "./histogram.mjs";
import { rSquared, stressScenario, weakFitWarning } from "./stress.mjs";
import { simulateGbmFan } from "./fan.mjs";

export interface PortfolioInput {
  weights: Record<string, number>;
  initialValue: number;
}

export interface RiskMetrics {
  var95_1d: number;
  var99_1d: number;
  var95_10d: number;
  cvar95: number;
  cvar99: number;
  maxDrawdown: number;
  annualVol: number;
  sharpe: number | null;
  // null when the ratio is unbounded (no downside / no drawdown) — the
  // engine serialises those as JSON null.
  sortino: number | null;
  calmar: number | null;
  cagr: number;
  skewness: number | null;
  excessKurtosis: number | null;
}

export interface MonteCarloPath {
  day: number;
  p5: number;
  p25: number;
  p50: number;
  p75: number;
  p95: number;
}

export interface ReturnDistribution {
  bucket: string;
  lo: number;
  mid: number;
  hi: number;
  count: number;
  isTail: boolean;
}

export interface StressResult {
  name: string;
  description: string;
  marketDrawdown: number;
  portfolioDrawdown: number;
  // Trading days in the scenario path (peak close to trough close).
  durationDays: number;
  // null when no recovery time is estimable (non-positive growth, or the
  // window ends at or above the pre-crisis level) — mirrors the engine's `recovery_days_estimate: number | null`.
  recoveryDays: number | null;
}

// Where the stress beta came from. The first five mirror the engine's
// `beta_source`; `demo_factor_loading` is the in-browser fallback.
export type BetaSource =
  | "explicit"
  | "market_returns"
  | "spy_leg"
  | "yfinance_spy"
  | "assumed"
  | "demo_factor_loading";

export interface StressBeta {
  value: number;
  source: BetaSource;
  note: string;
  // Reasons the equity-only scenarios may not represent the portfolio.
  warnings: string[];
}

// Same threshold as the engine (riskcanvas/scenarios/stress_test.py
// LOW_EQUITY_BETA): below it an S&P 500 shock barely moves the portfolio.
const LOW_EQUITY_BETA = 0.3;

export interface MonteCarloInfo {
  paths: number;
  horizonDays: number;
  method: string;
  drift: string; // "zero" (default), "historical" or "risk_free"
}

export interface PortfolioAnalysis {
  metrics: RiskMetrics;
  varDistribution: ReturnDistribution[];
  monteCarloFan: MonteCarloPath[];
  monteCarlo: MonteCarloInfo;
  stressResults: StressResult[];
  stressBeta: StressBeta;
  // null where a correlation is undefined (a zero-variance series, e.g. cash).
  correlationMatrix: { row: string; col: string; value: number | null }[];
  equityCurve: { day: number; equity: number; drawdown: number }[];
}

/**
 * SYNTHETIC one-factor return model — NOT market data.
 *
 * Every asset's daily return is
 *     r_i,t = drift_i + beta_i * MARKET_VOL * m_t + idioVol_i * e_i,t
 * where m_t (one shared market shock) and e_i,t (per-asset) are seeded,
 * unit-variance Student-t(4) draws: fat-tailed, and correlated across
 * assets only through the market factor. SPY is the market itself
 * (beta 1, no idiosyncratic part), so `market` below equals SPY's series.
 * The parameters are rough, plausible magnitudes chosen for the demo; they
 * are not estimated from any real price history.
 */
const MARKET_DRIFT = 0.0004;
const MARKET_VOL = 0.011;
const T_DF = 4;
const RISK_FREE_DAILY = 0.02 / 252;

const ASSET_PROFILES: Record<string, { drift: number; beta: number; idioVol: number }> = {
  SPY: { drift: MARKET_DRIFT, beta: 1.0, idioVol: 0.0 },
  QQQ: { drift: 0.0006, beta: 1.15, idioVol: 0.006 },
  AAPL: { drift: 0.0007, beta: 1.2, idioVol: 0.0122 },
  MSFT: { drift: 0.0006, beta: 1.1, idioVol: 0.0105 },
  NVDA: { drift: 0.001, beta: 1.7, idioVol: 0.0235 },
  TSLA: { drift: 0.0005, beta: 1.9, idioVol: 0.0317 },
  AGG: { drift: 0.0001, beta: 0.05, idioVol: 0.004 },
  GLD: { drift: 0.0002, beta: 0.1, idioVol: 0.0099 },
  BTC: { drift: 0.0008, beta: 1.3, idioVol: 0.0427 },
};
const DEFAULT_PROFILE = { drift: 0.0003, beta: 1.0, idioVol: 0.01 };

export const AVAILABLE_ASSETS = Object.keys(ASSET_PROFILES);
export const SYNTHETIC_DAYS = 750;

export interface SyntheticReturns {
  assets: Record<string, number[]>;
  market: number[];
}

/**
 * Generate the synthetic per-asset daily return series (and the market
 * series they load on) for the supplied symbols. These are POSTed to the real
 * Python engine (`returns` / `market_returns`), which computes VaR / CVaR /
 * stress / Monte-Carlo from them with numpy/scipy — real math on synthetic
 * inputs. Deterministic: the same symbols always give the same series.
 */
export function generateAssetReturns(symbols: string[], nDays = SYNTHETIC_DAYS): SyntheticReturns {
  const marketRand = mulberry32(hash("__market__"));
  const shocks = Array.from({ length: nDays }, () => studentT(marketRand));
  const market = shocks.map((z) => MARKET_DRIFT + MARKET_VOL * z);
  const assets: Record<string, number[]> = {};
  for (const s of symbols) {
    const p = ASSET_PROFILES[s] ?? DEFAULT_PROFILE;
    const rand = mulberry32(hash(s));
    assets[s] = shocks.map((z) => p.drift + p.beta * MARKET_VOL * z + p.idioVol * studentT(rand));
  }
  return { assets, market };
}

/** Normalised long-only weights over the symbols with a positive weight. */
export function normalisedWeights(weights: Record<string, number>): Record<string, number> {
  const held = Object.entries(weights).filter(([, w]) => w > 0);
  const total = held.reduce((acc, [, w]) => acc + w, 0);
  return Object.fromEntries(held.map(([s, w]) => [s, w / total]));
}

/** Exact beta of the synthetic portfolio to the synthetic market factor. */
export function syntheticPortfolioBeta(weights: Record<string, number>): number {
  return Object.entries(normalisedWeights(weights)).reduce(
    (acc, [s, w]) => acc + w * (ASSET_PROFILES[s] ?? DEFAULT_PROFILE).beta,
    0,
  );
}

function gaussian(rand: () => number): number {
  // Box-Muller
  const u = 1 - rand();
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Unit-variance Student-t(T_DF) draw: z / sqrt(chi2/df), rescaled. */
function studentT(rand: () => number): number {
  let chi2 = 0;
  for (let k = 0; k < T_DF; k++) chi2 += gaussian(rand) ** 2;
  return (gaussian(rand) / Math.sqrt(chi2 / T_DF)) * Math.sqrt((T_DF - 2) / T_DF);
}

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hash(s: string): number {
  let h = 2166136261;
  for (const c of s) {
    h ^= c.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Weighted portfolio daily series built from the synthetic asset series. */
export function portfolioSeries(
  weights: Record<string, number>,
  assets: Record<string, number[]>,
): number[] {
  const w = normalisedWeights(weights);
  const symbols = Object.keys(w);
  const n = symbols.length ? assets[symbols[0]].length : 0;
  return Array.from({ length: n }, (_, i) =>
    symbols.reduce((acc, s) => acc + w[s] * assets[s][i], 0),
  );
}

// Scenario reference data, mirroring the engine's SCENARIOS (S&P 500
// close-to-close peak-to-trough, re-verified against ^GSPC 2026-10-30).
// Short demo names (the live view shows the engine's full names); each maps
// to the engine scenario whose calibrated daily path the demo compounds.
export const SCENARIO_REFERENCE: { key: string; name: string; description: string }[] = [
  { key: "black_monday_1987", name: "Black Monday 1987", description: "S&P 500 -20.5% on Oct 19, 1987; -33.5% peak to trough" },
  { key: "dotcom_crash", name: "Dot-com Crash", description: "Technology bubble burst" },
  { key: "gfc_2008", name: "Global Financial Crisis", description: "Subprime mortgage crisis" },
  { key: "covid_crash_2020", name: "COVID-19 Crash", description: "-33.9% in 23 trading days" },
  { key: "european_debt_crisis", name: "European Debt Crisis", description: "Greek/Euro sovereign crisis, US downgrade" },
  { key: "rate_hike_2022", name: "Fed Rate Hike Bear", description: "2022 Fed tightening bear market (equity drawdown only)" },
];

/** In-browser fallback analysis; null when the portfolio holds nothing (no
 * asset, or every weight 0), never a throw on an empty series. */
export function analyzePortfolio(input: PortfolioInput): PortfolioAnalysis | null {
  const w = normalisedWeights(input.weights);
  const symbols = Object.keys(w);
  if (symbols.length === 0) return null;
  const { assets, market } = generateAssetReturns(symbols);
  const returns = portfolioSeries(input.weights, assets);

  const meanRet = returns.reduce((a, b) => a + b, 0) / returns.length;
  const stdRet = Math.sqrt(returns.reduce((acc, r) => acc + (r - meanRet) ** 2, 0) / returns.length);

  // Equity curve / drawdown over the synthetic history.
  const equity: { day: number; equity: number; drawdown: number }[] = [];
  let eq = input.initialValue;
  let peak = eq;
  let worstDrawdown = 0;   // unrounded: CAGR / Calmar use exact values
  returns.forEach((r, i) => {
    eq *= 1 + r;
    peak = Math.max(peak, eq);
    worstDrawdown = Math.min(worstDrawdown, (eq - peak) / peak);
    // Rounded for the chart only.
    equity.push({ day: i, equity: Math.round(eq), drawdown: Number(((eq - peak) / peak).toFixed(4)) });
  });

  // Historical VaR / CVaR
  const sorted = [...returns].sort((a, b) => a - b);
  const var95 = -sorted[Math.floor(0.05 * sorted.length)];
  const var99 = -sorted[Math.floor(0.01 * sorted.length)];
  const tail95 = sorted.slice(0, Math.floor(0.05 * sorted.length));
  const tail99 = sorted.slice(0, Math.floor(0.01 * sorted.length));
  const cvar95 = -tail95.reduce((a, b) => a + b, 0) / Math.max(tail95.length, 1);
  const cvar99 = -tail99.reduce((a, b) => a + b, 0) / Math.max(tail99.length, 1);

  const maxDrawdown = worstDrawdown;
  const annualVol = stdRet * Math.sqrt(252);
  // From the unrounded final value (the chart's equity is rounded to dollars).
  const cagr = Math.pow(eq / input.initialValue, 252 / returns.length) - 1;
  const sharpe = ((meanRet - RISK_FREE_DAILY) / stdRet) * Math.sqrt(252);
  // Sortino: same definition as the engine (performance.py) — mean excess
  // over the daily risk-free target divided by the downside deviation
  // sqrt(mean(min(r - T, 0)^2)) taken over ALL observations.
  const downDev = Math.sqrt(
    returns.reduce((acc, r) => acc + Math.min(r - RISK_FREE_DAILY, 0) ** 2, 0) / returns.length,
  );
  const excessAnnual = (meanRet - RISK_FREE_DAILY) * 252;
  const sortino = downDev > 0 ? excessAnnual / (downDev * Math.sqrt(252)) : null;
  const calmar = Math.abs(maxDrawdown) > 1e-10 ? cagr / Math.abs(maxDrawdown) : null;

  const m3 = returns.reduce((acc, r) => acc + Math.pow(r - meanRet, 3), 0) / returns.length;
  const m4 = returns.reduce((acc, r) => acc + Math.pow(r - meanRet, 4), 0) / returns.length;
  const skewness = m3 / Math.pow(stdRet, 3);
  const excessKurtosis = m4 / Math.pow(stdRet, 4) - 3;

  // Histogram for VaR distribution (shared with the live view).
  const buckets: ReturnDistribution[] = histogram(returns, 30, -var95);

  // In-browser Monte Carlo fan: zero-drift geometric Brownian motion at the
  // series' volatility (lib/fan.mjs), the engine's default drift.
  const N_PATHS = 1000;
  const HORIZON = 252;
  const rand = mulberry32(hash(symbols.join(",")));
  const monteCarloFan: MonteCarloPath[] = simulateGbmFan(stdRet, N_PATHS, HORIZON, 7, () => gaussian(rand));

  // Stress: the synthetic portfolio's exact factor beta times the engine's
  // calibrated S&P 500 daily path, compounded (lib/stress.mjs), as the engine.
  const beta = syntheticPortfolioBeta(input.weights);
  const logGrowth = returns.reduce((acc, r) => acc + Math.log1p(r), 0) / returns.length;
  const stressResults: StressResult[] = SCENARIO_REFERENCE.map((sc) => {
    const path = stressScenario(sc.key, beta);
    const portfolioDrawdown = path.drawdown;
    return {
      name: sc.name,
      description: sc.description,
      marketDrawdown: path.marketDrawdown,
      portfolioDrawdown,
      durationDays: path.tradingDays,
      // From the end of the window, compounding at the series' geometric mean
      // daily growth, as the engine.
      recoveryDays:
        logGrowth > 0 && path.totalReturn < 0 && path.totalReturn > -1
          ? Math.floor(-Math.log1p(path.totalReturn) / logGrowth)
          : null,
    };
  }).sort((a, b) => a.portfolioDrawdown - b.portfolioDrawdown);

  // Sample correlation of the synthetic series.
  const correlationMatrix: { row: string; col: string; value: number | null }[] = [];
  const corr = (x: number[], y: number[]) => {
    const mx = x.reduce((a, b) => a + b, 0) / x.length;
    const my = y.reduce((a, b) => a + b, 0) / y.length;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < x.length; i++) {
      sxy += (x[i] - mx) * (y[i] - my);
      sxx += (x[i] - mx) ** 2;
      syy += (y[i] - my) ** 2;
    }
    // Undefined for a zero-variance series, as in the engine (null, not NaN).
    return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null;
  };
  symbols.forEach((a) => {
    symbols.forEach((b) => {
      const c = corr(assets[a], assets[b]);
      correlationMatrix.push({ row: a, col: b, value: c === null ? null : Number(c.toFixed(2)) });
    });
  });

  return {
    metrics: {
      var95_1d: var95,
      var99_1d: var99,
      var95_10d: var95 * Math.sqrt(10),
      cvar95,
      cvar99,
      maxDrawdown,
      annualVol,
      sharpe,
      sortino,
      calmar,
      cagr,
      skewness,
      excessKurtosis,
    },
    varDistribution: buckets,
    monteCarloFan,
    monteCarlo: { paths: N_PATHS, horizonDays: HORIZON, method: "in-browser GBM", drift: "zero" },
    stressResults,
    stressBeta: {
      value: beta,
      source: "demo_factor_loading",
      // The loading is what generated the series; the engine estimates beta by
      // OLS on the sampled series, which fat tails can move (100% BTC: 1.30
      // here, 1.08 from the engine). The fit warning is the engine's.
      note: "In-browser demo: the synthetic portfolio's exact factor loading (weighted asset betas), not the engine's OLS estimate from the same series, which can differ.",
      warnings: [
        ...(beta < LOW_EQUITY_BETA
          ? [
              `Beta to the S&P 500 is ${beta.toFixed(2)}: the equity-only scenarios barely move this portfolio and do not capture its rate or credit risk.`,
            ]
          : []),
        ...[weakFitWarning(rSquared(returns, market))].filter((w): w is string => w !== null),
      ],
    },
    correlationMatrix,
    equityCurve: equity,
  };
}
