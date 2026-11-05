/**
 * Typed client for the RiskCanvas FastAPI engine.
 *
 * Reads NEXT_PUBLIC_API_URL (inlined at build; unset means http://localhost:8004
 * under `next dev` and NO engine in a production build, see lib/apiUrl.mjs) and calls the
 * REAL Python engine. Every call is wrapped so the caller can fall back to the
 * in-browser mock module (lib/mockRisk.ts) on ANY error — meaning the app still
 * builds and renders with no backend running.
 *
 * The frontend generates SYNTHETIC per-asset daily returns (lib/mockRisk.ts —
 * not market data) and POSTs them, so the engine runs FULLY OFFLINE (no
 * yfinance / no network on the backend): real numpy / scipy math on synthetic
 * inputs.
 */

import { resolveApiUrl } from "./apiUrl.mjs";
import { fetchEngineJson } from "./engineRequest.mjs";

export const API_URL: string | null = resolveApiUrl(process.env.NEXT_PUBLIC_API_URL, process.env.NODE_ENV);

function engineUrl(path: string): string {
  // No engine configured (a production build without NEXT_PUBLIC_API_URL): the
  // page does not call in that case; fail before any network call if anything does.
  if (API_URL === null) throw new Error("No engine configured for this build (NEXT_PUBLIC_API_URL unset)");
  return `${API_URL}${path}`;
}

// ---- Wire types (mirror riskcanvas/api/app.py response shapes) -------------

export interface PortfolioRequestBody {
  // Long-only: finite, >= 0, positive sum; one `returns` series per ticker.
  weights: Record<string, number>;
  name?: string;
  returns?: Record<string, number[]>;
  // S&P 500 proxy series aligned with `returns`; drives the stress beta.
  market_returns?: number[];
  beta?: number;
}

// A reason a result may not represent the portfolio (thin history, or a
// stress scenario that cannot describe it). Empty list when none apply.
export interface ApiWarning {
  code: string;
  message: string;
}

// Every portfolio endpoint reports the data window it actually used.
interface DataWindow {
  period: string; // "first price date → last price date"
  n_observations: number; // daily returns
  warnings: ApiWarning[];
}

export interface AnalyzeResponse extends DataWindow {
  portfolio: string;
  n_assets: number;
  risk: {
    var_95_1d: number;
    var_99_1d: number;
    var_95_10d: number;
    cvar_95: number;
    cvar_99: number;
    max_drawdown: number;
    annual_vol: number;
    daily_vol: number;
    // null when undefined (a zero-variance series).
    skewness: number | null;
    excess_kurtosis: number | null;
    [k: string]: number | null;
  };
  performance: {
    sharpe: number | null; // null at zero volatility
    // null when unbounded (no downside / no drawdown / no losing day).
    sortino: number | null;
    calmar: number | null;
    cagr: number;
    [k: string]: number | null;
  };
  // null where the correlation is undefined: a zero-variance series (a cash
  // leg, any constant series) correlates with nothing, itself included.
  correlation_matrix: Record<string, Record<string, number | null>>;
}

export interface VarResponse extends DataWindow {
  portfolio_value: number;
  var_report: Record<
    string,
    // Currency units at full precision (unrounded), positive = loss. *_1d:
    // historical 1-day; var_10d: var_1d * sqrt(10); *_parametric_1d: normal
    // closed form, 1-day. var_10d and the parametric figures are capped at
    // portfolio_value (a total loss).
    {
      var_1d: number;
      var_10d: number;
      cvar_1d: number;
      var_parametric_1d: number;
      cvar_parametric_1d: number;
    }
  >;
}

export interface StressResponse extends DataWindow {
  portfolio: string;
  beta_used: number;
  beta_source: "explicit" | "market_returns" | "spy_leg" | "yfinance_spy" | "assumed";
  beta_note: string;
  // R^2 of the beta fit; null when beta was supplied or assumed.
  beta_r_squared: number | null;
  // warnings: low_equity_beta, weak_equity_fit, assumed_beta (the scenarios
  // are equity-only), non_synchronous_closes, plus short_history.
  scenarios: {
    name: string;
    portfolio_drawdown: number;
    market_drawdown: number;
    duration_days: number; // trading days (the scenario path length)
    total_loss: number;
    // null when the engine cannot estimate a recovery time (portfolio mean
    // daily return <= 0, or no drawdown) — never a sentinel number.
    recovery_days_estimate: number | null;
  }[];
}

export interface MonteCarloResponse extends DataWindow {
  n_paths: number;
  horizon_days: number;
  method: string;
  seed: number;
  // Mean daily return simulated: "zero" (default), "historical", "risk_free".
  drift: "zero" | "historical" | "risk_free";
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  // Monte Carlo sampling error of each estimate (null where undefined):
  // simulation noise only, not estimation error (see standard_errors_note).
  standard_errors_note: string;
  standard_errors: Record<
    "var_95" | "var_99" | "cvar_95" | "cvar_99" | "prob_loss" | "expected_terminal",
    number | null
  >;
  prob_loss: number;
  expected_terminal: number;
  // Terminal-value percentiles (despite the name, not paths).
  percentile_paths: { p5: number; p25: number; p50: number; p75: number; p95: number };
  // Percentiles of simulated value at checkpoint days 0..horizon.
  percentile_bands: { day: number; p5: number; p25: number; p50: number; p75: number; p95: number }[];
  percentile_table: { percentile: number; terminal_value: number; return: number }[];
}

// ---- low-level fetch helpers ----------------------------------------------

const TIMEOUT_MS = 6000;

async function postJSON<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return fetchEngineJson(
    engineUrl(path),
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    { timeoutMs: TIMEOUT_MS, signal },
  );
}

async function getJSON<T>(path: string): Promise<T> {
  return fetchEngineJson(engineUrl(path), {}, { timeoutMs: TIMEOUT_MS });
}

// ---- public client --------------------------------------------------------

export async function checkHealth(): Promise<boolean> {
  try {
    const j = await getJSON<{ status: string }>("/health");
    return j.status === "ok";
  } catch {
    return false;
  }
}

export const api = {
  analyze: (body: PortfolioRequestBody, signal?: AbortSignal) =>
    postJSON<AnalyzeResponse>("/portfolio/analyze", body, signal),
  var: (body: PortfolioRequestBody, signal?: AbortSignal) =>
    postJSON<VarResponse>("/portfolio/var", body, signal),
  stress: (body: PortfolioRequestBody, signal?: AbortSignal) =>
    postJSON<StressResponse>("/portfolio/stress", body, signal),
  montecarlo: (
    body: PortfolioRequestBody,
    params: { n_paths: number; horizon_days: number; method: "gbm" | "t" | "bootstrap" },
    signal?: AbortSignal,
  ) =>
    postJSON<MonteCarloResponse>(
      `/portfolio/montecarlo?n_paths=${params.n_paths}&horizon_days=${params.horizon_days}&method=${params.method}`,
      body,
      signal,
    ),
};
