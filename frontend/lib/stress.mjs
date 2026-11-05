/**
 * The in-browser demo's stress scenarios, computed the way the engine does
 * (riskcanvas/scenarios/stress_test.py): the portfolio's path is beta times
 * the calibrated S&P 500 daily path, compounded daily, and its drawdown is
 * measured from pre-crisis capital. Multiplying the scenario's drawdown by beta
 * instead is not what the panel says and gives losses beyond 100% (100% TSLA,
 * beta 1.9: GFC -107.92%).
 */
import { SCENARIO_PATHS } from "./scenarioPaths.mjs";

/**
 * @param {string} key engine scenario key, e.g. "gfc_2008"
 * @param {number} beta portfolio beta to the S&P 500
 * @returns {{ drawdown: number, totalReturn: number, marketDrawdown: number, tradingDays: number }}
 */
export function stressScenario(key, beta) {
  const sc = SCENARIO_PATHS[key];
  if (!sc) throw new Error(`unknown scenario ${key}`);
  let equity = 1.0;
  let peak = 1.0;
  let drawdown = 0.0;
  for (const r of sc.returns) {
    const growth = 1 + beta * r;
    // The engine caps |beta| at 4 so that no scaled day loses 100% or more.
    if (!(growth > 0)) throw new Error(`beta ${beta} turns a ${r} day into a total loss`);
    equity *= growth;
    peak = Math.max(peak, equity);
    drawdown = Math.min(drawdown, equity / peak - 1);
  }
  return { drawdown, totalReturn: equity - 1, marketDrawdown: sc.marketDrawdown, tradingDays: sc.tradingDays };
}

// The engine's threshold (riskcanvas.scenarios.stress_test.LOW_R_SQUARED).
export const LOW_R_SQUARED = 0.3;

/**
 * The share of the portfolio's daily variance the market explains, as the
 * engine's `beta_fit`: the squared correlation, 0 for a constant portfolio,
 * null for a constant market (no fit exists).
 *
 * @param {number[]} portfolio daily returns
 * @param {number[]} market daily returns, same length
 * @returns {number | null}
 */
export function rSquared(portfolio, market) {
  const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
  const mp = mean(portfolio);
  const mm = mean(market);
  let spm = 0;
  let spp = 0;
  let smm = 0;
  for (let i = 0; i < portfolio.length; i++) {
    spm += (portfolio[i] - mp) * (market[i] - mm);
    spp += (portfolio[i] - mp) ** 2;
    smm += (market[i] - mm) ** 2;
  }
  if (!(Math.sqrt(smm) > 1e-12)) return null;
  if (!(Math.sqrt(spp) > 1e-12)) return 0;
  return Math.min(1, (spm * spm) / (smm * spp));
}

/**
 * The engine's `weak_equity_fit` warning, word for word; null when the market
 * explains enough of the portfolio (or no fit exists).
 *
 * @param {number | null} r2
 * @returns {string | null}
 */
export function weakFitWarning(r2) {
  if (r2 === null || r2 >= LOW_R_SQUARED) return null;
  return (
    `The S&P 500 explains ${(r2 * 100).toFixed(0)}% of this portfolio's daily ` +
    `variance (R^2 ${r2.toFixed(2)}): an equity-beta scenario captures ` +
    "only that part; its rate, credit, FX and idiosyncratic risk are not modelled."
  );
}
