// node --test lib/stress.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { SCENARIO_PATHS } from "./scenarioPaths.mjs";
import { LOW_R_SQUARED, rSquared, stressScenario, weakFitWarning } from "./stress.mjs";

test("beta 1 reproduces every documented S&P 500 drawdown", () => {
  for (const [key, sc] of Object.entries(SCENARIO_PATHS)) {
    const r = stressScenario(key, 1);
    assert.ok(Math.abs(r.drawdown - sc.marketDrawdown) < 1e-6, `${key}: ${r.drawdown}`);
    assert.ok(Math.abs(r.totalReturn - r.drawdown) < 1e-12, `${key} ends at its trough`);
  }
});

test("demo stress compounds beta x path: 100% TSLA (beta 1.9) is the engine's -79.94% GFC", () => {
  // The demo multiplied the drawdown by beta:
  // -0.568 * 1.9 = -107.92%, a loss beyond the portfolio's value.
  // Engine: StressTester(beta_to_spy=1.9).run_scenario(p, "gfc_2008").
  const r = stressScenario("gfc_2008", 1.9);
  assert.ok(Math.abs(r.drawdown - -0.7993905606730651) < 1e-12, String(r.drawdown));
});

test("no beta the demo can produce loses 100% or more", () => {
  for (const key of Object.keys(SCENARIO_PATHS)) {
    for (const beta of [0, 0.05, 0.5, 1, 1.5, 1.9, 2.5]) {
      const r = stressScenario(key, beta);
      assert.ok(r.drawdown > -1 && r.drawdown <= 0, `${key} beta ${beta}: ${r.drawdown}`);
    }
  }
  assert.equal(stressScenario("covid_crash_2020", 0).drawdown, 0);
});

test("rSquared is the engine's squared correlation", () => {
  const x = [0.01, -0.02, 0.03, -0.01, 0.02];
  assert.ok(Math.abs(rSquared(x.map((v) => 2 * v + 0.001), x) - 1) < 1e-12);
  assert.ok(Math.abs(rSquared([1, 1, -1, -1], [1, -1, 1, -1])) < 1e-12);
  assert.equal(rSquared([0.01, 0.01, 0.01], [0.01, -0.02, 0.03]), 0);
  assert.equal(rSquared([0.01, -0.02, 0.03], [0.01, 0.01, 0.01]), null);
});

test("the demo warns of a weak equity fit in the engine's words, on the engine's threshold", () => {
  // The engine warned for 100% BTC (R^2 0.08) while the demo showed nothing.
  assert.equal(
    weakFitWarning(0.08),
    "The S&P 500 explains 8% of this portfolio's daily variance (R^2 0.08): an equity-beta scenario captures only that part; its rate, credit, FX and idiosyncratic risk are not modelled.",
  );
  assert.equal(weakFitWarning(LOW_R_SQUARED), null);
  assert.ok(weakFitWarning(LOW_R_SQUARED - 0.001));
  assert.equal(weakFitWarning(null), null);
});
