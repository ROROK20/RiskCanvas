// node --test lib/dashboardState.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { analysedShares, heldFractions, RECOMPUTE_CAPTION } from "./dashboardState.mjs";

test("an emptied portfolio holds nothing (the page shows 'add an asset', not stale numbers)", () => {
  // {SPY: 0} crashed the fallback and left the
  // previous portfolio's numbers on screen.
  assert.equal(heldFractions({}), null);
  assert.equal(heldFractions({ SPY: 0, AGG: 0 }), null);
  assert.equal(heldFractions({ SPY: Number.NaN }), null);
});

test("held weights become fractions; zero weights are dropped", () => {
  assert.deepEqual(heldFractions({ SPY: 40, AGG: 0, QQQ: 10 }), { SPY: 0.4, QQQ: 0.1 });
});

test("the recompute caption says what is on screen", () => {
  assert.match(RECOMPUTE_CAPTION.engine, /last engine result/);
  assert.match(RECOMPUTE_CAPTION.demo, /last in-browser estimate/);
  assert.match(RECOMPUTE_CAPTION.preview, /in-browser preview/);
  // After an empty portfolio the caption claimed
  // "the last engine result" with nothing on screen.
  assert.match(RECOMPUTE_CAPTION.none, /no result yet/);
});

test("the analysed shares are the held weights rescaled to 100%", () => {
  // "Total: 90%" suggested 10% cash, but the
  // engine analyses SPY at 44.4%.
  const a = analysedShares({ SPY: 40, QQQ: 20, AAPL: 15, MSFT: 15, AGG: 0 });
  assert.ok(Math.abs(a.SPY - 40 / 90) < 1e-12 && Math.abs(a.QQQ - 20 / 90) < 1e-12);
  assert.ok(Math.abs(Object.values(a).reduce((x, y) => x + y, 0) - 1) < 1e-12);
  assert.equal("AGG" in a, false);
  assert.equal(analysedShares({ SPY: 0 }), null);
});
