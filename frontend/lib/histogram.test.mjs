// node --test lib/histogram.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { histogram, tailShare } from "./histogram.mjs";

const SERIES = [-0.031, -0.012, -0.004, 0.0, 0.002, 0.007, 0.011, 0.019, 0.026, 0.044];

test("every observation is counted, the maximum included", () => {
  // `r < hi` on every bucket dropped the best day.
  const h = histogram(SERIES, 30, -0.02);
  assert.equal(h.length, 30);
  assert.equal(h.reduce((a, b) => a + b.count, 0), SERIES.length);
  assert.equal(h[29].count, 1); // 0.044, the maximum
  assert.equal(h[0].count, 1); // -0.031, the minimum
});

test("tail flag follows the bucket midpoint", () => {
  const h = histogram(SERIES, 30, -0.02);
  assert.ok(h[0].isTail);
  assert.ok(!h[29].isTail);
  assert.equal(h[0].bucket, "-3.0%");
});

test("a constant series lands in one bucket", () => {
  const h = histogram([0.001, 0.001, 0.001], 30, -0.02);
  assert.equal(h.reduce((a, b) => a + b.count, 0), 3);
  assert.equal(h[0].count, 3);
});

test("the tail share is what the red bars hold, not a nominal 5%", () => {
  // "Red bars = worst 5%" while they held 7.2%.
  const h = histogram(SERIES, 30, -0.02);
  const red = h.filter((b) => b.isTail).reduce((a, b) => a + b.count, 0);
  assert.equal(tailShare(h), red / SERIES.length);
  assert.equal(tailShare([]), 0);
});
