// node --test lib/histogram.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { histogram, histogramAxis, pctLabel, tailShare } from "./histogram.mjs";

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

test("the buckets tile [min, max] edge to edge, each midpoint centred", () => {
  const h = histogram(SERIES, 30, -0.02);
  assert.equal(h[0].lo, -0.031);
  assert.equal(h[29].hi, 0.044);
  for (let i = 0; i < 30; i++) {
    assert.ok(Math.abs(h[i].mid - (h[i].lo + h[i].hi) / 2) < 1e-15, `bucket ${i}`);
    assert.equal(h[i].bucket, pctLabel(h[i].mid, 1));
    if (i) assert.ok(Math.abs(h[i].lo - h[i - 1].hi) < 1e-15, `gap before bucket ${i}`);
  }
});

// The B612 advance widths at 12px, as measured; histogramAxis rounds them up.
const PX = { "%": 10.89, "-": 5.48, ".": 3.69 };
const measured = (label) => [...label].reduce((w, c) => w + (PX[c] ?? 7.89), 0);
const axisOf = (lo, hi, width) => histogramAxis([{ lo, hi }], width);

function assertReadableScale(lo, hi, width, at) {
  const { domain, ticks, decimals } = axisOf(lo, hi, width);
  assert.deepEqual(domain, [lo, hi], at);
  const x = (t) => ((t - lo) / (hi - lo)) * width;
  const labels = ticks.map((t) => pctLabel(t, decimals));
  ticks.forEach((t, i) => {
    assert.ok(t >= lo && t <= hi, `${at}: ${labels[i]} outside the span`);
    const half = measured(labels[i]) / 2;
    assert.ok(x(t) - half >= 0 && x(t) + half <= width, `${at}: ${labels[i]} hangs past an end`);
    if (i) assert.ok(x(t) - half - (x(ticks[i - 1]) + measured(labels[i - 1]) / 2) >= 16, `${at}: ${labels[i - 1]} and ${labels[i]} crowd`);
  });
  assert.ok(ticks.length <= 8, `${at}: ${ticks.length} ticks`);
  if (ticks.length < 2) return ticks;
  // One step throughout, and a round one: 1, 2 or 5 x 10^k percent.
  const step = (ticks[1] - ticks[0]) * 100;
  for (let i = 2; i < ticks.length; i++) {
    assert.ok(Math.abs((ticks[i] - ticks[i - 1]) * 100 - step) < 1e-9, `${at}: uneven ${labels.join(" ")}`);
  }
  const mantissa = step / 10 ** Math.floor(Math.log10(step) + 1e-9);
  assert.ok([1, 2, 5].some((m) => Math.abs(mantissa - m) < 1e-9), `${at}: step ${step}%`);
  for (const t of ticks) assert.ok(Math.abs(Math.round((t * 100) / step) - (t * 100) / step) < 1e-9, `${at}: ${t} off the step`);
  assert.equal(new Set(labels).size, labels.length, `${at}: repeated label`);
  return ticks;
}

test("the return axis: evenly spaced round ticks, the last no closer than the rest", () => {
  // The category axis labelled bucket midpoints and always kept the last:
  // "-7.5% -5.0% -2.5% 0.0% 4.6%" at 375px, its last gap 108px against 59px.
  const lo = -0.0792;
  const hi = 0.0481;
  const ticks = assertReadableScale(lo, hi, 295, "375px phone");
  assert.deepEqual(ticks.map((t) => pctLabel(t, 0)), ["-6%", "-4%", "-2%", "0%", "2%", "4%"]);
  assert.deepEqual(assertReadableScale(lo, hi, 568, "1440px desktop"), ticks);
});

test("the return axis stays readable at every width and spread", () => {
  for (const width of [0, 40, 120, 240, 295, 334, 408, 568, 656, 1200]) {
    for (const [lo, hi] of [
      [-0.0792, 0.0481],
      [-0.004, 0.003],
      [-0.31, 0.27],
      [0.0012, 0.0019],
      [-0.6, -0.1],
      [-1e-5, 1e-5],
      [-0.06, 0.04],
    ]) {
      assertReadableScale(lo, hi, width, `${width}px over [${lo}, ${hi}]`);
    }
  }
});

test("the axis spans the buckets, and a constant series gets a valid one", () => {
  const h = histogram(SERIES, 30, -0.02);
  assert.deepEqual(histogramAxis(h, 400).domain, [-0.031, 0.044]);
  const flat = histogramAxis(histogram([0.001, 0.001], 30, -0.02), 400);
  assert.ok(flat.domain[0] < 0.001 && flat.domain[1] > 0.001);
  assert.ok(flat.ticks.length >= 2, String(flat.ticks));
  assert.deepEqual(histogramAxis(h, 0).ticks, []);
});
