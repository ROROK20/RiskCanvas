// node --test lib/*.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { fanBands, fanDomain, simulateGbmFan } from "./fan.mjs";

const POINTS = [
  { day: 0, p5: 1, p25: 1, p50: 1, p75: 1, p95: 1 },
  { day: 126, p5: 0.82, p25: 0.93, p50: 1.0, p75: 1.07, p95: 1.19 },
  { day: 252, p5: 0.75, p25: 0.9, p50: 1.01, p75: 1.12, p95: 1.31 },
];

test("bands are percentile ranges, not fills from zero", () => {
  // A number-valued Area fills down to 0, which would make the darkest band 0..p5.
  const b = fanBands(POINTS);
  assert.deepEqual(b[2], { day: 252, outer: [0.75, 1.31], inner: [0.9, 1.12], median: 1.01 });
  for (const x of b) {
    assert.ok(x.outer[0] > 0 && x.outer[0] <= x.inner[0] && x.inner[0] <= x.median);
    assert.ok(x.median <= x.inner[1] && x.inner[1] <= x.outer[1]);
  }
});

test("percentiles out of order are rejected", () => {
  assert.throws(() => fanBands([{ day: 1, p5: 1.1, p25: 1, p50: 1, p75: 1, p95: 1 }]), /day 1/);
});

test("the axis fits the bands instead of starting at 0", () => {
  assert.deepEqual(fanDomain(fanBands(POINTS)), [0.75, 1.35]);
});

test("the demo fan is geometric: never negative, unit expected growth", () => {
  // "in-browser GBM" took arithmetic steps 1 + sigma * z.
  let seed = 7;
  const uniform = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) + 0.5) / 2147483648;
  const normal = () => Math.sqrt(-2 * Math.log(uniform())) * Math.cos(2 * Math.PI * uniform());
  const fan = simulateGbmFan(0.6, 2000, 20, 5, normal);   // a 60%-a-day series
  assert.deepEqual(fan.map((p) => p.day), [0, 5, 10, 15, 20]);
  for (const p of fan) assert.ok(p.p5 > 0 && p.p5 <= p.p25 && p.p25 <= p.p50 && p.p50 <= p.p75 && p.p75 <= p.p95);
  // Median of zero-drift GBM: exp(-sigma^2 / 2 * t).
  const day1 = simulateGbmFan(0.01, 20000, 1, 1, normal)[1];
  assert.ok(Math.abs(day1.p50 - Math.exp(-0.00005)) < 0.0005, String(day1.p50));
});
