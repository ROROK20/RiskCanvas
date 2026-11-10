// node --test lib/*.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";

import { dayTicks, fanBands, fanDomain, simulateGbmFan } from "./fan.mjs";

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

// The fan's plot width at three screens: its chart box less the two scales (48 + 84 px).
const PHONE_320 = 156;
const PHONE_375 = 211;
const DESKTOP_1440 = 484;
const DIGIT_PX = 7.89; // B612 at 12px, measured

test("the day axis steps by a trading month, quarter or half year from day 0", () => {
  // The category axis labelled checkpoint days and always kept the last:
  // "0 20 45 71 96 121 146 171 197 222 252" at 1440px.
  const range = (step) => Array.from({ length: 252 / step + 1 }, (_, k) => k * step);
  assert.deepEqual(dayTicks(252, DESKTOP_1440), range(21));
  assert.deepEqual(dayTicks(252, PHONE_375), [0, 63, 126, 189, 252]);
  assert.deepEqual(dayTicks(252, PHONE_320), [0, 126, 252]);
});

test("the day axis stays evenly spaced and readable for any horizon the engine takes", () => {
  const steps = [1, 5, 21, 63, 126, 252, 504, 1260, 2520];
  for (const horizon of [1, 2, 5, 10, 21, 30, 100, 252, 260, 500, 1000, 2519, 2520]) {
    for (const width of [0, 20, 60, 120, PHONE_320, PHONE_375, 334, DESKTOP_1440, 900]) {
      const at = `horizon ${horizon} at ${width}px`;
      const ticks = dayTicks(horizon, width);
      assert.equal(ticks[0], 0, at);
      if (ticks.length < 2) continue;
      const step = ticks[1];
      assert.ok(steps.includes(step), `${at}: step ${step}`);
      ticks.forEach((t, k) => assert.equal(t, k * step, `${at}: ${ticks}`));
      // Every multiple of the step up to the horizon, and no forced end label.
      assert.ok(ticks.at(-1) <= horizon && horizon - ticks.at(-1) < step, `${at}: ${ticks}`);
      const x = (t) => (t / horizon) * width;
      const half = (t) => (String(t).length * DIGIT_PX) / 2;
      for (let k = 1; k < ticks.length; k++) {
        const gap = x(ticks[k]) - half(ticks[k]) - (x(ticks[k - 1]) + half(ticks[k - 1]));
        assert.ok(gap >= 16, `${at}: ${ticks[k - 1]} and ${ticks[k]} are ${gap.toFixed(1)}px apart`);
      }
    }
  }
});
