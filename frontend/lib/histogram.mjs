/**
 * Equal-width histogram of daily returns for the dashboard's distribution
 * chart, shared by the live engine view (liveRisk.ts) and the in-browser
 * demo (mockRisk.ts).
 *
 * `bucketCount` buckets span [min, max]. Every value lands in exactly one
 * bucket: each bucket is [lo, hi) except the last, which also holds the
 * maximum, so the best day of the series is counted. A constant series puts
 * every value in the first bucket.
 *
 * @param {number[]} values daily returns (fractions)
 * @param {number} bucketCount number of buckets (>= 1)
 * @param {number} tailThreshold a bucket whose midpoint is at or below this
 *   return is marked `isTail` (the dashboard passes -VaR95)
 * @returns {{ bucket: string, lo: number, mid: number, hi: number, count: number, isTail: boolean }[]}
 *   `lo`/`hi` are the bucket's edges and `mid` its midpoint (fractions);
 *   `bucket` is the midpoint as a label
 */
export function histogram(values, bucketCount, tailThreshold) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const size = (max - min) / bucketCount;
  const counts = new Array(bucketCount).fill(0);
  for (const r of values) {
    const i = size > 0 ? Math.floor((r - min) / size) : 0;
    counts[Math.min(Math.max(i, 0), bucketCount - 1)] += 1;
  }
  return counts.map((count, i) => {
    const mid = min + (i + 0.5) * size;
    const lo = min + i * size;
    const hi = i === bucketCount - 1 ? max : min + (i + 1) * size;
    return { bucket: `${(mid * 100).toFixed(1)}%`, lo, mid, hi, count, isTail: mid <= tailThreshold };
  });
}

/** A return as an axis label: `pctLabel(-0.06, 0)` is "-6%". */
export function pctLabel(value, decimals) {
  return `${(value * 100).toFixed(decimals)}%`;
}

// Advance widths of B612, the axis font, in em (measured at 12px in Chromium and
// WebKit: digit 7.89px, "%" 10.89px, "-" 5.48px, "." 3.69px), rounded up.
const EM = { digit: 0.66, "%": 0.91, "-": 0.46, ".": 0.31 };
const labelWidth = (label, fontSize) =>
  [...label].reduce((w, c) => w + (EM[c] ?? EM.digit) * fontSize, 0);

/**
 * The return axis under the histogram: it spans the buckets exactly, and its
 * ticks are every multiple of one round step (1, 2 or 5 x 10^k percent) inside
 * that span, so they are evenly spaced. The step is the smallest that keeps at
 * most `maxTicks` labels with at least `minGap` px between neighbours at this
 * width; a label that would hang past either end of the plot is dropped, never
 * nudged off its tick.
 *
 * @param {{ lo: number, hi: number }[]} buckets from `histogram`
 * @param {number} plotWidth px between the axis's two ends
 * @param {{ fontSize?: number, minGap?: number, maxTicks?: number }} [options]
 * @returns {{ domain: [number, number], ticks: number[], decimals: number }}
 *   label tick `t` as `pctLabel(t, decimals)`
 */
export function histogramAxis(buckets, plotWidth, { fontSize = 12, minGap = 16, maxTicks = 8 } = {}) {
  let lo = buckets[0].lo;
  let hi = buckets[buckets.length - 1].hi;
  // A constant series: an axis one percentage point wide, centred on the value.
  if (!(hi > lo)) [lo, hi] = [lo - 0.005, hi + 0.005];
  const domain = [lo, hi];
  const spanPct = (hi - lo) * 100;
  if (!(plotWidth > 0)) return { domain, ticks: [], decimals: 0 };
  const top = Math.ceil(Math.log10(spanPct)) + 1;
  for (let e = top - 4; e <= top; e++) {
    const decimals = Math.max(0, -e);
    for (const m of [1, 2, 5]) {
      const step = m * 10 ** e; // percent
      const first = Math.ceil((lo * 100) / step - 1e-9);
      const last = Math.floor((hi * 100) / step + 1e-9);
      if (last - first + 1 > maxTicks) continue;
      const ticks = [];
      for (let k = first; k <= last; k++) ticks.push(Number((k * step).toFixed(decimals)) / 100);
      const widths = ticks.map((t) => labelWidth(pctLabel(t, decimals), fontSize));
      const stepPx = (step / spanPct) * plotWidth;
      if (stepPx < Math.max(0, ...widths) + minGap) continue;
      const x = (t) => ((t - lo) / (hi - lo)) * plotWidth;
      return {
        domain,
        ticks: ticks.filter((t, i) => x(t) - widths[i] / 2 >= 0 && x(t) + widths[i] / 2 <= plotWidth),
        decimals,
      };
    }
  }
  return { domain, ticks: [], decimals: 0 };
}

/**
 * Share of all observations in the buckets marked ``isTail``. Whole buckets
 * are coloured, so this is not exactly the 5% the VaR threshold cuts at (the
 * bucket straddling it counts entirely): the dashboard states the share it
 * actually colours, which can be 7.2% of the days where the threshold cuts at 5%.
 *
 * @param {{ count: number, isTail: boolean }[]} buckets
 * @returns {number} fraction in [0, 1]
 */
export function tailShare(buckets) {
  const total = buckets.reduce((a, b) => a + b.count, 0);
  const tail = buckets.reduce((a, b) => a + (b.isTail ? b.count : 0), 0);
  return total ? tail / total : 0;
}
