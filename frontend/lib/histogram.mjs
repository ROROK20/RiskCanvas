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
 * @returns {{ bucket: string, count: number, isTail: boolean }[]}
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
    return { bucket: `${(mid * 100).toFixed(1)}%`, count, isTail: mid <= tailThreshold };
  });
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
