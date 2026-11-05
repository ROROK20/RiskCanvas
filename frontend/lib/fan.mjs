/**
 * Monte Carlo fan-chart bands from per-day percentiles of simulated value.
 *
 * Recharts draws an `Area` whose value is a `[low, high]` pair as the band
 * between the two; a plain number fills down to the axis baseline, which would
 * make the darkest region 0..p5. So the fan passes pairs: outer p5-p95, inner
 * p25-p75, and the median.
 *
 * @param {{day: number, p5: number, p25: number, p50: number, p75: number, p95: number}[]} points
 * @returns {{day: number, outer: [number, number], inner: [number, number], median: number}[]}
 */
export function fanBands(points) {
  return points.map((p) => {
    if (!(p.p5 <= p.p25 && p.p25 <= p.p50 && p.p50 <= p.p75 && p.p75 <= p.p95)) {
      throw new Error(`percentiles out of order on day ${p.day}`);
    }
    return { day: p.day, outer: [p.p5, p.p95], inner: [p.p25, p.p75], median: p.p50 };
  });
}

/**
 * Y-axis domain that fits the bands (rounded out to 0.05), instead of the
 * Recharts default that starts at 0.
 *
 * @param {{outer: [number, number]}[]} bands
 * @returns {[number, number]}
 */
export function fanDomain(bands) {
  const lo = Math.min(...bands.map((b) => b.outer[0]));
  const hi = Math.max(...bands.map((b) => b.outer[1]));
  return [Math.floor(lo * 20) / 20, Math.ceil(hi * 20) / 20];
}

/**
 * The in-browser demo's Monte Carlo fan: zero-drift geometric Brownian
 * motion, ``value *= exp(-sigma^2 / 2 + sigma * z)`` per day (unit expected
 * growth, never negative), sampled every ``step`` days. Arithmetic Gaussian
 * steps, ``value *= 1 + sigma * z``, are not GBM and go negative once a day's
 * shock passes -100%.
 *
 * @param {number} sigma daily volatility
 * @param {number} nPaths
 * @param {number} horizon trading days
 * @param {number} step record a fan point every `step` days (plus day 0)
 * @param {() => number} normal standard-normal draws
 * @returns {{day: number, p5: number, p25: number, p50: number, p75: number, p95: number}[]}
 */
export function simulateGbmFan(sigma, nPaths, horizon, step, normal) {
  const drift = -0.5 * sigma * sigma;
  const paths = [];
  for (let p = 0; p < nPaths; p++) {
    const path = [1.0];
    for (let d = 0; d < horizon; d++) {
      path.push(path[path.length - 1] * Math.exp(drift + sigma * normal()));
    }
    paths.push(path);
  }
  const fan = [];
  for (let d = 0; d <= horizon; d += step) {
    const values = paths.map((p) => p[d]).sort((a, b) => a - b);
    const at = (q) => values[Math.floor(q * nPaths)];
    fan.push({ day: d, p5: at(0.05), p25: at(0.25), p50: at(0.5), p75: at(0.75), p95: at(0.95) });
  }
  return fan;
}
