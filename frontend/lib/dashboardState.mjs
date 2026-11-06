/**
 * Dashboard state decisions, kept pure so they can be unit-tested: an emptied
 * portfolio (every asset removed, or every weight at 0) has nothing to analyse,
 * and while a recompute is in flight the caption says what the numbers on
 * screen are.
 */

/**
 * The held weights as fractions (percent / 100), or null when nothing is
 * held: no asset, or every weight 0 / not a finite number.
 *
 * @param {Record<string, number>} percentWeights
 * @returns {Record<string, number> | null}
 */
export function heldFractions(percentWeights) {
  const held = Object.entries(percentWeights).filter(([, w]) => Number.isFinite(w) && w > 0);
  return held.length ? Object.fromEntries(held.map(([s, w]) => [s, w / 100])) : null;
}

/**
 * While a recompute is in flight: what the numbers on screen ARE, by where
 * they came from.
 */
export const RECOMPUTE_CAPTION = {
  // Nothing on screen yet (e.g. the first asset after an empty portfolio).
  none: "Computing: no result yet.",
  preview: "Connecting to the engine; until it responds the numbers on this page are an in-browser preview.",
  engine: "Recomputing: the numbers on this page are the last engine result until the engine responds.",
  demo: "Recomputing: the numbers on this page are the last in-browser estimate until the engine responds.",
};

/**
 * The weights the engine actually analyses: the held weights rescaled to sum
 * to 1 (100% invested), or null when nothing is held. The API and the
 * in-browser estimate both normalise, so a slider total of 90% (AGG removed
 * from 40/20/15/15/10) is analysed as SPY 44.4%, not as 40% plus 10% cash.
 * The dashboard shows these shares whenever the sliders do not total 100%.
 *
 * @param {Record<string, number>} percentWeights
 * @returns {Record<string, number> | null}
 */
export function analysedShares(percentWeights) {
  const held = heldFractions(percentWeights);
  if (!held) return null;
  const total = Object.values(held).reduce((a, b) => a + b, 0);
  return Object.fromEntries(Object.entries(held).map(([s, w]) => [s, w / total]));
}
