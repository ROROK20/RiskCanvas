/**
 * How the page writes a figure. One definition, so the sentence at the top and the
 * instruments below it cannot disagree. Always en-US: the page is prerendered, so a
 * viewer's locale would be a hydration mismatch (lib/locale.test.mjs).
 */
export const fmtPct = (n: number) => `${(n * 100).toFixed(2)}%`;

export const fmtUsd = (n: number) => `$${Math.abs(n).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;

// Ratios the engine reports as null when undefined or unbounded.
export const fmtRatio = (n: number | null) => (n == null ? "N/A" : n.toFixed(2));
