/**
 * Width of a chart axis label, for placing ticks so their labels never touch.
 *
 * Advance widths of B612, the axis font, in em (measured at 12px in Chromium and
 * WebKit: digit 7.89px, "%" 10.89px, "-" 5.48px, "." 3.69px), rounded up.
 */
const EM = { digit: 0.66, "%": 0.91, "-": 0.46, ".": 0.31 };

/**
 * @param {string} label
 * @param {number} fontSize px
 * @returns {number} px, at least the rendered width
 */
export function labelWidth(label, fontSize) {
  return [...label].reduce((w, c) => w + (EM[c] ?? EM.digit) * fontSize, 0);
}
