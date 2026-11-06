// Scales drawn like instrument scales: a line and outward ticks, no grid behind the data.
// B612 rather than B612 Mono: same digit width, but a decimal point that does not take a full cell.
const FIGURES = "var(--font-b612), Arial, sans-serif";

export const AXIS_TICK = { fontSize: 12, fontFamily: FIGURES, fill: "var(--label)" };
export const AXIS_LINE = { stroke: "var(--rule-strong)" };
export const TICK_LINE = { stroke: "var(--rule-strong)" };

export const TOOLTIP = {
  contentStyle: {
    background: "var(--raised)",
    border: "1px solid var(--rule-strong)",
    borderRadius: 0,
    fontFamily: FIGURES,
    fontSize: "0.75rem",
    color: "var(--readout)",
  },
  labelStyle: { color: "var(--label)" },
  itemStyle: { color: "var(--readout)" },
};
