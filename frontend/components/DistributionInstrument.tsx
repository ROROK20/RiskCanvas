import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { histogramAxis, pctLabel, tailShare } from "@/lib/histogram.mjs";
import type { ReturnDistribution } from "@/lib/mockRisk";
import { AXIS_LINE, AXIS_TICK, TICK_LINE, TOOLTIP } from "./chartStyle";

const MARGIN = { top: 8, right: 8, bottom: 0, left: 0 };
const Y_AXIS_WIDTH = 40;

/** The daily returns as a histogram: graphite, with only the tail days lit. */
export function DistributionInstrument({ buckets }: { buckets: ReturnDistribution[] }) {
  return (
    <div>
      <h3 className="text-base font-bold">Return Distribution</h3>
      <p className="mb-4 mt-1 max-w-[65ch] text-sm text-label" data-testid="tail-caption">
        Amber bars = buckets beyond the 95% VaR ({(tailShare(buckets) * 100).toFixed(1)}% of days)
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <Histogram buckets={buckets} />
      </ResponsiveContainer>
    </div>
  );
}

/**
 * ResponsiveContainer passes its measured width and height to its child, and the
 * axis needs the width: how many round-number ticks fit depends on it.
 */
function Histogram({ buckets, width = 0, height = 0 }: { buckets: ReturnDistribution[]; width?: number; height?: number }) {
  const axis = histogramAxis(buckets, width - Y_AXIS_WIDTH - MARGIN.left - MARGIN.right, { fontSize: AXIS_TICK.fontSize });
  return (
    <BarChart width={width} height={height} data={buckets} barCategoryGap={1} margin={MARGIN}>
      <XAxis
        type="number"
        scale="linear"
        dataKey="mid"
        domain={axis.domain}
        ticks={axis.ticks}
        interval={0}
        tickFormatter={(v: number) => pctLabel(v, axis.decimals)}
        tick={AXIS_TICK}
        tickLine={TICK_LINE}
        axisLine={AXIS_LINE}
      />
      <YAxis tick={AXIS_TICK} tickLine={TICK_LINE} axisLine={AXIS_LINE} width={Y_AXIS_WIDTH} />
      <Tooltip
        {...TOOLTIP}
        cursor={{ fill: "var(--rule)" }}
        labelFormatter={(_, payload) => payload?.[0]?.payload?.bucket ?? ""}
      />
      <Bar dataKey="count" name="days" isAnimationActive={false}>
        {buckets.map((b, i) => (
          <Cell key={i} fill={b.isTail ? "var(--amber)" : "var(--bar)"} />
        ))}
      </Bar>
    </BarChart>
  );
}
