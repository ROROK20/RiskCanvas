import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { tailShare } from "@/lib/histogram.mjs";
import type { ReturnDistribution } from "@/lib/mockRisk";
import { AXIS_LINE, AXIS_TICK, TICK_LINE, TOOLTIP } from "./chartStyle";

/** The daily returns as a histogram: graphite, with only the tail days lit. */
export function DistributionInstrument({ buckets }: { buckets: ReturnDistribution[] }) {
  return (
    <div>
      <h3 className="text-base font-bold">Return Distribution</h3>
      <p className="mb-4 mt-1 max-w-[65ch] text-sm text-label" data-testid="tail-caption">
        Amber bars = buckets beyond the 95% VaR ({(tailShare(buckets) * 100).toFixed(1)}% of days)
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={buckets} barCategoryGap={1} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <XAxis
            dataKey="bucket"
            tick={AXIS_TICK}
            tickLine={TICK_LINE}
            axisLine={AXIS_LINE}
            interval="preserveStartEnd"
            minTickGap={20}
          />
          <YAxis tick={AXIS_TICK} tickLine={TICK_LINE} axisLine={AXIS_LINE} width={40} />
          <Tooltip {...TOOLTIP} cursor={{ fill: "var(--rule)" }} />
          <Bar dataKey="count" name="days" isAnimationActive={false}>
            {buckets.map((b) => (
              <Cell key={b.bucket} fill={b.isTail ? "var(--amber)" : "var(--bar)"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
