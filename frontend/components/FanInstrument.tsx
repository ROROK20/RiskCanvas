import { Area, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { dayTicks, fanBands, fanDomain } from "@/lib/fan.mjs";
import type { MonteCarloInfo, MonteCarloPath } from "@/lib/mockRisk";
import { AXIS_LINE, AXIS_TICK, TICK_LINE, TOOLTIP } from "./chartStyle";

// The right-hand scale names the percentiles at the horizon, lowest first.
const END_LABELS = ["p5", "p25", "p50", "p75", "p95"];
const MARGIN = { top: 8, right: 0, bottom: 0, left: 0 };
const Y_AXIS_WIDTH = 48;
const END_AXIS_WIDTH = 84;

/**
 * The simulated value by day. `outer` and `inner` are [low, high] pairs from
 * fanBands, so each band is really the space between its two percentiles.
 */
export function FanInstrument({ points, info }: { points: MonteCarloPath[]; info: MonteCarloInfo }) {
  return (
    <div>
      <h3 className="text-base font-bold">
        Monte Carlo Fan ({info.horizonDays} days, {info.paths.toLocaleString("en-US")} paths, {info.method}, {info.drift} drift)
      </h3>
      <p className="mb-4 mt-1 max-w-[65ch] text-sm text-label">
        Simulated portfolio value by day: 5th–95th percentile (outer band), 25th–75th (inner band), median (line)
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <Fan points={points} />
      </ResponsiveContainer>
    </div>
  );
}

/**
 * ResponsiveContainer passes its measured width and height to its child, and the
 * day axis needs the width: how many evenly spaced day ticks fit depends on it.
 * The axis is numeric, so each band point sits at its own day.
 */
function Fan({ points, width = 0, height = 0 }: { points: MonteCarloPath[]; width?: number; height?: number }) {
  const fan = fanBands(points);
  const domain = fanDomain(fan);
  const end = fan[fan.length - 1];
  const ends = [end.outer[0], end.inner[0], end.median, end.inner[1], end.outer[1]];
  const days = dayTicks(end.day, width - Y_AXIS_WIDTH - END_AXIS_WIDTH - MARGIN.left - MARGIN.right, {
    fontSize: AXIS_TICK.fontSize,
  });
  return (
    <ComposedChart width={width} height={height} data={fan} margin={MARGIN}>
      <XAxis
        type="number"
        scale="linear"
        dataKey="day"
        domain={[0, end.day]}
        ticks={days}
        interval={0}
        tick={AXIS_TICK}
        tickLine={TICK_LINE}
        axisLine={AXIS_LINE}
      />
      <YAxis
        tick={AXIS_TICK}
        tickLine={TICK_LINE}
        axisLine={AXIS_LINE}
        width={Y_AXIS_WIDTH}
        domain={domain}
        ticks={[domain[0], 1, domain[1]]}
        interval={0}
        tickFormatter={(v) => `${v.toFixed(2)}x`}
      />
      <YAxis
        yAxisId="ends"
        orientation="right"
        tick={AXIS_TICK}
        tickLine={TICK_LINE}
        axisLine={AXIS_LINE}
        width={END_AXIS_WIDTH}
        domain={domain}
        ticks={ends}
        interval={0}
        tickFormatter={(v, i) => `${END_LABELS[i]} ${v.toFixed(2)}x`}
      />
      <Tooltip
        {...TOOLTIP}
        labelFormatter={(day) => `Day ${day}`}
        formatter={(v: number | number[]) =>
          Array.isArray(v) ? `${v[0].toFixed(3)}x – ${v[1].toFixed(3)}x` : `${v.toFixed(3)}x`
        }
      />
      <Area type="monotone" dataKey="outer" name="5th–95th" stroke="none" fill="var(--band-outer)" fillOpacity={1} isAnimationActive={false} />
      <Area type="monotone" dataKey="inner" name="25th–75th" stroke="none" fill="var(--band-inner)" fillOpacity={1} isAnimationActive={false} />
      <ReferenceLine y={1.0} stroke="var(--rule-strong)" strokeDasharray="4 4" />
      <Line type="monotone" dataKey="median" name="median" stroke="var(--readout)" strokeWidth={2} dot={false} isAnimationActive={false} />
    </ComposedChart>
  );
}
