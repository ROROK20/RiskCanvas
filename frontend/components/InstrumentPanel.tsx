import type { CSSProperties } from "react";
import type { RiskMetrics } from "@/lib/mockRisk";
import { fmtPct, fmtRatio, fmtUsd } from "@/lib/format";

function ScaleReadout({ label, value, usd, level, primary }: { label: string; value: string; usd?: string; level: number; primary?: boolean }) {
  return (
    <div className="scale-readout" style={{ "--level": level } as CSSProperties}>
      <dt className="legend">{label}</dt>
      <dd className={primary ? "figure text-[2.5rem] font-bold leading-[3.25rem]" : "figure text-[2rem] leading-[3.25rem]"}>
        {value}
      </dd>
      {usd && <dd className="figure text-base">{usd}</dd>}
    </div>
  );
}

function DataLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="dataline">
      <dt className="legend">{label}</dt>
      <dd className="figure text-base">{value}</dd>
    </div>
  );
}

/**
 * The risk readouts, in soft white whatever their value: an ordinary VaR is not an
 * alarm, and a good Sharpe does not need a colour either. The order of this list is
 * what docs/capture.json records and tests/test_capture.py checks.
 */
export function InstrumentPanel({ metrics: m, initialValue }: { metrics: RiskMetrics; initialValue: number }) {
  const usd = (n: number) => fmtUsd(n * initialValue);
  const scale = Math.max(m.var95_1d, m.var99_1d, m.cvar95, m.var95_10d);
  const level = (n: number) => (scale > 0 ? Math.min(Math.max(n, 0) / scale, 1) : 0);
  return (
    <section aria-label="Risk readouts" className="mt-5 border-t border-rule pt-4">
      <dl className="panel">
        <ScaleReadout primary label="VaR 95% (1d)" value={fmtPct(m.var95_1d)} usd={usd(m.var95_1d)} level={level(m.var95_1d)} />
        <ScaleReadout label="VaR 99% (1d)" value={fmtPct(m.var99_1d)} usd={usd(m.var99_1d)} level={level(m.var99_1d)} />
        <ScaleReadout label="CVaR 95%" value={fmtPct(m.cvar95)} usd={usd(m.cvar95)} level={level(m.cvar95)} />
        <DataLine label="Max DD" value={fmtPct(m.maxDrawdown)} />
        <DataLine label="Sharpe" value={fmtRatio(m.sharpe)} />
        <DataLine label="Sortino" value={fmtRatio(m.sortino)} />
        <DataLine label="CAGR" value={fmtPct(m.cagr)} />
        <DataLine label="Annual Vol" value={fmtPct(m.annualVol)} />
        <DataLine label="Calmar" value={fmtRatio(m.calmar)} />
        <DataLine label="Skewness" value={fmtRatio(m.skewness)} />
        <DataLine label="Excess Kurt." value={fmtRatio(m.excessKurtosis)} />
        <ScaleReadout label="VaR 95% (10d)" value={fmtPct(m.var95_10d)} level={level(m.var95_10d)} />
      </dl>
      <p className="mt-2 text-xs text-label">The four scales run from zero to the largest of the four readings.</p>
    </section>
  );
}
