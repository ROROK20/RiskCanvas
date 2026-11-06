import clsx from "clsx";
import type { BetaSource, StressBeta, StressResult } from "@/lib/mockRisk";
import { fmtPct, fmtUsd } from "@/lib/format";

const BETA_SOURCE_LABEL: Record<BetaSource, string> = {
  explicit: "supplied explicitly",
  market_returns: "estimated by the engine from the synthetic market series",
  spy_leg: "estimated by the engine from the SPY series",
  yfinance_spy: "estimated by the engine vs SPY (yfinance)",
  assumed: "ASSUMED — no market series was available",
  demo_factor_loading: "in-browser: the synthetic portfolio's factor loading, not the engine's OLS estimate",
};

const AXIS = [0, 25, 50, 75, 100];

/** Deepest loss first; the sort is stable, so equal losses keep the engine's order. */
export function rankByLoss(results: StressResult[]): StressResult[] {
  return [...results].sort((a, b) => a.portfolioDrawdown - b.portfolioDrawdown);
}

function Rung({ rank, result: s, initialValue }: { rank: number; result: StressResult; initialValue: number }) {
  const worst = rank === 1;
  const depth = Math.min(Math.abs(s.portfolioDrawdown), 1);
  const market = Math.min(Math.abs(s.marketDrawdown), 1);
  return (
    <li data-testid="stress-rung" className="rung border-b border-rule py-4 lg:py-3.5">
      <div className="rung-rank font-mono text-sm leading-6 text-label">{rank}</div>
      <div className="rung-name">
        <div className="text-base font-bold leading-6">{s.name}</div>
        <div className="text-sm text-label">{s.description}</div>
      </div>
      <div className="rung-bar mt-3 lg:mt-1.5">
        <div className="stress-track">
          {AXIS.slice(1, -1).map((tick) => (
            <i key={tick} aria-hidden className="stress-grid" style={{ left: `${tick}%` }} />
          ))}
          <div
            aria-hidden
            className={clsx("stress-fill", worst ? "bg-amber" : "bg-amber-dim")}
            style={{ transform: `scaleX(${depth})` }}
          />
          <i aria-hidden className="market-tick" style={{ left: `${market * 100}%` }} />
        </div>
      </div>
      <div className="rung-loss mt-2 flex items-baseline gap-3 lg:mt-0 lg:block lg:text-right">
        <div
          data-testid="stress-loss"
          className={clsx("text-xl font-bold leading-7", worst ? "figure-amber" : "figure")}
        >
          {fmtPct(s.portfolioDrawdown)}
        </div>
        <div className="figure text-sm">{fmtUsd(s.portfolioDrawdown * initialValue)} loss</div>
      </div>
      <div className="rung-meta mt-1 flex flex-wrap gap-x-5 text-xs leading-5 text-label lg:mt-2">
        <span>Market: {fmtPct(s.marketDrawdown)}</span>
        <span>Duration: {s.durationDays} trading days</span>
        <span title="Trading days to regain the pre-crisis level, compounding at the portfolio's historical growth rate">
          Recovery at hist. growth: ~{s.recoveryDays != null && s.recoveryDays > 0 ? `${s.recoveryDays}d` : "N/A"}
        </span>
      </div>
    </li>
  );
}

/** The crisis scenarios as one ladder, deepest rung first and the only one fully lit. */
export function StressLadder({
  results,
  beta,
  initialValue,
}: {
  results: StressResult[];
  beta: StressBeta;
  initialValue: number;
}) {
  return (
    <section aria-labelledby="stress-heading" className="mt-8 border-t border-rule pt-5">
      <h2 id="stress-heading" className="text-base font-bold">
        Crisis Stress Tests
      </h2>
      <div className="mt-2 grid gap-x-12 gap-y-2 lg:grid-cols-2">
        <p className="max-w-[75ch] text-sm text-label" data-testid="stress-disclosure">
          Equity-market shocks only: calibrated synthetic S&amp;P 500 paths matched to each
          crisis&apos;s documented peak-to-trough drawdown (not historical replays); portfolio
          path = beta × S&amp;P path. No interest-rate, credit or FX factor, so bonds, gold and
          other low-beta holdings move only through their equity beta.
        </p>
        <div className="max-w-[75ch]">
          <p
            data-testid="stress-beta"
            title={beta.note}
            className={clsx("text-sm", beta.source === "assumed" ? "font-bold text-readout" : "text-label")}
          >
            Stress beta <span className="figure">{beta.value.toFixed(2)}</span> ({BETA_SOURCE_LABEL[beta.source]})
          </p>
          {beta.warnings.length > 0 && (
            <ul data-testid="stress-warnings" className="mt-2 space-y-1 text-sm font-bold text-readout">
              {beta.warnings.map((w) => (
                <li key={w}>⚠ {w}</li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="rung mt-6">
        <p className="rung-name mb-2 text-xs leading-5 text-label lg:mb-0">
          Bar: this portfolio. Tick: the S&amp;P 500 path.
        </p>
        <div aria-hidden className="rung-bar relative h-5 font-mono text-xs leading-5 text-label">
          {AXIS.map((tick) => (
            <span
              key={tick}
              className="absolute top-0"
              style={{ left: `${tick}%`, transform: `translateX(-${tick}%)` }}
            >
              {tick}%
            </span>
          ))}
        </div>
      </div>
      <ol className="border-t border-rule">
        {rankByLoss(results).map((s, i) => (
          <Rung key={s.name} rank={i + 1} result={s} initialValue={initialValue} />
        ))}
      </ol>
    </section>
  );
}
