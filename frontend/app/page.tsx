"use client";

import { useState, useEffect } from "react";
import {
  analyzePortfolio,
  AVAILABLE_ASSETS,
  SYNTHETIC_DAYS,
  type PortfolioAnalysis,
} from "@/lib/mockRisk";
import { API_URL } from "@/lib/api";
import { liveAnalyzePortfolio } from "@/lib/liveRisk";
import { analysedShares, heldFractions, RECOMPUTE_CAPTION } from "@/lib/dashboardState.mjs";
import { debouncedRequest, demoCopy, demoReason, ENGINE_DEBOUNCE_MS, NO_ENGINE } from "@/lib/engineRequest.mjs";
import { fmtPct, fmtUsd } from "@/lib/format";
import { AllocationStrip } from "@/components/AllocationStrip";
import { DistributionInstrument } from "@/components/DistributionInstrument";
import { FanInstrument } from "@/components/FanInstrument";
import { InstrumentPanel } from "@/components/InstrumentPanel";
import { rankByLoss, StressLadder } from "@/components/StressLadder";
import clsx from "clsx";

// Data source: "live" = real Python engine, "demo" = in-browser numbers (no engine
// in this build, or the engine failed: demoCopy says which), "loading" = waiting on
// the engine (with no figure on screen until something has answered).
type DataMode = "live" | "demo" | "loading" | "empty" | "error";
// Where the numbers on screen came from (the caption while recomputing).
type Shown = keyof typeof RECOMPUTE_CAPTION;

// Every mode-dependent string lives in ONE total map (demo's in demoCopy, by
// reason): adding a mode without copy is a compile error, so no secondary ternary
// can mislabel a state.
const MODE_COPY: Record<Exclude<DataMode, "demo">, { badge: string; title: string; footer: string; math: string }> = {
  live: {
    badge: "Live engine",
    title: "Connected to the live RiskCanvas Python engine",
    footer: "Live results from the Python engine (numpy/scipy) on synthetic inputs",
    math: "Every risk number on this page is computed by the Python engine (numpy/scipy) from these synthetic series; the return histogram is binned in your browser from the same series.",
  },
  loading: {
    badge: "Connecting…",
    title: "Connecting to the engine…",
    footer: "Connecting to the Python engine…",
    // Replaced by RECOMPUTE_CAPTION[shown]: what is on screen while loading.
    math: RECOMPUTE_CAPTION.none,
  },
  empty: {
    badge: "No assets",
    title: "The portfolio holds nothing",
    footer: "Add an asset to compute its risk",
    math: "The portfolio holds nothing: add an asset, or raise a weight above 0%.",
  },
  error: {
    badge: "Error",
    title: "The in-browser fallback failed",
    footer: "Error: see the message above",
    math: "Neither the engine nor the in-browser fallback produced a result.",
  },
};

// The panel shown instead of results, per mode (total map: no mode can fall
// through to another's text, e.g. "Every weight is 0%" while computing).
const NO_RESULT_COPY: Record<DataMode, { title: string; detail: string }> = {
  live: { title: "Computing…", detail: "Waiting for the first result." },
  demo: { title: "Computing…", detail: "Waiting for the first result." },
  loading: { title: "Computing…", detail: "Waiting for the first result." },
  empty: { title: "Add an asset to see its risk", detail: "Every weight is 0%: there is no portfolio to analyse." },
  error: { title: "No result to show", detail: "" },
};

// The status annunciator, by mode. A nominal state stays unlit; only a state the
// reader must weigh lights up.
const ANNUNCIATOR: Record<DataMode, string> = {
  live: "border-rule-strong text-label",
  demo: "border-readout font-bold text-readout",
  loading: "border-dashed border-rule-strong text-label",
  empty: "border-rule-strong text-label",
  error: "border-fault font-bold text-fault",
};

export default function Home() {
  const [weights, setWeights] = useState<Record<string, number>>({
    SPY: 40, QQQ: 20, AAPL: 15, MSFT: 15, AGG: 10,
  });
  const [initialValue, setInitialValue] = useState(1_000_000);
  const [valueInput, setValueInput] = useState("1000000");
  const valueValid = Number.isFinite(Number(valueInput)) && Number(valueInput) > 0;

  // A build with no engine has nothing to connect to, from its first paint on.
  const [mode, setMode] = useState<DataMode>(API_URL === null ? "demo" : "loading");
  const [whyDemo, setWhyDemo] = useState(NO_ENGINE);
  // Nothing yet: no figure is prerendered, or shown before the engine or the
  // in-browser estimate has answered.
  const [analysis, setAnalysis] = useState<PortfolioAnalysis | null>(null);
  const [shown, setShown] = useState<Shown>("none");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    const held = heldFractions(weights);
    if (!held) {
      // Nothing held: no call, and no stale numbers from the last portfolio.
      setAnalysis(null);
      setShown("none");
      setMode("empty");
      return;
    }
    const input = { weights: held, initialValue };

    // In-browser numbers, and why. A failure here is shown, never swallowed.
    const showEstimate = (reason: typeof NO_ENGINE) => {
      try {
        const local = analyzePortfolio(input);
        setAnalysis(local);
        setShown("demo");
        setWhyDemo(reason);
        setMode(local ? "demo" : "empty");
      } catch (e) {
        setAnalysis(null);
        setErrorMsg(e instanceof Error ? e.message : String(e));
        setMode("error");
      }
    };
    if (API_URL === null) {
      showEstimate(NO_ENGINE);
      return;
    }

    // Try the REAL engine first, once the inputs stop moving; the next change
    // aborts this request. Any engine failure falls back to the in-browser estimate.
    setMode("loading");
    return debouncedRequest((signal) => liveAnalyzePortfolio(input, signal), {
      delayMs: ENGINE_DEBOUNCE_MS,
      onResult: (live) => {
        setAnalysis(live);
        setShown("engine");
        setMode("live");
      },
      onError: (error) => showEstimate(demoReason(error)),
    });
  }, [weights, initialValue]);

  const copy = mode === "demo" ? demoCopy(whyDemo) : MODE_COPY[mode];
  const caption = mode === "loading" ? RECOMPUTE_CAPTION[shown] : copy.math;

  const totalWeight = Object.values(weights).reduce((a, b) => a + b, 0);
  // Both the engine and the in-browser estimate rescale the held weights to
  // 100% invested; show those shares whenever the sliders do not total 100%.
  const shares = analysedShares(weights);
  const rescaled = shares !== null && Math.abs(totalWeight - 100) >= 0.1;

  const addable = AVAILABLE_ASSETS.filter((a) => !(a in weights));

  const updateWeight = (asset: string, value: number) => {
    setWeights({ ...weights, [asset]: value });
  };

  const addAsset = (asset: string) => {
    if (!(asset in weights)) setWeights({ ...weights, [asset]: 10 });
  };

  const removeAsset = (asset: string) => {
    const copy = { ...weights };
    delete copy[asset];
    setWeights(copy);
  };

  const presets = {
    "60/40": { SPY: 60, AGG: 40 },
    "All Weather": { SPY: 30, AGG: 40, GLD: 15, QQQ: 15 },
    "Aggressive": { QQQ: 40, NVDA: 25, AAPL: 15, MSFT: 10, TSLA: 10 },
    "Diversified": { SPY: 40, QQQ: 15, AAPL: 10, MSFT: 10, AGG: 15, GLD: 10 },
  };

  return (
    <main className="min-h-screen" data-stale={mode === "loading"}>
      <header className="sticky top-0 z-10 border-b border-rule bg-bg">
        <div className="mx-auto flex max-w-[84rem] items-center justify-between gap-3 px-4 py-2.5 sm:px-8">
          <div className="min-w-0">
            <p className="text-base font-bold leading-5">RiskCanvas</p>
            <p className="hidden text-xs leading-4 text-label sm:block">Portfolio Risk Analytics</p>
          </div>
          <div className="flex items-center gap-2 sm:gap-4">
            <div
              role="status"
              title={copy.title}
              className={clsx("whitespace-nowrap border px-2 py-1 text-xs leading-4", ANNUNCIATOR[mode])}
            >
              {copy.badge}
            </div>
            <a
              href="https://github.com/ROROK20/RiskCanvas"
              className="whitespace-nowrap py-1 text-xs leading-4 text-label underline decoration-rule-strong underline-offset-4 hover:text-readout"
            >
              View Source
            </a>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[84rem] px-4 pb-10 pt-7 sm:px-8 lg:pt-8">
        {analysis ? (
          <Hero analysis={analysis} initialValue={initialValue} />
        ) : (
          <div data-testid="no-results" className="min-h-[9rem] lg:min-h-[11.5rem]">
            <h1 className="needs-script text-[1.75rem] leading-tight sm:text-[2.25rem] lg:text-[2.5rem]">
              {NO_RESULT_COPY[mode].title}
            </h1>
            <p className="needs-script mt-2 text-sm text-label">{mode === "error" ? errorMsg : NO_RESULT_COPY[mode].detail}</p>
            <noscript>
              <p className="text-[1.75rem] leading-tight sm:text-[2.25rem] lg:text-[2.5rem]">
                The figures need JavaScript. Turn it on to see them.
              </p>
            </noscript>
          </div>
        )}

        {/* Synthetic-input disclosure: true in every mode. */}
        <aside
          data-testid="synthetic-disclosure"
          className="mt-5 grid gap-x-8 gap-y-1 border border-rule-strong px-4 py-2.5 lg:grid-cols-[minmax(0,19rem)_minmax(0,1fr)]"
        >
          <p className="text-base font-bold leading-6 text-readout">Synthetic inputs, not market data.</p>
          <p className="max-w-[90ch] text-sm text-label">
            Each asset is a seeded {SYNTHETIC_DAYS}-day series from a one-factor model (shared
            fat-tailed market shock, plausible betas and volatilities) generated in your browser.{" "}
            <span data-testid="data-caption" className="text-readout">{caption}</span>
          </p>
        </aside>

        <section aria-labelledby="allocation-heading" className="mt-6">
          <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
            <h2 id="allocation-heading" className="text-base font-bold">Portfolio Allocation</h2>
            <div className="flex flex-wrap items-center gap-x-8 gap-y-2">
              <p
                data-testid="weight-total"
                className={clsx("text-sm", rescaled ? "font-bold text-readout" : "text-label")}
              >
                Total: {totalWeight.toFixed(0)}%{rescaled ? " → analysed as 100% invested" : ""}
              </p>
              <div className="flex items-center gap-2">
                <label htmlFor="portfolio-value" className="legend whitespace-nowrap">Portfolio Value ($)</label>
                <input
                  id="portfolio-value"
                  type="number"
                  value={valueInput}
                  min={1}
                  onChange={(e) => {
                    setValueInput(e.target.value);
                    const v = Number(e.target.value);
                    // Only a positive amount is used; anything else keeps the last valid value.
                    if (Number.isFinite(v) && v > 0) setInitialValue(v);
                  }}
                  className={clsx(
                    // iOS Safari zooms the page into a focused field under 16px, so 14px takes a
                    // wide screen AND a fine pointer (a mouse): a phone held sideways or an iPad,
                    // whose pointer is coarse, stays at 16px.
                    "min-h-8 w-32 min-w-0 border bg-raised px-2 font-mono text-base text-readout sm:[@media(pointer:fine)]:text-sm",
                    valueValid ? "border-rule-strong" : "border-fault",
                  )}
                  step={50_000}
                  data-testid="portfolio-value"
                />
              </div>
            </div>
          </div>
          {!valueValid && (
            <p className="mb-3 text-sm text-fault sm:text-right" data-testid="portfolio-value-error">
              Enter a positive amount; figures use the last valid value (${initialValue.toLocaleString("en-US")}).
            </p>
          )}

          <AllocationStrip shares={shares} decimals={rescaled ? 1 : 0} />

          <ul className="mt-2 grid grid-cols-1 gap-x-10 gap-y-0.5 sm:grid-cols-2 min-[1200px]:grid-cols-3">
            {Object.entries(weights).map(([asset, weight]) => (
              <li key={asset} className="flex min-h-8 items-center gap-2">
                <label htmlFor={`weight-${asset}`} className="w-11 shrink-0 font-mono text-sm font-bold">
                  {asset}
                </label>
                <input
                  id={`weight-${asset}`}
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={weight}
                  onChange={(e) => updateWeight(asset, Number(e.target.value))}
                  className="h-6 min-w-0 flex-1"
                />
                <div className="w-11 shrink-0 text-right font-mono text-sm text-readout">{weight}%</div>
                {rescaled && (
                  <div className="w-[5.25rem] shrink-0 whitespace-nowrap text-right text-xs text-readout" data-testid="analysed-share">
                    {shares && asset in shares ? `${(shares[asset] * 100).toFixed(1)}%` : "0.0%"} used
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => removeAsset(asset)}
                  className="h-7 w-7 shrink-0 text-xs text-label transition-colors duration-150 hover:text-readout"
                >
                  ✕<span className="sr-only"> Remove {asset}</span>
                </button>
              </li>
            ))}
          </ul>
          {rescaled && (
            <p className="mt-2 max-w-[75ch] text-sm text-readout" data-testid="rescale-note">
              The weights are rescaled to sum to 100% (the &ldquo;used&rdquo; column): the analysis is a
              fully invested portfolio, and dollar figures are for the full portfolio value, not
              part of it held as cash.
            </p>
          )}
          <div className="mt-2 flex flex-wrap items-center justify-between gap-x-10 gap-y-3">
            <div className="flex flex-wrap items-center gap-2">
              {addable.length > 0 && <span className="text-xs text-label">Add:</span>}
              {addable.map((a) => (
                <button
                  key={a}
                  type="button"
                  onClick={() => addAsset(a)}
                  className="min-h-7 border border-rule px-2 font-mono text-xs text-label transition-colors duration-150 hover:border-rule-strong hover:text-readout"
                >
                  + {a}
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="legend mr-1">Quick Presets</h3>
              {Object.entries(presets).map(([name, w]) => (
                <button
                  key={name}
                  type="button"
                  title={Object.keys(w).join(" · ")}
                  onClick={() => setWeights(w as Record<string, number>)}
                  className="min-h-7 border border-rule px-2 text-xs font-bold transition-colors duration-150 hover:border-rule-strong"
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        </section>

        {analysis && (
          <>
            <InstrumentPanel metrics={analysis.metrics} initialValue={initialValue} />
            <div className="mt-8 grid gap-x-12 gap-y-10 border-t border-rule pt-5 lg:grid-cols-2">
              <DistributionInstrument buckets={analysis.varDistribution} />
              <FanInstrument points={analysis.monteCarloFan} info={analysis.monteCarlo} />
            </div>
            <StressLadder results={analysis.stressResults} beta={analysis.stressBeta} initialValue={initialValue} />
          </>
        )}

        <footer className="mt-12 flex flex-wrap justify-between gap-x-6 gap-y-1 border-t border-rule pt-5 text-xs text-label">
          <p>
            RiskCanvas · Portfolio risk analytics · MIT License ·{" "}
            {/* From MODE_COPY, like the badge: a two-way ternary here once made
                the `loading` state claim "Demo data" on every cold load. */}
            {copy.footer}
          </p>
          <p>v0.1.0</p>
        </footer>
      </div>
    </main>
  );
}

/**
 * The point of the page, in plain words, built only from figures the instruments
 * below also show: the one-day VaR 95% in dollars and the deepest stress rung.
 */
function Hero({ analysis, initialValue }: { analysis: PortfolioAnalysis; initialValue: number }) {
  const worst = rankByLoss(analysis.stressResults)[0];
  return (
    <div className="lg:min-h-[11.5rem]">
      <h1 className="text-[1.75rem] leading-tight sm:text-[2.25rem] lg:text-[2.5rem]">
        <span className="block">
          On one day in twenty, this portfolio loses at least{" "}
          <span className="figure font-bold">{fmtUsd(analysis.metrics.var95_1d * initialValue)}</span>.
        </span>
        {worst && (
          <span className="mt-1 block">
            Its worst stress path, <span className="lg:whitespace-nowrap">{worst.name}</span>, draws it down{" "}
            <span className="figure-amber font-bold">{fmtPct(Math.abs(worst.portfolioDrawdown))}</span>, or{" "}
            <span className="figure-amber font-bold">{fmtUsd(worst.portfolioDrawdown * initialValue)}</span>.
          </span>
        )}
      </h1>
      <p className="mt-2 text-sm text-label">
        One-day VaR 95% on the synthetic series, and the deepest of the calibrated synthetic S&amp;P 500 paths below.
      </p>
    </div>
  );
}
