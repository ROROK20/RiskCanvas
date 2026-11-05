"use client";

import { useState, useEffect } from "react";
import {
  analyzePortfolio,
  AVAILABLE_ASSETS,
  SYNTHETIC_DAYS,
  type BetaSource,
  type PortfolioAnalysis,
} from "@/lib/mockRisk";
import { API_URL } from "@/lib/api";
import { liveAnalyzePortfolio } from "@/lib/liveRisk";
import { fanBands, fanDomain } from "@/lib/fan.mjs";
import { analysedShares, heldFractions, RECOMPUTE_CAPTION } from "@/lib/dashboardState.mjs";
import { debouncedRequest, demoCopy, demoReason, ENGINE_DEBOUNCE_MS, NO_ENGINE } from "@/lib/engineRequest.mjs";
import { tailShare } from "@/lib/histogram.mjs";
import {
  Shield, Github, AlertTriangle, Activity, Target, Zap, Briefcase, Wifi, WifiOff,
} from "lucide-react";
import {
  BarChart, Bar, ComposedChart, Area, Line, ResponsiveContainer,
  XAxis, YAxis, CartesianGrid, Tooltip, Cell, ReferenceLine,
} from "recharts";
import clsx from "clsx";

// Data source: "live" = real Python engine, "demo" = in-browser numbers (no engine
// in this build, or the engine failed: demoCopy says which), "loading" = waiting on
// the engine (the in-browser preview is on screen).
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
    math: "Every risk number below is computed by the Python engine (numpy/scipy) from these synthetic series; the return histogram is binned in your browser from the same series.",
  },
  loading: {
    badge: "Connecting…",
    title: "Connecting to the engine…",
    footer: "Connecting to the Python engine…",
    // Replaced by RECOMPUTE_CAPTION[shown]: what is on screen while loading.
    math: RECOMPUTE_CAPTION.preview,
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

const BETA_SOURCE_LABEL: Record<BetaSource, string> = {
  explicit: "supplied explicitly",
  market_returns: "estimated by the engine from the synthetic market series",
  spy_leg: "estimated by the engine from the SPY series",
  yfinance_spy: "estimated by the engine vs SPY (yfinance)",
  assumed: "ASSUMED — no market series was available",
  demo_factor_loading: "in-browser: the synthetic portfolio's factor loading, not the engine's OLS estimate",
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
  const [analysis, setAnalysis] = useState<PortfolioAnalysis | null>(() => {
    const held = heldFractions(weights);
    return held ? analyzePortfolio({ weights: held, initialValue }) : null;
  });
  const [shown, setShown] = useState<Shown>("preview");
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
    <main className="min-h-screen">
      <header className="border-b border-border bg-panel/30 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-br from-red-500 to-orange-500 rounded-lg flex items-center justify-center glow">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">RiskCanvas</h1>
              <p className="text-xs text-gray-500">Portfolio Risk Analytics</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div
              className={clsx(
                "flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md border",
                mode === "live" && "border-accent-green/40 bg-accent-green/10 text-accent-green",
                mode === "demo" && "border-yellow-500/40 bg-yellow-500/10 text-yellow-400",
                mode === "loading" && "border-border bg-panel/40 text-gray-400",
                mode === "empty" && "border-border bg-panel/40 text-gray-400",
                mode === "error" && "border-accent-red/40 bg-accent-red/10 text-accent-red",
              )}
              title={copy.title}
            >
              {mode === "live" ? (
                <Wifi className="w-3.5 h-3.5" />
              ) : (
                <WifiOff className="w-3.5 h-3.5" />
              )}
              {copy.badge}
            </div>
            <a href="https://github.com/ROROK20/RiskCanvas" className="flex items-center gap-2 px-3 py-1.5 text-xs border border-border hover:border-accent-blue rounded-md">
              <Github className="w-3.5 h-3.5" /> View Source
            </a>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Synthetic-input disclosure: true in every mode. */}
        <div
          data-testid="synthetic-disclosure"
          className="mb-6 rounded-lg border border-yellow-500/30 bg-yellow-500/5 px-4 py-3 text-xs text-gray-300"
        >
          <span className="font-semibold text-yellow-400">Synthetic inputs, not market data.</span>{" "}
          Each asset is a seeded {SYNTHETIC_DAYS}-day series from a one-factor model (shared
          fat-tailed market shock, plausible betas and volatilities) generated in your browser.{" "}
          <span data-testid="data-caption">{caption}</span>
        </div>

        {/* Portfolio Builder */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
          <div className="lg:col-span-2 bg-panel border border-border rounded-lg p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold flex items-center gap-2">
                <Briefcase className="w-4 h-4" /> Portfolio Allocation
              </h3>
              <div className={clsx(
                "text-xs px-2 py-1 rounded",
                Math.abs(totalWeight - 100) < 0.1 ? "bg-accent-green/20 text-accent-green" : "bg-yellow-500/20 text-yellow-400",
              )} data-testid="weight-total">
                Total: {totalWeight.toFixed(0)}%{rescaled ? " → analysed as 100% invested" : ""}
              </div>
            </div>
            <div className="space-y-3 mb-4">
              {Object.entries(weights).map(([asset, weight]) => (
                <div key={asset} className="flex items-center gap-3">
                  <div className="w-16 font-bold text-sm">{asset}</div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={1}
                    value={weight}
                    onChange={(e) => updateWeight(asset, Number(e.target.value))}
                    className="flex-1 accent-accent-blue"
                  />
                  <div className="w-12 text-right tabular-nums text-sm">{weight}%</div>
                  {rescaled && (
                    <div className="w-20 text-right tabular-nums text-[11px] text-yellow-400" data-testid="analysed-share">
                      {shares && asset in shares ? `${(shares[asset] * 100).toFixed(1)}%` : "0.0%"} used
                    </div>
                  )}
                  <button onClick={() => removeAsset(asset)} className="text-xs text-gray-500 hover:text-accent-red">✕</button>
                </div>
              ))}
            </div>
            {rescaled && (
              <p className="text-[11px] text-yellow-400 mb-3" data-testid="rescale-note">
                The weights are rescaled to sum to 100% (the &ldquo;used&rdquo; column): the analysis is a
                fully invested portfolio, and dollar figures are for the full portfolio value, not
                part of it held as cash.
              </p>
            )}
            <div className="flex flex-wrap gap-2 pt-3 border-t border-border">
              <span className="text-xs text-gray-500 self-center">Add:</span>
              {AVAILABLE_ASSETS.filter((a) => !(a in weights)).map((a) => (
                <button key={a} onClick={() => addAsset(a)} className="px-2 py-1 text-xs border border-border hover:border-accent-blue rounded">
                  + {a}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-panel border border-border rounded-lg p-5">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Target className="w-4 h-4" /> Quick Presets
            </h3>
            <div className="space-y-2 mb-4">
              {Object.entries(presets).map(([name, w]) => (
                <button
                  key={name}
                  onClick={() => setWeights(w as Record<string, number>)}
                  className="w-full text-left px-3 py-2 text-xs border border-border hover:border-accent-blue rounded transition-colors"
                >
                  <div className="font-bold">{name}</div>
                  <div className="text-gray-500">{Object.keys(w).join(" · ")}</div>
                </button>
              ))}
            </div>
            <div>
              <label className="text-xs text-gray-500 block mb-1">Portfolio Value ($)</label>
              <input
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
                  "w-full bg-bg border rounded px-2 py-1.5 text-sm focus:border-accent-blue outline-none",
                  valueValid ? "border-border" : "border-accent-red",
                )}
                step={50_000}
                data-testid="portfolio-value"
              />
              {!valueValid && (
                <p className="text-[11px] text-accent-red mt-1" data-testid="portfolio-value-error">
                  Enter a positive amount; figures use the last valid value (${initialValue.toLocaleString("en-US")}).
                </p>
              )}
            </div>
          </div>
        </div>

        {analysis ? (
          <Results analysis={analysis} initialValue={initialValue} />
        ) : (
          <div data-testid="no-results" className="bg-panel border border-border rounded-lg p-8 mb-6 text-center">
            <div className="text-sm font-semibold mb-1">{NO_RESULT_COPY[mode].title}</div>
            <div className="text-xs text-gray-500">{mode === "error" ? errorMsg : NO_RESULT_COPY[mode].detail}</div>
          </div>
        )}

        <footer className="mt-12 pt-6 border-t border-border text-xs text-gray-500 flex justify-between">
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

function Results({ analysis, initialValue }: { analysis: PortfolioAnalysis; initialValue: number }) {
  const m = analysis.metrics;
  const fmtPct = (n: number) => `${(n * 100).toFixed(2)}%`;
  const fmtMoney = (n: number) => `$${Math.abs(n * initialValue).toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
  // Ratios the engine reports as null when undefined or unbounded.
  const fmtRatio = (n: number | null) => (n == null ? "N/A" : n.toFixed(2));
  const beta = analysis.stressBeta;
  const fan = fanBands(analysis.monteCarloFan);
  return (
    <>
        {/* Key Metrics */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">VaR 95% (1d)</div>
            <div className="text-xl font-bold tabular-nums text-accent-red">{fmtPct(m.var95_1d)}</div>
            <div className="text-[10px] text-gray-500">{fmtMoney(m.var95_1d)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">VaR 99% (1d)</div>
            <div className="text-xl font-bold tabular-nums text-accent-red">{fmtPct(m.var99_1d)}</div>
            <div className="text-[10px] text-gray-500">{fmtMoney(m.var99_1d)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">CVaR 95%</div>
            <div className="text-xl font-bold tabular-nums text-accent-red">{fmtPct(m.cvar95)}</div>
            <div className="text-[10px] text-gray-500">{fmtMoney(m.cvar95)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">Max DD</div>
            <div className="text-xl font-bold tabular-nums text-accent-red">{fmtPct(m.maxDrawdown)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">Sharpe</div>
            <div className={clsx("text-xl font-bold tabular-nums", m.sharpe != null && m.sharpe > 1 ? "text-accent-green" : "")}>{fmtRatio(m.sharpe)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">Sortino</div>
            <div className="text-xl font-bold tabular-nums">{fmtRatio(m.sortino)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">CAGR</div>
            <div className={clsx("text-xl font-bold tabular-nums", m.cagr > 0 ? "text-accent-green" : "text-accent-red")}>{fmtPct(m.cagr)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">Annual Vol</div>
            <div className="text-xl font-bold tabular-nums">{fmtPct(m.annualVol)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">Calmar</div>
            <div className="text-xl font-bold tabular-nums">{fmtRatio(m.calmar)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">Skewness</div>
            <div className="text-xl font-bold tabular-nums">{fmtRatio(m.skewness)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">Excess Kurt.</div>
            <div className="text-xl font-bold tabular-nums">{fmtRatio(m.excessKurtosis)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-gray-500 uppercase">VaR 95% (10d)</div>
            <div className="text-xl font-bold tabular-nums text-accent-red">{fmtPct(m.var95_10d)}</div>
          </div>
        </div>

        {/* Return distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          <div className="bg-panel border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold mb-1 flex items-center gap-2">
              <Activity className="w-4 h-4" /> Return Distribution
            </h3>
            <p className="text-xs text-gray-500 mb-3" data-testid="tail-caption">
              Red bars = buckets beyond the 95% VaR ({(tailShare(analysis.varDistribution) * 100).toFixed(1)}% of days)
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={analysis.varDistribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1c2540" />
                <XAxis dataKey="bucket" stroke="#6b7280" tick={{ fontSize: 9 }} interval={2} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ background: "#0f1525", border: "1px solid #1c2540", fontSize: "0.75rem" }} />
                <Bar dataKey="count">
                  {analysis.varDistribution.map((b, i) => (
                    <Cell key={i} fill={b.isTail ? "#ef4444" : "#3b82f6"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-panel border border-border rounded-lg p-4">
            <h3 className="text-sm font-semibold mb-1 flex items-center gap-2">
              <Zap className="w-4 h-4" /> Monte Carlo Fan ({analysis.monteCarlo.horizonDays} days,{" "}
              {analysis.monteCarlo.paths.toLocaleString("en-US")} paths, {analysis.monteCarlo.method},{" "}
              {analysis.monteCarlo.drift} drift)
            </h3>
            <p className="text-xs text-gray-500 mb-3">
              Simulated portfolio value by day: 5th–95th percentile (outer band), 25th–75th (inner band), median (line)
            </p>
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={fan}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1c2540" />
                <XAxis dataKey="day" stroke="#6b7280" tick={{ fontSize: 10 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} domain={fanDomain(fan)}
                  tickFormatter={(v) => `${(v).toFixed(2)}x`} />
                <Tooltip contentStyle={{ background: "#0f1525", border: "1px solid #1c2540", fontSize: "0.75rem" }}
                  formatter={(v: number | number[]) =>
                    Array.isArray(v) ? `${v[0].toFixed(3)}x – ${v[1].toFixed(3)}x` : `${v.toFixed(3)}x`} />
                <ReferenceLine y={1.0} stroke="#6b7280" strokeDasharray="4 4" />
                <Area type="monotone" dataKey="outer" name="5th–95th" stroke="none" fill="#3b82f6" fillOpacity={0.15} />
                <Area type="monotone" dataKey="inner" name="25th–75th" stroke="none" fill="#3b82f6" fillOpacity={0.35} />
                <Line type="monotone" dataKey="median" name="median" stroke="#3b82f6" strokeWidth={2} dot={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Stress tests */}
        <div className="bg-panel border border-border rounded-lg p-5 mb-6">
          <h3 className="text-sm font-semibold flex items-center gap-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-accent-red" /> Crisis Stress Tests
          </h3>
          <p className="text-xs text-gray-500 mb-1" data-testid="stress-disclosure">
            Equity-market shocks only: calibrated synthetic S&amp;P 500 paths matched to each
            crisis&apos;s documented peak-to-trough drawdown (not historical replays); portfolio
            path = beta × S&amp;P path. No interest-rate, credit or FX factor, so bonds, gold and
            other low-beta holdings move only through their equity beta.
          </p>
          <p
            data-testid="stress-beta"
            title={beta.note}
            className={clsx("text-xs mb-4", beta.source === "assumed" ? "text-yellow-400" : "text-gray-400")}
          >
            Stress beta {beta.value.toFixed(2)} ({BETA_SOURCE_LABEL[beta.source]})
          </p>
          {beta.warnings.length > 0 && (
            <ul data-testid="stress-warnings" className="text-xs text-yellow-400 mb-4 -mt-2 space-y-1">
              {beta.warnings.map((w) => (
                <li key={w}>⚠ {w}</li>
              ))}
            </ul>
          )}
          <div className="space-y-3">
            {analysis.stressResults.map((s) => {
              const lossDollar = Math.abs(s.portfolioDrawdown * initialValue);
              const pct = Math.min(Math.abs(s.portfolioDrawdown), 1.0);
              return (
                <div key={s.name} className="border border-border rounded-lg p-3">
                  <div className="flex items-start justify-between mb-2 flex-wrap gap-2">
                    <div>
                      <div className="font-semibold text-sm">{s.name}</div>
                      <div className="text-xs text-gray-500">{s.description}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold tabular-nums text-accent-red">
                        {fmtPct(s.portfolioDrawdown)}
                      </div>
                      <div className="text-xs text-gray-500">
                        ${lossDollar.toLocaleString("en-US", { maximumFractionDigits: 0 })} loss
                      </div>
                    </div>
                  </div>
                  <div className="relative h-2 bg-bg/50 rounded overflow-hidden">
                    <div className="absolute top-0 left-0 h-full bg-accent-red/60" style={{ width: `${pct * 100}%` }} />
                  </div>
                  <div className="flex justify-between text-[10px] text-gray-500 mt-1">
                    <span>Market: {fmtPct(s.marketDrawdown)}</span>
                    <span>Duration: {s.durationDays} trading days</span>
                    <span title="Trading days to regain the pre-crisis level, compounding at the portfolio's historical growth rate">
                      Recovery at hist. growth: ~{s.recoveryDays != null && s.recoveryDays > 0 ? `${s.recoveryDays}d` : "N/A"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

    </>
  );
}
