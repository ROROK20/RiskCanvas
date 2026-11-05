/**
 * RiskCanvas scripted live-fire smoke driver — the repo's end-to-end gate.
 *
 * Proves in ONE free, repeatable command that RiskCanvas's documented crisis
 * drawdowns reproduce over HTTP for a beta-1 portfolio,
 * that the portfolio's beta actually drives the result (low beta scales down,
 * beta 0 is 0, a missing market series is declared "assumed"), and that the
 * real browser renders the engine's results with honest labels and zero
 * console errors, and that the offline demo mode (engine unreachable) still
 * computes the scenarios the engine's way.
 *
 * It self-provisions: spawns the FastAPI engine (:8004) and the Next.js
 * dashboard (:3004), waits for readiness, asserts, and tears BOTH down in a
 * `finally` (also on failure). Nothing is left running.
 *
 *   node e2e/smoke.mjs        # from the repo root; exit 0 = all pass, 1 = any fail
 *
 * Note: this drives the Playwright *library* from a plain .mjs — NOT
 * @playwright/test, whose TS spec loader breaks under Node 23. Playwright is a
 * devDependency of frontend/ (this repo's only npm package), so it is resolved
 * from there even though this file lives at the repo root.
 */
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { setTimeout as sleep } from "node:timers/promises";
import { connect } from "node:net";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const FRONTEND = join(ROOT, "frontend");
const require = createRequire(join(FRONTEND, "package.json"));
const { chromium } = require("playwright");

const API_PORT = 8004;
const UI_PORT = 3004;
const API = `http://127.0.0.1:${API_PORT}`;
const UI = `http://127.0.0.1:${UI_PORT}`;

// The documented S&P 500 peak-to-trough drawdowns (README
// §Stress Scenarios; re-verified against ^GSPC closes 2026-10-30), keyed by
// the engine's exact scenario name. The engine scales each scenario by the
// portfolio's beta and reports it as beta_used / beta_source, so a portfolio
// with beta exactly 1 must reproduce the table and a low-beta one must not.
const EXPECTED = {
  "Black Monday (Oct 1987)": -0.335,
  "Dot-com Crash (2000-2002)": -0.491,
  "Global Financial Crisis (2008-2009)": -0.568,
  "European Debt Crisis (2011)": -0.194,
  "COVID-19 Crash (Feb-Mar 2020)": -0.339,
  "Fed Rate Hike Bear Market (2022)": -0.254,
};
const MARKET_TOL = 1e-9; // market_drawdown IS the documented value, passed through
const PORT_TOL = 0.0005; // beta-1 portfolio dd = documented dd (wire rounds to 4 dp)
const LOW_BETA = 0.3;

const fmtPct = (n) => `${(n * 100).toFixed(2)}%`; // mirrors frontend page.tsx
const near = (a, b, tol) => Math.abs(a - b) <= tol;
// Same structure and strings, numbers within 1e-12: browsers and Node differ in
// the last bit of Math functions, which the 750-day synthetic series inherit.
function sameRequest(a, b, at = "body") {
  if (typeof a === "number" && typeof b === "number") {
    if (!near(a, b, 1e-12)) throw new Error(`${at}: ${a} vs ${b}`);
  } else if (a && b && typeof a === "object" && typeof b === "object") {
    const keys = Object.keys(a);
    if (keys.join() !== Object.keys(b).join()) throw new Error(`${at}: keys ${keys} vs ${Object.keys(b)}`);
    for (const k of keys) sameRequest(a[k], b[k], `${at}.${k}`);
  } else if (a !== b) {
    throw new Error(`${at}: ${a} vs ${b}`);
  }
}

// ---- pass/fail bookkeeping -------------------------------------------------
let pass = 0,
  fail = 0;
const ok = (n) => {
  pass++;
  console.log(`  ✓ ${n}`);
};
const bad = (n, e) => {
  fail++;
  console.log(`  ✗ ${n}${e ? " — " + e : ""}`);
};
async function check(name, fn) {
  try {
    await fn();
    ok(name);
  } catch (e) {
    bad(name, (e && e.message ? e.message : String(e)).split("\n")[0].slice(0, 180));
  }
}

// ---- server lifecycle -----------------------------------------------------
const children = [];
function launch(label, cmd, args, cwd) {
  // detached:true -> each server leads its own process group so teardown can
  // SIGKILL the whole tree (uvicorn, `next dev` compiler children).
  const child = spawn(cmd, args, { cwd, detached: true, stdio: ["ignore", "pipe", "pipe"] });
  child.label = label;
  child.errtail = [];
  const keep = (d) => {
    for (const line of String(d).split("\n")) {
      if (line.trim()) {
        child.errtail.push(line);
        if (child.errtail.length > 30) child.errtail.shift();
      }
    }
  };
  child.stderr.on("data", keep);
  child.stdout.on("data", keep);
  child.on("error", (e) => keep(`spawn error: ${e.message}`));
  children.push(child);
  return child;
}
function killAll() {
  for (const c of children) {
    if (c.pid == null || c.exitCode !== null) continue;
    try {
      process.kill(-c.pid, "SIGKILL");
    } catch {
      try {
        c.kill("SIGKILL");
      } catch {}
    }
  }
}
for (const sig of ["SIGINT", "SIGTERM"]) {
  process.on(sig, () => {
    killAll();
    process.exit(1);
  });
}
// A rejected promise nothing awaits would end the process before the `finally`
// below, leaving both servers holding their ports.
process.on("unhandledRejection", (e) => {
  console.log(`\n✗ FATAL — ${e && e.message ? e.message : String(e)}`);
  killAll();
  process.exit(1);
});

function portListening(port) {
  return new Promise((resolve) => {
    const s = connect({ host: "127.0.0.1", port }, () => {
      s.destroy();
      resolve(true);
    });
    s.on("error", () => resolve(false));
    s.setTimeout(1000, () => {
      s.destroy();
      resolve(false);
    });
  });
}
async function waitUntil(what, fn, timeoutMs, child) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (child && child.exitCode !== null) {
      throw new Error(`${child.label} exited early (code ${child.exitCode}): ${child.errtail.slice(-6).join(" | ")}`);
    }
    try {
      if (await fn()) return;
    } catch {}
    await sleep(400);
  }
  throw new Error(`timeout waiting for ${what} after ${timeoutMs}ms${child ? ": " + child.errtail.slice(-6).join(" | ") : ""}`);
}

// Deterministic per-asset daily returns => valid, fully-offline portfolios
// (`returns` field means the engine never touches yfinance).
function seriesOf(n, drift, amp, period) {
  return Array.from({ length: n }, (_, i) => drift + amp * Math.sin((2 * Math.PI * i) / period));
}
// x minus its sample projection on m: sample cov(result, m) is exactly 0.
function orthogonal(x, m) {
  const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
  const mx = mean(x), mm = mean(m);
  const xc = x.map((v) => v - mx), mc = m.map((v) => v - mm);
  const k = xc.reduce((a, v, i) => a + v * mc[i], 0) / mc.reduce((a, v) => a + v * v, 0);
  return xc.map((v, i) => v - k * mc[i] + 0.0001);
}
const SPY = seriesOf(300, 0.0004, 0.011, 23);
const AGG = orthogonal(seriesOf(300, 0.0001, 0.004, 29), SPY);
const CASES = {
  // 100% SPY: beta of SPY on its own SPY leg is exactly 1.
  market: { weights: { SPY: 1 }, name: "smoke beta-1", returns: { SPY } },
  // 30% SPY + 70% of a series uncorrelated with SPY: beta exactly 0.3.
  lowBeta: { weights: { SPY: LOW_BETA, AGG: 1 - LOW_BETA }, name: "smoke beta-0.3", returns: { SPY, AGG } },
  // An explicit beta of 0 must mean 0, not fall through to the 1.0 default.
  zeroBeta: { weights: { SPY: 1 }, name: "smoke beta-0", returns: { SPY }, beta: 0 },
  // No market series at all: the engine must SAY beta 1.0 is assumed.
  noMarket: { weights: { QQQ: 0.6, AGG: 0.4 }, name: "smoke no-market", returns: { QQQ: seriesOf(300, 0.0006, 0.014, 19), AGG } },
};

let browser;
const t0 = Date.now();
try {
  // Preflight: refuse to run against a stale server on our ports (that would be
  // a false pass on stale code). 8004/3004 are this repo's own ports.
  for (const [port, name] of [[API_PORT, "API"], [UI_PORT, "frontend"]]) {
    if (await portListening(port)) {
      throw new Error(`port ${port} (${name}) already in use — stop the stale process and re-run`);
    }
  }

  console.log("Starting servers:");
  const apiProc = launch("api", "python3", ["-m", "uvicorn", "riskcanvas.api.app:app", "--host", "127.0.0.1", "--port", String(API_PORT)], ROOT);
  await check("API up (/health ok)", () =>
    waitUntil(
      "API /health",
      async () => {
        const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2000) });
        return r.ok && (await r.json()).status === "ok";
      },
      30000,
      apiProc,
    ),
  );

  const uiProc = launch("frontend", "npm", ["run", "dev"], FRONTEND);
  await check("frontend listening (:3004)", () => waitUntil("frontend port", () => portListening(UI_PORT), 60000, uiProc));

  // ---- API live-fire: beta-scaled stress over HTTP ---------------------------
  const stress = async (body) => {
    const r = await fetch(`${API}/portfolio/stress`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15000),
    });
    if (r.status !== 200) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 120)}`);
    const j = await r.json();
    return { ...j, byName: Object.fromEntries((j.scenarios || []).map((s) => [s.name, s])) };
  };
  const out = {};
  console.log("API POST /portfolio/stress:");
  for (const [k, body] of Object.entries(CASES)) {
    await check(`POST /portfolio/stress (${body.name}) -> 200, 6 scenarios`, async () => {
      out[k] = await stress(body);
      if (out[k].scenarios.length !== 6) throw new Error(`got ${out[k].scenarios.length}`);
    });
  }
  const mkt1 = out.market;
  await check("beta-1 portfolio: beta_source spy_leg, beta_used 1", () => {
    if (!mkt1) throw new Error("no response");
    if (mkt1.beta_source !== "spy_leg") throw new Error(`beta_source ${mkt1.beta_source}`);
    if (!near(mkt1.beta_used, 1, 1e-6)) throw new Error(`beta_used ${mkt1.beta_used}`);
  });
  for (const [name, mkt] of Object.entries(EXPECTED)) {
    // Every path ends at its trough, so at beta 1 the end-of-window loss is
    // the max drawdown too: both must equal the table.
    await check(`API ${name}: market ${fmtPct(mkt)}, beta-1 drawdown and end-of-window loss = market`, () => {
      const s = mkt1 && mkt1.byName[name];
      if (!s) throw new Error("scenario missing from response");
      if (!near(s.market_drawdown, mkt, MARKET_TOL)) throw new Error(`market_drawdown ${s.market_drawdown} ≠ ${mkt}`);
      if (!near(s.portfolio_drawdown, mkt, PORT_TOL)) throw new Error(`portfolio_drawdown ${s.portfolio_drawdown} not within ${PORT_TOL} of ${mkt}`);
      if (!near(s.total_loss, mkt, PORT_TOL)) throw new Error(`total_loss ${s.total_loss} not within ${PORT_TOL} of ${mkt}`);
    });
  }
  await check(`low-beta portfolio: beta_used ${LOW_BETA}, every scenario scaled down`, () => {
    const l = out.lowBeta;
    if (!l) throw new Error("no response");
    if (l.beta_source !== "spy_leg" || !near(l.beta_used, LOW_BETA, 1e-6)) throw new Error(`beta ${l.beta_used} (${l.beta_source})`);
    for (const [name, mkt] of Object.entries(EXPECTED)) {
      const p = l.byName[name].portfolio_drawdown;
      if (!(p < 0 && p > 0.5 * mkt)) throw new Error(`${name}: ${p} not shallower than half of ${mkt}`);
    }
  });
  await check("explicit beta 0: beta_used 0, every drawdown 0", () => {
    const z = out.zeroBeta;
    if (!z) throw new Error("no response");
    if (z.beta_source !== "explicit" || z.beta_used !== 0) throw new Error(`beta ${z.beta_used} (${z.beta_source})`);
    for (const s of z.scenarios) if (s.portfolio_drawdown !== 0) throw new Error(`${s.name}: ${s.portfolio_drawdown}`);
  });
  await check('no market series: beta_source "assumed", stated in beta_note', () => {
    const a = out.noMarket;
    if (!a) throw new Error("no response");
    if (a.beta_source !== "assumed" || a.beta_used !== 1) throw new Error(`beta ${a.beta_used} (${a.beta_source})`);
    if (!/ASSUMED/.test(a.beta_note || "")) throw new Error(`beta_note: ${a.beta_note}`);
  });

  // ---- UI live-fire: real browser, real backend, 0 console errors ----------
  console.log("UI (headless Chromium @ :3004):");
  browser = await chromium.launch();
  const page = await (await browser.newContext()).newPage();
  const consoleErrors = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
  let firstAnalyzeBody = null;
  page.on("request", (r) => {
    if (!firstAnalyzeBody && r.method() === "POST" && r.url().endsWith("/portfolio/analyze")) firstAnalyzeBody = r.postData();
  });
  await page.goto(UI, { waitUntil: "domcontentloaded", timeout: 60000 });

  await check('dashboard shows "Live engine" (real backend, not demo)', async () => {
    // exact:true so this can't false-match running text; waitFor then blocks
    // through the brief demo→live flash on cold start until the badge settles on
    // the real-engine state.
    await page.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 30000 });
    const body = await page.innerText("body");
    if (body.includes("Demo data")) throw new Error("demo badge present");
  });

  // Full engine scenario names appear ONLY in live mode (the in-browser mock
  // uses short names), so asserting them + the documented market % proves the
  // render came from the real engine, not the fallback.
  await page.getByText("Global Financial Crisis (2008-2009)").first().waitFor({ timeout: 15000 });
  const uiText = await page.innerText("body");
  for (const [name, mkt] of Object.entries(EXPECTED)) {
    await check(`UI ${name}: rendered, Market ${fmtPct(mkt)}`, () => {
      if (!uiText.includes(name)) throw new Error("scenario name not rendered");
      if (!uiText.includes(`Market: ${fmtPct(mkt)}`)) throw new Error(`missing "Market: ${fmtPct(mkt)}"`);
    });
  }
  await check("UI discloses synthetic inputs", () => {
    if (!uiText.includes("Synthetic inputs, not market data.")) throw new Error("disclosure missing");
  });
  await check("UI shows the engine-estimated stress beta and its source", () => {
    const b = uiText.match(/Stress beta (-?\d+\.\d+) \(estimated by the engine from the synthetic market series\)/);
    if (!b) throw new Error("stress beta line missing or not engine-estimated");
    const v = Number(b[1]);
    if (!(v > 0.5 && v < 1.5)) throw new Error(`implausible beta ${v} for the default equity-heavy portfolio`);
  });
  await check("UI Monte Carlo label states the engine's path count", () => {
    if (!uiText.includes("5,000 paths, GBM")) throw new Error('missing "5,000 paths, GBM"');
  });
  await check("a non-positive Portfolio Value is refused, not used", async () => {
    // The input accepted 0 and negative amounts.
    const input = page.getByTestId("portfolio-value");
    await input.fill("-5");
    await page.getByTestId("portfolio-value-error").waitFor({ timeout: 10000 });
    if (!(await page.innerText("body")).includes("$16,217")) throw new Error("dollar VaR must use the last valid value");
    await input.fill("1000000");
  });
  await check("weights not totalling 100% show the rescaled shares the engine analyses", async () => {
    // Removing AGG showed "Total: 90%" (reads as
    // 10% cash) while the engine analysed SPY at 44.4%, unsaid.
    await page.getByRole("button", { name: "✕" }).last().click();   // AGG, the last row
    await page.getByText("analysed as 100% invested").waitFor({ timeout: 15000 });
    const total = await page.getByTestId("weight-total").innerText();
    if (!total.includes("Total: 90%")) throw new Error(`total reads "${total}"`);
    const spy = await page.getByTestId("analysed-share").first().innerText();
    if (!spy.includes("44.4% used")) throw new Error(`SPY share reads "${spy}"`);
    if (!(await page.getByTestId("rescale-note").isVisible())) throw new Error("no rescale note");
  });
  await check('emptied portfolio: "add an asset", no stale numbers, no crash', async () => {
    // Removing every asset crashed the in-browser
    // fallback and left the previous portfolio's results on screen under
    // "Connecting…". (Runs before the console-error check, which would see
    // that crash as a pageerror.)
    while ((await page.getByRole("button", { name: "✕" }).count()) > 0) {
      await page.getByRole("button", { name: "✕" }).first().click();
    }
    await page.getByText("Add an asset to see its risk").waitFor({ timeout: 15000 });
    const body = await page.innerText("body");
    if (!body.includes("No assets")) throw new Error('badge is not "No assets"');
    if (/VaR 95% \(1d\)|Crisis Stress Tests|Monte Carlo Fan/i.test(body)) throw new Error("stale results still on screen");
  });
  await check("0 console errors", () => {
    if (consoleErrors.length) throw new Error(`${consoleErrors.length}: ${consoleErrors.slice(0, 3).join(" | ")}`);
  });
  // tests/test_capture.py recomputes the published numbers from this request,
  // so it must still be what the default dashboard sends.
  await check("the default dashboard sends the request docs/capture_request.json records", () => {
    if (!firstAnalyzeBody) throw new Error("no POST /portfolio/analyze seen");
    sameRequest(JSON.parse(firstAnalyzeBody), JSON.parse(readFileSync(join(ROOT, "docs", "capture_request.json"), "utf8")));
  });

  // A prerendered page must not format numbers in the viewer's locale: "$16.217"
  // in a German browser against the server's "$16,217" is a hydration mismatch.
  console.log("UI in a German browser:");
  const german = await (await browser.newContext({ locale: "de-DE" })).newPage();
  const germanErrors = [];
  german.on("console", (m) => {
    if (m.type() === "error") germanErrors.push(m.text());
  });
  german.on("pageerror", (e) => germanErrors.push(`pageerror: ${e.message}`));
  await german.goto(UI, { waitUntil: "domcontentloaded", timeout: 60000 });
  await check("de-DE: the engine's numbers render, formatted as on the server, 0 console errors", async () => {
    await german.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 60000 });
    await german.getByText("Global Financial Crisis (2008-2009)").first().waitFor({ timeout: 30000 });
    const body = await german.innerText("body");
    if (!/\$\d{1,3},\d{3}/.test(body) || /\$\d{1,3}\.\d{3}\b/.test(body)) throw new Error("money is not formatted en-US");
    if (germanErrors.length) throw new Error(`${germanErrors.length}: ${germanErrors.slice(0, 2).join(" | ").slice(0, 160)}`);
  });

  // The sliders must not flood the engine (the free plan the README warns about):
  // ten steps at once are one ask of three POSTs, and a change aborts an ask in flight.
  console.log("UI slider burst:");
  const burst = await (await browser.newContext()).newPage();
  const burstErrors = [];
  burst.on("console", (m) => {
    if (m.type() === "error") burstErrors.push(m.text());
  });
  await burst.goto(UI, { waitUntil: "domcontentloaded", timeout: 60000 });
  await burst.getByText("Live engine", { exact: true }).first().waitFor({ timeout: 60000 });
  const stepSlider = (from, steps) =>
    burst.locator('input[type="range"]').first().evaluate((el, [start, n]) => {
      const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
      for (let i = 1; i <= n; i++) {
        setValue.call(el, String(start + i));
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
    }, [from, steps]);
  // `next dev` calls the engine on localhost, the API checks above on 127.0.0.1.
  const isEngineCall = (r) => r.method() === "POST" && /:8004\/portfolio\//.test(r.url());
  const isMonteCarlo = (r) => isEngineCall(r) && /\/portfolio\/montecarlo/.test(r.url());
  await check("ten slider steps at once send one ask: three POSTs", async () => {
    const posts = [];
    const onRequest = (r) => {
      if (isEngineCall(r)) posts.push(r.url());
    };
    burst.on("request", onRequest);
    await Promise.all([burst.waitForResponse((r) => isMonteCarlo(r.request()), { timeout: 60000 }), stepSlider(40, 10)]);
    await sleep(1000); // anything still to come would have been sent by now
    burst.off("request", onRequest);
    if (posts.length !== 3) throw new Error(`${posts.length} POSTs: ${posts.join(" ")}`);
  });
  await check("a change aborts the ask in flight", async () => {
    const failed = burst.waitForEvent("requestfailed", { predicate: isEngineCall, timeout: 30000 }).catch((e) => e);
    await Promise.all([burst.waitForRequest(isMonteCarlo, { timeout: 30000 }), stepSlider(50, 1)]);
    const answered = burst.waitForResponse((r) => isMonteCarlo(r.request()) && r.ok(), { timeout: 60000 });
    await Promise.all([answered, stepSlider(51, 1)]);
    const stale = await failed;
    if (stale instanceof Error) throw stale;
    if (!/ERR_ABORTED/.test(stale.failure()?.errorText || "")) throw new Error(`${stale.url()}: ${stale.failure()?.errorText}`);
  });
  await check("the burst leaves 0 console errors", () => {
    if (burstErrors.length) throw new Error(`${burstErrors.length}: ${burstErrors.slice(0, 2).join(" | ").slice(0, 160)}`);
  });

  // The engine answers, with an error: the page must not call it unreachable.
  console.log("UI with the engine answering 500:");
  const brokenCtx = await browser.newContext();
  const cors = { "access-control-allow-origin": "*", "access-control-allow-headers": "*", "access-control-allow-methods": "*" };
  await brokenCtx.route(/:8004\/portfolio\//, (route) =>
    route.request().method() === "OPTIONS"
      ? route.fulfill({ status: 204, headers: cors })
      : route.fulfill({ status: 500, headers: cors, contentType: "application/json", body: '{"detail":"boom"}' }),
  );
  const broken = await brokenCtx.newPage();
  await broken.goto(UI, { waitUntil: "domcontentloaded", timeout: 60000 });
  await check("engine error: the banner and footer say it answered with an error", async () => {
    await broken.getByText("Demo data", { exact: true }).first().waitFor({ timeout: 30000 });
    const caption = await broken.getByTestId("data-caption").innerText();
    if (!caption.includes("The engine answered with an error (HTTP 500)")) throw new Error(caption);
    const body = await broken.innerText("body");
    if (!body.includes("the engine answered with an error (HTTP 500)")) throw new Error("footer does not say so");
    if (/unreachable|could not be reached/i.test(body)) throw new Error("it calls the engine unreachable");
  });

  // Demo mode: the same dashboard with the engine unreachable from the
  // browser (its own context; the aborted requests' console noise is
  // expected there and not counted above). The demo
  // multiplied each drawdown by beta, so 100% TSLA (beta 1.9) printed a GFC
  // loss of -107.92%, beyond the portfolio's value.
  console.log("UI demo mode (engine blocked in the browser):");
  const demoCtx = await browser.newContext();
  await demoCtx.route(/:8004\//, (route) => route.abort());
  const demo = await demoCtx.newPage();
  await demo.goto(UI, { waitUntil: "domcontentloaded", timeout: 60000 });
  await check('demo mode: "Demo data" badge with the engine unreachable', async () => {
    await demo.getByText("Demo data", { exact: true }).first().waitFor({ timeout: 30000 });
    const caption = await demo.getByTestId("data-caption").innerText();
    if (!caption.includes("The engine could not be reached in time")) throw new Error(caption);
    if (/start the api/i.test(await demo.innerText("body"))) throw new Error("it tells a visitor to start the API");
  });
  // frontend/public/og.png is this state; docs/capture.json records its figures.
  await check("demo mode: the default cards are the ones the social-preview image shows", async () => {
    const recorded = JSON.parse(readFileSync(join(ROOT, "docs", "capture.json"), "utf8")).og.metric_cards;
    const body = await demo.innerText("body");
    const cards = (body.match(/VaR 95% \(1d\)[\s\S]*?VaR 95% \(10d\)\s*\S+/i) || [""])[0].replace(/\s+/g, " ");
    if (cards !== recorded) throw new Error(`page: ${cards} | og.png: ${recorded}`);
  });
  await check("demo mode: 100% TSLA (beta 1.9) compounds the path: GFC -79.94%, every loss above -100%", async () => {
    await demo.getByRole("button", { name: "+ TSLA" }).click();
    for (let i = 0; i < 5; i++) await demo.getByRole("button", { name: "✕" }).first().click();
    const held = await demo.getByRole("button", { name: "✕" }).count();
    if (held !== 1) throw new Error(`expected TSLA alone, ${held} assets held`);
    const gfc = demo.locator("div.rounded-lg.p-3", { hasText: "Global Financial Crisis" }).first();
    try {
      await gfc.getByText("-79.94%", { exact: true }).waitFor({ timeout: 15000 });
    } catch {
      throw new Error(`GFC card reads: ${(await gfc.innerText()).replace(/\s+/g, " ")}`);
    }
    const losses = (await demo.locator("div.rounded-lg.p-3 .text-lg.font-bold").allInnerTexts()).map((t) => parseFloat(t));
    if (losses.length !== 6) throw new Error(`expected 6 scenarios, got ${losses.length}`);
    const bad = losses.filter((v) => !(v > -100 && v <= 0));
    if (bad.length) throw new Error(`losses at or beyond -100%: ${bad.join(", ")}`);
  });
  await check("demo mode: 100% BTC states the factor loading is not the engine's estimate and warns of the weak fit, as the engine does", async () => {
    await demo.getByRole("button", { name: "+ BTC" }).click();
    await demo.getByRole("button", { name: "✕" }).first().click();
    const warnings = demo.getByTestId("stress-warnings");
    await warnings.getByText("The S&P 500 explains").waitFor({ timeout: 15000 });
    const text = await warnings.innerText();
    if (!/explains \d+% of this portfolio's daily variance/.test(text)) throw new Error(text);
    const label = await demo.getByTestId("stress-beta").innerText();
    if (!/1\.30/.test(label) || !/not the engine's OLS estimate/.test(label)) throw new Error(label);
  });
} catch (e) {
  fail++;
  console.log(`\n✗ FATAL — ${e && e.message ? e.message : String(e)}`);
} finally {
  if (browser) {
    try {
      await browser.close();
    } catch {}
  }
  killAll();
}

console.log(`\n${pass} passed, ${fail} failed  (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
process.exit(fail === 0 ? 0 : 1);
