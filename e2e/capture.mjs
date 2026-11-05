/**
 * Regenerate BOTH published images, in one run, each from the state it describes:
 *   docs/screenshot.png      the README screenshot (1440 px wide, full page, 2x),
 *                            from the live stack: badge "Live engine"
 *   frontend/public/og.png   the social-preview image every shared link shows
 *                            (1200 x 630 at 2x: banner, allocation, metric cards),
 *                            from a production build with no engine, which is what
 *                            a frontend-only deploy serves: badge "Demo data"
 * docs/capture.json, the SHA-256 of both plus each one's badge and cards, and
 * docs/capture_request.json, the /portfolio/analyze body the dashboard sent.
 * tests/test_capture.py fails if either image changes without the other, if the
 * preview stops being the demo state, or if the engine stops computing the
 * screenshot's cards from that request; e2e/smoke.mjs fails if the demo stops
 * showing the preview's cards.
 *
 *   node e2e/capture.mjs      # or: make capture   (from the repo root)
 *
 * It builds the frontend with NEXT_PUBLIC_API_URL pointed at the local engine,
 * starts the FastAPI engine (:8004) and `next start` (:3004), and captures; then
 * rebuilds without the variable, restarts `next start`, captures the preview, and
 * tears everything down.
 */
import { spawn, execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { readFileSync, writeFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const FRONTEND = join(ROOT, "frontend");
const require = createRequire(join(FRONTEND, "package.json"));
const { chromium } = require("playwright");
const API = "http://127.0.0.1:8004";
const UI = "http://127.0.0.1:3004";
const SHOT = join(ROOT, "docs", "screenshot.png");
const OG = join(FRONTEND, "public", "og.png");
const MANIFEST = join(ROOT, "docs", "capture.json");
const REQUEST = join(ROOT, "docs", "capture_request.json");

const children = [];
function launch(cmd, args, cwd, env) {
  const child = spawn(cmd, args, { cwd, env: { ...process.env, ...env }, detached: true, stdio: "ignore" });
  children.push(child);
  return child;
}
function killAll() {
  for (const c of children) {
    try { process.kill(-c.pid, "SIGKILL"); } catch { try { c.kill("SIGKILL"); } catch {} }
  }
}
for (const sig of ["SIGINT", "SIGTERM"]) process.on(sig, () => { killAll(); process.exit(1); });

async function up(url, ms) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) {
    try { if ((await fetch(url)).ok) return; } catch {}
    await sleep(500);
  }
  throw new Error(`${url} not up after ${ms} ms`);
}
const sha256 = (f) => createHash("sha256").update(readFileSync(f)).digest("hex");

function build(apiUrl) {
  const env = { ...process.env };
  if (apiUrl) env.NEXT_PUBLIC_API_URL = apiUrl;
  else delete env.NEXT_PUBLIC_API_URL;
  console.log(`building the frontend ${apiUrl ? "with NEXT_PUBLIC_API_URL=" + apiUrl : "without NEXT_PUBLIC_API_URL (no engine)"}`);
  execFileSync("npm", ["run", "build"], { cwd: FRONTEND, stdio: "inherit", env });
}
function startUi() {
  return launch("npx", ["next", "start", "-p", "3004", "-H", "127.0.0.1"], FRONTEND);
}
async function stop(child) {
  try { process.kill(-child.pid, "SIGKILL"); } catch {}
  const t0 = Date.now();
  while (Date.now() - t0 < 30000) {
    try { await fetch(UI); } catch { return; }
    await sleep(250);
  }
  throw new Error(`${UI} still answering after next start was stopped`);
}
const metricCards = (body) => (body.match(/VaR 95% \(1d\)[\s\S]*?VaR 95% \(10d\)\s*\S+/i) || [""])[0].replace(/\s+/g, " ");

let browser;
try {
  build(API);
  launch("python3", ["-m", "uvicorn", "riskcanvas.api.app:app", "--host", "127.0.0.1", "--port", "8004"], ROOT);
  let ui = startUi();
  await up(`${API}/health`, 120000);
  await up(UI, 120000);

  browser = await chromium.launch();
  const errors = [];
  let analyzeBody = null;
  const engineRequests = [];
  const open = async (viewport, zoom, badge, absent, says) => {
    const page = await (await browser.newContext({ viewport, deviceScaleFactor: 2 })).newPage();
    page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
    page.on("request", (r) => {
      if (r.url().startsWith(API)) engineRequests.push(r.url());
      if (!analyzeBody && r.method() === "POST" && r.url().endsWith("/portfolio/analyze")) analyzeBody = r.postData();
    });
    await page.goto(UI, { waitUntil: "domcontentloaded", timeout: 120000 });
    await page.getByText(badge, { exact: true }).first().waitFor({ timeout: 120000 });
    await page.getByText("Global Financial Crisis").first().waitFor({ timeout: 60000 });
    if (zoom) await page.evaluate((z) => { document.body.style.zoom = String(z); }, zoom);
    await page.waitForTimeout(2500); // chart animations
    const body = await page.innerText("body");
    for (const must of [badge, says, "Synthetic inputs, not market data."]) {
      if (!body.includes(must)) throw new Error(`capture is missing "${must}"`);
    }
    if (await page.getByText(absent, { exact: true }).count()) throw new Error(`capture shows the "${absent}" badge, not "${badge}"`);
    return { page, body };
  };

  const full = await open({ width: 1440, height: 900 }, undefined, "Live engine", "Demo data", "computed by the Python engine");
  await full.page.screenshot({ path: SHOT, fullPage: true });
  await full.page.context().close();

  await stop(ui);
  build(null);
  ui = startUi();
  await up(UI, 120000);
  engineRequests.length = 0;
  const og = await open({ width: 1200, height: 630 }, 0.75, "Demo data", "Live engine", "This deployment has no engine connected");
  await og.page.screenshot({ path: OG });
  if (engineRequests.length) throw new Error(`the no-engine build called the engine: ${engineRequests[0]}`);
  if (errors.length) throw new Error(`${errors.length} console errors: ${errors.slice(0, 3).join(" | ")}`);

  const manifest = {
    captured: new Date().toISOString().slice(0, 10),
    screenshot: { path: "docs/screenshot.png", sha256: sha256(SHOT), viewport: "1440x900 @2x, full page", badge: "Live engine" },
    og: {
      path: "frontend/public/og.png",
      sha256: sha256(OG),
      viewport: "1200x630 @2x, zoom 0.75",
      badge: "Demo data",
      build: "production, no NEXT_PUBLIC_API_URL",
      metric_cards: metricCards(og.body),
    },
    metric_cards: metricCards(full.body),
  };
  writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + "\n");
  writeFileSync(REQUEST, JSON.stringify(JSON.parse(analyzeBody)) + "\n");
  console.log(JSON.stringify(manifest, null, 2));
} finally {
  if (browser) await browser.close();
  killAll();
}
