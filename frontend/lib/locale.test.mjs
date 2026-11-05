// node --test lib/locale.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import test from "node:test";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const FRONTEND = join(dirname(fileURLToPath(import.meta.url)), "..");

function sources(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sources(path);
    return /\.tsx?$/.test(name) ? [path] : [];
  });
}

test("no number is formatted in the viewer's locale", () => {
  // The page is prerendered, so `toLocaleString(undefined, ...)` printed
  // "$16,217" on the server and "$16.217" in a German browser: a hydration
  // mismatch. The driver's de-DE pass (e2e/smoke.mjs) proves it in a browser.
  const files = ["app", "lib"].flatMap((d) => sources(join(FRONTEND, d)));
  assert.ok(files.length > 3, String(files.length));
  for (const file of files) {
    const hits = readFileSync(file, "utf8").match(/toLocale\w*\(\s*(undefined\s*,?[^)]*)?\)/g);
    assert.equal(hits, null, `${file}: ${hits}`);
  }
});
