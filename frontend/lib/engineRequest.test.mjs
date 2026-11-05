// node --test lib/engineRequest.test.mjs  (npm test; Node 20+, no dependencies)
import assert from "node:assert/strict";
import test from "node:test";
import { setTimeout as sleep } from "node:timers/promises";

import {
  debouncedRequest,
  demoCopy,
  demoReason,
  EngineHttpError,
  EngineUnreachableError,
  fetchEngineJson,
  NO_ENGINE,
} from "./engineRequest.mjs";

const fetchAnswering = (status, body = "{}") => async () => new Response(body, { status });
const fetchHanging = (seen) => (_url, init) =>
  new Promise((_resolve, reject) => {
    seen.signal = init.signal;
    const abort = () => reject(new DOMException("Aborted", "AbortError"));
    if (init.signal.aborted) abort();
    init.signal.addEventListener("abort", abort);
  });

test("a 4xx or 5xx is an engine error naming its status, not an unreachable engine", async () => {
  for (const status of [404, 422, 500, 503]) {
    const error = await fetchEngineJson("/x", {}, { timeoutMs: 1000, fetchImpl: fetchAnswering(status) }).catch((e) => e);
    assert.ok(error instanceof EngineHttpError);
    assert.deepEqual(demoReason(error), { kind: "engine_error", status });
  }
});

test("a failed connection is an unreachable engine", async () => {
  const fetchImpl = async () => {
    throw new TypeError("Failed to fetch");
  };
  const error = await fetchEngineJson("/x", {}, { timeoutMs: 1000, fetchImpl }).catch((e) => e);
  assert.ok(error instanceof EngineUnreachableError);
  assert.deepEqual(demoReason(error), { kind: "unreachable" });
});

test("no answer inside the timeout is an unreachable engine, and the request is aborted", async () => {
  const seen = {};
  const error = await fetchEngineJson("/x", {}, { timeoutMs: 5, fetchImpl: fetchHanging(seen) }).catch((e) => e);
  assert.ok(error instanceof EngineUnreachableError);
  assert.equal(seen.signal.aborted, true);
});

test("a reply that is not JSON is an engine error with no status", async () => {
  const error = await fetchEngineJson("/x", {}, { timeoutMs: 1000, fetchImpl: fetchAnswering(200, "<html>") }).catch((e) => e);
  assert.deepEqual(demoReason(error), { kind: "engine_error", status: null });
});

test("the caller's abort stops the request and is not reported as a failed engine", async () => {
  const seen = {};
  const caller = new AbortController();
  const pending = fetchEngineJson("/x", {}, { timeoutMs: 1000, signal: caller.signal, fetchImpl: fetchHanging(seen) }).catch((e) => e);
  caller.abort();
  const error = await pending;
  assert.equal(seen.signal.aborted, true);
  assert.equal(error.name, "AbortError");
  assert.equal(error instanceof EngineUnreachableError, false);
});

test("a caller that has already aborted never leaves the request running", async () => {
  const seen = {};
  const caller = new AbortController();
  caller.abort();
  await fetchEngineJson("/x", {}, { timeoutMs: 1000, signal: caller.signal, fetchImpl: fetchHanging(seen) }).catch((e) => e);
  assert.equal(seen.signal.aborted, true);
});

test("a good answer comes back parsed", async () => {
  const body = await fetchEngineJson("/x", {}, { timeoutMs: 1000, fetchImpl: fetchAnswering(200, '{"a":1}') });
  assert.deepEqual(body, { a: 1 });
});

test("a deployment with no engine does not claim one is down or tell visitors to start it", () => {
  const copy = demoCopy(NO_ENGINE);
  for (const text of Object.values(copy)) assert.doesNotMatch(text, /unreachable|start the api|could not be reached/i);
  assert.match(copy.math, /no engine connected/);
  assert.match(copy.footer, /no engine connected/);
});

test("the three reasons read differently, and all call the numbers in-browser estimates", () => {
  const copies = [NO_ENGINE, { kind: "unreachable" }, { kind: "engine_error", status: 500 }].map(demoCopy);
  assert.equal(new Set(copies.map((c) => c.math)).size, 3);
  assert.equal(new Set(copies.map((c) => c.footer)).size, 3);
  for (const copy of copies) {
    assert.equal(copy.badge, "Demo data");
    assert.match(copy.math, /approximate in-browser estimates/);
  }
  assert.match(copies[1].math, /could not be reached/);
  assert.match(copies[2].math, /answered with an error \(HTTP 500\)/);
  assert.match(copies[2].footer, /the engine answered with an error \(HTTP 500\)/);
  assert.match(demoCopy({ kind: "engine_error", status: null }).math, /could not be read/);
});

/** A request that records its signal and stays open until told. */
function recording() {
  const calls = [];
  const request = (signal) =>
    new Promise((resolve, reject) => {
      calls.push({ signal, resolve, reject });
    });
  return { calls, request };
}

test("a burst of changes sends one request, after the delay, for the newest inputs", async () => {
  const asked = [];
  let cancel = () => {};
  for (let step = 0; step < 10; step++) {
    cancel();
    cancel = debouncedRequest(async () => asked.push(step), { delayMs: 15, onResult() {}, onError() {} });
  }
  assert.deepEqual(asked, [], "nothing is sent before the delay");
  await sleep(50);
  assert.deepEqual(asked, [9]);
});

test("dropping a request that is still waiting means it never starts", async () => {
  const { calls, request } = recording();
  debouncedRequest(request, { delayMs: 10, onResult() {}, onError() {} })();
  await sleep(30);
  assert.equal(calls.length, 0);
});

test("dropping a request in flight aborts it, and its answer is ignored", async () => {
  const { calls, request } = recording();
  const reported = [];
  const handlers = { delayMs: 1, onResult: (r) => reported.push(r), onError: (e) => reported.push(e) };
  const cancel = debouncedRequest(request, handlers);
  await sleep(15);
  assert.equal(calls[0].signal.aborted, false);
  cancel();
  assert.equal(calls[0].signal.aborted, true);
  calls[0].resolve("stale");
  await sleep(1);
  assert.deepEqual(reported, []);
});

test("a dropped request's failure is ignored too", async () => {
  const { calls, request } = recording();
  const reported = [];
  const cancel = debouncedRequest(request, { delayMs: 1, onResult: (r) => reported.push(r), onError: (e) => reported.push(e) });
  await sleep(15);
  cancel();
  calls[0].reject(new DOMException("Aborted", "AbortError"));
  await sleep(1);
  assert.deepEqual(reported, []);
});

test("the newest request reports its result or its failure", async () => {
  const { calls, request } = recording();
  const results = [];
  const errors = [];
  const handlers = { delayMs: 1, onResult: (r) => results.push(r), onError: (e) => errors.push(e) };
  debouncedRequest(request, handlers);
  await sleep(15);
  calls[0].resolve("fresh");
  debouncedRequest(request, handlers);
  await sleep(15);
  calls[1].reject(new Error("engine failed"));
  await sleep(1);
  assert.deepEqual(results, ["fresh"]);
  assert.equal(errors[0].message, "engine failed");
});
