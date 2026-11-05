/**
 * The dashboard's calls to the engine, and what it says when they fail, kept pure
 * so they can be unit-tested.
 *
 * The copy tells apart a frontend-only deploy that has no engine, an engine that
 * cannot be reached in time and one answering with an error. Slider steps are
 * debounced and a newer one aborts the request in flight, since three POSTs per
 * step can saturate a small hosted engine.
 */

/** How long the inputs must stay still before the engine is asked. */
export const ENGINE_DEBOUNCE_MS = 300;

export class EngineUnreachableError extends Error {}

export class EngineHttpError extends Error {
  /**
   * @param {string} url
   * @param {number} status
   */
  constructor(url, status) {
    super(`${url} -> HTTP ${status}`);
    this.status = status;
  }
}

/**
 * The JSON answer to one engine request. Rejects with EngineHttpError on a 4xx/5xx
 * answer and EngineUnreachableError on no answer in time; when the caller's `signal`
 * aborts, with the abort itself, since nobody is waiting any more.
 *
 * @param {string} url
 * @param {RequestInit} init
 * @param {{ timeoutMs: number, signal?: AbortSignal, fetchImpl?: typeof fetch }} options
 * @returns {Promise<any>}
 */
export async function fetchEngineJson(url, init, { timeoutMs, signal, fetchImpl = globalThis.fetch }) {
  const controller = new AbortController();
  const abort = () => controller.abort();
  const timer = setTimeout(abort, timeoutMs);
  if (signal?.aborted) abort();
  signal?.addEventListener("abort", abort, { once: true });
  try {
    const res = await fetchImpl(url, { ...init, signal: controller.signal, cache: "no-store" });
    if (!res.ok) throw new EngineHttpError(url, res.status);
    return await res.json();
  } catch (error) {
    if (signal?.aborted || error instanceof EngineHttpError || error instanceof SyntaxError) throw error;
    throw new EngineUnreachableError(`${url}: no answer (${error instanceof Error ? error.message : String(error)})`);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}

/**
 * @typedef {{ kind: "no_engine" } | { kind: "unreachable" } | { kind: "engine_error", status: number | null }} DemoReason
 */

/** @type {DemoReason} */
export const NO_ENGINE = { kind: "no_engine" };

/**
 * Why the engine produced no result: it gave no answer, or an answer that is an
 * error (an HTTP status, or null for a reply that could not be used).
 *
 * @param {unknown} error
 * @returns {DemoReason}
 */
export function demoReason(error) {
  if (error instanceof EngineUnreachableError) return { kind: "unreachable" };
  return { kind: "engine_error", status: error instanceof EngineHttpError ? error.status : null };
}

const ESTIMATES = "the numbers below are approximate in-browser estimates from the same synthetic series.";

/**
 * Every string the page shows for in-browser numbers, by why it has no engine result.
 *
 * @param {DemoReason} reason
 * @returns {{ badge: string, title: string, footer: string, math: string }}
 */
export function demoCopy(reason) {
  const badge = "Demo data";
  if (reason.kind === "no_engine") {
    return {
      badge,
      title: "This deployment has no engine connected: the numbers are in-browser estimates",
      footer: "Demo data: this deployment has no engine connected",
      math: `This deployment has no engine connected, so ${ESTIMATES}`,
    };
  }
  if (reason.kind === "unreachable") {
    return {
      badge,
      title: "The engine could not be reached in time: the numbers are in-browser estimates",
      footer: "Demo data: the engine could not be reached in time",
      math: `The engine could not be reached in time, so ${ESTIMATES}`,
    };
  }
  const answered = reason.status === null ? "The engine sent a reply that could not be read" : `The engine answered with an error (HTTP ${reason.status})`;
  return {
    badge,
    title: `${answered}: the numbers are in-browser estimates`,
    footer: `Demo data: ${answered[0].toLowerCase()}${answered.slice(1)}`,
    math: `${answered}, so ${ESTIMATES}`,
  };
}

/**
 * Ask once, `delayMs` after the inputs last changed. The returned function drops this
 * request: if it is still waiting it never starts, if it is in flight it is aborted,
 * and neither callback runs afterwards. Call it whenever the inputs change again, so
 * only the newest request ever reports.
 *
 * @template T
 * @param {(signal: AbortSignal) => Promise<T>} request
 * @param {{ delayMs: number, onResult: (result: T) => void, onError: (error: unknown) => void }} handlers
 * @returns {() => void}
 */
export function debouncedRequest(request, { delayMs, onResult, onError }) {
  const controller = new AbortController();
  const timer = setTimeout(() => {
    request(controller.signal).then(
      (result) => {
        if (!controller.signal.aborted) onResult(result);
      },
      (error) => {
        if (!controller.signal.aborted) onError(error);
      },
    );
  }, delayMs);
  return () => {
    clearTimeout(timer);
    controller.abort();
  };
}
