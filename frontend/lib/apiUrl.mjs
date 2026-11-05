/**
 * Which engine URL the dashboard calls, or null for none (demo mode).
 *
 * `NEXT_PUBLIC_API_URL` is inlined by Next at BUILD time. When it is set,
 * that URL is used. When it is unset, only `next dev` (NODE_ENV
 * "development") falls back to http://localhost:8004, the local engine; a
 * production build gets null, makes no engine call at all and shows the
 * in-browser demo. Otherwise a public deploy without the variable would call
 * each VISITOR's own localhost:8004, and a visitor running the stack locally
 * would see "Live engine" on the public site, computed by their own machine.
 *
 * @param {string | undefined} configured NEXT_PUBLIC_API_URL
 * @param {string | undefined} nodeEnv NODE_ENV
 * @returns {string | null}
 */
export function resolveApiUrl(configured, nodeEnv) {
  const url = (configured ?? "").trim();
  if (url) return url.replace(/\/$/, "");
  return nodeEnv === "development" ? "http://localhost:8004" : null;
}
