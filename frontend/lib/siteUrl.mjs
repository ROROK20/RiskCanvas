/**
 * The site's public origin for absolute social-preview URLs, or null.
 *
 * `NEXT_PUBLIC_SITE_URL` (inlined at build) when set; otherwise
 * http://localhost:3004 under `next dev` only. A production build without it
 * gets null: layout.tsx then sets no metadataBase and no og / twitter image
 * (Next would otherwise resolve "/og.png" against localhost and advertise
 * http://localhost:3004/og.png to every crawler).
 *
 * @param {string | undefined} configured NEXT_PUBLIC_SITE_URL
 * @param {string | undefined} nodeEnv NODE_ENV
 * @returns {string | null}
 */
export function resolveSiteUrl(configured, nodeEnv) {
  const url = (configured ?? "").trim();
  if (url) return url.replace(/\/$/, "");
  return nodeEnv === "development" ? "http://localhost:3004" : null;
}
