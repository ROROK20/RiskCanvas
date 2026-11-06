const SCALE = [0, 25, 50, 75, 100];

// How wide a segment must be for its label, by the longer of its two lines: tickers
// run to 4 characters, shares to 5 ("10.7%"). globals.css holds the widths.
const fit = (asset: string, share: string) => (share.length >= 5 ? "l" : asset.length >= 4 || share.length >= 4 ? "m" : "s");

/**
 * The portfolio as the analysis sees it: one strip, every segment its analysed
 * share of 100%. All segments are one tone, since no holding is abnormal for being
 * held.
 */
export function AllocationStrip({
  shares,
  decimals,
}: {
  shares: Record<string, number> | null;
  decimals: 0 | 1;
}) {
  const entries = Object.entries(shares ?? {});
  const pct = (share: number) => `${(share * 100).toFixed(decimals)}%`;
  return (
    <div>
      <div
        role="img"
        aria-label={
          entries.length
            ? `Analysed allocation: ${entries.map(([a, s]) => `${a} ${pct(s)}`).join(", ")}`
            : "Nothing held"
        }
        className="flex h-12 gap-0.5 border border-rule"
      >
        {entries.map(([asset, share]) => (
          <div
            key={asset}
            title={`${asset} ${pct(share)}`}
            className="strip-segment min-w-0.5 bg-segment"
            style={{ flex: `${share} 1 0` }}
          >
            <div className="strip-label" data-fit={fit(asset, pct(share))}>
              <span className="font-mono font-bold">{asset}</span>
              <span>{pct(share)}</span>
            </div>
          </div>
        ))}
      </div>
      <div aria-hidden className="relative mt-1 h-5 font-mono text-xs leading-5 text-label">
        {SCALE.map((tick) => (
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
  );
}
