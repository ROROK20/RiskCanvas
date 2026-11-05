import type { Metadata } from "next";
import "./globals.css";
import { resolveSiteUrl } from "@/lib/siteUrl.mjs";

// Absolute og / twitter image URLs need the site's public origin. Without
// NEXT_PUBLIC_SITE_URL a production build ships no preview image rather
// than one pointing at localhost (lib/siteUrl.mjs).
const SITE_URL = resolveSiteUrl(process.env.NEXT_PUBLIC_SITE_URL, process.env.NODE_ENV);
// The preview shows the state a no-backend deploy serves: the "Demo data" badge
// and its in-browser figures (e2e/capture.mjs builds without an engine for it).
const OG_IMAGES = SITE_URL
  ? [{ url: "/og.png", width: 1200, height: 630, alt: "RiskCanvas portfolio risk dashboard showing its in-browser demo data" }]
  : [];

// True in both modes, engine or no engine: the static metadata cannot tell which.
const DESCRIPTION =
  "Interactive portfolio risk demo on synthetic inputs: VaR, CVaR, Monte Carlo and crisis stress tests. Open-source Python engine.";

export const metadata: Metadata = {
  title: {
    default: "RiskCanvas — Portfolio Risk Analytics & Stress Testing",
    template: "%s · RiskCanvas",
  },
  description: DESCRIPTION,
  keywords: [
    "VaR",
    "CVaR",
    "Monte Carlo",
    "stress testing",
    "portfolio risk",
    "quantitative finance",
  ],
  authors: [{ name: "Rohan Khiani", url: "https://github.com/ROROK20" }],
  ...(SITE_URL ? { metadataBase: new URL(SITE_URL) } : {}),
  openGraph: {
    title: "RiskCanvas — Portfolio Risk Analytics & Stress Testing",
    description: DESCRIPTION,
    type: "website",
    siteName: "RiskCanvas",
    images: OG_IMAGES,
  },
  twitter: {
    card: "summary_large_image",
    title: "RiskCanvas — Portfolio Risk Analytics & Stress Testing",
    description: DESCRIPTION,
    images: OG_IMAGES.map((i) => i.url),
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-mono antialiased">{children}</body>
    </html>
  );
}
