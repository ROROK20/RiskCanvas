import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        raised: "var(--raised)",
        rule: "var(--rule)",
        "rule-strong": "var(--rule-strong)",
        label: "var(--label)",
        readout: "var(--readout)",
        segment: "var(--segment)",
        amber: "var(--amber)",
        "amber-dim": "var(--amber-dim)",
        fault: "var(--fault)",
      },
      borderColor: {
        DEFAULT: "var(--rule)",
      },
      fontFamily: {
        sans: ["var(--font-b612)", "Arial", "sans-serif"],
        mono: ["var(--font-b612-mono)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
