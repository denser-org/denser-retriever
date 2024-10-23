const { createPreset } = require("fumadocs-ui/tailwind-plugin");
import { StaticShadows } from "open-props/src/shadows";

/** @type {import('tailwindcss').Config} */
module.exports = {
	darkMode: "class",
	content: [
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./content/**/*.{md,mdx}",
		"./mdx-components.{ts,tsx}",
		"./node_modules/fumadocs-ui/dist/**/*.js",
	],
	presets: [createPreset()],
	plugins: [],
	theme: {
		extend: {
			fontFamily: {
				sans: ["var(--font-geist-sans)", "Inter", "system-ui", "sans-serif"],
				mono: ["var(--font-geist-mono)", "Ubuntu Mono", "monospace"],
				display: ["var(--font-geist-display)", "Inter", "system-ui"],
				heading: ["var(--font-google-heading)", "Garamond", "system-ui"],
			},
			colors: {
				neutral: {
					50: "hsl(0 0% 76% / <alpha-value>)",
					100: "hsl(0 0% 71% / <alpha-value>)",
					200: "hsl(0 0% 62% / <alpha-value>)",
					300: "hsl(0 0% 49% / <alpha-value>)",
					400: "hsl(0 0% 33% / <alpha-value>)",
					500: "hsl(0 0% 23% / <alpha-value>)",
					600: "hsl(0 0% 16% / <alpha-value>)",
					700: "hsl(0 0% 11% / <alpha-value>)",
					800: "hsl(0 0% 7% / <alpha-value>)",
					900: "hsl(0 0% 4% / <alpha-value>)",
					950: "hsl(0 0% 3% / <alpha-value>)",
				},
			},
		},
		container: {
			center: true,
			padding: "2rem",
			screens: {
				"2xl": "1400px",
			},
		},
		boxShadow: {
			xs: StaticShadows["--shadow-1"],
			sm: StaticShadows["--shadow-2"],
			md: StaticShadows["--shadow-3"],
			lg: StaticShadows["--shadow-4"],
			xl: StaticShadows["--shadow-5"],
			"2xl": StaticShadows["--shadow-6"],
		},
		animation: {
			ripple: "ripple 3400ms ease infinite",
		},
		keyframes: {
			ripple: {
				"0%, 100%": {
					transform: "translate(-50%, -50%) scale(1)",
				},
				"50%": {
					transform: "translate(-50%, -50%) scale(0.9)",
				},
			},
		},
	},
};
