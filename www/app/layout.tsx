import { baseUrl, createMetadata } from "@/utils/metadata"
import { RootProvider } from "fumadocs-ui/provider"
import { GeistMono } from "geist/font/mono"
import { GeistSans } from "geist/font/sans"
import { Londrina_Solid } from "next/font/google"

import Script from "next/script"
import { Footer } from "../components/footer"
import "./global.css"

import type { Viewport } from "next"
import type { ReactNode } from "react"

const uni = Londrina_Solid({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-google-heading",
  display: "swap",
})

export const metadata = createMetadata({
  title: {
    template: "DenserRetriever • %s",
    default: "DenserRetriever • Docs",
  },
  description: "Bookmarks, Read-it-later, and RSS-Feeds",
  metadataBase: baseUrl,
})

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0A0A0A" },
    { media: "(prefers-color-scheme: light)", color: "#fff" },
  ],
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${GeistSans.className} ${uni.variable} ${GeistMono.className}`}
      suppressHydrationWarning
    >
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
      </head>
      {process.env.NODE_ENV === "production" && (
        <Script data-api="/a/e" data-domain="docs.denser.ai" src="/p.js" />
      )}
      <body className="overflow-x-hidden">
        <RootProvider>
          {children}
          <Footer />
        </RootProvider>
      </body>
    </html>
  )
}
