import { baseUrl, createMetadata } from "@/utils/metadata";
import { RootToggle } from "fumadocs-ui/components/layout/root-toggle";
import { RollButton } from "fumadocs-ui/components/roll-button";
import { DocsLayout } from "fumadocs-ui/layout";
import { RootProvider } from "fumadocs-ui/provider";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import { LibraryIcon, LucideIcon, PlugZapIcon } from "lucide-react";
import { layoutOptions } from "./layoutOptions";

import { GoogleAnalytics } from "@next/third-parties/google";
import "./global.css";

import { pageTree } from "@/app/source";
import type { Viewport } from "next";
import type { ReactNode } from "react";

export const metadata = createMetadata({
  title: {
    template: "DenserRetriever • %s",
    default: "DenserRetriever • Docs",
  },
  description: "Cutting-edge AI Retriever for RAG",
  metadataBase: baseUrl,
});

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0A0A0A" },
    { media: "(prefers-color-scheme: light)", color: "#fff" },
  ],
};

interface Mode {
  param: string;
  name: string;
  package: string;
  description: string;
  icon: LucideIcon;
}

const modes: Mode[] = [
  {
    param: "core",
    name: "Core",
    package: "DenserRetriever",
    description: "The core",
    icon: LibraryIcon,
  },
  {
    param: "api",
    name: "API",
    package: "fumadocs-mdx",
    description: "API Documentation",
    icon: PlugZapIcon,
  },
];

export default function RootDocsLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${GeistSans.className} ${GeistMono.className}`}
      suppressHydrationWarning
    >
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
      </head>
      <body className="overflow-x-hidden">
        <RootProvider>
          <DocsLayout
            {...layoutOptions}
            tree={pageTree}
            sidebar={{
              defaultOpenLevel: 0,
              banner: (
                <RootToggle
                  options={modes.map((mode) => {
                    return {
                      url: `/docs/${mode.param}`,
                      icon: (
                        <mode.icon
                          className="size-9 shrink-0 rounded-md bg-gradient-to-t from-background/80 p-1.5"
                          style={{
                            backgroundColor: `hsl(var(--${mode.param}-color)/.3)`,
                            color: `hsl(var(--${mode.param}-color))`,
                          }}
                        />
                      ),
                      title: mode.name,
                      description: mode.description,
                    };
                  })}
                />
              ),
            }}
          >
            {children}
            <RollButton />
          </DocsLayout>
        </RootProvider>
      </body>
      <GoogleAnalytics gaId="G-LBFKFW4G31" />
    </html>
  );
}
