import "fumadocs-ui/twoslash.css"
import { DocsLayout } from "fumadocs-ui/layout"
import { RollButton } from "fumadocs-ui/components/roll-button"
import type { ReactNode } from "react"
import { layoutOptions } from "./layoutOptions"

export default function RootDocsLayout({ children }: { children: ReactNode }) {
  return (
    <DocsLayout {...layoutOptions}>
      {children}
      <RollButton />
    </DocsLayout>
  )
}
