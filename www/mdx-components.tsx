import type { MDXComponents } from "mdx/types"
import { Screenshot } from "@/components/screenshot"
import defaultComponents from "fumadocs-ui/mdx"
import { Step, Steps } from "fumadocs-ui/components/steps"
import { Tab, Tabs } from "fumadocs-ui/components/tabs"
import { ImageZoom } from "fumadocs-ui/components/image-zoom"
import { File, Files } from "fumadocs-ui/components/files"
import { Callout } from "fumadocs-ui/components/callout"
import { AutoTypeTable } from "fumadocs-typescript/ui"
import { Accordion, Accordions } from "fumadocs-ui/components/accordion"
import { Popup, PopupTrigger, PopupContent } from "fumadocs-ui/twoslash/popup"

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    Popup,
    PopupTrigger,
    PopupContent,
    Accordions,
    Accordion,
    AutoTypeTable,
    Callout,
    Files,
    File,
    ImageZoom,
    Tab,
    Tabs,
    Step,
    Steps,
    Screenshot,
    ...defaultComponents,
    ...components,
  }
}
