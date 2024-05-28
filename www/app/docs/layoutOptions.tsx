import { pageTree } from "../source"
import { type DocsLayoutProps } from "fumadocs-ui/layout"
import { Logo } from "@/components/ui/icons"

export const layoutOptions: Omit<DocsLayoutProps, "children"> = {
  tree: pageTree,
  sidebar: {
    collapsible: false,
  },
  nav: {
    transparentMode: "top",
    title: (
      <>
        <Logo
          className="w-8 h-8 text-black dark:text-white"
          fill="currentColor"
        />
        <span className="font-semibold max-md:hidden">DenserRetriever</span>
      </>
    ),
    githubUrl: "https://github.com/denser-org/denser-retriever",
  },
  links: [
    {
      text: "Documentation",
      url: "/docs",
    },
    // {
    //   text: "About",
    //   url: "https://denser.ai/about",
    //   external: true,
    // },
  ],
}
