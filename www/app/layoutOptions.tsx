import { Logo } from "@/components/ui/icons";
import type { BaseLayoutProps } from "fumadocs-ui/layout";
import { GlobeIcon } from "lucide-react";

export const layoutOptions: BaseLayoutProps = {
  nav: {
    url: "/docs/core",
    title: (
      <>
        <Logo />
        <span className="text-foreground">DenserRetriever</span>
      </>
    ),
  },
  links: [
    {
      text: "Website",
      url: "https://retriever.denser.ai",
      active: "nested-url",
      icon: <GlobeIcon />,
    },
    // {
    //   text: "About",
    //   url: "https://denser.ai/about",
    //   external: true,
    // },
  ],
};
