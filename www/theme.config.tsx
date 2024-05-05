import { Code2, LibraryBig, SquircleIcon } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/router";
import {
  Card,
  Cards,
  DocsThemeConfig,
  Steps,
  Tab,
  Tabs,
  useConfig,
} from "nextra-theme-docs";
import { Frame } from "./components/frame";
import { Icons } from "./components/icons";
import { MainContainer } from "./components/main-container";

const config: DocsThemeConfig = {
  logo: (
    <div className="flex items-center">
      <SquircleIcon className="inline-block mr-2" />
      <span className="font-bold">DenserRetriever</span>
    </div>
  ),
  main: MainContainer,
  search: {
    placeholder: "Search...",
  },
  navbar: {
    extraContent: (
      <>
        <a
          className="hidden p-1 sm:inline-block hover:opacity-80"
          target="_blank"
          href="https://denser.ai/discord"
          aria-label="Denser Discord"
          rel="nofollow noreferrer"
        >
          <Icons.discord className="w-6 h-6" />
        </a>

        <a
          className="hidden p-1 sm:inline-block hover:opacity-80"
          target="_blank"
          href="https://x.com/denser_ai"
          aria-label="Denser X formerly known as Twitter"
          rel="nofollow noreferrer"
        >
          <svg
            aria-label="X formerly known as Twitter"
            fill="currentColor"
            width="24"
            height="24"
            viewBox="0 0 24 22"
          >
            <path d="M16.99 0H20.298L13.071 8.26L21.573 19.5H14.916L9.702 12.683L3.736 19.5H0.426L8.156 10.665L0 0H6.826L11.539 6.231L16.99 0ZM15.829 17.52H17.662L5.83 1.876H3.863L15.829 17.52Z"></path>
          </svg>
        </a>

        {/* <GithubMenuBadge /> */}

        {/* <ToAppButton /> */}
      </>
    ),
  },
  sidebar: {
    defaultMenuCollapseLevel: 1,
    toggleButton: true,
    titleComponent: ({ type, title, route }) => {
      const { asPath } = useRouter();
      if (type === "separator" && title === "Switcher") {
        return (
          <div className="hidden -mx-2 md:block">
            {[
              { title: "Docs", path: "/docs", Icon: LibraryBig },
              { title: "Examples", path: "/examples", Icon: Code2 },
            ].map((item) =>
              asPath.startsWith(item.path) ? (
                <div
                  key={item.path}
                  className="flex flex-row items-center gap-3 mb-3 group nx-text-primary-800 dark:nx-text-primary-600"
                >
                  <item.Icon className="p-1 border rounded dark:border-muted-foreground w-7 h-7 nx-bg-primary-100 dark:nx-bg-primary-400/10" />
                  {item.title}
                </div>
              ) : (
                <Link
                  href={item.path}
                  key={item.path}
                  className="flex flex-row items-center gap-3 mb-3 border-muted-foreground group hover:nx-text-primary-600/100"
                >
                  <item.Icon className="p-1 border rounded dark:border-muted-foreground w-7 h-7 group-hover:nx-bg-primary-400/30" />
                  {item.title}
                </Link>
              )
            )}
          </div>
        );
      }
      return title;
    },
  },
  editLink: {
    text: "Edit this page on GitHub",
  },
  toc: {
    backToTop: true,
  },
  docsRepositoryBase: "https://github.com/denser-org/retriever-docs/tree/main",
  footer: {
    text: (
      <span>
        2024 ©{" "}
        <a href="https://denser.ai" target="_blank" rel="noopener noreferrer">
          Denser.ai
        </a>
        . All rights reserved.
      </span>
    ),
  },
  head: () => {
    const { asPath, defaultLocale, locale } = useRouter();
    const { frontMatter, title: pageTitle } = useConfig();
    const url =
      "https://denser.ai" +
      (defaultLocale === locale ? asPath : `/${locale}${asPath}`);

    const description = frontMatter.description ?? "";

    // const title = frontMatter.title ?? pageTitle;

    // const section = asPath.startsWith("/docs")
    //   ? "Docs"
    //   : asPath.startsWith("/changelog/")
    //   ? "Changelog"
    //   : "";

    return (
      <>
        <meta name="theme-color" content="#000" />
        <meta property="og:url" content={url} />
        <meta httpEquiv="Content-Language" content="en" />

        <meta name="description" content={description} />
        <meta property="og:description" content={description} />

        {/* {video && <meta property="og:video" content={video} />} */}

        {/* <meta property="og:image" content={image} /> */}
        {/* <meta property="twitter:image" content={image} /> */}

        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:site:domain" content="langfuse.com" />
        <meta name="twitter:url" content="https://denser.ai" />

        <link
          rel="apple-touch-icon"
          sizes="180x180"
          href="/apple-touch-icon.png"
        />
        <link
          rel="icon"
          type="image/png"
          sizes="32x32"
          href="/favicon-32x32.png"
        />
        <link
          rel="icon"
          type="image/png"
          sizes="16x16"
          href="/favicon-16x16.png"
        />
      </>
    );
  },
  components: {
    Frame,
    Tabs,
    Tab,
    Steps,
    Card,
    Cards,
  },
};

export default config;
