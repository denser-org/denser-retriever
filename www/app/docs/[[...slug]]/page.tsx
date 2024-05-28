import { notFound } from "next/navigation"
import { ArrowSquareOut } from "@phosphor-icons/react/dist/ssr"
import { getPage, getPages } from "@/app/source"
import { DocsPage, DocsBody } from "fumadocs-ui/page"
import type { Metadata } from "next"

interface Param {
  slug: string[]
}

export const dynamicParams = false

export default async function Page({ params }: { params: Param }) {
  const page = getPage(params.slug)

  if (page == null) {
    notFound()
  }

  const path = `/content/docs/${page.file.path}`

  const MDX = page.data.exports.default

  return (
    <DocsPage
      toc={page.data.exports.toc}
      lastUpdate={page.data.exports.lastModified}
      tableOfContent={{
        enabled: page.file.path !== "api-reference.mdx",
        footer: (
          <a
            href={`https://github.com/denser-org/denser-retriever/blob/main/www/${path}`}
            target="_blank"
            rel="noreferrer noopener"
            className="flex items-baseline text-xs text-muted-foreground hover:text-foreground"
          >
            Edit on Github <ArrowSquareOut className="ml-1 size-3" />
          </a>
        ),
      }}
    >
      <DocsBody>
        <h1>{page.data.title}</h1>
        <MDX />
      </DocsBody>
    </DocsPage>
  )
}

export async function generateStaticParams() {
  return getPages().map((page) => ({
    slug: page.slugs,
  }))
}

export function generateMetadata({ params }: { params: { slug?: string[] } }) {
  const page = getPage(params.slug)

  if (page == null) notFound()

  const description =
    page.data.description ?? "The library for building documentation sites"

  const imageParams = new URLSearchParams()
  imageParams.set("title", page.data.title)
  imageParams.set("description", description)

  const image = {
    alt: "Banner",
    url: `/api/og/${params.slug?.[0]}?${imageParams.toString()}`,
    width: 1200,
    height: 630,
  }

  return {
    title: page.data.title,
    description: page.data.description,
    openGraph: {
      url: `/docs/${page.slugs.join("/")}`,
      images: image,
    },
    twitter: {
      images: image,
    },
  } satisfies Metadata
}
