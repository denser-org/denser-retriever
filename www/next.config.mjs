import createMDX from "fumadocs-mdx/config"
import {
  remarkInstall,
  rehypeCodeDefaultOptions,
} from "fumadocs-core/mdx-plugins"
import { transformerTwoslash } from "fumadocs-twoslash"

const withMDX = createMDX({
  mdxOptions: {
    rehypeCodeOptions: {
      transformers: [
        ...rehypeCodeDefaultOptions.transformers,
        transformerTwoslash(),
        {
          name: "fumadocs:remove-escape",
          code(element) {
            element.children.forEach((line) => {
              if (line.type !== "element") return

              line.children.forEach((child) => {
                if (child.type !== "element") return
                const textNode = child.children[0]
                if (!textNode || textNode.type !== "text") return

                textNode.value = textNode.value.replace(/\[\\!code/g, "[!code")
              })
            })

            return element
          },
        },
      ],
    },
    lastModifiedTime: "git",
    remarkPlugins: [remarkInstall],
  },
})

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  eslint: {
    ignoreDuringBuilds: true,
  }
}

export default withMDX(config)
