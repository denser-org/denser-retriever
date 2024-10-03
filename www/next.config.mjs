import createMDX from "fumadocs-mdx/config";

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  redirects: () => {
    return [
      {
        source: "/",
        destination: "/docs/core",
        permanent: true,
      },
    ];
  },
};

export default withMDX(config);
