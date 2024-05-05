/// <reference path="./.sst/platform/config.d.ts" />

export default $config({
  app(input) {
    return {
      name: "retriever-docs",
      removal: input?.stage === "production" ? "retain" : "remove",
      home: "aws",
      providers: {
        aws: {
          region: "us-west-2",
          profile: input?.stage === "production" ? undefined : "denser",
        },
      },
    };
  },
  async run() {
    new sst.aws.Nextjs("RetrieverDocs");
  },
});
