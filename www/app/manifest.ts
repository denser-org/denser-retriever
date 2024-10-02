import { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "DenserRetriever",
    short_name: "DenserRetriever",
    description: "DenserRetriever",
    start_url: "/",
    theme_color: "#1e293b",
    background_color: "#1e293b",
    display: "standalone",
    icons: [],
  };
}
