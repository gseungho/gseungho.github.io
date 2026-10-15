import type { MetadataRoute } from "next";
import { getAllPosts } from "@/lib/posts";

export const dynamic = "force-static";

const SITE = "https://gseungho.github.io";

// Submitted to Google Search Console so new posts get indexed.
export default function sitemap(): MetadataRoute.Sitemap {
  const posts = getAllPosts();
  const latest = posts[0]?.date;
  return [
    { url: `${SITE}/`, lastModified: latest },
    { url: `${SITE}/about`, lastModified: latest },
    ...["공부", "프로젝트", "논문리뷰"].map((c) => ({
      url: `${SITE}/category/${encodeURIComponent(c)}`,
      lastModified: latest,
    })),
    ...posts.map((p) => ({ url: `${SITE}/posts/${p.slug}`, lastModified: p.date })),
  ];
}
