import { getAllPosts } from "@/lib/posts";

export const dynamic = "force-static";

const SITE = "https://gseungho.github.io";

function escape(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

// RSS 2.0 feed, submitted to Naver Search Advisor so new posts get collected quickly.
export function GET() {
  const posts = getAllPosts();
  const items = posts
    .map((p) => {
      const url = `${SITE}/posts/${p.slug}`;
      const pubDate = new Date(`${p.date}T09:00:00+09:00`).toUTCString();
      return [
        "    <item>",
        `      <title>${escape(p.title)}</title>`,
        `      <link>${url}</link>`,
        `      <guid isPermaLink="true">${url}</guid>`,
        `      <description>${escape(p.description)}</description>`,
        `      <category>${escape(p.category)}</category>`,
        `      <pubDate>${pubDate}</pubDate>`,
        "    </item>",
      ].join("\n");
    })
    .join("\n");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Tensors &amp; Notebooks</title>
    <link>${SITE}/</link>
    <description>AI 논문 리뷰와 공부 중인 내용을 정리한 노트, 직접 만든 프로젝트의 기록을 담는 블로그입니다.</description>
    <language>ko</language>
    <atom:link href="${SITE}/rss.xml" rel="self" type="application/rss+xml" />
${items}
  </channel>
</rss>
`;
  return new Response(xml, { headers: { "Content-Type": "application/rss+xml; charset=utf-8" } });
}
