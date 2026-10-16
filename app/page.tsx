import { getAllPosts } from "@/lib/posts";
import PostCard from "@/components/PostCard";
import type { Metadata } from "next";

export const metadata: Metadata = { alternates: { canonical: "/" } };

// tells Google the site's name (otherwise results show "GitHub" for *.github.io)
const WEBSITE_JSON_LD = {
  "@context": "https://schema.org",
  "@type": "WebSite",
  name: "Tensors & Notebooks",
  url: "https://gseungho.github.io/",
};

export default function Home() {
  const posts = getAllPosts();
  return (
    <div className="space-y-8">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(WEBSITE_JSON_LD) }}
      />
      {/* Voltagent Hero Band */}
      <section className="pt-6">
        <div className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#00d992] mb-3">
          npx tensors-and-notebooks --log
        </div>
        <h1 className="text-[2.85rem] md:text-[3.75rem] font-normal leading-none tracking-[-0.035em] text-(--text-strong) mb-6 font-sans">
          AI &amp; Mathematics.
        </h1>
        <p className="text-base md:text-lg text-(--text) max-w-xl leading-relaxed">
          AI 논문 리뷰와 공부 중인 내용을 정리한 노트, 직접 만든 프로젝트의 기록을 담는 블로그입니다.
        </p>
      </section>

      {/* Posts Section */}
      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-[0.25em] text-[#8b949e]">
            Recent Publications
          </h2>
          <span className="font-mono text-xs text-[#8b949e]">{posts.length} articles</span>
        </div>
        <div className="divide-y divide-(--border) border-y border-(--border)">
          {posts.map((post) => (
            <PostCard key={post.slug} post={post} />
          ))}
        </div>
      </section>
    </div>
  );
}
