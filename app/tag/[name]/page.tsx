import { getAllPosts, getAllTags } from "@/lib/posts";
import PostCard from "@/components/PostCard";
import type { Metadata } from "next";

export async function generateStaticParams() {
  return getAllTags().map((name) => ({ name }));
}

export async function generateMetadata({ params }: { params: Promise<{ name: string }> }): Promise<Metadata> {
  const { name } = await params;
  const tag = decodeURIComponent(name).normalize("NFC");
  return {
    title: `#${tag}`,
    description: `#${tag} 태그가 붙은 글 모음입니다.`,
    alternates: { canonical: `/tag/${encodeURIComponent(tag)}` },
  };
}

export default async function TagPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const tag = decodeURIComponent(name).normalize("NFC");
  const posts = getAllPosts().filter((p) => p.tags.map(t => t.normalize("NFC")).includes(tag));

  return (
    <div className="space-y-10">
      {/* Tag Header Band */}
      <section className="py-2">
        <div className="font-mono text-xs font-semibold uppercase tracking-[0.25em] text-(--accent) mb-3">
          folder / tags
        </div>
        <h1 className="text-3xl md:text-4xl font-semibold leading-tight text-(--text-strong) tracking-tight mb-3">
          #{tag}
        </h1>
        <div className="flex items-center gap-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-(--accent) animate-pulse" />
          <span className="font-mono text-sm text-(--text-muted)">{posts.length} index files available</span>
        </div>

        <div className="mt-8 border-b border-dashed border-(--border) w-full opacity-60" />
      </section>

      {/* Posts Section */}
      <section className="space-y-6">
        <div className="divide-y divide-(--border) border-y border-(--border)">
          {posts.map((post) => (
            <PostCard key={post.slug} post={post} />
          ))}
        </div>
      </section>
    </div>
  );
}
