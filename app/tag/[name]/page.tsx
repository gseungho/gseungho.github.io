import { getAllPosts, getAllTags } from "@/lib/posts";
import PostCard from "@/components/PostCard";

export async function generateStaticParams() {
  return getAllTags().map((name) => ({ name: encodeURIComponent(name) }));
}

export default async function TagPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const tag = decodeURIComponent(name);
  const posts = getAllPosts().filter((p) => p.tags.includes(tag));

  return (
    <div className="space-y-10">
      {/* Tag Header Band */}
      <section className="py-2">
        <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.25em] text-(--accent) mb-2">
          folder / tags
        </div>
        <h1 className="text-2xl md:text-3xl font-normal leading-tight text-(--text-strong) tracking-tight mb-2">
          #{tag}
        </h1>
        <div className="flex items-center gap-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-(--accent) animate-pulse" />
          <span className="font-mono text-xs text-(--text-muted)">{posts.length} index files available</span>
        </div>

        <div className="mt-8 border-b border-dashed border-(--border) w-full opacity-60" />
      </section>

      {/* Posts Section */}
      <section className="space-y-6">
        <div className="grid grid-cols-1 gap-4">
          {posts.map((post) => (
            <PostCard key={post.slug} post={post} />
          ))}
        </div>
      </section>
    </div>
  );
}
