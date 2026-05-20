import { getAllPosts, getAllCategories } from "@/lib/posts";
import PostCard from "@/components/PostCard";

export async function generateStaticParams() {
  return getAllCategories().map((name) => ({ name: encodeURIComponent(name) }));
}

export default async function CategoryPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const category = decodeURIComponent(name);
  const posts = getAllPosts().filter((p) => p.category === category);

  return (
    <div className="space-y-10">
      {/* Category Header Band */}
      <section className="py-2">
        <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.25em] text-[#00d992] mb-2">
          folder / categories
        </div>
        <h1 className="text-2xl md:text-3xl font-normal leading-tight text-[#ffffff] tracking-tight mb-2">
          {category}
        </h1>
        <div className="flex items-center gap-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#00d992] animate-pulse" />
          <span className="font-mono text-xs text-[#8b949e]">{posts.length} index files available</span>
        </div>
        
        {/* Voltagent Signature Dashed Divider */}
        <div className="mt-8 border-b border-dashed border-[#4f5d75]/40 w-full" />
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
