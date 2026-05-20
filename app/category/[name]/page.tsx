import { getAllPosts, getAllCategories } from "@/lib/posts";
import PostCard from "@/components/PostCard";

export async function generateStaticParams() {
  return getAllCategories().map((name) => ({ name: encodeURIComponent(name) }));
}

export default async function CategoryPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const category = decodeURIComponent(name);
  const posts = getAllPosts().filter((p) => p.category === category);

  // Group by subcategory; posts without subcategory go under ""
  const groups = new Map<string, typeof posts>();
  for (const post of posts) {
    const key = post.subcategory ?? "";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(post);
  }

  // Sort: no-subcategory first, then alphabetically
  const sortedKeys = [...groups.keys()].sort((a, b) => {
    if (a === "") return -1;
    if (b === "") return 1;
    return a.localeCompare(b);
  });

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

        <div className="mt-8 border-b border-dashed border-[#4f5d75]/40 w-full" />
      </section>

      {/* Posts Section */}
      <section className="space-y-10">
        {sortedKeys.map((key) => (
          <div key={key || "__root__"}>
            {key && (
              <div className="mb-4">
                <span className="font-mono text-xs text-[#8b949e]">~/</span>
                <span className="font-mono text-sm text-[#00d992]">{key.replace(/\//g, " / ")}</span>
                <div className="mt-2 border-b border-[#3d3a39] w-full" />
              </div>
            )}
            <div className="grid grid-cols-1 gap-4">
              {groups.get(key)!.map((post) => (
                <PostCard key={post.slug} post={post} />
              ))}
            </div>
          </div>
        ))}
      </section>
    </div>
  );
}
