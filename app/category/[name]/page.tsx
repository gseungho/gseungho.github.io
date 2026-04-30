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
    <div>
      <div className="mb-8">
        <p style={{ color: "var(--text-muted)", fontSize: "0.85rem" }} className="mb-1">카테고리</p>
        <h1 className="text-2xl font-bold">{category}</h1>
        <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }} className="mt-1">
          {posts.length}개의 글
        </p>
      </div>
      <div className="flex flex-col gap-4">
        {posts.map((post) => (
          <PostCard key={post.slug} post={post} />
        ))}
      </div>
    </div>
  );
}
