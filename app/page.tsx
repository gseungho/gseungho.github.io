import { getAllPosts } from "@/lib/posts";
import PostCard from "@/components/PostCard";

export default function Home() {
  const posts = getAllPosts();
  return (
    <div>
      <div className="mb-10">
        <h1 className="text-3xl font-bold mb-2">gseungho&apos;s log</h1>
        <p style={{ color: "var(--text-muted)" }}>창업 일지, AI 논문 리뷰, 프로젝트 기록을 남기는 공간</p>
      </div>
      <div className="flex flex-col gap-4">
        {posts.map((post) => (
          <PostCard key={post.slug} post={post} />
        ))}
      </div>
    </div>
  );
}
