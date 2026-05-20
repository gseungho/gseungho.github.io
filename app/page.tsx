import { getAllPosts } from "@/lib/posts";
import PostCard from "@/components/PostCard";

export default function Home() {
  const posts = getAllPosts();
  return (
    <div className="space-y-12">
      {/* Voltagent Hero Band */}
      <section className="py-6">
        <div className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-[#00d992] mb-3">
          npx gseungho --log
        </div>
        <h1 className="text-[2.85rem] md:text-[3.75rem] font-normal leading-none tracking-[-0.035em] text-[#ffffff] mb-6 font-sans">
          AI & Entrepreneurship.
        </h1>
        <p className="text-base md:text-lg text-[#bdbdbd] max-w-xl leading-relaxed">
          창업 일지, AI 논문 리뷰, 그리고 고도화된 소프트웨어 프로젝트 개발 기록을 담는 엔지니어링 저널입니다.
        </p>
        
        {/* Voltagent Signature Dashed Divider */}
        <div className="mt-12 border-b border-dashed border-[#4f5d75]/40 w-full" />
      </section>

      {/* Posts Section */}
      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold uppercase tracking-[0.25em] text-[#8b949e]">
            Recent Publications
          </h2>
          <span className="font-mono text-xs text-[#8b949e]">{posts.length} articles</span>
        </div>
        <div className="grid grid-cols-1 gap-4">
          {posts.map((post) => (
            <PostCard key={post.slug} post={post} />
          ))}
        </div>
      </section>
    </div>
  );
}
