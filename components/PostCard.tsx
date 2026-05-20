import Link from "next/link";
import type { PostMeta } from "@/lib/posts";

export default function PostCard({ post }: { post: PostMeta }) {
  return (
    <Link href={`/posts/${post.slug}`} className="block group">
      <article className="bg-[#101010] border border-[#3d3a39] rounded-lg p-5 transition-all duration-300 cursor-pointer group-hover:border-[#00d992] group-hover:bg-[#161616]">
        <div className="flex items-center justify-between gap-4 mb-3">
          {/* Voltagent button-pill-tag style */}
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="inline-block px-2.5 py-0.5 text-xs font-semibold text-[#f2f2f2] bg-[#101010] border border-[#3d3a39] rounded-full transition-colors duration-300 group-hover:border-[#00d992]/60 group-hover:text-[#00d992] shrink-0">
              {post.category}
            </span>
            {post.subcategory && (
              <span className="font-mono text-[11px] text-[#8b949e] truncate">
                {post.subcategory.replace(/\//g, " / ")}
              </span>
            )}
          </div>
          <span className="font-mono text-xs text-[#8b949e]">{post.date}</span>
        </div>
        
        <h2 className="text-lg font-semibold text-[#ffffff] mb-2 tracking-tight transition-colors duration-200 group-hover:text-[#00d992]/90">
          {post.title}
        </h2>
        
        {post.description && (
          <p className="text-sm text-[#bdbdbd] line-clamp-2 leading-relaxed">
            {post.description}
          </p>
        )}
      </article>
    </Link>
  );
}
