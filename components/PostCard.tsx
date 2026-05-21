import Link from "next/link";
import type { PostMeta } from "@/lib/posts";

export default function PostCard({ post }: { post: PostMeta }) {
  return (
    <Link href={`/posts/${post.slug}`} className="block group">
      <article className="bg-(--bg) border border-(--border) rounded-lg p-5 transition-all duration-300 cursor-pointer group-hover:border-(--accent) group-hover:bg-(--bg-soft)">
        <div className="flex items-center justify-between gap-4 mb-3">
          {/* Voltagent button-pill-tag style */}
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="inline-block px-2.5 py-0.5 text-xs font-semibold text-(--text-strong) bg-(--bg) border border-(--border) rounded-full transition-colors duration-300 group-hover:border-(--accent)/60 group-hover:text-(--accent) shrink-0">
              {post.category}
            </span>
            {post.subcategory && (
              <span className="font-mono text-[11px] text-(--text-muted) truncate">
                {post.subcategory.replace(/\//g, " / ")}
              </span>
            )}
          </div>
          <span className="font-mono text-xs text-(--text-muted)">{post.date}</span>
        </div>
        
        <h2 className="text-lg font-semibold text-(--text-strong) mb-2 tracking-tight transition-colors duration-200 group-hover:text-(--accent-soft)">
          {post.title}
        </h2>
        
        {post.description && (
          <p className="text-sm text-(--text) line-clamp-2 leading-relaxed">
            {post.description}
          </p>
        )}
      </article>
    </Link>
  );
}
