import Link from "next/link";
import type { PostMeta } from "@/lib/posts";

// One row of a post list. Rows are separated by lines drawn by the parent list (divide-y).
export default function PostCard({ post }: { post: PostMeta }) {
  return (
    <article className="py-7">
      <Link href={`/posts/${post.slug}`} className="block group">
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

        <h2 className="text-xl md:text-2xl font-semibold leading-snug break-keep text-(--text-strong) mb-2 tracking-tight transition-colors duration-200 group-hover:text-(--accent-soft)">
          {post.title}
        </h2>

        {post.description && (
          <p className="text-[0.95rem] text-(--text) leading-relaxed">
            {post.description}
          </p>
        )}
      </Link>

      {post.tags.length > 0 && (
        <div className="flex gap-2 flex-wrap mt-3">
          {post.tags.map((tag) => (
            <Link
              key={tag}
              href={`/tag/${encodeURIComponent(tag)}`}
              className="bg-(--bg-soft) text-(--text-muted) border border-(--border) text-xs px-2 py-0.5 rounded-sm transition-colors duration-200 hover:text-(--accent) hover:border-(--accent)"
            >
              #{tag}
            </Link>
          ))}
        </div>
      )}
    </article>
  );
}
