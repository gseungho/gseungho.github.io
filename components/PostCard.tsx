import Link from "next/link";
import type { PostMeta } from "@/lib/posts";

export default function PostCard({ post }: { post: PostMeta }) {
  return (
    <Link href={`/posts/${post.slug}`}>
      <article
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border)",
          borderRadius: "10px",
          padding: "1.2rem 1.4rem",
        }}
        className="hover:border-[var(--accent)] transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-2 mb-2">
          <span
            style={{ background: "var(--bg-hover)", color: "var(--accent)", fontSize: "0.75rem" }}
            className="px-2 py-0.5 rounded-full"
          >
            {post.category}
          </span>
          <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>{post.date}</span>
        </div>
        <h2 style={{ fontSize: "1.05rem", fontWeight: 600 }} className="mb-1">
          {post.title}
        </h2>
        {post.description && (
          <p style={{ color: "var(--text-muted)", fontSize: "0.88rem" }}>{post.description}</p>
        )}
      </article>
    </Link>
  );
}
