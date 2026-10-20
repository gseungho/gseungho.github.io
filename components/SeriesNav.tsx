import Link from "next/link";
import type { PostMeta } from "@/lib/posts";

function NavCard({ post, label, next }: { post: PostMeta; label: string; next?: boolean }) {
  return (
    <Link
      href={`/posts/${post.slug}`}
      className={[
        "group block rounded-md border border-(--border) px-4 py-3 transition-colors duration-200 hover:border-(--accent)",
        next ? "md:col-start-2 md:text-right" : "",
      ].join(" ")}
    >
      <span className="block text-sm text-(--text-muted) group-hover:text-(--accent)">{label}</span>
      <span className="mt-1 block text-base leading-snug break-keep text-(--text-strong)">{post.title}</span>
    </Link>
  );
}

// Previous and next post in the same folder, in the sidebar's reading order. The page renders it after
// the body, so it sits below the footnotes.
export default function SeriesNav({ prev, next }: { prev?: PostMeta; next?: PostMeta }) {
  if (!prev && !next) return null;
  return (
    <nav aria-label="이전 글과 다음 글" className="grid gap-3 md:grid-cols-2">
      {prev && <NavCard post={prev} label="← 이전 글" />}
      {next && <NavCard post={next} label="다음 글 →" next />}
    </nav>
  );
}
