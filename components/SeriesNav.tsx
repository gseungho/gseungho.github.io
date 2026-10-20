import Link from "next/link";
import type { PostMeta } from "@/lib/posts";

// No box: a small label with an arrow that nudges outward on hover, and the title, which takes the
// accent colour on hover like the titles in the post lists.
function NavLink({ post, next }: { post: PostMeta; next?: boolean }) {
  return (
    <Link href={`/posts/${post.slug}`} className={["group block", next ? "md:col-start-2 md:text-right" : ""].join(" ")}>
      <span className={["flex items-center gap-1.5 text-sm text-(--text-muted)", next ? "md:justify-end" : ""].join(" ")}>
        {!next && (
          <span aria-hidden="true" className="transition-transform duration-200 group-hover:-translate-x-1">
            ←
          </span>
        )}
        {next ? "다음 글" : "이전 글"}
        {next && (
          <span aria-hidden="true" className="transition-transform duration-200 group-hover:translate-x-1">
            →
          </span>
        )}
      </span>
      <span className="mt-1.5 block text-lg font-semibold leading-snug tracking-tight break-keep text-(--text-strong) transition-colors duration-200 group-hover:text-(--accent-soft)">
        {post.title}
      </span>
    </Link>
  );
}

// Previous and next post in the same folder, in the sidebar's reading order. The page renders it after
// the body, so it sits below the footnotes. From 1440px the sidebar (SidebarShell) lists the folder
// beside the post at all times, so the links only show on narrower screens, where it is a drawer.
export default function SeriesNav({ prev, next }: { prev?: PostMeta; next?: PostMeta }) {
  if (!prev && !next) return null;
  return (
    <nav aria-label="이전 글과 다음 글" className="grid gap-x-12 gap-y-7 md:grid-cols-2 min-[1440px]:hidden">
      {prev && <NavLink post={prev} />}
      {next && <NavLink post={next} next />}
    </nav>
  );
}
