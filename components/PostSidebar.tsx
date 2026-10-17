import Link from "next/link";
import type { PostMeta } from "@/lib/posts";
import { buildTree, countPosts, sortedFolders, type Folder } from "@/components/FolderTree";
import { categoryLabel } from "@/lib/categories";
import SidebarShell from "@/components/SidebarShell";

// Short label for the list: "시계열분석 (6): 정상성과 차분, 언제 …" -> "6. 정상성과 차분",
// "ViT 리뷰: inductive bias는 …" -> "ViT 리뷰". The full title stays in the tooltip.
function navTitle(title: string): string {
  const series = title.match(/^.+?\((\d+)\):\s*([^,]+)/);
  if (series) return `${series[1]}. ${series[2].trim()}`;
  const colon = title.indexOf(":");
  return colon > 0 ? title.slice(0, colon) : title;
}

function contains(folder: Folder, slug: string): boolean {
  return folder.posts.some((p) => p.slug === slug) || [...folder.folders.values()].some((f) => contains(f, slug));
}

// reading order inside a folder: oldest first, numbered slugs in numeric order
function inReadingOrder(posts: PostMeta[]): PostMeta[] {
  return [...posts].sort(
    (a, b) => a.date.localeCompare(b.date) || a.slug.localeCompare(b.slug, undefined, { numeric: true })
  );
}

function PostList({ posts, current }: { posts: PostMeta[]; current: string }) {
  if (posts.length === 0) return null;
  return (
    <ul className="py-1">
      {inReadingOrder(posts).map((post) => {
        const active = post.slug === current;
        return (
          <li key={post.slug}>
            <Link
              href={`/posts/${post.slug}`}
              title={post.title}
              aria-current={active ? "page" : undefined}
              className={[
                "block -ml-[11px] border-l pl-2.5 py-1 text-sm leading-snug break-keep transition-colors duration-200",
                active
                  ? "border-(--accent) text-(--accent)"
                  : "border-transparent text-(--text-muted) hover:text-(--text)",
              ].join(" ")}
            >
              {navTitle(post.title)}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

function SideFolder({ folder, current }: { folder: Folder; current: string }) {
  return (
    <details open={contains(folder, current)} className="folder">
      <summary className="flex items-center gap-1.5 py-1 select-none">
        <svg className="folder-chevron w-3 h-3 shrink-0 text-(--text-muted)" viewBox="0 0 10 10" fill="currentColor" aria-hidden="true">
          <path d="M3 1l5 4-5 4z" />
        </svg>
        <span className="font-mono text-[13px] text-(--accent) truncate">{folder.name}</span>
        <span className="ml-auto pl-2 font-mono text-xs text-(--text-muted)">{countPosts(folder)}</span>
      </summary>
      <div className="ml-1 pl-2.5 border-l border-(--border)">
        {sortedFolders(folder).map((child) => (
          <SideFolder key={child.name} folder={child} current={current} />
        ))}
        <PostList posts={folder.posts} current={current} />
      </div>
    </details>
  );
}

// Left-hand list of the posts in the current post's category, folder by folder. SidebarShell places it:
// fixed at the left edge on wide screens, a drawer behind a button on narrow ones.
export default function PostSidebar({ posts, current, category }: { posts: PostMeta[]; current: string; category: string }) {
  const root = buildTree(posts);
  return (
    <SidebarShell>
      <Link
        href={`/category/${category}`}
        className="block font-mono text-[11px] tracking-widest uppercase text-(--text-muted) hover:text-(--accent) mb-3"
      >
        {categoryLabel(category)}
      </Link>
      {sortedFolders(root).map((folder) => (
        <SideFolder key={folder.name} folder={folder} current={current} />
      ))}
      <PostList posts={root.posts} current={current} />
    </SidebarShell>
  );
}
