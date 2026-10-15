import Link from "next/link";
import type { PostMeta } from "@/lib/posts";

type Folder = {
  name: string;
  folders: Map<string, Folder>;
  posts: PostMeta[];
};

// "Deep Learning/Transformer" puts a post in Deep Learning > Transformer.
// Posts without a subcategory stay at the root.
function buildTree(posts: PostMeta[]): Folder {
  const root: Folder = { name: "", folders: new Map(), posts: [] };
  for (const post of posts) {
    let node = root;
    const parts = (post.subcategory ?? "")
      .split("/")
      .map((part) => part.trim().normalize("NFC"))
      .filter(Boolean);
    for (const part of parts) {
      let child = node.folders.get(part);
      if (!child) {
        child = { name: part, folders: new Map(), posts: [] };
        node.folders.set(part, child);
      }
      node = child;
    }
    node.posts.push(post);
  }
  return root;
}

function countPosts(folder: Folder): number {
  let total = folder.posts.length;
  for (const child of folder.folders.values()) total += countPosts(child);
  return total;
}

// date of the newest post anywhere under the folder ("YYYY-MM-DD" compares as text)
function latestDate(folder: Folder): string {
  let latest = "";
  for (const post of folder.posts) if (post.date > latest) latest = post.date;
  for (const child of folder.folders.values()) {
    const date = latestDate(child);
    if (date > latest) latest = date;
  }
  return latest;
}

// most recently updated folder first; same date falls back to the name
function sortedFolders(folder: Folder): Folder[] {
  return [...folder.folders.values()].sort(
    (a, b) => latestDate(b).localeCompare(latestDate(a)) || a.name.localeCompare(b.name)
  );
}

function PostRows({ posts }: { posts: PostMeta[] }) {
  return (
    <ul>
      {posts.map((post) => (
        <li key={post.slug}>
          <Link href={`/posts/${post.slug}`} className="group flex flex-col gap-1 py-4 sm:flex-row sm:items-baseline sm:gap-4">
            <span className="font-mono text-sm text-(--text-muted) shrink-0">{post.date}</span>
            <span className="min-w-0">
              <span className="block text-lg font-semibold text-(--text-strong) leading-snug tracking-tight break-keep transition-colors duration-200 group-hover:text-(--accent)">
                {post.title}
              </span>
              {post.description && (
                <span className="block mt-1.5 text-[0.95rem] text-(--text) leading-relaxed">
                  {post.description}
                </span>
              )}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}

function FolderNode({ folder }: { folder: Folder }) {
  return (
    <details open className="folder">
      <summary className="flex items-center gap-2.5 py-2.5 px-2 -mx-2 rounded-md select-none transition-colors duration-200 hover:bg-(--bg-soft)">
        <svg
          className="folder-chevron w-3 h-3 shrink-0 text-(--text-muted)"
          viewBox="0 0 10 10"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M3 1l5 4-5 4z" />
        </svg>
        <span className="font-mono text-base text-(--accent)">{folder.name}</span>
        <span className="ml-auto font-mono text-sm text-(--text-muted)">{countPosts(folder)}</span>
      </summary>
      <div className="ml-1 pl-4 border-l border-(--border)">
        {sortedFolders(folder).map((child) => (
          <FolderNode key={child.name} folder={child} />
        ))}
        <PostRows posts={folder.posts} />
      </div>
    </details>
  );
}

export default function FolderTree({ posts }: { posts: PostMeta[] }) {
  const root = buildTree(posts);
  return (
    <div>
      {sortedFolders(root).map((folder) => (
        <FolderNode key={folder.name} folder={folder} />
      ))}
      <PostRows posts={root.posts} />
    </div>
  );
}
