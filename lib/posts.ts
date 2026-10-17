import fs from "fs";
import path from "path";
import matter from "gray-matter";
import GithubSlugger from "github-slugger";
import katex from "katex";

const postsDir = path.join(process.cwd(), "content/posts");

export type Heading = { depth: 2 | 3; text: string; html: string; id: string };

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Heading text for the table of contents: $...$ rendered with KaTeX like the post body,
// everything else escaped as plain text.
function headingHtml(source: string): string {
  return source
    .split(/(\$[^$]+\$)/g)
    .map((part) =>
      /^\$[^$]+\$$/.test(part)
        ? katex.renderToString(part.slice(1, -1), { throwOnError: false })
        : escapeHtml(part)
    )
    .join("");
}

/**
 * Pull the ## / ### headings out of the markdown source for the table of contents.
 * Ids are produced with github-slugger so they match the ids rehype-slug writes
 * into the rendered HTML.
 */
export function getHeadings(content: string): Heading[] {
  const slugger = new GithubSlugger();
  const headings: Heading[] = [];
  let inFence = false;

  for (const line of content.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;

    const m = /^(#{2,3})\s+(.*\S)\s*$/.exec(line);
    if (!m) continue;

    // strip the inline markers that never reach the rendered text content
    const source = m[2]
      .replace(/`([^`]*)`/g, "$1")
      .replace(/\*\*([^*]*)\*\*/g, "$1")
      .replace(/\*([^*]*)\*/g, "$1")
      .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
      .trim();
    // rehype-slug sees inline math as its LaTeX source, so the id uses the text without the $ signs
    const text = source.replace(/\$([^$]+)\$/g, "$1");

    headings.push({ depth: m[1].length as 2 | 3, text, html: headingHtml(source), id: slugger.slug(text) });
  }

  return headings;
}

export type PostMeta = {
  slug: string;
  title: string;
  date: string;
  category: string;
  subcategory?: string;
  tags: string[];
  description: string;
};

export type Post = PostMeta & { content: string };

export function getAllPosts(): PostMeta[] {
  const files = fs.readdirSync(postsDir);
  return files
    .filter((f) => f.endsWith(".md"))
    .map((filename) => {
      const slug = filename.replace(/\.md$/, "");
      const raw = fs.readFileSync(path.join(postsDir, filename), "utf8");
      const { data } = matter(raw);
      return {
        slug,
        title: data.title ?? "",
        date: data.date ?? "",
        category: data.category ?? "기타",
        subcategory: data.subcategory,
        tags: data.tags ?? [],
        description: data.description ?? "",
      };
    })
    .sort((a, b) => (a.date < b.date ? 1 : -1));
}

export function getPostBySlug(slug: string): Post {
  const raw = fs.readFileSync(path.join(postsDir, `${slug}.md`), "utf8");
  const { data, content } = matter(raw);
  return {
    slug,
    title: data.title ?? "",
    date: data.date ?? "",
    category: data.category ?? "기타",
    subcategory: data.subcategory,
    tags: data.tags ?? [],
    description: data.description ?? "",
    content,
  };
}

export function getAllCategories(): string[] {
  const posts = getAllPosts();
  return [...new Set(posts.map((p) => p.category))];
}

export function getAllTags(): string[] {
  const posts = getAllPosts();
  const allTags = posts.flatMap((p) => p.tags);
  return [...new Set(allTags)];
}
