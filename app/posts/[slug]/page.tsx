import { getAllPosts, getPostBySlug } from "@/lib/posts";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import Link from "next/link";

export async function generateStaticParams() {
  return getAllPosts().map((p) => ({ slug: p.slug }));
}

export default async function PostPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const post = getPostBySlug(slug);

  return (
    <article>
      <div className="mb-8">
        <div className="flex items-center gap-2 mb-3">
          <Link
            href={`/category/${post.category}`}
            style={{ color: "var(--accent)", fontSize: "0.85rem" }}
            className="hover:underline"
          >
            {post.category}
          </Link>
          <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>· {post.date}</span>
        </div>
        <h1 className="text-3xl font-bold mb-2">{post.title}</h1>
        {post.description && (
          <p style={{ color: "var(--text-muted)" }}>{post.description}</p>
        )}
        {post.tags.length > 0 && (
          <div className="flex gap-2 mt-3 flex-wrap">
            {post.tags.map((tag) => (
              <span
                key={tag}
                style={{ background: "var(--bg-card)", color: "var(--text-muted)", fontSize: "0.78rem" }}
                className="px-2 py-0.5 rounded-full border border-[var(--border)]"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}
      </div>
      <hr style={{ borderColor: "var(--border)", marginBottom: "2rem" }} />
      <div className="prose">
        <MDXRemote
          source={post.content}
          options={{
            mdxOptions: {
              remarkPlugins: [remarkGfm, remarkMath],
              rehypePlugins: [rehypeKatex, rehypeHighlight],
            },
          }}
        />
      </div>
      <div className="mt-12">
        <Link href="/" style={{ color: "var(--accent)" }} className="hover:underline text-sm">
          ← 목록으로
        </Link>
      </div>
    </article>
  );
}
