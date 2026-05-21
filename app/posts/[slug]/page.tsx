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
    <article className="space-y-8">
      {/* Post Header */}
      <header className="space-y-4">
        <div className="flex items-center gap-3">
          <Link
            href={`/category/${post.category}`}
            className="inline-block px-2.5 py-0.5 text-xs font-semibold text-(--accent) bg-(--bg-soft)/60 border border-(--border) rounded-full hover:border-(--accent)/80 transition-colors"
          >
            {post.category}
          </Link>
          <span className="font-mono text-xs text-(--text-muted)">{post.date}</span>
        </div>
        
        <h1 className="text-2xl md:text-3.5xl font-semibold leading-tight text-(--text-strong) tracking-tight">
          {post.title}
        </h1>
        
        {post.description && (
          <p className="text-base text-(--text) leading-relaxed italic border-l-2 border-(--border) pl-3">
            {post.description}
          </p>
        )}
        
        {post.tags.length > 0 && (
          <div className="flex gap-2 flex-wrap pt-1">
            {post.tags.map((tag) => (
              <Link
                key={tag}
                href={`/tag/${encodeURIComponent(tag)}`}
                className="bg-(--bg-soft) text-(--text-muted) border border-(--border) text-xs px-2 py-0.5 rounded-sm transition-colors duration-200 hover:text-(--accent) hover:border-(--accent) cursor-pointer"
              >
                #{tag}
              </Link>
            ))}
          </div>
        )}
      </header>

      {/* Voltagent Dashed Line Section Divider */}
      <div className="border-b border-dashed border-(--border) w-full opacity-60" />

      {/* Post Body (MDX Content) */}
      <div className="prose max-w-none">
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

      {/* Voltagent Dashed Line Section Divider */}
      <div className="border-b border-dashed border-(--border) w-full mt-12 opacity-60" />

      {/* Back Button (button-outline-on-dark style) */}
      <div className="pt-4">
        <Link 
          href="/" 
          className="inline-flex items-center gap-1.5 px-4 py-2 border border-(--border) rounded-md bg-(--bg) text-(--text) hover:text-(--accent) hover:border-(--accent) text-sm transition-all duration-300 font-medium cursor-pointer"
        >
          <span>← Back to Journal</span>
        </Link>
      </div>
    </article>
  );
}
