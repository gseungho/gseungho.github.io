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
            className="inline-block px-2.5 py-0.5 text-xs font-semibold text-[#00d992] bg-[#1a1a1a]/60 border border-[#3d3a39] rounded-full hover:border-[#00d992]/80 transition-colors"
          >
            {post.category}
          </Link>
          <span className="font-mono text-xs text-[#8b949e]">{post.date}</span>
        </div>
        
        <h1 className="text-2xl md:text-3.5xl font-semibold leading-tight text-[#ffffff] tracking-tight">
          {post.title}
        </h1>
        
        {post.description && (
          <p className="text-base text-[#bdbdbd] leading-relaxed italic border-l-2 border-[#3d3a39] pl-3">
            {post.description}
          </p>
        )}
        
        {post.tags.length > 0 && (
          <div className="flex gap-2 flex-wrap pt-1">
            {post.tags.map((tag) => (
              <span
                key={tag}
                className="bg-[#1a1a1a] text-[#8b949e] border border-[#3d3a39] text-xs px-2 py-0.5 rounded-sm transition-colors duration-200 hover:text-[#ffffff] hover:border-[#8b949e]"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}
      </header>

      {/* Voltagent Dashed Line Section Divider */}
      <div className="border-b border-dashed border-[#4f5d75]/30 w-full" />

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
      <div className="border-b border-dashed border-[#4f5d75]/30 w-full mt-12" />

      {/* Back Button (button-outline-on-dark style) */}
      <div className="pt-4">
        <Link 
          href="/" 
          className="inline-flex items-center gap-1.5 px-4 py-2 border border-[#3d3a39] rounded-md bg-[#101010] text-[#bdbdbd] hover:text-[#00d992] hover:border-[#00d992] text-sm transition-all duration-300 font-medium cursor-pointer"
        >
          <span>← Back to Journal</span>
        </Link>
      </div>
    </article>
  );
}
