import { getAllPosts, getAllCategories } from "@/lib/posts";
import FolderTree from "@/components/FolderTree";
import type { Metadata } from "next";

export async function generateStaticParams() {
  const defaultCategories = ["공부", "프로젝트", "논문리뷰", "창업일지"];
  const allCategories = Array.from(new Set([
    ...defaultCategories,
    ...getAllCategories()
  ]));
  return allCategories.map((name) => ({ name }));
}

export async function generateMetadata({ params }: { params: Promise<{ name: string }> }): Promise<Metadata> {
  const { name } = await params;
  const category = decodeURIComponent(name).normalize("NFC");
  return {
    title: category,
    description: `Tensors & Notebooks의 ${category} 글 모음입니다.`,
    alternates: { canonical: `/category/${encodeURIComponent(category)}` },
  };
}

export default async function CategoryPage({ params }: { params: Promise<{ name: string }> }) {
  const { name } = await params;
  const category = decodeURIComponent(name).normalize("NFC");
  const posts = getAllPosts().filter((p) => p.category.normalize("NFC") === category);

  // Render Empty State if no posts are found in this category
  if (posts.length === 0) {
    return (
      <div className="space-y-6">
        <section className="pt-2">
          <div className="font-mono text-xs font-semibold uppercase tracking-[0.25em] text-(--accent) mb-3">
            folder / categories
          </div>
          <h1 className="text-3xl md:text-4xl font-semibold leading-tight text-(--text-strong) tracking-tight mb-3">
            {category}
          </h1>
          <div className="flex items-center gap-2">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-(--text-muted) opacity-50" />
            <span className="font-mono text-sm text-(--text-muted)">0 index files available</span>
          </div>
        </section>

        <section className="py-16 text-center border border-dashed border-(--border) rounded-lg bg-(--bg-soft)/30">
          <div className="text-4xl mb-4">📂</div>
          <h3 className="text-lg font-medium text-(--text-strong) mb-1">아직 작성된 글이 없습니다</h3>
          <p className="text-sm text-(--text-muted)">이 카테고리에 첫 번째 글을 작성해 보세요.</p>
        </section>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Category Header Band */}
      <section className="pt-2">
        <div className="font-mono text-xs font-semibold uppercase tracking-[0.25em] text-(--accent) mb-3">
          folder / categories
        </div>
        <h1 className="text-3xl md:text-4xl font-semibold leading-tight text-(--text-strong) tracking-tight mb-3">
          {category}
        </h1>
        <div className="flex items-center gap-2">
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-(--accent) animate-pulse" />
          <span className="font-mono text-sm text-(--text-muted)">{posts.length} index files available</span>
        </div>
      </section>

      {/* Folder tree built from each post's subcategory path */}
      <section>
        <FolderTree posts={posts} />
      </section>
    </div>
  );
}
