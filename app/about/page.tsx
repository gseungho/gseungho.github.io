import Link from "next/link";
import { getAllPosts, type PostMeta } from "@/lib/posts";

const AFFILIATION: { name: string; href?: string; note?: string }[] = [
  { name: "세종대학교 소프트웨어학과" },
  { name: "수학통계학과 복수전공" },
  { name: "ISDS Lab", href: "https://sites.google.com/view/isdslab/home", note: "2025.02 ~" },
];

const INTERESTS: { name: string; note?: string }[] = [
  { name: "Model Compression", note: "Pruning" },
  { name: "Class-Imbalanced Learning", note: "Loss Reweighting" },
  { name: "Dynamic Compute" },
  { name: "Efficient Sequence Models", note: "Linear Attention, State Space Models" },
];

const AWARDS = [
  {
    date: "2025-11-29",
    title: "우수논문상",
    venue: "한국지능시스템학회(KIIS) 2025년 추계학술대회",
    paper: "클래스 불균형 학습을 위한 로그 스케일 가중 교차 엔트로피",
    authors: "고승호, 이현석, 윤진희",
  },
];

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <section className="space-y-4">
      <h2 className="text-xs font-semibold uppercase tracking-[0.25em] text-(--text-muted)">{label}</h2>
      {children}
    </section>
  );
}

function PostRows({ posts }: { posts: PostMeta[] }) {
  return (
    <ul className="border-y border-(--border) divide-y divide-(--border)">
      {posts.map((post) => (
        <li key={post.slug}>
          <Link href={`/posts/${post.slug}`} className="group flex flex-col gap-1 py-3.5 sm:flex-row sm:items-baseline sm:gap-4">
            <span className="font-mono text-sm text-(--text-muted) shrink-0">{post.date}</span>
            <span className="text-base font-medium text-(--text-strong) leading-relaxed break-keep transition-colors duration-200 group-hover:text-(--accent)">
              {post.title}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}

export default function AboutPage() {
  const posts = getAllPosts();
  const reviews = posts.filter((p) => p.category === "논문리뷰");
  const projects = posts.filter((p) => p.category === "프로젝트");
  const studies = posts.filter((p) => p.category === "공부");

  return (
    <div className="space-y-10">
      <section className="py-2">
        <div className="font-mono text-xs font-semibold uppercase tracking-[0.25em] text-(--accent) mb-3">
          about
        </div>
        <h1 className="text-3xl md:text-4xl font-semibold leading-tight text-(--text-strong) tracking-tight mb-1">
          고승호
        </h1>
        <div className="font-mono text-sm text-(--text-muted) mb-5">Seung Ho Go</div>
        <p className="text-base text-(--text) leading-relaxed max-w-xl">
          매주 ISDS Lab 세미나에서 발표한 논문을 리뷰로 정리합니다. 공부 중인 내용을 정리한 노트와 직접 만든 프로젝트의 기록도 함께 남깁니다.
        </p>

        <div className="mt-8 border-b border-dashed border-(--border) w-full opacity-60" />
      </section>

      <Section label="Affiliation">
        <ul className="space-y-2 text-base text-(--text)">
          {AFFILIATION.map(({ name, href, note }) => (
            <li key={name}>
              {href ? (
                <a
                  href={href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-(--text-strong) underline decoration-(--border) underline-offset-4 transition-colors duration-200 hover:text-(--accent) hover:decoration-(--accent)"
                >
                  {name}
                </a>
              ) : (
                name
              )}
              {note && <span className="ml-2 font-mono text-sm text-(--text-muted)">{note}</span>}
            </li>
          ))}
        </ul>
      </Section>

      <Section label="Research Interests">
        <ul className="space-y-2 text-base text-(--text)">
          {INTERESTS.map(({ name, note }) => (
            <li key={name}>
              <span className="text-(--text-strong)">{name}</span>
              {note && <span className="text-(--text-muted)"> ({note})</span>}
            </li>
          ))}
        </ul>
      </Section>

      <Section label="Awards">
        <ul className="border-y border-(--border) divide-y divide-(--border)">
          {AWARDS.map((award) => (
            <li key={`${award.date}-${award.title}`} className="flex flex-col gap-1 py-3.5 sm:flex-row sm:items-baseline sm:gap-4">
              <span className="font-mono text-sm text-(--text-muted) shrink-0">{award.date}</span>
              <div className="text-base leading-relaxed">
                <div className="text-(--text-strong)">
                  {award.title} · {award.venue}
                </div>
                <div className="text-(--text-muted)">
                  {award.paper} ({award.authors})
                </div>
              </div>
            </li>
          ))}
        </ul>
      </Section>

      {reviews.length > 0 && (
        <Section label="Paper Reviews">
          <PostRows posts={reviews} />
        </Section>
      )}

      {projects.length > 0 && (
        <Section label="Projects">
          <PostRows posts={projects} />
        </Section>
      )}

      {studies.length > 0 && (
        <Section label="Study">
          <PostRows posts={studies} />
        </Section>
      )}

      <Section label="Contact">
        <ul className="space-y-2 text-base text-(--text)">
          <li>
            <span className="inline-block w-20 font-mono text-sm text-(--text-muted)">Email</span>
            {/* written out rather than linked, to keep it away from address scrapers */}
            <span className="font-mono">gseungho999 [at] gmail.com</span>
          </li>
          <li>
            <span className="inline-block w-20 font-mono text-sm text-(--text-muted)">GitHub</span>
            <a
              href="https://github.com/gseungho"
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono transition-colors duration-200 hover:text-(--accent)"
            >
              github.com/gseungho
            </a>
          </li>
        </ul>
      </Section>
    </div>
  );
}
