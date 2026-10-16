import fs from "fs";
import path from "path";
import Link from "next/link";
import { getAllPosts } from "@/lib/posts";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "About",
  description:
    "Seung Ho Go (고승호), undergraduate researcher at the ISDS Lab, Sejong University, working on model compression and pruning.",
  alternates: { canonical: "/about" },
};

const SEJONG = "https://www.sejong.ac.kr/";
const ISDS_LAB = "https://sites.google.com/view/isdslab/home";
const ADVISOR = "https://sites.google.com/view/isdslab/members/professor";

const INTERESTS: { name: string; note?: string }[] = [
  { name: "Model Compression", note: "Pruning" },
  { name: "Class-Imbalanced Learning", note: "Loss Reweighting" },
  { name: "Dynamic Compute" },
  { name: "Efficient Sequence Models", note: "Linear Attention, State Space Models" },
];

const NEWS: { date: string; text: string }[] = [
  { date: "Nov. 2025", text: "Received the Outstanding Paper Award at the KIIS 2025 Fall Conference." },
  { date: "Feb. 2025", text: "Joined the Intelligent Systems and Data Science Lab at Sejong University." },
];

const PUBLICATIONS = [
  {
    title: "Log-Scaled Weighted Cross-Entropy for Class-Imbalanced Learning",
    original: "클래스 불균형 학습을 위한 로그 스케일 가중 교차 엔트로피",
    authors: [
      { name: "Seung Ho Go", me: true },
      { name: "Hyunseok Lee" },
      { name: "Jin Hee Yoon", corresponding: true, href: ADVISOR },
    ],
    venue: "Korean Institute of Intelligent Systems (KIIS) Fall Conference, 2025",
    note: "Outstanding Paper Award",
  },
];

const EXPERIENCES = [
  { name: "Intelligent Systems and Data Science Lab, Sejong University", href: ISDS_LAB, period: "Feb. 2025 - Present" },
];

const AWARDS = [
  { date: "Nov. 2025", title: "Outstanding Paper Award", venue: "KIIS 2025 Fall Conference" },
];

// Profile photo: drop public/images/profile.{jpg,jpeg,png,webp} in and it shows up next to the name.
const PHOTO = ["jpg", "jpeg", "png", "webp"]
  .map((ext) => `/images/profile.${ext}`)
  .find((p) => fs.existsSync(path.join(process.cwd(), "public", p)));

function Ext({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-(--text-strong) underline decoration-(--border) underline-offset-4 transition-colors duration-200 hover:text-(--accent) hover:decoration-(--accent)"
    >
      {children}
    </a>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <section className="space-y-5">
      <h2 className="flex items-center gap-3 text-sm font-semibold uppercase tracking-[0.2em] text-(--text-strong)">
        <span className="h-4 w-1 rounded-full bg-(--accent)" aria-hidden="true" />
        {label}
      </h2>
      {children}
    </section>
  );
}

// date column on wide screens, date above the text on phones
function DatedRow({ date, children }: { date: string; children: React.ReactNode }) {
  return (
    <li className="flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:gap-4">
      <span className="font-mono text-sm text-(--text-muted) shrink-0 sm:w-24">{date}</span>
      <div className="text-base text-(--text) leading-relaxed">{children}</div>
    </li>
  );
}

export default function AboutPage() {
  const posts = getAllPosts();
  const count = (category: string) => posts.filter((p) => p.category === category).length;
  const blog = [
    { label: "Paper Reviews", href: "/category/논문리뷰", n: count("논문리뷰") },
    { label: "Study Notes", href: "/category/공부", n: count("공부") },
    { label: "Projects", href: "/category/프로젝트", n: count("프로젝트") },
  ];

  return (
    <div className="space-y-14">
      <section className="py-2">
        <div className="font-mono text-xs font-semibold uppercase tracking-[0.25em] text-(--accent) mb-3">
          about
        </div>
        <div className="flex items-center gap-5 md:gap-6 mb-6">
          {PHOTO && (
            <img
              src={PHOTO}
              alt="Seung Ho Go"
              className="w-30 h-30 md:w-36 md:h-36 rounded-full object-cover border border-(--border) shrink-0"
            />
          )}
          <div>
            <h1 className="text-3xl md:text-4xl font-semibold leading-tight text-(--text-strong) tracking-tight mb-1">
              Seung Ho Go
            </h1>
            <div className="text-sm text-(--text-muted)">고승호 · Undergraduate Researcher</div>
          </div>
        </div>
        <p className="text-base text-(--text) leading-relaxed">
          I am an undergraduate student in the Department of Software at <Ext href={SEJONG}>Sejong University</Ext>,
          double majoring in Mathematics and Statistics. Since February 2025, I have been an undergraduate researcher at
          the <Ext href={ISDS_LAB}>Intelligent Systems and Data Science Lab</Ext>, advised by{" "}
          <Ext href={ADVISOR}>Jin Hee Yoon</Ext>. My current research focuses on model compression, especially pruning.
          I am also interested in class-imbalanced learning and models that adapt their computation to each input.
        </p>
      </section>

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

      <Section label="News">
        <ul className="space-y-3">
          {NEWS.map(({ date, text }) => (
            <DatedRow key={date + text} date={date}>
              {text}
            </DatedRow>
          ))}
        </ul>
      </Section>

      <Section label="Education">
        <div className="text-base leading-relaxed">
          <div className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-0.5">
            <span className="text-(--text-strong) font-medium">Sejong University, Seoul</span>
            <span className="font-mono text-sm text-(--text-muted)">Mar. 2024 - Aug. 2027 (expected)</span>
          </div>
          <div className="text-(--text)">B.S. in Software, Double Major in Mathematics and Statistics</div>
          <div className="text-(--text-muted)">
            Advisor: Prof. <Ext href={ADVISOR}>Jin Hee Yoon</Ext>
          </div>
        </div>
      </Section>

      <Section label="Publications">
        <p className="text-sm text-(--text-muted)">† corresponding author</p>
        <ul className="space-y-5">
          {PUBLICATIONS.map((pub) => (
            <li key={pub.title} className="text-base leading-relaxed">
              <div className="text-(--text-strong) font-medium">{pub.title}</div>
              {pub.original && <div className="text-sm text-(--text-muted)">{pub.original}</div>}
              <div className="text-(--text)">
                {pub.authors.map((a, i) => (
                  <span key={a.name}>
                    {i > 0 && (i === pub.authors.length - 1 ? ", and " : ", ")}
                    <span className={a.me ? "font-semibold text-(--text-strong)" : undefined}>{a.name}</span>
                    {a.corresponding && "†"}
                  </span>
                ))}
              </div>
              <div className="text-(--text-muted) italic">{pub.venue}</div>
              {pub.note && <div className="text-sm text-(--accent) mt-0.5">{pub.note}</div>}
            </li>
          ))}
        </ul>
      </Section>

      <Section label="Research Experience">
        <ul className="space-y-2 text-base">
          {EXPERIENCES.map((e) => (
            <li key={e.name} className="flex flex-col sm:flex-row sm:items-baseline sm:justify-between gap-0.5">
              <Ext href={e.href}>{e.name}</Ext>
              <span className="font-mono text-sm text-(--text-muted) shrink-0">{e.period}</span>
            </li>
          ))}
        </ul>
      </Section>

      <Section label="Awards">
        <ul className="space-y-3">
          {AWARDS.map((a) => (
            <DatedRow key={a.date + a.title} date={a.date}>
              <span className="text-(--text-strong)">{a.title}</span>, {a.venue}
            </DatedRow>
          ))}
        </ul>
      </Section>

      <Section label="Blog">
        <ul className="space-y-2 text-base">
          {blog.map((b) => (
            <li key={b.href}>
              <Link href={b.href} className="group inline-flex items-baseline gap-2">
                <span className="text-(--text-strong) transition-colors duration-200 group-hover:text-(--accent)">{b.label}</span>
                <span className="font-mono text-sm text-(--text-muted)">{b.n}</span>
              </Link>
            </li>
          ))}
        </ul>
      </Section>

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
