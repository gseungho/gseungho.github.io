"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/category/공부정리", label: "공부정리" },
  { href: "/category/논문리뷰", label: "논문리뷰" },
  { href: "/category/창업일지", label: "창업일지" },
];

export default function Header() {
  const pathname = usePathname();
  return (
    <header
      style={{ borderBottom: "1px solid var(--border)", background: "var(--bg)" }}
      className="sticky top-0 z-50"
    >
      <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="text-lg font-bold" style={{ color: "var(--text)" }}>
          gseungho&apos;s log
        </Link>
        <nav className="flex gap-6 text-sm">
          {NAV.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              style={{ color: pathname === href ? "var(--accent)" : "var(--text-muted)" }}
              className="hover:opacity-80 transition-opacity"
            >
              {label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
