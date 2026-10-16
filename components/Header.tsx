"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import ThemeToggle from "./ThemeToggle";
import { categoryLabel } from "@/lib/categories";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/category/공부", label: categoryLabel("공부") },
  { href: "/category/프로젝트", label: categoryLabel("프로젝트") },
  { href: "/category/논문리뷰", label: categoryLabel("논문리뷰") },
  { href: "/about", label: "About" },
];

function decode(path: string): string {
  try {
    return decodeURIComponent(path);
  } catch {
    return path;
  }
}

export default function Header() {
  const pathname = decode(usePathname());
  const [open, setOpen] = useState(false);

  // close the mobile menu after navigating
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  const isActive = (href: string) => pathname === href || (href !== "/" && pathname.startsWith(href));

  return (
    <header className="sticky top-0 z-50 bg-(--bg)/80 backdrop-blur-md border-b border-(--border) transition-colors duration-300">
      <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between gap-3">
        <Link href="/" className="group flex items-center gap-1.5 text-base font-semibold tracking-tight text-(--text-strong) shrink-0">
          <span>Tensors &amp; Notebooks</span>
          <span className="text-(--accent) transition-transform duration-300 group-hover:scale-125">✦</span>
        </Link>
        <div className="flex items-center gap-3 md:gap-5">
          {/* wide screens: links in a row */}
          <nav className="hidden md:flex gap-5 text-sm font-medium whitespace-nowrap">
            {NAV.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={`transition-colors duration-200 ${
                  isActive(href) ? "text-(--accent)" : "text-(--text-muted) hover:text-(--text-strong)"
                }`}
              >
                {label}
              </Link>
            ))}
          </nav>
          <div className="md:border-l border-(--border) md:pl-4 h-5 flex items-center shrink-0">
            <ThemeToggle />
          </div>
          {/* narrow screens: menu button */}
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-label={open ? "메뉴 닫기" : "메뉴 열기"}
            aria-expanded={open}
            aria-controls="mobile-nav"
            className="md:hidden flex items-center justify-center w-9 h-9 rounded-md border border-(--border) bg-(--bg-soft) text-(--text-muted) hover:text-(--accent) hover:border-(--accent) transition-colors duration-200 cursor-pointer"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" className="w-5 h-5" aria-hidden="true">
              {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
            </svg>
          </button>
        </div>
      </div>

      {open && (
        <nav id="mobile-nav" className="md:hidden border-t border-(--border) bg-(--bg)">
          <ul className="max-w-4xl mx-auto px-6 py-2">
            {NAV.map(({ href, label }) => (
              <li key={href}>
                <Link
                  href={href}
                  onClick={() => setOpen(false)}
                  className={`block py-3 text-base font-medium transition-colors duration-200 ${
                    isActive(href) ? "text-(--accent)" : "text-(--text) hover:text-(--text-strong)"
                  }`}
                >
                  {label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </header>
  );
}
