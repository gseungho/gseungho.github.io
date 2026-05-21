"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import ThemeToggle from "./ThemeToggle";

const NAV = [
  { href: "/", label: "Home" },
  { href: "/category/공부", label: "공부" },
  { href: "/category/프로젝트", label: "프로젝트" },
  { href: "/category/논문리뷰", label: "논문리뷰" },
];

export default function Header() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-50 bg-(--bg)/80 backdrop-blur-md border-b border-(--border) transition-colors duration-300">
      <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="group flex items-center gap-1.5 text-base font-semibold tracking-tight text-(--text-strong)">
          <span>gseungho&apos;s log</span>
          <span className="text-(--accent) transition-transform duration-300 group-hover:scale-125">✦</span>
        </Link>
        <div className="flex items-center gap-5">
          <nav className="flex gap-5 text-sm font-medium">
            {NAV.map(({ href, label }) => {
              const isActive = pathname === href || (href !== "/" && pathname.startsWith(href));
              return (
                <Link
                  key={href}
                  href={href}
                  className={`transition-colors duration-200 ${
                    isActive 
                      ? "text-(--accent)" 
                      : "text-(--text-muted) hover:text-(--text-strong)"
                  }`}
                >
                  {label}
                </Link>
              );
            })}
          </nav>
          <div className="border-l border-(--border) pl-4 h-5 flex items-center">
            <ThemeToggle />
          </div>
        </div>
      </div>
    </header>
  );
}
