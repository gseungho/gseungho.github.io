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
    <header className="sticky top-0 z-50 bg-[#101010]/80 backdrop-blur-md border-b border-[#3d3a39]">
      <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="group flex items-center gap-1.5 text-base font-semibold tracking-tight text-[#ffffff]">
          <span>gseungho&apos;s log</span>
          <span className="text-[#00d992] transition-transform duration-300 group-hover:scale-125">✦</span>
        </Link>
        <nav className="flex gap-5 text-sm font-medium">
          {NAV.map(({ href, label }) => {
            const isActive = pathname === href || (href !== "/" && pathname.startsWith(href));
            return (
              <Link
                key={href}
                href={href}
                className={`transition-colors duration-200 ${
                  isActive 
                    ? "text-[#00d992]" 
                    : "text-[#8b949e] hover:text-[#ffffff]"
                }`}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
