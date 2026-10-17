"use client";

import { useEffect, useState } from "react";

// Holds the post list. 1440px and wider: always shown, fixed at the left edge (mirrors the table of
// contents). Narrower: hidden behind a floating "목록" button and opened as a drawer from the left.
export default function SidebarShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);

  // Escape closes the drawer; the page behind it does not scroll while it is open
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = overflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-expanded={open}
        aria-controls="post-sidebar"
        className="min-[1440px]:hidden fixed bottom-5 left-5 z-40 inline-flex items-center gap-2 px-3.5 py-2 rounded-full border border-(--border) bg-(--bg-soft) text-sm text-(--text) shadow-lg hover:text-(--accent) hover:border-(--accent) transition-colors duration-200"
      >
        <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
          <path d="M2.5 4h11M2.5 8h11M2.5 12h7" strokeLinecap="round" />
        </svg>
        <span>목록</span>
      </button>

      {open && (
        <div className="min-[1440px]:hidden fixed inset-0 z-[55] bg-black/50" onClick={() => setOpen(false)} aria-hidden="true" />
      )}

      <nav
        id="post-sidebar"
        aria-label="글 목록"
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("a")) setOpen(false);
        }}
        className={[
          open ? "block" : "hidden",
          "fixed inset-y-0 left-0 z-[60] w-72 max-w-[85vw] overflow-y-auto bg-(--bg) border-r border-(--border) px-5 pt-5 pb-10",
          "min-[1440px]:block min-[1440px]:inset-y-auto min-[1440px]:top-28 min-[1440px]:left-6 min-[1440px]:z-auto min-[1440px]:w-52",
          "min-[1440px]:max-h-[calc(100vh-10rem)] min-[1440px]:bg-transparent min-[1440px]:border-0 min-[1440px]:p-0",
        ].join(" ")}
      >
        <div className="flex justify-end min-[1440px]:hidden">
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="목록 닫기"
            className="p-1 -mr-1 text-(--text-muted) hover:text-(--accent) transition-colors"
          >
            <svg className="w-5 h-5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        {children}
      </nav>
    </>
  );
}
