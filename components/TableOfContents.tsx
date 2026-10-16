"use client";

import { useEffect, useMemo, useState } from "react";
import type { Heading } from "@/lib/posts";

export default function TableOfContents({ headings }: { headings: Heading[] }) {
  const [activeId, setActiveId] = useState<string>("");
  // only the top-level sections; h3 ids still exist for direct links
  const items = useMemo(() => headings.filter((h) => h.depth === 2), [headings]);

  useEffect(() => {
    if (items.length === 0) return;

    const nodes = items
      .map((h) => document.getElementById(h.id))
      .filter((el): el is HTMLElement => el !== null);
    if (nodes.length === 0) return;

    // The heading closest above the top of the reading area is the active one.
    const onScroll = () => {
      const line = 120;
      let current = nodes[0];
      for (const node of nodes) {
        if (node.getBoundingClientRect().top <= line) current = node;
        else break;
      }
      // once the page is scrolled to the bottom, the last section is the active one
      if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 2) {
        current = nodes[nodes.length - 1];
      }
      setActiveId(current.id);
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [items]);

  if (items.length < 2) return null;

  return (
    <nav
      aria-label="목차"
      className="hidden min-[1440px]:block fixed top-28 right-6 w-52 max-h-[calc(100vh-10rem)] overflow-y-auto"
    >
      <p className="font-mono text-[10px] tracking-widest uppercase text-(--text-muted) mb-3">
        On this page
      </p>
      <ul className="space-y-1 border-l border-(--border)">
        {items.map((h) => {
          const active = h.id === activeId;
          return (
            <li key={h.id}>
              <a
                href={`#${h.id}`}
                className={[
                  "block border-l -ml-px pl-3 py-1 text-xs leading-snug transition-colors duration-200",
                  active
                    ? "border-(--accent) text-(--accent)"
                    : "border-transparent text-(--text-muted) hover:text-(--text)",
                ].join(" ")}
              >
                {h.text}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
