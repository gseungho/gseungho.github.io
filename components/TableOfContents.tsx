"use client";

import { useEffect, useState } from "react";
import type { Heading } from "@/lib/posts";

export default function TableOfContents({ headings }: { headings: Heading[] }) {
  const [activeId, setActiveId] = useState<string>("");

  useEffect(() => {
    if (headings.length === 0) return;

    const nodes = headings
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
  }, [headings]);

  if (headings.length < 2) return null;

  return (
    <nav
      aria-label="목차"
      className="hidden xl:block fixed top-28 left-[calc(50%+24rem+2rem)] w-56 max-h-[calc(100vh-10rem)] overflow-y-auto pr-2"
    >
      <p className="font-mono text-[10px] tracking-widest uppercase text-(--text-muted) mb-3">
        On this page
      </p>
      <ul className="space-y-1 border-l border-(--border)">
        {headings.map((h) => {
          const active = h.id === activeId;
          return (
            <li key={h.id}>
              <a
                href={`#${h.id}`}
                className={[
                  "block border-l -ml-px py-1 text-xs leading-snug transition-colors duration-200",
                  h.depth === 3 ? "pl-6" : "pl-3",
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
