"use client";

import { useEffect, useState } from "react";

export default function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark">("dark");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const savedTheme = localStorage.getItem("theme") as "light" | "dark" | null;
    if (savedTheme) {
      setTheme(savedTheme);
    } else {
      const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      setTheme(prefersDark ? "dark" : "light");
    }
  }, []);

  const toggleTheme = () => {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    localStorage.setItem("theme", nextTheme);
    
    // HTML 클래스 교체
    if (nextTheme === "dark") {
      document.documentElement.classList.add("dark");
      document.documentElement.classList.remove("light");
    } else {
      document.documentElement.classList.add("light");
      document.documentElement.classList.remove("dark");
    }
  };

  // SSR Hydration mismatch 방지용 (마운트되기 전에는 자리를 지키는 플레이스홀더를 보여줌)
  if (!mounted) {
    return <div className="w-9 h-9" />;
  }

  return (
    <button
      onClick={toggleTheme}
      className="relative flex items-center justify-center w-9 h-9 rounded-md border border-[var(--border)] bg-[var(--bg-soft)] text-[var(--text-muted)] hover:text-[var(--accent)] hover:border-[var(--accent)] transition-all duration-300 cursor-pointer shadow-sm active:scale-95 group overflow-hidden"
      aria-label={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
    >
      <div className="relative w-5 h-5 flex items-center justify-center">
        {/* Sun Icon (Light Mode로 전환 가능할 때 또는 Light Mode 상태에서 보여줄 아이콘) */}
        {/* 여기서는 현재가 Light Mode일 때 해(Sun) 아이콘을 보여줍니다. */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
          className={`absolute w-5 h-5 transition-all duration-300 transform ${
            theme === "dark" ? "rotate-90 scale-0 opacity-0" : "rotate-0 scale-100 opacity-100"
          }`}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 3v2.25m0 13.5V21M9.75 12h4.5M9.75 12a2.25 2.25 0 0 1-4.5 0M9.75 12a2.25 2.25 0 0 0-4.5 0M14.25 12a2.25 2.25 0 0 1 4.5 0M14.25 12a2.25 2.25 0 0 0 4.5 0M9.75 12A2.25 2.25 0 0 1 12 9.75m0 0a2.25 2.25 0 0 1 2.25 2.25M12 9.75a2.25 2.25 0 0 0-2.25 2.25M12 9.75a2.25 2.25 0 0 1 2.25 2.25m-2.25 0a2.25 2.25 0 0 0 2.25 2.25M12 14.25a2.25 2.25 0 0 0 2.25-2.25m-2.25 2.25a2.25 2.25 0 0 1-2.25-2.25M12 14.25A2.25 2.25 0 0 1 9.75 12m2.25 2.25a2.25 2.25 0 0 0 2.25-2.25"
          />
        </svg>

        {/* Moon Icon (Dark Mode일 때 보여줄 아이콘) */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
          className={`absolute w-5 h-5 transition-all duration-300 transform ${
            theme === "dark" ? "rotate-0 scale-100 opacity-100" : "-rotate-90 scale-0 opacity-0"
          }`}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M21.752 15.002A9.72 9.72 0 0 1 18 15.75c-5.385 0-9.75-4.365-9.75-9.75 0-1.33.266-2.597.748-3.752A9.753 9.753 0 0 0 3 11.25C3 16.635 7.365 21 12.75 21a9.753 9.753 0 0 0 9.002-5.998Z"
          />
        </svg>
      </div>
    </button>
  );
}
