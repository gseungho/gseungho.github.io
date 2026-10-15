import type { Metadata } from "next";
import "./globals.css";
import Header from "@/components/Header";
import GoatCounter from "@/components/GoatCounter";

export const metadata: Metadata = {
  metadataBase: new URL("https://gseungho.github.io"),
  // pages set their own title; the template appends the site name
  title: { default: "Tensors & Notebooks", template: "%s | Tensors & Notebooks" },
  description: "AI 논문 리뷰와 공부 중인 내용을 정리한 노트, 직접 만든 프로젝트의 기록을 담는 블로그입니다.",
  openGraph: { siteName: "Tensors & Notebooks", locale: "ko_KR", type: "website" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" className="scroll-smooth">
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var saved = localStorage.getItem('theme');
                  var theme = 'dark';
                  if (saved === 'light' || saved === 'dark') {
                    theme = saved;
                  } else {
                    var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                    theme = prefersDark ? 'dark' : 'light';
                  }
                  document.documentElement.classList.add(theme);
                  document.documentElement.classList.remove(theme === 'dark' ? 'light' : 'dark');
                } catch (e) {}
              })();
            `
          }}
        />
      </head>
      <body className="min-h-screen flex flex-col bg-(--bg) text-(--text) antialiased transition-colors duration-300">
        <Header />
        <main className="flex-grow max-w-3xl w-full mx-auto px-6 py-10">{children}</main>
        <footer className="border-t border-(--border) py-10 px-6 mt-auto transition-colors duration-300">
          <div className="max-w-3xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-(--text-muted)">
            <div>
              <span>© {new Date().getFullYear()} gseungho. All rights reserved.</span>
            </div>
            <div className="flex gap-4">
              <a 
                href="https://github.com/gseungho" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="hover:text-(--accent) transition-colors"
              >
                GitHub
              </a>
              <span>·</span>
              <span className="font-mono text-[10px] tracking-wider uppercase text-(--text-muted)">VOLTAGENT SPEC</span>
            </div>
          </div>
        </footer>
        <GoatCounter />
      </body>
    </html>
  );
}
