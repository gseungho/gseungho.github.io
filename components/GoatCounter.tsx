"use client";

import Script from "next/script";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

declare global {
  interface Window {
    goatcounter?: { count?: (vars?: { path?: string }) => void };
  }
}

// "/posts/a/", "/posts/a.html" and "/posts/a" are the same page; "/index.html" is "/".
function normalize(path: string): string {
  let p = path.replace(/\/index\.html$/, "/").replace(/\.html$/, "");
  if (p.length > 1) p = p.replace(/\/+$/, "");
  return p || "/";
}

// GoatCounter page views. Client-side navigation does not reload the page, so the
// automatic count on load is turned off and every pathname change is counted here.
// count.js skips localhost on its own.
export default function GoatCounter() {
  const pathname = usePathname();

  useEffect(() => {
    // before count.js has loaded this is a no-op; onLoad below counts the first view
    window.goatcounter?.count?.({ path: normalize(pathname) });
  }, [pathname]);

  return (
    <Script
      src="https://gc.zgo.at/count.js"
      data-goatcounter="https://gseungho.goatcounter.com/count"
      data-goatcounter-settings='{"no_onload": true}'
      strategy="afterInteractive"
      onLoad={() => window.goatcounter?.count?.({ path: normalize(window.location.pathname) })}
    />
  );
}
