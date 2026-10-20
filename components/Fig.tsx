import type { ReactNode } from "react";

// max width of the image; without it the image fills the column
const SIZES = { xs: "max-w-xs", sm: "max-w-sm", md: "max-w-md", xl: "max-w-xl" } as const;

// A figure in a post body: <Fig src="/images/posts/<slug>/a.svg" alt="…">그림 1. …</Fig>
// The caption is the children, so $...$ math, links and bold work in it. Keep it on the same line as
// the tags: a caption on its own line becomes a <p> inside <figcaption>.
export default function Fig({
  src,
  alt,
  size,
  children,
}: {
  src: string;
  alt: string;
  size?: keyof typeof SIZES;
  children?: ReactNode;
}) {
  return (
    <figure className="my-6 text-center">
      {/* static export: images are served as they are, next/image has nothing to optimize */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className={size ? `rounded-lg mx-auto ${SIZES[size]}` : "rounded-lg mx-auto"} />
      {children && <figcaption className="mt-2 text-sm text-gray-500">{children}</figcaption>}
    </figure>
  );
}
