// Runs after `next build` (npm "postbuild") and fails the build when a post renders wrong.
// MDX breaks silently: a bad `$$` block or a `**` that does not close still builds, so the source is
// compared with out/posts/<slug>.html. Usage: node scripts/check-posts.mjs [slug ...]
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const postsDir = path.join(root, "content/posts");
const outDir = path.join(root, "out/posts");

// Greek letters, ℓ, √, sub/superscript digits and letters: math typed as text renders in the text
// font, not like the KaTeX in the body. Write it as $...$ instead.
const UNICODE_MATH = /[Ͱ-Ͽ⁰-₟²³¹ʰ-ʸ√ℓ]/u;

// Lines of the markdown body with front matter, fenced code and $$ blocks removed.
function bodyLines(md) {
  const lines = md.split(/\r?\n/);
  const out = [];
  let inFront = lines[0]?.trim() === "---";
  let inFence = false;
  let inDisplay = false;
  lines.forEach((line, i) => {
    if (i === 0 && inFront) return;
    if (inFront) {
      if (line.trim() === "---") inFront = false;
      return;
    }
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence;
      return;
    }
    if (inFence) return;
    if (line.trim() === "$$") {
      inDisplay = !inDisplay;
      return;
    }
    if (!inDisplay) out.push({ no: i + 1, line });
  });
  return out;
}

function checkSource(md) {
  const problems = [];
  const lines = bodyLines(md);
  for (const { no, line } of lines) {
    if (/^##\s+(다음 글|앞 글)\s*$/.test(line)) {
      problems.push(`line ${no}: "${line.trim()}" 절은 쓰지 않는다 (이전 글/다음 글 카드가 자동으로 붙는다)`);
    }
    const text = line
      .replace(/`[^`]*`/g, "")
      .replace(/\$\$[^$]*\$\$/g, "")
      .replace(/\$[^$]*\$/g, "")
      .replace(/\balt="[^"]*"/g, "")
      .replace(/\]\([^)]*\)/g, "]");
    const m = text.match(UNICODE_MATH);
    if (m) problems.push(`line ${no}: "${m[0]}" outside $...$: ${line.trim().slice(0, 80)}`);
  }
  const count = (re) => lines.filter(({ line }) => re.test(line)).length;
  const images = lines.reduce((n, { line }) => n + (line.match(/<img\b|<Fig\b/g) ?? []).length, 0);
  return { problems, h2: count(/^## /), h3: count(/^### /), images };
}

function checkRendered(html, src) {
  const problems = [];
  let body = html.slice(html.indexOf("prose max-w-none"));
  // the RSC payload at the end repeats the raw markdown; only the rendered part counts
  body = body.slice(0, body.indexOf("__next_f"));
  const h2 = (body.match(/<h2\b/g) ?? []).length - (body.match(/id="footnote-label"/g) ?? []).length;
  const h3 = (body.match(/<h3\b/g) ?? []).length;
  const images = (body.match(/<img\b/g) ?? []).length;
  if (h2 !== src.h2) problems.push(`## headings: ${src.h2} in the source, ${h2} rendered`);
  if (h3 !== src.h3) problems.push(`### headings: ${src.h3} in the source, ${h3} rendered`);
  if (images !== src.images) problems.push(`images: ${src.images} in the source, ${images} rendered`);

  const text = body
    .replace(/<pre[\s\S]*?<\/pre>/g, "")
    .replace(/<code[\s\S]*?<\/code>/g, "")
    .replace(/<annotation[\s\S]*?<\/annotation>/g, "")
    .replace(/<[^>]+>/g, " ");
  if (text.includes("**")) problems.push(`literal ** in the rendered text (bold did not close)`);
  if (text.includes("$")) problems.push(`literal $ in the rendered text (math did not render)`);
  if (/<figcaption[^>]*>\s*<p>/.test(body)) {
    problems.push(`a caption became a paragraph: keep <Fig …>caption</Fig> on one line`);
  }
  return problems;
}

const only = process.argv.slice(2);
const files = fs
  .readdirSync(postsDir)
  .filter((f) => f.endsWith(".md"))
  .filter((f) => only.length === 0 || only.includes(f.replace(/\.md$/, "")));

let failed = 0;
for (const file of files) {
  const slug = file.replace(/\.md$/, "");
  const src = checkSource(fs.readFileSync(path.join(postsDir, file), "utf8"));
  const htmlPath = path.join(outDir, `${slug}.html`);
  const problems = [...src.problems];
  if (fs.existsSync(htmlPath)) problems.push(...checkRendered(fs.readFileSync(htmlPath, "utf8"), src));
  else problems.push(`no ${path.relative(root, htmlPath)}: run next build first`);
  if (problems.length > 0) {
    failed += 1;
    console.error(`✗ ${slug}`);
    for (const p of problems) console.error(`    ${p}`);
  }
}
console.log(`check-posts: ${files.length - failed}/${files.length} posts ok`);
if (failed > 0) process.exit(1);
