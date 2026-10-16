// Display names for categories. Front matter and URLs keep the Korean names.
const LABELS: Record<string, string> = {
  공부: "Study",
  프로젝트: "Projects",
  논문리뷰: "Paper Reviews",
};

export function categoryLabel(category: string): string {
  return LABELS[category.normalize("NFC")] ?? category;
}
