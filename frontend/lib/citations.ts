export type Citation = {
  index: number;
  display_index: number;
  text: string;
  label: string;
};

export function linkCitationsHtml(
  html: string,
  citations: Citation[],
  scope: string
): string {
  if (!html || !citations.length || !scope) {
    return html;
  }
  const indices = new Set(citations.map((citation) => citation.display_index));
  return html.replace(/\[(.*?)\]/g, (match, content) => {
    if (!/^[0-9,\s-]+$/.test(content)) {
      return match;
    }
    const linked = content.replace(/\d+/g, (value) => {
      const idx = Number.parseInt(value, 10);
      if (!indices.has(idx)) {
        return value;
      }
      return `<a href="#citation-${scope}-${idx}">${value}</a>`;
    });
    return `[${linked}]`;
  });
}
