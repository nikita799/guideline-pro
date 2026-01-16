import MarkdownIt from "markdown-it";

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true
}).enable("table");

export function renderMarkdown(text: string): string {
  if (!text) {
    return "";
  }
  return md.render(text);
}
