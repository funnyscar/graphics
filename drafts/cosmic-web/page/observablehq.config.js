import {readFileSync} from "node:fs";
import texmath from "markdown-it-texmath";
import katex from "katex";

const katexCss = readFileSync(new URL("./katex-inline.css", import.meta.url), "utf-8");

export default {
  title: "The Cosmic Web",
  root: "src",
  pager: false,
  toc: true,
  footer: false,
  head:
    '<meta name="viewport" content="width=device-width, initial-scale=1">' +
    `<style>${katexCss}</style>`,
  markdownIt: (md) =>
    md.use(texmath, {
      engine: katex,
      delimiters: "dollars",
      katexOptions: {throwOnError: false},
    }),
};
