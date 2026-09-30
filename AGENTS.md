# Repository Guidelines

## Project Structure & Module Organization

This repository is a Chinese-language learning notebook for LLM training and architecture. Keep material grouped by topic:

- `README.md` defines the repository scope and high-level study outline. It is also published verbatim as the website home page, so keep it reader-facing prose.
- `model_arch/` contains architecture notes; the current chapters are `Advanced LLMs 1 (Attentions).md` through `Advanced LLMs 5 (Mixture of Experts).md`.
- `infra/` documents training and inference systems. Add a topic directory when a subject needs multiple articles, as with `infra/slime/`.
- `pre-training/`, `mid-training/`, and `post-training/` are reserved for training-recipe notes.
- `report/` indexes vendor technical reports and stores locally referenced PDFs. It is intentionally **not** published (the PDFs are gitignored).
- `images/` holds diagrams embedded by Markdown articles.
- `tools/` holds `sync.py` (the Obsidian → Hugo converter) and `status.json` (the publishing manifest). See `tools/README.md`.
- `docs/` is the Hugo project root for the published site. See `docs/README.md`.

Place new content in the narrowest applicable directory and update the nearest `README.md` index when readers need a discoverable entry point.

## Development and Verification

Notes are the source of truth and are never modified by tooling. Publishing is driven by `tools/sync.py`, which reads `tools/status.json` and generates the Hugo content under `docs/content/` (gitignored — never edit or commit it).

```bash
python3 tools/sync.py --check                     # validate manifest/sources without writing
python3 tools/sync.py && hugo serve --source docs # regenerate + preview at localhost:1313/llm/
```

`--check` fails on manifest errors (missing source file, duplicate slug, unknown `pin`, internal link pointing at an unpublished note) and only warns about missing images and notes absent from `status.json`. Publishing is always explicit: a note that is not listed in `status.json` is not published.

Pushing to `main` re-runs the converter in CI (`.github/workflows/gh-pages.yaml`) and deploys to GitHub Pages, so committed notes go live without any local build.

Before committing, confirm that:

- the note you added is listed in `tools/status.json` with an explicit `date` (most notes have no git history to infer one from) and an ASCII `slug`;
- `python3 tools/sync.py --check` is clean;
- relative links resolve from the article that contains them;
- image embeds such as `![[sparse_attn.png]]` point at a file that exists in `images/`;
- new report links point to an existing PDF or a stable official URL.

Use `git status` and `git diff --check` from this directory to review intended changes and catch whitespace errors.

## Writing Style & Naming

Write clear Chinese prose; retain standard English technical terms, model names, and code identifiers where they improve precision. Use ATX headings (`##`), short paragraphs, and unordered lists for outlines. Prefer descriptive Markdown filenames in `PascalCase` for focused topics (for example, `Attentions.md`) and lowercase hyphenated names for supporting articles (for example, `slime-training.md`). Name images in lowercase kebab-case and use meaningful names such as `streaming-llm.png`.

## Commit & Pull Request Guidelines

Git history currently has only `initial commit :)`, so no established commit convention exists. Use concise imperative subjects scoped to the change, for example `docs: add MoE routing notes` or `report: index GLM-5 paper`. Keep each commit focused.

Pull requests should summarize the topic, list affected paths, link sources for factual claims, and include rendered screenshots when changing layout-heavy Markdown or diagrams. Note any added large PDFs or image assets explicitly.
