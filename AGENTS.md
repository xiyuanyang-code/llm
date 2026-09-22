# Repository Guidelines

## Project Structure & Module Organization

This repository is a Chinese-language learning notebook for LLM training and architecture. Keep material grouped by topic:

- `README.md` defines the repository scope and high-level study outline.
- `model_arch/` contains architecture notes; `Attentions.md` is the current attention-mechanism chapter.
- `infra/` documents training and inference systems. Add a topic directory when a subject needs multiple articles, as with `infra/slime/`.
- `pre-training/`, `mid-training/`, and `post-training/` are reserved for training-recipe notes.
- `report/` indexes vendor technical reports and stores locally referenced PDFs.
- `images/` holds diagrams embedded by Markdown articles.

Place new content in the narrowest applicable directory and update the nearest `README.md` index when readers need a discoverable entry point.

## Development and Verification

There is no build tool, package manifest, or automated test command. Validate documentation changes locally by opening the edited Markdown in a renderer such as Obsidian or GitHub preview. Before committing, confirm that:

- relative links resolve from the article that contains them;
- image paths such as `../images/sparse_attn.png` render correctly;
- new report links point to an existing PDF or a stable official URL.

Use `git status` and `git diff --check` from this directory to review intended changes and catch whitespace errors.

## Writing Style & Naming

Write clear Chinese prose; retain standard English technical terms, model names, and code identifiers where they improve precision. Use ATX headings (`##`), short paragraphs, and unordered lists for outlines. Prefer descriptive Markdown filenames in `PascalCase` for focused topics (for example, `Attentions.md`) and lowercase hyphenated names for supporting articles (for example, `slime-training.md`). Name images in lowercase kebab-case and use meaningful names such as `streaming-llm.png`.

## Commit & Pull Request Guidelines

Git history currently has only `initial commit :)`, so no established commit convention exists. Use concise imperative subjects scoped to the change, for example `docs: add MoE routing notes` or `report: index GLM-5 paper`. Keep each commit focused.

Pull requests should summarize the topic, list affected paths, link sources for factual claims, and include rendered screenshots when changing layout-heavy Markdown or diagrams. Note any added large PDFs or image assets explicitly.
