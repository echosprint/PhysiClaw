# PhysiClaw documentation

These are the source docs for **[docs.physiclaw.ai](https://docs.physiclaw.ai)**. They live
here, next to the code, so docs ship in the same pull request as the change they describe.

On every push to `main` that touches `docs/`, a GitHub Action mirrors this directory into
the renderer ([physiclaw/docs-site](https://github.com/physiclaw/docs-site)), which builds
and deploys the static site. You don't build anything here; just edit Markdown.

## Authoring

- **One file per page**, `.md` for prose or `.mdx` when you use a component.
- **Bilingual by suffix:** `intro.mdx` is English (the source of truth); `intro.zh.mdx` is
  its 简体中文 translation. A page **must** have the English file; the `.zh` sibling is
  optional and falls back to English when missing.
- **Frontmatter is just `title` + `description`:**
  ```mdx
  ---
  title: Quickstart
  description: Get the server talking to your agent in ten minutes.
  ---
  ```
- **No `import` statements.** Write callouts as directives (`:::note`, `:::tip[Title]`,
  `:::caution`, `:::danger`) and components ([Card, CardGrid, Steps, Tabs, FileTree,
  LinkCard, Badge, …](https://starlight.astro.build/components/)) as bare tags. The
  renderer injects imports during the build.

## Navigation: `docs.json`

The sidebar is defined in [`docs.json`](./docs.json) (validated against
[`docs.schema.json`](./docs.schema.json)). Each section lists its pages **in order**, by
slug (the path under `docs/`, no extension):

```json
{
  "label": "Start",
  "translations": { "zh-CN": "开始" },
  "items": ["start/introduction", "start/installation", "start/quickstart"]
}
```

To **add a page**: create `docs/<section>/<name>.mdx` (and optionally `.zh.mdx`), then add
its slug to the right section's `items` in `docs.json`. To **reorder**, move the slug. The
build fails fast on an unknown/duplicate slug or a page missing its English source, and
warns about any page not listed in a section.

> Note: the repo ignores `*.png` / `*.jpg`. If a doc needs an image, add an exception in
> `.gitignore` (e.g. `!docs/**/*.png`) so it isn't dropped.
