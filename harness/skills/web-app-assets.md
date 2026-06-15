---
applies_to: [html, css, react, vue, angular]
---

## Web App — Asset Reference Contract

### When this skill applies
Any workspace that ships HTML — pure static sites, React/Vue/Angular apps, or anything with `index.html` at the root. Targets the class of bug where the LLM emits `<link href="src/styles.css">` but never creates the CSS file, or splits one logical asset across two non-matching paths in different artifacts.

### The rule
Every local reference in generated HTML/CSS/JS MUST resolve to a file that the same plan will write to disk:

- HTML: `<link href>`, `<script src>`, `<img src>`, `<source src>`, `<video src>`, `<audio src>`, `<a href>`, `<iframe src>`, `<embed src>`, `<object data>`
- CSS: `url(...)`, `@import "..."`
- JS/TS: relative-path `import "./foo.js"` and dynamic `import("./foo.js")`

If the file inventory in `SPEC_ARCHITECTURE.md` lists `style.css` at the project root, then `index.html` must write `<link href="style.css">` — NOT `<link href="src/styles.css">`, `<link href="styles/main.css">`, or any other path-renamed variant. The architecture inventory is authoritative.

### What gets skipped (don't worry about these)
External URLs (`https://`, `//cdn...`), `mailto:`, `tel:`, `data:`, anchor-only refs (`#section`), and bare-module JS imports (`import "react"`) are not validated — bundlers and the browser resolve those.

### Why this matters
The harness runs a static asset-reference scanner inside lintgate AND inside the Makefile `build` target. Both will fail loudly if a generated reference doesn't resolve — and the LLM repair loop has to spend tokens reconciling the mismatch. Writing the right reference the first time avoids the repair round-trip entirely.

### Common patches the LLM gets wrong
- **Pluralization drift**: architecture says `style.css`, HTML says `styles.css`. Mass noun confusion. Pick one form and use it in both places.
- **Directory prefix drift**: architecture lists assets at workspace root, then `index.html` references them as if they live under `src/` or `assets/` because that's the convention for JS modules. Static assets sit wherever the architecture inventory says they sit.
- **Forgetting to create the asset at all**: the LLM writes the `<link>` and never includes a `CREATE_FILE` for the CSS. If you reference it, you must create it in the same set of patches.
- **Module-vs-script confusion**: writing `<script src="src/main.js">` (no `type="module"`) while `main.js` uses ES `import`. If the JS uses `import`, the HTML tag must be `<script type="module" src="...">`.
