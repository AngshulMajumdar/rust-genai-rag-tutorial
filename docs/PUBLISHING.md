# Publishing the tutorial

The `docs/` directory is the complete static tutorial site.

## GitHub Pages

In the repository settings:

1. Open **Pages**.
2. Choose **Deploy from a branch**.
3. Select the `main` branch.
4. Select the `/docs` folder.
5. Save.

The site needs no custom workflow and no site generator. `docs/index.html` is the entry page and `docs/styles.css` is the only presentation dependency.

## Medium

The article body uses ordinary semantic HTML: headings, paragraphs, ordered/unordered lists, blockquotes, figures and code blocks. It does not depend on JavaScript or GitHub-only widgets.

Two versions are committed:

- `index.html` — the styled GitHub Pages version.
- `medium.html` — the same complete tutorial with the site chrome and stylesheet removed.

After Pages is live, either import the public article URL into Medium or paste the content from `medium.html` into the Medium editor. Check code-block formatting and image captions after import because Medium controls its own presentation.

## Source of truth

Edit `tutorial_body.html`, then run:

```bash
python tools/site/build_site.py
```

That regenerates both public versions from the same article body so they cannot drift apart.
