"""Build the static GitHub Pages tutorial and a minimal Medium-friendly mirror.

The article body lives in docs/tutorial_body.html.  This script only wraps that
semantic HTML; it has no third-party dependencies.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
BODY = (DOCS / "tutorial_body.html").read_text(encoding="utf-8")

TITLE = "Build Multilingual RAG in Rust: an Indic-language tutorial"
DESC = (
    "Build a multilingual RAG system in Rust over a 50-document Indic HR corpus: "
    "Unicode tokenization, BM25, dense retrieval, RRF, Ollama, agentic search and evaluation."
)

index = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{TITLE}</title>
<meta name=\"description\" content=\"{DESC}\">
<link rel=\"stylesheet\" href=\"styles.css\">
</head>
<body>
<header class=\"sitebar\"><div class=\"sitebar-inner\">
<div class=\"brand\">Rust × Indic RAG</div>
<nav><a href=\"#why-rust\">Why Rust</a><a href=\"#corpus\">Corpus</a><a href=\"#bm25\">Retrieval</a><a href=\"#agentic\">Agentic RAG</a><a href=\"#evaluation\">Evaluation</a></nav>
</div></header>
<main>
{BODY}
</main>
<footer>Complete Rust tutorial over the included Suryodaya multilingual Indic HR corpus.</footer>
</body>
</html>
"""

medium = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{TITLE}</title>
<meta name=\"description\" content=\"{DESC}\">
</head>
<body>
{BODY}
</body>
</html>
"""

(DOCS / "index.html").write_text(index, encoding="utf-8")
(DOCS / "medium.html").write_text(medium, encoding="utf-8")
print("wrote docs/index.html and docs/medium.html")
