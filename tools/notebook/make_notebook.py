"""Builds notebooks/hr_rag_chatbot.ipynb (Rust / evcxr kernel) from the library sources in rust/src, so the
notebook always carries the same code as the crate. Run from the repository root: python3 tools/notebook/make_notebook.py"""
import json, re, sys, os
src = sys.argv[1] if len(sys.argv) > 1 else "rust/src"; out = sys.argv[2] if len(sys.argv) > 2 else "notebooks/hr_rag_chatbot.ipynb"
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s, hidden=False):
    m = {"jupyter": {"source_hidden": True}} if hidden else {}
    cells.append({"cell_type": "code", "metadata": m, "execution_count": None, "outputs": [], "source": s.strip("\n")})

def module(name):
    s = open(os.path.join(src, name + ".rs")).read()
    s = "\n".join(l for l in s.splitlines() if not l.startswith("//!"))
    i = s.find("#[cfg(test)]")
    if i >= 0: s = s[:i]
    s = s.replace(", clap::ValueEnum", "")
    assert "clap" not in s, name
    body = "\n".join(("    " + l) if l.strip() else "" for l in s.strip("\n").splitlines())
    return f"pub mod {name} {{\n{body}\n}}"

md("""
# Suryodaya HR chatbot (Rust): RAG baseline

A self-contained notebook for the **Rust (evcxr)** kernel. It asks where your data is, builds a multilingual search
index over the HR-policy documents, runs a retrieval-augmented chatbot on a local Ollama model, and measures its
accuracy on the multi-hop test set.

**Needs:** the Ollama app running on this computer. Nothing else: the notebook downloads the models it needs
(about 5 GB the first time) and compiles its own code (a few minutes the first time, seconds afterwards).

**Run:** *Run All*. The *Data* cell opens the standard Windows file window (on top of the browser, starting in
Downloads): choose the data zip, or any file inside the folder you extracted it to.
""")
code("""
:opt 2
:dep anyhow = "1"
:dep serde = { version = "1", features = ["derive"] }
:dep serde_json = "1"
:dep bincode = "1.3"
:dep ureq = { version = "2", default-features = false, features = ["json"] }
:dep zip = { version = "2", default-features = false, features = ["deflate"] }
:dep evcxr_input = "1"
""")
md("""
## Settings
""")
code("""
const OLLAMA: &str = "http://localhost:11434";   // Ollama server
const EMBED_MODEL: &str = "bge-m3";               // multilingual embeddings (all 8 languages)
const CHAT_MODEL: &str = "gemma3:4b";             // main chatbot model
const SMALL_MODEL: &str = "gemma3:1b";            // small model (the one the fine-tuning step trains)
const EVERY: usize = 10;                          // accuracy runs use every 10th question (100 of 1,000)
""")
md("""
## Library code

Tokenizer, chunker, BM25 + dense index, Ollama client, retrieval, the chatbot, scoring, and data loading. The cells are
collapsed; click a cell's left bar to read the code.
""")
for m in ["text", "chunk", "index", "ollama", "retrieve", "rag", "eval", "session", "data"]:
    code(module(m), hidden=True)
code("""
use anyhow::Result;
use data::DataPaths;
use rag::Config;
use retrieve::Mode;
use session::{compare, details, Report, Session};
println!("library ready");
""")
md("""
## 1. Data

A file window opens: choose the data **zip** (it is extracted next to itself, once), or **any file inside** the extracted
data folder. If the test set is not next to the documents, a second window asks for it (Cancel = chat only). The index
and all results are written to a `hr_rag_work` folder beside the documents.
""")
code("""
let paths: DataPaths = data::choose(evcxr_input::get_string)?;
""")
code("""
let mut s: Session = Session::open_dirs(&paths.corpus, paths.testset.as_deref(), &paths.work, OLLAMA)?;
""")
md("""
## 2. Retrieval with keywords only (no model needed)

Single-shot BM25 (word + character 4-gram) with the raw question as the query. *all facts retrieved* = every gold
supporting fact is in the top 10; *fact recall* = share of gold facts in the top 10.
""")
code("""
s.retrieval(Mode::Bm25, 10)?
""")
md("""
## 3. Models

Checks Ollama and downloads whatever is missing (bge-m3 about 1.2 GB, gemma3:4b about 3.3 GB, gemma3:1b about 0.8 GB).
""")
code("""
s.ensure_models(&[EMBED_MODEL, CHAT_MODEL, SMALL_MODEL])?;
""")
md("""
## 4. Dense embeddings and hybrid retrieval

Embeds every chunk once (a few minutes on the GPU) and saves the vectors with the index; re-running skips it.
""")
code("""
s.embed(EMBED_MODEL)?;
""")
code("""
s.retrieval(Mode::Hybrid, 10)?
""")
md("""
## 5. Chatbot

`Config::new(model)`: agentic strategy (plans one search per needed fact, in the language of the document that holds
it), hybrid retrieval, up to 12 passages. `.simple()` = one search and one answer. Three test questions first (the
reference answer is shown under the bot's), then a chat box: type questions in any of the 8 languages, empty line to stop.
""")
code("""
let bot: Config = Config::new(CHAT_MODEL);
""")
code("""
s.ask_item("HI-029", &bot)?
""")
code("""
s.ask_item("PA-069", &bot)?
""")
code("""
s.ask_item("XL-REC-001", &bot)?
""")
code("""
loop {
    let q: String = evcxr_input::get_string("Your question (empty = stop): ");
    if q.trim().is_empty() { break; }
    println!("{}\\n", s.ask(q.trim(), &bot)?);
    std::io::Write::flush(&mut std::io::stdout())?;
    std::thread::sleep(std::time::Duration::from_millis(400)); // let the answer reach the page before the next box
};
""")
md("""
## 6. Accuracy

Every `EVERY`-th question (100 questions, same language and hop-type mix as the full set). Each run saves its answers
in `hr_rag_work/runs/<run>.jsonl` and resumes where it stopped if interrupted: just run the cell again. *numeric match* =
every number in the reference short answer appears in the bot's answer.
""")
code("""
let grid: Vec<Config> = vec![
    Config::new(CHAT_MODEL).simple().mode(Mode::Bm25),
    Config::new(CHAT_MODEL).simple(),
    Config::new(CHAT_MODEL),
    Config::new(SMALL_MODEL),
];
let mut reports: Vec<Report> = Vec::new();
""")
code("""
reports.clear();
for c in &grid {
    reports.push(s.eval(c, EVERY)?);
};
""")
code("""
compare(&reports.iter().collect::<Vec<_>>())
""")
md("Per-language and per-hop-type detail of each run (also saved as `hr_rag_work/runs/<run>.md`):")
code("""
details(&reports.iter().collect::<Vec<_>>())
""")
md("""
## 7. Full test set (optional, several hours)

Set `FULL` to `true` to run all 1,000 questions with the agentic bot on both models. Resumable.
""")
code("""
const FULL: bool = false;
if FULL {
    let full: Vec<Report> = vec![
        s.eval_named(&Config::new(CHAT_MODEL), 1, "full_agentic_main")?,
        s.eval_named(&Config::new(SMALL_MODEL), 1, "full_agentic_small")?,
    ];
    compare(&full.iter().collect::<Vec<_>>()).evcxr_display();
};
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Rust", "language": "rust", "name": "rust"},
      "language_info": {"codemirror_mode": "rust", "file_extension": ".rs", "mimetype": "text/rust", "name": "Rust", "pygment_lexer": "rust", "version": ""}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]): c["id"] = f"c{i:02d}"
json.dump(nb, open(out, "w"), ensure_ascii=False, indent=1)
print(len(cells), "cells")
