# Build Multilingual RAG in Rust

**A complete, runnable Rust tutorial for building a multilingual retrieval-augmented generation system over a real Indic-language HR corpus.**

Python is excellent for getting a RAG experiment running. This repository asks the next question: **what does the same system look like when indexing, retrieval, generation orchestration, evaluation, and the command-line application are written as one Rust project?**

The tutorial uses the included **Suryodaya Indic HR Multi-hop QA corpus** instead of a toy English dataset:

- 50 long HR-policy documents
- 675,913 words
- 8 Indian languages: Bengali, Hindi, Kannada, Marathi, Odia, Punjabi, Tamil and Telugu
- 6,315 structured gold facts
- 1,000 multi-hop questions
- 100 explicitly cross-language questions
- long-document, cross-section and cross-document reasoning

The complete tutorial is in [`docs/`](docs/). That folder is ready to be used directly as the source for **GitHub Pages**, and the tutorial is intentionally written with ordinary headings, paragraphs, images and code blocks so the same article can be imported into or pasted into **Medium** without relying on GitHub-only components.

## Why Rust for RAG?

Rust does not make a model smarter. It makes the **system around the model** easier to turn into a real application.

- **One compiled program.** Indexing, search, chat and evaluation live in the same crate.
- **No Python runtime in the final CLI.** The application is a native executable; Ollama remains the local model server.
- **Explicit Unicode handling.** Indic digits, matras, virama and script-aware tokenization are implemented directly rather than delegated to a regex assumption that may be wrong for the text.
- **Predictable memory ownership.** The chunk store, posting lists and dense vectors have explicit lifetimes and representations.
- **Safe concurrency is available when you need it.** The current tutorial keeps the core logic simple, but Rust lets the implementation scale without changing languages later.
- **Benchmark code and application code can be the same code.** The exact modules used by the CLI are also exposed as a library and notebook-friendly API.

If your objective is only to sketch an idea, Python will usually be faster to write. If your objective is to understand a RAG system down to the index and then ship the same implementation as a binary, Rust is a very good fit.

## Start here

Read the tutorial:

**[`docs/index.html`](docs/index.html)**

For GitHub Pages, set the publishing source to the repository's `main` branch and the `/docs` folder. No custom workflow is required.

## Quick run

The Rust application uses a local Ollama server for embeddings and generation.

```bash
ollama pull bge-m3
ollama pull gemma3:4b

cd rust
cargo build --release
cargo run --release -- index
cargo run --release -- embed --model bge-m3
cargo run --release -- search "मातृत्व अवकाश" --mode hybrid --k 5
cargo run --release -- chat --model gemma3:4b --mode hybrid --strategy agentic --verbose
```

From `rust/`, the CLI defaults to `--repo ..`, so it automatically sees the included `data/corpus` and `data/testset` folders.

## What the tutorial builds

The implementation is deliberately decomposed into small Rust modules:

```text
rust/src/
  text.rs       Unicode normalisation, Indic-aware tokens, character n-grams
  chunk.rs      heading-aware corpus chunking
  index.rs      BM25 posting lists, dense-vector storage
  retrieve.rs   BM25, dense and hybrid RRF retrieval
  ollama.rs     local embedding and chat client
  rag.rs        simple RAG and agentic multi-hop RAG
  eval.rs       gold-evidence mapping and evaluation
  session.rs    notebook-friendly owned API
  data.rs       dataset discovery and zip/folder handling
  main.rs       command-line interface
```

The tutorial explains these modules in that order, using code from the repository rather than pseudocode.

## The multilingual Indic HR corpus

The data is under `data/corpus/<Language>/`. Each policy has a Markdown document and a structured fact ledger.

```text
data/corpus/
  Bengali/   4 documents
  Hindi/     9 documents
  Kannada/   4 documents
  Marathi/   6 documents
  Odia/      5 documents
  Punjabi/   5 documents
  Tamil/    10 documents
  Telugu/    7 documents
```

The documents were authored directly in their target language rather than being eight translations of the same policy set. The test set therefore contains both ordinary multilingual retrieval problems and genuine cross-language multi-hop questions.

Example cross-language question (`XL-REC-001`):

> সূর্যোদয় ইন্টিগ্রেটেড সলিউশনস-এর বাংলা নিয়োগ নির্দেশিকা অনুযায়ী সম্পূর্ণ পূর্বতন কর্মজীবন যাচাই প্রক্রিয়া সম্পন্ন করার জন্য বরাদ্দ সময়সীমা কত, এবং তেলুগু নির্দেশিকায় উল্লিখিত নেপথ্য ধৃবীকরণ (నేపథ్య ధృవీకరణ)-র প্রমিত সময়সীমার (సాధారణ కాలపరిమితి) তুলনায় তা কত কার্যদিবস বেশি?

It requires one fact from a Bengali recruitment policy and one from a Telugu recruitment policy, followed by a comparison.

## Repository layout

```text
.
├── README.md
├── CITATION.cff
├── data/
│   ├── corpus/
│   └── testset/
├── docs/                 complete GitHub Pages + Medium tutorial
│   ├── index.html
│   ├── medium.html
│   ├── tutorial_body.html
│   ├── styles.css
│   ├── PUBLISHING.md
│   └── assets/
├── rust/                 primary implementation taught by the tutorial
├── rag/                  Python reference retrieval implementation
├── eval/                 reference evaluation scripts
├── notebooks/            Rust/evcxr notebook interface
├── prompts/              answering, planning and judging prompts
├── results/              stored reference results and search logs
├── notes/                corpus-construction notes
└── tools/                reproducibility and site utilities
```

The Python code remains as a reference implementation for parity checks. The tutorial itself is Rust-first.

## Reproduce the retrieval baseline

From `rust/`:

```bash
cargo run --release -- index
cargo run --release -- retrieval --mode bm25 --k 10
```

The stored reference baseline reports global single-shot retrieval at top 10 of approximately **80.8% supporting-fact recall** and **63.8% all-facts recall** over the 1,000 questions. Cross-language questions are intentionally difficult for a single raw query; that is the motivation for the later agentic section of the tutorial.

## Run the full RAG evaluator

```bash
cargo run --release -- embed --model bge-m3
cargo run --release -- eval \
  --model gemma3:4b \
  --strategy agentic \
  --mode hybrid \
  --every 10 \
  --out runs/gemma3-4b-agentic-hybrid.jsonl
```

Remove `--every 10` for all 1,000 questions. Prediction files are append-only and the evaluator resumes completed IDs, so an interrupted run does not need to start over.

## Repository hygiene

This repository is intentionally suitable for direct upload to GitHub:

- no dotfiles
- no dot-directories
- no generated `target/`, `store/`, `runs/` or `hr_rag_work/` directories are committed
- the Pages site lives entirely in `docs/`

Because there is deliberately no `.gitignore`, do **not** upload generated Cargo or run directories after executing the code locally.

## Citation

Citation metadata is provided in `CITATION.cff`.
