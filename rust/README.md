# hr-rag

The Rust implementation taught by the repository tutorial.

It builds a multilingual RAG system over the included Suryodaya Indic HR corpus: Unicode-aware preprocessing, heading-aware chunking, BM25 and character n-gram indexes, optional dense embeddings through Ollama, reciprocal-rank hybrid retrieval, local generation, agentic multi-hop search, evaluation and a CLI.

The full walk-through is in [`../docs/index.html`](../docs/index.html).

## Modules

| module | responsibility |
|---|---|
| `text.rs` | Indic-digit normalization, Unicode word tokens, char 4-grams, sentence splitting and script detection |
| `chunk.rs` | section-aware chunking; 220 words is a merge target, while an individual source paragraph is kept intact even if longer |
| `index.rs` | word and char-gram BM25 posting lists plus optional dense-vector storage |
| `retrieve.rs` | lexical, dense and hybrid retrieval with reciprocal rank fusion |
| `ollama.rs` | minimal client for local `/api/embed`, `/api/chat`, model listing and pulling |
| `rag.rs` | single-shot and agentic plan-retrieve-answer strategies |
| `eval.rs` | test-set loading, gold-evidence mapping, numeric/evidence metrics and resumable scoring |
| `session.rs` | owned API used from the Rust Jupyter kernel |
| `data.rs` | locate the included dataset from a repository, extracted folder or zip |
| `main.rs` | command-line application |

## Prerequisites

Install Rust and Ollama, start Ollama, then pull an embedding model and a chat model:

```bash
ollama pull bge-m3
ollama pull gemma3:4b
```

## Build

Run commands from this `rust` folder:

```bash
cargo build --release
```

The CLI defaults to `--repo ..`, so it reads the corpus and test set from the parent repository.

## Build the lexical index

```bash
cargo run --release -- index
```

This writes `store/index.bin`. The corpus currently produces 5,507 chunks. Chunk construction is heading-aware: paragraphs sharing a heading are merged toward a 220-word target, but oversized source paragraphs are never split merely to satisfy that target.

## Add dense embeddings

```bash
cargo run --release -- embed --model bge-m3
```

The vectors are L2-normalized before being stored. Dense retrieval therefore uses a dot product equivalent to cosine similarity for stored rows.

## Search

BM25 only:

```bash
cargo run --release -- search "மகப்பேறு விடுப்பு" --mode bm25 --k 5
```

Hybrid lexical + dense:

```bash
cargo run --release -- search "মাতৃত্বকালীন ছুটি" --mode hybrid --k 5
```

Restrict a search to a language or document when you already know the target source:

```bash
cargo run --release -- search "notice period" --lang Tamil --mode dense
```

## Chat

```bash
cargo run --release -- chat \
  --model gemma3:4b \
  --mode hybrid \
  --strategy agentic \
  --verbose
```

The assistant answers in the detected language of the question, returns cited chunk IDs and can re-plan searches when evidence is missing.

## Retrieval evaluation

No generator is needed for BM25 retrieval evaluation:

```bash
cargo run --release -- retrieval --mode bm25 --k 10
```

For dense or hybrid evaluation, build embeddings first.

## End-to-end evaluation

A quick 100-question slice:

```bash
cargo run --release -- eval \
  --model gemma3:4b \
  --strategy agentic \
  --mode hybrid \
  --every 10 \
  --out runs/gemma3-4b-agentic-hybrid.jsonl
```

Full 1,000-question run:

```bash
cargo run --release -- eval \
  --model gemma3:4b \
  --strategy agentic \
  --mode hybrid \
  --out runs/gemma3-4b-agentic-hybrid.jsonl
```

The output file is append-only. Existing question IDs are skipped on restart.

## Score an existing run

```bash
cargo run --release -- score \
  runs/gemma3-4b-agentic-hybrid.jsonl \
  --report runs/gemma3-4b-agentic-hybrid.md
```

An external judge file can be added with `--judge`, or judge input can be exported with `--export-judge`.

## Rust notebook API

`../notebooks/hr_rag_chatbot.ipynb` targets the Rust `evcxr` kernel. The same functionality is available as a library:

```rust
use hr_rag::*;

let d = hr_rag::data::locate("../")?;
let mut s = Session::open_dirs(
    &d.corpus,
    d.testset.as_deref(),
    &d.work,
    "http://localhost:11434",
)?;

s.ensure_models(&["bge-m3", "gemma3:4b"])?;
s.embed("bge-m3")?;
let answer = s.ask("नोटिस अवधि कितनी है?", &Config::new("gemma3:4b"))?;
println!("{answer}");
```

## Generated directories

Local runs can create `target/`, `store/`, `runs/` and `hr_rag_work/`. They are outputs, not source files, and are intentionally absent from this repository.
