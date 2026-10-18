//! hr-rag: multilingual RAG chatbot + evaluator for the Suryodaya Indic HR MH-QA benchmark.
use hr_rag::{chunk, eval, index, ollama, rag, retrieve};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "hr-rag", version, about = "Multilingual HR-policy RAG chatbot and evaluator (Ollama backend)")]
struct Cli {
    /// Repository root (contains data/corpus and data/testset)
    #[arg(long, global = true, default_value = "..")]
    repo: PathBuf,
    /// Index file
    #[arg(long, global = true, default_value = "store/index.bin")]
    store: PathBuf,
    /// Ollama base URL
    #[arg(long, global = true, default_value = "http://localhost:11434")]
    ollama: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct GenArgs {
    /// Ollama chat model used for planning and answering
    #[arg(long, default_value = "gemma3:4b")]
    model: String,
    #[arg(long, value_enum, default_value = "hybrid")]
    mode: retrieve::Mode,
    #[arg(long, value_enum, default_value = "agentic")]
    strategy: rag::Strategy,
    /// chunks retrieved per query
    #[arg(long, default_value_t = 6)]
    k: usize,
    /// max chunks in the answer prompt
    #[arg(long, default_value_t = 12)]
    max_ctx: usize,
    /// agentic re-planning rounds
    #[arg(long, default_value_t = 2)]
    rounds: usize,
    /// model context window (tokens); 12 Indic-script chunks need ~4-7k tokens
    #[arg(long, default_value_t = 12288)]
    num_ctx: usize,
}

impl GenArgs {
    fn cfg(&self) -> rag::Config {
        rag::Config {
            model: self.model.clone(),
            mode: self.mode,
            strategy: self.strategy,
            k: self.k,
            max_ctx: self.max_ctx,
            rounds: self.rounds,
            num_ctx: self.num_ctx,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Chunk the corpus and build the BM25 indexes
    Index,
    /// Add dense embeddings (Ollama embedding model, e.g. bge-m3) to the index
    Embed {
        #[arg(long, default_value = "bge-m3")]
        model: String,
        #[arg(long, default_value_t = 32)]
        batch: usize,
    },
    /// Search the corpus
    Search {
        query: String,
        #[arg(long)]
        lang: Option<String>,
        #[arg(long)]
        doc: Option<String>,
        #[arg(long, default_value_t = 5)]
        k: usize,
        #[arg(long, value_enum, default_value = "bm25")]
        mode: retrieve::Mode,
    },
    /// Interactive chatbot
    Chat {
        #[command(flatten)]
        g: GenArgs,
        /// print retrieved chunk ids and planned queries
        #[arg(long)]
        verbose: bool,
    },
    /// Run the test set through the chatbot; appends to a JSONL file and resumes if interrupted
    Eval {
        #[command(flatten)]
        g: GenArgs,
        /// predictions file (JSONL)
        #[arg(long)]
        out: PathBuf,
        /// only the first N questions (after --every)
        #[arg(long)]
        limit: Option<usize>,
        /// take every n-th question (stratified quick runs: --every 10 -> 100 questions)
        #[arg(long, default_value_t = 1)]
        every: usize,
    },
    /// Score a predictions file; optionally merge external judge verdicts and/or export a judging file
    Score {
        preds: PathBuf,
        /// JSON {id: {"label": "correct|partial|incorrect"}}
        #[arg(long)]
        judge: Option<PathBuf>,
        /// write items for an external judge (question, reference, prediction)
        #[arg(long)]
        export_judge: Option<PathBuf>,
        /// write the metric table (markdown) here as well
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Write chunks (JSONL) and gold chunk ids per question (JSON) for inspection or for the fine-tuning step
    Export {
        #[arg(long, default_value = "store")]
        dir: PathBuf,
    },
    /// Retrieval-only check: recall of gold evidence for single-shot search with the raw question
    Retrieval {
        #[arg(long, value_enum, default_value = "bm25")]
        mode: retrieve::Mode,
        #[arg(long, default_value_t = 10)]
        k: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let corpus = cli.repo.join("data/corpus");
    let testset = cli.repo.join("data/testset/suryodaya_qa_testset_1000.json");
    let ol = ollama::Ollama::new(&cli.ollama);
    match cli.cmd {
        Cmd::Index => {
            let chunks = chunk::chunk_corpus(&corpus)?;
            let n = chunks.len();
            let st = index::Store::build(chunks);
            st.save(&cli.store)?;
            println!("indexed {n} chunks -> {}", cli.store.display());
        }
        Cmd::Embed { model, batch } => {
            let mut st = index::Store::load(&cli.store)?;
            let texts: Vec<String> = st.chunks.iter().map(index::index_text).collect();
            let mut all: Vec<f32> = Vec::new();
            let mut dim = 0;
            for (b, part) in texts.chunks(batch).enumerate() {
                let vs = ol.embed(&model, part)?;
                for v in vs {
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                    dim = v.len();
                    all.extend(v.iter().map(|x| x / n));
                }
                eprint!("\rembedded {}/{}", ((b + 1) * batch).min(texts.len()), texts.len());
            }
            eprintln!();
            st.emb = Some(all);
            st.dim = dim;
            st.emb_model = Some(model.clone());
            st.save(&cli.store)?;
            println!("stored {} embeddings (dim {dim}, model {model})", texts.len());
        }
        Cmd::Search { query, lang, doc, k, mode } => {
            let st = index::Store::load(&cli.store)?;
            let r = retrieve::Retriever { store: &st, ollama: Some(&ol) };
            for i in r.search(&query, lang.as_deref(), doc.as_deref(), k, mode)? {
                let c = &st.chunks[i];
                println!("=== [{}] {} | {} | {}\n{}\n", c.id, c.language, c.doc_title, c.heading, c.text);
            }
        }
        Cmd::Chat { g, verbose } => {
            let st = index::Store::load(&cli.store)?;
            if g.mode != retrieve::Mode::Bm25 && st.emb.is_none() {
                bail!("index has no embeddings: run `hr-rag embed` or use --mode bm25");
            }
            let rag = rag::Rag::new(&st, &ol, g.cfg());
            println!("HR assistant ({}, {:?}/{:?}). Ask in any of the 8 languages; empty line to quit.", g.model, g.strategy, g.mode);
            let stdin = std::io::stdin();
            loop {
                print!("\n> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
                    break;
                }
                let p = rag.run("chat", line.trim());
                if let Some(e) = &p.error {
                    println!("error: {e}");
                    continue;
                }
                println!("\n{}\n\n{}", p.short_answer, p.answer);
                for id in &p.cited_chunks {
                    if let Some(i) = st.pos(id) {
                        let c = &st.chunks[i];
                        println!("  - {} | {} | {}", c.language, c.doc_title, c.heading);
                    }
                }
                if !p.missing_info.trim().is_empty() {
                    println!("  (not found: {})", p.missing_info);
                }
                if verbose {
                    println!("  queries: {}\n  retrieved: {}\n  {} LLM calls, {:.1}s", p.queries.join(" | "), p.retrieved.join(", "), p.llm_calls, p.latency_ms as f64 / 1000.0);
                }
            }
        }
        Cmd::Eval { g, out, limit, every } => {
            let st = index::Store::load(&cli.store)?;
            if g.mode != retrieve::Mode::Bm25 && st.emb.is_none() {
                bail!("index has no embeddings: run `hr-rag embed` or use --mode bm25");
            }
            let items = eval::load_testset(&testset)?;
            let mut sel: Vec<&eval::Item> = items.iter().step_by(every.max(1)).collect();
            if let Some(l) = limit {
                sel.truncate(l);
            }
            let done = eval::read_predictions(&out)?;
            if let Some(d) = out.parent() {
                std::fs::create_dir_all(d)?;
            }
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&out)?;
            let rag = rag::Rag::new(&st, &ol, g.cfg());
            let todo: Vec<&&eval::Item> = sel.iter().filter(|it| !done.contains_key(&it.id)).collect();
            eprintln!("{} selected, {} already done, {} to run with {} ({:?}/{:?})", sel.len(), sel.len() - todo.len(), todo.len(), g.model, g.strategy, g.mode);
            for (n, it) in todo.iter().enumerate() {
                let p = rag.run(&it.id, &it.question);
                writeln!(f, "{}", serde_json::to_string(&p)?)?;
                f.flush()?;
                let ok = eval::numeric_match(&it.short_answer, &p).map(|b| if b { "ok" } else { "--" }).unwrap_or("..");
                eprintln!("[{}/{}] {} {} {:.1}s{}", n + 1, todo.len(), it.id, ok, p.latency_ms as f64 / 1000.0, p.error.as_ref().map(|e| format!(" ERROR {e}")).unwrap_or_default());
            }
            let preds = eval::read_predictions(&out)?;
            let ev = eval::load_evidence(&corpus)?;
            let gold = eval::gold_chunks(&st, &items, &ev);
            println!("{}", eval::print_table(&eval::score(&items, &preds, &gold, None)));
        }
        Cmd::Score { preds, judge, export_judge, report } => {
            let st = index::Store::load(&cli.store)?;
            let items = eval::load_testset(&testset)?;
            let p = eval::read_predictions(&preds)?;
            let ev = eval::load_evidence(&corpus)?;
            let gold = eval::gold_chunks(&st, &items, &ev);
            let j: Option<HashMap<String, String>> = match judge {
                Some(path) => {
                    let v: HashMap<String, serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
                    Some(v.into_iter().map(|(k, x)| (k, x["label"].as_str().unwrap_or("").to_string())).collect())
                }
                None => None,
            };
            let table = eval::print_table(&eval::score(&items, &p, &gold, j.as_ref()));
            println!("{table}");
            if let Some(r) = report {
                std::fs::write(r, &table)?;
            }
            if let Some(path) = export_judge {
                let out: Vec<serde_json::Value> = items
                    .iter()
                    .filter_map(|it| p.get(&it.id).map(|x| (it, x)))
                    .map(|(it, x)| serde_json::json!({"id": it.id, "question": it.question, "gold_short_answer": it.short_answer,
                        "gold_answer": it.answer, "system_short_answer": x.short_answer, "system_answer": x.answer}))
                    .collect();
                std::fs::write(&path, serde_json::to_string_pretty(&out)?)?;
                println!("wrote {} items for judging -> {}", out.len(), path.display());
            }
        }
        Cmd::Export { dir } => {
            let st = index::Store::load(&cli.store)?;
            let items = eval::load_testset(&testset)?;
            let ev = eval::load_evidence(&corpus)?;
            let gold = eval::gold_chunks(&st, &items, &ev);
            std::fs::create_dir_all(&dir)?;
            let mut f = std::io::BufWriter::new(std::fs::File::create(dir.join("chunks.jsonl"))?);
            for c in &st.chunks {
                writeln!(f, "{}", serde_json::json!({"id": c.id, "language": c.language, "document": c.document,
                    "doc_title": c.doc_title, "heading": c.heading, "text": c.text}))?;
            }
            let g: std::collections::BTreeMap<_, _> = gold.into_iter().collect();
            std::fs::write(dir.join("gold_chunks.json"), serde_json::to_string_pretty(&g)?)?;
            println!("wrote {} chunks and gold ids for {} questions -> {}", st.chunks.len(), g.len(), dir.display());
        }
        Cmd::Retrieval { mode, k } => {
            let st = index::Store::load(&cli.store)?;
            let items = eval::load_testset(&testset)?;
            let ev = eval::load_evidence(&corpus)?;
            let gold = eval::gold_chunks(&st, &items, &ev);
            let r = retrieve::Retriever { store: &st, ollama: Some(&ol) };
            let mut preds = HashMap::new();
            for (n, it) in items.iter().enumerate() {
                let ids = r.search(&it.question, None, None, k, mode)?;
                preds.insert(it.id.clone(), rag::Prediction { id: it.id.clone(), retrieved: ids.iter().map(|&i| st.chunks[i].id.clone()).collect(), ..Default::default() });
                if mode != retrieve::Mode::Bm25 {
                    eprint!("\r{}/{}", n + 1, items.len());
                }
            }
            eprintln!();
            println!("single-shot retrieval, mode {mode:?}, k={k} (numeric/judge columns not applicable)\n");
            println!("{}", eval::print_table(&eval::score(&items, &preds, &gold, None)));
        }
    }
    Ok(())
}
