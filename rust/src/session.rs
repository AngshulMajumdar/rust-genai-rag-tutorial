//! Owned, notebook-friendly API: everything a Jupyter (evcxr) cell needs, with rich HTML display.
use crate::eval::{self, Item, Row};
use crate::index::{self, Store};
use crate::ollama::Ollama;
use crate::rag::{Config, Prediction, Rag};
use crate::retrieve::{Mode, Retriever};
use anyhow::{bail, Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn show_html(h: &str) {
    println!("EVCXR_BEGIN_CONTENT text/html\n{h}\nEVCXR_END_CONTENT");
}

/// Accuracy report for one run (rows: ALL, one per question language, Cross-language, one per hop type).
#[derive(Clone)]
pub struct Report {
    pub name: String,
    pub rows: BTreeMap<String, Row>,
}

const COLS: [&str; 10] = ["group", "n", "numeric match %", "judge correct %", "all facts retrieved %", "fact recall %", "all facts cited %", "LLM calls/q", "s/q", "errors"];

impl Report {
    fn keys(&self) -> Vec<&String> {
        let mut k: Vec<&String> = self.rows.keys().collect();
        k.sort_by_key(|k| (k.as_str() != "ALL", !k.starts_with("lang:"), k.to_string()));
        k
    }
    fn cells(r: &Row) -> Vec<String> {
        vec![
            r.n.to_string(),
            Row::pct(r.auto_ok, r.auto_n),
            Row::pct(r.judge_correct, r.judged),
            Row::pct(r.all_retrieved, r.n),
            Row::pct(r.facts_retrieved, r.facts),
            Row::pct(r.all_cited, r.n),
            format!("{:.2}", r.llm_calls as f64 / r.n.max(1) as f64),
            format!("{:.1}", r.latency_ms as f64 / 1000.0 / r.n.max(1) as f64),
            r.errors.to_string(),
        ]
    }
    /// Numeric-match accuracy (%) for a group ("ALL", "Hindi", "Cross-language", "hop:cross_document", ...).
    pub fn accuracy(&self, group: &str) -> Option<f64> {
        let r = self.rows.get(group).or_else(|| self.rows.get(&format!("lang:{group}")))?;
        (r.auto_n > 0).then(|| 100.0 * r.auto_ok as f64 / r.auto_n as f64)
    }
    pub fn markdown(&self) -> String {
        eval::print_table(&self.rows)
    }
    pub fn html(&self) -> String {
        let mut h = format!("<b>{}</b><table><tr>{}</tr>", esc(&self.name), COLS.iter().map(|c| format!("<th>{c}</th>")).collect::<String>());
        for k in self.keys() {
            let style = if k == "ALL" { " style=\"font-weight:bold\"" } else { "" };
            h += &format!("<tr{style}><td>{}</td>{}</tr>", esc(k.trim_start_matches("lang:")), Self::cells(&self.rows[k]).iter().map(|c| format!("<td>{c}</td>")).collect::<String>());
        }
        h + "</table>"
    }
    /// Rendered automatically by the evcxr Jupyter kernel when a cell ends with a Report.
    pub fn evcxr_display(&self) {
        show_html(&self.html());
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n{}", self.name, self.markdown())
    }
}

/// Side-by-side summary of several runs (numeric match per language plus retrieval / citation figures).
pub struct Comparison(pub String, pub String);
impl Comparison {
    pub fn evcxr_display(&self) {
        show_html(&self.0);
    }
    pub fn markdown(&self) -> &str {
        &self.1
    }
}
impl std::fmt::Display for Comparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.1)
    }
}

/// All per-group tables of several runs in one display (evcxr shows only one HTML block per cell).
pub fn details(runs: &[&Report]) -> Comparison {
    Comparison(runs.iter().map(|r| r.html()).collect::<Vec<_>>().join("<br>"), runs.iter().map(|r| r.to_string()).collect::<Vec<_>>().join("\n"))
}

pub fn compare(runs: &[&Report]) -> Comparison {
    let langs = ["Bengali", "Hindi", "Kannada", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu", "Cross-language"];
    let mut head = vec!["run".to_string(), "ALL".into()];
    head.extend(langs.iter().map(|s| s.to_string()));
    head.extend(["all facts retrieved %", "all facts cited %", "LLM calls/q", "s/q", "errors"].map(String::from));
    let mut rows = Vec::new();
    for r in runs {
        let a = r.rows.get("ALL").cloned().unwrap_or_default();
        let mut v = vec![r.name.clone(), r.accuracy("ALL").map_or("-".into(), |x| format!("{x:.1}"))];
        v.extend(langs.iter().map(|l| r.accuracy(l).map_or("-".into(), |x| format!("{x:.1}"))));
        v.push(Row::pct(a.all_retrieved, a.n));
        v.push(Row::pct(a.all_cited, a.n));
        v.push(format!("{:.2}", a.llm_calls as f64 / a.n.max(1) as f64));
        v.push(format!("{:.1}", a.latency_ms as f64 / 1000.0 / a.n.max(1) as f64));
        v.push(a.errors.to_string());
        rows.push(v);
    }
    let mut h = String::from("<b>numeric-match accuracy % by question language</b><table><tr>");
    h += &head.iter().map(|c| format!("<th>{}</th>", esc(c))).collect::<String>();
    h += "</tr>";
    for v in &rows {
        h += &format!("<tr>{}</tr>", v.iter().map(|c| format!("<td>{}</td>", esc(c))).collect::<String>());
    }
    h += "</table>";
    let mut md = format!("| {} |\n|{}\n", head.join(" | "), "---|".repeat(head.len()));
    for v in &rows {
        md += &format!("| {} |\n", v.join(" | "));
    }
    Comparison(h, md)
}

/// One chatbot answer with its sources; renders as HTML in Jupyter.
pub struct Answer {
    pub question: String,
    pub prediction: Prediction,
    /// (language, document title, section heading) of each cited passage
    pub sources: Vec<(String, String, String)>,
    /// reference short answer, when the question comes from the test set
    pub reference: Option<String>,
}

impl Answer {
    pub fn evcxr_display(&self) {
        let p = &self.prediction;
        let mut h = format!("<div><i>{}</i><br><b>{}</b><p>{}</p>", esc(&self.question), esc(&p.short_answer), esc(&p.answer));
        if let Some(r) = &self.reference {
            h += &format!("<p><small>reference: {}</small></p>", esc(r));
        }
        if let Some(e) = &p.error {
            h += &format!("<p style=\"color:#b00\">error: {}</p>", esc(e));
        }
        if !self.sources.is_empty() {
            h += "<ul>";
            for (l, t, s) in &self.sources {
                h += &format!("<li>{} | {} | {}</li>", esc(l), esc(t), esc(s));
            }
            h += "</ul>";
        }
        if !p.missing_info.trim().is_empty() {
            h += &format!("<p><small>not found: {}</small></p>", esc(&p.missing_info));
        }
        h += &format!("<small>{} LLM calls, {:.1} s; queries: {}</small></div>", p.llm_calls, p.latency_ms as f64 / 1000.0, esc(&p.queries.join(" | ")));
        show_html(&h);
    }
}

impl std::fmt::Display for Answer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let p = &self.prediction;
        writeln!(f, "{}\n\n{}\n\n{}", self.question, p.short_answer, p.answer)?;
        if let Some(r) = &self.reference {
            writeln!(f, "reference: {r}")?;
        }
        for (l, t, s) in &self.sources {
            writeln!(f, "  - {l} | {t} | {s}")?;
        }
        Ok(())
    }
}

/// Owns the index, the Ollama client, the test set and the gold evidence.
pub struct Session {
    /// folder with one sub-folder per language holding NN_<topic>.md (and NN_<topic>.facts.json)
    pub corpus: PathBuf,
    /// the test set, if there is one (chat works without it; retrieval/eval need it)
    pub testset: Option<PathBuf>,
    /// where the index and the run files are written
    pub work: PathBuf,
    pub store_path: PathBuf,
    pub runs_dir: PathBuf,
    pub store: Store,
    pub ollama: Ollama,
    pub items: Vec<Item>,
    pub gold: HashMap<String, Vec<Vec<String>>>,
}

/// Remove the \\?\ prefix Windows puts on canonical paths, so paths print normally.
pub fn plain(p: PathBuf) -> PathBuf {
    match p.to_str().and_then(|s| s.strip_prefix(r"\\?\")) {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => p,
    }
}

/// Walk up from `start` to the folder that contains data/corpus and data/testset.
pub fn find_repo(start: &Path) -> Result<PathBuf> {
    let start = if start.as_os_str().is_empty() { std::env::current_dir()? } else { start.to_path_buf() };
    let start = plain(std::fs::canonicalize(&start).with_context(|| format!("{} does not exist", start.display()))?);
    for d in start.ancestors() {
        if d.join("data/corpus").is_dir() && d.join("data/testset/suryodaya_qa_testset_1000.json").is_file() {
            return Ok(d.to_path_buf());
        }
    }
    bail!("no data/corpus and data/testset found in {} or any parent folder; pass the repository folder to Session::open", start.display())
}

/// Fingerprint of the corpus files (names and sizes), so a changed corpus triggers re-indexing.
fn corpus_sig(corpus: &Path) -> Result<String> {
    let mut v: Vec<(String, u64)> = Vec::new();
    for lang in std::fs::read_dir(corpus)? {
        let lang = lang?.path();
        if !lang.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&lang)? {
            let f = f?;
            let name = f.file_name().to_string_lossy().to_string();
            if name.ends_with(".md") || name.ends_with(".facts.json") {
                v.push((format!("{}/{name}", lang.file_name().unwrap().to_string_lossy()), f.metadata()?.len()));
            }
        }
    }
    v.sort();
    let mut h: u64 = 0xcbf29ce484222325;
    for (n, l) in &v {
        for b in n.bytes().chain(l.to_le_bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{}:{h:016x}", v.len()))
}

impl Session {
    /// Open the repository (`""` = current folder or any parent) with Ollama at localhost:11434.
    pub fn open(path: &str) -> Result<Session> {
        Self::open_with(path, "http://localhost:11434")
    }

    pub fn open_with(path: &str, ollama_url: &str) -> Result<Session> {
        let repo = find_repo(Path::new(path))?;
        Self::open_dirs(
            &repo.join("data").join("corpus"),
            Some(&repo.join("data").join("testset").join("suryodaya_qa_testset_1000.json")),
            &repo.join("rust"),
            ollama_url,
        )
    }

    /// Open explicit locations. The index is built on first use (about a second), reused afterwards,
    /// and rebuilt automatically if the corpus files change.
    pub fn open_dirs(corpus: &Path, testset: Option<&Path>, work: &Path, ollama_url: &str) -> Result<Session> {
        let store_path = work.join("store").join("index.bin");
        let runs_dir = work.join("runs");
        let sig = corpus_sig(corpus)?;
        let store = match Store::load(&store_path) {
            Ok(s) if s.sig == sig => s,
            _ => {
                let mut s = Store::build(crate::chunk::chunk_corpus(corpus)?);
                if s.chunks.is_empty() {
                    bail!("no NN_*.md documents found under {}", corpus.display());
                }
                s.sig = sig;
                s.save(&store_path)?;
                s
            }
        };
        let items = match testset {
            Some(t) => eval::load_testset(t)?,
            None => Vec::new(),
        };
        let gold = eval::gold_chunks(&store, &items, &eval::load_evidence(corpus)?);
        let s = Session {
            corpus: corpus.to_path_buf(),
            testset: testset.map(|t| t.to_path_buf()),
            work: work.to_path_buf(),
            store_path,
            runs_dir,
            store,
            ollama: Ollama::new(ollama_url),
            items,
            gold,
        };
        println!("{}", s.status());
        Ok(s)
    }

    pub fn status(&self) -> String {
        let docs: std::collections::BTreeSet<_> = self.store.chunks.iter().map(|c| (&c.language, &c.document)).collect();
        let langs: std::collections::BTreeSet<_> = self.store.chunks.iter().map(|c| &c.language).collect();
        format!(
            "corpus  {}\ntest set {}\nwork    {}\n{} documents in {} languages, {} chunks, {} test questions; embeddings: {}",
            self.corpus.display(),
            self.testset.as_ref().map_or("(none: chat only)".into(), |t| t.display().to_string()),
            self.work.display(),
            docs.len(),
            langs.len(),
            self.store.chunks.len(),
            self.items.len(),
            match (&self.store.emb_model, self.store.emb.is_some()) {
                (Some(m), true) => format!("{m} (dim {})", self.store.dim),
                _ => "none (BM25 only until embed() is run)".into(),
            }
        )
    }

    /// Re-chunk and re-index the corpus. Drops embeddings.
    pub fn rebuild_index(&mut self) -> Result<()> {
        let mut st = Store::build(crate::chunk::chunk_corpus(&self.corpus)?);
        st.sig = corpus_sig(&self.corpus)?;
        st.save(&self.store_path)?;
        self.store = st;
        self.gold = eval::gold_chunks(&self.store, &self.items, &eval::load_evidence(&self.corpus)?);
        println!("{}", self.status());
        Ok(())
    }

    fn need_testset(&self) -> Result<()> {
        if self.items.is_empty() {
            bail!("no test set loaded: retrieval and accuracy runs need one");
        }
        Ok(())
    }

    /// Check that Ollama is running and download any of `models` that are missing.
    pub fn ensure_models(&self, models: &[&str]) -> Result<()> {
        let have = self.ollama.models()?;
        println!("Ollama at {} has: {}", self.ollama.base(), if have.is_empty() { "(no models)".into() } else { have.join(", ") });
        for m in models {
            self.ollama.pull(m)?;
        }
        Ok(())
    }

    /// Embed all chunks with an Ollama embedding model and store the vectors in the index (skipped if already done).
    pub fn embed(&mut self, model: &str) -> Result<()> {
        if self.store.emb.is_some() && self.store.emb_model.as_deref() == Some(model) {
            println!("embeddings for {model} already in the index");
            return Ok(());
        }
        let texts: Vec<String> = self.store.chunks.iter().map(index::index_text).collect();
        let t0 = std::time::Instant::now();
        let mut all = Vec::with_capacity(texts.len() * 1024);
        let mut dim = 0;
        let batch = 16;
        for (b, part) in texts.chunks(batch).enumerate() {
            for v in self.ollama.embed(model, part)? {
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                dim = v.len();
                all.extend(v.iter().map(|x| x / n));
            }
            let done = ((b + 1) * batch).min(texts.len());
            if b % 25 == 24 || done == texts.len() {
                eprintln!("embedded {done}/{} ({:.0} s)", texts.len(), t0.elapsed().as_secs_f64());
            }
        }
        self.store.emb = Some(all);
        self.store.dim = dim;
        self.store.emb_model = Some(model.to_string());
        self.store.save(&self.store_path)?;
        println!("stored {} embeddings (dim {dim}) in {}", texts.len(), self.store_path.display());
        Ok(())
    }

    fn check_mode(&self, mode: Mode) -> Result<()> {
        if mode != Mode::Bm25 && self.store.emb.is_none() {
            bail!("mode {mode:?} needs embeddings: run session.embed(\"bge-m3\") first, or use Mode::Bm25");
        }
        Ok(())
    }

    /// Search the corpus; returns (chunk id, language, document title, heading, text).
    pub fn search(&self, query: &str, lang: Option<&str>, k: usize, mode: Mode) -> Result<Vec<(String, String, String, String, String)>> {
        self.check_mode(mode)?;
        let r = Retriever { store: &self.store, ollama: Some(&self.ollama) };
        Ok(r.search(query, lang, None, k, mode)?
            .into_iter()
            .map(|i| {
                let c = &self.store.chunks[i];
                (c.id.clone(), c.language.clone(), c.doc_title.clone(), c.heading.clone(), c.text.clone())
            })
            .collect())
    }

    /// Ask the chatbot one question (any of the 8 languages).
    pub fn ask(&self, question: &str, cfg: &Config) -> Result<Answer> {
        self.check_mode(cfg.mode)?;
        let p = Rag::new(&self.store, &self.ollama, cfg.clone()).run("chat", question);
        let sources = p
            .cited_chunks
            .iter()
            .filter_map(|id| self.store.pos(id))
            .map(|i| {
                let c = &self.store.chunks[i];
                (c.language.clone(), c.doc_title.clone(), c.heading.clone())
            })
            .collect();
        Ok(Answer { question: question.to_string(), prediction: p, sources, reference: None })
    }

    /// A test-set question by id, e.g. "HI-029".
    pub fn item(&self, id: &str) -> Result<&Item> {
        self.items.iter().find(|it| it.id == id).with_context(|| format!("no question {id}"))
    }

    /// Ask a test-set question and show the reference short answer next to the chatbot's.
    pub fn ask_item(&self, id: &str, cfg: &Config) -> Result<Answer> {
        let it = self.item(id)?;
        let mut a = self.ask(&it.question, cfg)?;
        a.reference = Some(it.short_answer.clone());
        Ok(a)
    }

    fn report(&self, name: &str, preds: &HashMap<String, Prediction>, judge: Option<&HashMap<String, String>>) -> Report {
        Report { name: name.to_string(), rows: eval::score(&self.items, preds, &self.gold, judge) }
    }

    /// Single-shot retrieval with the raw question (no LLM): evidence recall per group.
    pub fn retrieval(&self, mode: Mode, k: usize) -> Result<Report> {
        self.need_testset()?;
        self.check_mode(mode)?;
        let r = Retriever { store: &self.store, ollama: Some(&self.ollama) };
        let mut preds = HashMap::new();
        for (n, it) in self.items.iter().enumerate() {
            let ids = r.search(&it.question, None, None, k, mode)?;
            preds.insert(it.id.clone(), Prediction { id: it.id.clone(), retrieved: ids.iter().map(|&i| self.store.chunks[i].id.clone()).collect(), ..Default::default() });
            if mode != Mode::Bm25 && (n + 1) % 200 == 0 {
                eprintln!("{}/{}", n + 1, self.items.len());
            }
        }
        Ok(self.report(&format!("retrieval only, {mode:?}, k={k}"), &preds, None))
    }

    /// The questions a run uses: every `every`-th question (10 -> 100 questions with the same language/hop mix).
    pub fn subset(&self, every: usize) -> Vec<&Item> {
        self.items.iter().step_by(every.max(1)).collect()
    }

    /// Run the test set (every `every`-th question) through the chatbot. Predictions are appended to
    /// rust/runs/<run name>.jsonl, so an interrupted run resumes where it stopped; the report is also written as .md.
    pub fn eval(&self, cfg: &Config, every: usize) -> Result<Report> {
        self.eval_named(cfg, every, &cfg.name())
    }

    pub fn eval_named(&self, cfg: &Config, every: usize, name: &str) -> Result<Report> {
        self.need_testset()?;
        self.check_mode(cfg.mode)?;
        std::fs::create_dir_all(&self.runs_dir)?;
        let out = self.runs_dir.join(format!("{name}.jsonl"));
        let done = eval::read_predictions(&out)?;
        let sel = self.subset(every);
        let todo: Vec<&&Item> = sel.iter().filter(|it| !done.contains_key(&it.id)).collect();
        eprintln!("{name}: {} questions, {} already done, {} to run", sel.len(), sel.len() - todo.len(), todo.len());
        let rag = Rag::new(&self.store, &self.ollama, cfg.clone());
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&out)?;
        let t0 = std::time::Instant::now();
        let (mut ok, mut seen) = (0usize, 0usize);
        for (n, it) in todo.iter().enumerate() {
            let p = rag.run(&it.id, &it.question);
            writeln!(f, "{}", serde_json::to_string(&p)?)?;
            f.flush()?;
            if let Some(m) = eval::numeric_match(&it.short_answer, &p) {
                seen += 1;
                ok += m as usize;
            }
            if let Some(e) = &p.error {
                eprintln!("{} ERROR {e}", it.id);
            }
            if (n + 1) % 10 == 0 || n + 1 == todo.len() {
                let el = t0.elapsed().as_secs_f64();
                eprintln!(
                    "[{}/{}] running numeric match {:.1}% | {:.1} s/q | about {:.0} min left",
                    n + 1,
                    todo.len(),
                    100.0 * ok as f64 / seen.max(1) as f64,
                    el / (n + 1) as f64,
                    el / (n + 1) as f64 * (todo.len() - n - 1) as f64 / 60.0
                );
            }
        }
        let all = eval::read_predictions(&out)?;
        let ids: std::collections::HashSet<&str> = sel.iter().map(|it| it.id.as_str()).collect();
        let preds: HashMap<String, Prediction> = all.into_iter().filter(|(k, _)| ids.contains(k.as_str())).collect();
        let rep = self.report(name, &preds, None);
        std::fs::write(self.runs_dir.join(format!("{name}.md")), rep.markdown())?;
        Ok(rep)
    }

    /// Score an existing run (optionally with judge verdicts {id: {"label": ...}}).
    pub fn score(&self, name: &str, judge: Option<&str>) -> Result<Report> {
        let preds = eval::read_predictions(&self.runs_dir.join(format!("{name}.jsonl")))?;
        if preds.is_empty() {
            bail!("no predictions in {}", self.runs_dir.join(format!("{name}.jsonl")).display());
        }
        let j: Option<HashMap<String, String>> = match judge {
            Some(p) => {
                let v: HashMap<String, serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(p)?)?;
                Some(v.into_iter().map(|(k, x)| (k, x["label"].as_str().unwrap_or("").to_string())).collect())
            }
            None => None,
        };
        Ok(self.report(name, &preds, j.as_ref()))
    }
}
