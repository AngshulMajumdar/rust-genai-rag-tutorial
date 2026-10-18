//! RAG answering: a single-shot pipeline and an agentic plan-retrieve-answer loop.
use crate::index::Store;
use crate::ollama::{Msg, Ollama};
use crate::retrieve::{Mode, Retriever};
use crate::text::detect_language;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Instant;

pub const LANGUAGES: [&str; 8] = ["Bengali", "Hindi", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu", "Kannada"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Strategy {
    /// Retrieve once with the question, answer once.
    Simple,
    /// Plan per-fact queries (in each document's language), retrieve, answer; re-plan if facts are missing.
    Agentic,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub model: String,
    pub mode: Mode,
    pub strategy: Strategy,
    /// Chunks retrieved per query.
    pub k: usize,
    /// Maximum chunks placed in the answer prompt.
    pub max_ctx: usize,
    pub rounds: usize,
    pub num_ctx: usize,
}

impl Config {
    /// Defaults used for the reported runs: hybrid retrieval, agentic strategy, k 6, 12 passages, 2 rounds, 12k context.
    pub fn new(model: &str) -> Self {
        Config { model: model.to_string(), mode: Mode::Hybrid, strategy: Strategy::Agentic, k: 6, max_ctx: 12, rounds: 2, num_ctx: 12288 }
    }
    pub fn simple(mut self) -> Self {
        self.strategy = Strategy::Simple;
        self
    }
    pub fn agentic(mut self) -> Self {
        self.strategy = Strategy::Agentic;
        self
    }
    pub fn mode(mut self, m: Mode) -> Self {
        self.mode = m;
        self
    }
    /// Short run name such as "gemma3-4b_agentic_hybrid".
    pub fn name(&self) -> String {
        let m: String = self.model.chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' || c == '-' { c } else { '-' }).collect();
        format!("{m}_{:?}_{:?}", self.strategy, self.mode).to_lowercase()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Prediction {
    pub id: String,
    pub short_answer: String,
    pub answer: String,
    pub cited_chunks: Vec<String>,
    pub retrieved: Vec<String>,
    pub queries: Vec<String>,
    pub missing_info: String,
    pub llm_calls: usize,
    pub latency_ms: u128,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Deserialize, Default)]
struct AnswerOut {
    #[serde(default)]
    short_answer: String,
    #[serde(default)]
    answer: String,
    #[serde(default)]
    cited_chunks: Vec<String>,
    #[serde(default)]
    missing_info: String,
}

#[derive(Deserialize, Default)]
struct PlanOut {
    #[serde(default)]
    queries: Vec<PlanQuery>,
}
#[derive(Deserialize, Default, Clone)]
struct PlanQuery {
    #[serde(default)]
    query: String,
    #[serde(default)]
    language: String,
}

fn answer_schema() -> Value {
    json!({"type": "object",
           "properties": {"short_answer": {"type": "string"}, "answer": {"type": "string"},
                          "cited_chunks": {"type": "array", "items": {"type": "string"}},
                          "missing_info": {"type": "string"}},
           "required": ["short_answer", "answer", "cited_chunks", "missing_info"]})
}
fn plan_schema() -> Value {
    json!({"type": "object",
           "properties": {"queries": {"type": "array", "maxItems": 4, "items": {"type": "object",
               "properties": {"query": {"type": "string"}, "language": {"type": "string", "enum": LANGUAGES}},
               "required": ["query", "language"]}}},
           "required": ["queries"]})
}

pub const ANSWER_SYSTEM: &str = "You are the HR-policy assistant of Suryodaya Integrated Solutions Pvt. Ltd. \
Answer ONLY from the numbered context passages; never use outside knowledge. A question may need facts from several \
passages, possibly in different languages: find each fact, then combine them and do exactly the arithmetic or comparison \
asked (keep working days and calendar days apart). Write in the language of the question. Fields: short_answer = the \
minimal final answer (a number with its unit, yes/no with the decisive figure, a date, or a short phrase); answer = 2-4 \
sentences with the figures you used and the reasoning; cited_chunks = ids of the passages you used; missing_info = a \
short description of any needed fact that is NOT in the context (empty string if nothing is missing). Never invent a figure.";

const PLAN_SYSTEM: &str = "You plan searches over the HR-policy documents of one company, written in 8 Indian languages. \
Break the question into the separate facts it needs and write one short keyword-style search query per fact (at most 4). \
Write each query in the language of the document most likely to contain that fact, using that document's own terms, and \
give that language.";

pub struct Rag<'a> {
    pub store: &'a Store,
    pub ollama: &'a Ollama,
    pub cfg: Config,
    doc_list: String,
}

fn parse<T: for<'de> Deserialize<'de> + Default>(s: &str) -> Option<T> {
    serde_json::from_str(s).ok().or_else(|| {
        let a = s.find('{')?;
        let b = s.rfind('}')?;
        serde_json::from_str(&s[a..=b]).ok()
    })
}

impl<'a> Rag<'a> {
    pub fn new(store: &'a Store, ollama: &'a Ollama, cfg: Config) -> Self {
        let mut seen = std::collections::BTreeSet::new();
        for c in &store.chunks {
            seen.insert(format!("{} | {}", c.language, c.doc_title));
        }
        let doc_list = seen.into_iter().collect::<Vec<_>>().join("\n");
        Rag { store, ollama, cfg, doc_list }
    }

    fn retriever(&self) -> Retriever<'_> {
        Retriever { store: self.store, ollama: Some(self.ollama) }
    }

    fn context(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&i| {
                let c = &self.store.chunks[i];
                format!("[{}] ({} | {} | {})\n{}", c.id, c.language, c.doc_title, c.heading, c.text)
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    fn answer(&self, q: &str, ids: &[usize]) -> Result<AnswerOut> {
        let lang = detect_language(q);
        let msgs = [
            Msg { role: "system", content: ANSWER_SYSTEM.to_string() },
            Msg { role: "user", content: format!("Context passages:\n\n{}\n\nQuestion ({lang}): {q}", self.context(ids)) },
        ];
        let raw = self.ollama.chat(&self.cfg.model, &msgs, Some(&answer_schema()), self.cfg.num_ctx)?;
        Ok(parse::<AnswerOut>(&raw).unwrap_or(AnswerOut { short_answer: raw.chars().take(300).collect(), ..Default::default() }))
    }

    fn plan(&self, q: &str, missing: Option<&str>) -> Result<Vec<PlanQuery>> {
        let extra = missing.map(|m| format!("\n\nA first attempt could not find: {m}\nPlan new queries for the missing facts only.")).unwrap_or_default();
        let msgs = [
            Msg { role: "system", content: PLAN_SYSTEM.to_string() },
            Msg { role: "user", content: format!("Documents (language | title):\n{}\n\nQuestion: {q}{extra}", self.doc_list) },
        ];
        let raw = self.ollama.chat(&self.cfg.model, &msgs, Some(&plan_schema()), self.cfg.num_ctx)?;
        Ok(parse::<PlanOut>(&raw).unwrap_or_default().queries.into_iter().filter(|p| !p.query.trim().is_empty()).take(4).collect())
    }

    pub fn run(&self, id: &str, q: &str) -> Prediction {
        let t0 = Instant::now();
        let mut p = Prediction { id: id.to_string(), ..Default::default() };
        if let Err(e) = self.run_inner(q, &mut p) {
            p.error = Some(e.to_string());
        }
        p.latency_ms = t0.elapsed().as_millis();
        p
    }

    fn run_inner(&self, q: &str, p: &mut Prediction) -> Result<()> {
        let r = self.retriever();
        let c = &self.cfg;
        let mut ids: Vec<usize> = Vec::new();
        let push = |ids: &mut Vec<usize>, new: Vec<usize>| {
            for i in new {
                if !ids.contains(&i) {
                    ids.push(i);
                }
            }
        };
        match c.strategy {
            Strategy::Simple => {
                push(&mut ids, r.search(q, None, None, c.max_ctx, c.mode)?);
                let a = self.answer(q, &ids)?;
                p.llm_calls = 1;
                self.fill(p, a, &ids);
            }
            Strategy::Agentic => {
                // the question itself is always one query; plan queries add one result list each
                let mut lists: Vec<Vec<usize>> = vec![r.search(q, None, None, c.max_ctx.max(c.k), c.mode)?];
                let mut missing: Option<String> = None;
                for round in 0..c.rounds.max(1) {
                    let plan = self.plan(q, missing.as_deref())?;
                    p.llm_calls += 1;
                    for pq in &plan {
                        let lang = LANGUAGES.iter().find(|l| l.eq_ignore_ascii_case(pq.language.trim())).copied();
                        let tag = format!("{}: {}", lang.unwrap_or("any"), pq.query.trim());
                        if p.queries.contains(&tag) {
                            continue;
                        }
                        p.queries.push(tag);
                        lists.push(r.search(&pq.query, lang, None, c.k, c.mode)?);
                    }
                    // the question's own top half is always kept (so a weak plan can never do worse than simple
                    // retrieval on those slots); the rest is filled round-robin so every planned query contributes
                    ids.clear();
                    push(&mut ids, lists[0].iter().take(c.max_ctx / 2).copied().collect());
                    let depth = lists.iter().map(|l| l.len()).max().unwrap_or(0);
                    'fill: for d in 0..depth {
                        for l in &lists {
                            if let Some(&i) = l.get(d) {
                                push(&mut ids, vec![i]);
                                if ids.len() >= c.max_ctx {
                                    break 'fill;
                                }
                            }
                        }
                    }
                    let a = self.answer(q, &ids)?;
                    p.llm_calls += 1;
                    let miss = a.missing_info.trim().to_string();
                    self.fill(p, a, &ids);
                    if miss.is_empty() || round + 1 == c.rounds.max(1) {
                        break;
                    }
                    missing = Some(miss);
                }
            }
        }
        Ok(())
    }

    fn fill(&self, p: &mut Prediction, a: AnswerOut, ids: &[usize]) {
        p.short_answer = a.short_answer;
        p.answer = a.answer;
        p.missing_info = a.missing_info;
        let valid: std::collections::HashSet<&str> = ids.iter().map(|&i| self.store.chunks[i].id.as_str()).collect();
        p.cited_chunks = a.cited_chunks.into_iter().map(|s| s.trim().trim_matches(|c| c == '[' || c == ']').to_string()).filter(|s| valid.contains(s.as_str())).collect();
        p.retrieved = ids.iter().map(|&i| self.store.chunks[i].id.clone()).collect();
    }
}
