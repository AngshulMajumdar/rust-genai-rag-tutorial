//! Test-set loading, gold-evidence mapping, and scoring (ports eval/retrieval_baseline.py and eval/score.py).
use crate::index::Store;
use crate::rag::Prediction;
use crate::text::{numbers, sentences, squash};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Fact {
    pub language: String,
    pub document: String,
    pub subject: String,
    pub aspect: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Item {
    pub id: String,
    pub language: String,
    pub hop_type: String,
    pub question: String,
    pub short_answer: String,
    pub answer: String,
    pub supporting_facts: Vec<Fact>,
}

pub fn load_testset(path: &Path) -> Result<Vec<Item>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

/// (language, document, subject, aspect) -> evidence text, from data/corpus/*/*.facts.json
pub fn load_evidence(corpus: &Path) -> Result<HashMap<(String, String, String, String), String>> {
    #[derive(Deserialize)]
    struct F {
        subject: String,
        aspect: String,
        evidence: Option<String>,
    }
    let mut m = HashMap::new();
    for lang in std::fs::read_dir(corpus)? {
        let lang = lang?.path();
        if !lang.is_dir() {
            continue;
        }
        let l = lang.file_name().unwrap().to_string_lossy().to_string();
        for f in std::fs::read_dir(&lang)? {
            let f = f?.path();
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            if let Some(doc) = name.strip_suffix(".facts.json") {
                let v: Vec<F> = serde_json::from_str(&std::fs::read_to_string(&f)?)?;
                for x in v {
                    m.insert((l.clone(), doc.to_string(), x.subject, x.aspect), x.evidence.unwrap_or_default());
                }
            }
        }
    }
    Ok(m)
}

/// Gold chunk ids per supporting fact (same procedure as eval/retrieval_baseline.py).
pub fn gold_chunks(store: &Store, items: &[Item], ev: &HashMap<(String, String, String, String), String>) -> HashMap<String, Vec<Vec<String>>> {
    let ctext: Vec<String> = store.chunks.iter().map(|c| squash(&c.text)).collect();
    let mut by_doc: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (i, c) in store.chunks.iter().enumerate() {
        by_doc.entry((c.language.clone(), c.document.clone())).or_default().push(i);
    }
    let mut out = HashMap::new();
    for it in items {
        let mut g = Vec::new();
        for f in &it.supporting_facts {
            let e = ev.get(&(f.language.clone(), f.document.clone(), f.subject.clone(), f.aspect.clone())).cloned().unwrap_or_default();
            let cand = by_doc.get(&(f.language.clone(), f.document.clone())).cloned().unwrap_or_default();
            let parts: Vec<String> = sentences(&e).into_iter().filter(|p| p.trim().chars().count() > 8).map(|p| squash(&p)).collect();
            let mut ids: Vec<String> = cand
                .iter()
                .filter(|&&i| parts.iter().any(|p| !p.is_empty() && ctext[i].contains(p.as_str())))
                .map(|&i| store.chunks[i].id.clone())
                .collect();
            ids.sort();
            if ids.is_empty() && !cand.is_empty() {
                let evc: Vec<char> = squash(&e).chars().collect();
                let last = if evc.len() > 40 { evc.len() - 40 } else { 1 };
                let windows: Vec<String> = (0..last).step_by(20).map(|j| evc[j.min(evc.len())..(j + 40).min(evc.len())].iter().collect()).collect();
                let score = |i: usize| windows.iter().filter(|w| !w.is_empty() && ctext[i].contains(w.as_str())).map(|w| w.chars().count()).max().unwrap_or(0);
                let mut best = cand[0];
                let mut bs = score(best);
                for &i in &cand[1..] {
                    let s = score(i);
                    if s > bs {
                        best = i;
                        bs = s;
                    }
                }
                if bs > 0 {
                    ids = vec![store.chunks[best].id.clone()];
                }
            }
            g.push(ids);
        }
        out.insert(it.id.clone(), g);
    }
    out
}

pub fn group_of(it: &Item) -> String {
    if it.hop_type == "cross_language" { "Cross-language".into() } else { it.language.clone() }
}

#[derive(Default, Clone, Serialize)]
pub struct Row {
    pub n: usize,
    pub auto_n: usize,
    pub auto_ok: usize,
    pub facts: usize,
    pub facts_retrieved: usize,
    pub all_retrieved: usize,
    pub facts_cited: usize,
    pub all_cited: usize,
    pub errors: usize,
    pub judged: usize,
    pub judge_correct: usize,
    pub judge_partial: usize,
    pub llm_calls: usize,
    pub latency_ms: u128,
}

impl Row {
    pub fn pct(a: usize, b: usize) -> String {
        if b == 0 { "-".into() } else { format!("{:.1}", 100.0 * a as f64 / b as f64) }
    }
}

/// Is every number in the reference short answer present in the prediction?
pub fn numeric_match(gold_short: &str, p: &Prediction) -> Option<bool> {
    let g = numbers(gold_short);
    if g.is_empty() {
        return None;
    }
    let mut pn = numbers(&p.short_answer);
    pn.extend(numbers(&p.answer));
    Some(g.iter().all(|x| pn.iter().any(|y| (x - y).abs() < 1e-9)))
}

pub fn score(
    items: &[Item],
    preds: &HashMap<String, Prediction>,
    gold: &HashMap<String, Vec<Vec<String>>>,
    judge: Option<&HashMap<String, String>>,
) -> BTreeMap<String, Row> {
    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    for it in items {
        let Some(p) = preds.get(&it.id) else { continue };
        let retrieved: HashSet<&str> = p.retrieved.iter().map(|s| s.as_str()).collect();
        let cited: HashSet<&str> = p.cited_chunks.iter().map(|s| s.as_str()).collect();
        let g: Vec<&Vec<String>> = gold.get(&it.id).map(|v| v.iter().filter(|x| !x.is_empty()).collect()).unwrap_or_default();
        let hit_r: Vec<bool> = g.iter().map(|ids| ids.iter().any(|i| retrieved.contains(i.as_str()))).collect();
        let hit_c: Vec<bool> = g.iter().map(|ids| ids.iter().any(|i| cited.contains(i.as_str()))).collect();
        let answered = !(p.short_answer.is_empty() && p.answer.is_empty());
        let auto = if answered { numeric_match(&it.short_answer, p) } else { None };
        let verdict = judge.and_then(|j| j.get(&it.id));
        for key in ["ALL".to_string(), format!("lang:{}", group_of(it)), format!("hop:{}", it.hop_type)] {
            let r = rows.entry(key).or_default();
            r.n += 1;
            if let Some(a) = auto {
                r.auto_n += 1;
                r.auto_ok += a as usize;
            }
            r.facts += hit_r.len();
            r.facts_retrieved += hit_r.iter().filter(|&&x| x).count();
            r.all_retrieved += hit_r.iter().all(|&x| x) as usize;
            r.facts_cited += hit_c.iter().filter(|&&x| x).count();
            r.all_cited += hit_c.iter().all(|&x| x) as usize;
            r.errors += p.error.is_some() as usize;
            r.llm_calls += p.llm_calls;
            r.latency_ms += p.latency_ms;
            if let Some(v) = verdict {
                r.judged += 1;
                r.judge_correct += (v == "correct") as usize;
                r.judge_partial += (v == "partial") as usize;
            }
        }
    }
    rows
}

pub fn print_table(rows: &BTreeMap<String, Row>) -> String {
    let mut s = String::from(
        "| group | n | numeric match % | judge correct % | all facts retrieved % | fact recall (retrieved) % | all facts cited % | LLM calls/q | s/q | errors |\n|---|---|---|---|---|---|---|---|---|---|\n",
    );
    let mut keys: Vec<&String> = rows.keys().collect();
    keys.sort_by_key(|k| (k.as_str() != "ALL", !k.starts_with("lang:"), k.to_string()));
    for k in keys {
        let r = &rows[k];
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.2} | {:.1} | {} |\n",
            k.trim_start_matches("lang:"),
            r.n,
            Row::pct(r.auto_ok, r.auto_n),
            Row::pct(r.judge_correct, r.judged),
            Row::pct(r.all_retrieved, r.n),
            Row::pct(r.facts_retrieved, r.facts),
            Row::pct(r.all_cited, r.n),
            r.llm_calls as f64 / r.n.max(1) as f64,
            r.latency_ms as f64 / 1000.0 / r.n.max(1) as f64,
            r.errors
        ));
    }
    s
}

pub fn read_predictions(path: &Path) -> Result<HashMap<String, Prediction>> {
    let mut m = HashMap::new();
    if path.exists() {
        for line in std::fs::read_to_string(path)?.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let p: Prediction = serde_json::from_str(line)?;
            m.insert(p.id.clone(), p);
        }
    }
    Ok(m)
}
