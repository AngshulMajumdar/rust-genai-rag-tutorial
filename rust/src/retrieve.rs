//! Retrieval: BM25 (word + char RRF, identical to rag/search.py), dense cosine, or hybrid.
use crate::index::Store;
use crate::ollama::Ollama;
use crate::text::{char_grams, word_tokens};
use anyhow::{bail, Result};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Mode {
    Bm25,
    Dense,
    Hybrid,
}

pub struct Retriever<'a> {
    pub store: &'a Store,
    pub ollama: Option<&'a Ollama>,
}

const RRF_K: f32 = 60.0;
const DEPTH: usize = 200;

fn ranked(sc: HashMap<u32, f32>) -> Vec<u32> {
    let mut v: Vec<(u32, f32)> = sc.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v.into_iter().take(DEPTH).map(|x| x.0).collect()
}

impl<'a> Retriever<'a> {
    fn allowed(&self, lang: Option<&str>, doc: Option<&str>) -> Option<Vec<bool>> {
        if lang.is_none() && doc.is_none() {
            return None;
        }
        Some(
            self.store
                .chunks
                .iter()
                .map(|c| lang.map_or(true, |l| c.language.eq_ignore_ascii_case(l)) && doc.map_or(true, |d| c.document == d))
                .collect(),
        )
    }

    fn dense_rank(&self, q: &str, allowed: Option<&[bool]>) -> Result<Vec<u32>> {
        let (Some(emb), Some(ol), Some(model)) = (&self.store.emb, self.ollama, &self.store.emb_model) else {
            bail!("dense retrieval needs embeddings: run `hr-rag embed` (and have Ollama running)")
        };
        let qv = ol.embed(model, &[q.to_string()])?.remove(0);
        let nq = qv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let d = self.store.dim;
        let mut sc = HashMap::new();
        for i in 0..self.store.chunks.len() {
            if let Some(a) = allowed {
                if !a[i] {
                    continue;
                }
            }
            let row = &emb[i * d..(i + 1) * d];
            let s: f32 = row.iter().zip(&qv).map(|(a, b)| a * b).sum::<f32>() / nq;
            sc.insert(i as u32, s);
        }
        Ok(ranked(sc))
    }

    /// Top-k chunk indices for `q`, optionally restricted to a language and/or document.
    pub fn search(&self, q: &str, lang: Option<&str>, doc: Option<&str>, k: usize, mode: Mode) -> Result<Vec<usize>> {
        let allowed = self.allowed(lang, doc);
        let a = allowed.as_deref();
        let mut lists: Vec<Vec<u32>> = Vec::new();
        if mode != Mode::Dense {
            lists.push(ranked(self.store.word.score(&word_tokens(q), a)));
            lists.push(ranked(self.store.char.score(&char_grams(q), a)));
        }
        if mode != Mode::Bm25 {
            lists.push(self.dense_rank(q, a)?);
        }
        let mut fused: HashMap<u32, f32> = HashMap::new();
        for l in &lists {
            for (r, &i) in l.iter().enumerate() {
                *fused.entry(i).or_default() += 1.0 / (RRF_K + r as f32 + 1.0);
            }
        }
        let mut v: Vec<(u32, f32)> = fused.into_iter().collect();
        v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
        Ok(v.into_iter().take(k).map(|x| x.0 as usize).collect())
    }
}
