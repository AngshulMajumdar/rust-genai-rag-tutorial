//! BM25 indexes (word + char 4-gram), optional dense embeddings, and the on-disk store.
use crate::chunk::Chunk;
use crate::text::{char_grams, word_tokens};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub struct Bm25 {
    k1: f32,
    b: f32,
    avg: f32,
    len: Vec<u32>,
    post: HashMap<String, Vec<(u32, u32)>>,
    idf: HashMap<String, f32>,
}

impl Bm25 {
    pub fn build(docs: &[Vec<String>]) -> Self {
        let n = docs.len();
        let len: Vec<u32> = docs.iter().map(|d| d.len() as u32).collect();
        let avg = len.iter().map(|&x| x as f32).sum::<f32>() / n.max(1) as f32;
        let mut post: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        for (i, toks) in docs.iter().enumerate() {
            let mut tf: HashMap<&str, u32> = HashMap::new();
            for t in toks {
                *tf.entry(t.as_str()).or_default() += 1;
            }
            for (t, f) in tf {
                post.entry(t.to_string()).or_default().push((i as u32, f));
            }
        }
        let idf = post
            .iter()
            .map(|(t, p)| {
                let df = p.len() as f32;
                (t.clone(), (1.0 + (n as f32 - df + 0.5) / (df + 0.5)).ln())
            })
            .collect();
        Bm25 { k1: 1.4, b: 0.75, avg, len, post, idf }
    }

    /// Scores for the distinct query tokens (query term frequency ignored, as in the Python index).
    pub fn score(&self, q: &[String], allowed: Option<&[bool]>) -> HashMap<u32, f32> {
        let mut sc: HashMap<u32, f32> = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for t in q {
            if !seen.insert(t) {
                continue;
            }
            let (Some(p), Some(&idf)) = (self.post.get(t), self.idf.get(t)) else { continue };
            for &(i, f) in p {
                if let Some(a) = allowed {
                    if !a[i as usize] {
                        continue;
                    }
                }
                let f = f as f32;
                let dl = self.len[i as usize] as f32;
                *sc.entry(i).or_default() += idf * f * (self.k1 + 1.0) / (f + self.k1 * (1.0 - self.b + self.b * dl / self.avg));
            }
        }
        sc
    }
}

#[derive(Serialize, Deserialize)]
pub struct Store {
    pub chunks: Vec<Chunk>,
    pub word: Bm25,
    pub char: Bm25,
    /// Row-major, L2-normalised embeddings (chunks.len() x dim), if `embed` has been run.
    pub emb: Option<Vec<f32>>,
    pub dim: usize,
    pub emb_model: Option<String>,
    /// fingerprint of the corpus files the index was built from
    pub sig: String,
}

pub fn index_text(c: &Chunk) -> String {
    format!("{} {} {}", c.doc_title, c.heading, c.text)
}

impl Store {
    pub fn build(chunks: Vec<Chunk>) -> Self {
        let texts: Vec<String> = chunks.iter().map(index_text).collect();
        let word = Bm25::build(&texts.iter().map(|t| word_tokens(t)).collect::<Vec<_>>());
        let char = Bm25::build(&texts.iter().map(|t| char_grams(t)).collect::<Vec<_>>());
        Store { chunks, word, char, emb: None, dim: 0, emb_model: None, sig: String::new() }
    }
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        let f = std::io::BufWriter::new(std::fs::File::create(path)?);
        bincode::serialize_into(f, self)?;
        Ok(())
    }
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let f = std::io::BufReader::new(std::fs::File::open(path).map_err(|e| anyhow::anyhow!("{}: {e} (run `hr-rag index` first)", path.display()))?);
        Ok(bincode::deserialize_from(f)?)
    }
    pub fn pos(&self, id: &str) -> Option<usize> {
        self.chunks.iter().position(|c| c.id == id)
    }
}
