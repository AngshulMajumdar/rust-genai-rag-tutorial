//! Section-aware chunking; a faithful port of `chunk_doc` in `rag/ragcore.py`,
//! so chunk ids and texts are identical to the Python store.
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const MAX_WORDS: usize = 220;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub language: String,
    pub document: String,
    pub doc_title: String,
    pub heading: String,
    pub text: String,
}

fn parse_heading(s: &str) -> Option<(usize, String)> {
    let hashes = s.chars().take_while(|&c| c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &s[hashes..];
    // Python: ^(#{1,6})\s+(.*)$  -> at least one whitespace after the hashes
    if !rest.starts_with(|c: char| c.is_whitespace()) {
        return None;
    }
    Some((hashes, rest.trim_start().to_string()))
}

pub fn chunk_doc(lang: &str, doc: &str, raw: &str) -> Vec<Chunk> {
    let mut title: Option<String> = None;
    let mut heading_path: Vec<String> = Vec::new();
    let mut blocks: Vec<(Vec<String>, String)> = Vec::new();
    let mut para: Vec<String> = Vec::new();
    let flush = |para: &mut Vec<String>, blocks: &mut Vec<(Vec<String>, String)>, hp: &Vec<String>| {
        if !para.is_empty() {
            blocks.push((hp.clone(), para.join(" ")));
            para.clear();
        }
    };
    for ln in raw.split('\n') {
        let s = ln.trim();
        if s.starts_with("<!--") {
            continue;
        }
        if let Some((lvl, txt)) = parse_heading(s) {
            flush(&mut para, &mut blocks, &heading_path);
            let txt = txt.replace("**", "").trim().to_string();
            if lvl == 2 && title.is_none() {
                title = Some(txt);
                heading_path.clear();
                continue;
            }
            if lvl == 1 {
                continue;
            }
            let keep = lvl.saturating_sub(3).min(heading_path.len());
            heading_path.truncate(keep);
            heading_path.push(txt);
            continue;
        }
        if s.is_empty() || s == "---" {
            flush(&mut para, &mut blocks, &heading_path);
            continue;
        }
        para.push(s.to_string());
    }
    flush(&mut para, &mut blocks, &heading_path);

    let mut out_raw: Vec<(String, String)> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut cur_h: Option<Vec<String>> = None;
    let mut cur_n = 0usize;
    for (h, p) in blocks {
        let n = p.split_whitespace().count();
        if cur_h.as_ref() != Some(&h) || cur_n + n > MAX_WORDS {
            if !cur.is_empty() {
                let hd = cur_h.as_ref().map(|v| v.join(" > ")).unwrap_or_default();
                out_raw.push((hd, cur.join("\n")));
            }
            cur.clear();
            cur_n = 0;
            cur_h = Some(h);
        }
        cur.push(p);
        cur_n += n;
    }
    if !cur.is_empty() {
        let hd = cur_h.as_ref().map(|v| v.join(" > ")).unwrap_or_default();
        out_raw.push((hd, cur.join("\n")));
    }
    let prefix: String = lang.chars().take(2).collect::<String>().to_uppercase();
    let title = title.unwrap_or_else(|| doc.to_string());
    out_raw
        .into_iter()
        .enumerate()
        .map(|(i, (heading, text))| Chunk {
            id: format!("{prefix}:{doc}:{i:04}"),
            language: lang.to_string(),
            document: doc.to_string(),
            doc_title: title.clone(),
            heading,
            text,
        })
        .collect()
}

/// Chunk every `<corpus>/<Language>/NN_*.md`, in the same order as the Python builder.
pub fn chunk_corpus(corpus: &Path) -> anyhow::Result<Vec<Chunk>> {
    let mut paths = Vec::new();
    for lang in std::fs::read_dir(corpus)? {
        let lang = lang?.path();
        if !lang.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&lang)? {
            let f = f?.path();
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            let b = name.as_bytes();
            if name.ends_with(".md") && b.len() > 3 && b[0].is_ascii_digit() && b[1].is_ascii_digit() && b[2] == b'_' {
                paths.push(f);
            }
        }
    }
    paths.sort();
    let mut chunks = Vec::new();
    for p in paths {
        let lang = p.parent().unwrap().file_name().unwrap().to_string_lossy().to_string();
        let doc = p.file_stem().unwrap().to_string_lossy().to_string();
        let raw = std::fs::read_to_string(&p)?;
        chunks.extend(chunk_doc(&lang, &doc, &raw));
    }
    Ok(chunks)
}
