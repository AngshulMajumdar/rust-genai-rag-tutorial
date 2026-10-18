//! Text normalisation and tokenisation shared by indexing, retrieval and scoring.
//! Mirrors `rag/ragcore.py` (after the Indic tokenizer fix).

/// Zero code points of the Indic digit blocks handled (Devanagari .. Kannada).
const DIGIT_BLOCKS: [u32; 8] = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6];

/// Map any Indic digit to its ASCII digit.
pub fn ascii_digit(c: char) -> char {
    let u = c as u32;
    for b in DIGIT_BLOCKS {
        if u >= b && u < b + 10 {
            return char::from_u32('0' as u32 + (u - b)).unwrap();
        }
    }
    c
}

/// Digits to ASCII, strip markdown emphasis (`**`, `__`, backticks), lowercase.
pub fn norm(s: &str) -> String {
    let s: String = s.chars().map(ascii_digit).collect();
    s.replace("**", "").replace("__", "").replace('`', "").to_lowercase()
}

fn is_word_char(c: char) -> bool {
    let u = c as u32;
    c.is_alphanumeric()
        || c == '_'
        || (0x0900..=0x0963).contains(&u) // Devanagari letters + vowel signs (danda excluded)
        || (0x0966..=0x0D7F).contains(&u) // Bengali .. Malayalam blocks, incl. matras
        || c == '\u{200C}'
        || c == '\u{200D}'
}

/// Word tokens: maximal runs of word characters (Indic words kept whole).
pub fn word_tokens(s: &str) -> Vec<String> {
    let n = norm(s);
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in n.chars() {
        if is_word_char(c) {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Character n-grams (n=4) over `#word#`, as in the Python index.
pub fn char_grams(s: &str) -> Vec<String> {
    let n = 4;
    let mut out = Vec::new();
    for w in word_tokens(s) {
        let cs: Vec<char> = format!("#{w}#").chars().collect();
        if cs.len() <= n {
            out.push(cs.iter().collect());
        } else {
            for i in 0..=cs.len() - n {
                out.push(cs[i..i + n].iter().collect());
            }
        }
    }
    out
}

/// Remove all whitespace after `norm` (used for verbatim evidence matching).
pub fn squash(s: &str) -> String {
    norm(s).chars().filter(|c| !c.is_whitespace()).collect()
}

/// Split into sentences after । ॥ . ? ! followed by whitespace.
pub fn sentences(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        if matches!(c, '।' | '॥' | '.' | '?' | '!') && i + 1 < chars.len() && chars[i + 1].is_whitespace() {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() {
                out.push(std::mem::take(&mut cur));
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// All numbers (as f64) in a string, Indic digits normalised, thousands commas removed.
pub fn numbers(s: &str) -> Vec<f64> {
    let t: String = s.chars().map(ascii_digit).filter(|&c| c != ',').collect();
    let cs: Vec<char> = t.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        if cs[i].is_ascii_digit() {
            let st = i;
            while i < cs.len() && cs[i].is_ascii_digit() {
                i += 1;
            }
            if i + 1 < cs.len() && cs[i] == '.' && cs[i + 1].is_ascii_digit() {
                i += 1;
                while i < cs.len() && cs[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let tok: String = cs[st..i].iter().collect();
            if let Ok(v) = tok.parse::<f64>() {
                out.push(v);
            }
        } else {
            i += 1;
        }
    }
    out
}

/// Detect the dominant Indic script of a string and return the language name.
pub fn detect_language(s: &str) -> &'static str {
    let mut counts = [0usize; 8];
    for c in s.chars() {
        let u = c as u32;
        let idx = match u {
            0x0900..=0x097F => 0, // Devanagari: Hindi/Marathi (disambiguated below)
            0x0980..=0x09FF => 1,
            0x0A00..=0x0A7F => 2,
            0x0B00..=0x0B7F => 3,
            0x0B80..=0x0BFF => 4,
            0x0C00..=0x0C7F => 5,
            0x0C80..=0x0CFF => 6,
            _ => continue,
        };
        counts[idx] += 1;
    }
    let (best, n) = counts.iter().enumerate().max_by_key(|x| x.1).unwrap();
    if *n == 0 {
        return "English";
    }
    match best {
        0 => {
            // Marathi-specific markers
            if s.contains("आहे") || s.contains("च्या") || s.contains("ळ") { "Marathi" } else { "Hindi" }
        }
        1 => "Bengali",
        2 => "Punjabi",
        3 => "Odia",
        4 => "Tamil",
        5 => "Telugu",
        _ => "Kannada",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tokens_keep_matras() {
        assert_eq!(word_tokens("परिवीक्षा अवधि। মোট ১১২টি পদ"), vec!["परिवीक्षा", "अवधि", "মোট", "112টি", "পদ"]);
    }
    #[test]
    fn nums() {
        assert_eq!(numbers("₹১,২০,০০০ and 4.5 दिन"), vec![120000.0, 4.5]);
    }
    #[test]
    fn sent() {
        assert_eq!(sentences("एक। दो. तीन").len(), 3);
    }
}
