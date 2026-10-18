//! Choose the data with the system file window (Windows: the standard Explorer "Open" dialog), or by path.
//! Accepts the downloaded .zip (extracted next to itself, once), the extracted folder, or any file inside it.
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct DataPaths {
    /// folder whose sub-folders (one per language) hold NN_<topic>.md documents
    pub corpus: PathBuf,
    /// test-set JSON, if one was found or chosen
    pub testset: Option<PathBuf>,
    /// folder for the index and run files (next to the corpus)
    pub work: PathBuf,
}

const SKIP: [&str; 6] = ["hr_rag_work", "target", "node_modules", "__MACOSX", "AppData", "$RECYCLE.BIN"];
const MAX_DIRS: usize = 4000;

fn home() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from)
}

/// Trim spaces and quotes (Windows "Copy as path" adds them), expand ~, and resolve a bare name such as
/// "Downloads" against the current folder, the home folder and the usual user folders.
pub fn clean(input: &str) -> PathBuf {
    let s = input.trim().trim_matches(|c| c == '"' || c == '\'').trim();
    if s.is_empty() {
        return PathBuf::new();
    }
    let p = match (s.strip_prefix('~'), home()) {
        (Some(rest), Some(h)) => h.join(rest.trim_start_matches(['/', '\\'])),
        _ => PathBuf::from(s),
    };
    if p.is_absolute() || p.exists() {
        return p;
    }
    if let Some(h) = home() {
        for base in [h.clone(), h.join("Downloads"), h.join("Desktop"), h.join("Documents"), h.join("OneDrive"), h.join("OneDrive").join("Desktop"), h.join("OneDrive").join("Documents")] {
            if base.join(&p).exists() {
                return base.join(&p);
            }
        }
    }
    p
}

fn is_doc(name: &str) -> bool {
    let b = name.as_bytes();
    name.ends_with(".md") && b.len() > 5 && b[0].is_ascii_digit() && b[1].is_ascii_digit() && b[2] == b'_'
}

fn subdirs(d: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    v.retain(|p| {
        let n = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        !n.starts_with('.') && !SKIP.contains(&n.as_str())
    });
    v.sort();
    v
}

/// Number of NN_*.md documents in the language sub-folders of `d`.
fn corpus_docs(d: &Path) -> usize {
    subdirs(d)
        .iter()
        .map(|l| std::fs::read_dir(l).map(|r| r.filter_map(|e| e.ok()).filter(|e| is_doc(&e.file_name().to_string_lossy())).count()).unwrap_or(0))
        .sum()
}

/// Folders under `root` (breadth-first, at most `depth` levels and MAX_DIRS folders).
fn walk(root: &Path, depth: usize) -> Vec<PathBuf> {
    let mut out = vec![root.to_path_buf()];
    let mut level = vec![root.to_path_buf()];
    for _ in 0..depth {
        let mut next = Vec::new();
        for d in &level {
            for s in subdirs(d) {
                if out.len() >= MAX_DIRS {
                    return out;
                }
                out.push(s.clone());
                next.push(s);
            }
        }
        level = next;
    }
    out
}

fn find_corpus(root: &Path) -> Option<PathBuf> {
    walk(root, 5)
        .into_iter()
        .map(|d| (corpus_docs(&d), d))
        .filter(|(n, _)| *n > 0)
        .max_by_key(|(n, d)| (*n, d.file_name().map_or(false, |x| x == "corpus"), std::cmp::Reverse(d.components().count())))
        .map(|(_, d)| d)
}

fn find_testset(root: &Path) -> Option<PathBuf> {
    let mut c: Vec<PathBuf> = walk(root, 4)
        .iter()
        .flat_map(|d| std::fs::read_dir(d).into_iter().flatten().filter_map(|e| e.ok()).map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            p.is_file() && n.ends_with(".json") && (n.contains("testset") || n.contains("test_set"))
        })
        .filter(|p| crate::eval::load_testset(p).map(|v| !v.is_empty()).unwrap_or(false))
        .collect();
    c.sort();
    c.into_iter().next()
}

/// Does this zip contain language folders with NN_*.md documents?
fn zip_has_corpus(zip: &Path) -> bool {
    let Ok(f) = std::fs::File::open(zip) else { return false };
    let Ok(a) = zip::ZipArchive::new(std::io::BufReader::new(f)) else { return false };
    let found = a.file_names().any(|n| {
        let parts: Vec<&str> = n.split(['/', '\\']).collect();
        parts.len() >= 2 && is_doc(parts[parts.len() - 1])
    });
    found
}

/// Extract `zip` into a folder named after it (next to it), unless that was done already.
pub fn unzip(zip: &Path) -> Result<PathBuf> {
    let stem = zip.file_stem().context("bad zip name")?.to_string_lossy().to_string();
    let dest = zip.parent().unwrap_or(Path::new(".")).join(&stem);
    if dest.is_dir() && find_corpus(&dest).is_some() {
        println!("using the copy already extracted in {}", dest.display());
        return Ok(dest);
    }
    if !zip_has_corpus(zip) {
        bail!("{} does not contain the HR documents (language folders with files like 01_recruitment.md)", zip.display());
    }
    let f = std::fs::File::open(zip).with_context(|| format!("cannot open {}", zip.display()))?;
    let mut a = zip::ZipArchive::new(std::io::BufReader::new(f)).with_context(|| format!("{} is not a valid zip file", zip.display()))?;
    // a zip whose entries all sit in one top folder named like the zip is extracted beside it, not one level deeper
    let prefix = format!("{stem}/");
    let self_contained = a.file_names().all(|n| n.replace('\\', "/").starts_with(&prefix));
    let target = if self_contained { zip.parent().unwrap_or(Path::new(".")).to_path_buf() } else { dest.clone() };
    println!("extracting {} -> {}", zip.display(), dest.display());
    a.extract(&target).with_context(|| format!("cannot extract into {}", target.display()))?;
    Ok(dest)
}

/// Zips directly inside `dir` that contain the documents (names with "suryodaya" first, then newest).
fn zips_in(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<(bool, std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().map_or(false, |e| e.eq_ignore_ascii_case("zip")))
        .filter(|p| zip_has_corpus(p))
        .map(|p| {
            let named = p.file_name().map_or(false, |n| n.to_string_lossy().to_lowercase().contains("suryodaya"));
            let t = p.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
            (named, t, p)
        })
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
    v.into_iter().map(|x| x.2).collect()
}

fn finish(corpus: PathBuf) -> Result<DataPaths> {
    let base = corpus.parent().unwrap_or(&corpus).to_path_buf();
    // the test set is usually beside the corpus folder (data/testset) or one level further up
    let testset = find_testset(&base).or_else(|| base.parent().and_then(find_testset));
    let work = base.join("hr_rag_work");
    std::fs::create_dir_all(&work).with_context(|| format!("cannot write to {}", work.display()))?;
    Ok(DataPaths { corpus, testset, work })
}

/// Resolve a zip, a folder, or any file inside the data folder to the corpus, test set and work folder.
pub fn locate_path(p: &Path) -> Result<DataPaths> {
    if p.as_os_str().is_empty() {
        bail!("no path given");
    }
    if !p.exists() {
        bail!("{} does not exist", p.display());
    }
    let p = crate::session::plain(std::fs::canonicalize(p)?);
    if p.is_file() && p.extension().map_or(false, |e| e.eq_ignore_ascii_case("zip")) {
        let root = unzip(&p)?;
        return finish(find_corpus(&root).context("the extracted zip has no documents")?);
    }
    let dir = if p.is_file() { p.parent().unwrap().to_path_buf() } else { p.clone() };
    // the folder itself (or something below it) holds the documents
    if let Some(c) = find_corpus(&dir) {
        return finish(c);
    }
    // a file in the folder that holds the downloaded zip (e.g. Downloads)
    if let Some(z) = zips_in(&dir).into_iter().next() {
        println!("using {}", z.display());
        let root = unzip(&z)?;
        return finish(find_corpus(&root).context("the extracted zip has no documents")?);
    }
    // a file deeper inside the extracted data (e.g. rust/README.md, or a document itself): go up through the
    // repository's own layout only, never into unrelated sibling folders
    for a in dir.ancestors().skip(1).take(3) {
        for c in [a.to_path_buf(), a.join("corpus"), a.join("data").join("corpus")] {
            if corpus_docs(&c) > 0 {
                return finish(c);
            }
        }
    }
    bail!("no HR documents found at {} (expected the data zip, or a folder with language sub-folders holding files like 01_recruitment.md)", p.display())
}

pub fn locate(input: &str) -> Result<DataPaths> {
    locate_path(&clean(input))
}

// ---------------------------------------------------------------- native file window

pub fn utf16_base64(s: &str) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = s.encode_utf16().flat_map(|u| u.to_le_bytes()).collect();
    let mut out = String::new();
    for ch in bytes.chunks(3) {
        let n = (ch[0] as u32) << 16 | (*ch.get(1).unwrap_or(&0) as u32) << 8 | *ch.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= ch.len() {
                out.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// PowerShell script for the standard Windows "Open" window, kept on top of other windows.
pub fn dialog_script(title: &str, filter: &str, start: Option<&Path>) -> String {
    let q = |s: &str| s.replace('\'', "''");
    let start = match start {
        Some(p) => format!("$d.InitialDirectory = '{}'", q(&p.display().to_string())),
        None => "try { $d.InitialDirectory = (New-Object -ComObject Shell.Application).NameSpace('shell:Downloads').Self.Path } catch {}".into(),
    };
    format!(
        "Add-Type -AssemblyName System.Windows.Forms\n\
         [System.Windows.Forms.Application]::EnableVisualStyles()\n\
         [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false\n\
         $d = New-Object System.Windows.Forms.OpenFileDialog\n\
         $d.Title = '{}'\n\
         $d.Filter = '{}'\n\
         $d.Multiselect = $false\n\
         {start}\n\
         $owner = New-Object System.Windows.Forms.Form -Property @{{TopMost = $true; ShowInTaskbar = $false}}\n\
         if ($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Out.Write($d.FileName) }}\n\
         $owner.Dispose()\n",
        q(title),
        q(filter)
    )
}

/// Open the system file window. Ok(None) = cancelled; Err = no file window available here.
pub fn pick_file(title: &str, filter: &str, start: Option<&Path>) -> Result<Option<PathBuf>> {
    let out = if cfg!(windows) {
        std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-STA", "-ExecutionPolicy", "Bypass", "-EncodedCommand"])
            .arg(utf16_base64(&dialog_script(title, filter, start)))
            .output()
            .context("cannot start PowerShell to open the file window")?
    } else {
        let mut r = std::process::Command::new("zenity").args(["--file-selection", &format!("--title={title}")]).output();
        if r.is_err() {
            r = std::process::Command::new("kdialog").args(["--getopenfilename", ".", "--title", title]).output();
        }
        r.context("no file window available (zenity/kdialog not installed)")?
    };
    let s = String::from_utf8_lossy(&out.stdout).trim_start_matches('\u{feff}').trim().to_string();
    if s.is_empty() {
        if !out.status.success() && !out.stderr.is_empty() && cfg!(windows) {
            bail!("file window failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        return Ok(None);
    }
    Ok(Some(PathBuf::from(s)))
}

fn report(d: &DataPaths) {
    println!("corpus   {}", d.corpus.display());
    println!("test set {}", d.testset.as_ref().map_or("(none: chat only)".to_string(), |t| t.display().to_string()));
    println!("work     {}", d.work.display());
}

/// Choose the data in the system file window: the downloaded .zip, or any file inside the extracted folder.
/// Falls back to typing a path (with `read`, e.g. evcxr_input::get_string) only if no file window can be shown.
pub fn choose(read: impl Fn(&str) -> String) -> Result<DataPaths> {
    let title = "Choose the HR data: the downloaded .zip, or any file inside the extracted folder";
    let filter = "Data zip (*.zip)|*.zip|All files (*.*)|*.*";
    println!("A file window has opened: choose the data zip (or any file inside the extracted data folder).");
    let mut start: Option<PathBuf> = None;
    for _ in 0..3 {
        match pick_file(title, filter, start.as_deref()) {
            Ok(Some(p)) => match locate_path(&p) {
                Ok(mut d) => {
                    if d.testset.is_none() {
                        println!("No test set next to the documents. Choose the test-set .json, or Cancel for chat only.");
                        if let Ok(Some(t)) = pick_file("Choose the test-set .json (Cancel = chat only)", "Test set (*.json)|*.json", d.corpus.parent()) {
                            match crate::eval::load_testset(&t) {
                                Ok(v) if !v.is_empty() => d.testset = Some(t),
                                _ => println!("{} is not a test set; continuing without one", t.display()),
                            }
                        }
                    }
                    report(&d);
                    return Ok(d);
                }
                Err(e) => {
                    println!("{e:#}\nChoose again.");
                    start = p.parent().map(|x| x.to_path_buf());
                }
            },
            Ok(None) => bail!("nothing chosen; run this cell again to open the file window"),
            Err(e) => {
                println!("{e:#}; type the path instead.");
                return ask(read);
            }
        }
    }
    bail!("no usable data chosen; run this cell again")
}

/// Text fallback: ask for a path until a usable one is given; up to 5 tries.
pub fn ask(read: impl Fn(&str) -> String) -> Result<DataPaths> {
    for _ in 0..5 {
        let input = read("Path to the data zip or folder (a bare name like Downloads is looked up in your user folders): ");
        match locate(&input) {
            Ok(mut d) => {
                if d.testset.is_none() {
                    let t = clean(&read("No test set found. Path to the test-set .json (Enter = none, chat only): "));
                    if !t.as_os_str().is_empty() {
                        match crate::eval::load_testset(&t) {
                            Ok(v) if !v.is_empty() => d.testset = Some(t),
                            _ => println!("could not read a test set from {}; continuing without one", t.display()),
                        }
                    }
                }
                report(&d);
                return Ok(d);
            }
            Err(e) => println!("{e:#}. Try again."),
        }
    }
    bail!("no usable data location given")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn base64_utf16() {
        // "ab" in UTF-16LE = 61 00 62 00
        assert_eq!(utf16_base64("ab"), "YQBiAA==");
        assert_eq!(utf16_base64("a"), "YQA=");
        assert_eq!(utf16_base64("abc"), "YQBiAGMA");
    }
}
