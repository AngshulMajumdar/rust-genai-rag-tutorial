//! Minimal blocking client for a local Ollama server (plain HTTP, no TLS needed).
use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

pub struct Ollama {
    base: String,
    agent: ureq::Agent,
}

#[derive(Clone, Debug)]
pub struct Msg {
    pub role: &'static str,
    pub content: String,
}

impl Ollama {
    pub fn new(base: &str) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(900))
            .build();
        Ollama { base: base.trim_end_matches('/').to_string(), agent }
    }

    fn post(&self, path: &str, body: Value) -> Result<Value> {
        let url = format!("{}{}", self.base, path);
        let resp = self.agent.post(&url).send_json(body).map_err(|e| match e {
            ureq::Error::Status(code, r) => anyhow!("Ollama {url} returned {code}: {}", r.into_string().unwrap_or_default()),
            other => anyhow!("cannot reach Ollama at {url}: {other} (is `ollama serve` running?)"),
        })?;
        Ok(resp.into_json()?)
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    /// Names of the locally installed models (e.g. "gemma3:4b", "bge-m3:latest").
    pub fn models(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/tags", self.base);
        let v: Value = self
            .agent
            .get(&url)
            .call()
            .map_err(|e| anyhow!("cannot reach Ollama at {url}: {e} (start the Ollama app or run `ollama serve`)"))?
            .into_json()?;
        Ok(v["models"].as_array().map(|a| a.iter().filter_map(|m| m["name"].as_str().map(String::from)).collect()).unwrap_or_default())
    }

    /// Is `model` installed? ("bge-m3" matches "bge-m3:latest")
    pub fn has_model(&self, model: &str) -> Result<bool> {
        let want = if model.contains(':') { model.to_string() } else { format!("{model}:latest") };
        Ok(self.models()?.iter().any(|m| *m == want || m == model))
    }

    /// Download `model` unless it is installed already; prints progress to stderr.
    pub fn pull(&self, model: &str) -> Result<()> {
        if self.has_model(model)? {
            eprintln!("{model}: already installed");
            return Ok(());
        }
        let url = format!("{}/api/pull", self.base);
        let resp = self.agent.post(&url).send_json(json!({"model": model, "stream": true})).map_err(|e| anyhow!("pull {model} failed: {e}"))?;
        let mut last = String::new();
        for line in std::io::BufRead::lines(std::io::BufReader::new(resp.into_reader())) {
            let v: Value = serde_json::from_str(&line?)?;
            if let Some(e) = v["error"].as_str() {
                return Err(anyhow!("pull {model}: {e}"));
            }
            let status = v["status"].as_str().unwrap_or("").to_string();
            let msg = match (v["completed"].as_u64(), v["total"].as_u64()) {
                (Some(c), Some(t)) if t > 0 => format!("{status} {}%", c * 100 / t / 10 * 10),
                _ => status,
            };
            if msg != last {
                eprintln!("{model}: {msg}");
                last = msg;
            }
        }
        if !self.has_model(model)? {
            return Err(anyhow!("pull {model} finished but the model is not listed"));
        }
        Ok(())
    }

    /// Embed a batch of texts; returns one vector per input.
    pub fn embed(&self, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let v = self.post("/api/embed", json!({"model": model, "input": inputs, "truncate": true}))?;
        let arr = v["embeddings"].as_array().context("no `embeddings` in /api/embed response")?;
        arr.iter()
            .map(|e| e.as_array().context("bad embedding").map(|x| x.iter().map(|f| f.as_f64().unwrap_or(0.0) as f32).collect()))
            .collect()
    }

    /// Chat completion. `schema` (a JSON schema) constrains the output to JSON when given.
    pub fn chat(&self, model: &str, msgs: &[Msg], schema: Option<&Value>, num_ctx: usize) -> Result<String> {
        let messages: Vec<Value> = msgs.iter().map(|m| json!({"role": m.role, "content": m.content})).collect();
        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "keep_alive": "30m",
            "options": {"temperature": 0.0, "seed": 42, "num_ctx": num_ctx}
        });
        if let Some(s) = schema {
            body["format"] = s.clone();
        }
        let v = self.post("/api/chat", body)?;
        v["message"]["content"].as_str().map(|s| s.to_string()).context("no message.content in /api/chat response")
    }
}
