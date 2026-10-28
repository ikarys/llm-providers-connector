//! Port of openfox `llm/client.ts`: one streaming call to an
//! OpenAI-compatible `/chat/completions` endpoint.
//!
//! Not ported yet: the provider shims of openfox `requestBody`
//! (llama.cpp `chat_template_kwargs.thinking`, Qwen3 `enable_thinking`);
//! the standard `reasoning_effort` field is sent when set.

use super::endpoint::Endpoint;
use super::params::ReasoningEffort;
use serde_json::{Value, json};
use std::io::BufReader;
use std::sync::atomic::AtomicBool;

/// What the stream yields while it runs (openfox `StreamEvent`).
#[derive(Debug, PartialEq)]
pub enum Delta {
    Text(String),
    Thinking(String),
    /// A tool call's name is known before its arguments finish streaming.
    ToolPreparing {
        index: usize,
        name: String,
    },
}

/// The request for one streaming chat completion.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest<'a> {
    pub model: &'a str,
    pub system_prompt: &'a str,
    pub messages: &'a [Value],
    pub tools: &'a Value,
    pub max_tokens: u32,
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// A tool call as streamed, before fennec's policy parses its arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamedToolCall {
    /// May be empty; the caller decides what to do about it.
    pub id: String,
    pub name: String,
    /// The raw accumulated argument text; no JSON parsing here.
    pub arguments: String,
}

#[derive(Debug, Default, PartialEq)]
pub struct Completion {
    pub content: String,
    pub thinking: String,
    pub tool_calls: Vec<StreamedToolCall>,
    pub finish_reason: String,
    pub prompt_tokens: u64,
    /// The part of the prompt served from the server's cache.
    pub cached_tokens: u64,
    pub completion_tokens: u64,
}

/// Streams one chat completion, forwarding deltas to `on` and returning the
/// accumulated completion.
pub fn stream(
    endpoint: &Endpoint,
    req: &ChatRequest,
    stop: &AtomicBool,
    on: &mut dyn FnMut(Delta),
) -> Result<Completion, String> {
    let mut messages = vec![json!({"role": "system", "content": req.system_prompt})];
    messages.extend_from_slice(req.messages);
    let mut body = json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if let Some(effort) = req.reasoning_effort {
        body["reasoning_effort"] = json!(effort);
    }
    if req.tools.as_array().is_some_and(|t| !t.is_empty()) {
        body["tools"] = req.tools.clone();
        body["tool_choice"] = json!("auto");
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into();
    let url = endpoint.url("chat/completions");
    let mut call = agent.post(&url).header("Content-Type", "application/json");
    if let Some(key) = &endpoint.api_key {
        call = call.header("Authorization", &format!("Bearer {key}"));
    }
    let resp = call.send_json(&body).map_err(|e| e.to_string())?;
    let status = resp.status();
    let mut body = resp.into_body();
    if !status.is_success() {
        let text = body.read_to_string().unwrap_or_default();
        return Err(format!("LLM HTTP {status}: {text}"));
    }
    super::sse::parse(BufReader::new(body.into_reader()), stop, on)
}
