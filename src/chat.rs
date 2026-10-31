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
use std::time::Duration;

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

/// The request body. Its bytes are the server's prompt cache key: the same
/// request must serialize identically from one fennec version to the next.
fn body(req: &ChatRequest) -> Value {
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
    body
}

/// Streams one chat completion, forwarding deltas to `on` and returning the
/// accumulated completion.
pub fn stream(
    endpoint: &Endpoint,
    req: &ChatRequest,
    stop: &AtomicBool,
    on: &mut dyn FnMut(Delta),
) -> Result<Completion, String> {
    let body = body(req);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        // An endpoint that is down otherwise hangs each attempt for the OS's
        // TCP timeout (about two minutes) before the turn can report it.
        .timeout_connect(Some(Duration::from_secs(10)))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact bytes sent today. A change here voids every session's prompt
    /// cache on the next call: update it only on purpose.
    #[test]
    fn the_request_body_bytes_are_stable() {
        let messages = [json!({"role": "user", "content": "hi"})];
        let tools = json!([{"type": "function", "function": {"name": "read_file"}}]);
        let req = ChatRequest {
            model: "qwen3.8-27b",
            system_prompt: "You are fennec.",
            messages: &messages,
            tools: &tools,
            max_tokens: 32_000,
            reasoning_effort: Some(ReasoningEffort::XHigh),
        };
        assert_eq!(
            serde_json::to_string(&body(&req)).unwrap(),
            r#"{"max_tokens":32000,"messages":[{"content":"You are fennec.","role":"system"},{"content":"hi","role":"user"}],"model":"qwen3.8-27b","reasoning_effort":"xhigh","stream":true,"stream_options":{"include_usage":true},"tool_choice":"auto","tools":[{"function":{"name":"read_file"},"type":"function"}]}"#
        );
    }
}
