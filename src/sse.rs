//! Port of openfox `llm/streaming.ts`: reads an SSE chat-completion stream,
//! accumulating tool-call fragments by index as the OpenAI protocol sends them.

use super::chat::{Completion, Delta, StreamedToolCall};
use serde_json::Value;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Default)]
struct PendingCall {
    id: String,
    name: String,
    arguments: String,
}

/// Parses an SSE chat-completion stream, accumulating tool-call fragments
/// by index as the OpenAI protocol sends them.
pub(crate) fn parse(
    reader: impl BufRead,
    stop: &AtomicBool,
    on: &mut dyn FnMut(Delta),
) -> Result<Completion, String> {
    let mut out = Completion::default();
    let mut calls: Vec<PendingCall> = Vec::new();
    for line in reader.lines() {
        if stop.load(Ordering::Relaxed) {
            return Err("Aborted".into());
        }
        let line = line.map_err(|e| format!("stream read failed: {e}"))?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let chunk: Value = serde_json::from_str(data).map_err(|e| format!("bad SSE chunk: {e}"))?;
        if let Some(err) = chunk.get("error") {
            return Err(format!("LLM error: {err}"));
        }
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            out.prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
            out.completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
            out.cached_tokens = usage["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0);
        }
        let Some(choice) = chunk["choices"].get(0) else {
            continue;
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            out.finish_reason = reason.to_string();
        }
        let delta = &choice["delta"];
        // Same field precedence as openfox `getThinking`.
        let thinking = ["reasoning", "reasoning_content", "thinking"]
            .iter()
            .find_map(|k| delta[k].as_str());
        if let Some(t) = thinking.filter(|t| !t.is_empty()) {
            out.thinking.push_str(t);
            on(Delta::Thinking(t.to_string()));
        }
        if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            out.content.push_str(t);
            on(Delta::Text(t.to_string()));
        }
        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            let index = tc["index"].as_u64().unwrap_or(0) as usize;
            if calls.len() <= index {
                calls.resize_with(index + 1, PendingCall::default);
            }
            let call = &mut calls[index];
            if let Some(id) = tc["id"].as_str() {
                call.id = id.to_string();
            }
            if let Some(name) = tc["function"]["name"].as_str().filter(|n| !n.is_empty()) {
                call.name.push_str(name);
                on(Delta::ToolPreparing {
                    index,
                    name: call.name.clone(),
                });
            }
            if let Some(args) = tc["function"]["arguments"].as_str() {
                call.arguments.push_str(args);
            }
        }
    }
    out.tool_calls = calls
        .into_iter()
        .filter(|c| !c.name.is_empty())
        .map(|c| StreamedToolCall {
            id: c.id,
            name: c.name,
            arguments: c.arguments,
        })
        .collect();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_accumulates_text_thinking_and_fragmented_tool_calls() {
        let stream = r#"data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}

data: {"choices":[{"delta":{"content":"Hi"}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\"pa"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a\"}"}}]},"finish_reason":"tool_calls"}]}
data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":8}}}
data: [DONE]
"#;
        let mut deltas = Vec::new();
        let c = parse(stream.as_bytes(), &AtomicBool::new(false), &mut |d| {
            deltas.push(d)
        })
        .unwrap();
        assert_eq!(c.content, "Hi");
        assert_eq!(c.thinking, "hmm");
        assert_eq!(c.finish_reason, "tool_calls");
        assert_eq!(
            (c.prompt_tokens, c.cached_tokens, c.completion_tokens),
            (10, 8, 5)
        );
        assert_eq!(c.tool_calls.len(), 1);
        assert_eq!(c.tool_calls[0].id, "c1");
        assert_eq!(c.tool_calls[0].arguments, "{\"path\":\"a\"}");
        assert_eq!(deltas.len(), 3);
    }
}
