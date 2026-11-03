//! Reads an SSE chat-completion stream, accumulating tool-call fragments
//! by index as the OpenAI protocol sends them.

use super::chat::{Completion, Delta, StreamedToolCall};
use serde_json::Value;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The longest a healthy stream goes without a line after it has started
/// answering: sub-second gaps while decoding. A longer silence means the
/// connection is dead, so it is cut here — and the model-gate slot freed. A
/// total budget would cut a slow-but-alive generation; this idle one does not.
const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

/// The longest a healthy stream may take to produce its first delta (a token,
/// usage, or finish chunk). A long prefill on a big context can take minutes,
/// so this is much larger than `IDLE_TIMEOUT`; a longer silence means the
/// server never started answering.
const FIRST_TOKEN_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// How long a receive slice is: short enough to notice `stop` promptly, long
/// enough to avoid busy-waiting.
const POLL: Duration = Duration::from_millis(200);

#[derive(Default)]
struct PendingCall {
    id: String,
    name: String,
    arguments: String,
}

/// Parses an SSE chat-completion stream, accumulating tool-call fragments
/// by index as the OpenAI protocol sends them.
pub(crate) fn parse(
    reader: impl BufRead + Send + 'static,
    stop: &AtomicBool,
    on: &mut dyn FnMut(Delta),
) -> Result<Completion, String> {
    parse_with_idle_timeout(reader, stop, on, FIRST_TOKEN_TIMEOUT, IDLE_TIMEOUT)
}

fn parse_with_idle_timeout<R>(
    reader: R,
    stop: &AtomicBool,
    on: &mut dyn FnMut(Delta),
    first_token: Duration,
    idle: Duration,
) -> Result<Completion, String>
where
    R: BufRead + Send + 'static,
{
    // A dead connection blocks the reader forever. A reader thread plus a
    // bounded receive turns that block into a timeout: before the first delta
    // the limit is `first_token`, after it `idle`, reset by every received
    // line, so a slow stream survives.
    // ponytail: on a timeout the reader thread stays blocked on the socket
    // until the server closes it or sends a line (then its send fails and it
    // exits); ureq gives no handle to close the socket.
    let (tx, rx) = mpsc::channel::<Result<String, String>>();
    std::thread::spawn(move || {
        for line in reader.lines() {
            match line {
                Ok(l) => {
                    if tx.send(Ok(l)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(format!("stream read failed: {e}")));
                    return;
                }
            }
        }
        // EOF: the channel closes; the receive below sees Disconnected.
    });

    let mut out = Completion::default();
    let mut calls: Vec<PendingCall> = Vec::new();
    let mut first_token_seen = false;
    let mut last_line = Instant::now();
    loop {
        let line = match rx.recv_timeout(POLL) {
            Ok(Ok(line)) => {
                last_line = Instant::now();
                line
            }
            Ok(Err(e)) => return Err(e),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::Relaxed) {
                    return Err("Aborted".into());
                }
                let limit = if first_token_seen { idle } else { first_token };
                if last_line.elapsed() >= limit {
                    return if first_token_seen {
                        Err(format!(
                            "stream idle for {} s: the server stopped sending",
                            idle.as_secs()
                        ))
                    } else {
                        Err(format!(
                            "no token after {} s: the server never started answering",
                            first_token.as_secs()
                        ))
                    };
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        if stop.load(Ordering::Relaxed) {
            return Err("Aborted".into());
        }
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
            first_token_seen = true;
        }
        let Some(choice) = chunk["choices"].get(0) else {
            continue;
        };
        if let Some(reason) = choice["finish_reason"].as_str() {
            out.finish_reason = reason.to_string();
            first_token_seen = true;
        }
        let delta = &choice["delta"];
        // Same field precedence as the thinking delta below.
        let thinking = ["reasoning", "reasoning_content", "thinking"]
            .iter()
            .find_map(|k| delta[k].as_str());
        if let Some(t) = thinking.filter(|t| !t.is_empty()) {
            out.thinking.push_str(t);
            on(Delta::Thinking(t.to_string()));
            first_token_seen = true;
        }
        if let Some(t) = delta["content"].as_str().filter(|t| !t.is_empty()) {
            out.content.push_str(t);
            on(Delta::Text(t.to_string()));
            first_token_seen = true;
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
                first_token_seen = true;
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
    use std::io::{BufReader, Cursor, Read};

    /// Yields `first` once if present, then blocks on a live-but-silent
    /// channel: a connection that delivered data and then went quiet, or one
    /// that is silent from the start.
    struct ThenBlocks {
        first: Option<Vec<u8>>,
        rx: mpsc::Receiver<Vec<u8>>,
    }

    impl Read for ThenBlocks {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let chunk = match self.first.take() {
                Some(c) => c,
                None => self
                    .rx
                    .recv()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::BrokenPipe, e))?,
            };
            let n = chunk.len().min(buf.len());
            buf[..n].copy_from_slice(&chunk[..n]);
            Ok(n)
        }
    }

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
        let c = parse(
            Cursor::new(stream.as_bytes().to_vec()),
            &AtomicBool::new(false),
            &mut |d| deltas.push(d),
        )
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

    #[test]
    fn a_stream_that_goes_silent_is_cut_by_the_idle_timeout() {
        let first = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n";
        // The sender stays alive (dropped at the end) so the reader blocks
        // after `first` rather than seeing a closed channel.
        let (keep_sender, rx) = mpsc::channel::<Vec<u8>>();
        let reader = BufReader::new(ThenBlocks {
            first: Some(first.to_vec()),
            rx,
        });
        let stop = AtomicBool::new(false);
        let mut deltas = Vec::new();
        let err = parse_with_idle_timeout(
            reader,
            &stop,
            &mut |d| deltas.push(d),
            Duration::from_millis(50),
            Duration::from_millis(50),
        )
        .unwrap_err();
        assert!(err.contains("idle"));
        // The one token before the silence was delivered.
        assert_eq!(deltas.len(), 1);
        drop(keep_sender);
    }

    #[test]
    fn a_silent_stream_is_cut_by_the_first_token_budget_not_the_idle_budget() {
        // The sender stays alive so the reader blocks from the start.
        let (keep_sender, rx) = mpsc::channel::<Vec<u8>>();
        let reader = BufReader::new(ThenBlocks { first: None, rx });
        let stop = AtomicBool::new(false);
        let mut deltas = Vec::new();
        let err = parse_with_idle_timeout(
            reader,
            &stop,
            &mut |d| deltas.push(d),
            Duration::from_millis(150),
            Duration::from_millis(30),
        )
        .unwrap_err();
        assert!(err.contains("no token"));
        assert!(deltas.is_empty());
        drop(keep_sender);
    }

    #[test]
    fn stop_while_silent_aborts_before_the_budgets() {
        let (keep_sender, rx) = mpsc::channel::<Vec<u8>>();
        let reader = BufReader::new(ThenBlocks { first: None, rx });
        let stop = AtomicBool::new(false);
        let mut deltas = Vec::new();
        let started = Instant::now();
        let err = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(50));
                stop.store(true, Ordering::Relaxed);
            });
            parse_with_idle_timeout(
                reader,
                &stop,
                &mut |d| deltas.push(d),
                Duration::from_secs(30),
                Duration::from_secs(30),
            )
            .unwrap_err()
        });
        assert_eq!(err, "Aborted");
        assert!(deltas.is_empty());
        assert!(started.elapsed() < Duration::from_secs(1));
        drop(keep_sender);
    }
}
