//! Auto-config for an endpoint that does not list its models' limits
//! (no `/models` route, no `max_model_len`): which reasoning efforts the
//! model accepts, and the limits it names when refusing an oversized
//! `max_tokens` (read from error wording, so indicative).
//! The effort probes ask for `max_tokens: 1`, not 0, which some gateways
//! refuse.
//! This module does not handle thinking/non-thinking param combos,
//! rejected-param detection, reasoning in history, or the ollama/llama.cpp/LM
//! Studio native routes: its request has no per-model extra params to put
//! them in yet.

use super::endpoint::Endpoint;
use super::params::ReasoningEffort::{self, *};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Duration;

/// The efforts tried in turn ('none' is the non-thinking path).
const EFFORT_CANDIDATES: [ReasoningEffort; 7] = [None, Minimal, Low, Medium, High, XHigh, Max];

/// Asked for in the oversized call: far above any model's limit, so the
/// endpoint refuses it and, often, says what the limit is.
const OVERSIZED_MAX_TOKENS: u64 = 1_000_000_000;

/// Below this, a number in an error message is not a token limit.
const MIN_LIMIT: u64 = 1024;

/// What a probe learned about a model. `None`: nothing said it.
#[derive(Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbedModel {
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
    /// The efforts the endpoint accepts, distinct ones only; empty when it
    /// refused every probe.
    pub reasoning_efforts: Vec<ReasoningEffort>,
    /// The effort to run with: `high` when accepted, else the strongest.
    pub default_reasoning_effort: Option<ReasoningEffort>,
}

/// Probes one `model`. Fails only when the
/// model cannot be called at all.
pub fn probe_model(endpoint: &Endpoint, model: &str) -> Result<ProbedModel, String> {
    call(endpoint, model, 1, Option::None)?.map_err(|text| format!("{model}: {text}"))?;
    let mut probed = ProbedModel::default();

    let probes: Vec<(ReasoningEffort, Option<u64>)> = std::thread::scope(|scope| {
        let handles: Vec<_> = EFFORT_CANDIDATES
            .iter()
            .map(|&effort| {
                scope.spawn(move || {
                    let accepted = call(endpoint, model, 1, Some(effort))
                        .ok()
                        .and_then(Result::ok);
                    (effort, accepted)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("an effort probe panicked"))
            .collect()
    });
    let accepted: Vec<(ReasoningEffort, u64)> = probes
        .into_iter()
        .filter_map(|(effort, tokens)| tokens.map(|tokens| (effort, tokens)))
        .collect();
    probed.reasoning_efforts = distinct_efforts(&accepted);
    probed.default_reasoning_effort = default_effort(&probed.reasoning_efforts);

    if let Err(text) = call(endpoint, model, OVERSIZED_MAX_TOKENS, Option::None)? {
        probed.context_window = limit_after(
            &text,
            &["context length", "context window", "context_length"],
        );
        probed.max_tokens = limit_after(
            &text,
            &[
                "max_tokens",
                "max_completion_tokens",
                "max output",
                "maximum output",
                "output tokens",
            ],
        )
        // A context limit is not an output limit: the same number twice means
        // the message named only the context.
        .filter(|max| Some(*max) != probed.context_window);
    }
    Ok(probed)
}

/// The probe's dedup: an effort that changes the prompt
/// (a distinct `prompt_tokens`) is a real level; efforts that map to the same
/// prompt are one level, the first named kept. An effort whose prompt matches
/// the call without one is kept too: it is the endpoint's default level.
fn distinct_efforts(accepted: &[(ReasoningEffort, u64)]) -> Vec<ReasoningEffort> {
    let mut seen = Vec::new();
    let mut distinct = Vec::new();
    for (effort, tokens) in accepted {
        if seen.contains(tokens) {
            continue;
        }
        seen.push(*tokens);
        distinct.push(*effort);
    }
    // All the same prompt: the endpoint ignores the field, every value "works".
    if seen.len() <= 1 {
        return accepted.iter().map(|(effort, _)| *effort).collect();
    }
    distinct
}

/// `high` when accepted, else the strongest accepted; none when no effort is
/// accepted.
fn default_effort(accepted: &[ReasoningEffort]) -> Option<ReasoningEffort> {
    accepted
        .contains(&High)
        .then_some(High)
        .or_else(|| accepted.last().copied())
}

/// One non-streaming chat completion. `Err` is a transport failure;
/// `Ok(Err(text))` the endpoint refusing the request, with its message;
/// `Ok(Ok(n))` the prompt tokens it counted.
fn call(
    endpoint: &Endpoint,
    model: &str,
    max_tokens: u64,
    effort: Option<ReasoningEffort>,
) -> Result<Result<u64, String>, String> {
    let mut body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "say hi in one word"}],
        "max_tokens": max_tokens,
    });
    if let Some(effort) = effort {
        body["reasoning_effort"] = json!(effort);
    }
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into();
    let url = endpoint.url("chat/completions");
    let mut req = agent.post(&url).header("Content-Type", "application/json");
    if let Some(key) = &endpoint.api_key {
        req = req.header("Authorization", &format!("Bearer {key}"));
    }
    let resp = req.send_json(&body).map_err(|e| format!("{url}: {e}"))?;
    let status = resp.status();
    let text = resp.into_body().read_to_string().unwrap_or_default();
    let value = serde_json::from_str::<Value>(&text).ok();
    if status.is_success() {
        let tokens = value
            .as_ref()
            .and_then(|v| v.pointer("/usage/prompt_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        return Ok(Ok(tokens));
    }
    // The message, when the body is OpenAI's `{"error": {"message": …}}`.
    let message = value
        .as_ref()
        .and_then(|v| v.pointer("/error/message"))
        .and_then(Value::as_str)
        .map_or(text, str::to_string);
    Ok(Err(format!("HTTP {status}: {message}")))
}

/// The first token limit named after one of `keys` (case-insensitive) in
/// `text`: a number of at least `MIN_LIMIT`, not the one the probe sent.
fn limit_after(text: &str, keys: &[&str]) -> Option<u64> {
    let lower = text.to_lowercase();
    keys.iter()
        .filter_map(|key| lower.find(key).map(|at| &lower[at + key.len()..]))
        .find_map(|rest| {
            rest.split(|c: char| !c.is_ascii_digit())
                .filter_map(|word| word.parse::<u64>().ok())
                .find(|n| *n >= MIN_LIMIT && *n != OVERSIZED_MAX_TOKENS)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_level_is_kept_and_aliases_collapse() {
        // A Qwen3.8 template: xhigh renders the same prompt as no effort
        // (its default), and medium and high would be one level.
        let accepted = [(None, 13), (Low, 41), (Medium, 11), (High, 11), (XHigh, 53)];
        assert_eq!(distinct_efforts(&accepted), vec![None, Low, Medium, XHigh]);
    }

    #[test]
    fn an_endpoint_ignoring_the_field_accepts_every_effort() {
        let accepted = [(Low, 20), (High, 20)];
        assert_eq!(distinct_efforts(&accepted), vec![Low, High]);
    }
}
