# llm-providers-connector

A small, synchronous Rust client for OpenAI-compatible chat endpoints: local
servers (llama.cpp, vLLM, Ollama, LM Studio) as well as hosted ones. It
streams chat completions with tool calls and reasoning, lists and probes
models, estimates costs, and seals API keys at rest.

The crate knows the wire protocol, the shape of a provider and the per-model
preferences. It does not know your store or your web API, so it fits a CLI, a
desktop app or a server alike.

- No async runtime: HTTP goes through [`ureq`](https://crates.io/crates/ureq).
- Four dependencies: `serde`, `serde_json`, `ureq`, `chacha20poly1305`.
- Edition 2024, no `unsafe`.

## Install

Not on crates.io yet. Pin the git tag:

```toml
[dependencies]
llm-providers-connector = { git = "https://github.com/ikarys/llm-providers-connector", tag = "v0.1.2" }
```

## What is in the box

| Need | Types and functions |
| --- | --- |
| Point at a server | `Endpoint { base_url, api_key }` |
| Stream a chat completion | `stream`, `ChatRequest`, `Delta`, `Completion`, `StreamedToolCall` |
| Know which models a server has | `list_models` → `ListedModel` |
| Learn a model's limits by asking the server | `probe_model` → `ProbedModel`, `LimitSource` |
| Built-in knowledge of public model families | `catalog_entry` → `CatalogEntry` |
| Describe providers and the model in use | `Provider`, `ModelRef`, `ApiKey`, `ModelPrefs`, `ModelParams`, `ReasoningEffort` |
| Put a price on tokens | `Price`, `Cost` |
| Keep API keys unreadable in a copied database | `MasterKey` |

## Stream a completion

`stream` sends one chat completion request and forwards every delta to a
callback while it runs. The returned `Completion` holds the accumulated text,
the thinking (reasoning content, when the server sends it), the tool calls with
their raw argument text, the finish reason and the token counts. `stop` lets
you cancel a running stream from another thread.

```rust,no_run
use std::sync::atomic::AtomicBool;
use llm_providers_connector::{ChatRequest, Delta, Endpoint, ReasoningEffort, stream};
use serde_json::json;

fn main() -> Result<(), String> {
    let endpoint = Endpoint { base_url: "http://127.0.0.1:8000/v1".into(), api_key: None };
    let messages = vec![json!({"role": "user", "content": "Name three Rust HTTP crates."})];
    let tools = json!([]);
    let req = ChatRequest {
        model: "qwen3.8-27b",
        system_prompt: "Answer briefly.",
        messages: &messages,
        tools: &tools,
        max_tokens: 2048,
        reasoning_effort: Some(ReasoningEffort::Low),
    };
    let stop = AtomicBool::new(false);
    let done = stream(&endpoint, &req, &stop, &mut |d| match d {
        Delta::Text(t) => print!("{t}"),
        Delta::Thinking(_) => {}
        Delta::ToolPreparing { name, .. } => eprintln!("calling {name}…"),
    })?;
    println!("\n{} prompt tokens ({} cached), {} completion tokens, finish: {}",
        done.prompt_tokens, done.cached_tokens, done.completion_tokens, done.finish_reason);
    Ok(())
}
```

`messages` and `tools` are plain OpenAI JSON (`serde_json::Value`): the crate
does not impose a message type on you. Tool calls come back as
`StreamedToolCall { id, name, arguments }` with the arguments as the raw
accumulated string; parsing and policy are the caller's.

The request body is serialized deterministically on purpose: its bytes are the
server's prompt cache key, so a stable prefix (system prompt, tools, earlier
messages) keeps hitting the cache from one call to the next.

Reasoning models: set `max_tokens` generously. The thinking counts against it,
and a tight budget returns an empty `content` with the reasoning in `thinking`.

## Models: list, probe, catalog

Local servers list their models but rarely say what they can do.
`list_models` asks the endpoint (vLLM's `max_model_len` is read when present);
`probe_model` goes further and sends deliberately oversized or effort-tagged
requests, reading the limits the server names in its error messages.

```rust,no_run
use llm_providers_connector::{Endpoint, catalog_entry, list_models, probe_model};

fn main() -> Result<(), String> {
    let endpoint = Endpoint { base_url: "http://127.0.0.1:8000/v1".into(), api_key: None };
    for m in list_models(&endpoint)? {
        let probed = probe_model(&endpoint, &m.id)?;
        println!("{}: context {:?} ({:?}), max_tokens {:?}, efforts {:?}",
            m.id, probed.context_window, probed.context_source, probed.max_tokens, probed.reasoning_efforts);
    }
    if let Some(entry) = catalog_entry("gpt-5-mini") {
        println!("catalog says: context {:?}, default effort {:?}", entry.context_window, entry.default_reasoning_effort);
    }
    Ok(())
}
```

`ProbedModel` reports where each limit comes from (`LimitSource::Endpoint` or
`LimitSource::Catalog`), so a UI can show "measured" against "assumed". The
catalog is a short built-in table of public model families (Qwen 3, DeepSeek,
OpenAI reasoning models, …) with their accepted reasoning efforts and context
windows; it is a fallback, not a source of truth.

## Providers, preferences and prices

A `Provider` is one endpoint plus the models it serves, each with optional
`ModelPrefs` (max tokens, context window, compaction threshold, reasoning
effort, price). A `ModelRef { provider, model }` names the model in use.
`ModelParams::resolve` turns defaults, preferences and the listed context
window into the concrete parameters of one call.

`Price` is dollars per million tokens, split into fresh prompt tokens, cached
prompt tokens and completion tokens; `Price::costs` gives a `Cost` broken down
the same way, `Price::cost` the total.

```rust
use llm_providers_connector::Price;

let price = Price { input_per_m: 1.0, cached_per_m: 0.1, output_per_m: 5.0 };
let cost = price.costs(100_000, 60_000, 10_000);
assert_eq!((cost.input, cost.cached, cost.output), (0.04, 0.006, 0.05));
assert!((cost.total() - 0.096).abs() < 1e-9);
```

All of these derive `Serialize` / `Deserialize` (camelCase), so they go
straight into a settings file or a database column.

## API keys at rest

An `ApiKey` is `Env(var)`, `Plain(key)` (typed by a human, on its way in only)
or `Sealed(blob)`. `MasterKey` is a 32-byte key kept in a file of your choice
(created with mode 0600 on first use); it seals keys with ChaCha20-Poly1305 and
a fresh nonce each time, so a copied database reveals nothing without that
file.

```rust,no_run
use std::path::Path;
use llm_providers_connector::{ApiKey, MasterKey};

fn main() -> Result<(), String> {
    let master = MasterKey::load_or_create(Path::new("app.db.key"))?;
    let stored = ApiKey::Sealed(master.seal("sk-live-…"));
    let key: Option<String> = stored.resolve(|sealed| master.open(sealed))?;
    assert!(key.is_some());
    Ok(())
}
```

`ApiKey::resolve` is the one place that turns a stored key back into the
string to send: a missing environment variable yields `None` (the server will
answer 401), not an error.

## Develop

```bash
just test   # cargo test, README examples included
just lint   # rustfmt + clippy with -D warnings
just dev    # tests in watch mode when cargo-watch is installed
```

CI runs lint and tests on every push and pull request.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
