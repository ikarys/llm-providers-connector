# llm-providers-connector

A small synchronous Rust client for OpenAI-compatible LLM endpoints. It was
extracted from [fennec](https://github.com/ikarys/fennec) so that several
projects can share it.

What it does:

- `Endpoint` / `Provider` / `ModelRef`: an OpenAI-compatible base URL, its API
  key and the models it serves, with per-model preferences (`ModelPrefs`).
- `stream`: streaming chat completions with tool calls, reasoning content and
  token usage (prompt, cached, completion) in a `Completion`.
- `list_models` and `probe_model`: ask an endpoint which models it serves and
  what limits they have (context window, max tokens, reasoning efforts), which
  matters for local servers whose models are in no public catalog.
- `catalog_entry` and `Price`: known limits and prices for public models.
- `MasterKey`: seals API keys at rest (ChaCha20-Poly1305) so a copied database
  reveals no key.

No async runtime: HTTP goes through `ureq`. Four dependencies.

## Use

```toml
[dependencies]
llm-providers-connector = { git = "https://github.com/ikarys/llm-providers-connector", tag = "v0.1.0" }
```

```rust
use llm_providers_connector::{Endpoint, list_models};

let endpoint = Endpoint { base_url: "http://127.0.0.1:8000/v1".into(), api_key: None };
for m in list_models(&endpoint)? {
    println!("{} ({:?} tokens)", m.id, m.context_window);
}
```

## Develop

`just test`, `just lint` (rustfmt + clippy). CI runs both.

## License

AGPL-3.0-only, like fennec.
