//! Talking to OpenAI-compatible model endpoints and owning the per-model
//! parameters for a call.
//!
//! Fennec-free: this crate knows only the wire protocol and the shape of
//! model preferences; it does not know fennec's domains, store, or web API.

mod chat;
mod endpoint;
mod listing;
mod params;
mod sse;

pub use chat::{ChatRequest, Completion, Delta, StreamedToolCall, stream};
pub use endpoint::Endpoint;
pub use listing::{ListedModel, list_models};
pub use params::{ModelParams, ModelPrefs, ReasoningEffort};
