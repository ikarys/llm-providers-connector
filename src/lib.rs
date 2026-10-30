//! Talking to OpenAI-compatible model providers, owning the per-model
//! parameters for a call, and sealing provider API keys at rest.
//!
//! This crate knows only the wire protocol, provider shape, and model
//! preferences; it does not know a host app's store or web API.

mod catalog;
mod chat;
mod endpoint;
mod listing;
mod params;
mod price;
mod probe;
mod provider;
mod secret;
mod sse;

pub use catalog::{CatalogEntry, catalog_entry};
pub use chat::{ChatRequest, Completion, Delta, StreamedToolCall, stream};
pub use endpoint::Endpoint;
pub use listing::{ListedModel, list_models};
pub use params::{ModelParams, ModelPrefs, ReasoningEffort};
pub use price::{Cost, Price};
pub use probe::{LimitSource, ProbedModel, probe_model};
pub use provider::{ApiKey, ModelRef, Provider};
pub use secret::MasterKey;
