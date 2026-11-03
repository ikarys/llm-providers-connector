//! Model catalog and `known` table: what well-known model families
//! advertise, for an endpoint that does not say (no `/models`, no
//! `max_model_len`).
//!
//! Matching normalizes the id first (gateway prefixes such as Databricks'
//! `system.ai.` / `databricks-` and an org path dropped, dots read as
//! dashes), then matches by family prefix, so
//! `databricks-deepseek-v4-1-flash-1017` is a DeepSeek V4.
//! This module does not gate images per model.

use super::params::ReasoningEffort::{self, *};

/// What a model family advertises.
#[derive(Debug, Clone, PartialEq)]
pub struct CatalogEntry {
    /// The reasoning efforts the family accepts, in display order (empty: unknown).
    pub reasoning_efforts: &'static [ReasoningEffort],
    pub default_reasoning_effort: Option<ReasoningEffort>,
    pub context_window: Option<u64>,
}

/// Family prefixes of the normalized id, most specific first.
const CATALOG: &[(&str, CatalogEntry)] = &[
    (
        "deepseek-v4",
        CatalogEntry {
            reasoning_efforts: &[None, Low, High, Max],
            default_reasoning_effort: Some(High),
            context_window: Some(1_000_000),
        },
    ),
    (
        "glm-5-3",
        CatalogEntry {
            reasoning_efforts: &[Low, High, Max],
            default_reasoning_effort: Some(Max),
            context_window: Some(1_000_000),
        },
    ),
    (
        "glm-5-2",
        CatalogEntry {
            reasoning_efforts: &[None, Minimal, Low, Medium, High, XHigh, Max],
            default_reasoning_effort: Some(Max),
            context_window: Some(1_000_000),
        },
    ),
    (
        "glm-5",
        CatalogEntry {
            reasoning_efforts: &[],
            default_reasoning_effort: Option::None,
            context_window: Some(1_000_000),
        },
    ),
    (
        "glm-4",
        CatalogEntry {
            reasoning_efforts: &[],
            default_reasoning_effort: Option::None,
            context_window: Some(128_000),
        },
    ),
    ("qwen3-6", QWEN3),
    ("qwen3-7", QWEN3),
    ("qwen3-8", QWEN3),
    (
        "kimi-k3",
        CatalogEntry {
            reasoning_efforts: &[Low, High, Max],
            default_reasoning_effort: Some(Max),
            context_window: Option::None,
        },
    ),
    ("gpt-5", OPENAI_REASONING),
    ("o1", OPENAI_REASONING),
    ("o3", OPENAI_REASONING),
    ("o4", OPENAI_REASONING),
];

const QWEN3: CatalogEntry = CatalogEntry {
    reasoning_efforts: &[Low, Medium, High],
    default_reasoning_effort: Some(Medium),
    context_window: Option::None,
};

const OPENAI_REASONING: CatalogEntry = CatalogEntry {
    reasoning_efforts: &[Minimal, Low, Medium, High],
    default_reasoning_effort: Some(Medium),
    context_window: Option::None,
};

/// What the catalog knows of `model_id`, if its family is listed.
pub fn catalog_entry(model_id: &str) -> Option<&'static CatalogEntry> {
    let id = normalized(model_id);
    CATALOG
        .iter()
        .find(|(family, _)| {
            id.strip_prefix(family)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('-'))
        })
        .map(|(_, entry)| entry)
}

/// The id as the catalog spells families: lowercase, no org path, no
/// gateway prefix, dots as dashes.
fn normalized(model_id: &str) -> String {
    let id = model_id
        .rsplit('/')
        .next()
        .unwrap_or(model_id)
        .to_lowercase();
    let id = id.strip_prefix("system.ai.").unwrap_or(&id);
    let id = id.strip_prefix("databricks-").unwrap_or(id);
    id.replace('.', "-")
}
