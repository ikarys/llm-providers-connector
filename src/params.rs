//! The per-model parameters a turn runs with, kept independent of any host
//! app so other products can reuse the same precedence and wire names.

use serde::{Deserialize, Serialize};

use crate::price::Price;
use crate::provider::ModelRef;

/// `ReasoningEffort`: how much a model may think.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    #[serde(rename = "xhigh")]
    XHigh,
    Max,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ModelPrefs {
    pub max_tokens: Option<u32>,
    /// `contextWindow`; overrides what the endpoint lists.
    pub context_window: Option<u64>,
    /// `compactionThreshold`; 0 disables compaction.
    pub compaction_threshold: Option<f64>,
    /// `reasoningEffort`; `None` sends no `reasoning_effort` field.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// What the model's tokens cost; none means the model has no price (a local model, or not set).
    pub price: Option<Price>,
}

/// The concrete parameters one turn runs with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelParams {
    /// The provider the model belongs to.
    pub provider: String,
    pub model: String,
    pub max_tokens: u32,
    pub context_window: u64,
    pub compact_at: f64,
    pub reasoning_effort: Option<ReasoningEffort>,
}

impl ModelParams {
    /// Chooses the concrete parameters for `selected`.
    ///
    /// Precedence: per-model preferences > `defaults`. The context window
    /// additionally falls back to the endpoint's current listing before the
    /// built-in default.
    pub fn resolve(
        defaults: &ModelParams,
        selected: ModelRef,
        prefs: Option<&ModelPrefs>,
        listed_window: Option<u64>,
    ) -> ModelParams {
        ModelParams {
            provider: selected.provider,
            model: selected.model,
            max_tokens: prefs
                .and_then(|p| p.max_tokens)
                .unwrap_or(defaults.max_tokens),
            context_window: prefs
                .and_then(|p| p.context_window)
                .or(listed_window)
                .unwrap_or(defaults.context_window),
            compact_at: prefs
                .and_then(|p| p.compaction_threshold)
                .unwrap_or(defaults.compact_at),
            reasoning_effort: prefs
                .and_then(|p| p.reasoning_effort)
                .or(defaults.reasoning_effort),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_effort_wire_names_are_stable() {
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::XHigh).unwrap(),
            "\"xhigh\""
        );
        assert_eq!(
            serde_json::from_str::<ReasoningEffort>("\"xhigh\"").unwrap(),
            ReasoningEffort::XHigh
        );
        assert_eq!(
            serde_json::to_string(&ReasoningEffort::Minimal).unwrap(),
            "\"minimal\""
        );
    }
}
