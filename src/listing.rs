//! The endpoint's model listing (`/models`), owned by the provider crate so
//! callers can refresh it without environment settings.

use super::endpoint::Endpoint;
use serde::Serialize;
use serde_json::Value;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListedModel {
    pub id: String,
    pub context_window: Option<u64>,
}

/// Lists the models the endpoint serves. `max_model_len` is vLLM's field.
pub fn list_models(endpoint: &Endpoint) -> Result<Vec<ListedModel>, String> {
    let url = endpoint.url("models");
    // Bounded: a settings page or startup call must not hang on a
    // packet-dropping endpoint.
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(5)))
        .build()
        .into();
    let mut req = agent.get(&url);
    if let Some(key) = &endpoint.api_key {
        req = req.header("Authorization", &format!("Bearer {key}"));
    }
    let value: Value = req
        .call()
        .and_then(|r| r.into_body().read_json())
        .map_err(|e| format!("{url}: {e}"))?;
    let data = value
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Ok(data
        .into_iter()
        .filter_map(|m| {
            let id = m.get("id")?.as_str()?.to_string();
            (!id.is_empty()).then(|| ListedModel {
                id,
                context_window: m.get("max_model_len").and_then(Value::as_u64),
            })
        })
        .collect())
}
