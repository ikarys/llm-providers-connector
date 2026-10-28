//! OpenAI-compatible providers and the source of their API keys.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::params::ModelPrefs;

/// Where a provider's API key comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ApiKey {
    /// The name of an environment variable that holds the key.
    Env(String),
    /// The key sealed with a `MasterKey`.
    Sealed(String),
    /// A key typed by a human, on its way in only: seal it before storing it.
    Plain(String),
}

impl ApiKey {
    /// The key itself; `open` unseals a sealed one. A missing environment
    /// variable is no key (the server answers 401), not an error.
    pub fn resolve(
        &self,
        open: impl FnOnce(&str) -> Result<String, String>,
    ) -> Result<Option<String>, String> {
        match self {
            Self::Env(var) => Ok(std::env::var(var).ok()),
            Self::Sealed(sealed) => open(sealed).map(Some),
            Self::Plain(plain) => Ok(Some(plain.clone())),
        }
    }
}

/// An OpenAI-compatible endpoint and the per-model overrides of its models.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Provider {
    /// Stable and without `/`.
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub api_key: Option<ApiKey>,
    /// Keyed by the model id the provider lists.
    pub models: BTreeMap<String, ModelPrefs>,
}

/// A model of a provider: what a call runs on. `model` is the provider's own
/// id, sent as is on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef {
    pub provider: String,
    pub model: String,
}

impl Provider {
    /// Rejects what would make a call impossible; the base URL is the caller's.
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err(format!("provider {}: id must not be empty", self.id));
        }
        if self.id.contains('/') {
            return Err(format!("provider {}: id must not contain '/'", self.id));
        }

        match &self.api_key {
            Some(ApiKey::Env(var)) if var.is_empty() => {
                return Err(format!(
                    "provider {}: api_key env must not be empty",
                    self.id
                ));
            }
            Some(ApiKey::Plain(key)) if key.is_empty() => {
                return Err(format!(
                    "provider {}: api_key plain must not be empty",
                    self.id
                ));
            }
            _ => {}
        }
        for (model, prefs) in &self.models {
            let field = |rule: &str| format!("provider {}, model {model}: {rule}", self.id);
            if model.is_empty() {
                return Err(field("model id must not be empty"));
            }
            if prefs.max_tokens.is_some_and(|v| v < 1) {
                return Err(field("max_tokens must be >= 1"));
            }
            if prefs.context_window.is_some_and(|v| v < 1024) {
                return Err(field("context_window must be >= 1024"));
            }
            if prefs
                .compaction_threshold
                .is_some_and(|v| !(0.0..=1.0).contains(&v))
            {
                return Err(field("compaction_threshold must be in 0.0..=1.0"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_prefs() -> ModelPrefs {
        ModelPrefs {
            max_tokens: Some(1),
            context_window: Some(1024),
            compaction_threshold: Some(1.0),
            reasoning_effort: None,
        }
    }

    fn valid_provider() -> Provider {
        Provider {
            id: "local".into(),
            name: "Local".into(),
            base_url: "http://127.0.0.1:8000/v1".into(),
            api_key: Some(ApiKey::Env("LLM_PROVIDERS_TEST_KEY".into())),
            models: BTreeMap::from([("m".into(), valid_prefs())]),
        }
    }

    #[test]
    fn api_key_uses_externally_tagged_wire_names() {
        assert_eq!(
            serde_json::to_string(&ApiKey::Env("K".into())).unwrap(),
            r#"{"env":"K"}"#
        );
        assert_eq!(
            serde_json::to_string(&ApiKey::Sealed("s".into())).unwrap(),
            r#"{"sealed":"s"}"#
        );
        assert_eq!(
            serde_json::to_string(&ApiKey::Plain("p".into())).unwrap(),
            r#"{"plain":"p"}"#
        );

        assert_eq!(
            serde_json::from_str::<ApiKey>(r#"{"env":"K"}"#).unwrap(),
            ApiKey::Env("K".into())
        );
        assert_eq!(
            serde_json::from_str::<ApiKey>(r#"{"sealed":"s"}"#).unwrap(),
            ApiKey::Sealed("s".into())
        );
        assert_eq!(
            serde_json::from_str::<ApiKey>(r#"{"plain":"p"}"#).unwrap(),
            ApiKey::Plain("p".into())
        );
    }

    #[test]
    fn resolve_env_unset_is_no_key() {
        // Never set by anything.
        let var = "LLM_PROVIDERS_TEST_UNSET";
        let key = ApiKey::Env(var.into());
        assert_eq!(key.resolve(|_| unreachable!()).unwrap(), None);
    }

    #[test]
    fn resolve_sealed_calls_open() {
        let key = ApiKey::Sealed("sealed-data".into());
        let resolved = key
            .resolve(|sealed| {
                assert_eq!(sealed, "sealed-data");
                Ok("key".into())
            })
            .unwrap();
        assert_eq!(resolved, Some("key".into()));
    }

    #[test]
    fn resolve_plain_returns_key() {
        let key = ApiKey::Plain("plain-key".into());
        assert_eq!(
            key.resolve(|_| unreachable!()).unwrap(),
            Some("plain-key".into())
        );
    }

    #[test]
    fn validate_accepts_valid_provider() {
        valid_provider().validate().unwrap();
    }

    #[test]
    fn validate_rejects_bad_id() {
        let mut provider = valid_provider();
        provider.id = "".into();
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("id must not be empty")
        );

        let mut provider = valid_provider();
        provider.id = "a/b".into();
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("id must not contain '/'")
        );
    }

    #[test]
    fn validate_rejects_empty_api_key_variants() {
        let mut provider = valid_provider();
        provider.api_key = Some(ApiKey::Env("".into()));
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("api_key env must not be empty")
        );

        let mut provider = valid_provider();
        provider.api_key = Some(ApiKey::Plain("".into()));
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("api_key plain must not be empty")
        );
    }

    #[test]
    fn validate_rejects_bad_model_fields() {
        let mut provider = valid_provider();
        provider.models.get_mut("m").unwrap().max_tokens = Some(0);
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("provider local, model m: max_tokens must be >= 1")
        );

        let mut provider = valid_provider();
        provider
            .models
            .insert("unset".into(), ModelPrefs::default());
        provider.validate().unwrap();

        let mut provider = valid_provider();
        provider.models.get_mut("m").unwrap().context_window = Some(1023);
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("context_window must be >= 1024")
        );

        let mut provider = valid_provider();
        provider.models.get_mut("m").unwrap().compaction_threshold = Some(1.0001);
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("compaction_threshold must be in 0.0..=1.0")
        );

        let mut provider = valid_provider();
        provider.models.get_mut("m").unwrap().compaction_threshold = Some(f64::NAN);
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("compaction_threshold must be in 0.0..=1.0")
        );

        let mut provider = valid_provider();
        provider.models.insert("".into(), valid_prefs());
        assert!(
            provider
                .validate()
                .err()
                .unwrap()
                .contains("model id must not be empty")
        );
    }
}
