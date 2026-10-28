//! Where an OpenAI-compatible endpoint lives and how to address its paths.

#[derive(Debug, Clone, PartialEq)]
pub struct Endpoint {
    pub base_url: String,
    pub api_key: Option<String>,
}

impl Endpoint {
    /// `path` without a leading slash: `chat/completions`, `models`.
    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}/{path}", self.base_url.trim_end_matches('/'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_joins_with_one_slash() {
        let endpoint = Endpoint {
            base_url: "http://127.0.0.1:8000/v1/".into(),
            api_key: None,
        };
        assert_eq!(
            endpoint.url("chat/completions"),
            "http://127.0.0.1:8000/v1/chat/completions"
        );
    }
}
