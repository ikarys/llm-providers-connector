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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;

    // The crate once built ureq without a TLS provider: every https endpoint failed after the TCP
    // connect, without a handshake. 0x16 is the first byte of a TLS ClientHello record.
    #[test]
    fn https_endpoint_gets_a_tls_handshake() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut first = [0u8; 1];
            socket.read_exact(&mut first).map(|_| first[0]).ok()
        });
        let endpoint = Endpoint {
            base_url: format!("https://127.0.0.1:{port}/v1"),
            api_key: None,
        };
        let err = list_models(&endpoint).unwrap_err();
        assert_eq!(
            server.join().unwrap(),
            Some(0x16),
            "no TLS handshake, error: {err}"
        );
    }
}
