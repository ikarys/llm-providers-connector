//! What a model's tokens cost, in dollars per million tokens, as the human sets it per model; an
//! estimate (Insight), never sent to the model.

use serde::{Deserialize, Serialize};

/// Dollars per million tokens, as set on the Settings page.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Price {
    /// Prompt tokens not read from cache.
    pub input_per_m: f64,
    /// Prompt tokens re-read from cache.
    pub cached_per_m: f64,
    /// Completion tokens.
    pub output_per_m: f64,
}

impl Price {
    /// The estimate in dollars for one usage summary.
    pub fn cost(&self, prompt_tokens: u64, cached_tokens: u64, completion_tokens: u64) -> f64 {
        let fresh = prompt_tokens.saturating_sub(cached_tokens);
        (fresh as f64 * self.input_per_m
            + cached_tokens as f64 * self.cached_per_m
            + completion_tokens as f64 * self.output_per_m)
            / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_price_prices_a_million_prompt_tokens() {
        let price = Price {
            input_per_m: 3.0,
            cached_per_m: 0.3,
            output_per_m: 15.0,
        };
        let cost = price.cost(1_000_000, 900_000, 100_000);
        assert!((cost - 2.07).abs() < 1e-9);
    }

    #[test]
    fn cached_tokens_above_prompt_do_not_underflow() {
        let price = Price {
            input_per_m: 3.0,
            cached_per_m: 0.3,
            output_per_m: 15.0,
        };
        let cost = price.cost(0, 1_000_000, 0);
        assert!((cost - 0.3).abs() < 1e-9);
    }
}
