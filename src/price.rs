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

/// An estimate split by kind of token, in dollars.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cost {
    /// Prompt tokens not read from cache.
    pub input: f64,
    /// Tokens read from the server's cache.
    pub cached: f64,
    /// Generated tokens.
    pub output: f64,
}

impl Cost {
    pub fn total(&self) -> f64 {
        self.input + self.cached + self.output
    }
}

impl std::ops::AddAssign for Cost {
    fn add_assign(&mut self, other: Cost) {
        self.input += other.input;
        self.cached += other.cached;
        self.output += other.output;
    }
}

impl Price {
    /// The estimate in dollars for one usage summary.
    pub fn cost(&self, prompt_tokens: u64, cached_tokens: u64, completion_tokens: u64) -> f64 {
        self.costs(prompt_tokens, cached_tokens, completion_tokens)
            .total()
    }

    /// The same estimate, split by kind of token.
    pub fn costs(&self, prompt_tokens: u64, cached_tokens: u64, completion_tokens: u64) -> Cost {
        let fresh = prompt_tokens.saturating_sub(cached_tokens);
        Cost {
            input: fresh as f64 * self.input_per_m / 1_000_000.0,
            cached: cached_tokens as f64 * self.cached_per_m / 1_000_000.0,
            output: completion_tokens as f64 * self.output_per_m / 1_000_000.0,
        }
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
    fn costs_split_the_estimate_by_kind_of_token() {
        let price = Price {
            input_per_m: 3.0,
            cached_per_m: 0.3,
            output_per_m: 15.0,
        };
        let cost = price.costs(1_000_000, 900_000, 100_000);
        assert!((cost.input - 0.3).abs() < 1e-9);
        assert!((cost.cached - 0.27).abs() < 1e-9);
        assert!((cost.output - 1.5).abs() < 1e-9);
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
