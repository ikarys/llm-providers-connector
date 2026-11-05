# Library crate: `dev` runs the tests in watch mode when cargo-watch is installed, once otherwise.
dev:
    cargo watch -x test 2>/dev/null || cargo test

build:
    cargo build

test:
    cargo test

lint:
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
