//! A small coding agent with a ratatui front end.
//!
//! The binary in `main.rs` is a thin shell over this library: it builds the
//! agent, enters the terminal, and runs the event loop. Exposing the modules
//! here lets the integration tests drive the same stream -> event -> render
//! path the binary runs, rather than a reimplementation of it.
//!
//! The README is the crate's front page, and it is included rather than
//! copied: a `rust` block in it is a doc test, so the example in the README
//! is compiled by `cargo test --doc` and cannot drift from the API it
//! describes. A second copy of the example here would be a second thing to
//! keep in step, and the one that gets forgotten.
#![doc = include_str!("../README.md")]
// The lint restrictions in `Cargo.toml` are for production code. In a test a
// panic *is* the assertion, and `unwrap` on a fixture that cannot fail is
// clearer than a `?` that would report the failure as a test error rather than
// as a line number. The exception is scoped to `cfg(test)` so it cannot leak
// into a release build, and it is declared here rather than in each of the
// fifteen test modules so a new one cannot forget it.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

pub mod agent;
pub mod app;
pub mod approval;
pub mod banner;
pub mod cli;
pub mod config;
pub mod context;
pub mod error;
pub mod event;
pub mod filesystem;
pub mod logging;
pub mod patch;
pub mod process;
pub mod prompts;
pub mod render;
pub mod runtime;
pub mod search;
pub mod session;
pub mod tools;
pub mod trust;
pub mod tui;
pub mod workspace;
