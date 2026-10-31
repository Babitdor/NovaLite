//! Provider selection: which provider is used, and what happens when it
//! cannot be built.
//!
//! Two rules run through this file:
//!
//! * **A missing credential is an error, not a panic.** The agent is built
//!   both at startup and from the `/model` picker, and a panic in the picker
//!   takes the whole TUI down over a typo in the environment.
//! * **An unknown provider is an error, not a fallback.** Silently using
//!   Ollama when `LLM_PROVIDER` says something else sends the prompt to a
//!   provider the user did not ask for, and the only sign of it is the
//!   answer.
//!
//! The environment is global and shared by parallel tests, so nothing here
//! reads it: `build_for_provider` takes the credentials as a closure.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rust_agent::agent::{build_for_provider, parse_provider};

mod common;

/// `build_for_provider` against a throwaway session.
///
/// The provider dispatch is what these cases are about, not the session, so
/// each one gets a fresh temporary workspace and an approving gate. The
/// session is a value rather than a process global, which is what lets the
/// cases run in any order.
fn build(
    provider: &str,
    model: &str,
    credential: impl Fn(&str) -> Option<String>,
) -> anyhow::Result<rust_agent::agent::Agents> {
    build_for_provider(provider, model, credential, &common::session())
}

/// A credential lookup that finds nothing, as an unset environment does.
fn no_credentials(_: &str) -> Option<String> {
    None
}

/// A credential lookup that finds `value` for `var` and nothing else.
fn only(var: &'static str, value: &'static str) -> impl Fn(&str) -> Option<String> {
    move |asked| (asked == var).then(|| value.to_string())
}

/// The error from a build that must fail.
///
/// `Result::expect_err` needs `Debug` on the success type, and rig's `Agent`
/// does not implement it, so the failure is unwrapped by hand.
fn error_from<T>(result: anyhow::Result<T>) -> anyhow::Error {
    match result {
        Ok(_) => panic!("the build should have failed"),
        Err(e) => e,
    }
}

// ------------------------------------------------------------- the name

#[test]
fn an_absent_or_blank_provider_is_ollama() {
    assert_eq!(parse_provider(None), "ollama");
    assert_eq!(parse_provider(Some("")), "ollama");
    assert_eq!(parse_provider(Some("   ")), "ollama");
}

#[test]
fn a_provider_name_is_trimmed_and_lowercased() {
    // `LLM_PROVIDER=OpenAI` is not a typo the user should have to discover.
    assert_eq!(parse_provider(Some("  OpenAI ")), "openai");
    assert_eq!(parse_provider(Some("ANTHROPIC")), "anthropic");
    assert_eq!(parse_provider(Some("ollama")), "ollama");
}

#[test]
fn an_unknown_provider_is_kept_so_it_can_be_reported() {
    // The name is not silently rewritten to `ollama`: the error has to be
    // able to say what the user actually typed.
    assert_eq!(
        parse_provider(Some("openai-compatible")),
        "openai-compatible"
    );
}

// --------------------------------------------------------- the dispatch

#[test]
fn ollama_needs_no_credential() {
    // The local daemon is the default and the only provider that works out
    // of the box, so it must build with nothing in the environment.
    assert!(build("ollama", "llama3", no_credentials).is_ok());
}

#[test]
fn a_missing_anthropic_key_is_an_error_that_names_the_variable() {
    let err = error_from(build("anthropic", "claude-sonnet-4", no_credentials));
    let message = err.to_string();

    assert!(
        message.contains("ANTHROPIC_API_KEY"),
        "the error must name the variable to set, got: {message}"
    );
    assert!(
        message.contains("anthropic"),
        "the error must name the provider that needs it, got: {message}"
    );
}

#[test]
fn a_missing_openai_key_is_an_error_that_names_the_variable() {
    let err = error_from(build("openai", "gpt-5", no_credentials));
    let message = err.to_string();

    assert!(
        message.contains("OPENAI_API_KEY"),
        "the error must name the variable to set, got: {message}"
    );
}

#[test]
fn a_blank_credential_counts_as_missing() {
    // `set OPENAI_API_KEY=` leaves an empty variable, which would otherwise
    // build an agent that fails every request with a 401.
    let blank = |_: &str| Some("   ".to_string());
    assert!(build("openai", "gpt-5", blank).is_err());
}

#[test]
fn a_present_credential_builds_the_agent() {
    // The other half of the rule: the error must not fire when the key is
    // there, or the check would be useless.
    assert!(build(
        "anthropic",
        "claude-sonnet-4",
        only("ANTHROPIC_API_KEY", "sk-ant-test")
    )
    .is_ok());
    assert!(build("openai", "gpt-5", only("OPENAI_API_KEY", "sk-test")).is_ok());
}

#[test]
fn an_unknown_provider_is_an_error_that_names_it() {
    let err = error_from(build("openai-compatible", "gpt-5", no_credentials));
    let message = err.to_string();

    assert!(
        message.contains("openai-compatible"),
        "the error must quote what the user typed, got: {message}"
    );
    assert!(
        message.contains("ollama") && message.contains("anthropic") && message.contains("openai"),
        "the error must list the providers that do work, got: {message}"
    );
}

#[test]
fn an_unknown_provider_is_rejected_before_any_credential_is_read() {
    // The dispatch must not read a credential for a provider it does not
    // know: a lookup that panics proves the order.
    let panicking = |var: &str| -> Option<String> {
        panic!("a credential was read for an unknown provider: {var}")
    };
    assert!(build("gemini", "gemini-2", panicking).is_err());
}

#[test]
fn the_ollama_base_url_is_configurable_and_defaults_to_localhost() {
    // The base URL is read through the same lookup, so a test can point the
    // daemon somewhere else without touching the environment.
    assert!(build(
        "ollama",
        "llama3",
        only("OLLAMA_BASE_URL", "http://127.0.0.1:11500")
    )
    .is_ok());
    // A blank base URL falls back rather than building a client with no host.
    let blank = |_: &str| Some(String::new());
    assert!(build("ollama", "llama3", blank).is_ok());
}
