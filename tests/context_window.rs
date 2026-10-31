//! The context window requested from an Ollama daemon, and the per-result
//! budget derived from it.
//!
//! The daemon picks its own default from the GPU's VRAM, and on a 6 GB card
//! that is 4096 tokens. A coding agent overruns that in a few tool
//! round-trips, and the daemon then rejects the request outright with
//! "request (N tokens) exceeds the available context size (4096 tokens)".
//! These pin the resolution rules for `LLM_NUM_CTX` and
//! `LLM_TOOL_RESULT_CHARS`.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rust_agent::agent::{
    default_window, parse_num_ctx, parse_tool_result_chars, tool_result_budget_chars,
    DEFAULT_COMPACT_AT, DEFAULT_NUM_CTX,
};

/// The provider whose window the local default is sized for.
const LOCAL: &str = "ollama";

#[test]
fn an_absent_value_uses_the_providers_default() {
    assert_eq!(parse_num_ctx(None, LOCAL), DEFAULT_NUM_CTX);
}

#[test]
fn each_provider_gets_the_window_it_actually_serves() {
    // One global default was wrong at both ends. A hosted model with a 200k
    // window was budgeted at 32k, so every percentage the user was shown was
    // computed against a window six times too small and compaction fired at a
    // sixth of the real limit. The local default is now the same 256k: the
    // window is what every other budget is a fraction of, so a small one
    // costs the agent its memory of the run.
    //
    // The note that used to sit here said a 6 GB card "fails outright" at
    // 128k. Measured on this machine, it does not: a 0.8B model served
    // 262144 entirely on the GPU, and a 5.2 GB 8B model clamped the request
    // to its own 128000 maximum and served it with 43% of the work on the
    // CPU. A large window on a small card costs speed, not correctness, which
    // is why the default can be raised and `LLM_NUM_CTX` is the knob for a
    // model that cannot afford it.
    assert_eq!(default_window("anthropic"), 200_000);
    assert_eq!(default_window("openai"), 128_000);
    assert_eq!(default_window("ollama"), DEFAULT_NUM_CTX);
    assert_eq!(
        default_window("something-new"),
        DEFAULT_NUM_CTX,
        "an unknown provider gets the local default, not a guess"
    );
}

#[test]
fn the_override_beats_the_provider_default() {
    // `LLM_NUM_CTX` is an override over the provider default, not the only
    // source: a user who has not set it gets the window their provider serves,
    // and a user who has set it gets what they asked for on any provider.
    assert_eq!(parse_num_ctx(Some("8192"), "anthropic"), 8192);
    assert_eq!(parse_num_ctx(Some("8192"), LOCAL), 8192);
}

#[test]
fn a_configured_value_is_used() {
    assert_eq!(parse_num_ctx(Some("8192"), LOCAL), 8192);
    assert_eq!(
        parse_num_ctx(Some("  8192  "), LOCAL),
        8192,
        "surrounding space is ignored"
    );
}

#[test]
fn a_zero_window_falls_back_rather_than_failing_every_request() {
    // A zero window would make every request fail, which is never what the
    // user meant to ask for.
    assert_eq!(parse_num_ctx(Some("0"), LOCAL), DEFAULT_NUM_CTX);
}

#[test]
fn an_unparseable_value_falls_back() {
    for raw in ["", "  ", "lots", "-1", "1.5", "4096.0"] {
        assert_eq!(
            parse_num_ctx(Some(raw), LOCAL),
            DEFAULT_NUM_CTX,
            "{raw:?} should fall back to the default"
        );
    }
}

#[test]
fn the_default_clears_the_daemons_vram_derived_floor() {
    // 4096 is what a 6 GB card gets, and it is the value that produced the
    // rejection. The default has to be comfortably above it, or the fix
    // changes nothing. Checked in a const block so it is a compile-time
    // assertion rather than a test that can only ever pass.
    const {
        assert!(
            DEFAULT_NUM_CTX > 4096,
            "the default window must clear the daemon's 4096 floor"
        )
    };
}

// ------------------------------------------------- the tool-result budget

#[test]
fn an_absent_tool_result_budget_defers_to_the_window() {
    // `None` is the signal to derive the budget from the window, not a budget
    // of zero.
    assert_eq!(parse_tool_result_chars(None), None);
}

#[test]
fn a_configured_tool_result_budget_is_used() {
    assert_eq!(parse_tool_result_chars(Some("2000")), Some(2000));
    assert_eq!(
        parse_tool_result_chars(Some("  2000  ")),
        Some(2000),
        "surrounding space is ignored"
    );
}

#[test]
fn a_zero_or_unparseable_tool_result_budget_defers_to_the_window() {
    // A zero budget would elide every result to nothing, which is never what
    // the user meant to ask for.
    for raw in ["0", "", "  ", "lots", "-1", "2000.0", "2k"] {
        assert_eq!(
            parse_tool_result_chars(Some(raw)),
            None,
            "{raw:?} should defer to the window-derived default"
        );
    }
}

#[test]
fn the_tool_result_budget_scales_with_the_window() {
    // A fixed cap would be 6% of a 32k window but half of a 4,096 one, and
    // 4,096 is what a 6 GB card serves by default. Deriving it from the
    // window keeps the fraction constant.
    assert_eq!(tool_result_budget_chars(32_768), 16_384);
    assert_eq!(tool_result_budget_chars(4_096), 2_048);
    assert_eq!(tool_result_budget_chars(0), 0);
}

#[test]
fn several_max_size_tool_results_fit_before_compaction_fires() {
    // The budget bounds one result; compaction bounds the accumulation. For
    // the two to work together, several max-size results have to fit under
    // the threshold, or a run overruns the window before compaction, which
    // only runs once a run has settled, can fire.
    let window = DEFAULT_NUM_CTX;
    // ~4 characters per token, the same conversion the budget uses.
    let budget_tokens = (tool_result_budget_chars(window) / 4) as u64;
    let threshold = window as u64 * DEFAULT_COMPACT_AT as u64 / 100;

    assert!(
        budget_tokens * 4 <= threshold,
        "four max-size results ({}) must fit under the compaction threshold \
         ({threshold}), or a run dies mid-flight",
        budget_tokens * 4
    );
}
