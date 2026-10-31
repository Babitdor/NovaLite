//! The model-call budget for one prompt.
//!
//! rig's `max_turns` counts every model call, not tool calls, so a budget that
//! is too small fails a prompt mid-exploration with "reached the max turns
//! limit". These pin the resolution rules for `LLM_MAX_TURNS`.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rust_agent::agent::{is_turn_limit, parse_max_turns, DEFAULT_MAX_TURNS};

#[test]
fn an_absent_value_uses_the_default() {
    assert_eq!(parse_max_turns(None), DEFAULT_MAX_TURNS);
}

#[test]
fn a_configured_value_is_used() {
    assert_eq!(parse_max_turns(Some("25")), 25);
    assert_eq!(
        parse_max_turns(Some("  25  ")),
        25,
        "surrounding space is ignored"
    );
}

#[test]
fn a_zero_budget_falls_back_rather_than_failing_every_prompt() {
    // Zero would emit no model call at all, so every prompt would fail. That
    // is never what the user meant to ask for.
    assert_eq!(parse_max_turns(Some("0")), DEFAULT_MAX_TURNS);
}

#[test]
fn an_unparseable_value_falls_back() {
    for raw in ["", "  ", "ten", "-1", "1.5"] {
        assert_eq!(
            parse_max_turns(Some(raw)),
            DEFAULT_MAX_TURNS,
            "{raw:?} should fall back to the default"
        );
    }
}

#[test]
fn the_default_leaves_room_for_a_real_task() {
    // A tool round-trip costs two model calls, so a budget of 10 was about
    // five tool calls: not enough to explore and then answer. 40 was about
    // twenty, which a task that explores, edits, runs the tests and then
    // fixes what they report still overruns. Checked in a const block so the
    // assertion is made at compile time.
    const {
        assert!(
            DEFAULT_MAX_TURNS >= 40,
            "the default budget must leave room for a task, not just a round trip"
        )
    };
}

#[test]
fn the_turn_limit_failure_is_recognised() {
    // rig reports this as `PromptError::MaxTurns`, and the text is what
    // reaches the UI: `Event` is `Clone`, so it cannot carry the error
    // itself. This is the exact wording rig uses.
    assert!(is_turn_limit("reached the max turns limit of 100"));
    assert!(
        is_turn_limit("Reached the max turns limit of 40"),
        "the match is case-insensitive, because the phrasing is not ours"
    );
}

#[test]
fn another_failure_is_not_mistaken_for_the_turn_limit() {
    // The caller replaces the message when this matches, so a false positive
    // would hide the real error behind advice that does not apply to it.
    for other in [
        "connection refused",
        "exceeds the available context size",
        "the daemon went away",
        "the model returned no content",
    ] {
        assert!(
            !is_turn_limit(other),
            "{other:?} is not a turn-limit failure"
        );
    }
}
