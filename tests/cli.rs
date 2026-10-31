//! Command-line parsing.
//!
//! The rules here are the ones every command-line tool follows, and each of
//! them is tested because the alternative is a confusing failure: before
//! this, `nova-lite --versoin` tried to canonicalise a path called
//! `--versoin` and reported that it could not be used as the workspace.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rust_agent::cli::{parse_args, usage, version, Command};
use std::ffi::OsString;
use std::path::PathBuf;

/// Parse a command line written as a slice of `&str`.
fn parse(args: &[&str]) -> Result<Command, String> {
    let owned: Vec<OsString> = args.iter().map(OsString::from).collect();
    parse_args(&owned)
}

/// The directory a `Run` names, or a panic if the command is not a run.
fn run_dir(args: &[&str]) -> Option<PathBuf> {
    match parse(args) {
        Ok(Command::Run { dir, .. }) => dir,
        other => panic!("expected a run, got {other:?}"),
    }
}

/// Whether a `Run` was asked to skip the per-action prompt.
fn run_auto(args: &[&str]) -> bool {
    match parse(args) {
        Ok(Command::Run { auto_approve, .. }) => auto_approve,
        other => panic!("expected a run, got {other:?}"),
    }
}

// ------------------------------------------------------------- the run

#[test]
fn no_arguments_runs_in_the_current_directory() {
    // `None` rather than a resolved path: parsing touches no filesystem, so
    // the caller decides what "the current directory" means.
    assert_eq!(run_dir(&[]), None);
}

#[test]
fn a_directory_is_taken_as_the_workspace() {
    assert_eq!(run_dir(&["src"]), Some(PathBuf::from("src")));
    assert_eq!(
        run_dir(&[r"C:\work\proj"]),
        Some(PathBuf::from(r"C:\work\proj"))
    );
}

#[test]
fn a_directory_that_looks_like_an_option_can_still_be_named() {
    // `--` ends the options, which is the only way to name a directory whose
    // name starts with a dash.
    assert_eq!(run_dir(&["--", "--weird"]), Some(PathBuf::from("--weird")));
    assert_eq!(run_dir(&["--", "-h"]), Some(PathBuf::from("-h")));
}

// -------------------------------------------------------- auto-approve

#[test]
fn the_prompt_is_the_default() {
    // The gate is the default, so a user who types nothing gets asked. A
    // flag that had to be given to *get* the prompt would be a flag nobody
    // gives.
    assert!(!run_auto(&[]));
    assert!(!run_auto(&["src"]));
}

#[test]
fn auto_approve_is_recognised_in_both_spellings() {
    assert!(run_auto(&["--auto-approve"]));
    assert!(run_auto(&["--yolo"]));
}

#[test]
fn auto_approve_does_not_consume_the_directory() {
    // The flag and the directory are independent: either order, and the
    // directory is still the directory.
    assert_eq!(
        run_dir(&["--auto-approve", "src"]),
        Some(PathBuf::from("src"))
    );
    assert_eq!(run_dir(&["src", "--yolo"]), Some(PathBuf::from("src")));
    assert!(run_auto(&["src", "--yolo"]));
}

#[test]
fn auto_approve_after_the_terminator_is_a_directory() {
    // `--` means "everything after this is an argument", so the flag is not
    // a flag there. Reading it as one would make a directory named
    // `--yolo` unnameable.
    assert_eq!(run_dir(&["--", "--yolo"]), Some(PathBuf::from("--yolo")));
    assert!(!run_auto(&["--", "--yolo"]));
}

// --------------------------------------------------------- help/version

#[test]
fn help_and_version_are_recognised_in_both_spellings() {
    assert_eq!(parse(&["--help"]), Ok(Command::Help));
    assert_eq!(parse(&["-h"]), Ok(Command::Help));
    assert_eq!(parse(&["--version"]), Ok(Command::Version));
    assert_eq!(parse(&["-V"]), Ok(Command::Version));
}

#[test]
fn help_wins_wherever_it_appears() {
    // `nova-lite --help` must not be read as a request to use a directory
    // called `--help`, and neither must the reverse order.
    assert_eq!(parse(&["--help", "src"]), Ok(Command::Help));
    assert_eq!(parse(&["src", "--help"]), Ok(Command::Help));
    assert_eq!(parse(&["--version", "--help"]), Ok(Command::Version));
}

#[test]
fn the_version_line_names_the_program_and_a_version() {
    let line = version();
    assert!(line.starts_with("nova-lite "), "got: {line}");
    let number = line.trim_start_matches("nova-lite ");
    assert!(
        number.split('.').count() >= 2 && number.chars().next().is_some_and(|c| c.is_ascii_digit()),
        "the version should look like a version, got: {number}"
    );
}

// ------------------------------------------------------------- errors

#[test]
fn an_unknown_option_is_an_error_that_quotes_it() {
    let err = parse(&["--versoin"]).expect_err("a typo must not be a directory");
    assert!(err.contains("--versoin"), "got: {err}");
    assert!(
        err.contains("--help"),
        "the error should say where to look: {err}"
    );
}

#[test]
fn a_short_unknown_option_is_an_error_too() {
    assert!(parse(&["-x"]).is_err());
    assert!(parse(&["-"]).is_err(), "a bare dash is not a directory");
}

#[test]
fn a_second_directory_is_an_error() {
    let err = parse(&["one", "two"]).expect_err("only one directory may be named");
    assert!(err.contains("two"), "got: {err}");
}

#[test]
fn an_option_after_the_directory_is_still_an_option() {
    // Options are not positional: `nova-lite src --nope` is a typo, not a
    // request to use two directories.
    assert!(parse(&["src", "--nope"]).is_err());
}

// ------------------------------------------------------------- the text

#[test]
fn the_usage_documents_the_argument_the_keys_and_the_environment() {
    let text = usage();

    for expected in [
        "USAGE:",
        "nova-lite [OPTIONS] [DIRECTORY]",
        "--help",
        "--version",
        "--auto-approve",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OLLAMA_BASE_URL",
        "LLM_NUM_CTX",
        "LLM_MAX_TURNS",
        "LLM_COMPACT_AT",
        "LLM_TOOL_RESULT_CHARS",
        "LLM_EVICT_CHARS",
        "NOVA_APPROVAL",
    ] {
        assert!(
            text.contains(expected),
            "the usage should mention {expected}"
        );
    }
}

#[test]
fn the_usage_documents_the_keys_the_loop_actually_handles() {
    // The usage is the only place a new user sees the key map, so a key that
    // moved must move here too. These are the bindings `handle_input` has.
    let text = usage();
    for expected in [
        "Shift+Enter",
        "Esc",
        "Ctrl+C",
        "Up, Down",
        "PageUp/Down",
        "Ctrl+Home/End",
        "Ctrl+A/E/U/K/W",
        "Ctrl+T",
        "Ctrl+G",
        "Tab",
    ] {
        assert!(
            text.contains(expected),
            "the usage should mention {expected}"
        );
    }
}

#[test]
fn the_usage_starts_with_the_version_line() {
    assert!(usage().starts_with(&version()));
}
