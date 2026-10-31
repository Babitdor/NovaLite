//! Bounded, ignore-aware search, end to end.
//!
//! `src/search.rs` tests the walker on its own. This file drives the tools
//! that use it, through a real session, because the properties that matter are
//! about what the *model* is handed: a path it can put back into another tool
//! call, a result that says when it is incomplete, and a search that stops
//! when the run is cancelled.
//!
//! The tree every test builds is the same shape, because the ignore rules are
//! about shape:
//!
//! ```text
//! .gitignore          target/
//! src/lib.rs          needle
//! src/main.rs         needle
//! src/nested/.gitignore   *.tmp
//! src/nested/keep.rs  needle
//! src/nested/drop.tmp needle
//! target/debug/out.rs needle
//! .hidden/secret.rs   needle
//! README.md           needle
//! ```
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;

use rig::tool::Tool;
use rig::tool::ToolContext;
use rust_agent::config::{Config, Limits};
use rust_agent::runtime::Runtime;
use rust_agent::tools::{
    Glob, GlobArgs, Grep, GrepArgs, ListDir, ListDirArgs, ReadFile, ReadFileArgs,
};

/// A tree with the shapes the ignore rules care about, and the session over it.
fn tree() -> (tempfile::TempDir, Arc<Runtime>) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let root = dir.path();

    std::fs::create_dir_all(root.join("src/nested")).unwrap();
    std::fs::create_dir_all(root.join("target/debug")).unwrap();
    std::fs::create_dir_all(root.join(".hidden")).unwrap();

    std::fs::write(root.join(".gitignore"), "target/\n").unwrap();
    std::fs::write(root.join("src/nested/.gitignore"), "*.tmp\n").unwrap();
    std::fs::write(root.join("src/lib.rs"), "pub fn needle() {}\n").unwrap();
    std::fs::write(root.join("src/main.rs"), "// needle\nfn main() {}\n").unwrap();
    std::fs::write(root.join("src/nested/keep.rs"), "needle kept\n").unwrap();
    std::fs::write(root.join("src/nested/drop.tmp"), "needle dropped\n").unwrap();
    std::fs::write(root.join("target/debug/out.rs"), "needle built\n").unwrap();
    std::fs::write(root.join(".hidden/secret.rs"), "needle hidden\n").unwrap();
    std::fs::write(root.join("README.md"), "needle in a readme\n").unwrap();

    let rt = common::session_in(root);
    (dir, rt)
}

async fn grep(rt: &Arc<Runtime>, pattern: &str) -> Result<String, String> {
    grep_with(rt, pattern, false, false).await
}

async fn grep_with(
    rt: &Arc<Runtime>,
    pattern: &str,
    hidden: bool,
    ignored: bool,
) -> Result<String, String> {
    Grep::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            GrepArgs {
                pattern: pattern.to_string(),
                path: ".".to_string(),
                hidden,
                ignored,
            },
        )
        .await
        .map_err(|e| e.to_string())
}

async fn glob(rt: &Arc<Runtime>, pattern: &str) -> Result<String, String> {
    Glob::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            GlobArgs {
                pattern: pattern.to_string(),
                path: ".".to_string(),
                hidden: false,
                ignored: false,
            },
        )
        .await
        .map_err(|e| e.to_string())
}

async fn list(rt: &Arc<Runtime>, path: &str) -> Result<String, String> {
    ListDir::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            ListDirArgs {
                path: path.to_string(),
            },
        )
        .await
        .map_err(|e| e.to_string())
}

async fn read(
    rt: &Arc<Runtime>,
    path: &str,
    range: Option<(usize, usize)>,
) -> Result<String, String> {
    ReadFile::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            ReadFileArgs {
                path: path.to_string(),
                start_line: range.map(|(s, _)| s),
                end_line: range.map(|(_, e)| e),
            },
        )
        .await
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------- the walk

#[tokio::test]
async fn grep_respects_the_ignore_files() {
    let (_dir, rt) = tree();
    let out = grep(&rt, "needle").await.expect("grep failed");

    assert!(out.contains("src/lib.rs:1:"), "unexpected: {out}");
    assert!(out.contains("src/main.rs:1:"), "unexpected: {out}");
    assert!(out.contains("src/nested/keep.rs:1:"), "unexpected: {out}");
    assert!(out.contains("README.md:1:"), "unexpected: {out}");
    assert!(
        !out.contains("target/"),
        "a .gitignore rule must exclude the build tree: {out}"
    );
    assert!(
        !out.contains("drop.tmp"),
        "a nested .gitignore must apply to its own directory: {out}"
    );
    assert!(
        !out.contains(".hidden"),
        "a dot-directory is not searched by default: {out}"
    );
}

#[tokio::test]
async fn a_nested_ignore_file_applies_to_its_own_directory_only() {
    let (_dir, rt) = tree();
    // `*.tmp` is written in `src/nested/.gitignore`, so it must not reach
    // `src/`. A walker that read one ignore file for the whole tree would
    // pass the previous test and fail this one.
    std::fs::write(rt.workspace().root().join("src/keep.tmp"), "needle top\n").unwrap();

    let out = grep(&rt, "needle").await.expect("grep failed");
    assert!(out.contains("src/keep.tmp"), "unexpected: {out}");
    assert!(!out.contains("nested/drop.tmp"), "unexpected: {out}");
}

#[tokio::test]
async fn asking_for_ignored_files_includes_them() {
    let (_dir, rt) = tree();
    let out = grep_with(&rt, "needle", false, true)
        .await
        .expect("grep failed");

    assert!(out.contains("target/debug/out.rs"), "unexpected: {out}");
    assert!(out.contains("nested/drop.tmp"), "unexpected: {out}");
    assert!(
        !out.contains(".hidden"),
        "asking for ignored files is not asking for hidden ones: {out}"
    );
}

#[tokio::test]
async fn asking_for_hidden_files_includes_them() {
    let (_dir, rt) = tree();
    let out = grep_with(&rt, "needle", true, false)
        .await
        .expect("grep failed");

    assert!(out.contains(".hidden/secret.rs"), "unexpected: {out}");
    assert!(
        !out.contains("target/"),
        "asking for hidden files is not asking for ignored ones: {out}"
    );
}

#[tokio::test]
async fn a_result_path_is_relative_and_uses_forward_slashes() {
    let (_dir, rt) = tree();
    let out = grep(&rt, "needle").await.expect("grep failed");

    // The model puts this path back into another tool call, and the tools
    // resolve against the root. An absolute path would be refused as outside
    // the workspace, so a result that reported one would be unusable.
    assert!(out.contains("src/lib.rs:1:"), "unexpected: {out}");
    assert!(
        !out.contains(&rt.workspace().root().display().to_string()),
        "a result must not carry the absolute root: {out}"
    );
    assert!(
        !out.contains('\\'),
        "a result must use forward slashes: {out}"
    );
}

#[tokio::test]
async fn a_search_that_finds_nothing_says_so() {
    let (_dir, rt) = tree();
    let out = grep(&rt, "no-such-token-anywhere")
        .await
        .expect("grep failed");
    assert!(out.contains("No matches"), "unexpected: {out}");
}

#[tokio::test]
async fn a_bad_pattern_is_an_error_not_a_panic() {
    let (_dir, rt) = tree();
    let err = grep(&rt, "a(b")
        .await
        .expect_err("an unclosed group is invalid");
    assert!(err.contains("invalid regex"), "unexpected: {err}");
}

// ------------------------------------------------------------- the bounds

/// A session over `dir` whose search limits are `limits`.
fn session_with_limits(dir: &std::path::Path, limits: Limits) -> Arc<Runtime> {
    let mut config = Config::defaults();
    config.limits = limits;
    common::session_in_with(dir, config)
}

#[tokio::test]
async fn the_match_limit_stops_the_walk_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..50 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), "needle\n").unwrap();
    }
    let mut limits = Config::defaults().limits;
    limits.max_search_matches = 10;
    let rt = session_with_limits(dir.path(), limits);

    let out = grep(&rt, "needle").await.expect("grep failed");
    let hits = out.lines().filter(|l| l.contains(":1:")).count();
    assert_eq!(hits, 10, "the limit must be the number returned: {out}");
    assert!(
        out.contains("incomplete") && out.contains("match limit"),
        "a capped result must say it is capped: {out}"
    );
}

#[tokio::test]
async fn the_byte_limit_stops_the_walk_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..50 {
        std::fs::write(
            dir.path().join(format!("f{i}.txt")),
            format!("needle {}\n", "x".repeat(200)),
        )
        .unwrap();
    }
    let mut limits = Config::defaults().limits;
    limits.max_search_bytes = 500;
    let rt = session_with_limits(dir.path(), limits);

    let out = grep(&rt, "needle").await.expect("grep failed");
    assert!(
        out.contains("incomplete") && out.contains("output limit"),
        "a capped result must say it is capped: {out}"
    );
}

#[tokio::test]
async fn the_file_limit_stops_the_walk_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..50 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), "nothing here\n").unwrap();
    }
    let mut limits = Config::defaults().limits;
    limits.max_search_files = 5;
    let rt = session_with_limits(dir.path(), limits);

    let out = grep(&rt, "needle").await.expect("grep failed");
    assert!(
        out.contains("incomplete") && out.contains("file limit"),
        "a capped result must say it is capped: {out}"
    );
}

#[tokio::test]
async fn a_file_over_the_size_cap_is_skipped_and_named() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("small.txt"), "needle\n").unwrap();
    std::fs::write(dir.path().join("big.txt"), "needle\n".repeat(5_000)).unwrap();
    let mut limits = Config::defaults().limits;
    limits.max_search_file_bytes = 1_000;
    let rt = session_with_limits(dir.path(), limits);

    let out = grep(&rt, "needle").await.expect("grep failed");
    assert!(out.contains("small.txt"), "unexpected: {out}");
    assert!(
        !out.contains("big.txt:1:"),
        "a file over the cap must not be searched: {out}"
    );
    assert!(
        out.contains("big.txt") && out.contains("over the search limit"),
        "the skipped file must be named: {out}"
    );
}

#[tokio::test]
async fn a_binary_file_is_skipped_rather_than_searched() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "needle\n").unwrap();
    std::fs::write(dir.path().join("a.bin"), b"needle\x00\x01\x02needle\n").unwrap();
    let rt = common::session_in(dir.path());

    let out = grep(&rt, "needle").await.expect("grep failed");
    assert!(out.contains("a.txt:1:"), "unexpected: {out}");
    assert!(
        !out.contains("a.bin:1:"),
        "a binary file must not be searched: {out}"
    );
    assert!(
        out.contains("binary file(s) skipped"),
        "the skip must be reported: {out}"
    );
}

#[tokio::test]
async fn a_cancelled_search_ends_early_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..200 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), "needle\n").unwrap();
    }
    let rt = common::session_in(dir.path());
    // Stop before the walk starts. The point is that the flag is consulted at
    // all: a walk that ignored it would return all two hundred.
    rt.cancel().stop();

    let out = grep(&rt, "needle").await.expect("grep failed");
    assert!(
        out.contains("cancelled"),
        "a stopped search must say it was stopped: {out}"
    );
    assert!(
        !out.contains("f199.txt"),
        "a stopped search must not have walked the whole tree: {out}"
    );
}

// ------------------------------------------------------------------ glob

#[tokio::test]
async fn glob_respects_the_ignore_files_too() {
    let (_dir, rt) = tree();
    let out = glob(&rt, "**/*.rs").await.expect("glob failed");

    assert!(out.contains("src/lib.rs"), "unexpected: {out}");
    assert!(out.contains("src/nested/keep.rs"), "unexpected: {out}");
    assert!(
        !out.contains("target/debug/out.rs"),
        "glob and grep must agree on what is in the workspace: {out}"
    );
    assert!(!out.contains(".hidden"), "unexpected: {out}");
}

#[tokio::test]
async fn glob_is_bounded_too() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..50 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), "").unwrap();
    }
    let mut limits = Config::defaults().limits;
    limits.max_search_matches = 7;
    let rt = session_with_limits(dir.path(), limits);

    let out = glob(&rt, "*.txt").await.expect("glob failed");
    let hits = out.lines().filter(|l| l.ends_with(".txt")).count();
    assert_eq!(hits, 7, "the limit must be the number returned: {out}");
    assert!(out.contains("incomplete"), "unexpected: {out}");
}

#[tokio::test]
async fn a_bad_glob_is_an_error_not_a_panic() {
    let (_dir, rt) = tree();
    let err = glob(&rt, "a[")
        .await
        .expect_err("an unclosed class is invalid");
    assert!(err.contains("invalid glob"), "unexpected: {err}");
}

// -------------------------------------------------------------- list_dir

#[tokio::test]
async fn list_dir_shows_what_the_walker_would_hide() {
    let (_dir, rt) = tree();
    let out = list(&rt, ".").await.expect("list failed");

    // An explicit `list_dir` is a request for what is in the directory the
    // caller named. Hiding an ignored entry from it would make the tool lie
    // about that directory, so the ignore rules do not apply here.
    assert!(out.contains("target/"), "unexpected: {out}");
    assert!(out.contains(".hidden/"), "unexpected: {out}");
    assert!(out.contains("src/"), "unexpected: {out}");
}

#[tokio::test]
async fn list_dir_is_bounded() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..50 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), "").unwrap();
    }
    let mut limits = Config::defaults().limits;
    limits.max_search_matches = 6;
    let rt = session_with_limits(dir.path(), limits);

    let out = list(&rt, ".").await.expect("list failed");
    let hits = out.lines().filter(|l| l.ends_with(".txt")).count();
    assert_eq!(hits, 6, "the limit must be the number returned: {out}");
    assert!(out.contains("incomplete"), "unexpected: {out}");
}

// ------------------------------------------------------- line-range reads

#[tokio::test]
async fn a_line_range_reads_one_part_of_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let body: String = (1..=100).map(|i| format!("line {i}\n")).collect();
    std::fs::write(dir.path().join("long.txt"), &body).unwrap();
    let rt = common::session_in(dir.path());

    let out = read(&rt, "long.txt", Some((40, 42)))
        .await
        .expect("read failed");
    assert!(out.contains("line 40"), "unexpected: {out}");
    assert!(out.contains("line 42"), "unexpected: {out}");
    assert!(!out.contains("line 39"), "unexpected: {out}");
    assert!(!out.contains("line 43"), "unexpected: {out}");
    assert!(
        out.contains("lines 40-42 of 100"),
        "the range must be stated so the model knows what it did not see: {out}"
    );
}

#[tokio::test]
async fn a_line_range_past_the_end_is_clamped_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("short.txt"), "one\ntwo\n").unwrap();
    let rt = common::session_in(dir.path());

    let out = read(&rt, "short.txt", Some((1, 99)))
        .await
        .expect("read failed");
    assert!(out.contains("one"), "unexpected: {out}");
    assert!(out.contains("two"), "unexpected: {out}");
    assert!(out.contains("lines 1-2 of 2"), "unexpected: {out}");
}

#[tokio::test]
async fn a_backwards_line_range_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("short.txt"), "one\ntwo\n").unwrap();
    let rt = common::session_in(dir.path());

    let err = read(&rt, "short.txt", Some((5, 2)))
        .await
        .expect_err("a backwards range is a mistake, not an empty result");
    assert!(err.contains("before start_line"), "unexpected: {err}");
}

#[tokio::test]
async fn reading_a_range_still_counts_as_having_read_the_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "alpha\nbeta\ngamma\n").unwrap();
    let rt = common::session_in(dir.path());

    // The digest the service takes is over the whole file, so a range read
    // must leave the session able to edit the file. If it did not, the model
    // would be told to read a file it had just read.
    read(&rt, "f.txt", Some((2, 2))).await.expect("read failed");
    let edited = rust_agent::tools::EditFile::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            rust_agent::tools::EditFileArgs {
                path: "f.txt".to_string(),
                edits: vec![rust_agent::tools::EditBlockArgs {
                    old_string: "beta".to_string(),
                    new_string: "BETA".to_string(),
                    replace_all: false,
                }],
            },
        )
        .await;
    assert!(
        edited.is_ok(),
        "a range read must count as a read: {edited:?}"
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("f.txt")).unwrap(),
        "alpha\nBETA\ngamma\n"
    );
}
