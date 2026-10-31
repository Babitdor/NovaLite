//! Two sessions in one process do not see each other's workspace.
//!
//! The workspace root used to be a process global: the first caller to set it
//! won, and every tool afterwards resolved against that one directory. That
//! made the tests order-dependent (whichever ran first fixed the root for the
//! rest of the binary) and made it impossible to run two agents over two
//! projects in one process.
//!
//! The root is now a value the session carries. These cases are what proves
//! it: the same relative path, in two sessions, is two different files, and a
//! path that is inside one session is outside the other.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rig::tool::{Tool, ToolContext};
use rust_agent::runtime::Runtime;
use rust_agent::tools::{ReadFile, ReadFileArgs, WriteFile, WriteFileArgs};
use std::sync::Arc;

mod common;

/// Write `content` to `path` through `rt`'s tools.
async fn write(rt: &Arc<Runtime>, path: &str, content: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    WriteFile::new(rt.clone())
        .call(
            &mut ctx,
            WriteFileArgs {
                path: path.to_string(),
                content: content.to_string(),
            },
        )
        .await
        .map_err(|e| e.to_string())
}

/// Read `path` through `rt`'s tools.
async fn read(rt: &Arc<Runtime>, path: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    ReadFile::new(rt.clone())
        .call(
            &mut ctx,
            ReadFileArgs {
                path: path.to_string(),
                start_line: None,
                end_line: None,
            },
        )
        .await
        .map_err(|e| e.to_string())
}

#[tokio::test]
async fn the_same_relative_path_is_two_different_files_in_two_sessions() {
    let a = common::session();
    let b = common::session();

    write(&a, "note.txt", "from a\n")
        .await
        .expect("writing in a");
    write(&b, "note.txt", "from b\n")
        .await
        .expect("writing in b");

    assert_eq!(read(&a, "note.txt").await.unwrap(), "from a\n");
    assert_eq!(read(&b, "note.txt").await.unwrap(), "from b\n");
}

#[tokio::test]
async fn a_file_written_in_one_session_is_not_visible_in_the_other() {
    let a = common::session();
    let b = common::session();

    write(&a, "only_in_a.txt", "x").await.expect("writing in a");

    // The same relative path in the other session is a missing file, not the
    // one the first session wrote: the two roots are different directories.
    let err = read(&b, "only_in_a.txt")
        .await
        .expect_err("b must not see a's file");
    assert!(
        !err.contains("outside the workspace"),
        "the file is inside b's root, it is simply absent: {err}"
    );
}

#[tokio::test]
async fn one_sessions_root_is_outside_the_other() {
    let a = common::session();
    let b = common::session();

    assert_ne!(
        a.workspace().root(),
        b.workspace().root(),
        "two sessions must not share a root"
    );

    // An absolute path inside `a` is outside `b`, so `b` refuses it. This is
    // the confinement rule holding across sessions rather than within one.
    let inside_a = a.workspace().root().join("note.txt");
    write(&a, "note.txt", "x").await.expect("writing in a");

    let err = read(&b, &inside_a.display().to_string())
        .await
        .expect_err("b must refuse a path inside a");
    assert!(
        err.contains("outside the workspace"),
        "the refusal should name the reason: {err}"
    );
}

#[tokio::test]
async fn a_session_keeps_its_own_budgets() {
    // The budgets are part of the session too, so two sessions can disagree
    // about them without either reading the environment again.
    let mut config = rust_agent::config::Config::defaults()
        .with_approval(rust_agent::config::ApprovalMode::Auto);
    config.context.evict_chars = 1234;
    config.limits.max_turns = 7;

    let small = common::session_with(config);
    let default = common::session();

    assert_eq!(small.config().context.evict_chars, 1234);
    assert_eq!(small.config().limits.max_turns, 7);
    assert_ne!(default.config().context.evict_chars, 1234);
    assert_ne!(default.config().limits.max_turns, 7);
}
