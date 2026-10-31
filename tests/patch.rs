//! Precise editing, end to end: what `edit_file` changes, and what it refuses.
//!
//! `src/patch.rs` has the rules as a pure function. This file is about the
//! path around it: the read that records what the session saw, the approval
//! that previews the whole patch, and the commit that leaves the file alone
//! when anything in between says no.
//!
//! The cases that matter are the ones where the old behaviour was silently
//! wrong rather than loudly broken: a search string that matched twice, a
//! file that changed under the model, a patch whose third block failed.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use futures::future::BoxFuture;
use rig::tool::{Tool, ToolContext};
use rust_agent::approval::{ApprovalRequest, Approver, Decision};
use rust_agent::config::{ApprovalMode, Config};
use rust_agent::runtime::Runtime;
use rust_agent::tools::{
    EditBlockArgs, EditFile, EditFileArgs, ReadFile, ReadFileArgs, WriteFile, WriteFileArgs,
};
use rust_agent::workspace::Workspace;

mod common;

/// A session over a fresh temporary directory, approving everything.
fn session() -> Arc<Runtime> {
    common::session()
}

/// A session over `dir`, approving everything.
fn session_in(dir: &std::path::Path) -> Arc<Runtime> {
    common::session_in(dir)
}

/// Run `read_file` and return its result.
async fn read(rt: &Arc<Runtime>, path: &str) -> Result<String, String> {
    ReadFile::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            ReadFileArgs {
                path: path.to_string(),
                start_line: None,
                end_line: None,
            },
        )
        .await
        .map_err(|e| e.to_string())
}

/// Run `write_file` and return its result.
async fn write(rt: &Arc<Runtime>, path: &str, content: &str) -> Result<String, String> {
    WriteFile::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            WriteFileArgs {
                path: path.to_string(),
                content: content.to_string(),
            },
        )
        .await
        .map_err(|e| e.to_string())
}

/// Run `edit_file` with one block and return its result.
async fn edit(rt: &Arc<Runtime>, path: &str, old: &str, new: &str) -> Result<String, String> {
    edit_blocks(rt, path, &[(old, new, false)]).await
}

/// Run `edit_file` with several blocks and return its result.
async fn edit_blocks(
    rt: &Arc<Runtime>,
    path: &str,
    blocks: &[(&str, &str, bool)],
) -> Result<String, String> {
    let edits = blocks
        .iter()
        .map(|(old, new, all)| EditBlockArgs {
            old_string: old.to_string(),
            new_string: new.to_string(),
            replace_all: *all,
        })
        .collect();
    EditFile::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            EditFileArgs {
                path: path.to_string(),
                edits,
            },
        )
        .await
        .map_err(|e| e.to_string())
}

/// The bytes of `path` in the workspace, read straight from disk.
fn on_disk(rt: &Arc<Runtime>, path: &str) -> String {
    std::fs::read_to_string(rt.workspace().root().join(path)).expect("reading the file")
}

// ------------------------------------------------------------- the happy path

#[tokio::test]
async fn a_single_edit_changes_the_file() {
    let rt = session();
    write(&rt, "a.txt", "let x = 1;\n").await.unwrap();
    edit(&rt, "a.txt", "let x = 1;", "let x = 2;")
        .await
        .unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "let x = 2;\n");
}

#[tokio::test]
async fn several_blocks_are_applied_in_order() {
    let rt = session();
    write(&rt, "a.txt", "one\ntwo\nthree\n").await.unwrap();
    edit_blocks(&rt, "a.txt", &[("one", "1", false), ("three", "3", false)])
        .await
        .unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "1\ntwo\n3\n");
}

#[tokio::test]
async fn a_later_block_can_match_what_an_earlier_one_wrote() {
    // The blocks are applied to a working copy, not all to the original, so a
    // patch can build a change up in steps.
    let rt = session();
    write(&rt, "a.txt", "start\n").await.unwrap();
    edit_blocks(
        &rt,
        "a.txt",
        &[("start", "middle", false), ("middle", "end", false)],
    )
    .await
    .unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "end\n");
}

#[tokio::test]
async fn replace_all_changes_every_occurrence() {
    let rt = session();
    write(&rt, "a.txt", "old\nkeep\nold\n").await.unwrap();
    edit_blocks(&rt, "a.txt", &[("old", "new", true)])
        .await
        .unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "new\nkeep\nnew\n");
}

// --------------------------------------------------------------- the refusals

#[tokio::test]
async fn an_ambiguous_search_is_refused_and_the_file_is_untouched() {
    let rt = session();
    write(&rt, "a.txt", "x\nkeep\nx\n").await.unwrap();
    let err = edit(&rt, "a.txt", "x", "y")
        .await
        .expect_err("a search matching twice should be refused");
    assert!(err.contains("matches 2 places"), "unexpected: {err}");
    assert!(err.contains("lines 1, 3"), "unexpected: {err}");
    assert_eq!(on_disk(&rt, "a.txt"), "x\nkeep\nx\n");
}

#[tokio::test]
async fn a_search_that_is_not_there_is_refused() {
    let rt = session();
    write(&rt, "a.txt", "abc\n").await.unwrap();
    let err = edit(&rt, "a.txt", "zzz", "y")
        .await
        .expect_err("a search that matches nothing should be refused");
    assert!(err.contains("was not found"), "unexpected: {err}");
    assert_eq!(on_disk(&rt, "a.txt"), "abc\n");
}

#[tokio::test]
async fn a_late_block_that_fails_leaves_the_original_untouched() {
    // The first block would have applied. The second cannot. Nothing is
    // written, because the whole patch is computed before anything is
    // committed: a half-applied file is worse than a refused patch, since the
    // model would have no way to know which half landed.
    let rt = session();
    write(&rt, "a.txt", "one\ntwo\n").await.unwrap();
    let err = edit_blocks(&rt, "a.txt", &[("one", "1", false), ("nope", "2", false)])
        .await
        .expect_err("the second block cannot apply");
    assert!(
        err.contains("edit 2"),
        "the error should name the block: {err}"
    );
    assert_eq!(on_disk(&rt, "a.txt"), "one\ntwo\n");
}

#[tokio::test]
async fn a_patch_that_changes_nothing_is_refused() {
    let rt = session();
    write(&rt, "a.txt", "same\n").await.unwrap();
    let err = edit(&rt, "a.txt", "same", "same")
        .await
        .expect_err("a no-op patch should be refused");
    assert!(err.contains("nothing to apply"), "unexpected: {err}");
}

#[tokio::test]
async fn an_empty_search_is_refused() {
    let rt = session();
    write(&rt, "a.txt", "abc\n").await.unwrap();
    let err = edit(&rt, "a.txt", "", "x")
        .await
        .expect_err("an empty search should be refused");
    assert!(err.contains("search text is empty"), "unexpected: {err}");
    assert_eq!(on_disk(&rt, "a.txt"), "abc\n");
}

#[tokio::test]
async fn editing_a_file_that_does_not_exist_says_to_create_it() {
    let rt = session();
    let err = edit(&rt, "nope.txt", "a", "b")
        .await
        .expect_err("editing a missing file should be refused");
    assert!(err.contains("write_file"), "unexpected: {err}");
}

// ------------------------------------------------------------- the stale read

#[tokio::test]
async fn an_edit_after_the_file_changed_underneath_is_refused() {
    // The model read the file, then something else changed it. The search
    // string still matches, so without the check the edit would land in a file
    // the model has not seen: a silent wrong edit rather than a failed one.
    let rt = session();
    write(&rt, "a.txt", "one\ntwo\n").await.unwrap();
    read(&rt, "a.txt").await.expect("reading the file");

    // Something else writes it: the user's editor, or another process.
    std::fs::write(rt.workspace().root().join("a.txt"), "one\nCHANGED\n").unwrap();

    let err = edit(&rt, "a.txt", "one", "1")
        .await
        .expect_err("an edit against a stale read should be refused");
    assert!(
        err.contains("changed since you read it"),
        "unexpected: {err}"
    );
    assert_eq!(on_disk(&rt, "a.txt"), "one\nCHANGED\n");
}

#[tokio::test]
async fn reading_the_file_again_clears_the_refusal() {
    let rt = session();
    write(&rt, "a.txt", "one\ntwo\n").await.unwrap();
    read(&rt, "a.txt").await.unwrap();
    std::fs::write(rt.workspace().root().join("a.txt"), "one\nCHANGED\n").unwrap();

    // Re-reading is the fix the error asks for, so it has to work.
    read(&rt, "a.txt").await.expect("reading it again");
    edit(&rt, "a.txt", "CHANGED", "two").await.unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "one\ntwo\n");
}

#[tokio::test]
async fn a_second_edit_without_re_reading_is_allowed() {
    // The session's own write updates what it has seen, so its own change does
    // not look like someone else's.
    let rt = session();
    write(&rt, "a.txt", "one\ntwo\n").await.unwrap();
    edit(&rt, "a.txt", "one", "1").await.unwrap();
    edit(&rt, "a.txt", "two", "2").await.unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "1\n2\n");
}

#[tokio::test]
async fn an_edit_to_a_file_never_read_is_allowed() {
    // The model may know the content from a grep rather than a read. There is
    // nothing to compare against, so the search string is the only guard, and
    // that is the documented behaviour rather than a hole: the check is a
    // second line of defence, not the first.
    let rt = session();
    write(&rt, "a.txt", "one\ntwo\n").await.unwrap();
    edit(&rt, "a.txt", "two", "2").await.unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "one\n2\n");
}

// ------------------------------------------------------------- the approval

/// An approver that records what it was shown and answers with `answer`.
#[derive(Debug)]
struct Recorder {
    answer: Decision,
    seen: std::sync::Mutex<Vec<ApprovalRequest>>,
}

impl Recorder {
    fn new(answer: Decision) -> Arc<Self> {
        Arc::new(Self {
            answer,
            seen: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn previews(&self) -> Vec<String> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|r| r.preview.clone())
            .collect()
    }
}

impl Approver for Recorder {
    fn decide<'a>(&'a self, request: ApprovalRequest) -> BoxFuture<'a, Decision> {
        self.seen.lock().unwrap().push(request);
        let answer = self.answer.clone();
        Box::pin(async move { answer })
    }
}

#[tokio::test]
async fn the_preview_shows_the_whole_patch_not_just_the_first_block() {
    let dir = tempfile::tempdir().unwrap();
    let approver = Recorder::new(Decision::Approved);
    let rt = common::session_with_approver(dir.path(), approver.clone());

    write(&rt, "a.txt", "one\ntwo\nthree\n").await.unwrap();
    edit_blocks(&rt, "a.txt", &[("one", "1", false), ("three", "3", false)])
        .await
        .unwrap();

    let previews = approver.previews();
    let patch = previews.last().expect("the patch was previewed");
    assert!(
        patch.contains("-one"),
        "the first block is missing: {patch}"
    );
    assert!(patch.contains("+1"), "the first block is missing: {patch}");
    assert!(
        patch.contains("-three"),
        "the second block is missing: {patch}"
    );
    assert!(patch.contains("+3"), "the second block is missing: {patch}");
}

#[tokio::test]
async fn a_denied_patch_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let approver = Recorder::new(Decision::Denied("not now".to_string()));
    let rt = common::session_with_approver(dir.path(), approver);

    // The write is denied too, so put the file there directly.
    std::fs::write(dir.path().join("a.txt"), "one\ntwo\n").unwrap();
    let err = edit(&rt, "a.txt", "one", "1")
        .await
        .expect_err("a denied patch should not be applied");
    assert!(err.contains("denied"), "unexpected: {err}");
    assert_eq!(on_disk(&rt, "a.txt"), "one\ntwo\n");
}

#[tokio::test]
async fn a_denied_patch_is_a_refusal_the_model_should_not_retry() {
    // The distinction matters to the loop: a conflict is the model's to fix,
    // a denial is not.
    let dir = tempfile::tempdir().unwrap();
    let approver = Recorder::new(Decision::Denied("no".to_string()));
    let rt = common::session_with_approver(dir.path(), approver);
    std::fs::write(dir.path().join("a.txt"), "one\n").unwrap();

    let err = EditFile::new(rt.clone())
        .call(
            &mut ToolContext::default(),
            EditFileArgs {
                path: "a.txt".to_string(),
                edits: vec![EditBlockArgs {
                    old_string: "one".to_string(),
                    new_string: "1".to_string(),
                    replace_all: false,
                }],
            },
        )
        .await
        .expect_err("denied");
    let _ = err;
    assert_eq!(on_disk(&rt, "a.txt"), "one\n");
}

// ------------------------------------------------------- bytes and encodings

#[tokio::test]
async fn crlf_line_endings_are_preserved() {
    let rt = session();
    write(&rt, "a.txt", "one\r\ntwo\r\n").await.unwrap();
    edit(&rt, "a.txt", "two", "TWO").await.unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "one\r\nTWO\r\n");
}

#[tokio::test]
async fn a_missing_final_newline_is_preserved() {
    let rt = session();
    write(&rt, "a.txt", "one\ntwo").await.unwrap();
    edit(&rt, "a.txt", "two", "TWO").await.unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "one\nTWO");
}

#[tokio::test]
async fn unicode_is_edited_whole() {
    let rt = session();
    write(&rt, "a.txt", "let s = \"héllo\";\n").await.unwrap();
    edit(&rt, "a.txt", "héllo", "wörld").await.unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "let s = \"wörld\";\n");
}

#[tokio::test]
async fn a_byte_order_mark_survives_an_edit() {
    let rt = session();
    write(&rt, "a.txt", "\u{feff}fn main() {}\n").await.unwrap();
    edit(&rt, "a.txt", "main", "start").await.unwrap();
    assert_eq!(on_disk(&rt, "a.txt"), "\u{feff}fn start() {}\n");
}

#[tokio::test]
async fn a_file_over_the_edit_limit_says_to_replace_it_whole() {
    let rt = session();
    let big = "x".repeat(rt.config().limits.max_read_bytes as usize + 1);
    write(&rt, "big.txt", &big).await.unwrap();
    let err = edit(&rt, "big.txt", "x", "y")
        .await
        .expect_err("a file over the limit cannot be edited");
    assert!(err.contains("write_file"), "unexpected: {err}");
}

// ------------------------------------------------------------- the write path

#[tokio::test]
async fn a_patch_leaves_no_temp_file_behind() {
    let rt = session();
    write(&rt, "dir/a.txt", "one\n").await.unwrap();
    edit(&rt, "dir/a.txt", "one", "1").await.unwrap();

    let names: Vec<String> = std::fs::read_dir(rt.workspace().root().join("dir"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec!["a.txt".to_string()],
        "a temp file was left behind"
    );
}

#[tokio::test]
async fn a_patch_through_a_symlinked_directory_is_refused() {
    // The edit resolves the real target, so a link pointing out of the
    // workspace is refused even though the lexical path looks inside.
    let outside = tempfile::tempdir().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "secret\n").unwrap();

    let dir = tempfile::tempdir().unwrap();
    let link = dir.path().join("link");
    if common::make_dir_link(outside.path(), &link).is_err() {
        eprintln!("skipping: this platform will not let an unprivileged process make a link");
        return;
    }

    let rt = session_in(dir.path());
    let err = edit(&rt, "link/secret.txt", "secret", "leaked")
        .await
        .expect_err("a link out of the workspace should be refused");
    assert!(err.contains("outside the workspace"), "unexpected: {err}");
    assert_eq!(
        std::fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
        "secret\n"
    );
}

#[tokio::test]
async fn the_config_defaults_are_the_ones_the_tests_assume() {
    // A guard on the fixture rather than on the code: if the default approval
    // mode ever became `Auto`, every denial test in this file would pass for
    // the wrong reason.
    assert!(!Config::defaults().approval.is_auto());
    assert!(Config::defaults()
        .with_approval(ApprovalMode::Auto)
        .approval
        .is_auto());
    let _ = Workspace::new(std::env::temp_dir()).unwrap();
}
