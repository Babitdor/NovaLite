//! Filesystem confinement: a path that is lexically inside the workspace but
//! *reaches* outside it.
//!
//! The lexical check in `workspace::resolve` refuses `../secrets` and
//! `C:\Windows`. It cannot refuse `link/secret.txt`, where `link` is a
//! symlink or a Windows junction, because the string is inside the root and
//! only the filesystem knows where it leads. These tests are about the second
//! check that follows the link and re-tests the real target.
//!
//! The link-creating tests need a privilege: on Windows, `symlink_file` needs
//! developer mode or an elevated process, and a junction needs neither. Each
//! test tries a symlink, falls back to a junction, and reports plainly if
//! neither is available rather than passing silently.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig::tool::{Tool, ToolContext};
use rust_agent::approval::{ActionKind, Decision, ScriptedApprover};
use rust_agent::runtime::Runtime;
use rust_agent::tools;

/// A workspace over a fresh temp directory in auto mode.
///
/// Auto mode is deliberate: these tests are about the confinement check
/// itself, which is what refuses an escape when nobody can be asked. In
/// prompt mode the same path is a question (covered in `tests/approval.rs`),
/// and an approving answer legitimately grants the link's real target.
fn session(dir: &Path) -> Arc<Runtime> {
    common::session_in(dir)
}

/// A directory outside the workspace, with a file in it, and a link to it
/// from inside the workspace.
///
/// Returns `None` when the platform will not let this process create a link,
/// which is a fact about the machine rather than a failure of the code.
fn escape_fixture() -> Option<(tempfile::TempDir, tempfile::TempDir, PathBuf)> {
    let outside = tempfile::tempdir().expect("a temporary directory");
    std::fs::write(outside.path().join("secret.txt"), "classified\n").unwrap();

    let inside = tempfile::tempdir().expect("a temporary directory");
    let link = inside.path().join("link");
    if make_dir_link(outside.path(), &link).is_err() {
        return None;
    }
    Some((outside, inside, link))
}

/// Point `link` at the directory `target`, by symlink if the process may, by
/// junction if not.
///
/// Two definitions rather than one `cfg`-gated body: a body whose last
/// statement is a `return` inside a `cfg` block is a needless return on the
/// platform where the other block is compiled out, and the lint is right.
#[cfg(windows)]
fn make_dir_link(target: &Path, link: &Path) -> std::io::Result<()> {
    if std::os::windows::fs::symlink_dir(target, link).is_ok() {
        return Ok(());
    }
    // A junction needs no privilege. `mklink /J` is the only way to make one
    // from the standard library's surface.
    let out = std::process::Command::new("cmd")
        .args(["/C", "mklink", "/J"])
        .arg(link)
        .arg(target)
        .output()?;
    if out.status.success() {
        return Ok(());
    }
    Err(std::io::Error::other("neither a symlink nor a junction"))
}

#[cfg(not(windows))]
fn make_dir_link(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

/// Point `link` at the file `target`, by symlink if the process may.
#[cfg(windows)]
fn make_file_link(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_file(target, link)
}

#[cfg(not(windows))]
fn make_file_link(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

/// Run `read_file` and return its result.
async fn read(rt: &Arc<Runtime>, path: &str) -> Result<String, String> {
    tools::ReadFile::new(rt.clone())
        .call(
            &mut ToolContext::new(),
            tools::ReadFileArgs {
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
    tools::WriteFile::new(rt.clone())
        .call(
            &mut ToolContext::new(),
            tools::WriteFileArgs {
                path: path.to_string(),
                content: content.to_string(),
            },
        )
        .await
        .map_err(|e| e.to_string())
}

/// Run `edit_file` and return its result.
async fn edit(rt: &Arc<Runtime>, path: &str, old: &str, new: &str) -> Result<String, String> {
    tools::EditFile::new(rt.clone())
        .call(
            &mut ToolContext::new(),
            tools::EditFileArgs {
                path: path.to_string(),
                edits: vec![tools::EditBlockArgs {
                    old_string: old.to_string(),
                    new_string: new.to_string(),
                    replace_all: false,
                }],
            },
        )
        .await
        .map_err(|e| e.to_string())
}

/// Run `list_dir` and return its result.
async fn list(rt: &Arc<Runtime>, path: &str) -> Result<String, String> {
    tools::ListDir::new(rt.clone())
        .call(
            &mut ToolContext::new(),
            tools::ListDirArgs {
                path: path.to_string(),
            },
        )
        .await
        .map_err(|e| e.to_string())
}

/// Every entry in `dir`, so a test can assert that nothing was left behind.
fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("the directory is readable")
        .map(|e| {
            e.expect("a readable entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

// ------------------------------------------------------------- the escape

#[tokio::test]
async fn a_link_out_of_the_workspace_is_refused_for_reading() {
    let Some((outside, inside, _link)) = escape_fixture() else {
        eprintln!("skipped: this process may not create a directory link");
        return;
    };
    let rt = session(inside.path());

    let err = read(&rt, "link/secret.txt").await.expect_err("outside");
    assert!(err.contains("outside the workspace"), "got: {err}");
    // The file is still there and still readable by its real path: the check
    // refused the access, it did not delete anything.
    assert_eq!(
        std::fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
        "classified\n"
    );
}

#[tokio::test]
async fn a_link_out_of_the_workspace_is_refused_for_writing() {
    let Some((outside, inside, _link)) = escape_fixture() else {
        eprintln!("skipped: this process may not create a directory link");
        return;
    };
    let rt = session(inside.path());

    let err = write(&rt, "link/planted.txt", "pwned\n")
        .await
        .expect_err("outside");
    assert!(err.contains("outside the workspace"), "got: {err}");
    assert!(
        !outside.path().join("planted.txt").exists(),
        "a write through a link must not land outside the workspace"
    );
}

#[tokio::test]
async fn a_link_out_of_the_workspace_is_refused_for_editing() {
    let Some((outside, inside, _link)) = escape_fixture() else {
        eprintln!("skipped: this process may not create a directory link");
        return;
    };
    let rt = session(inside.path());

    let err = edit(&rt, "link/secret.txt", "classified", "leaked")
        .await
        .expect_err("outside");
    assert!(err.contains("outside the workspace"), "got: {err}");
    assert_eq!(
        std::fs::read_to_string(outside.path().join("secret.txt")).unwrap(),
        "classified\n"
    );
}

#[tokio::test]
async fn a_link_out_of_the_workspace_is_refused_for_listing() {
    let Some((_outside, inside, _link)) = escape_fixture() else {
        eprintln!("skipped: this process may not create a directory link");
        return;
    };
    let rt = session(inside.path());

    let err = list(&rt, "link").await.expect_err("outside");
    assert!(err.contains("outside the workspace"), "got: {err}");
}

#[tokio::test]
async fn a_link_that_stays_inside_the_workspace_is_allowed() {
    // The check is on where the link *leads*, not on the fact that it is a
    // link. Refusing every link would refuse a legitimate layout, and a rule
    // that is wrong about the safe case is a rule the user learns to work
    // around.
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("real")).unwrap();
    std::fs::write(dir.path().join("real/a.txt"), "hello\n").unwrap();
    if make_dir_link(&dir.path().join("real"), &dir.path().join("alias")).is_err() {
        eprintln!("skipped: this process may not create a directory link");
        return;
    }
    let rt = session(dir.path());

    let out = read(&rt, "alias/a.txt").await.expect("inside the root");
    assert!(out.contains("hello"), "got: {out}");
}

#[tokio::test]
async fn a_link_that_does_not_exist_yet_is_still_checked() {
    // The target does not exist, so there is nothing to canonicalise at the
    // end of the path. The check has to be on the deepest ancestor that does
    // exist, or a write to a new file through a link would slip past.
    let Some((outside, inside, _link)) = escape_fixture() else {
        eprintln!("skipped: this process may not create a directory link");
        return;
    };
    let rt = session(inside.path());

    let err = write(&rt, "link/deep/new.txt", "pwned\n")
        .await
        .expect_err("outside");
    assert!(err.contains("outside the workspace"), "got: {err}");
    assert!(!outside.path().join("deep").exists());
}

// ------------------------------------------------------- path forms

#[tokio::test]
async fn a_verbatim_path_is_refused() {
    // `\\?\C:\work` is not normalised by the OS, so `..` inside it stays
    // literal and the lexical check sees a path the filesystem will not.
    let dir = tempfile::tempdir().unwrap();
    let rt = session(dir.path());
    let verbatim = format!(r"\\?\{}\a.txt", dir.path().display());

    let err = read(&rt, &verbatim).await.expect_err("verbatim");
    assert!(err.contains("verbatim"), "got: {err}");
}

#[tokio::test]
async fn a_unc_path_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let rt = session(dir.path());

    let err = read(&rt, r"\\server\share\a.txt").await.expect_err("unc");
    assert!(err.contains("UNC"), "got: {err}");
}

#[tokio::test]
async fn a_device_path_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let rt = session(dir.path());

    let err = read(&rt, r"\\.\C:").await.expect_err("device");
    assert!(err.contains("verbatim or device"), "got: {err}");
}

#[tokio::test]
async fn a_drive_relative_path_is_refused() {
    // `C:foo` means "foo under the current directory on drive C", which is
    // not the workspace and is not knowable from the string.
    let dir = tempfile::tempdir().unwrap();
    let rt = session(dir.path());

    let err = read(&rt, "C:foo.txt").await.expect_err("drive-relative");
    assert!(err.contains("drive"), "got: {err}");
}

#[tokio::test]
async fn an_alternate_data_stream_is_refused() {
    // `a.txt:stream` is a second file attached to the first, which the
    // containment check does not see.
    let dir = tempfile::tempdir().unwrap();
    let rt = session(dir.path());

    let err = read(&rt, "a.txt:hidden").await.expect_err("ads");
    assert!(err.contains("alternate data stream"), "got: {err}");
}

#[tokio::test]
async fn a_nul_byte_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let rt = session(dir.path());

    let err = read(&rt, "a\0b").await.expect_err("nul");
    assert!(err.contains("NUL"), "got: {err}");
}

#[tokio::test]
async fn an_empty_path_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let rt = session(dir.path());

    let err = read(&rt, "").await.expect_err("empty");
    assert!(err.contains("empty"), "got: {err}");
}

#[tokio::test]
async fn a_traversal_is_put_to_the_user() {
    // A traversal is not refused on sight any more: it is a question, and the
    // user's answer is what decides. What this pins is that the question is
    // always asked, in every shape a traversal can take, and that a refusal
    // is the end of it.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("project");
    std::fs::create_dir(&root).unwrap();
    let paths = ["../escape.txt", "a/../../escape.txt", "..", "../"];
    let approver = Arc::new(ScriptedApprover::new(
        paths
            .iter()
            .map(|_| Decision::Denied("stay in the workspace".to_string()))
            .collect(),
    ));
    let rt = common::session_with_approver(&root, approver.clone());

    for path in paths {
        let err = read(&rt, path).await.expect_err(path);
        assert!(err.contains("stay in the workspace"), "{path}: {err}");
    }
    assert_eq!(
        approver.count(),
        paths.len(),
        "every traversal is a question, not a refusal"
    );
    // The question names the directory, not the path as written: `..` and
    // `a/../..` are the same question, and only the resolved form says so.
    let asked = approver.seen();
    assert!(
        asked
            .iter()
            .all(|r| r.kind == ActionKind::Access && r.preview.contains("outside the workspace")),
        "{asked:?}"
    );
}

// ------------------------------------------------------- non-regular files

#[tokio::test]
async fn a_directory_is_not_read_as_a_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    let rt = session(dir.path());

    let err = read(&rt, "sub").await.expect_err("a directory");
    assert!(err.contains("is a directory"), "got: {err}");
    assert!(
        err.contains("list_dir"),
        "the error should say what to do: {err}"
    );
}

#[tokio::test]
async fn a_directory_is_not_edited_as_a_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sub")).unwrap();
    let rt = session(dir.path());

    let err = edit(&rt, "sub", "a", "b").await.expect_err("a directory");
    assert!(err.contains("not a regular file"), "got: {err}");
}

#[tokio::test]
async fn a_link_is_marked_as_one_in_a_listing() {
    // The listing is of this directory. A name that leads somewhere else is
    // not a name in it, and saying so is what stops the model from reading
    // through it and being refused for a reason it cannot see.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("real.txt"), "hello\n").unwrap();
    if make_file_link(&dir.path().join("real.txt"), &dir.path().join("alias.txt")).is_err() {
        eprintln!("skipped: this process may not create a file link");
        return;
    }
    let rt = session(dir.path());

    let out = list(&rt, ".").await.expect("the root lists");
    assert!(out.contains("alias.txt@"), "got: {out}");
    assert!(out.contains("real.txt"), "got: {out}");
}

// ------------------------------------------------------- safe replacement

#[tokio::test]
async fn a_write_leaves_no_temp_file_behind() {
    let dir = tempfile::tempdir().unwrap();
    let rt = session(dir.path());

    write(&rt, "a.txt", "hello\n").await.expect("approved");
    assert_eq!(entries(dir.path()), vec!["a.txt".to_string()]);
}

#[tokio::test]
async fn a_failed_replacement_preserves_the_original() {
    // The target is a directory, so the rename cannot land. The old code fell
    // back to a truncating write here, which is the failure this exists to
    // prevent: a file that is neither the old content nor the new one.
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("a.txt")).unwrap();
    std::fs::write(dir.path().join("a.txt/keep.txt"), "keep\n").unwrap();
    let rt = session(dir.path());

    let err = write(&rt, "a.txt", "clobbered\n")
        .await
        .expect_err("the rename cannot land");
    assert!(!err.is_empty(), "the failure is reported");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt/keep.txt")).unwrap(),
        "keep\n",
        "the directory and its contents must survive"
    );
    assert_eq!(
        entries(dir.path()),
        vec!["a.txt".to_string()],
        "the temp file must be cleaned up"
    );
}

#[cfg(windows)]
#[tokio::test]
async fn a_read_only_file_is_not_replaced() {
    // On Windows a rename over a read-only file is refused, and the refusal
    // must leave the file alone. On Unix the permission is on the directory,
    // so the same write succeeds; that is a platform difference, not a bug,
    // which is why this case is gated.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.txt");
    std::fs::write(&path, "original\n").unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_readonly(true);
    std::fs::set_permissions(&path, perms).unwrap();

    let rt = session(dir.path());
    let result = write(&rt, "a.txt", "replacement\n").await;

    // Restore before asserting, so a failure does not leave a read-only file
    // in the temp tree.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perms.set_readonly(false);
    std::fs::set_permissions(&path, perms).unwrap();

    assert!(result.is_err(), "a read-only file must not be replaced");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "original\n");
    assert_eq!(entries(dir.path()), vec!["a.txt".to_string()]);
}

#[tokio::test]
async fn a_write_preserves_the_file_permissions() {
    // The rename installs a new file, so the target's permissions have to be
    // carried over: a script that was executable must stay executable.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.sh");
    std::fs::write(&path, "#!/bin/sh\necho old\n").unwrap();
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_readonly(true);
    std::fs::set_permissions(&path, perms).unwrap();

    let rt = session(dir.path());
    // The write may be refused on Windows (read-only target) and succeed on
    // Unix. Either way the permission must not have been widened.
    let _ = write(&rt, "run.sh", "#!/bin/sh\necho new\n").await;

    let after = std::fs::metadata(&path).unwrap().permissions();
    assert!(
        after.readonly(),
        "a read-only file must not come back writable"
    );
}

// ------------------------------------------------------------- size bounds

#[tokio::test]
async fn a_read_of_a_huge_file_is_bounded_and_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.txt");
    // Just over the limit, so the test does not spend a second writing.
    let limit = rust_agent::config::DEFAULT_MAX_READ_BYTES as usize;
    let mut content = "x".repeat(limit + 1024);
    content.push('\n');
    std::fs::write(&path, &content).unwrap();

    let rt = session(dir.path());
    let out = read(&rt, "big.txt").await.expect("readable");

    assert!(
        out.len() < content.len(),
        "the result must be smaller than the file"
    );
    assert!(
        out.contains("only the first"),
        "the model must be told the read was partial: {}",
        &out[out.len().saturating_sub(300)..]
    );
}

#[tokio::test]
async fn an_edit_of_a_huge_file_is_refused_with_a_way_forward() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("big.txt");
    let limit = rust_agent::config::DEFAULT_MAX_READ_BYTES as usize;
    std::fs::write(&path, "x".repeat(limit + 1024)).unwrap();

    let rt = session(dir.path());
    let err = edit(&rt, "big.txt", "x", "y").await.expect_err("too large");
    assert!(err.contains("over the"), "got: {err}");
    assert!(
        err.contains("write_file"),
        "the error should say what to do: {err}"
    );
}
