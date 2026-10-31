//! The workspace confinement: what the tools may touch, and what they refuse.
//!
//! These drive the real tools rather than the helper they call, so they prove
//! the boundary sits in the tool call path. The root is a directory of this
//! test binary's own under the temp dir, so nothing here can reach the real
//! project.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rig::tool::{Tool, ToolContext};
use rust_agent::runtime::Runtime;
use rust_agent::tools::{
    EditBlockArgs, EditFile, EditFileArgs, Glob, GlobArgs, Grep, GrepArgs, ListDir, ListDirArgs,
    ReadFile, ReadFileArgs, Shell, ShellArgs, WriteFile, WriteFileArgs,
};
use rust_agent::workspace;
use std::path::PathBuf;
use std::sync::Arc;

mod common;

/// The workspace root for this test binary.
fn root() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("nova_lite_ws_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("creating the test workspace");
    dir
}

/// The session every case in this file runs against.
///
/// One per binary, over a directory of this binary's own. The cases are about
/// what the tools refuse, and a shared root is what makes the refusals
/// comparable: a case that wrote `inside/note.txt` and a case that read it
/// back have to be looking at the same directory. The root is a value rather
/// than a process global, so this is a fixture the file owns, not state the
/// tools reach for.
fn session() -> Arc<Runtime> {
    static SESSION: std::sync::OnceLock<Arc<Runtime>> = std::sync::OnceLock::new();
    SESSION.get_or_init(|| common::session_in(root())).clone()
}

/// A path outside the workspace, for the refusal cases.
fn outside(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("nova_lite_outside_{}_{}", std::process::id(), name))
}

async fn read(path: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    ReadFile::new(session())
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

async fn write(path: &str, content: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    WriteFile::new(session())
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

async fn edit(path: &str, old: &str, new: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    EditFile::new(session())
        .call(
            &mut ctx,
            EditFileArgs {
                path: path.to_string(),
                edits: vec![EditBlockArgs {
                    old_string: old.to_string(),
                    new_string: new.to_string(),
                    replace_all: false,
                }],
            },
        )
        .await
        .map_err(|e| e.to_string())
}

async fn list(path: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    ListDir::new(session())
        .call(
            &mut ctx,
            ListDirArgs {
                path: path.to_string(),
            },
        )
        .await
        .map_err(|e| e.to_string())
}

async fn grep(pattern: &str, path: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    Grep::new(session())
        .call(
            &mut ctx,
            GrepArgs {
                pattern: pattern.to_string(),
                path: path.to_string(),
                hidden: false,
                ignored: false,
            },
        )
        .await
        .map_err(|e| e.to_string())
}

async fn glob(pattern: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    Glob::new(session())
        .call(
            &mut ctx,
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

async fn shell(command: &str) -> Result<String, String> {
    let mut ctx = ToolContext::default();
    Shell::new(session())
        .call(
            &mut ctx,
            ShellArgs {
                command: command.to_string(),
            },
        )
        .await
        .map_err(|e| e.to_string())
}

/// Every refusal must say the path is outside the workspace, so the model can
/// tell "you may not go there" from "that file is missing".
fn assert_refused(result: Result<String, String>, what: &str) {
    let err = result.expect_err(&format!("{what} should have been refused"));
    assert!(
        err.contains("outside the workspace"),
        "{what} was refused for the wrong reason: {err}"
    );
}

// ------------------------------------------------------------ inside the root

#[tokio::test]
async fn a_relative_path_inside_the_workspace_is_read() {
    let _ = root();
    write("inside/note.txt", "hello from inside\n")
        .await
        .expect("writing inside the workspace");
    let content = read("inside/note.txt").await.expect("reading it back");
    assert_eq!(content, "hello from inside\n");
}

#[tokio::test]
async fn a_relative_path_is_resolved_against_the_root_not_the_process_cwd() {
    // The process's working directory is the crate root, which is not the
    // workspace, so a tool that ignored the root would write there instead.
    let root = root();
    write("cwd_probe.txt", "workspace copy\n").await.unwrap();
    assert!(
        root.join("cwd_probe.txt").exists(),
        "the write should have landed in the workspace root"
    );
    assert!(
        !std::path::Path::new("cwd_probe.txt").exists(),
        "the write must not have landed in the process's working directory"
    );
}

#[tokio::test]
async fn a_glob_is_anchored_to_the_workspace_root() {
    let _ = root();
    write("globbed/one.txt", "x").await.unwrap();
    let hits = glob("**/*.txt").await.expect("a relative pattern is fine");
    assert!(hits.contains("globbed/one.txt"), "unexpected hits: {hits}");
    // The hits are relative to the root, which is what makes them usable: the
    // model puts one back into another tool call, and an absolute path would
    // be refused as outside the workspace. It is also what makes the anchoring
    // structural rather than a check: a relative path cannot name a file
    // outside the root, so there is nothing to escape.
    assert!(
        hits.lines()
            .all(|line| !line.starts_with('/') && !line.contains(':')),
        "every hit should be a relative path: {hits}"
    );
}

#[tokio::test]
async fn the_shell_runs_in_the_workspace() {
    let root = root();
    // `cd` with no argument prints the current directory.
    let out = shell("cd").await.expect("running `cd`");
    let printed = out.trim().replace('\\', "/").to_lowercase();
    assert!(
        printed.contains(&workspace::key(&root)),
        "the shell should run in the workspace root ({printed})"
    );
}

// ----------------------------------------------------------- outside the root

#[tokio::test]
async fn read_file_refuses_a_path_that_climbs_out() {
    let _ = root();
    assert_refused(read("../secrets.txt").await, "read_file ../");
}

#[tokio::test]
async fn read_file_refuses_an_absolute_path_outside() {
    let _ = root();
    let target = outside("read.txt");
    std::fs::write(&target, "not yours").unwrap();
    assert_refused(
        read(&target.display().to_string()).await,
        "read_file on an absolute outside path",
    );
    let _ = std::fs::remove_file(&target);
}

#[tokio::test]
async fn write_file_refuses_a_path_that_climbs_out_and_writes_nothing() {
    let _ = root();
    let target = outside("written.txt");
    let _ = std::fs::remove_file(&target);

    // The same file, named by climbing out of the root rather than by its
    // absolute path, so the refusal is what stops it.
    let escaped = format!("../{}", target.file_name().unwrap().to_string_lossy());
    assert_refused(write(&escaped, "nope").await, "write_file ../");
    assert!(
        !target.exists(),
        "a refused write must not have created {}",
        target.display()
    );
}

#[tokio::test]
async fn edit_file_refuses_a_path_that_climbs_out() {
    let _ = root();
    assert_refused(edit("../secrets.txt", "a", "b").await, "edit_file ../");
}

#[tokio::test]
async fn list_dir_refuses_a_path_that_climbs_out() {
    let _ = root();
    assert_refused(list("..").await, "list_dir ..");
}

#[tokio::test]
async fn grep_refuses_a_path_that_climbs_out() {
    let _ = root();
    assert_refused(grep("anything", "..").await, "grep ..");
}

#[tokio::test]
async fn glob_refuses_a_pattern_that_climbs_out() {
    let _ = root();
    assert_refused(glob("../**/*.txt").await, "glob ../");
}

#[tokio::test]
async fn glob_refuses_an_absolute_pattern_outside() {
    let _ = root();
    let pattern = format!("{}/**/*.txt", outside("dir").display());
    assert_refused(glob(&pattern).await, "glob on an absolute outside pattern");
}

// ------------------------------------------------------- argument and size

#[tokio::test]
async fn edit_file_refuses_an_empty_search_string() {
    let _ = root();
    write("empty_old.txt", "original\n")
        .await
        .expect("writing the file");

    // `contains("")` is always true and `replacen("", new, 1)` inserts at the
    // front, so without the guard this "succeeds" by prepending to the file.
    let err = edit("empty_old.txt", "", "PREPENDED\n")
        .await
        .expect_err("an empty old_string should be refused");
    assert!(
        err.contains("search text is empty"),
        "refused for the wrong reason: {err}"
    );

    let content = read("empty_old.txt").await.expect("reading it back");
    assert_eq!(
        content, "original\n",
        "a refused edit must leave the file untouched"
    );
}

#[tokio::test]
async fn read_file_says_so_when_the_path_is_a_directory() {
    let _ = root();
    write("a_dir/note.txt", "x")
        .await
        .expect("creating the directory");

    // Without the check this is an opaque OS error ("access is denied" on
    // Windows), which tells the model nothing about what to do instead.
    let err = read("a_dir")
        .await
        .expect_err("reading a directory should fail");
    assert!(
        err.contains("is a directory") && err.contains("list_dir"),
        "the error should name the problem and the tool to use: {err}"
    );
}

#[tokio::test]
async fn read_file_keeps_the_front_of_a_file_over_the_read_limit() {
    let _ = root();
    // Just over the 4 MiB read limit, so the bounded path is taken. The
    // content is one repeated line, so the assertion does not depend on where
    // the cut lands.
    let line = "0123456789abcdef\n";
    let big = line.repeat(4 * 1024 * 1024 / line.len() + 64);
    write("big.txt", &big).await.expect("writing the big file");

    let out = read("big.txt").await.expect("reading the big file");
    assert!(
        out.contains("only the first"),
        "a truncated read should say so: {}",
        &out[..out.len().min(200)]
    );
    assert!(
        out.starts_with("0123456789abcdef"),
        "the front of the file should be kept"
    );
    // The result is elided to the model's budget, so it stays small however
    // large the file is: that is the point of bounding the read. The bound is
    // the budget plus the truncation notice, which `read_file` appends after
    // the elision rather than inside it. Asserted against the budget the
    // session actually uses rather than a literal, so the case keeps meaning
    // something when the window changes.
    let budget = session().tool_result_budget();
    assert!(
        out.len() <= budget + 1024,
        "the result should be elided to the budget plus its notice, got {} bytes \
         against a {budget}-character budget",
        out.len()
    );
}

#[tokio::test]
async fn a_small_file_is_read_whole_and_unannotated() {
    let _ = root();
    write("small.txt", "just this\n")
        .await
        .expect("writing the file");
    let out = read("small.txt").await.expect("reading it back");
    assert_eq!(
        out, "just this\n",
        "a file under the limit must be returned verbatim"
    );
}

/// A command that writes far more than the keep limit, spelled for the shell
/// the runner actually uses.
///
/// This is the one place a test names a platform, and it is deliberate. The
/// two shells have no common loop syntax, and the fixture-process trick the
/// process tests use does not help here: the `shell` tool takes a command
/// *string*, so the fixture would have to be invoked through the same shell
/// anyway, and quoting an executable path through both `cmd` and `sh` is more
/// platform-specific code than the loop it would replace. The assertion below
/// is about the runner's output bound, which is the same on both.
#[cfg(windows)]
const FLOOD: &str = "for /L %i in (1,1,60000) do @echo 0123456789012345678901234567890123456789";
#[cfg(unix)]
const FLOOD: &str =
    "i=0; while [ $i -lt 60000 ]; do echo 0123456789012345678901234567890123456789; i=$((i+1)); done";

#[tokio::test]
async fn shell_discards_output_past_the_keep_limit_without_hanging() {
    let _ = root();
    // ~3 MiB, comfortably over the 1 MiB keep limit. The command must still
    // finish: the excess is read and dropped, not left in the pipe, because a
    // child blocked on a full pipe never exits.
    let out = shell(FLOOD).await.expect("the command should finish");

    assert!(
        out.contains("bytes of output were dropped"),
        "the truncation should be reported: {}",
        &out[..out.len().min(200)]
    );
    // Both ends are kept, not just the front. A compiler prints its
    // diagnostics last, so a head-only bound would show the progress lines
    // and hide the errors, which is the case the bound exists to survive.
    assert!(
        out.starts_with("0123456789"),
        "the front of the output should be kept"
    );
    // The result is elided a second time, to the model's budget, so the
    // structure is: head, an elision marker, the tail, then the runner's own
    // notice. The tail is what sits between the two markers, and it has to be
    // substantial rather than a few stray characters.
    let elision = out
        .find("elided")
        .expect("the result is elided to the budget");
    let notice = out
        .find("bytes of output were dropped")
        .expect("the runner's notice is present");
    assert!(
        notice > elision,
        "the runner's notice comes after the elision marker"
    );
    let tail = &out[elision..notice];
    assert!(
        tail.len() > 1000,
        "the end of the output should be kept too, got {} bytes between the markers",
        tail.len()
    );
    assert!(
        tail.contains("0123456789"),
        "the kept tail should be the command's own output"
    );
    let budget = session().tool_result_budget();
    assert!(
        out.len() <= budget,
        "the result should be elided to the budget, got {} bytes against {budget}",
        out.len()
    );
}

#[tokio::test]
async fn shell_output_under_the_limit_is_returned_whole() {
    let _ = root();
    let out = shell("echo hello").await.expect("the command should run");
    assert_eq!(out.trim(), "hello");
    assert!(
        !out.contains("more than"),
        "a small result must not be annotated: {out:?}"
    );
}

// ------------------------------------------------------------- the write path

/// The names in a directory, sorted, so an assertion does not depend on the
/// order the filesystem hands them back.
fn entries(dir: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(root().join(dir))
        .expect("listing the directory")
        .map(|e| {
            e.expect("an entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    names
}

#[tokio::test]
async fn a_write_replaces_the_content_and_leaves_no_temp_file() {
    let _ = root();
    write("litter/note.txt", "first\n")
        .await
        .expect("writing the file");
    write("litter/note.txt", "second\n")
        .await
        .expect("rewriting the file");
    edit("litter/note.txt", "second", "third")
        .await
        .expect("editing the file");

    // The write goes through a sibling temp file, so a stray one would be
    // picked up by glob, grep and list_dir and read as part of the project.
    assert_eq!(
        entries("litter"),
        vec!["note.txt".to_string()],
        "the directory should hold only the file"
    );
    assert_eq!(
        read("litter/note.txt").await.expect("reading it back"),
        "third\n"
    );
}

#[tokio::test]
async fn a_write_that_cannot_replace_the_file_leaves_it_intact() {
    let _ = root();
    write("locked/note.txt", "original\n")
        .await
        .expect("writing the file");
    let path = root().join("locked/note.txt");

    let original = std::fs::metadata(&path)
        .expect("reading the metadata")
        .permissions();
    let mut readonly = original.clone();
    readonly.set_readonly(true);
    std::fs::set_permissions(&path, readonly).expect("making the file readonly");

    let result = write("locked/note.txt", "replacement\n").await;

    // Restore the permissions captured above rather than clearing the bit:
    // `set_readonly(false)` would make the file world-writable on Unix, and a
    // failure here must not leave a readonly file in the temp workspace for
    // the next run.
    std::fs::set_permissions(&path, original).expect("restoring the permissions");

    assert!(
        result.is_err(),
        "writing a readonly file should have failed"
    );
    assert_eq!(
        read("locked/note.txt").await.expect("reading it back"),
        "original\n",
        "a failed write must leave the original content"
    );
    assert_eq!(
        entries("locked"),
        vec!["note.txt".to_string()],
        "a failed write must not leave a temp file behind"
    );
}
