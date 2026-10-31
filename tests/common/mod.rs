//! Shared fixtures for the integration tests.
//!
//! Every test that builds an agent needs a *session*: a workspace, an
//! approval gate, a process runner and a set of budgets. Building one per
//! test, over its own temporary directory, is what keeps the tests
//! independent. The workspace root used to be a process global, so two tests
//! in one binary shared a root and the order they ran in could change the
//! result; a session is a value, so that cannot happen.
//!
//! The approval gate here approves everything. That is deliberate: these
//! tests are about the stream, the render and the input map, not about the
//! gate, and a test that had to answer a prompt would be testing the prompt.
//! `tests/approval.rs` is where the gate itself is exercised, with a
//! scripted responder.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
#![allow(dead_code)] // each test binary uses a different subset

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;

use rust_agent::config::{ApprovalMode, Config};
use rust_agent::runtime::Runtime;
use rust_agent::tools;
use rust_agent::workspace::Workspace;

thread_local! {
    /// The scratch directory the test running on this thread is using.
    ///
    /// A `TempDir` deletes its directory when it is dropped, and the runtime
    /// holds a path rather than a handle, so the directory has to outlive the
    /// helper that made it. A thread-local does that: libtest runs each test on
    /// its own thread, and the thread's locals are dropped when the test ends.
    ///
    /// Leaking the directory instead (what `TempDir::keep` did) left 83
    /// directories and 264 MB in `%TEMP%` before this was written. `%TEMP%` is
    /// not swept on Windows the way it is on some other systems, so a leak
    /// there is permanent.
    static SCRATCH: RefCell<Option<Scratch>> = const { RefCell::new(None) };
}

/// The directories one test has asked for, removed when the test ends.
struct Scratch {
    root: PathBuf,
    made: usize,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Reported rather than swallowed: a directory that cannot be removed
        // is how the leak this replaced went unnoticed for so long. It is not
        // a test failure, so it is a message and not a panic.
        if let Err(e) = std::fs::remove_dir_all(&self.root) {
            eprintln!("could not remove {}: {e}", self.root.display());
        }
    }
}

/// A fresh, empty directory for the test running on this thread.
///
/// The path is derived from the test's own name, so a run whose threads never
/// unwind (or a run with `--test-threads=1`, where the tests share the main
/// thread and its locals are never dropped) leaves at most one directory per
/// test rather than one per call. The root is removed before it is used, so a
/// test never inherits the last run's files.
fn scratch() -> PathBuf {
    SCRATCH.with(|cell| {
        let mut slot = cell.borrow_mut();
        let slot = slot.get_or_insert_with(|| {
            let root = scratch_root();
            let _ = std::fs::remove_dir_all(&root);
            Scratch { root, made: 0 }
        });
        slot.made += 1;
        let dir = slot.root.join(slot.made.to_string());
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        dir
    })
}

/// Where a test's scratch directory lives.
fn scratch_root() -> PathBuf {
    let test = std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .replace("::", "-");
    let binary = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "tests".to_string());
    std::env::temp_dir()
        .join("nova-lite-tests")
        .join(binary)
        .join(test)
}

/// A session over a fresh temporary directory, approving everything.
pub fn session() -> Arc<Runtime> {
    session_with(Config::defaults().with_approval(ApprovalMode::Auto))
}

/// A session over a fresh temporary directory, with `config` in force.
pub fn session_with(config: Config) -> Arc<Runtime> {
    session_in_with(scratch(), config)
}

/// A session over `dir`, approving everything.
///
/// For a test that needs to know where the workspace is, so it can put a file
/// in it, or that needs to read a file the repository already has.
pub fn session_in(dir: impl AsRef<std::path::Path>) -> Arc<Runtime> {
    session_in_with(dir, Config::defaults().with_approval(ApprovalMode::Auto))
}

/// A session over `dir`, with `config` in force.
pub fn session_in_with(dir: impl AsRef<std::path::Path>, config: Config) -> Arc<Runtime> {
    let workspace = Workspace::new(dir).expect("the directory is a valid root");
    Runtime::new(config, workspace)
}

/// A session over `dir` whose gate answers from `approver`.
///
/// The gate is the thing under test in `tests/approval.rs`, so it is the one
/// thing those tests do not take from the default: a session that approved
/// everything would make every denial test pass for the wrong reason.
pub fn session_with_approver(
    dir: impl AsRef<std::path::Path>,
    approver: Arc<dyn rust_agent::approval::Approver>,
) -> Arc<Runtime> {
    let workspace = Workspace::new(dir).expect("the directory is a valid root");
    Runtime::with_approver(Config::defaults(), workspace, approver)
}

/// A session over the crate root, approving everything.
///
/// For a test that reads a file the repository actually has (`Cargo.toml`,
/// `src/lib.rs`). A temporary workspace is the right default, but a test that
/// asserts on the *contents* of a real file has to be pointed at the real
/// tree.
pub fn session_at_crate_root() -> Arc<Runtime> {
    session_in(env!("CARGO_MANIFEST_DIR"))
}

/// A session over a fresh, minimal crate, approving everything.
///
/// For a test that runs a real `cargo check`: the check has to have something
/// to check, and pointing it at the repository would make the test write into
/// the tree it is testing. A crate with no dependencies compiles in a moment
/// and leaves nothing behind but its own temporary directory.
pub fn session_over_a_crate() -> Arc<Runtime> {
    session_over_a_crate_with(Config::defaults().with_approval(ApprovalMode::Auto))
}

/// A session over a fresh, minimal crate, with `config` in force.
pub fn session_over_a_crate_with(config: Config) -> Arc<Runtime> {
    let dir = scratch();
    std::fs::create_dir_all(dir.join("src")).expect("a source directory");
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.0.0\"\nedition = \"2021\"\n",
    )
    .expect("a manifest");
    std::fs::write(
        dir.join("src/lib.rs"),
        "pub fn answer() -> u32 {\n    42\n}\n",
    )
    .expect("a source file");
    session_in_with(dir, config)
}

/// The two agents a session runs, over one mock model.
///
/// The app is given both, because the summariser must have no tools and the
/// only way to guarantee that is to build it that way. A test that drives a
/// compaction needs this rather than a hand-built agent, so that what it
/// exercises is what the binary runs.
pub fn agents(
    model: impl Into<rig::DynModel<rig::operation::Completion>>,
    rt: &Arc<Runtime>,
) -> rust_agent::agent::Agents {
    rust_agent::agent::Agents::build(model, rt).expect("the agents build")
}

/// An agent with the real tool set registered against `rt`.
///
/// The tools are the production ones, bound to the session, so a test that
/// drives a tool call goes through the same workspace check, the same process
/// runner and the same budgets the binary does.
pub fn agent_with_tools(
    model: impl Into<rig::DynModel<rig::operation::Completion>>,
    rt: &Arc<Runtime>,
) -> rig::agent::Agent {
    rig::agent::AgentBuilder::new(model)
        .tool(tools::ReadFile::new(rt.clone()))
        .tool(tools::WriteFile::new(rt.clone()))
        .tool(tools::EditFile::new(rt.clone()))
        .tool(tools::Shell::new(rt.clone()))
        .tool(tools::ListDir::new(rt.clone()))
        .tool(tools::Grep::new(rt.clone()))
        .tool(tools::Glob::new(rt.clone()))
        .build()
}

/// An agent with only `read_file` registered, against `rt`.
///
/// The narrower set is what a test wants when it is asserting on the
/// transcript rather than on what the tools can do: a model that emits a call
/// to a tool that is not registered gets an error result, which is a
/// different test.
pub fn agent_with_read_only(
    model: impl Into<rig::DynModel<rig::operation::Completion>>,
    rt: &Arc<Runtime>,
) -> rig::agent::Agent {
    rig::agent::AgentBuilder::new(model)
        .tool(tools::ReadFile::new(rt.clone()))
        .build()
}

/// Point `link` at the directory `target`, by symlink if the process may, by
/// junction if not.
///
/// The junction fallback is what makes the confinement tests run at all on a
/// Windows machine without developer mode: `symlink_dir` needs a privilege an
/// unprivileged process does not have, and a test that silently skips is a
/// test that proves nothing. A junction is a reparse point like any other, so
/// it exercises the same resolution path.
#[cfg(windows)]
pub fn make_dir_link(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
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

/// Point `link` at the directory `target`.
#[cfg(not(windows))]
pub fn make_dir_link(target: &std::path::Path, link: &std::path::Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}
