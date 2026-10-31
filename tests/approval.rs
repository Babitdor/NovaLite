//! The approval boundary: nothing is written or run without an answer.
//!
//! These tests drive the real tools through the real session, so what they
//! assert is the property the gate exists for: a denied action leaves no
//! trace, and an approved one is the action that was previewed. The gate is
//! the only thing they replace, because a test that approved everything would
//! make every denial test pass for the wrong reason.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use std::sync::Arc;

use futures::future::BoxFuture;
use rig::tool::{Tool, ToolContext};
use rust_agent::approval::{ActionKind, ApprovalRequest, Approver, Decision, ScriptedApprover};
use rust_agent::config::{ApprovalMode, Config};
use rust_agent::error::Error;
use rust_agent::runtime::Runtime;
use rust_agent::tools;
use rust_agent::workspace::Workspace;

/// A session over a fresh temporary directory, with `approver` as the gate.
///
/// The directory is returned so the test can look inside it: the point of
/// most of these cases is what is *not* there.
fn session(approver: Arc<dyn Approver>) -> (tempfile::TempDir, Arc<Runtime>) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let rt = common::session_with_approver(dir.path(), approver);
    (dir, rt)
}

/// A session whose workspace is nested one level inside a temporary
/// directory, with `approver` as the gate.
///
/// What is outside the workspace here is the temporary directory itself,
/// which this test owns. The plain [`session`] helper puts the root *at* the
/// temporary directory, so `..` from it is the shared system temp directory,
/// and two tests writing `../escape.txt` there at once would be testing each
/// other.
fn nested_session(approver: Arc<dyn Approver>) -> (tempfile::TempDir, Arc<Runtime>) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let root = dir.path().join("project");
    std::fs::create_dir(&root).expect("the workspace root");
    let rt = common::session_with_approver(&root, approver);
    (dir, rt)
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

/// Run `shell` and return its result.
async fn shell(rt: &Arc<Runtime>, command: &str) -> Result<String, String> {
    tools::Shell::new(rt.clone())
        .call(
            &mut ToolContext::new(),
            tools::ShellArgs {
                command: command.to_string(),
            },
        )
        .await
        .map_err(|e| e.to_string())
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

/// Every entry in `dir`, so a test can assert that nothing was left behind.
fn entries(dir: &std::path::Path) -> Vec<String> {
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

// ------------------------------------------------------------- denials

#[tokio::test]
async fn a_denied_write_creates_no_file() {
    let approver = Arc::new(ScriptedApprover::new(vec![Decision::Denied(
        "not this one".to_string(),
    )]));
    let (dir, rt) = session(approver.clone());

    let err = write(&rt, "a.txt", "hello\n").await.expect_err("denied");
    assert!(err.contains("denied"), "got: {err}");
    assert!(
        err.contains("not this one"),
        "the reason reaches the model: {err}"
    );

    assert_eq!(entries(dir.path()), Vec::<String>::new());
    assert_eq!(
        approver.count(),
        1,
        "the action was asked about exactly once"
    );
}

#[tokio::test]
async fn a_denied_write_creates_no_directory_either() {
    // The parent directories are made as part of the commit, after the
    // answer. A denial that still created `src/` would leave the workspace
    // changed by an action the user refused.
    let approver = Arc::new(ScriptedApprover::new(vec![Decision::Denied(
        "no".to_string(),
    )]));
    let (dir, rt) = session(approver);

    write(&rt, "src/deep/a.txt", "hello\n")
        .await
        .expect_err("denied");
    assert_eq!(entries(dir.path()), Vec::<String>::new());
}

#[tokio::test]
async fn a_denied_command_runs_nothing() {
    let approver = Arc::new(ScriptedApprover::new(vec![Decision::Denied(
        "no".to_string(),
    )]));
    let (dir, rt) = session(approver);

    let marker = dir.path().join("ran.txt");
    let command = format!("echo hi > \"{}\"", marker.display());
    shell(&rt, &command).await.expect_err("denied");

    assert!(!marker.exists(), "a denied command must not have run");
}

#[tokio::test]
async fn a_denied_edit_leaves_the_file_untouched() {
    let approver = Arc::new(ScriptedApprover::new(vec![Decision::Denied(
        "no".to_string(),
    )]));
    let (dir, rt) = session(approver);
    let path = dir.path().join("a.txt");
    std::fs::write(&path, "old\n").unwrap();

    edit(&rt, "a.txt", "old", "new").await.expect_err("denied");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "old\n");
}

// ------------------------------------------------------------ approvals

#[tokio::test]
async fn an_approved_write_lands() {
    let approver = Arc::new(ScriptedApprover::approving());
    let (dir, rt) = session(approver.clone());

    let out = write(&rt, "a.txt", "hello\n").await.expect("approved");
    assert!(out.contains("a.txt"), "got: {out}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "hello\n"
    );
    assert_eq!(approver.count(), 1);
}

#[tokio::test]
async fn the_prompt_names_the_path_and_shows_the_change() {
    let approver = Arc::new(ScriptedApprover::approving());
    let (dir, rt) = session(approver.clone());
    std::fs::write(dir.path().join("a.txt"), "old\n").unwrap();

    edit(&rt, "a.txt", "old", "new").await.expect("approved");

    let seen = approver.seen();
    assert_eq!(seen.len(), 1);
    let request = &seen[0];
    assert_eq!(request.kind, ActionKind::Patch);
    assert_eq!(request.paths, vec!["a.txt".to_string()]);
    assert!(request.preview.contains("-old"), "{}", request.preview);
    assert!(request.preview.contains("+new"), "{}", request.preview);
    assert!(request.title.contains("a.txt"), "{}", request.title);
}

#[tokio::test]
async fn a_new_file_is_previewed_as_an_addition() {
    let approver = Arc::new(ScriptedApprover::approving());
    let (_dir, rt) = session(approver.clone());

    write(&rt, "a.txt", "one\ntwo\n").await.expect("approved");

    let request = &approver.seen()[0];
    assert_eq!(request.kind, ActionKind::Create);
    assert_eq!(request.preview, "+one\n+two\n");
}

#[tokio::test]
async fn the_command_prompt_names_the_directory_and_the_timeout() {
    let approver = Arc::new(ScriptedApprover::approving());
    let (_dir, rt) = session(approver.clone());

    shell(&rt, "echo hi").await.expect("approved");

    let request = &approver.seen()[0];
    assert_eq!(request.kind, ActionKind::Command);
    assert_eq!(request.preview, "echo hi");
    assert!(
        request.detail.contains("timeout"),
        "the deadline is part of what is being approved: {}",
        request.detail
    );
}

// ------------------------------------------------------- stale previews

/// Approves, but changes the file first: the user took a while to answer and
/// an editor saved in the meantime.
#[derive(Debug)]
struct EditsWhileAsking {
    path: std::path::PathBuf,
    content: String,
}

impl Approver for EditsWhileAsking {
    fn decide<'a>(&'a self, _request: ApprovalRequest) -> BoxFuture<'a, Decision> {
        Box::pin(async move {
            tokio::fs::write(&self.path, &self.content)
                .await
                .expect("the fixture file is writable");
            Decision::Approved
        })
    }
}

#[tokio::test]
async fn a_file_that_changed_after_the_preview_is_not_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.txt");
    std::fs::write(&path, "old\n").unwrap();

    let approver = Arc::new(EditsWhileAsking {
        path: path.clone(),
        content: "someone else's work\n".to_string(),
    });
    let rt = common::session_with_approver(dir.path(), approver);

    let err = edit(&rt, "a.txt", "old", "new").await.expect_err("stale");
    assert!(err.contains("changed after it was read"), "got: {err}");
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "someone else's work\n",
        "the other writer's content must survive"
    );
}

#[tokio::test]
async fn a_file_that_appeared_after_the_preview_is_not_overwritten() {
    // The preview said "create"; by the time the answer came back the file
    // exists. Writing would destroy content the user never saw.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.txt");

    let approver = Arc::new(EditsWhileAsking {
        path: path.clone(),
        content: "surprise\n".to_string(),
    });
    let rt = common::session_with_approver(dir.path(), approver);

    let err = write(&rt, "a.txt", "mine\n").await.expect_err("stale");
    assert!(err.contains("changed after it was read"), "got: {err}");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "surprise\n");
}

// --------------------------------------------------- refusals that are not
//                                                      questions

#[tokio::test]
async fn an_edit_that_cannot_match_is_refused_without_asking() {
    // There is nothing to decide: the search text is not in the file, so the
    // prompt would be a question with one answer. Asking anyway would train
    // the user to approve without reading.
    let approver = Arc::new(ScriptedApprover::approving());
    let (dir, rt) = session(approver.clone());
    std::fs::write(dir.path().join("a.txt"), "old\n").unwrap();

    let err = edit(&rt, "a.txt", "not there", "new")
        .await
        .expect_err("no match");
    assert!(err.contains("not found"), "got: {err}");
    assert_eq!(approver.count(), 0, "nothing was asked");
}

#[tokio::test]
async fn an_empty_search_string_is_refused_without_asking() {
    let approver = Arc::new(ScriptedApprover::approving());
    let (dir, rt) = session(approver.clone());
    std::fs::write(dir.path().join("a.txt"), "old\n").unwrap();

    let err = edit(&rt, "a.txt", "", "new").await.expect_err("empty");
    assert!(err.contains("empty"), "got: {err}");
    assert_eq!(approver.count(), 0);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "old\n"
    );
}

#[tokio::test]
async fn a_cancelled_run_denies_without_asking() {
    // The run is being abandoned, so there is no point putting a question to
    // the user about it. The action is refused, and the refusal is a
    // cancellation rather than a denial so the model is not told the user
    // said no to something they were never asked.
    let approver = Arc::new(ScriptedApprover::approving());
    let (dir, rt) = session(approver.clone());
    rt.cancel().stop();

    let err = write(&rt, "a.txt", "hello\n").await.expect_err("cancelled");
    assert!(err.contains("cancelled"), "got: {err}");
    assert_eq!(approver.count(), 0, "a cancelled run asks nothing");
    assert_eq!(entries(dir.path()), Vec::<String>::new());
}

// -------------------------------------------------------- auto-approval

#[tokio::test]
async fn auto_approval_skips_the_question_but_not_the_boundary() {
    // The mode skips the prompt, not the workspace check: a path outside the
    // root is refused in auto mode exactly as it is in prompt mode. If it
    // were not, `--auto-approve` would be a way to leave the workspace.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("project");
    std::fs::create_dir(&root).unwrap();
    let rt = common::session_in_with(&root, Config::defaults().with_approval(ApprovalMode::Auto));

    let err = write(&rt, "../escape.txt", "hello\n")
        .await
        .expect_err("outside");
    assert!(err.contains("outside the workspace"), "got: {err}");
    assert!(!dir.path().join("escape.txt").exists());
}

#[tokio::test]
async fn auto_approval_still_writes_inside_the_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let rt = common::session_in_with(
        dir.path(),
        Config::defaults().with_approval(ApprovalMode::Auto),
    );

    write(&rt, "a.txt", "hello\n").await.expect("approved");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "hello\n"
    );
}

#[tokio::test]
async fn a_prompt_mode_session_with_no_channel_denies_everything() {
    // The default when the mode is `Prompt` and no UI has been attached. A
    // gate with nobody behind it stays shut: the alternative is a build that
    // silently writes files because the prompt was never wired up.
    let dir = tempfile::tempdir().unwrap();
    let rt = common::session_in_with(dir.path(), Config::defaults());

    let err = write(&rt, "a.txt", "hello\n").await.expect_err("denied");
    assert!(err.contains("denied"), "got: {err}");
    assert_eq!(entries(dir.path()), Vec::<String>::new());
}

// ------------------------------------------------------------- the journal

#[tokio::test]
async fn a_completed_action_is_journalled_as_it_happens() {
    // The model's history is only appended to on success, so a run that is
    // cancelled after an edit would leave the file changed and the model
    // unaware. The journal is what the orchestration layer reads to tell it.
    let approver = Arc::new(ScriptedApprover::approving());
    let (_dir, rt) = session(approver);

    assert!(rt.actions().is_empty());
    write(&rt, "a.txt", "hello\n").await.expect("approved");

    let entries = rt.actions().entries();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].kind, ActionKind::Create);
    assert!(
        entries[0].summary.contains("a.txt"),
        "{}",
        entries[0].summary
    );
}

#[tokio::test]
async fn a_denied_action_is_not_journalled() {
    let approver = Arc::new(ScriptedApprover::new(vec![Decision::Denied(
        "no".to_string(),
    )]));
    let (_dir, rt) = session(approver);

    write(&rt, "a.txt", "hello\n").await.expect_err("denied");
    assert!(
        rt.actions().is_empty(),
        "nothing happened, so nothing is recorded"
    );
}

// ------------------------------------------------------------ the workspace

#[tokio::test]
async fn a_path_outside_the_workspace_is_asked_about_not_refused() {
    // The boundary is a question, not a wall. A reference crate beside the
    // project is a normal thing to need, so the user is asked about the
    // directory; what they are *not* asked is the same question again for the
    // next file in it.
    let approver = Arc::new(ScriptedApprover::approving());
    let (dir, rt) = nested_session(approver.clone());

    write(&rt, "../escape.txt", "hello\n")
        .await
        .expect("granted");
    assert!(
        dir.path().join("escape.txt").exists(),
        "the approved write lands"
    );
    assert_eq!(
        approver.count(),
        2,
        "the directory is asked about, then the write is"
    );

    write(&rt, "../escape2.txt", "hello\n")
        .await
        .expect("already granted");
    assert_eq!(
        approver.count(),
        3,
        "the grant is remembered, so only the write is asked about"
    );
}

#[tokio::test]
async fn a_path_through_a_link_outside_is_asked_about_by_its_real_target() {
    // A link is inside the workspace by its name and outside by where it
    // leads. The question has to be about the place the path really reaches,
    // or approving it grants a directory the read never uses and the same
    // question comes back on the next attempt.
    let dir = tempfile::tempdir().expect("a temporary directory");
    let root = dir.path().join("project");
    let outside = dir.path().join("outside");
    std::fs::create_dir(&root).expect("the workspace root");
    std::fs::create_dir(&outside).expect("the outside directory");
    std::fs::write(outside.join("secret.txt"), "classified\n").expect("the secret");
    if common::make_dir_link(&outside, &root.join("link")).is_err() {
        eprintln!("skipped: this process may not create a directory link");
        return;
    }
    let approver = Arc::new(ScriptedApprover::approving());
    let rt = common::session_with_approver(&root, approver.clone());

    let out = read(&rt, "link/secret.txt").await.expect("granted");
    assert!(out.contains("classified"), "got: {out}");

    let asked = approver.seen();
    assert_eq!(asked.len(), 1, "one directory question: {asked:?}");
    assert_eq!(asked[0].kind, ActionKind::Access);

    let requested = asked[0]
        .preview
        .lines()
        .find_map(|line| line.strip_prefix("requested: "))
        .expect("the preview names the requested path");
    assert_eq!(
        rust_agent::workspace::key(std::path::Path::new(requested)),
        rust_agent::workspace::key(&outside.join("secret.txt")),
        "the question names where the link leads: {:?}",
        asked[0]
    );

    let granted = asked[0]
        .detail
        .strip_prefix("approving grants ")
        .and_then(|rest| rest.strip_suffix(" for the rest of this session"))
        .expect("the detail names the granted directory");
    assert_eq!(
        rust_agent::workspace::key(std::path::Path::new(granted)),
        rust_agent::workspace::key(&outside),
        "the grant is the directory the read uses: {:?}",
        asked[0]
    );

    // The shell scan has to see the same escape: the command names a path
    // that exists through the link, so the words alone would look contained.
    let scan = Workspace::new(&root).expect("a workspace");
    let reach = scan.reach_of("cat link/secret.txt");
    assert_eq!(reach.outside.len(), 1, "{reach:?}");
    assert_eq!(
        rust_agent::workspace::key(&reach.outside[0]),
        rust_agent::workspace::key(&outside.join("secret.txt")),
        "{reach:?}"
    );
}

#[tokio::test]
async fn a_command_that_reaches_outside_is_asked_about_first() {
    // The shell is not confined, so the gate is the only thing between a
    // command and the rest of the disk. What this pins is that the gate is
    // not blind: the path in the command is resolved and put up as the same
    // directory question a file tool would have to ask, before the command
    // itself is put up at all.
    let approver = Arc::new(ScriptedApprover::approving());
    let (dir, rt) = nested_session(approver.clone());
    std::fs::write(dir.path().join("secret.txt"), "shh\n").unwrap();

    shell(&rt, "cat ../secret.txt").await.expect("granted");
    let asked = approver.seen();
    assert_eq!(asked.len(), 2, "the directory, then the command: {asked:?}");
    assert_eq!(asked[0].kind, ActionKind::Access);
    assert!(
        asked[0].preview.contains("secret.txt"),
        "the question names what the command reaches: {:?}",
        asked[0]
    );
    assert_eq!(asked[1].kind, ActionKind::Command);
    assert!(
        asked[1].detail.contains("reaches"),
        "the command prompt says what it reaches: {:?}",
        asked[1]
    );
}

#[tokio::test]
async fn a_command_that_reaches_outside_and_is_refused_never_runs() {
    let approver = Arc::new(ScriptedApprover::new(vec![Decision::Denied(
        "not that directory".to_string(),
    )]));
    let (dir, rt) = nested_session(approver.clone());
    let outside = dir.path().join("escape.txt");

    let err = shell(&rt, "echo hi > ../escape.txt")
        .await
        .expect_err("denied");
    assert!(err.contains("not that directory"), "got: {err}");
    assert_eq!(approver.count(), 1, "the command is never put up");
    assert!(!outside.exists(), "nothing ran");
}

#[tokio::test]
async fn a_command_that_stays_inside_is_not_asked_about_twice() {
    // The common case must stay one question. A boundary check that prompted
    // for `cargo check` would be trained away within a session.
    let approver = Arc::new(ScriptedApprover::approving());
    let (_dir, rt) = nested_session(approver.clone());

    shell(&rt, "echo hello").await.expect("approved");
    assert_eq!(approver.count(), 1, "only the command is asked about");
}

#[tokio::test]
async fn auto_approval_refuses_a_command_that_reaches_outside() {
    // Auto mode answers for the work, not for where it happens. The refusal
    // comes from the directory question, which auto mode never grants.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("project");
    std::fs::create_dir(&root).unwrap();
    let rt = common::session_in_with(&root, Config::defaults().with_approval(ApprovalMode::Auto));

    let err = shell(&rt, "cat ../secret.txt").await.expect_err("outside");
    assert!(err.contains("outside the workspace"), "got: {err}");
}

#[tokio::test]
async fn auto_approval_refuses_a_command_whose_reach_cannot_be_read() {
    // A substitution can name any path once it runs, and auto mode has
    // nobody to show it to. In prompt mode the same command is allowed: the
    // user sees it named in the prompt and decides.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("project");
    std::fs::create_dir(&root).unwrap();
    let rt = common::session_in_with(&root, Config::defaults().with_approval(ApprovalMode::Auto));

    let err = shell(&rt, "cat $(cat list.txt)").await.expect_err("opaque");
    assert!(err.contains("cannot be read"), "got: {err}");

    let approver = Arc::new(ScriptedApprover::approving());
    let (_dir, prompt) = nested_session(approver.clone());
    shell(&prompt, "echo $(echo hi)").await.expect("approved");
    let asked = approver.seen();
    assert!(
        asked[0].detail.contains("cannot be read from the command"),
        "the prompt says the reach is not the whole story: {:?}",
        asked[0]
    );
}

#[tokio::test]
async fn a_path_outside_the_workspace_that_is_refused_is_not_touched() {
    // The denial is the whole point of asking: nothing outside the root is
    // read or written unless the user says so, and the write is never put to
    // them, because the answer to the first question settled it.
    let approver = Arc::new(ScriptedApprover::new(vec![Decision::Denied(
        "not that directory".to_string(),
    )]));
    let (dir, rt) = nested_session(approver.clone());

    let err = write(&rt, "../escape.txt", "hello\n")
        .await
        .expect_err("denied");
    assert!(err.contains("not that directory"), "got: {err}");
    assert_eq!(approver.count(), 1, "the write is never reached");
    assert!(!dir.path().join("escape.txt").exists());
}

#[tokio::test]
async fn a_denial_is_a_refusal_the_model_must_not_retry() {
    // The distinction the model needs: a denial is a decision, not a
    // failure, so re-issuing the same call is not a fix.
    let err = Error::Denied("no".to_string());
    assert!(err.is_refusal());
    assert!(Error::Cancelled.is_refusal());
    assert!(Error::Stale {
        path: "a.txt".to_string()
    }
    .is_refusal());
    assert!(!Error::invalid("bad argument").is_refusal());
}

#[tokio::test]
async fn a_session_can_be_pointed_at_a_workspace_it_was_not_built_with() {
    // The gate is per session, not per process: two sessions in one process
    // have two gates, and a denial in one does not affect the other.
    let denying = Arc::new(ScriptedApprover::new(vec![Decision::Denied(
        "no".to_string(),
    )]));
    let (dir_a, rt_a) = session(denying);
    let (dir_b, rt_b) = session(Arc::new(ScriptedApprover::approving()));

    write(&rt_a, "a.txt", "hello\n").await.expect_err("denied");
    write(&rt_b, "a.txt", "hello\n").await.expect("approved");

    assert!(!dir_a.path().join("a.txt").exists());
    assert!(dir_b.path().join("a.txt").exists());
}

#[tokio::test]
async fn the_workspace_root_is_the_one_the_session_was_built_with() {
    // Compared through `key`, which drops the `\\?\` verbatim prefix Windows
    // adds on canonicalisation: the prefix is an implementation detail of how
    // the root was resolved, not part of which directory it is.
    let dir = tempfile::tempdir().unwrap();
    let workspace = Workspace::new(dir.path()).unwrap();
    let rt = Runtime::with_approver(
        Config::defaults().with_approval(ApprovalMode::Auto),
        workspace,
        Arc::new(ScriptedApprover::approving()),
    );
    assert_eq!(rt.workspace().key(), rust_agent::workspace::key(dir.path()));
    assert_eq!(
        rt.workspace().name(),
        dir.path().file_name().unwrap().to_string_lossy()
    );
}
