//! The run loop, end to end: stages, verification, and self-correction.
//!
//! The loop the model is asked to follow is analyze, gather, plan, execute,
//! verify, finish. Until the session recorder existed that loop was only in
//! the prompt, and the app could not tell a run that changed three files and
//! checked them from one that changed three files and stopped. These tests
//! drive the real app over a scripted model and assert on what the run says
//! about itself, and on what it leaves behind when it is interrupted.
//!
//! The model is scripted rather than live: a live provider would make the
//! assertions depend on what a model chose to do, and the thing under test is
//! the loop's bookkeeping, not the model's judgement.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use rig::completion::Usage;
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent};

use rust_agent::app::App;
use rust_agent::config::{ApprovalMode, Config};
use rust_agent::event::Event;
use rust_agent::runtime::Runtime;
use rust_agent::session::{Outcome, Stage, Terminal};

mod common;

type Rx = tokio::sync::mpsc::Receiver<Event>;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// A turn that answers with `text`.
fn turn(text: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::Text(text.to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new().input_tokens(100).output_tokens(5))),
    ]
}

/// A turn that calls one tool and then ends.
fn call(id: &str, name: &str, arguments: serde_json::Value) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call(id, name, arguments),
        MockStreamEvent::FinalResponse(mock_final(Usage::new().input_tokens(100).output_tokens(5))),
    ]
}

/// Pump events until the app settles.
///
/// "Settled" is idle for a short grace period rather than a particular event:
/// a run that fails starts a compaction and then a second run, so the end of
/// the work is not the end of the stream.
async fn pump(app: &mut App, rx: &mut Rx) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let idle = !app.is_streaming() && !app.compacting;
        let wait = if idle {
            std::time::Duration::from_millis(200)
        } else {
            std::time::Duration::from_secs(20)
        };
        match tokio::time::timeout(wait, rx.recv()).await {
            Ok(Some(event)) => app.handle_event(event),
            Ok(None) => break,
            Err(_) if idle => break,
            Err(_) => panic!("the app did not settle within 20s"),
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the app did not settle within 20s"
        );
    }
}

/// Send `prompt` and pump until the app settles.
fn run_to_idle(app: &mut App, rx: &mut Rx, prompt: &str) {
    runtime().block_on(async {
        app.send_message(prompt.to_string());
        pump(app, rx).await;
    });
}

/// The transcript as one string, for asserting on what the user was told.
fn transcript(app: &App) -> String {
    app.messages
        .iter()
        .map(|m| m.content.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The tool results recorded against the run's calls.
///
/// `transcript` joins the chat messages; a tool result lives on the block the
/// call opened, which is where a skipped call's explanation lands.
fn tool_results(app: &App) -> String {
    app.messages
        .iter()
        .filter_map(|m| m.tool.as_ref())
        .map(|b| b.result.clone())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A session over a fresh temporary directory, approving everything.
fn session() -> Arc<Runtime> {
    common::session()
}

#[test]
fn a_run_that_changes_a_file_and_checks_it_reports_verified() {
    // The whole loop in one script: read, change, check, answer. The check
    // runs after the change, so the run can say the change was verified rather
    // than merely made.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        call(
            "call_2",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        call(
            "call_3",
            "shell",
            serde_json::json!({ "command": "cargo check" }),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    let text = transcript(&app);
    assert!(
        text.contains("Verified:"),
        "the run should report that its change was checked:\n{text}"
    );
    assert_eq!(
        app.session().outcome(Terminal::Finished),
        Outcome::Verified,
        "the session should agree with the transcript"
    );
    assert_eq!(app.session().changed(), vec!["note.txt"]);
    assert_eq!(app.session().verifications().len(), 1);
}

#[test]
fn a_run_that_changes_a_file_and_never_checks_it_reports_unverified() {
    // The failure the recorder exists to catch. The run looks successful: the
    // model answered, the file is on disk. What it did not do is check
    // anything, and the transcript has to say so rather than let "done" stand
    // for "works".
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "write_file",
            serde_json::json!({ "path": "src/note.rs", "content": "// hello\n" }),
        ),
        turn("done"),
    ]);
    let rt = session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    let text = transcript(&app);
    assert!(
        text.contains("Unverified:"),
        "a change that was never checked must be reported as unverified:\n{text}"
    );
    assert!(
        text.contains("cargo check"),
        "the report should name the check that would settle it:\n{text}"
    );
    assert!(matches!(
        app.session().outcome(Terminal::Finished),
        Outcome::Unverified { .. }
    ));
}

#[test]
fn a_run_that_changes_nothing_says_nothing_about_verification() {
    // A question that needed no edit has nothing to verify, and a line on
    // every answer would be noise. The absence of the line is the report.
    let model = MockCompletionModel::from_stream_turns(vec![turn("42")]);
    let rt = session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "what is the answer");

    let text = transcript(&app);
    assert!(
        !text.contains("Verified:") && !text.contains("Unverified:"),
        "a run that changed nothing should say nothing:\n{text}"
    );
    assert_eq!(
        app.session().outcome(Terminal::Finished),
        Outcome::NoChanges
    );
}

#[test]
fn a_call_that_repeats_is_answered_not_run() {
    // The loop the failure count cannot see, and the cheap way out of it. The
    // second identical call is not run: the model is told its result is
    // already in the conversation, and the run carries on to its answer. A
    // model that has lost track of a result needs telling, not killing.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        call(
            "call_2",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        turn("I read the file twice."),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read Cargo.toml");

    let text = transcript(&app);
    assert!(
        !text.contains("Stopping the run"),
        "one repeat is answered, not fatal:\n{text}"
    );
    let results = tool_results(&app);
    assert!(
        results.contains("was already called with these exact arguments"),
        "the model should be told what it already has:\n{results}"
    );
    assert!(
        text.contains("I read the file twice."),
        "the run should reach its answer:\n{text}"
    );
}

#[test]
fn a_call_that_keeps_repeating_stops_the_run() {
    // Being told did not land. Every call succeeds, so nothing fails and
    // nothing is counted; without this guard the run would spend its whole
    // turn budget making the same call. The fourth turn is scripted to prove
    // the run never reaches it.
    let repeat = || {
        call(
            "call_n",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        )
    };
    let model = MockCompletionModel::from_stream_turns(vec![
        repeat(),
        repeat(),
        repeat(),
        turn("I read the file three times."),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read Cargo.toml");

    let text = transcript(&app);
    assert!(
        text.contains("the same call has now been made 3 times in a row"),
        "the run should say why it stopped:\n{text}"
    );
    assert!(
        text.contains("read_file") && text.contains("Cargo.toml"),
        "the report should name the call that repeated:\n{text}"
    );
    assert!(
        !text.contains("I read the file three times."),
        "the run stopped before the fourth turn:\n{text}"
    );
}

#[test]
fn a_call_that_differs_is_not_a_loop() {
    // The guard must not fire on a run that is making progress. Two reads of
    // two different files are two calls, and the run has to reach its answer.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        call(
            "call_2",
            "read_file",
            serde_json::json!({ "path": "src/lib.rs" }),
        ),
        turn("both files read."),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read two files");

    let text = transcript(&app);
    assert!(
        !text.contains("Stopping the run"),
        "two different calls are not a loop:\n{text}"
    );
    assert!(text.contains("both files read."), "{text}");
}

#[test]
fn the_runs_turn_budget_is_the_configured_one() {
    // The budget used to be read from the environment inside the run, so a
    // session configured with a different limit was ignored and the two
    // numbers could disagree. A one-turn budget over a three-turn script is
    // what tells them apart: only the first turn's call may run.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        call(
            "call_2",
            "read_file",
            serde_json::json!({ "path": "src/lib.rs" }),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate_with(
        Config::defaults()
            .with_approval(ApprovalMode::Auto)
            .with_max_turns(1),
    );
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read a file");

    assert_eq!(app.calls, 1, "a one-turn budget runs one turn's calls");
    let text = transcript(&app);
    assert!(
        !text.contains("done"),
        "the third turn is past the budget:\n{text}"
    );
}

#[test]
fn a_run_that_uses_its_whole_budget_says_how_to_raise_it() {
    // rig's own text for this is "reached the max turns limit of N", which
    // names the number but not the knob: a user who hits it cannot raise the
    // budget without reading the README. The app replaces it with a message
    // that names `LLM_MAX_TURNS` and says the conversation is kept, so the
    // prompt can be sent again rather than retyped.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        call("call_2", "list_dir", serde_json::json!({ "path": "." })),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate_with(
        Config::defaults()
            .with_approval(ApprovalMode::Auto)
            .with_max_turns(2),
    );
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read a file");

    assert_eq!(app.calls, 2, "a two-turn budget runs two turns");
    let text = transcript(&app);
    assert!(
        text.contains("LLM_MAX_TURNS"),
        "the failure must name the knob that raises the budget:\n{text}"
    );
    assert!(
        text.contains("budget of 2"),
        "the message must name the budget that was used:\n{text}"
    );
    assert!(
        !text.contains("reached the max turns limit"),
        "the raw rig text is a dead end and should be replaced:\n{text}"
    );
}

#[test]
fn a_denied_check_leaves_the_run_unverified() {
    // The user declining the verification command is not a failure of the run,
    // but it does mean nothing checked the change. The run must not report
    // itself verified on the strength of a command that never ran.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        call(
            "call_2",
            "shell",
            serde_json::json!({ "command": "cargo check" }),
        ),
        turn("done"),
    ]);
    // The write is approved and the check is not: the point is a run whose
    // change landed and whose verification did not.
    let dir = tempfile::tempdir().expect("a temporary directory").keep();
    let rt = common::session_with_approver(
        dir,
        Arc::new(rust_agent::approval::ScriptedApprover::new(vec![
            rust_agent::approval::Decision::Approved,
            rust_agent::approval::Decision::Denied("not now".to_string()),
        ])),
    );
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    let text = transcript(&app);
    assert!(
        text.contains("Unverified:"),
        "a denied check must not count as verification:\n{text}"
    );
    assert!(
        app.session().verifications().is_empty(),
        "a command that was denied never ran, so it is not a check"
    );
}

#[test]
fn a_check_that_fails_and_is_not_corrected_is_reported_as_the_reason() {
    // A failed check is the most important thing the run knows. Dropping it
    // would leave the outcome claiming nothing was ever run, which is a
    // different and less useful statement than "the check failed".
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        call(
            "call_2",
            "shell",
            serde_json::json!({ "command": "cargo check --definitely-not-a-flag" }),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    let text = transcript(&app);
    assert!(
        text.contains("Unverified:"),
        "a failed check leaves the change unverified:\n{text}"
    );
    assert_eq!(
        app.session().verifications().len(),
        1,
        "the check ran, so it is recorded whether or not it passed"
    );
    assert!(!app.session().verifications()[0].passed, "the check failed");
}

#[test]
fn a_correction_after_a_failed_check_is_counted() {
    // The correction budget is what bounds the loop. A change made after a
    // failed check is a correction; a change made before any check is the
    // first attempt, and counting it would spend the budget before the model
    // had anything to correct.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        call(
            "call_2",
            "shell",
            serde_json::json!({ "command": "cargo check --definitely-not-a-flag" }),
        ),
        call(
            "call_3",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello again\n" }),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    assert_eq!(
        app.session().corrections(),
        1,
        "the second write follows a failed check, so it is a correction"
    );
}

#[test]
fn the_same_failure_twice_stops_the_run() {
    // The second identical failure is the evidence that the first correction
    // did not address the cause. A third attempt at the same error is a loop,
    // and the run stops and says so rather than spending the rest of the turn
    // budget on it.
    let failing = || {
        call(
            "call_x",
            "shell",
            serde_json::json!({ "command": "cargo check --definitely-not-a-flag" }),
        )
    };
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        failing(),
        call(
            "call_2",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello again\n" }),
        ),
        failing(),
        turn("this should never be reached"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    let text = transcript(&app);
    assert!(
        text.contains("Stopping the run:"),
        "the run should say why it stopped:\n{text}"
    );
    assert!(
        text.contains("the same failure has now repeated 2 times"),
        "the stop should name the failure that repeated:\n{text}"
    );
    // The model's last turn may still appear: Rig's engine makes the next
    // completion call as soon as the tool results are in, so by the time the
    // app has read the second failure the call is already in flight. What the
    // stop guarantees is that the run ends here and says why, not that the
    // provider was never asked again.
    assert!(
        app.session().corrections() < rust_agent::session::MAX_CORRECTIONS,
        "the run should stop on the repeat, not on the correction budget"
    );
}

#[test]
fn an_interrupted_run_still_remembers_what_it_changed() {
    // The history is appended to only on a final response, so a run that is
    // cancelled leaves the model's transcript as it was. The edits are on disk
    // either way, and the journal is what remembers them: a record that
    // depended on the run reaching its end would lose exactly the runs whose
    // effects most need accounting for.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        turn("done"),
    ]);
    let rt = session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    let actions = rt.actions().entries();
    assert_eq!(actions.len(), 1, "the write should be journalled");
    assert!(
        actions[0].summary.contains("note.txt"),
        "the entry should name the file: {}",
        actions[0].summary
    );
    assert_eq!(
        actions[0].call.as_deref(),
        Some("call_1"),
        "the entry should be attributed to the call that asked for it"
    );
    assert!(
        actions[0].run > 0,
        "the entry should be attributed to the run that made it"
    );
}

#[test]
fn a_second_run_does_not_inherit_the_first_runs_changes() {
    // The session recorder is per run. Carrying the previous run's changes
    // forward would make this run's outcome claim edits it never made, and
    // would report a question as unverified because of a file someone else
    // changed.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        turn("done"),
        turn("42"),
    ]);
    let rt = session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");
    assert!(app.session().has_changed());

    run_to_idle(&mut app, &mut rx, "what is the answer");

    assert!(
        !app.session().has_changed(),
        "the second run changed nothing of its own"
    );
    assert_eq!(
        app.session().outcome(Terminal::Finished),
        Outcome::NoChanges
    );
    assert_eq!(
        rt.actions().entries().len(),
        1,
        "the journal keeps both runs' actions, but only the first run wrote"
    );
}

#[test]
fn a_malformed_call_is_returned_to_the_model_rather_than_ending_the_run() {
    // A call whose arguments do not parse is a recoverable failure: the model
    // is told what was wrong with what it sent, and gets to try again. Ending
    // the run instead would turn a typo into a lost session.
    let model = MockCompletionModel::from_stream_turns(vec![
        call("call_1", "read_file", serde_json::json!({ "path": 42 })),
        turn("sorry, let me try again"),
    ]);
    let rt = common::session_at_crate_root();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read the manifest");

    let text = transcript(&app);
    assert!(
        text.contains("sorry, let me try again"),
        "the run should continue after a malformed call:\n{text}"
    );
    assert_eq!(
        model.request_count(),
        2,
        "the model should have been asked again with the error in hand"
    );
}

#[test]
fn a_tool_call_written_into_the_message_is_returned_to_the_model() {
    // A small model routinely answers with the call in its message body
    // instead of through the tool-call channel. rig sees a tool-free turn,
    // treats it as the final answer, and the run ends having done nothing.
    // The turn is retried with feedback instead.
    //
    // The spelling here is the bare object, with no fence and no tag, because
    // that is what a 7B model actually emits: the eval harness caught three
    // tasks failing this way with the retry counter still at zero.
    let model = MockCompletionModel::from_stream_turns(vec![
        turn("{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}"),
        turn("I have read the manifest."),
    ]);
    let rt = common::session_at_crate_root();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read the manifest");

    let text = transcript(&app);
    assert!(
        text.contains("I have read the manifest."),
        "the run should continue after a call written as text:\n{text}"
    );
    // The feedback is not part of the visible transcript, so it is read off the
    // request the model actually received. Asserting on the transcript would
    // have passed for a retry that told the model nothing.
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "the model should have been asked again");
    let second = format!("{:?}", requests[1].messages_for_telemetry());
    assert!(
        second.contains("tool-call interface"),
        "the model should be told how to send the call:\n{second}"
    );
    assert_eq!(
        rt.repair_report().prose_tool_call_retries,
        1,
        "the retry should be counted, or a sweep cannot tell this from a model \
         that never tried"
    );
}

#[test]
fn a_model_that_keeps_writing_calls_as_text_is_retried_at_most_twice() {
    // rig imposes no retry limit of its own here, and every retry spends the
    // turn budget. A model that never learns the lesson has to degrade to the
    // existing stop path rather than loop until the budget runs out.
    let model = MockCompletionModel::from_stream_turns(vec![
        turn("{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}"),
        turn("{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}"),
        turn("{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}"),
        turn("{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}"),
        turn("{\"name\": \"read_file\", \"arguments\": {\"path\": \"Cargo.toml\"}}"),
    ]);
    let rt = common::session_at_crate_root();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read the manifest");

    assert_eq!(
        model.request_count(),
        3,
        "the first turn plus two retries, and then the run ends"
    );
    assert_eq!(
        rt.repair_report().prose_tool_call_retries,
        2,
        "the cap is what stops the loop, so it is what the counter should show"
    );
}

#[test]
fn the_stage_advances_from_reading_to_changing_to_checking() {
    // The stage is what the run's status reports, and it is derived from the
    // calls rather than declared by the model: a run that says it is verifying
    // while it is still writing is a run whose status line is a claim rather
    // than a report.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        call(
            "call_2",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        call(
            "call_3",
            "shell",
            serde_json::json!({ "command": "cargo check" }),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    assert_eq!(
        app.session().stage(),
        Stage::Verify,
        "the last thing the run did was check its change"
    );
    assert!(!Stage::Verify.is_editing(), "verifying is not editing");
    assert!(Stage::Execute.is_editing(), "executing is editing");
}

#[test]
fn a_cancelled_run_is_not_reported_as_unverified() {
    // Cancellation is not a failure to verify. Saying "unverified" would
    // suggest the run finished and skipped a step, when what happened is that
    // the user stopped it.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "write_file",
            serde_json::json!({ "path": "note.txt", "content": "hello\n" }),
        ),
        turn("done"),
    ]);
    let rt = session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "add a note");

    assert_eq!(
        app.session().outcome(Terminal::Cancelled),
        Outcome::Cancelled,
        "a cancelled run reports cancellation, not a missing check"
    );
}
