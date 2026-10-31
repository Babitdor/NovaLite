//! Stall protection on the model call.
//!
//! A local daemon is the whole point of this program, and a local daemon is
//! the one thing that can fail in a way the user cannot tell apart from the
//! model thinking: it stops talking, and the UI sits on a spinner until they
//! give up and press Ctrl-C. Two bounds exist for that, and both are checked
//! here.
//!
//! The first is an idle timeout on the stream. It bounds the gap *between*
//! items rather than the length of the run, so a slow model is left alone and
//! a silent one is abandoned.
//!
//! The second is a single retry, and only for the failure that resolves
//! itself: a daemon that drops the very first request while it loads weights.
//! It is guarded on nothing having happened yet, because a run that has
//! already executed a tool has changed the workspace and replaying it would
//! repeat those changes.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::Duration;

use rig::completion::Usage;
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent};

use rust_agent::app::App;
use rust_agent::config::{ApprovalMode, Config};
use rust_agent::event::Event;

mod common;

type Rx = tokio::sync::mpsc::Receiver<Event>;

/// Build an app whose agent answers from `turns`, with `config` in force.
///
/// The real tool set is registered, because the case that matters most is a
/// tool running *inside* the stream: the gap between two items then includes
/// the time the tool took, which is the same shape as a daemon that has gone
/// quiet.
fn app_with(config: Config, turns: Vec<Vec<MockStreamEvent>>) -> (App, Rx) {
    let model = MockCompletionModel::from_stream_turns(turns);
    let rt = common::session_with(config.with_approval(ApprovalMode::Auto));
    let agent = rig::agent::AgentBuilder::new(model.clone())
        .tool(rust_agent::tools::ReadFile::new(rt.clone()))
        .tool(rust_agent::tools::WriteFile::new(rt.clone()))
        .tool(rust_agent::tools::EditFile::new(rt.clone()))
        .tool(rust_agent::tools::Shell::new(rt.clone()))
        .tool(rust_agent::tools::ListDir::new(rt.clone()))
        .tool(rust_agent::tools::Grep::new(rt.clone()))
        .tool(rust_agent::tools::Glob::new(rt.clone()))
        .build();
    let summariser = rig::agent::AgentBuilder::new(model).build();
    App::new(
        rust_agent::agent::Agents {
            main: agent,
            summariser,
        },
        rt,
    )
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Send `prompt` and pump the app's event channel until the run ends.
///
/// Returns every event the run produced, in order, so a test can assert on
/// what the user would have been told rather than only on the final state.
/// The deadline is a backstop: a run that hangs is the failure these tests
/// exist to catch, and a test that hangs instead of failing reports nothing.
fn run(app: &mut App, rx: &mut Rx, prompt: &str) -> Vec<Event> {
    let mut seen = Vec::new();
    let rt = runtime();
    rt.block_on(async {
        app.send_message(prompt.to_string());
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    let finished = matches!(event, Event::Finished(_) | Event::Error(_));
                    seen.push(event.clone());
                    app.handle_event(event);
                    if finished {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => panic!("the run neither finished nor failed within 30s"),
            }
        }
    });
    seen
}

/// The idle timeout, short enough that a test does not have to wait.
fn impatient() -> Config {
    let mut config = Config::defaults();
    config.stream_idle_timeout = Duration::from_secs(1);
    config
}

fn error_text(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Error(text) => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn retries(events: &[Event]) -> usize {
    events
        .iter()
        .filter(|event| matches!(event, Event::StreamRetry { .. }))
        .count()
}

// ------------------------------------------------------------ idle timeout

#[test]
fn a_stream_that_goes_quiet_is_abandoned_rather_than_waited_on() {
    // The stand-in for a stalled daemon is a tool that outlives the timeout.
    // Both leave `next()` pending with nothing to yield, which is the
    // condition the timeout is written against, and a real local daemon
    // cannot be scripted into a stall.
    let (mut app, mut rx) = app_with(
        impatient(),
        vec![vec![
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "shell".to_string(),
                arguments: serde_json::json!({"command": "ping -n 6 127.0.0.1"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ]],
    );

    let events = run(&mut app, &mut rx, "run a slow command");

    let text = error_text(&events);
    assert!(
        text.contains("LLM_STREAM_IDLE_TIMEOUT_SECS"),
        "the stall should be reported, and should name the knob that bounds \
         it so the user can raise it: {text:?}"
    );
    assert!(
        !app.is_streaming(),
        "the run should have ended rather than left the app on a spinner"
    );
}

#[test]
fn a_run_that_keeps_talking_is_never_timed_out() {
    // The timeout is on the gap between items, not on the run. A turn that
    // takes longer than the timeout in total, but never pauses for longer
    // than it, must survive: timing out a working model is worse than
    // waiting for a stalled one.
    let (mut app, mut rx) = app_with(
        impatient(),
        vec![vec![
            MockStreamEvent::Text("thinking".to_string()),
            MockStreamEvent::Text(" still thinking".to_string()),
            MockStreamEvent::Text(" and answering".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ]],
    );

    let events = run(&mut app, &mut rx, "a question");

    assert!(
        error_text(&events).is_empty(),
        "a stream that never went quiet must not be abandoned: {:?}",
        error_text(&events)
    );
    assert!(
        app.messages
            .iter()
            .any(|m| m.content.contains("and answering")),
        "the answer should have reached the transcript"
    );
}

// ---------------------------------------------------------------- the retry

#[test]
fn a_stream_that_fails_before_its_first_item_is_retried_once() {
    // What a daemon cold-loading weights does to the first request: the
    // connection is refused, nothing has been read, and the same request
    // works a moment later.
    let (mut app, mut rx) = app_with(
        Config::defaults(),
        vec![
            vec![MockStreamEvent::error("connection refused")],
            vec![
                MockStreamEvent::Text("the answer".to_string()),
                MockStreamEvent::FinalResponse(mock_final(Usage::new())),
            ],
        ],
    );

    let events = run(&mut app, &mut rx, "a question");

    assert_eq!(retries(&events), 1, "the failure should be retried once");
    assert!(
        app.messages
            .iter()
            .any(|m| m.content.contains("the answer")),
        "the retry should have carried the run to completion, got: {:#?}",
        app.messages.iter().map(|m| &m.content).collect::<Vec<_>>()
    );
    assert!(
        error_text(&events).is_empty(),
        "the second attempt succeeded, so nothing should have been reported \
         as a failure: {:?}",
        error_text(&events)
    );
}

#[test]
fn a_second_failure_is_reported_rather_than_retried_again() {
    // One retry is for the failure that resolves itself. A daemon that is
    // simply not there fails the same way twice, and re-sending a third time
    // would only delay telling the user.
    let (mut app, mut rx) = app_with(
        Config::defaults(),
        vec![
            vec![MockStreamEvent::error("connection refused")],
            vec![MockStreamEvent::error("connection refused")],
        ],
    );

    let events = run(&mut app, &mut rx, "a question");

    assert_eq!(retries(&events), 1, "the retry is not a loop");
    assert!(
        error_text(&events).contains("connection refused"),
        "the failure should reach the user: {:?}",
        error_text(&events)
    );
}

#[test]
fn a_run_that_has_already_used_a_tool_is_not_replayed() {
    // The first turn calls a tool; the second turn's stream then fails. The
    // retry is for a connection that never got going, and this one did: a
    // tool has run, and re-sending the prompt would repeat whatever it did.
    // A tool that wrote a file is not something to run twice.
    let (mut app, mut rx) = app_with(
        Config::defaults(),
        vec![
            vec![
                MockStreamEvent::ToolCall {
                    id: "t0".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                    call_id: None,
                },
                MockStreamEvent::FinalResponse(mock_final(Usage::new())),
            ],
            vec![MockStreamEvent::error("the daemon went away")],
        ],
    );

    let events = run(&mut app, &mut rx, "list the directory");

    assert_eq!(
        retries(&events),
        0,
        "a run that has already used a tool must not be replayed"
    );
    assert!(
        error_text(&events).contains("the daemon went away"),
        "the failure should reach the user: {:?}",
        error_text(&events)
    );
}

#[test]
fn a_later_run_in_the_same_session_is_still_retried() {
    // The guard is on the run, not on the session. A session that has used a
    // tool once is the normal case, and a retry that could never fire again
    // after the first tool call would be no protection at all for the long
    // session it exists for.
    let (mut app, mut rx) = app_with(
        Config::defaults(),
        vec![
            vec![
                MockStreamEvent::ToolCall {
                    id: "t0".to_string(),
                    name: "list_dir".to_string(),
                    arguments: serde_json::json!({"path": "."}),
                    call_id: None,
                },
                MockStreamEvent::FinalResponse(mock_final(Usage::new())),
            ],
            // The model's second call in that run, answering the tool result.
            vec![
                MockStreamEvent::Text("the directory is empty".to_string()),
                MockStreamEvent::FinalResponse(mock_final(Usage::new())),
            ],
            // The second run: the daemon drops the request before answering.
            vec![MockStreamEvent::error("connection refused")],
            vec![
                MockStreamEvent::Text("the second answer".to_string()),
                MockStreamEvent::FinalResponse(mock_final(Usage::new())),
            ],
        ],
    );

    run(&mut app, &mut rx, "list the directory");
    let events = run(&mut app, &mut rx, "a second question");

    assert_eq!(
        retries(&events),
        1,
        "a run that has not yet used a tool should still be retried, even \
         though the session has"
    );
    assert!(
        app.messages
            .iter()
            .any(|m| m.content.contains("the second answer")),
        "the retry should have carried the second run to completion, got: {:#?}",
        app.messages.iter().map(|m| &m.content).collect::<Vec<_>>()
    );
}

#[test]
fn a_size_rejection_is_left_to_the_recovery_that_can_fix_it() {
    // A size rejection arrives the same way a dropped connection does, but
    // re-sending the identical request fails identically. The app has a
    // recovery for it, compact the history and retry against a smaller one,
    // and an internal retry here would spend the scripted turn and pre-empt
    // it.
    let (mut app, mut rx) = app_with(
        Config::defaults(),
        vec![
            vec![MockStreamEvent::error(
                "This model's maximum context length is 8192 tokens",
            )],
            vec![
                MockStreamEvent::Text("the answer".to_string()),
                MockStreamEvent::FinalResponse(mock_final(Usage::new())),
            ],
        ],
    );

    let events = run(&mut app, &mut rx, "a question");

    assert_eq!(
        retries(&events),
        0,
        "a size rejection must not be retried unchanged"
    );
    assert!(
        error_text(&events).contains("maximum context length"),
        "the rejection should reach the app's own recovery: {:?}",
        error_text(&events)
    );
}

// ------------------------------------------------------- the timeout config

#[test]
fn the_idle_timeout_is_configurable_and_never_zero() {
    use rust_agent::config::{parse_stream_idle_timeout, DEFAULT_STREAM_IDLE_TIMEOUT_SECS};

    assert_eq!(
        parse_stream_idle_timeout(None),
        Duration::from_secs(DEFAULT_STREAM_IDLE_TIMEOUT_SECS),
        "an absent setting takes the default"
    );
    assert_eq!(
        parse_stream_idle_timeout(Some("45")),
        Duration::from_secs(45),
        "a configured setting is used"
    );
    // A zero timeout would abandon every stream before its first token, which
    // is never what the user meant; so would a negative or unparseable one.
    for raw in ["0", "0.0", "-1", "", "  ", "soon"] {
        assert_eq!(
            parse_stream_idle_timeout(Some(raw)),
            Duration::from_secs(DEFAULT_STREAM_IDLE_TIMEOUT_SECS),
            "{raw:?} should fall back to the default"
        );
    }
}

/// A tool runs inside the stream, so the gap between two items includes the
/// time the tool took. A command may legitimately run until the command
/// timeout kills it, and a stream timeout at or below that figure would race it
/// and abandon a run that was working.
///
/// Checked at compile time rather than in a test: both figures are constants, so
/// a test would only restate them, and this way a change that inverts the
/// relationship does not build.
const _: () = assert!(
    rust_agent::config::DEFAULT_STREAM_IDLE_TIMEOUT_SECS
        > rust_agent::config::DEFAULT_COMMAND_TIMEOUT_SECS,
    "the stream idle timeout must exceed the command timeout, or a command that \
     runs to its own limit is reported as a stalled model"
);
