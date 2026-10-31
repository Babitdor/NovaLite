//! The request budget: what one model request may occupy.
//!
//! Every model call re-sends the whole conversation, so the size of a request
//! is the size of the session rather than the size of the prompt. The window
//! is a hard limit: a request over it is rejected outright, and a run that is
//! rejected mid-flight has already made its edits.
//!
//! Four things are checked here. The oldest completed exchanges are dropped
//! when a request does not fit, and only whole exchanges go, so every tool
//! result keeps the call it answers. A request that cannot be made to fit is
//! refused with a size the user can act on, and the model is never called. The
//! summarisation call advertises no tools, so the one call that is handed a
//! whole conversation cannot act on it. And a provider's size rejection buys
//! exactly one compaction-and-retry, which a run that has already changed
//! files does not get.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use rig::completion::{CompletionRequest, Usage};
use rig::message::{
    AssistantContent, CallId, Message as RigMessage, ToolCall, ToolFunction, ToolName, UserContent,
};
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent, MockTurn};

use rust_agent::agent::build_agent;
use rust_agent::app::App;
use rust_agent::config::{ApprovalMode, Config};
use rust_agent::context::tokens_for;
use rust_agent::event::Event;
use rust_agent::runtime::Runtime;

mod common;

type Rx = tokio::sync::mpsc::Receiver<Event>;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// A session over the crate root with `window` as the context window.
///
/// The window is a property of the session rather than of the agent, so a
/// test that wants a small one builds a session with it. The workspace is the
/// crate root because the scripted calls read `Cargo.toml`.
fn session_with_window(window: u32) -> Arc<Runtime> {
    let config = Config::defaults()
        .with_approval(ApprovalMode::Auto)
        .with_window(window);
    common::session_in_with(env!("CARGO_MANIFEST_DIR"), config)
}

/// A turn that answers with `text` and reports `input_tokens` as the prompt
/// length of that call.
fn turn(text: &str, input_tokens: u64) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::Text(text.to_string()),
        MockStreamEvent::FinalResponse(mock_final(
            Usage::new().input_tokens(input_tokens).output_tokens(5),
        )),
    ]
}

/// A user turn carrying a tool result.
fn result(call: &str, name: &str, text: &str) -> RigMessage {
    RigMessage::tool_result(CallId::from_wire(call), ToolName::new(name).unwrap(), text)
}

/// One completed exchange: a question, the model's call, its result, and the
/// answer to it.
///
/// The shape a prune has to respect. A tool result with no call before it is a
/// request the provider rejects, so a cut in the middle of one of these turns
/// a request that is merely too large into one that is malformed. The call is
/// a real one rather than a bare assistant turn: rig drops a tool result whose
/// call is not in the history, and it folds a user turn that follows a tool
/// result into the same message, so an exchange built any other way would not
/// reach the provider in the shape it was written in.
///
/// `result_chars` is the size of the tool result and `prose_chars` the size of
/// the question and of the answer. The two are separate because they are
/// bounded by different things: a tool result is elided to the per-result
/// budget, and prose is not elided at all, so a conversation whose bulk is
/// prose is the one that has to be pruned.
fn exchange(n: usize, result_chars: usize, prose_chars: usize) -> Vec<RigMessage> {
    let call = ToolCall::from_wire(
        format!("call_{n}"),
        ToolFunction::new(
            ToolName::new("read_file").unwrap(),
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
    );
    vec![
        RigMessage::user(format!("question {n} {}", "q".repeat(prose_chars))),
        RigMessage::from(AssistantContent::ToolCall(call)),
        result(&format!("call_{n}"), "read_file", &"x".repeat(result_chars)),
        RigMessage::assistant(format!("answer {n} {}", "a".repeat(prose_chars))),
    ]
}

/// The conversation in `request`, without the leading system instructions.
///
/// The preamble is placed as the first message of the history, so the body is
/// what the run carried plus the prompt.
fn body(request: &CompletionRequest) -> &[RigMessage] {
    match request.chat_history.first() {
        Some(RigMessage::System { .. }) => &request.chat_history[1..],
        _ => &request.chat_history[..],
    }
}

/// The text of the last message of `request`, which is the prompt.
fn prompt_text(request: &CompletionRequest) -> String {
    let Some(RigMessage::User { content }) = request.chat_history.last() else {
        return String::new();
    };
    content
        .iter()
        .filter_map(|item| match item {
            UserContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether `message` starts a fresh exchange rather than continuing one.
fn is_exchange_start(message: &RigMessage) -> bool {
    match message {
        RigMessage::User { content } => !content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_))),
        _ => false,
    }
}

/// Run one prompt with `history` already carried, and return the request the
/// model was sent.
///
/// The history is passed to the runner rather than accumulated by a run: the
/// budget hook sees exactly what the runner was given, so a test can hand it a
/// conversation of a known size and observe what comes out the other side.
fn request_with_history(window: u32, history: Vec<RigMessage>) -> CompletionRequest {
    let model = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let rt = session_with_window(window);
    let agent = build_agent(model.clone(), &rt).unwrap();

    runtime().block_on(async {
        agent
            .prompt("now")
            .history(history)
            .max_turns(1)
            .await
            .unwrap();
    });

    let requests = model.requests();
    assert_eq!(requests.len(), 1, "one model call was scripted");
    requests.into_iter().next().unwrap()
}

// ------------------------------------------------------------- the pruning

#[test]
fn a_request_inside_the_window_is_sent_whole() {
    // The default window, and a conversation far smaller than it. Nothing
    // should be dropped: a prune that fired here would be taking away context
    // the model was entitled to.
    let history: Vec<RigMessage> = (0..3).flat_map(|n| exchange(n, 200, 20)).collect();
    let request = request_with_history(rust_agent::agent::DEFAULT_NUM_CTX, history.clone());

    assert_eq!(
        body(&request).len(),
        history.len() + 1,
        "every message should have been sent, plus the prompt"
    );
    assert_eq!(prompt_text(&request), "now");
}

#[test]
fn a_request_over_the_window_drops_the_oldest_exchanges() {
    // Eight exchanges of 30 kB of tool result and 4 kB of prose each, against
    // a 20k window. The tool results are elided to the per-result budget
    // first, and what is left still does not fit, so the oldest exchanges go.
    let history: Vec<RigMessage> = (0..8).flat_map(|n| exchange(n, 30_000, 4_000)).collect();
    let request = request_with_history(20_000, history.clone());

    let sent = body(&request);
    assert!(
        sent.len() < history.len() + 1,
        "the request should have been pruned: {} messages sent of {}",
        sent.len(),
        history.len() + 1
    );
    assert!(
        sent.len() > 1,
        "the most recent exchange must survive: {} messages sent",
        sent.len()
    );
    assert_eq!(
        prompt_text(&request),
        "now",
        "the prompt is the result of the call the model just made and is never dropped"
    );
}

#[test]
fn a_prune_cuts_only_at_an_exchange_start() {
    // A tool result with no call before it is a request the provider rejects,
    // so the cut has to land on a question rather than in the middle of an
    // exchange. This is the property that makes the prune safe to do at all.
    let history: Vec<RigMessage> = (0..8).flat_map(|n| exchange(n, 30_000, 4_000)).collect();
    let request = request_with_history(20_000, history);

    let sent = body(&request);
    assert!(
        is_exchange_start(&sent[0]),
        "the first message sent must start an exchange, got {:?}",
        sent[0]
    );
    // And every tool result still has the call it answers, in order.
    let mut calls: Vec<CallId> = Vec::new();
    for message in sent {
        if let RigMessage::Assistant(turn) = message {
            for part in &turn.content {
                if let rig::message::AssistantContent::ToolCall(call) = part {
                    calls.push(call.id.clone());
                }
            }
        }
        if let RigMessage::User { content } = message {
            for item in content {
                if let UserContent::ToolResult(result) = item {
                    assert!(
                        calls.contains(&result.call),
                        "a tool result was sent without the call it answers: {:?}",
                        result.call
                    );
                }
            }
        }
    }
}

#[test]
fn a_prompt_that_alone_cannot_fit_is_refused_without_calling_the_model() {
    // The prompt is not in the history, so no amount of pruning can make room
    // for it. Sending it would be rejected by the provider after the run had
    // already made its edits; refusing it here costs nothing and says why.
    let model = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let rt = session_with_window(20_000);
    let agent = build_agent(model.clone(), &rt).unwrap();

    let error = runtime()
        .block_on(async {
            agent
                .prompt("x".repeat(200_000))
                .max_turns(1)
                .await
                .map(|_| ())
        })
        .expect_err("a prompt over the window must be refused");

    let text = error.to_string();
    assert!(
        text.contains("LLM_NUM_CTX"),
        "the error should say what to change:\n{text}"
    );
    assert!(
        text.contains("14880"),
        "the error should name the budget it was measured against:\n{text}"
    );
    assert_eq!(
        model.request_count(),
        0,
        "the model must not be called for a request that cannot fit"
    );
}

#[test]
fn an_empty_history_does_not_make_an_oversized_prompt_fit() {
    // An empty history "fits" a zero budget trivially, so a check that only
    // measured the history would let a prompt that is over the window on its
    // own through. The request is measured as a whole for exactly this case.
    let model = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let rt = session_with_window(6_000);
    let agent = build_agent(model.clone(), &rt).unwrap();

    let error = runtime()
        .block_on(async {
            agent
                .prompt("x".repeat(200_000))
                .history(Vec::<RigMessage>::new())
                .max_turns(1)
                .await
                .map(|_| ())
        })
        .expect_err("a prompt over the window must be refused");

    assert!(
        error.to_string().contains("LLM_NUM_CTX"),
        "the error should say what to change:\n{error}"
    );
    assert_eq!(model.request_count(), 0, "the model must not be called");
}

// ------------------------------------------------------- the recorded figure

#[test]
fn the_recorded_estimate_describes_the_request_that_was_sent() {
    // The app shows this figure beside the provider's own usage report, so it
    // has to be the size of the request that went out rather than of one that
    // was measured and then pruned.
    let model = MockCompletionModel::from_turns([MockTurn::text("done")]);
    let rt = session_with_window(20_000);
    let agent = build_agent(model.clone(), &rt).unwrap();

    runtime().block_on(async {
        agent
            .prompt("now")
            .history(
                (0..8)
                    .flat_map(|n| exchange(n, 30_000, 4_000))
                    .collect::<Vec<_>>(),
            )
            .max_turns(1)
            .await
            .unwrap();
    });

    let estimate = rt
        .last_estimate()
        .expect("the hook records an estimate on every model call");
    assert!(
        !estimate.exact,
        "rig exposes no token counter, so the figure is an estimate and must say so"
    );
    assert_eq!(
        estimate.tokens,
        tokens_for(estimate.parts.total()),
        "the figure is the request's characters over the average token length"
    );

    // And it is the request that was sent: the same figure the request itself
    // measures to, not the pre-prune one. The prompt is the last message of
    // the body and is not part of the history, so it is left out of the sum.
    let request = model.requests().into_iter().next().unwrap();
    let sent = body(&request);
    let sent_chars: usize = sent[..sent.len() - 1]
        .iter()
        .map(rust_agent::context::message_chars)
        .sum();
    assert_eq!(
        estimate.parts.history, sent_chars,
        "the recorded history size must be the history that was sent"
    );
}

// ------------------------------------------------------- the summariser

#[test]
fn the_summarisation_call_advertises_no_tools() {
    // Compaction is handed a whole conversation. The guarantee that it cannot
    // act on it is that the agent making the call has no tools registered, so
    // there is nothing for it to call: an instruction not to use them would be
    // a request, and a request can be ignored.
    let model = MockCompletionModel::from_stream_turns(vec![
        turn("the answer", 100),
        turn("a summary of the conversation", 100),
    ]);
    let rt = common::session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt);

    run_to_idle(&mut app, &mut rx, "a question");
    compact_to_idle(&mut app, &mut rx);

    let requests = model.requests();
    assert_eq!(
        requests.len(),
        2,
        "one working call and one summarisation call"
    );
    assert!(
        !requests[0].tools.is_empty(),
        "the working agent advertises its tools"
    );
    assert!(
        requests[1].tools.is_empty(),
        "the summarisation call must advertise no tools, got {}",
        requests[1].tools.len()
    );
}

// ------------------------------------------------------- the recovery retry

/// Pump events until the app is idle and nothing more is coming.
///
/// A run that fails ends with `Event::Error` rather than `Event::Finished`,
/// and a failure the app recovers from starts a compaction and then a second
/// run, so "idle" is the condition rather than a particular event. The short
/// grace period while idle is what distinguishes "done" from "about to start
/// something".
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
        if !prompt.is_empty() {
            app.send_message(prompt.to_string());
        }
        pump(app, rx).await;
    });
}

/// Start a compaction and pump until it settles.
///
/// `compact` spawns the summarisation run, so it has to be called from inside
/// the runtime rather than beside it.
fn compact_to_idle(app: &mut App, rx: &mut Rx) {
    runtime().block_on(async {
        app.compact();
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

#[test]
fn a_size_rejection_compacts_and_retries_once() {
    // The provider refusing a request for its size is the one failure
    // compaction can fix. The run that failed only read, so redoing it against
    // a smaller history is safe and is the whole point.
    let model = MockCompletionModel::from_stream_turns(vec![
        turn("the answer", 100),
        vec![MockStreamEvent::error(
            "This model's maximum context length is 8192 tokens",
        )],
        turn("a summary of the conversation", 100),
        turn("the retried answer", 100),
    ]);
    let rt = common::session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt);

    run_to_idle(&mut app, &mut rx, "a question");
    assert!(
        app.history_len() > 0,
        "the first run should leave a history"
    );

    run_to_idle(&mut app, &mut rx, "a much longer question");

    let text = transcript(&app);
    assert!(
        text.contains("Compacting the conversation and retrying once"),
        "the user should be told what is happening:\n{text}"
    );
    assert!(
        text.contains("the retried answer"),
        "the retry should have run and answered:\n{text}"
    );
    assert!(
        app.recovery_used(),
        "the one attempt should have been spent"
    );
    assert!(
        !app.is_streaming() && !app.compacting,
        "the app should be idle afterwards"
    );
    assert_eq!(
        model.request_count(),
        4,
        "the first run, the rejected call, the summary, and the retry"
    );
}

#[test]
fn a_rejection_after_a_write_is_not_retried() {
    // The retry re-issues the prompt, and a run that has already changed files
    // would re-issue the calls that made those changes. The journal is the
    // guard: it records each completed action as it happens, so a run that
    // wrote is not replayed even though its history was never appended to.
    let model = MockCompletionModel::from_stream_turns(vec![
        vec![
            MockStreamEvent::tool_call(
                "call_1",
                "write_file",
                serde_json::json!({ "path": "note.txt", "content": "hello" }),
            ),
            MockStreamEvent::FinalResponse(mock_final(
                Usage::new().input_tokens(100).output_tokens(5),
            )),
        ],
        turn("wrote it", 100),
        vec![MockStreamEvent::error(
            "This model's maximum context length is 8192 tokens",
        )],
    ]);
    let rt = common::session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "write a note");
    assert!(
        !rt.actions().is_empty(),
        "the run should have recorded the write it made"
    );

    run_to_idle(&mut app, &mut rx, "a much longer question");

    let text = transcript(&app);
    assert!(
        text.contains("Error:"),
        "the failure should be reported:\n{text}"
    );
    assert!(
        !text.contains("Compacting the conversation"),
        "a run that changed files must not be replayed:\n{text}"
    );
    assert_eq!(
        model.request_count(),
        3,
        "the write, the answer to it, and the rejected call: no summarisation"
    );
}

#[test]
fn a_request_refused_locally_does_not_trigger_a_compaction() {
    // The budget hook refuses a request the prompt alone makes impossible, and
    // its message is not a provider's size rejection. Compaction cannot fix a
    // prompt that does not fit on its own, so spending the one attempt on it
    // would only cost a model call and leave the user where they started.
    let model = MockCompletionModel::from_stream_turns(vec![turn("done", 10)]);
    let rt = session_with_window(20_000);
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt);

    run_to_idle(&mut app, &mut rx, &"x".repeat(200_000));

    let text = transcript(&app);
    assert!(
        text.contains("LLM_NUM_CTX"),
        "the user should be told what to change:\n{text}"
    );
    assert!(
        !text.contains("Compacting the conversation"),
        "a locally refused request must not be retried:\n{text}"
    );
    assert_eq!(
        model.request_count(),
        0,
        "the model must not be called for a request that cannot fit"
    );
    assert!(!app.recovery_used(), "no attempt should have been spent");
}
