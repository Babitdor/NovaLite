//! Tool-result eviction: the per-request bound on what the model is sent.
//!
//! The history is re-sent on every model call, so a run that makes many tool
//! calls grows the prompt by a result per call. Compaction cannot rescue that
//! (it runs after a run has settled, and a run that overruns dies before it
//! fires), so the results in one request are bounded instead.
//!
//! The checks that matter are: the budget resolves to something sane, the
//! most recent results survive while the older ones are elided, and the
//! message structure a provider requires is left intact.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rig::completion::CompletionRequest;
use rig::message::{CallId, Message as RigMessage, ToolName, ToolResultContent, UserContent};
use rig::test_utils::{MockCompletionModel, MockTurn};

use rust_agent::agent::{
    build_agent, evict_stale_results, eviction_budget_chars, parse_evict_chars,
    tool_result_budget_chars, DEFAULT_NUM_CTX,
};

mod common;

/// A user turn carrying a tool result.
fn result(name: &str, text: &str) -> RigMessage {
    RigMessage::tool_result(
        CallId::from_wire("call_1"),
        ToolName::new(name).unwrap(),
        text,
    )
}

/// A user turn carrying a tool result with no text in it.
fn structured_result() -> RigMessage {
    RigMessage::User {
        content: vec![UserContent::ToolResult(rig::message::ToolResult {
            call: CallId::from_wire("call_1"),
            name: ToolName::new("read_file").unwrap(),
            content: vec![ToolResultContent::Json {
                value: serde_json::json!({ "rows": 3 }),
            }],
            is_error: false,
        })],
    }
}

/// The text of the tool results in `message`, joined.
fn result_text(message: &RigMessage) -> String {
    let RigMessage::User { content } = message else {
        return String::new();
    };
    content
        .iter()
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(
                result
                    .content
                    .iter()
                    .filter_map(ToolResultContent::as_text)
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The text of every tool result in `history`, joined.
fn history_results(history: &[RigMessage]) -> String {
    history
        .iter()
        .map(result_text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

// ------------------------------------------------------------- the budget

#[test]
fn the_eviction_budget_is_a_fraction_of_the_window() {
    // A quarter of the window, in characters. The exact fraction is a policy
    // choice; that it scales with the window and leaves room for the rest of
    // the conversation is not.
    assert_eq!(eviction_budget_chars(32_768), 32_768);
    assert_eq!(eviction_budget_chars(8_192), 8_192);
    assert!(
        eviction_budget_chars(8_192) < 8_192 * 4,
        "the budget must leave room for the prompt and the answer"
    );
}

#[test]
fn one_full_size_result_never_trips_the_eviction() {
    // A single result is capped at the tool-result budget, so a budget that
    // did not exceed it would elide the result the model had just asked for.
    let window = DEFAULT_NUM_CTX;
    let mut history = vec![result(
        "read_file",
        &"x".repeat(tool_result_budget_chars(window)),
    )];

    assert!(
        !evict_stale_results(&mut history, eviction_budget_chars(window)),
        "one full-size result should fit the default budget"
    );
}

#[test]
fn a_configured_eviction_budget_is_used() {
    assert_eq!(parse_evict_chars(Some("500")), Some(500));
    assert_eq!(parse_evict_chars(Some(" 500 ")), Some(500));
}

#[test]
fn an_unusable_eviction_budget_falls_back_to_the_window() {
    // A zero budget would elide every result the moment it was made, which is
    // never what the user meant to ask for.
    assert_eq!(parse_evict_chars(None), None);
    assert_eq!(parse_evict_chars(Some("0")), None);
    assert_eq!(parse_evict_chars(Some("nonsense")), None);
}

// ------------------------------------------------------- the eviction itself

#[test]
fn a_history_inside_the_budget_is_left_alone() {
    let mut history = vec![result("read_file", "short"), result("grep", "also short")];

    assert!(
        !evict_stale_results(&mut history, 1_000),
        "nothing should be reported as elided"
    );
    assert_eq!(history_results(&history), "short\nalso short");
}

#[test]
fn the_most_recent_results_are_kept_and_the_older_ones_elided() {
    // The work rests on the recent results, so the budget is filled from the
    // recent end: the oldest result is the one that goes.
    let mut history = vec![
        result("read_file", &"a".repeat(100)),
        result("grep", &"b".repeat(100)),
        result("glob", &"c".repeat(100)),
    ];

    assert!(evict_stale_results(&mut history, 250));

    let text = history_results(&history);
    assert!(
        text.contains(&"c".repeat(100)),
        "the newest result should survive:\n{text}"
    );
    assert!(
        text.contains(&"b".repeat(100)),
        "the second newest result should survive:\n{text}"
    );
    assert!(
        !text.contains(&"a".repeat(100)),
        "the oldest result should be gone:\n{text}"
    );
}

#[test]
fn an_elided_result_says_which_tool_and_how_much() {
    // The model has to be able to tell a result it has already read from one
    // that was taken away, and know to call the tool again.
    let mut history = vec![result("read_file", &"a".repeat(100))];

    assert!(evict_stale_results(&mut history, 10));

    let text = history_results(&history);
    assert!(
        text.contains("elided"),
        "the placeholder should say so:\n{text}"
    );
    assert!(
        text.contains("read_file"),
        "the placeholder should name the tool:\n{text}"
    );
    assert!(
        text.contains("100"),
        "the placeholder should say how much was dropped:\n{text}"
    );
}

#[test]
fn eliding_a_result_keeps_the_call_it_answers() {
    // A provider rejects a tool result whose call it cannot match, so the
    // message and the call id have to survive the rewrite.
    let mut history = vec![
        RigMessage::user("read the manifest"),
        RigMessage::assistant("reading"),
        result("read_file", &"a".repeat(100)),
    ];

    assert!(evict_stale_results(&mut history, 10));

    assert_eq!(history.len(), 3, "no message should be dropped");
    let RigMessage::User { content } = &history[2] else {
        panic!("the tool result should still be a user turn");
    };
    let UserContent::ToolResult(result) = &content[0] else {
        panic!("the tool result should still be a tool result");
    };
    assert_eq!(result.call, CallId::from_wire("call_1"));
    assert_eq!(result.name.as_str(), "read_file");
}

#[test]
fn a_history_with_no_tool_results_is_left_alone() {
    let mut history = vec![RigMessage::user("hello"), RigMessage::assistant("hi")];

    assert!(
        !evict_stale_results(&mut history, 1),
        "a conversation with no results has nothing to elide"
    );
}

#[test]
fn a_result_that_exactly_fits_is_kept() {
    let mut history = vec![result("read_file", &"a".repeat(100))];

    assert!(
        !evict_stale_results(&mut history, 100),
        "a result that exactly fills the budget should be kept"
    );
    assert_eq!(history_results(&history), "a".repeat(100));
}

#[test]
fn a_result_one_character_over_the_budget_is_elided() {
    let mut history = vec![result("read_file", &"a".repeat(100))];

    assert!(
        evict_stale_results(&mut history, 99),
        "a result one character over the budget should be elided"
    );
    assert!(!history_results(&history).contains(&"a".repeat(100)));
}

#[test]
fn a_structured_result_is_left_alone() {
    // There is no text in it to shrink, so the eviction has nothing to do.
    let mut history = vec![structured_result()];

    assert!(!evict_stale_results(&mut history, 1));
    assert_eq!(history.len(), 1);
}

// ------------------------------------------------------------- the hook

/// The tool results in the request the model was sent, joined.
fn sent_results(request: &CompletionRequest) -> String {
    history_results(&request.chat_history)
}

/// The text of the prompt the model was sent.
///
/// The prompt is the last message of the request's history, and it is the one
/// message the eviction never touches: it is the result of the call the model
/// just made.
fn sent_prompt(request: &CompletionRequest) -> String {
    request
        .chat_history
        .last()
        .map(result_text)
        .unwrap_or_default()
}

/// Run a prompt that reads the same file twice through the production agent,
/// and return the requests the model was sent.
///
/// Two reads are what it takes to observe the eviction: the result of the
/// call just made is the *prompt* of the next call, not part of its history,
/// so only the result of the call before it is in the history to be bounded.
fn two_read_requests(budget: usize) -> Vec<CompletionRequest> {
    let model = MockCompletionModel::from_turns([
        MockTurn::tool_call(
            "call_1",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        MockTurn::tool_call(
            "call_2",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        MockTurn::text("done"),
    ]);
    // The budget is a property of the session, not of the agent: it is one of
    // the values the runtime carries, so a test that wants a small one builds
    // a session with it rather than passing it to the builder. The workspace
    // is the crate root, because the scripted calls read `Cargo.toml`.
    let mut config = rust_agent::config::Config::defaults()
        .with_approval(rust_agent::config::ApprovalMode::Auto);
    config.context.evict_chars = budget;
    let rt = common::session_in_with(env!("CARGO_MANIFEST_DIR"), config);
    let agent = build_agent(model.clone(), &rt).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        // Three model calls: two tool calls, then the answer to them. The app
        // sets this on the runner rather than the agent, so the test does too.
        agent
            .prompt("read the manifest twice")
            .max_turns(3)
            .await
            .unwrap();
    });

    model.requests()
}

#[test]
fn the_hook_bounds_what_the_model_is_sent() {
    // The production agent, built the way the app builds it, with a budget
    // small enough to observe.
    let requests = two_read_requests(10);
    assert_eq!(
        requests.len(),
        3,
        "both tool calls should have been answered"
    );

    // The result the model just asked for is sent whole: it is the prompt of
    // the call, and eliding it would take away the answer to the question the
    // model just asked.
    assert!(
        sent_prompt(&requests[1]).contains("[package]"),
        "the newest result should be sent whole:\n{}",
        sent_prompt(&requests[1])
    );

    // The result from the call before it is in the history, and over budget.
    let sent = sent_results(&requests[2]);
    assert!(
        sent.contains("elided"),
        "the over-budget result should have been elided:\n{sent}"
    );
    assert!(
        sent.contains("read_file"),
        "the placeholder should name the tool:\n{sent}"
    );
}

#[test]
fn the_hook_leaves_a_request_inside_the_budget_alone() {
    // The same run with the default budget: the result is small enough to
    // send, so the model sees the file rather than a placeholder.
    let requests = two_read_requests(eviction_budget_chars(DEFAULT_NUM_CTX));

    let sent = sent_results(&requests[2]);
    assert!(
        !sent.contains("elided"),
        "a result inside the budget should be sent whole:\n{sent}"
    );
    assert!(
        sent.contains("[package]"),
        "the model should see the file itself:\n{sent}"
    );
}
