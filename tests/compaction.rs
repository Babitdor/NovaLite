//! Compaction: the `/compact` command and the automatic trigger.
//!
//! Compaction replaces the model's history with a summary of it, so the
//! checks that matter are: the threshold resolves to something sane, the
//! trigger fires at the boundary and not before it, and a failed
//! summarisation never costs the user their conversation.
//!
//! The threshold is measured against the *last* model call's prompt length,
//! not the run's cumulative token total: each call re-sends the whole
//! conversation, so the latest figure is the size of the context.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use rig::completion::Usage;
use rig::message::{
    AssistantContent, CallId, Message as RigMessage, ToolName, ToolResultContent, UserContent,
};
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent};

use rust_agent::agent::{
    default_window, history_chars, parse_compact_at, parse_tail_chars, rebuild_history,
    should_compact, tail_budget_chars, tail_start, tool_result_budget_chars, COMPACT_PROMPT,
    DEFAULT_COMPACT_AT, DEFAULT_NUM_CTX, SUMMARY_PREFIX,
};
use rust_agent::app::{App, Role};
use rust_agent::config::Config;
use rust_agent::event::Event;
use rust_agent::render;
use rust_agent::tui::{handle_input, InputResult};

mod common;

type Rx = tokio::sync::mpsc::Receiver<Event>;

/// The context window the app under test is actually configured with.
///
/// Read from `Config` rather than from the environment. The change these tests
/// pin is that the compaction trigger follows the *resolved* configuration, so
/// a test that computed its own threshold by re-reading the environment would
/// pass whether or not the app did.
fn window() -> u64 {
    Config::defaults().context.window as u64
}

/// The token count at which the app under test compacts.
fn threshold() -> u64 {
    window() * Config::defaults().context.compact_at as u64 / 100
}

fn app_with(turns: Vec<Vec<MockStreamEvent>>) -> (App, Rx) {
    let model = MockCompletionModel::from_stream_turns(turns);
    let rt = common::session();
    App::new(common::agents(model, &rt), rt)
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

/// Type `text` and press Enter, exactly as the event loop does.
fn submit(app: &mut App, text: &str) {
    app.set_input(text);
    if let InputResult::Send(text) = handle_input(app, key(KeyCode::Enter)) {
        app.send_message(text);
    }
}

fn paint(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|f| render::draw(f, app)).unwrap();
    let buffer = terminal.backend().buffer();
    let area = *buffer.area();
    (0..area.height)
        .map(|y| {
            (0..area.width)
                .map(|x| buffer.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "))
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
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

/// Send `prompt` and pump events until the run finishes.
fn run_prompt(app: &mut App, rx: &mut Rx, prompt: &str) {
    let rt = runtime();
    rt.block_on(async {
        app.send_message(prompt.to_string());
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    let finished = matches!(event, Event::Finished(_));
                    app.handle_event(event);
                    if finished {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => panic!("the run did not finish within 20s"),
            }
        }
    });
}

/// Send `prompt` and keep pumping until a compaction run reports back.
///
/// The auto-trigger fires from `Event::Finished`, so a run that crosses the
/// threshold starts a second model call after the first has ended; stopping
/// at `Finished` would miss it.
fn run_prompt_through_compaction(app: &mut App, rx: &mut Rx, prompt: &str) {
    let rt = runtime();
    rt.block_on(async {
        app.send_message(prompt.to_string());
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    let compacted = matches!(event, Event::Compacted(_));
                    app.handle_event(event);
                    if compacted {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => panic!("no compaction within 20s"),
            }
        }
    });
}

/// Run `/compact` and pump until it reports back.
fn run_compact(app: &mut App, rx: &mut Rx) {
    let rt = runtime();
    rt.block_on(async {
        submit(app, "/compact");
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    let compacted = matches!(event, Event::Compacted(_));
                    app.handle_event(event);
                    if compacted {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => panic!("no compaction within 20s"),
            }
        }
    });
}

// ------------------------------------------------------- threshold config

#[test]
fn an_absent_threshold_uses_the_default() {
    assert_eq!(parse_compact_at(None), DEFAULT_COMPACT_AT);
}

#[test]
fn a_configured_threshold_is_used() {
    assert_eq!(parse_compact_at(Some("70")), 70);
    assert_eq!(
        parse_compact_at(Some("  70  ")),
        70,
        "surrounding space is ignored"
    );
}

#[test]
fn the_default_sits_inside_the_requested_band() {
    // The request was 80%. A default outside it would be a different feature
    // from the one asked for.
    assert_eq!(
        DEFAULT_COMPACT_AT, 80,
        "the default must be 80%, got {DEFAULT_COMPACT_AT}"
    );
}

#[test]
fn the_default_leaves_room_for_the_summarisation_call() {
    // The summarisation call sends the history it is summarising, so the
    // threshold has to leave the window room for that request plus the
    // summary it returns. Asserted against the real windows rather than a
    // bare number: 80% is only safe because the remainder is large.
    for window in [
        default_window("ollama"),
        default_window("openai"),
        default_window("anthropic"),
    ] {
        let sent = window as u64 * DEFAULT_COMPACT_AT as u64 / 100;
        let left = window as u64 - sent;
        assert!(
            left >= 4_000,
            "at {DEFAULT_COMPACT_AT}% of a {window}-token window only {left} \
             tokens are left for the summary"
        );
    }
}

#[test]
fn a_threshold_with_no_headroom_falls_back() {
    // The summarisation call sends the history it is summarising, so a
    // threshold at or near the window leaves it nothing to send with.
    for raw in ["96", "99", "100", "255"] {
        assert_eq!(
            parse_compact_at(Some(raw)),
            DEFAULT_COMPACT_AT,
            "{raw:?} leaves no headroom and should fall back"
        );
    }
}

#[test]
fn a_threshold_that_would_fire_every_turn_falls_back() {
    for raw in ["0", "1", "9"] {
        assert_eq!(
            parse_compact_at(Some(raw)),
            DEFAULT_COMPACT_AT,
            "{raw:?} would compact almost every turn and should fall back"
        );
    }
}

#[test]
fn an_unparseable_threshold_falls_back() {
    for raw in ["", "  ", "most", "-1", "65.0", "65%"] {
        assert_eq!(
            parse_compact_at(Some(raw)),
            DEFAULT_COMPACT_AT,
            "{raw:?} should fall back to the default"
        );
    }
}

// ------------------------------------------------------------ the trigger

#[test]
fn the_trigger_fires_at_the_boundary_and_not_before_it() {
    // Asserted at both ends: a `<=` guard would pass the "above" case while
    // letting the "below" case through, and vice versa.
    let window = 1000;
    let pct = 65;

    assert!(
        !should_compact(649, window, pct),
        "just below the threshold must not compact"
    );
    assert!(
        should_compact(650, window, pct),
        "exactly at the threshold must compact: the threshold is the point \
         compaction is wanted, not the point after which it is"
    );
    assert!(
        should_compact(651, window, pct),
        "above the threshold must compact"
    );
}

#[test]
fn an_unknown_window_never_compacts() {
    // A provider that reports no context size gives nothing to measure
    // against, and compacting on a figure that is not known would throw the
    // conversation away for no reason.
    assert!(!should_compact(1_000_000, 0, 65));
}

#[test]
fn the_trigger_scales_with_the_configured_window() {
    // The same token count is over the threshold in a small window and under
    // it in a large one, which is the whole point of measuring a percentage.
    assert!(should_compact(8_000, 8_192, 65));
    assert!(!should_compact(8_000, 32_768, 65));
}

// ------------------------------------------------------- the /compact path

#[test]
fn compacting_an_empty_conversation_says_so_and_starts_nothing() {
    let (mut app, _rx) = app_with(Vec::new());

    submit(&mut app, "/compact");

    assert!(
        !app.compacting,
        "there is nothing to summarise, so no run should start"
    );
    let last = app.messages.last().expect("a message");
    assert!(
        last.content.contains("Nothing to compact"),
        "the user should be told why nothing happened, got {:?}",
        last.content
    );
}

#[test]
fn the_compact_command_replaces_the_history_with_a_summary() {
    // The first turn answers the prompt; the second is the summarisation
    // call, which the mock serves from the same script.
    let (mut app, mut rx) = app_with(vec![
        turn("the answer", 100),
        turn("The user asked a question and it was answered.", 100),
    ]);

    run_prompt(&mut app, &mut rx, "a question");
    let before = app.history_len();
    assert!(
        before > 0,
        "the run should have left a history to compact, got {before}"
    );

    run_compact(&mut app, &mut rx);

    assert_eq!(
        app.history_len(),
        1,
        "the history should be the summary alone, not the summary plus the \
         conversation it summarises"
    );
    assert!(!app.compacting, "the compaction run should have settled");
    assert!(!app.is_streaming(), "the app should be idle afterwards");
}

#[test]
fn the_summary_is_what_the_model_carries_forward() {
    let (mut app, mut rx) = app_with(vec![
        turn("the answer", 100),
        turn("SUMMARY-MARKER: the user asked a question.", 100),
    ]);

    run_prompt(&mut app, &mut rx, "a question");
    run_compact(&mut app, &mut rx);

    let screen = paint(&app, 100, 30);
    assert!(
        screen.contains("Compacted"),
        "the transcript should mark where the model's memory was compacted:\n{screen}"
    );
    assert!(
        screen.contains("the answer"),
        "the visible transcript must survive compaction: the user keeps their \
         scrollback, only the model's context is rewritten:\n{screen}"
    );
}

#[test]
fn a_failed_compaction_keeps_the_conversation() {
    // Losing the conversation to a failed summarisation is a far worse
    // outcome than being over the threshold, so the history must survive.
    let (mut app, mut rx) = app_with(vec![
        turn("the answer", 100),
        vec![MockStreamEvent::error("the daemon went away")],
    ]);

    run_prompt(&mut app, &mut rx, "a question");
    let before = app.history_len();

    run_compact(&mut app, &mut rx);

    assert_eq!(
        app.history_len(),
        before,
        "a failed summarisation must leave the history untouched"
    );
    assert!(!app.compacting, "the app must not stay stuck compacting");
    let screen = paint(&app, 100, 30);
    assert!(
        screen.contains("Compaction failed"),
        "the failure should be reported:\n{screen}"
    );
}

#[test]
fn a_prompt_sent_during_compaction_is_queued_not_dropped() {
    // A run started mid-compaction would replay the pre-compaction history
    // and silently undo the summary, so the prompt waits. It has to wait in
    // the queue rather than be refused: the input box is already cleared by
    // the time `send_message` sees the text, so a refusal loses it.
    let (mut app, mut rx) = app_with(vec![turn("the answer", 100), turn("a summary", 100)]);

    run_prompt(&mut app, &mut rx, "a question");
    app.compacting = true;

    submit(&mut app, "another question");

    assert_eq!(
        app.queued(),
        1,
        "the prompt should be waiting in the queue, not dropped"
    );
    let last = app.messages.last().expect("a message");
    assert!(
        last.content.contains("Compacting"),
        "the prompt should be held back with an explanation, got {:?}",
        last.content
    );
}

// ------------------------------------------------------ the automatic path

#[test]
fn a_run_over_the_threshold_compacts_itself() {
    let over = threshold() + 1_000;

    let (mut app, mut rx) = app_with(vec![
        turn("the answer", over),
        turn("a summary of the conversation", over),
    ]);

    run_prompt_through_compaction(&mut app, &mut rx, "a question");

    assert_eq!(
        app.history_len(),
        1,
        "a run past the threshold should have compacted its own history"
    );
    assert!(!app.compacting, "the compaction should have settled");
}

#[test]
fn a_prompt_queued_during_an_automatic_compaction_is_sent_afterwards() {
    // The auto-trigger fires from `Event::Finished`, which then drains the
    // queue. Draining while `compacting` is set used to pop the prompt and
    // hand it to a `send_message` that refused it, so a prompt typed ahead
    // during the run that crossed the threshold was lost silently.
    let over = threshold() + 1_000;

    let (mut app, mut rx) = app_with(vec![
        turn("the answer", over),                    // crosses the threshold
        turn("a summary of the conversation", over), // the compaction call
        turn("the second answer", 100),              // the queued prompt's run
    ]);

    let rt = runtime();
    rt.block_on(async {
        app.send_message("a question".to_string());
        // Typed ahead while the first run is in flight, which is what puts a
        // prompt in the queue in the first place.
        app.send_message("a second question".to_string());
        assert_eq!(app.queued(), 1, "the second prompt should be queued");

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    app.handle_event(event);
                    if app
                        .messages
                        .iter()
                        .any(|m| m.content.contains("the second answer"))
                    {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => panic!("the queued prompt was never sent"),
            }
        }
    });

    assert!(
        app.messages
            .iter()
            .any(|m| m.content.contains("the second answer")),
        "the queued prompt should have run once the summary landed, got: {:#?}",
        app.messages.iter().map(|m| &m.content).collect::<Vec<_>>()
    );
    assert_eq!(app.queued(), 0, "the queue should be empty");
}

#[test]
fn prompts_queued_across_a_compaction_keep_their_order() {
    // `drain_pending` must not pop while a compaction is in flight. Popping
    // hands the prompt to a `send_message` that re-queues it, which moves it
    // to the back, so two prompts typed ahead would run in the wrong order.
    let over = threshold() + 1_000;

    let (mut app, mut rx) = app_with(vec![
        turn("the answer", over),                    // crosses the threshold
        turn("a summary of the conversation", over), // the compaction call
        turn("answer to the first", 100),
        turn("answer to the second", 100),
    ]);

    let rt = runtime();
    rt.block_on(async {
        app.send_message("a question".to_string());
        app.send_message("first queued".to_string());
        app.send_message("second queued".to_string());
        assert_eq!(app.queued(), 2, "both prompts should be queued");

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    app.handle_event(event);
                    if app
                        .messages
                        .iter()
                        .any(|m| m.content.contains("answer to the second"))
                    {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => panic!("the queued prompts were never both sent"),
            }
        }
    });

    let at = |needle: &str| {
        app.messages
            .iter()
            .position(|m| m.role == Role::User && m.content == needle)
    };
    let first = at("first queued");
    let second = at("second queued");
    assert!(
        first.is_some() && second.is_some(),
        "both queued prompts should have run, got: {:#?}",
        app.messages
            .iter()
            .map(|m| (m.role, &m.content))
            .collect::<Vec<_>>()
    );
    assert!(
        first < second,
        "the prompts should run in the order they were typed, got {first:?} then {second:?}"
    );
}

#[test]
fn a_run_under_the_threshold_does_not_compact() {
    let under = threshold() / 2;

    // Only one turn is scripted: if the app wrongly started a compaction run
    // it would consume a second turn, and the mock fails a call with no
    // scripted turn left rather than repeating one.
    let (mut app, mut rx) = app_with(vec![turn("the answer", under)]);

    run_prompt(&mut app, &mut rx, "a question");

    assert!(
        !app.compacting,
        "a run under the threshold must not start a compaction"
    );
    assert!(
        app.history_len() > 1,
        "the history should still hold the whole conversation, got {}",
        app.history_len()
    );
}

#[test]
fn the_context_size_is_the_last_calls_prompt_not_the_runs_total() {
    // `usage_in` accumulates across every call of a run, so it grows without
    // bound and would cross the threshold on a long run that never actually
    // filled the window. The context size is the latest call's prompt length.
    let (mut app, mut rx) = app_with(vec![turn("the answer", 1_000)]);

    run_prompt(&mut app, &mut rx, "a question");

    assert_eq!(
        app.context_tokens, 1_000,
        "the context size should be the last call's input tokens"
    );
}

// ------------------------------------------------------------- the footer

#[test]
fn the_footer_shows_how_full_the_context_is() {
    // The threshold has to be visible before it fires, or compaction reads as
    // the agent losing the conversation for no reason.
    let (mut app, _rx) = app_with(Vec::new());
    app.context_tokens = window() / 2;

    let screen = paint(&app, 120, 30);

    assert!(
        screen.contains("ctx 50%"),
        "the footer should show the context usage:\n{screen}"
    );
}

#[test]
fn the_footer_says_so_when_the_context_size_is_unknown() {
    let (app, _rx) = app_with(Vec::new());

    let screen = paint(&app, 120, 30);

    assert!(
        screen.contains("ctx —"),
        "an unknown context size must not be painted as a number:\n{screen}"
    );
}

#[test]
fn a_context_far_over_the_window_does_not_wrap_its_percentage() {
    // A percentage above 255 narrowed to `u8` wraps, so a context ten times
    // the window would paint a *smaller* figure than one just over it. The
    // clamp has to happen before the narrowing, not after.
    let (mut app, _rx) = app_with(Vec::new());
    app.context_tokens = window() * 10;

    let screen = paint(&app, 120, 30);

    assert!(
        screen.contains("ctx 999%"),
        "an over-full context should clamp at 999%, not wrap:\n{screen}"
    );
}

#[test]
fn the_help_text_documents_the_compact_command() {
    let (mut app, _rx) = app_with(Vec::new());

    submit(&mut app, "/help");

    let help = app
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        help.contains("/compact"),
        "the help text should list /compact:\n{help}"
    );
    assert!(
        help.contains("LLM_COMPACT_AT"),
        "the help text should document the threshold:\n{help}"
    );
}

// ------------------------------------------- the tail kept across a compaction

/// A user turn carrying text, which is where a retained tail may start.
fn prompt(text: &str) -> RigMessage {
    RigMessage::user(text)
}

/// An assistant turn carrying text.
fn answer(text: &str) -> RigMessage {
    RigMessage::assistant(text)
}

/// A user turn carrying a tool result, which is not a place a tail may start.
fn result(text: &str) -> RigMessage {
    RigMessage::tool_result(
        CallId::from_wire("call_1"),
        ToolName::new("read_file").unwrap(),
        text,
    )
}

/// The text a message carries, for asserting on what a request was sent.
fn text_of(message: &RigMessage) -> String {
    match message {
        RigMessage::System { content } => content.clone(),
        RigMessage::User { content } => content
            .iter()
            .map(|item| match item {
                UserContent::Text(text) => text.text.clone(),
                UserContent::ToolResult(result) => result
                    .content
                    .iter()
                    .filter_map(ToolResultContent::as_text)
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        RigMessage::Assistant(turn) => turn
            .content
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

#[test]
fn the_tail_keeps_the_most_recent_turns() {
    let history = vec![
        prompt("first"),
        answer("one"),
        prompt("second"),
        answer("two"),
    ];

    assert_eq!(
        tail_start(&history, 10_000),
        2,
        "the tail should start at the most recent prompt"
    );
}

#[test]
fn the_tail_never_starts_at_the_first_message() {
    // A compaction that evicted nothing would send the summariser an empty
    // conversation, so the first message is never part of the tail.
    let history = vec![prompt("only"), answer("one")];

    assert_eq!(
        tail_start(&history, 10_000),
        history.len(),
        "a single turn has no tail to keep"
    );
}

#[test]
fn the_tail_never_starts_mid_turn() {
    // A tool result with no call before it is a request the provider rejects,
    // so the tail has to start at the prompt that opened the turn.
    let history = vec![
        prompt("first"),
        answer("one"),
        result("the file"),
        answer("two"),
        prompt("second"),
        answer("three"),
    ];

    assert_eq!(
        tail_start(&history, 10_000),
        4,
        "the tail should start at the prompt, not at the tool result"
    );
}

#[test]
fn a_tail_over_the_budget_falls_back_to_the_most_recent_turn() {
    // The most recent turn is the work in flight, so it is returned even when
    // it does not fit, for `rebuild_history` to elide down.
    let history = vec![
        prompt("first"),
        answer("one"),
        prompt("second"),
        answer(&"x".repeat(5_000)),
    ];

    assert_eq!(
        tail_start(&history, 1_000),
        2,
        "an over-budget tail should still start at the most recent prompt"
    );
}

#[test]
fn the_tail_is_elided_to_fit_the_budget() {
    let history = [
        prompt("first"),
        answer("one"),
        prompt("second"),
        result(&"x".repeat(20_000)),
        answer("two"),
    ];

    let rebuilt = rebuild_history(history[2..].to_vec(), "a summary", 4_000);

    assert_eq!(
        rebuilt.len(),
        5,
        "the summary, the acknowledgement and the three tail messages"
    );
    assert!(
        history_chars(&rebuilt[2..]) <= 4_000,
        "the tail should fit the budget, got {} characters",
        history_chars(&rebuilt[2..])
    );
    assert!(
        text_of(&rebuilt[3]).contains("elided"),
        "the elided result should say so: {}",
        text_of(&rebuilt[3])
    );
}

#[test]
fn a_tail_that_cannot_be_made_to_fit_is_dropped() {
    // The point of compacting is to leave the window room. A tail that
    // overran it would only make the next run compact again, so a turn whose
    // own words do not fit is dropped and the summary is the whole memory.
    let history = [prompt("first"), answer("one"), prompt(&"x".repeat(5_000))];

    let rebuilt = rebuild_history(history[2..].to_vec(), "a summary", 1_000);

    assert_eq!(
        rebuilt.len(),
        1,
        "an over-budget tail with no tool result to elide should be dropped"
    );
    assert!(text_of(&rebuilt[0]).contains("a summary"));
}

#[test]
fn the_summary_is_anchored_on_the_previous_one() {
    // The previous summary is part of the evicted prefix, so the summariser
    // is sent it and each summary is built on the last rather than
    // re-derived from a conversation that no longer exists.
    let history = vec![
        RigMessage::user(format!("{SUMMARY_PREFIX}\n\nan earlier summary")),
        prompt("second"),
        answer("two"),
    ];

    assert_eq!(
        tail_start(&history, 10_000),
        1,
        "the previous summary should be evicted, not kept"
    );
}

#[test]
fn a_rebuilt_history_never_opens_with_two_user_messages() {
    // `tail[0]` is a non-tool-result user turn by construction, so the summary
    // and the tail would otherwise be two consecutive user messages. Providers
    // that enforce strict alternation reject that outright, and the local
    // OpenAI-compatible servers this is aimed at are the strictest of all.
    let history = [
        prompt("first"),
        answer("one"),
        prompt("second"),
        answer("two"),
    ];

    let rebuilt = rebuild_history(history[2..].to_vec(), "a summary", 4_000);

    assert!(
        matches!(rebuilt[0], RigMessage::User { .. }),
        "the summary is a user turn"
    );
    assert!(
        matches!(rebuilt[1], RigMessage::Assistant(_)),
        "an assistant turn must sit between the summary and the tail, got {:?}",
        rebuilt[1]
    );
    assert!(
        matches!(rebuilt[2], RigMessage::User { .. }),
        "the tail still starts with the user turn it started with"
    );
    for pair in rebuilt.windows(2) {
        assert!(
            !(matches!(pair[0], RigMessage::User { .. })
                && matches!(pair[1], RigMessage::User { .. })),
            "two consecutive user messages: {pair:?}"
        );
    }
}

#[test]
fn a_dropped_tail_needs_no_acknowledgement() {
    // With no tail there is nothing to separate, and an assistant turn with
    // nothing after it would be a reply to a question nobody asked.
    let history = [prompt("first"), answer("one"), prompt(&"x".repeat(5_000))];

    let rebuilt = rebuild_history(history[2..].to_vec(), "a summary", 1_000);

    assert_eq!(rebuilt.len(), 1, "the summary alone");
}

#[test]
fn the_tail_of_the_conversation_survives_a_compaction() {
    let (mut app, mut rx) = app_with(vec![
        turn("the first answer", 100),
        turn("the second answer", 100),
        turn("a summary of the conversation", 100),
    ]);

    run_prompt(&mut app, &mut rx, "the first question");
    run_prompt(&mut app, &mut rx, "the second question");
    assert_eq!(app.history_len(), 4, "two turns should be four messages");

    run_compact(&mut app, &mut rx);

    assert_eq!(
        app.history_len(),
        4,
        "the summary, the acknowledgement and the most recent turn should survive"
    );
    let screen = paint(&app, 120, 30);
    assert!(
        screen.contains("keeping the last 2"),
        "the transcript should say what was kept:\n{screen}"
    );
}

#[test]
fn the_summariser_is_sent_only_what_is_evicted() {
    // Sending the whole history would make the summary a second copy of the
    // turns that survive, and would cost the most expensive call in the app
    // the full window for no gain.
    let model = MockCompletionModel::from_stream_turns(vec![
        turn("the first answer", 100),
        turn("the second answer", 100),
        turn("a summary of the conversation", 100),
    ]);
    let rt = common::session();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt);

    run_prompt(&mut app, &mut rx, "the first question");
    run_prompt(&mut app, &mut rx, "the second question");
    run_compact(&mut app, &mut rx);

    let requests = model.requests();
    let compaction = requests.last().expect("the compaction call");
    let sent = compaction
        .chat_history
        .iter()
        .map(text_of)
        .collect::<Vec<_>>()
        .join("\n---\n");

    assert!(
        sent.contains("the first question"),
        "the evicted prefix should be sent to the summariser:\n{sent}"
    );
    assert!(
        !sent.contains("the second question"),
        "the kept tail must not be sent to the summariser:\n{sent}"
    );
    assert!(
        sent.contains(COMPACT_PROMPT),
        "the summariser should be asked for a summary:\n{sent}"
    );
}

// ------------------------------------------------------ the tail budget

#[test]
fn an_absent_tail_budget_defers_to_the_window() {
    assert_eq!(parse_tail_chars(None), None);
}

#[test]
fn a_configured_tail_budget_is_used() {
    assert_eq!(parse_tail_chars(Some("4000")), Some(4_000));
    assert_eq!(
        parse_tail_chars(Some("  4000  ")),
        Some(4_000),
        "surrounding space is ignored"
    );
}

#[test]
fn a_zero_or_unparseable_tail_budget_defers_to_the_window() {
    // A zero budget would drop the tail of every compaction, which is never
    // what the user meant to ask for.
    for raw in ["0", "", "  ", "lots", "-1", "4000.0", "4k"] {
        assert_eq!(
            parse_tail_chars(Some(raw)),
            None,
            "{raw:?} should defer to the window-derived default"
        );
    }
}

#[test]
fn the_tail_budget_scales_with_the_window() {
    assert_eq!(tail_budget_chars(32_768), 32_768);
    assert_eq!(tail_budget_chars(4_096), 4_096);
    assert_eq!(tail_budget_chars(0), 0);
}

#[test]
fn the_tail_budget_is_a_fraction_of_the_window() {
    // Both budgets are fractions of the window, so their ratio holds whatever
    // the window is. The tail holds two max-size tool results: one is not
    // enough to keep a turn's shape, and more would leave the summary and the
    // run that follows it no room.
    for window in [4_096, 32_768, 131_072] {
        assert_eq!(
            tail_budget_chars(window),
            2 * tool_result_budget_chars(window),
            "the tail budget should be twice the per-result budget at {window}"
        );
    }
}

#[test]
fn the_tail_budget_holds_a_max_size_tool_result() {
    // A turn whose only bulk is one tool result must not lose its tail: the
    // tail budget has to be at least the per-result budget, or the elision
    // has nothing to work with.
    let window = DEFAULT_NUM_CTX;
    assert!(
        tail_budget_chars(window) >= tool_result_budget_chars(window),
        "the tail budget must hold at least one max-size tool result"
    );
}
