//! End-to-end checks of the stream -> event -> render path.
//!
//! These drive the real `App::send_message` against rig's scripted mock
//! model, so the `MultiTurnStreamItem` variants the mock can produce are
//! exercised through the same code the binary runs, and the result is
//! asserted on the painted buffer rather than on internal state.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use rig::completion::Usage;
use rig::run::CompletionCall;
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent};

use rust_agent::app::{App, ChatMessage, Role};
use rust_agent::event::Event;
use rust_agent::render;
use rust_agent::tools;

mod common;

type Rx = tokio::sync::mpsc::Receiver<Event>;

/// Build an app whose agent answers from `turns`, one scripted stream per
/// model call, with the real tool set registered.
///
/// The workspace is the crate root: the scripted calls read real files
/// (`src/app.rs`, `Cargo.toml`), and a temporary directory would make every
/// one of them fail. Nothing here writes, so the shared root is read-only in
/// practice.
fn app_with(turns: Vec<Vec<MockStreamEvent>>) -> (App, Rx) {
    let model = MockCompletionModel::from_stream_turns(turns);
    let rt = common::session_at_crate_root();
    let agent = rig::agent::AgentBuilder::new(model.clone())
        .tool(tools::ReadFile::new(rt.clone()))
        .tool(tools::WriteFile::new(rt.clone()))
        .tool(tools::EditFile::new(rt.clone()))
        .tool(tools::Shell::new(rt.clone()))
        .tool(tools::ListDir::new(rt.clone()))
        .tool(tools::Grep::new(rt.clone()))
        .tool(tools::Glob::new(rt.clone()))
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

/// Send `prompt` and pump the app's event channel until the run finishes,
/// calling `after_each` with the app after every event is applied. Sampling
/// inside the loop is what makes a liveness claim testable: state that only
/// exists while the run is in flight is gone by the time it ends.
///
/// The app is handed over mutably so a case can act on it mid-run, which is
/// the only way to test what a run does to a viewport the user has moved.
fn drive(app: &mut App, rx: &mut Rx, prompt: &str, mut after_each: impl FnMut(&mut App)) {
    let rt = runtime();
    rt.block_on(async {
        app.send_message(prompt.to_string());
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    let finished = matches!(event, Event::Finished(_));
                    app.handle_event(event);
                    after_each(app);
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

/// Send `prompt`, run to completion, and return the painted screen.
fn run_and_paint(app: &mut App, rx: &mut Rx, prompt: &str) -> Vec<String> {
    drive(app, rx, prompt, |_| {});
    paint(app, 100, 30)
}

/// Render `app` into an offscreen buffer and return it as one string per row.
fn paint(app: &App, width: u16, height: u16) -> Vec<String> {
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
        .collect()
}

fn screen_text(rows: &[String]) -> String {
    rows.join("\n")
}

#[test]
fn the_answer_being_written_carries_a_caret() {
    // The write point, which the spinner cannot show: the spinner says the
    // run is working, the caret says where the next token lands. Sampled
    // mid-run, because that is the only time it exists — the tick is zero
    // throughout, so the caret is on its visible frame.
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("partial".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);

    let mut seen_while_writing = false;
    drive(&mut app, &mut rx, "hi", |app| {
        if app.is_streaming() {
            let screen = screen_text(&paint(app, 100, 30));
            seen_while_writing |= screen.contains(render::CARET[0]);
        }
    });

    assert!(
        seen_while_writing,
        "the answer being written should carry a caret"
    );
    let after = screen_text(&paint(&app, 100, 30));
    assert!(
        !after.contains(render::CARET[0]),
        "a finished answer must not carry one:\n{after}"
    );
}

#[test]
fn a_user_message_is_the_prompt_glyph_and_no_label() {
    let (mut app, _rx) = app_with(Vec::new());
    app.messages.push(ChatMessage::new(
        Role::User,
        "first line\nsecond line".to_string(),
    ));

    let screen = screen_text(&paint(&app, 60, 30));
    assert!(
        screen.contains("❯ first line"),
        "a user's row opens with the prompt glyph:\n{screen}"
    );
    assert!(
        screen.contains("  second line"),
        "a continuation indents to sit under the text, not under the glyph:\n{screen}"
    );
    assert!(
        !screen.contains("You"),
        "a user's row carries no label of its own:\n{screen}"
    );
}

#[test]
fn text_stream_paints_into_the_transcript() {
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("Hello ".to_string()),
        MockStreamEvent::Text("from the mock.".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new().input_tokens(11).output_tokens(4))),
    ]]);

    let rows = run_and_paint(&mut app, &mut rx, "hi");
    let screen = screen_text(&rows);

    assert!(
        screen.contains("Hello from the mock."),
        "streamed text missing from the transcript:\n{screen}"
    );
    assert!(
        screen.contains("❯ hi"),
        "the user's row is the prompt glyph and their words:\n{screen}"
    );
    assert!(
        !screen.contains('◆'),
        "an answer carries no label row of its own:\n{screen}"
    );
    assert!(
        screen.contains("11 in") && screen.contains("4 out"),
        "usage missing from the footer:\n{screen}"
    );
    assert!(
        !app.is_streaming(),
        "the app should be idle once the run finished"
    );
}

#[test]
fn reasoning_is_painted_while_the_turn_is_in_flight() {
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::ReasoningDelta {
            id: "r0".to_string(),
            reasoning: "weighing the options".to_string(),
        },
        MockStreamEvent::Text("done".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);

    // Reasoning is a live indicator: it is cleared when the run ends, so the
    // screen has to be sampled while the turn is still streaming.
    let mut live: Option<String> = None;
    drive(&mut app, &mut rx, "think", |app| {
        if live.is_none() && !app.reasoning.is_empty() {
            live = Some(screen_text(&paint(app, 100, 30)));
        }
    });

    let live = live.expect("reasoning never reached the app");
    assert!(
        live.contains("reasoning"),
        "live reasoning should be labelled on screen:\n{live}"
    );
    assert!(
        live.contains("weighing the options"),
        "reasoning text missing:\n{live}"
    );
    assert!(
        app.reasoning.is_empty(),
        "reasoning should be cleared once the run ends"
    );
}

#[test]
fn tool_call_and_result_appear_in_the_activity_timeline() {
    let (mut app, mut rx) = app_with(vec![
        vec![
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "list_dir".to_string(),
                arguments: serde_json::json!({"path": "."}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::Text("the directory is listed".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
    ]);

    let rows = run_and_paint(&mut app, &mut rx, "list the directory");
    let screen = screen_text(&rows);

    assert!(
        screen.contains("list_dir"),
        "the tool call should be named in its row:\n{screen}"
    );
    assert!(
        screen.contains("path=."),
        "the call's arguments should be shown:\n{screen}"
    );
    assert!(
        screen.contains("the directory is listed"),
        "the follow-up turn's text should reach the transcript:\n{screen}"
    );
    assert!(
        app.calls >= 2,
        "both model calls should be counted, got {}",
        app.calls
    );
}

#[test]
fn a_fenced_code_block_renders_without_literal_backticks() {
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("Try this:\n\n```rust\nlet x = 1;\n```\n".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);

    let rows = run_and_paint(&mut app, &mut rx, "show me");
    let screen = screen_text(&rows);

    assert!(
        screen.contains("let x = 1;"),
        "the code block's body should be painted:\n{screen}"
    );
    assert!(
        !screen.contains("```"),
        "the fence should not be painted literally:\n{screen}"
    );
}

#[test]
fn one_tool_call_produces_one_call_row() {
    let (mut app, mut rx) = app_with(vec![
        vec![
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "list_dir".to_string(),
                arguments: serde_json::json!({"path": "src"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::Text("done".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
    ]);

    run_and_paint(&mut app, &mut rx, "list src");

    // The stream reports the call twice: once as a placeholder while its
    // arguments stream, once when the turn commits. That is one call, so the
    // timeline must show one `ToolCall` row for it, not two.
    let call_rows = app
        .activity
        .iter()
        .filter(|a| a.kind == rust_agent::app::ActivityKind::ToolCall)
        .count();
    assert_eq!(call_rows, 1, "activity timeline: {:#?}", app.activity);
    assert!(
        !app.activity
            .iter()
            .any(|a| a.title.contains("streaming arguments")),
        "the placeholder should have been replaced: {:#?}",
        app.activity
    );
}

#[test]
fn text_from_consecutive_turns_is_separated() {
    // A tool-using run streams text, calls a tool, then streams more text.
    // The two runs are separate paragraphs, not one concatenated sentence.
    let (mut app, mut rx) = app_with(vec![
        vec![
            MockStreamEvent::Text("Let me look.".to_string()),
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "list_dir".to_string(),
                arguments: serde_json::json!({"path": "."}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::Text("Here is the answer.".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
    ]);

    run_and_paint(&mut app, &mut rx, "go");

    // The tool block is the last message now, so find the assistant text
    // rather than assuming it is last.
    let text = &app
        .messages
        .iter()
        .rev()
        .find(|m| m.role == rust_agent::app::Role::Assistant)
        .expect("the assistant message")
        .content;
    assert!(
        text.contains("Let me look.\n\nHere is the answer."),
        "turns should be separated by a blank line, got: {text:?}"
    );
    assert!(
        !text.contains("look.Here"),
        "turns must not be concatenated: {text:?}"
    );
}

/// A narrow terminal still paints the whole conversation.
///
/// There was once a side panel that was dropped below 80 columns; it is gone,
/// so the transcript owns the width at every size and nothing is dropped.
#[test]
fn a_narrow_terminal_still_paints_the_transcript() {
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("narrow".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);

    run_and_paint(&mut app, &mut rx, "hi");

    // Every size paints the same conversation, with no column reserved.
    for (w, h) in [(40u16, 12u16), (79, 20), (80, 20), (200, 60)] {
        let painted = paint(&app, w, h).join("\n");
        assert!(
            painted.contains("narrow"),
            "text missing at {w}x{h}:\n{painted}"
        );
    }
}

#[test]
fn scrolling_leaves_follow_mode_and_clamps_at_the_top() {
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("x".repeat(400)),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);

    run_and_paint(&mut app, &mut rx, "long");

    assert!(app.auto_scroll, "a finished run follows the tail");

    // A frame in a window shorter than the transcript, so there is somewhere
    // to scroll to: the offset is clamped against the height the renderer
    // reports, and a window that fits the whole transcript reports zero.
    paint(&app, 100, 8);
    assert!(
        app.scroll_max() > 3,
        "the transcript should overflow the window"
    );

    app.scroll_by(-5);
    assert!(!app.auto_scroll, "a manual move leaves follow mode");
    assert_eq!(app.scroll, 0, "scrolling up from the top clamps at zero");

    app.scroll_by(3);
    assert_eq!(app.scroll, 3);

    app.scroll_to_bottom();
    assert!(app.auto_scroll, "End returns to follow mode");
}

#[test]
fn a_run_that_ends_does_not_pull_the_view_away_from_where_the_user_put_it() {
    // The viewport is the user's, not the run's. A user who scrolls up during
    // a run to re-read an earlier answer is reading; a run that finishes and
    // jumps them to the bottom takes the answer away at the moment they
    // stopped needing to watch it. What the app owes them instead is the
    // footer's count of what is below.
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("x".repeat(400)),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);

    let mut scrolled_at: Option<u16> = None;
    drive(&mut app, &mut rx, "long", |app| {
        // A window shorter than the transcript, so there is somewhere to
        // scroll to and the offset is clamped against a real height.
        paint(app, 100, 8);
        if scrolled_at.is_none() && app.scroll_max() > 3 {
            app.scroll_by(3);
            scrolled_at = Some(app.scroll);
        }
    });

    let scrolled_at = scrolled_at.expect("the transcript never overflowed the window");
    assert!(
        !app.auto_scroll,
        "the run ending re-armed follow mode and moved the view"
    );
    assert_eq!(app.scroll, scrolled_at, "the run ending moved the viewport");

    // And the footer says what the user is missing, rather than claiming to
    // be following a viewport it is not at.
    let screen = screen_text(&paint(&app, 100, 8));
    assert!(
        screen.contains("below"),
        "the footer should count what is below the view:\n{screen}"
    );
    assert!(
        !screen.contains("following"),
        "the footer still claims to be following:\n{screen}"
    );
}

// ------------------------------------------------------------ tool blocks

/// A run that calls one tool and then answers.
fn tool_run() -> (App, Rx) {
    app_with(vec![
        vec![
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "list_dir".to_string(),
                arguments: serde_json::json!({"path": "src"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::Text("the directory is listed".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
    ])
}

/// A run that explores, then runs a command, then answers: two categories, so
/// the grouping has something to separate.
fn mixed_run() -> (App, Rx) {
    app_with(vec![
        vec![
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "list_dir".to_string(),
                arguments: serde_json::json!({"path": "src"}),
                call_id: None,
            },
            MockStreamEvent::ToolCall {
                id: "t1".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "src/app.rs"}),
                call_id: None,
            },
            MockStreamEvent::ToolCall {
                id: "t2".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "src/render.rs"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::ToolCall {
                id: "t3".to_string(),
                name: "shell".to_string(),
                arguments: serde_json::json!({"command": "echo hello"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::Text("done".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
    ])
}

/// True when a painted row holds nothing at all. The transcript is borderless,
/// so a separator row is blank rather than a run of panel edge.
fn is_blank_row(row: &str) -> bool {
    row.trim().is_empty()
}

/// Paint narrow, and tall enough for the whole transcript, so an assertion
/// cannot be satisfied by a footer row instead, nor missed because the
/// transcript scrolled past it.
fn transcript_whole(app: &App) -> String {
    screen_text(&paint(app, 60, 40))
}

/// Paint narrow and shorter: the transcript is everything above the input box,
/// so an assertion cannot be satisfied by a row outside it.
fn transcript_only(app: &App) -> String {
    screen_text(&paint(app, 60, 30))
}

/// A call in a later turn must not take over an earlier turn's block.
///
/// A part's position is what identifies a placeholder before its call has an
/// id, and positions restart at zero on every model call. So the second turn's
/// first part is position 0 again, and a lookup that matched on position alone
/// would find the *first* turn's block, rename it to the new call, and leave
/// the new turn's own placeholder waiting forever. The transcript would then
/// show one call twice and another not at all.
///
/// The discriminator is each call's own arguments: the tool names alone cannot
/// tell the two apart, because the hijacked block is renamed to the new call's
/// name as well.
#[test]
fn concurrent_tool_arguments_stay_with_their_own_call() {
    // Two calls can stream their arguments at the same time. The stream
    // identifies each fragment by its part, not by arrival order, so the
    // first call's arguments must not land on the second call's placeholder
    // just because the second placeholder is newer.
    let (mut app, mut rx) = app_with(vec![
        vec![
            MockStreamEvent::tool_call_name_delta("call_a", "read_file"),
            MockStreamEvent::tool_call_name_delta("call_b", "grep"),
            MockStreamEvent::tool_call_arguments_delta("call_a", r#"{"path":"Cargo.toml"}"#),
            MockStreamEvent::tool_call_arguments_delta(
                "call_b",
                r#"{"pattern":"fn main","path":"src"}"#,
            ),
            MockStreamEvent::tool_call_end("call_a"),
            MockStreamEvent::tool_call_end("call_b"),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::Text("done".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
    ]);

    // The two calls' arguments carry distinct text, so an argument that
    // belongs to one is unmistakable in the other's row.
    let read_marker = "Cargo.toml";
    let grep_marker = "fn main";
    let mut saw_both_with_arguments = false;

    drive(&mut app, &mut rx, "look", |app| {
        let streaming: Vec<&rust_agent::app::ToolBlock> = app
            .messages
            .iter()
            .filter_map(|m| m.tool.as_ref())
            .filter(|b| b.streaming)
            .collect();
        if streaming.len() != 2 {
            return;
        }
        let Some(read) = streaming.iter().find(|b| b.name == "read_file") else {
            return;
        };
        let Some(grep) = streaming.iter().find(|b| b.name == "grep") else {
            return;
        };

        // A block that has arguments already must hold its own call's and
        // nobody else's. Before any arguments arrive both are empty, which is
        // the only state where nothing can be attributed yet.
        for (block, own, other) in [
            (*read, read_marker, grep_marker),
            (*grep, grep_marker, read_marker),
        ] {
            if block.args.is_empty() {
                continue;
            }
            assert!(
                block.args.contains(own),
                "the {} block holds arguments that are not its own: {:?}",
                block.name,
                block.args
            );
            assert!(
                !block.args.contains(other),
                "the {} block holds the other call's arguments: {:?}",
                block.name,
                block.args
            );
        }

        // The activity timeline is the other live view of the same call, and
        // it has the same identity problem: a row that shows the newest
        // fragment under the oldest call is just as misleading as the block.
        let call_rows: Vec<&rust_agent::app::Activity> = app
            .activity
            .iter()
            .filter(|a| a.kind == rust_agent::app::ActivityKind::ToolCall)
            .collect();
        for (name, own, other) in [
            ("read_file", read_marker, grep_marker),
            ("grep", grep_marker, read_marker),
        ] {
            let Some(row) = call_rows.iter().find(|a| a.title.starts_with(name)) else {
                continue;
            };
            if row.detail.is_empty() {
                continue;
            }
            assert!(
                row.detail.contains(own),
                "the {name} activity row holds arguments that are not its own: {:?}",
                row.detail
            );
            assert!(
                !row.detail.contains(other),
                "the {name} activity row holds the other call's arguments: {:?}",
                row.detail
            );
        }

        saw_both_with_arguments |= !read.args.is_empty() && !grep.args.is_empty();
    });

    assert!(
        saw_both_with_arguments,
        "the run never showed both calls streaming their own arguments"
    );
}

#[test]
fn a_call_in_a_later_turn_does_not_take_over_an_earlier_turns_block() {
    let (mut app, mut rx) = mixed_run();
    run_and_paint(&mut app, &mut rx, "look around");
    let screen = transcript_whole(&app);

    // Every call of both turns is present, with the arguments it was made
    // with. `list_dir` is the one the hijack erased: it was the first turn's
    // position-0 block, so it is the one that went missing.
    for args in [
        "path=src",
        "path=src/app.rs",
        "path=src/render.rs",
        "command=echo hello",
    ] {
        assert!(
            screen.contains(args),
            "the call made with {args} is missing from the transcript:\n{screen}"
        );
    }

    // And no block is left claiming to be in flight: a placeholder that was
    // never bound is closed out when the provider call that opened it ends.
    assert!(
        !screen.contains("streaming…"),
        "a finished run left a call streaming:\n{screen}"
    );

    // The blocks are one per call, not one call painted twice.
    let blocks = app
        .messages
        .iter()
        .filter_map(|m| m.tool.as_ref())
        .collect::<Vec<_>>();
    assert_eq!(blocks.len(), 4, "one block per call");
    let mut names: Vec<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["list_dir", "read_file", "read_file", "shell"],
        "each call keeps its own name"
    );
    for block in &blocks {
        assert!(block.done, "every call of a finished run has returned");
        assert!(!block.streaming, "no call is still streaming");
        assert!(block.call.is_some(), "every block is bound to its call");
    }
}

#[test]
fn a_tool_call_is_collapsed_by_default() {
    let (mut app, mut rx) = tool_run();
    run_and_paint(&mut app, &mut rx, "list src");
    let screen = transcript_only(&app);

    assert!(
        screen.contains("list_dir"),
        "the call should be named in the transcript:\n{screen}"
    );

    // The marker is asserted on the call's own row, not on the screen: the
    // group heading above it carries a `▾` of its own, because the group is
    // open by default. The two markers mean different things.
    let row = screen
        .lines()
        .find(|r| r.contains("list_dir"))
        .expect("the call row");
    assert!(
        row.contains('▸'),
        "a collapsed block shows a closed marker: {row:?}"
    );
    assert!(
        !row.contains('▾'),
        "the block itself should not be expanded: {row:?}"
    );
    assert!(
        !screen.contains("main.rs"),
        "a collapsed block must not paint its result body:\n{screen}"
    );
}

#[test]
fn expanding_a_tool_block_reveals_its_result() {
    let (mut app, mut rx) = tool_run();
    run_and_paint(&mut app, &mut rx, "list src");

    let idx = app.last_tool_index().expect("a tool block");
    app.toggle_expanded(idx);

    // Painted tall enough for the whole transcript: expanding a block makes
    // it as tall as the listing it holds, and at a fixed short height the
    // heading carrying the marker scrolls off the top. The assertion would
    // then fail for a reason that has nothing to do with the marker, and
    // would start failing again every time a file is added to `src/`.
    let screen = transcript_whole(&app);
    assert!(
        screen.contains('▾'),
        "an expanded block shows an open marker:\n{screen}"
    );
    assert!(
        screen.contains("main.rs"),
        "the result body should be painted once expanded:\n{screen}"
    );
    assert!(
        screen
            .lines()
            .any(|row| row.trim_start().starts_with("▏ main.rs")),
        "each result line hangs off a rule in the border colour:\n{screen}"
    );
    assert!(
        !screen
            .lines()
            .any(|row| row.starts_with('│') && row.contains(".rs")),
        "a result line is ruled, not boxed in:\n{screen}"
    );

    app.toggle_expanded(idx);
    let screen = transcript_whole(&app);
    assert!(
        !screen.contains("main.rs"),
        "collapsing should hide the body again:\n{screen}"
    );
}

#[test]
fn the_tool_block_is_ordered_before_the_answer_it_produced() {
    let (mut app, mut rx) = tool_run();
    run_and_paint(&mut app, &mut rx, "list src");

    let tool_at = app
        .messages
        .iter()
        .position(|m| m.role == rust_agent::app::Role::Tool)
        .expect("a tool block");
    let answer_at = app
        .messages
        .iter()
        .position(|m| {
            m.role == rust_agent::app::Role::Assistant
                && m.content.contains("the directory is listed")
        })
        .expect("the answer");

    assert!(
        tool_at < answer_at,
        "the call must be painted before the answer it produced, got order: {:?}",
        app.messages
            .iter()
            .map(|m| m.role)
            .collect::<Vec<rust_agent::app::Role>>()
    );
}

#[test]
fn messages_are_separated_by_blank_lines() {
    let (mut app, mut rx) = tool_run();
    run_and_paint(&mut app, &mut rx, "list src");
    let rows = paint(&app, 60, 30);

    // A message has no label row, so each side of the exchange is located by
    // its own text: the question opens with the prompt glyph, and the reply is
    // the row the answer starts on.
    let question = rows
        .iter()
        .position(|r| r.contains("❯ list src"))
        .expect("the question's row");
    assert!(
        is_blank_row(&rows[question + 1]),
        "a blank line should follow the question:\n{}",
        screen_text(&rows)
    );

    let reply = rows
        .iter()
        .position(|r| r.trim_start().starts_with("the directory is listed"))
        .expect("the reply's row");
    assert!(
        is_blank_row(&rows[reply - 1]),
        "a blank line should precede the reply:\n{}",
        screen_text(&rows)
    );
}

#[test]
fn toggling_all_expands_every_block() {
    let (mut app, mut rx) = tool_run();
    run_and_paint(&mut app, &mut rx, "list src");

    app.toggle_all_expanded();
    let screen = transcript_only(&app);
    assert!(
        screen.contains("main.rs"),
        "every block should be expanded:\n{screen}"
    );

    app.toggle_all_expanded();
    let screen = transcript_only(&app);
    assert!(
        !screen.contains("main.rs"),
        "every block should be collapsed again:\n{screen}"
    );
}

// ------------------------------------------------------------ tool groups

#[test]
fn calls_of_one_kind_are_grouped_under_a_heading() {
    let (mut app, mut rx) = mixed_run();
    run_and_paint(&mut app, &mut rx, "look around");
    let screen = transcript_whole(&app);

    assert!(
        screen.contains("EXPLORE"),
        "the reading calls should be headed by their category:\n{screen}"
    );
    assert!(
        screen.contains("RUN"),
        "the shell call should be headed by its own category:\n{screen}"
    );

    let rows: Vec<&str> = screen.lines().collect();
    let heading = rows
        .iter()
        .position(|r| r.contains("EXPLORE"))
        .expect("the EXPLORE heading");
    let first_call = rows
        .iter()
        .position(|r| r.contains("list_dir"))
        .expect("the first call");
    assert!(
        heading < first_call,
        "the heading should precede the calls it heads:\n{screen}"
    );

    // The calls sit under the heading rather than beside it, which is what
    // makes the run read as one block.
    let indent = |row: &str| row.chars().take_while(|c| *c == ' ').count();
    assert!(
        indent(rows[first_call]) > indent(rows[heading]),
        "the calls should be indented under their heading:\n{screen}"
    );
}

#[test]
fn a_group_heading_counts_its_calls() {
    let (mut app, mut rx) = mixed_run();
    run_and_paint(&mut app, &mut rx, "look around");
    let screen = transcript_whole(&app);

    let heading = screen
        .lines()
        .find(|r| r.contains("EXPLORE"))
        .expect("the EXPLORE heading");
    assert!(
        heading.contains("3 calls"),
        "the heading should count the run: {heading:?}"
    );

    let run = screen
        .lines()
        .find(|r| r.contains("RUN"))
        .expect("the RUN heading");
    assert!(
        run.contains("1 call") && !run.contains("1 calls"),
        "a single call should not be pluralised: {run:?}"
    );
}

#[test]
fn calls_of_different_kinds_are_not_merged_into_one_group() {
    // read_file, shell, read_file: the shell call splits the run, so the two
    // reads are two groups rather than one. Grouping follows the order the
    // calls happened in; it does not reorder them by kind.
    let (mut app, mut rx) = app_with(vec![
        vec![
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "src/app.rs"}),
                call_id: None,
            },
            MockStreamEvent::ToolCall {
                id: "t1".to_string(),
                name: "shell".to_string(),
                arguments: serde_json::json!({"command": "echo hi"}),
                call_id: None,
            },
            MockStreamEvent::ToolCall {
                id: "t2".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "src/render.rs"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::Text("done".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
    ]);
    run_and_paint(&mut app, &mut rx, "look around");

    let kinds: Vec<&str> = app
        .tool_groups()
        .iter()
        .map(|g| g.category.label())
        .collect();
    assert_eq!(
        kinds,
        vec!["EXPLORE", "RUN", "EXPLORE"],
        "a call of another kind should split the run"
    );
}

#[test]
fn folding_a_group_replaces_its_calls_with_a_digest() {
    let (mut app, mut rx) = mixed_run();
    run_and_paint(&mut app, &mut rx, "look around");

    let first = app.tool_groups().first().expect("a group").first;
    app.toggle_group(first);
    let screen = transcript_whole(&app);

    assert!(
        !screen.contains("list_dir  path"),
        "a folded group should not list its calls:\n{screen}"
    );
    assert!(
        screen.contains("read_file ×2"),
        "a folded group should say what it was made of:\n{screen}"
    );
    assert!(
        screen.contains("3 calls"),
        "a folded group should still count its calls:\n{screen}"
    );
    assert!(
        screen.contains("shell"),
        "folding one group must not fold another:\n{screen}"
    );
}

#[test]
fn folding_a_group_keeps_every_call_in_the_conversation() {
    // The grouping organises the transcript; it must not drop anything from
    // it. A folded group hides its calls from the screen, not from the app.
    let (mut app, mut rx) = mixed_run();
    run_and_paint(&mut app, &mut rx, "look around");

    let calls = |app: &App| {
        app.messages
            .iter()
            .filter(|m| m.role == rust_agent::app::Role::Tool)
            .count()
    };
    let before = calls(&app);
    assert_eq!(before, 4, "the run made four calls");

    let first = app.tool_groups().first().expect("a group").first;
    app.toggle_group(first);

    assert_eq!(calls(&app), before, "folding must not drop a call");
}

#[test]
fn folding_all_groups_folds_every_one_and_unfolds_them_again() {
    let (mut app, mut rx) = mixed_run();
    run_and_paint(&mut app, &mut rx, "look around");

    app.toggle_all_groups();
    assert!(
        app.tool_groups()
            .iter()
            .all(|g| app.is_group_folded(g.first)),
        "every group should be folded"
    );
    let screen = transcript_whole(&app);
    assert!(
        screen.contains("read_file ×2") && screen.contains("shell"),
        "each folded group should show its digest:\n{screen}"
    );
    assert!(
        !screen.contains("list_dir  path"),
        "no folded group should list its calls:\n{screen}"
    );

    app.toggle_all_groups();
    assert!(
        app.tool_groups()
            .iter()
            .all(|g| !app.is_group_folded(g.first)),
        "every group should be open again"
    );
}

#[test]
fn a_group_heading_is_never_cut_to_make_room_for_its_tail() {
    // At 30 columns the heading and its tail cannot both fit. The heading is
    // the content, so the tail sheds rather than the label being cut to
    // `EXPLO` to make room for a line count.
    let (mut app, mut rx) = mixed_run();
    run_and_paint(&mut app, &mut rx, "look around");

    let screen = screen_text(&paint(&app, 30, 40));
    assert!(
        screen.contains("EXPLORE"),
        "the heading should survive a narrow column:\n{screen}"
    );
    assert!(
        !screen.contains("953 lines"),
        "the tail should shed before the heading is cut:\n{screen}"
    );
}

// ---------------------------------------------------------------- layout

#[test]
fn a_wrapped_answer_keeps_its_gutter_on_every_line() {
    // A long paragraph wraps across several rows. `Paragraph`'s own wrapping
    // starts each continuation at column 0, so the gutter has to be applied
    // per visual line; otherwise the answer falls out of its own margin and
    // the reply stops reading as one block.
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("word ".repeat(60)),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);

    run_and_paint(&mut app, &mut rx, "long");

    let rows = paint(&app, 60, 30);
    let body: Vec<&String> = rows.iter().filter(|r| r.contains("word")).collect();
    assert!(
        body.len() > 1,
        "the answer should wrap across rows:\n{}",
        screen_text(&rows)
    );

    // The gutter is the body indent, so every visual line of the answer has to
    // start at the same column as the first one.
    let indent = |row: &str| row.chars().take_while(|c| *c == ' ').count();
    let first = indent(body[0]);
    assert!(
        first > 0,
        "the answer should be indented under its label: {:?}",
        body[0]
    );
    for row in &body {
        assert_eq!(
            indent(row),
            first,
            "every wrapped line should keep the gutter: {row:?}"
        );
    }
}

#[test]
fn the_transcript_is_inset_from_the_column_edge() {
    let (mut app, mut rx) = tool_run();
    run_and_paint(&mut app, &mut rx, "list src");

    let rows = paint(&app, 60, 30);
    let content: Vec<&String> = rows
        .iter()
        .filter(|r| r.contains("list_dir") || r.contains("list src"))
        .collect();
    assert!(
        !content.is_empty(),
        "no content rows to measure:\n{}",
        screen_text(&rows)
    );

    // The transcript has no border of its own: a box around every answer would
    // only eat columns. What insets the text is the column's padding.
    for row in content {
        assert!(
            row.starts_with("  "),
            "content should be inset from the column edge: {row:?}"
        );
        assert!(
            !row.starts_with('│') && !row.starts_with('┌'),
            "the transcript should be borderless: {row:?}"
        );
    }
}

#[test]
fn the_transcript_column_is_not_cut_by_a_rule() {
    let (mut app, mut rx) = tool_run();
    run_and_paint(&mut app, &mut rx, "list src");

    // The rail and the activity panel are gone: the transcript is the whole
    // width at every size. What would come back is a fixed column split, so
    // the check is that no single column is ruled down most of the body, and
    // that neither panel's label survives.
    for width in [100u16, 140, 300] {
        let rows = paint(&app, width, 30);
        let screen = screen_text(&rows);

        assert!(
            screen.contains("Agent online"),
            "the header should still carry the run's status:\n{screen}"
        );
        for label in ["TOOLS", "CONTEXT", "ACTIVITY"] {
            assert!(
                !screen.contains(label),
                "the {label} panel is gone, so its label must not return:\n{screen}"
            );
        }

        // The input box is drawn on the last rows only, so it is left out:
        // its own border would read as a rule.
        let body = &rows[..rows.len().saturating_sub(4)];
        for col in 0..width as usize {
            let ruled = body
                .iter()
                .filter(|r| r.chars().nth(col) == Some('│'))
                .count();
            assert!(
                ruled < 20,
                "a rule cuts the column at {col} ({ruled} rows) at {width} wide:\n{screen}"
            );
        }
    }
}

#[test]
fn a_narrow_footer_drops_its_key_hints_rather_than_clipping() {
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("hi".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new().input_tokens(11).output_tokens(4))),
    ]]);
    run_and_paint(&mut app, &mut rx, "hi");

    // The accounting is what the footer is for, so it survives a narrow
    // terminal; the key hints are the first thing to go, because a hint cut
    // off mid-word is worse than an absent one.
    let narrow = paint(&app, 60, 20).join("\n");
    assert!(
        narrow.contains("11 in"),
        "the token count should survive a narrow footer:\n{narrow}"
    );
    assert!(
        !narrow.contains("Esc stop"),
        "the key hints should be dropped rather than clipped:\n{narrow}"
    );

    let wide = paint(&app, 140, 20).join("\n");
    assert!(
        wide.contains("Esc stop"),
        "a wide footer should show the key hints:\n{wide}"
    );
}

#[test]
fn consecutive_tool_calls_are_grouped_without_a_blank_line() {
    // A run of calls is one group. A blank line between each would break the
    // run into unrelated fragments and push the answer off the screen.
    let (mut app, mut rx) = app_with(vec![
        vec![
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "list_dir".to_string(),
                arguments: serde_json::json!({"path": "src"}),
                call_id: None,
            },
            MockStreamEvent::ToolCall {
                id: "t1".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "src/main.rs"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
        vec![
            MockStreamEvent::Text("done".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new())),
        ],
    ]);

    run_and_paint(&mut app, &mut rx, "go");

    // Two calls are two blocks, each committed with its own arguments. The
    // stream opens a placeholder per call and reports the committed calls
    // afterwards, so a block that is still `streaming` here means a
    // placeholder was never filled in and its result went to the wrong block.
    let blocks: Vec<&rust_agent::app::ToolBlock> = app
        .messages
        .iter()
        .filter_map(|m| m.tool.as_ref())
        .collect();
    assert_eq!(
        blocks.len(),
        2,
        "two calls should be two blocks: {blocks:#?}"
    );
    assert_eq!(blocks[0].name, "list_dir", "{blocks:#?}");
    assert_eq!(blocks[1].name, "read_file", "{blocks:#?}");
    assert!(
        !blocks[0].streaming,
        "the first call's placeholder was never committed: {blocks:#?}"
    );
    assert!(
        !blocks[1].streaming,
        "the second call's placeholder was never committed: {blocks:#?}"
    );

    let rows = paint(&app, 60, 30);
    let first = rows
        .iter()
        .position(|r| r.contains("list_dir"))
        .expect("the first call");
    let second = rows
        .iter()
        .position(|r| r.contains("read_file"))
        .expect("the second call");
    assert_eq!(
        second,
        first + 1,
        "consecutive calls should be adjacent:\n{}",
        screen_text(&rows)
    );
}

// ------------------------------------------------------- the view's bounds

#[test]
fn a_frame_reports_the_transcripts_scrollable_height() {
    // The key handler clamps the scroll offset against this, so a frame that
    // never reports it would pin the view to the top: every scroll key would
    // be clamped to zero and the transcript would look frozen.
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("one\ntwo\nthree\nfour\nfive\nsix\nseven\neight".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);

    run_and_paint(&mut app, &mut rx, "go");
    assert_eq!(
        app.scroll_max(),
        0,
        "a short transcript has nothing to scroll"
    );

    // A tall transcript in a short window does have somewhere to go.
    let mut long = String::new();
    for i in 0..200 {
        long.push_str(&format!("line {i}\n"));
    }
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text(long),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);
    run_and_paint(&mut app, &mut rx, "go");
    paint(&app, 60, 20);

    assert!(
        app.scroll_max() > 0,
        "the frame must report the height the scroll keys clamp against"
    );
}

#[test]
fn the_layout_survives_a_terminal_too_short_for_its_parts() {
    // The header, the body, the input box and the footer are fixed heights,
    // and the input box grows with the prompt. On a terminal shorter than
    // their sum the layout has to give something up rather than panic or
    // paint outside the buffer.
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("hello".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);
    run_and_paint(&mut app, &mut rx, "hi");

    // Counting the painted rows proves nothing: they come from the buffer, so
    // the count is the requested height whatever the layout did. What the
    // layout has to get right is *which* parts survive. The box the user
    // types into is the only interactive part of the screen, so it is the
    // last thing to be given up and has to be on screen at every height.
    for height in 1..14u16 {
        for width in [20u16, 40, 100] {
            let screen = screen_text(&paint(&app, width, height));
            assert!(
                screen.contains('╭'),
                "the input box must survive {width}x{height}:\n{screen}"
            );
        }
    }

    // And with a multi-line prompt, which asks for the tallest input box.
    app.set_input("line\n".repeat(20));
    for height in 1..14u16 {
        let screen = screen_text(&paint(&app, 60, height));
        assert!(
            screen.contains('╭'),
            "the input box must survive 60x{height} with a tall prompt:\n{screen}"
        );
    }
}

#[test]
fn a_short_terminal_gives_up_its_parts_in_order_of_usefulness() {
    // The chrome is fixed-height, so on a terminal too short for it the
    // constraints would sum to more than the height and ratatui's solver
    // would drop rows arbitrarily: the same height could show the input box
    // on one run and a blank gap on the next. The order is fixed instead, and
    // it is the order of usefulness: the box the user types into, then the
    // status line, then the header.
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("hello".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);
    run_and_paint(&mut app, &mut rx, "hi");

    let one = screen_text(&paint(&app, 60, 1));
    assert!(
        one.contains('╭') && !one.contains("calls"),
        "one row is the input box alone:\n{one}"
    );

    let two = screen_text(&paint(&app, 60, 2));
    assert!(
        two.contains('╭') && two.contains("calls"),
        "two rows add the status line:\n{two}"
    );

    let three = screen_text(&paint(&app, 60, 3));
    assert!(
        three.contains('╭') && three.contains("calls") && three.contains("Agent online"),
        "three rows add the header:\n{three}"
    );
}

#[test]
fn a_short_terminal_keeps_the_box_the_user_types_into() {
    // The input box is the only interactive part of the screen, so it is the
    // last thing to be squeezed: a terminal with room for the header and the
    // box must still show the box, even with a prompt too tall to fit.
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("hello".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);
    run_and_paint(&mut app, &mut rx, "hi");
    app.set_input("line\n".repeat(20));

    for height in 7..14u16 {
        let rows = paint(&app, 60, height);
        let screen = screen_text(&rows);
        assert!(
            screen.contains('╭') && screen.contains('╰'),
            "the box's borders are missing at 60x{height}:\n{screen}"
        );
        // At least three rows: two borders and a line of text between them.
        // It grows with the terminal, but never past a third of the column.
        let top = rows
            .iter()
            .position(|r| r.contains('╭'))
            .expect("top border");
        let bottom = rows
            .iter()
            .position(|r| r.contains('╰'))
            .expect("bottom border");
        let box_h = bottom - top + 1;
        assert!(
            box_h >= 3,
            "the box should keep a content row at 60x{height}, got {box_h}:\n{screen}"
        );
        assert!(
            box_h <= (height / 3).max(3) as usize,
            "the box should not take more than a third of the column at \
             60x{height}, got {box_h}:\n{screen}"
        );
    }

    // With a prompt that fits, the marker and the text are on screen.
    app.set_input("hello");
    let screen = screen_text(&paint(&app, 60, 7));
    assert!(
        screen.contains("❯ hello"),
        "the prompt should be visible at 60x7:\n{screen}"
    );
}

// ------------------------------------------------------------- the footer

#[test]
fn the_footer_hints_name_the_keys_that_are_actually_bound() {
    // The footer is a second copy of the key map, and it went stale the same
    // way the help text did: it still said `↑↓ scroll` after Up/Down became
    // the prompt history, and `Esc quit` after Esc became "stop the run".
    // A hint that names the wrong key is worse than no hint, because the user
    // tries it and the app does something else.
    let (mut app, mut rx) = app_with(vec![vec![
        MockStreamEvent::Text("hello".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]]);
    run_and_paint(&mut app, &mut rx, "hi");

    // Wide enough that the footer is not shedding segments from the right.
    let screen = screen_text(&paint(&app, 200, 40));

    for expected in ["Enter send", "PgUp/PgDn", "Ctrl+T/G fold", "Esc stop"] {
        assert!(
            screen.contains(expected),
            "the footer should hint {expected}:\n{screen}"
        );
    }

    // And it must not still name the map it replaced.
    for stale in ["↑↓ scroll", "Esc quit"] {
        assert!(
            !screen.contains(stale),
            "the footer still hints {stale}, which is no longer bound:\n{screen}"
        );
    }

    // The follow-mode label names the key that returns to the tail, which is
    // `Ctrl+End` now that `End` is the caret's line.
    app.scroll_by(-5);
    let screen = screen_text(&paint(&app, 200, 40));
    assert!(
        screen.contains("Ctrl+End to follow"),
        "the scrolled label should name the key that follows again:\n{screen}"
    );
}

// ------------------------------------------------------- the status line

/// The header row, which is where the run's status is painted.
fn header(app: &App) -> String {
    paint(app, 100, 20)[0].clone()
}

#[test]
fn the_status_line_names_the_act_rather_than_the_tool() {
    // The tool's name is the mechanism; the act is what the user is watching
    // for. A read and a grep are both searching, a write is a change to the
    // project, and a command that checks the change is the run verifying
    // itself. The categories are the ones the transcript already groups by, so
    // the two surfaces agree about what kind of thing is happening.
    let (mut app, _rx) = app_with(Vec::new());
    app.set_streaming(true, "thinking");

    for (name, arguments, expected) in [
        (
            "read_file",
            serde_json::json!({ "path": "src/app.rs" }),
            "searching",
        ),
        (
            "grep",
            serde_json::json!({ "pattern": "fn main" }),
            "searching",
        ),
        (
            "write_file",
            serde_json::json!({ "path": "src/x.rs", "content": "" }),
            "applying changes",
        ),
        (
            "shell",
            serde_json::json!({ "command": "cargo test --locked" }),
            "running checks",
        ),
        (
            "shell",
            serde_json::json!({ "command": "rm -rf target" }),
            "running shell",
        ),
    ] {
        app.handle_event(Event::ToolCallStarted(Box::new(call_with(
            "call_1", name, arguments,
        ))));
        let header = header(&app);
        assert!(
            header.contains(expected),
            "a {name} call should read {expected:?}, but the header was:\n{header}"
        );
    }
}

#[tokio::test]
async fn a_run_blocked_on_approval_says_so_instead_of_spinning() {
    // A run waiting on the user is the one state the user has to act on. A
    // spinner would say the opposite: that the run is working and the user
    // should wait.
    let (mut app, _rx) = app_with(Vec::new());
    app.set_streaming(true, "thinking");

    let spinning = header(&app);
    assert!(
        spinning.contains("thinking…"),
        "a working run spins:\n{spinning}"
    );

    // The real producer: the approver the binary installs, asking the UI. A
    // hand-built request would prove the renderer reads a field, not that the
    // field is ever filled.
    let (tx, mut requests) = tokio::sync::mpsc::unbounded_channel();
    let approver = rust_agent::approval::InteractiveApprover::new(tx);
    let asked = tokio::spawn(async move {
        use rust_agent::approval::Approver;
        approver
            .decide(rust_agent::approval::ApprovalRequest::new(
                rust_agent::approval::ActionKind::Write,
                "write src/x.rs",
            ))
            .await
    });

    app.approval = requests.recv().await;
    assert!(app.is_awaiting_approval(), "the request reached the app");

    let blocked = header(&app);
    assert!(
        blocked.contains("awaiting approval"),
        "a blocked run says what it is waiting for:\n{blocked}"
    );
    assert!(
        !blocked.contains("thinking…"),
        "and stops claiming to be working:\n{blocked}"
    );

    // Answering it puts the run back to work, so the state is not sticky.
    app.answer_approval(true);
    let resumed = header(&app);
    assert!(
        resumed.contains("thinking…"),
        "an answered run goes back to work:\n{resumed}"
    );
    assert!(
        asked.await.expect("the asker did not panic").is_approved(),
        "the answer reached the approver"
    );
}

#[test]
fn a_failed_result_reads_as_a_correction_rather_than_as_thinking() {
    // A failed result is the model's cue to correct itself. Saying so is more
    // use than the generic word: a user watching a run that keeps failing
    // wants to know the agent knows, and a run that is correcting is not the
    // same as one that is stuck.
    let (mut app, _rx) = app_with(Vec::new());
    app.set_streaming(true, "thinking");

    let mut failed = result("call_1", "shell", "error: expected `;`");
    failed.is_error = true;
    app.handle_event(Event::ToolResult(Box::new(failed)));

    let after_failure = header(&app);
    assert!(
        after_failure.contains("correcting"),
        "a failed result reads as a correction:\n{after_failure}"
    );

    app.handle_event(Event::ToolResult(Box::new(result(
        "call_2",
        "shell",
        "test result: ok",
    ))));
    let after_success = header(&app);
    assert!(
        after_success.contains("thinking"),
        "a result that worked leaves the model to think:\n{after_success}"
    );
}

// ------------------------------------------------- calls and results by id

/// A call as the stream reports it, with the id the result will carry.
fn call(id: &str, name: &str) -> rig::message::ToolCall {
    rig::message::ToolCall {
        id: rig::message::CallId::from_wire(id),
        function: rig::message::ToolFunction::new(
            rig::message::ToolName::new(name).unwrap(),
            serde_json::json!({ "path": "a.rs" }),
        ),
        native: None,
    }
}

/// A call with the arguments the status line has to read.
fn call_with(id: &str, name: &str, arguments: serde_json::Value) -> rig::message::ToolCall {
    rig::message::ToolCall {
        id: rig::message::CallId::from_wire(id),
        function: rig::message::ToolFunction::new(
            rig::message::ToolName::new(name).unwrap(),
            arguments,
        ),
        native: None,
    }
}

/// The result of the call with this id.
fn result(id: &str, name: &str, text: &str) -> rig::message::ToolResult {
    rig::message::ToolResult {
        call: rig::message::CallId::from_wire(id),
        name: rig::message::ToolName::new(name).unwrap(),
        content: vec![rig::message::ToolResultContent::Text(
            rig::message::Text::new(text),
        )],
        is_error: false,
    }
}

/// The result of the call with this id, as rig reports a call that did not
/// succeed.
///
/// rig sets `is_error` for anything that is not a success, a refusal and a
/// cancellation included, so a test that wants the transcript to see a
/// refusal has to set it the way the real path does.
fn error_result(id: &str, name: &str, text: &str) -> rig::message::ToolResult {
    let mut r = result(id, name, text);
    r.is_error = true;
    r
}

/// The result body recorded against the block for `name`.
fn result_for(app: &App, name: &str) -> Option<String> {
    app.messages
        .iter()
        .filter_map(|m| m.tool.as_ref())
        .find(|b| b.name == name)
        .map(|b| b.result.clone())
}

#[test]
fn a_result_lands_on_the_block_its_call_opened() {
    // Two calls in one turn, and their results delivered in the order the
    // calls were made. The association is by call id, so the second result
    // must not land on the first block even though the first block is the
    // one still waiting for a result when it arrives.
    let (mut app, _rx) = app_with(Vec::new());

    app.handle_event(Event::ToolCallStreaming {
        name: "read_file".to_string(),
        part: 0,
    });
    app.handle_event(Event::ToolCallIdentified {
        part: 0,
        call: "call_a".to_string(),
    });
    app.handle_event(Event::ToolCallStarted(Box::new(call(
        "call_a",
        "read_file",
    ))));
    app.handle_event(Event::ToolCallStreaming {
        name: "grep".to_string(),
        part: 1,
    });
    app.handle_event(Event::ToolCallIdentified {
        part: 1,
        call: "call_b".to_string(),
    });
    app.handle_event(Event::ToolCallStarted(Box::new(call("call_b", "grep"))));

    app.handle_event(Event::ToolResult(Box::new(result(
        "call_a",
        "read_file",
        "the manifest",
    ))));
    app.handle_event(Event::ToolResult(Box::new(result(
        "call_b",
        "grep",
        "three matches",
    ))));

    assert_eq!(
        result_for(&app, "read_file").as_deref(),
        Some("the manifest"),
        "the first result belongs to the first call"
    );
    assert_eq!(
        result_for(&app, "grep").as_deref(),
        Some("three matches"),
        "the second result belongs to the second call"
    );
}

#[test]
fn a_result_that_arrives_out_of_order_still_finds_its_call() {
    // The case position alone cannot handle. Rig surfaces a batch's results
    // in call order, but a batch that is interrupted surfaces none, and a
    // block whose call was skipped by a hook never committed at all. Keying
    // on the id is what makes the association independent of arrival order.
    let (mut app, _rx) = app_with(Vec::new());

    for (part, id, name) in [(0, "call_a", "read_file"), (1, "call_b", "grep")] {
        app.handle_event(Event::ToolCallStreaming {
            name: name.to_string(),
            part,
        });
        app.handle_event(Event::ToolCallIdentified {
            part,
            call: id.to_string(),
        });
        app.handle_event(Event::ToolCallStarted(Box::new(call(id, name))));
    }

    // The second call's result first.
    app.handle_event(Event::ToolResult(Box::new(result(
        "call_b",
        "grep",
        "three matches",
    ))));
    app.handle_event(Event::ToolResult(Box::new(result(
        "call_a",
        "read_file",
        "the manifest",
    ))));

    assert_eq!(
        result_for(&app, "grep").as_deref(),
        Some("three matches"),
        "the out-of-order result still finds its own call"
    );
    assert_eq!(
        result_for(&app, "read_file").as_deref(),
        Some("the manifest"),
        "and the later one finds the block the first result skipped"
    );
}

#[test]
fn a_call_whose_part_never_ended_still_gets_its_result() {
    // A provider that does not end the part leaves the placeholder unbound.
    // The fallback is the oldest block still waiting, which is what the
    // transcript did before ids were carried at all: the association is
    // exact when the id is known and no worse than it was when it is not.
    let (mut app, _rx) = app_with(Vec::new());

    app.handle_event(Event::ToolCallStreaming {
        name: "read_file".to_string(),
        part: 0,
    });
    app.handle_event(Event::ToolCallStarted(Box::new(call(
        "call_a",
        "read_file",
    ))));
    app.handle_event(Event::ToolResult(Box::new(result(
        "call_a",
        "read_file",
        "the manifest",
    ))));

    assert_eq!(
        result_for(&app, "read_file").as_deref(),
        Some("the manifest"),
        "the result reaches the placeholder the commit filled"
    );
}

#[test]
fn a_placeholder_whose_call_never_committed_is_closed_out() {
    // A part opens a block before its call has an id. If the provider call
    // ends with the block still unbound, the call was dropped: rig rejects a
    // call it cannot route, and a rejected call is never executed. The block
    // must not be left waiting, because part positions restart at zero on the
    // next model call, so the next turn's first part would bind to it and the
    // transcript would show the wrong call under the wrong name.
    let (mut app, _rx) = app_with(Vec::new());

    app.handle_event(Event::ToolCallStreaming {
        name: "read_file".to_string(),
        part: 0,
    });
    // The provider call ends without the part ever naming a call.
    app.handle_event(Event::Completion(Box::new(CompletionCall::new(
        0,
        Usage::new(),
        serde_json::Value::Null,
    ))));

    let dropped = app
        .messages
        .iter()
        .find_map(|m| m.tool.as_ref())
        .expect("the part opened a block");
    assert!(!dropped.streaming, "a dropped call is not still streaming");
    assert!(dropped.done, "a dropped call is closed out");
    assert!(
        dropped.part.is_none(),
        "and is no longer a candidate for a later part"
    );

    // The next turn's first part opens its own block rather than binding to
    // the dropped one.
    app.handle_event(Event::ToolCallStreaming {
        name: "grep".to_string(),
        part: 0,
    });
    app.handle_event(Event::ToolCallIdentified {
        part: 0,
        call: "call_b".to_string(),
    });
    app.handle_event(Event::ToolCallStarted(Box::new(call("call_b", "grep"))));

    let blocks: Vec<&rust_agent::app::ToolBlock> = app
        .messages
        .iter()
        .filter_map(|m| m.tool.as_ref())
        .collect();
    assert_eq!(blocks.len(), 2, "the dropped call and the new one");
    assert_eq!(
        blocks[0].name, "read_file",
        "the dropped call keeps the name it was opened with"
    );
    assert_eq!(
        blocks[1].name, "grep",
        "the new call gets a block of its own"
    );
    assert_eq!(
        blocks[1].call.as_deref(),
        Some("call_b"),
        "and the new call's id is bound to it"
    );
}

#[test]
fn a_denied_call_reaches_the_model_with_its_reason_and_is_not_painted_as_a_failure() {
    // Two things have to hold at once, and they are the two halves of the
    // same defect. The model must read *why* the action was refused, because
    // a model told only "the tool failed" retries the call the user just
    // declined. And the transcript must not paint the refusal with the cross
    // of a broken build, because a user who said no has not seen anything go
    // wrong.
    let dir = tempfile::tempdir().unwrap();
    let approver = std::sync::Arc::new(rust_agent::approval::ScriptedApprover::new(vec![
        rust_agent::approval::Decision::Denied("not this one".to_string()),
    ]));
    let rt = common::session_with_approver(dir.path(), approver);
    let model = MockCompletionModel::from_stream_turns(vec![
        vec![
            MockStreamEvent::tool_call(
                "call_1",
                "write_file",
                serde_json::json!({ "path": "a.txt", "content": "hi\n" }),
            ),
            MockStreamEvent::FinalResponse(mock_final(Usage::new().input_tokens(100))),
        ],
        vec![
            MockStreamEvent::Text("done".to_string()),
            MockStreamEvent::FinalResponse(mock_final(Usage::new().input_tokens(100))),
        ],
    ]);
    let agent = rig::agent::AgentBuilder::new(model.clone())
        .tool(tools::WriteFile::new(rt.clone()))
        .build();
    let summariser = rig::agent::AgentBuilder::new(model).build();
    let (mut app, mut rx) = App::new(
        rust_agent::agent::Agents {
            main: agent,
            summariser,
        },
        rt,
    );
    drive(&mut app, &mut rx, "write a file", |_| {});

    let block = app
        .messages
        .iter()
        .filter_map(|m| m.tool.as_ref())
        .find(|b| b.name == "write_file")
        .expect("the denied call has a block");

    assert!(
        block.result.contains("not this one"),
        "the model must read the reason it was refused, not a generic failure: {:?}",
        block.result
    );
    assert_eq!(
        block.outcome,
        rust_agent::app::ToolOutcome::Denied,
        "a refusal is not a failure: {:?}",
        block.outcome
    );
    assert!(
        !block.is_error(),
        "a denial must not be counted as an error"
    );
    assert!(
        !dir.path().join("a.txt").exists(),
        "a denied write must leave no file behind"
    );
}

#[test]
fn a_result_for_a_call_that_never_opened_a_block_is_not_lost() {
    // A hook-skipped call reports no commit, so no block was opened for it.
    // Dropping the result would leave the model's answer unexplained in the
    // transcript, which is the one thing the transcript exists to prevent.
    let (mut app, _rx) = app_with(Vec::new());

    app.handle_event(Event::ToolResult(Box::new(result(
        "call_orphan",
        "read_file",
        "the manifest",
    ))));

    assert_eq!(
        result_for(&app, "read_file").as_deref(),
        Some("the manifest"),
        "a result with no block makes one"
    );
}

// ------------------------------------------------------- how a call ended

/// The block for `name`, which is the one the assertions below are about.
fn block_for<'a>(app: &'a App, name: &str) -> &'a rust_agent::app::ToolBlock {
    app.messages
        .iter()
        .filter_map(|m| m.tool.as_ref())
        .find(|b| b.name == name)
        .unwrap_or_else(|| panic!("no block for {name}"))
}

#[test]
fn a_cancelled_run_does_not_leave_a_call_looking_live() {
    // A call that was in flight when the run was abandoned never returns a
    // result, so nothing else will ever fill its block in. Left as it stands
    // it reads as still running for the rest of the session, and the user
    // waits on a call that is never coming back.
    let (mut app, _rx) = app_with(Vec::new());

    app.handle_event(Event::ToolCallStreaming {
        name: "shell".to_string(),
        part: 0,
    });
    app.handle_event(Event::ToolCallIdentified {
        part: 0,
        call: "call_1".to_string(),
    });
    app.handle_event(Event::ToolCallStarted(Box::new(call_with(
        "call_1",
        "shell",
        serde_json::json!({ "command": "cargo test" }),
    ))));

    assert!(
        !block_for(&app, "shell").done,
        "the call is in flight before the run is cancelled"
    );

    app.handle_event(Event::Cancelled);

    let block = block_for(&app, "shell");
    assert!(!block.streaming, "a cancelled call is not still streaming");
    assert!(block.done, "a cancelled call has ended");
    assert_eq!(
        block.outcome,
        rust_agent::app::ToolOutcome::Cancelled,
        "a cancelled call is not a failed one"
    );
    assert!(
        !block.is_error(),
        "a cancellation must not be counted as an error"
    );
}

#[test]
fn a_result_cut_to_fit_the_budget_is_marked_as_cut() {
    // The model is told in the text that the result was elided. The user is
    // not reading the text, so the row has to say it too: a truncated result
    // that reads as complete is the one way it can mislead.
    let (mut app, _rx) = app_with(Vec::new());

    app.handle_event(Event::ToolResult(Box::new(result(
        "call_1",
        "read_file",
        &format!(
            "fn main() {{}}\n\n{}4000 of 9000 characters; the first 200 and last 200 are \
             shown. Narrow the request (grep for the part you need) to see the rest.] ...\n\n\
             fn tail() {{}}",
            rust_agent::tools::ELISION_MARKER
        ),
    ))));

    assert!(
        block_for(&app, "read_file").truncated,
        "a result the budget cut must say so"
    );

    // And a result that was not cut must not claim to be.
    let (mut app, _rx) = app_with(Vec::new());
    app.handle_event(Event::ToolResult(Box::new(result(
        "call_1",
        "read_file",
        "fn main() {}",
    ))));
    assert!(
        !block_for(&app, "read_file").truncated,
        "a whole result is not truncated"
    );
}

#[test]
fn a_long_session_releases_old_bodies_but_keeps_their_rows() {
    // The transcript is the user's record of the session, so a row is never
    // dropped. The text is what grows without bound, and it is the part that
    // is only read again by scrolling back and expanding a row.
    let (mut app, _rx) = app_with(Vec::new());
    let cap = rust_agent::app::MAX_RETAINED_BODIES;

    for i in 0..cap + 5 {
        app.handle_event(Event::ToolResult(Box::new(result(
            &format!("call_{i}"),
            "read_file",
            &format!("body {i}"),
        ))));
    }

    let blocks: Vec<&rust_agent::app::ToolBlock> = app
        .messages
        .iter()
        .filter_map(|m| m.tool.as_ref())
        .collect();
    assert_eq!(blocks.len(), cap + 5, "every call keeps its row");

    let released = blocks.iter().filter(|b| b.released).count();
    assert_eq!(released, 5, "only the overflow is released");

    // The oldest are the ones released, and the newest keep their text: a run
    // is still working from the results it just read.
    assert_eq!(blocks[0].result, rust_agent::app::RELEASED_BODY);
    assert_eq!(blocks[4].result, rust_agent::app::RELEASED_BODY);
    assert_eq!(blocks[5].result, "body 5");
    assert_eq!(blocks[cap + 4].result, format!("body {}", cap + 4));

    // A released row still says what the call was and how it ended.
    assert_eq!(blocks[0].name, "read_file");
    assert!(blocks[0].done);
    assert_eq!(blocks[0].outcome, rust_agent::app::ToolOutcome::Ok);
}

#[test]
fn a_denial_is_not_counted_as_an_error_in_its_group() {
    // The group heading counts errors, and a denial is not one. Counting it
    // would tell the user their agent is broken when it is doing what they
    // said, and the count is the only summary the folded group shows.
    let (mut app, _rx) = app_with(Vec::new());

    app.handle_event(Event::ToolResult(Box::new(error_result(
        "call_1",
        "write_file",
        "denied: not this one. The user declined this action; do not repeat it.",
    ))));
    app.handle_event(Event::ToolResult(Box::new(result(
        "call_2",
        "read_file",
        "the manifest",
    ))));

    assert_eq!(
        block_for(&app, "write_file").outcome,
        rust_agent::app::ToolOutcome::Denied,
        "the refusal is read from the text the library writes for it"
    );
    assert!(
        !block_for(&app, "write_file").is_error(),
        "a denial is not an error"
    );
    assert!(
        !block_for(&app, "read_file").is_error(),
        "a result that worked is not an error"
    );
}
