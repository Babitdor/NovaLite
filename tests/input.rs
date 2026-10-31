//! Key handling: what each key does to the input buffer and the view.
//!
//! Two rules run through the whole map and are worth stating once, because
//! both have been broken here before:
//!
//! * **A letter must always be typeable.** An arm in `handle_input` runs
//!   before the input box ever sees the key, so any shortcut bound to a bare
//!   letter makes that letter impossible to type. Every shortcut carries a
//!   modifier.
//! * **A shortcut must not type.** The mirror image: Ctrl+T folds a tool
//!   block and must leave the buffer alone.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use rig::completion::Usage;
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent};

use rust_agent::app::App;
use rust_agent::event::Event;
use rust_agent::tools;
use rust_agent::tui::{handle_input, InputResult};

mod common;

type Rx = tokio::sync::mpsc::Receiver<Event>;

fn app_with(turns: Vec<Vec<MockStreamEvent>>) -> (App, Rx) {
    let model = MockCompletionModel::from_stream_turns(turns);
    let rt = common::session();
    let agent = rig::agent::AgentBuilder::new(model.clone())
        .tool(tools::ReadFile::new(rt.clone()))
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

fn new_app() -> App {
    app_with(vec![vec![
        MockStreamEvent::Text("ok".to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new())),
    ]])
    .0
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn key_with(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
    KeyEvent::new(code, modifiers)
}

fn ctrl(c: char) -> KeyEvent {
    key_with(KeyCode::Char(c), KeyModifiers::CONTROL)
}

/// Type `text` one character at a time, as the event loop does.
fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        handle_input(app, key(KeyCode::Char(c)));
    }
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
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
                    let finished = matches!(event, Event::Finished(_) | Event::Cancelled);
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

// ------------------------------------------------------------- the buffer

#[test]
fn typing_appends_and_backspace_removes() {
    let mut app = new_app();

    type_text(&mut app, "hello");
    assert_eq!(app.input(), "hello");

    handle_input(&mut app, key(KeyCode::Backspace));
    assert_eq!(app.input(), "hell");
}

#[test]
fn enter_sends_and_clears_the_buffer() {
    let mut app = new_app();
    app.set_input("do the thing");

    match handle_input(&mut app, key(KeyCode::Enter)) {
        InputResult::Send(text) => assert_eq!(text, "do the thing"),
        _ => panic!("plain Enter should send"),
    }
    assert!(app.input().is_empty(), "the buffer is cleared on send");
}

#[test]
fn shift_enter_inserts_a_newline_instead_of_sending() {
    let mut app = new_app();
    app.set_input("line one");

    let result = handle_input(&mut app, key_with(KeyCode::Enter, KeyModifiers::SHIFT));

    assert!(matches!(result, InputResult::Continue));
    assert_eq!(app.input(), "line one\n");
    assert_eq!(app.caret(), 9, "the caret follows the newline");
}

#[test]
fn tab_indents_and_unknown_keys_are_ignored() {
    let mut app = new_app();
    handle_input(&mut app, key(KeyCode::Tab));
    assert_eq!(app.input(), "    ");

    let before = app.input().to_string();
    assert!(matches!(
        handle_input(&mut app, key(KeyCode::F(5))),
        InputResult::Continue
    ));
    assert_eq!(app.input(), before, "an unhandled key changes nothing");
}

#[test]
fn a_press_event_is_what_the_loop_acts_on() {
    // The loop filters on `KeyEventKind`; a Release must not type a
    // character twice on terminals that report both.
    let press = KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE);
    assert_eq!(press.kind, KeyEventKind::Press);
}

// -------------------------------------------------------------- the caret

#[test]
fn the_caret_moves_and_edits_in_the_middle_of_the_buffer() {
    let mut app = new_app();
    app.set_input("hello world");
    assert_eq!(app.caret(), 11, "a fresh buffer puts the caret at the end");

    handle_input(&mut app, key(KeyCode::Left));
    handle_input(&mut app, key(KeyCode::Left));
    assert_eq!(app.caret(), 9);

    // Typing goes in at the caret, not at the end.
    type_text(&mut app, "X");
    assert_eq!(app.input(), "hello worXld");
    assert_eq!(app.caret(), 10);

    handle_input(&mut app, key(KeyCode::Backspace));
    assert_eq!(app.input(), "hello world");

    // Delete removes the character *under* the caret, unlike Backspace.
    handle_input(&mut app, key(KeyCode::Delete));
    assert_eq!(app.input(), "hello word");
    assert_eq!(app.caret(), 9, "Delete does not move the caret");
}

#[test]
fn home_and_end_are_the_carets_line_not_the_transcript() {
    let mut app = new_app();
    app.set_input("first line\nsecond line");

    handle_input(&mut app, key(KeyCode::Home));
    assert_eq!(
        app.caret(),
        11,
        "Home goes to the start of the caret's line"
    );

    handle_input(&mut app, key(KeyCode::End));
    assert_eq!(app.caret(), 22, "End goes to the end of the caret's line");

    // Ctrl+Home/Ctrl+End are the transcript's ends, not the caret's.
    app.scroll = 5;
    app.auto_scroll = false;
    handle_input(&mut app, ctrl('e'));
    assert_eq!(app.caret(), 22);
    handle_input(&mut app, key_with(KeyCode::End, KeyModifiers::CONTROL));
    assert!(app.auto_scroll, "Ctrl+End returns to follow mode");
}

#[test]
fn readline_bindings_edit_the_line() {
    let mut app = new_app();
    app.set_input("keep this tail");

    // Ctrl+A then Ctrl+K kills the whole line.
    handle_input(&mut app, ctrl('a'));
    assert_eq!(app.caret(), 0);
    handle_input(&mut app, ctrl('k'));
    assert_eq!(app.input(), "");

    // Ctrl+U kills back to the start of the line, leaving the tail.
    app.set_input("drop this keep");
    handle_input(&mut app, ctrl('a'));
    handle_input(&mut app, ctrl('e'));
    handle_input(&mut app, ctrl('u'));
    assert_eq!(app.input(), "", "Ctrl+U from the end clears the line");

    app.set_input("drop this keep");
    for _ in 0..4 {
        handle_input(&mut app, key(KeyCode::Left));
    }
    handle_input(&mut app, ctrl('u'));
    assert_eq!(
        app.input(),
        "keep",
        "Ctrl+U keeps what is right of the caret"
    );
}

#[test]
fn ctrl_w_deletes_the_word_before_the_caret() {
    let mut app = new_app();
    app.set_input("keep drop");

    handle_input(&mut app, ctrl('w'));
    assert_eq!(app.input(), "keep ");
    assert_eq!(app.caret(), 5);
}

#[test]
fn ctrl_arrows_move_by_word() {
    let mut app = new_app();
    app.set_input("one two three");

    handle_input(&mut app, key_with(KeyCode::Left, KeyModifiers::CONTROL));
    assert_eq!(app.caret(), 8, "back to the start of `three`");

    handle_input(&mut app, key_with(KeyCode::Left, KeyModifiers::CONTROL));
    assert_eq!(app.caret(), 4, "back to the start of `two`");

    handle_input(&mut app, key_with(KeyCode::Right, KeyModifiers::CONTROL));
    assert_eq!(app.caret(), 8, "forward past `two` and its space");
}

#[test]
fn a_multibyte_prompt_is_edited_by_character_not_byte() {
    // The caret is a character index. A byte offset would slice a multi-byte
    // glyph in half and panic, so this is the test that keeps the two apart.
    let mut app = new_app();
    app.set_input("héllo → wörld");

    // Three characters back from the end puts the caret after the `ö`, so
    // Backspace removes that glyph and not the `l` beside it.
    for _ in 0..3 {
        handle_input(&mut app, key(KeyCode::Left));
    }
    handle_input(&mut app, key(KeyCode::Backspace));
    assert_eq!(app.input(), "héllo → wrld");

    handle_input(&mut app, key(KeyCode::Home));
    handle_input(&mut app, key(KeyCode::Delete));
    assert_eq!(app.input(), "éllo → wrld");

    // And a paste of multi-byte text lands whole.
    app.set_input("");
    app.insert_str("日本語");
    assert_eq!(app.input(), "日本語");
    assert_eq!(app.caret(), 3, "three characters, not nine bytes");
}

#[test]
fn the_input_box_grows_with_the_prompt_and_then_stops() {
    let mut app = new_app();
    assert_eq!(app.input_height(), 3, "one row plus the border");

    app.set_input("one\ntwo\nthree");
    assert_eq!(app.input_height(), 5);

    // A pasted file must not push the conversation off the screen.
    app.set_input("line\n".repeat(40));
    assert_eq!(
        app.input_height(),
        rust_agent::app::MAX_INPUT_ROWS + 2,
        "the box stops growing and scrolls instead"
    );
}

// ------------------------------------------------------------ the history

#[test]
fn up_and_down_recall_sent_prompts() {
    let (mut app, mut rx) = app_with(vec![
        vec![MockStreamEvent::FinalResponse(mock_final(Usage::new()))],
        vec![MockStreamEvent::FinalResponse(mock_final(Usage::new()))],
    ]);

    run_prompt(&mut app, &mut rx, "first prompt");
    run_prompt(&mut app, &mut rx, "second prompt");
    assert_eq!(app.history(), ["first prompt", "second prompt"]);

    // A draft typed before browsing is restored by Down past the newest.
    app.set_input("half-typed");
    handle_input(&mut app, key(KeyCode::Up));
    assert_eq!(app.input(), "second prompt");
    handle_input(&mut app, key(KeyCode::Up));
    assert_eq!(app.input(), "first prompt");
    handle_input(&mut app, key(KeyCode::Up));
    assert_eq!(app.input(), "first prompt", "Up stops at the oldest");

    handle_input(&mut app, key(KeyCode::Down));
    assert_eq!(app.input(), "second prompt");
    handle_input(&mut app, key(KeyCode::Down));
    assert_eq!(app.input(), "half-typed", "the draft comes back");
}

#[test]
fn a_repeated_prompt_is_not_stored_twice() {
    let (mut app, mut rx) = app_with(vec![
        vec![MockStreamEvent::FinalResponse(mock_final(Usage::new()))],
        vec![MockStreamEvent::FinalResponse(mock_final(Usage::new()))],
    ]);

    run_prompt(&mut app, &mut rx, "same");
    run_prompt(&mut app, &mut rx, "same");
    assert_eq!(app.history(), ["same"]);
}

// ------------------------------------------------------------- the view

#[test]
fn scroll_keys_move_the_view_and_leave_follow_mode() {
    let mut app = new_app();
    // The renderer reports the transcript's height; without a frame there is
    // nothing to scroll, so the test stands in for one.
    app.set_scroll_max(100);
    app.scroll = 20;
    app.auto_scroll = true;

    // Up/Down are the prompt history now, so the transcript moves with the
    // page keys and Ctrl+Up/Ctrl+Down.
    handle_input(&mut app, key_with(KeyCode::Up, KeyModifiers::CONTROL));
    assert_eq!(app.scroll, 19);
    assert!(!app.auto_scroll, "a manual move leaves follow mode");

    handle_input(&mut app, key(KeyCode::PageDown));
    assert_eq!(app.scroll, 29);

    handle_input(&mut app, key(KeyCode::PageUp));
    assert_eq!(app.scroll, 19);

    handle_input(&mut app, key_with(KeyCode::Home, KeyModifiers::CONTROL));
    assert_eq!(app.scroll, 0, "Ctrl+Home clamps at the top");

    handle_input(&mut app, key_with(KeyCode::End, KeyModifiers::CONTROL));
    assert!(app.auto_scroll, "Ctrl+End returns to follow mode");
}

#[test]
fn scrolling_past_the_end_parks_on_the_last_row() {
    // The offset is a `u16` and the transcript's height is only known after
    // layout, so an unclamped `scroll_by` would run the offset up to the
    // ceiling and the next PageUp would appear to do nothing.
    let mut app = new_app();
    app.set_scroll_max(30);

    for _ in 0..20 {
        handle_input(&mut app, key(KeyCode::PageDown));
    }
    assert_eq!(app.scroll, 30, "the view parks on the last row");

    handle_input(&mut app, key(KeyCode::PageUp));
    assert_eq!(app.scroll, 20, "and the next PageUp moves it");
}

#[test]
fn a_transcript_that_shrinks_under_the_view_does_not_swallow_the_next_press() {
    let mut app = new_app();
    app.set_scroll_max(500);
    app.scroll = 400;

    // `/clear` or a folded group can leave the stored offset above the new
    // end; the next press has to move from the end, not from the stale value.
    app.set_scroll_max(10);
    handle_input(&mut app, key(KeyCode::PageUp));
    assert_eq!(app.scroll, 0);
}

// ------------------------------------------------------------- the help

#[test]
fn the_help_text_documents_the_keys_the_loop_handles() {
    // The help text is the only place a user sees the key map, and it went
    // stale once already: it still said Up/Down scrolled the transcript and
    // that Esc quit, after both had changed.
    let mut app = new_app();
    app.send_message("/help".to_string());
    let help = app
        .messages
        .last()
        .expect("the help text is pushed as a message")
        .content
        .clone();

    for expected in [
        "`Up`/`Down` recall previous prompts",
        "`PageUp`/`PageDown` scroll the transcript",
        "`Ctrl+Home`/`Ctrl+End`",
        "`Ctrl+A`/`Ctrl+E`",
        "`Ctrl+W`",
        "`Ctrl+T`",
        "`Ctrl+G`",
        "stop the run in flight",
        "LLM_PROVIDER",
        "OLLAMA_BASE_URL",
    ] {
        assert!(
            help.contains(expected),
            "the help should mention {expected}"
        );
    }

    // And it must not still describe the map it replaced.
    assert!(
        !help.contains("`Up`/`Down` scroll the transcript"),
        "the help still describes the old key map"
    );
    assert!(
        !help.contains("`Esc` or `Ctrl+C` quit"),
        "the help still says Esc quits, which is now the second press"
    );
}

// ------------------------------------------------------- the letter traps
#[test]
fn a_bare_t_types_the_letter_and_does_not_toggle_a_tool_block() {
    // `t` was bound to the tool-block fold with no modifier, and the arm ran
    // before the input box, so the letter could not be typed at all.
    let mut app = new_app();

    type_text(&mut app, "test");
    assert_eq!(
        app.input(),
        "test",
        "every letter must reach the input buffer"
    );
}

#[test]
fn ctrl_t_toggles_the_newest_tool_block_without_typing() {
    let mut app = new_app();
    app.set_input("keep");

    let result = handle_input(&mut app, ctrl('t'));

    assert!(matches!(result, InputResult::Continue));
    assert_eq!(
        app.input(),
        "keep",
        "a shortcut must not type into the buffer"
    );
}

#[test]
fn ctrl_g_folds_the_newest_tool_group_without_typing() {
    let mut app = new_app();
    app.set_input("keep");

    let result = handle_input(&mut app, ctrl('g'));

    assert!(matches!(result, InputResult::Continue));
    assert_eq!(
        app.input(),
        "keep",
        "a shortcut must not type into the buffer"
    );
}

#[test]
fn a_bare_g_types_the_letter_and_does_not_fold_a_group() {
    // Same trap as the bare `t`: an arm in `handle_input` runs before the
    // input box, so binding the letter would make it untypeable.
    let mut app = new_app();

    type_text(&mut app, "grep");
    assert_eq!(
        app.input(),
        "grep",
        "every letter must reach the input buffer"
    );
}

// ------------------------------------------------------- interrupting a run

#[test]
fn escape_and_ctrl_c_quit_when_nothing_is_running() {
    let mut app = new_app();
    assert!(matches!(
        handle_input(&mut app, key(KeyCode::Esc)),
        InputResult::Quit
    ));

    let mut app = new_app();
    assert!(matches!(
        handle_input(&mut app, ctrl('c')),
        InputResult::Quit
    ));
}

#[test]
fn escape_cancels_the_run_in_flight_and_a_second_press_quits() {
    let (mut app, _rx) = app_with(vec![vec![MockStreamEvent::FinalResponse(mock_final(
        Usage::new(),
    ))]]);

    let rt = runtime();
    rt.block_on(async {
        app.send_message("go".to_string());
        assert!(app.is_streaming(), "the run is in flight");

        // The first press asks the run to stop and keeps the app alive: the
        // user needs the keyboard back, not an exit.
        assert!(matches!(
            handle_input(&mut app, key(KeyCode::Esc)),
            InputResult::Continue
        ));
        assert!(app.is_cancelling());

        // The second press quits, so a run that will not stop cannot trap
        // the user in the app.
        assert!(matches!(
            handle_input(&mut app, key(KeyCode::Esc)),
            InputResult::Quit
        ));
    });
}

#[test]
fn a_prompt_typed_during_a_run_is_queued_and_sent_after_it() {
    let (mut app, mut rx) = app_with(vec![
        vec![MockStreamEvent::FinalResponse(mock_final(Usage::new()))],
        vec![MockStreamEvent::FinalResponse(mock_final(Usage::new()))],
    ]);

    let rt = runtime();
    rt.block_on(async {
        app.send_message("first".to_string());
        // A second run would interleave into the same transcript and fight
        // over the assistant message, so the prompt waits its turn rather
        // than being refused, which would lose what the user typed.
        app.send_message("second".to_string());
        assert_eq!(app.queued(), 1);

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    let finished = matches!(event, Event::Finished(_) | Event::Cancelled);
                    app.handle_event(event);
                    if finished && app.queued() == 0 {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => panic!("the queued run did not finish within 20s"),
            }
        }
    });

    assert_eq!(app.queued(), 0, "the queue drained");
    let prompts: Vec<&str> = app
        .messages
        .iter()
        .filter(|m| m.role == rust_agent::app::Role::User)
        .map(|m| m.content.as_str())
        .collect();
    assert_eq!(
        prompts,
        ["first", "second"],
        "both prompts were sent, in order"
    );
}
