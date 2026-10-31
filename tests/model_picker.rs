//! The `/model` command: listing Ollama's models, choosing one, and the
//! overlay that shows the list.
//!
//! The list itself comes from `ollama list`, which is not guaranteed to be
//! installed, so the parsing is tested against captured output and the picker
//! is driven with a list injected directly. The one thing that must not be
//! faked is the wiring: `/model` has to open the picker, and Enter has to
//! rebuild the agent, so those go through the real command and key paths.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::Terminal;
use rig::test_utils::MockCompletionModel;

use rust_agent::agent;
use rust_agent::app::{App, ModelPicker};
use rust_agent::render;
use rust_agent::tui::{handle_input, InputResult};

mod common;

fn app() -> App {
    let model =
        MockCompletionModel::from_stream_turns(Vec::<Vec<rig::test_utils::MockStreamEvent>>::new());
    let rt = common::session();
    App::new(common::agents(model, &rt), rt).0
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// Type `text` and press Enter, exactly as the event loop does: the key
/// handler returns the text and the loop hands it to `send_message`. Going
/// through both is what proves `/model` is wired to the picker.
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

/// The column a byte offset in `row` falls on.
///
/// `str::rfind` reports a byte offset, and a painted frame is full of
/// multi-byte box-drawing glyphs, so two offsets are only comparable once
/// they are converted to columns.
fn column_of(row: &str, byte: usize) -> usize {
    row[..byte].chars().count()
}

/// A picker holding `models`, as if `ollama list` had just returned.
fn with_models(app: &mut App, models: &[&str]) {
    app.picker = Some(ModelPicker {
        models: models.iter().map(|s| s.to_string()).collect(),
        selected: 0,
        loading: false,
        error: None,
    });
}

// ------------------------------------------------------------- parsing

#[test]
fn ollama_list_output_parses_into_model_names() {
    // Captured from `ollama list` on this machine, header included.
    let stdout = "\
NAME                                     ID              SIZE      MODIFIED     \n\
ingloryon/k2-horizon:0.9b                677f6e7a4eed    1.1 GB    9 hours ago     \n\
tev1:0.8b                                9bbd32fdb335    811 MB    4 days ago      \n\
deepseek-v4.1-flash:cloud                e04da138d31e    -         3 weeks ago     \n";

    let models = agent::parse_ollama_list(stdout);

    assert_eq!(
        models,
        vec![
            "ingloryon/k2-horizon:0.9b",
            "tev1:0.8b",
            "deepseek-v4.1-flash:cloud",
        ],
        "the header must be dropped and the first column taken"
    );
}

#[test]
fn an_empty_ollama_list_parses_to_no_models() {
    // A daemon with nothing pulled prints the header and nothing else.
    let stdout =
        "NAME                                     ID              SIZE      MODIFIED     \n";
    assert!(agent::parse_ollama_list(stdout).is_empty());
}

#[test]
fn a_model_name_with_a_slash_survives_parsing() {
    // Namespaced models are common; the name is the whole first column, not
    // the part before the slash.
    let stdout = "NAME  ID  SIZE  MODIFIED\nlibrary/llama3.2:3b  abc  2 GB  now\n";
    assert_eq!(
        agent::parse_ollama_list(stdout),
        vec!["library/llama3.2:3b"]
    );
}

// -------------------------------------------------------------- command

#[test]
fn the_model_command_opens_the_picker() {
    let mut app = app();
    assert!(app.picker.is_none(), "the picker starts closed");

    // The fetch runs on a blocking task, so this needs the runtime the binary
    // always has.
    runtime().block_on(async { submit(&mut app, "/model") });

    let picker = app.picker.as_ref().expect("/model should open the picker");
    assert!(
        picker.loading,
        "the picker opens in a loading state while `ollama list` runs"
    );
}

#[test]
fn the_picker_is_modal_so_typing_does_not_reach_the_prompt() {
    let mut app = app();
    with_models(&mut app, &["a:1b", "b:2b"]);
    app.set_input("");

    for c in "hello".chars() {
        handle_input(&mut app, key(KeyCode::Char(c)));
    }

    assert!(
        app.input().is_empty(),
        "keys belong to the picker while it is open, got input {:?}",
        app.input()
    );
}

#[test]
fn arrows_move_the_highlight_and_clamp_at_the_ends() {
    let mut app = app();
    with_models(&mut app, &["a:1b", "b:2b", "c:3b"]);

    handle_input(&mut app, key(KeyCode::Down));
    assert_eq!(app.picker.as_ref().unwrap().selected, 1);

    handle_input(&mut app, key(KeyCode::Up));
    handle_input(&mut app, key(KeyCode::Up));
    assert_eq!(
        app.picker.as_ref().unwrap().selected,
        0,
        "the highlight must not run off the top"
    );

    for _ in 0..10 {
        handle_input(&mut app, key(KeyCode::Down));
    }
    assert_eq!(
        app.picker.as_ref().unwrap().selected,
        2,
        "the highlight must not run off the bottom"
    );
}

#[test]
fn escape_closes_the_picker_without_switching() {
    let mut app = app();
    with_models(&mut app, &["a:1b", "b:2b"]);
    let before = app.model_name.clone();

    handle_input(&mut app, key(KeyCode::Esc));

    assert!(app.picker.is_none(), "Esc should close the picker");
    assert_eq!(app.model_name, before, "Esc must not switch the model");
}

#[test]
fn enter_switches_to_the_highlighted_model() {
    let mut app = app();
    with_models(&mut app, &["a:1b", "b:2b"]);
    handle_input(&mut app, key(KeyCode::Down));

    handle_input(&mut app, key(KeyCode::Enter));

    assert!(app.picker.is_none(), "choosing closes the picker");
    assert_eq!(
        app.model_name, "b:2b",
        "the highlighted model should become the active one"
    );
}

#[test]
fn switching_reports_the_new_model_in_the_transcript() {
    let mut app = app();
    with_models(&mut app, &["a:1b", "b:2b"]);
    handle_input(&mut app, key(KeyCode::Down));

    handle_input(&mut app, key(KeyCode::Enter));

    let last = app.messages.last().expect("a message");
    assert!(
        last.content.contains("b:2b"),
        "the switch should be visible in the conversation, got {:?}",
        last.content
    );
}

#[test]
fn enter_on_an_empty_list_does_not_switch_or_panic() {
    // `ollama list` can legitimately return nothing, and Enter is still a
    // key the user can press.
    let mut app = app();
    with_models(&mut app, &[]);
    let before = app.model_name.clone();

    handle_input(&mut app, key(KeyCode::Enter));

    assert_eq!(app.model_name, before, "there is nothing to switch to");
}

// -------------------------------------------------------------- overlay

#[test]
fn the_picker_paints_the_models_and_marks_the_current_one() {
    let mut app = app();
    let current = app.model_name.clone();
    with_models(&mut app, &[&current, "other:1b"]);

    let screen = paint(&app, 100, 30);

    assert!(
        screen.contains("select a model"),
        "the overlay should be titled:\n{screen}"
    );
    assert!(
        screen.contains("other:1b"),
        "the list should be painted:\n{screen}"
    );
    assert!(
        screen.contains("(current)"),
        "the active model should be marked:\n{screen}"
    );
    assert!(
        screen.contains("Enter switch"),
        "the key hint should fit inside the border:\n{screen}"
    );
}

#[test]
fn the_picker_reports_a_failed_list_rather_than_an_empty_one() {
    // "no models" and "could not ask" are different problems, and only one of
    // them is fixed by pulling a model.
    let mut app = app();
    app.picker = Some(ModelPicker {
        models: Vec::new(),
        selected: 0,
        loading: false,
        error: Some("`ollama list` failed: command not found".to_string()),
    });

    let screen = paint(&app, 100, 30);

    assert!(
        screen.contains("could not list models"),
        "a failed fetch should say so:\n{screen}"
    );
    assert!(
        !screen.contains("no models found"),
        "a failed fetch is not an empty list:\n{screen}"
    );
}

#[test]
fn the_picker_stays_inside_the_conversation_panel() {
    // The overlay is centred on the transcript column and clamped to it. The
    // transcript is the whole width, so what the clamp protects now is the
    // input box and the footer underneath: a popup that grew to the column
    // edge would paint over the prompt the user is about to answer.
    let mut app = app();
    with_models(
        &mut app,
        &["a-very-long-model-name-that-is-quite-wide:latest", "b:2b"],
    );

    let rows: Vec<String> = paint(&app, 100, 30).lines().map(str::to_string).collect();

    let top = rows
        .iter()
        .position(|r| r.contains("select a model"))
        .expect("the overlay");
    let overlay_left = column_of(
        &rows[top],
        rows[top].find('╭').expect("the overlay's left corner"),
    );
    let overlay_right = column_of(
        &rows[top],
        rows[top].rfind('╮').expect("the overlay's right corner"),
    );

    assert!(
        overlay_left > 0 && overlay_right < 99,
        "the overlay should sit inside the column (left {overlay_left}, right \
         {overlay_right}):\n{}",
        rows.join("\n")
    );
    // The input box is painted on the overlay's frame, so the picker covering
    // the whole screen would take it with it.
    assert!(
        rows.iter().any(|r| r.contains("Ask anything")),
        "the input box stays on screen under the overlay:\n{}",
        rows.join("\n")
    );
}

#[test]
fn a_long_list_scrolls_to_keep_the_highlight_visible() {
    // `ollama list` on a machine with many models returns more than the
    // overlay can show, so the highlight has to stay in view as it moves.
    let mut app = app();
    let models: Vec<String> = (0..40).map(|i| format!("model-{i:02}:latest")).collect();
    app.picker = Some(ModelPicker {
        models,
        selected: 0,
        loading: false,
        error: None,
    });

    for _ in 0..39 {
        handle_input(&mut app, key(KeyCode::Down));
    }
    assert_eq!(app.picker.as_ref().unwrap().selected, 39);

    let screen = paint(&app, 100, 30);
    assert!(
        screen.contains("model-39:latest"),
        "the highlighted model must be scrolled into view:\n{screen}"
    );
    assert!(
        !screen.contains("model-00:latest"),
        "the top of the list should have scrolled away:\n{screen}"
    );
}

#[test]
fn the_picker_does_not_paint_when_it_is_closed() {
    let app = app();
    let screen = paint(&app, 100, 30);
    assert!(
        !screen.contains("select a model"),
        "a closed picker must not paint:\n{screen}"
    );
}

#[test]
fn a_short_picker_list_still_fits_its_border() {
    // The popup is sized from the list, so a one-model list must not produce a
    // box too small for its own hint line.
    let mut app = app();
    with_models(&mut app, &["only:1b"]);

    let screen = paint(&app, 100, 30);
    assert!(
        screen.contains("Enter switch"),
        "the hint must fit even for a one-model list:\n{screen}"
    );
    assert!(
        screen.contains("only:1b"),
        "the single model should be painted:\n{screen}"
    );
}

#[test]
fn the_help_text_documents_the_model_command() {
    let mut app = app();
    submit(&mut app, "/help");

    let help = app
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        help.contains("/model"),
        "the help text should list /model:\n{help}"
    );
}

#[test]
fn input_result_is_continue_while_the_picker_is_open() {
    // The picker must not be able to quit the app or send a prompt by
    // accident: every key it handles is a Continue.
    let mut app = app();
    with_models(&mut app, &["a:1b"]);

    for code in [
        KeyCode::Char('q'),
        KeyCode::Enter,
        KeyCode::Esc,
        KeyCode::Up,
        KeyCode::Down,
    ] {
        assert!(
            matches!(handle_input(&mut app, key(code)), InputResult::Continue),
            "{code:?} should not quit or send while the picker is open"
        );
        if app.picker.is_none() {
            with_models(&mut app, &["a:1b"]);
        }
    }
}
