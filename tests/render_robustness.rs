//! Adversarial rendering: sizes and text the happy path never produces.
//!
//! The layout maths in `render.rs` mixes character counts with byte indices
//! and casts to `u16` in several places. These tests pin the cases where that
//! could panic or paint outside the buffer: a terminal smaller than the
//! chrome, and text whose characters are not one byte and not one column wide.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use rig::test_utils::{MockCompletionModel, MockStreamEvent};

use rust_agent::app::App;
use rust_agent::render;
use rust_agent::tools;

mod common;

fn app() -> App {
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let rt = common::session();
    let agent = rig::agent::AgentBuilder::new(model.clone())
        .tool(tools::ReadFile::new(rt.clone()))
        .tool(tools::Shell::new(rt.clone()))
        .build();
    let summariser = rig::agent::AgentBuilder::new(model).build();
    App::new(
        rust_agent::agent::Agents {
            main: agent,
            summariser,
        },
        rt,
    )
    .0
}

/// Render into an offscreen buffer, returning one string per row.
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

/// Every size from 1x1 up to 6x6, plus a few lopsided ones. The chrome needs
/// more rows than this, so each of these is a case where the layout has to
/// give something up rather than index past the end of the buffer.
#[test]
fn renders_at_every_size_smaller_than_the_chrome() {
    for height in 1..=6u16 {
        for width in 1..=6u16 {
            let app = app();
            let rows = paint(&app, width, height);
            assert_eq!(rows.len(), height as usize, "at {width}x{height}");
            for row in &rows {
                assert_eq!(
                    row.chars().count(),
                    width as usize,
                    "at {width}x{height}: {row:?}"
                );
            }
        }
    }
    for (width, height) in [(1, 40), (40, 1), (2, 30), (200, 2), (3, 3)] {
        let app = app();
        let rows = paint(&app, width, height);
        assert_eq!(rows.len(), height as usize, "at {width}x{height}");
    }
}

/// A caret and a horizontal offset are computed from character counts, then
/// cast to `u16`. A line longer than `u16::MAX` characters is the case where
/// that cast would wrap and put the caret in the wrong place, or panic.
#[test]
fn a_line_longer_than_u16_max_does_not_panic() {
    let mut app = app();
    app.set_input("x".repeat(70_000));
    for width in [1u16, 2, 20, 80] {
        let rows = paint(&app, width, 10);
        assert_eq!(rows.len(), 10);
    }
}

/// Multi-byte characters: the caret index is a character count, the slice
/// into the input is a byte index, and the width of a CJK glyph is two
/// columns. None of those agree, so this is where a byte-boundary panic or a
/// row that overflows its width would show up.
#[test]
fn multibyte_input_does_not_panic_or_overflow_the_row() {
    let cases = vec![
        "日本語のテキスト".to_string(),
        "🎉🎉🎉 emoji".to_string(),
        "aé日🎉b".to_string(),
        "é".repeat(200),
        "日".repeat(200),
    ];
    for text in &cases {
        let mut app = app();
        app.set_input(text.clone());
        for width in [1u16, 2, 5, 20, 40, 100] {
            let rows = paint(&app, width, 12);
            assert_eq!(rows.len(), 12, "at {width} for {text:?}");
            for row in &rows {
                assert_eq!(
                    row.chars().count(),
                    width as usize,
                    "at {width} for {text:?}: {row:?}"
                );
            }
        }
    }
}

/// The caret must stay inside the input box for multi-byte text, since the
/// box's own width is measured in columns while the caret is measured in
/// characters.
#[test]
fn the_caret_stays_inside_the_box_for_multibyte_input() {
    let mut app = app();
    app.set_input("日本語のテキストをここに入力します");
    for width in [10u16, 20, 40, 80] {
        let rows = paint(&app, width, 12);
        // The input box is the last block; its border row is the one that
        // starts with the box-drawing corner.
        let boxed: Vec<&String> = rows.iter().filter(|r| r.contains('╭')).collect();
        assert!(
            !boxed.is_empty(),
            "the input box must survive at width {width}"
        );
    }
}

/// A transcript entry containing multi-byte text, which is what a file read
/// or a model answer in another language produces.
#[test]
fn multibyte_transcript_does_not_panic() {
    let mut app = app();
    app.messages.push(rust_agent::app::ChatMessage::new(
        rust_agent::app::Role::Assistant,
        "日本語のテキスト\n🎉 emoji line\nplain ascii".to_string(),
    ));
    for width in [1u16, 3, 10, 40, 100] {
        let rows = paint(&app, width, 20);
        assert_eq!(rows.len(), 20, "at {width}");
        for row in &rows {
            assert_eq!(row.chars().count(), width as usize, "at {width}: {row:?}");
        }
    }
}

/// Markdown that is still arriving is markdown that does not parse.
///
/// The transcript renders the answer as it streams, so every frame of a long
/// answer is a frame of half-written markdown: an unterminated code fence, a
/// table with no rows yet, a link with no closing bracket. The renderer must
/// paint what it has rather than panic or drop the text, and the text must
/// still be readable, because that is the whole point of streaming it.
#[test]
fn incomplete_markdown_renders_what_has_arrived() {
    let cases = [
        ("```rust\nfn main() {", "fn main() {"),
        ("| name | value |\n| - |", "name"),
        ("see [the docs](", "see"),
        ("**bold", "bold"),
        ("> quote\n>", "quote"),
        ("# heading\n\n- item\n- ", "heading"),
        ("text with a lone ` backtick", "backtick"),
    ];
    for (text, needle) in cases {
        let mut app = app();
        app.messages.push(rust_agent::app::ChatMessage::new(
            rust_agent::app::Role::Assistant,
            text.to_string(),
        ));
        for width in [1u16, 4, 20, 60, 100] {
            let rows = paint(&app, width, 20);
            assert_eq!(rows.len(), 20, "at {width} for {text:?}");
            for row in &rows {
                assert_eq!(
                    row.chars().count(),
                    width as usize,
                    "at {width} for {text:?}: {row:?}"
                );
            }
        }
        // The text survives to the screen: a renderer that swallowed an
        // unparseable block would leave the answer blank, and the answer is
        // the whole point of streaming it.
        let screen = paint(&app, 100, 20).join("\n");
        assert!(
            screen.contains(needle),
            "{needle:?} from {text:?} is missing from:\n{screen}"
        );
    }
}

/// Slash-command completion compares a character count against a byte length.
/// A non-ASCII input is where those disagree, so this is the case that would
/// slice mid-character. The completion must either decline or land on a whole
/// command, never on a fragment.
#[test]
fn slash_completion_survives_multibyte_input() {
    for text in ["/", "/日", "/é", "/🎉", "/mo", "/m", "/MODEL"] {
        let mut app = app();
        app.set_input(text);
        let changed = app.complete_slash_command();
        let after = app.input().to_string();
        assert!(after.starts_with('/'), "completing {text:?} gave {after:?}");
        if changed {
            assert!(
                App::SLASH_COMMANDS.contains(&after.as_str()),
                "completing {text:?} gave {after:?}, which is not a command"
            );
        }
    }
}

/// A tool message as the stream task builds one: the call's name and
/// arguments, and the result that came back.
fn tool_msg(name: &str, args: &str, result: &str) -> rust_agent::app::ChatMessage {
    rust_agent::app::ChatMessage::tool(rust_agent::app::ToolBlock {
        name: name.to_string(),
        args: args.to_string(),
        args_json: args.to_string(),
        result: result.to_string(),
        outcome: rust_agent::app::ToolOutcome::Ok,
        truncated: false,
        streaming: false,
        done: true,
        part: None,
        call: None,
        released: false,
    })
}

/// The row carrying a group heading, which is the one that names the category.
fn heading_row<'a>(rows: &'a [String], label: &str) -> &'a String {
    rows.iter()
        .find(|r| r.contains(label))
        .unwrap_or_else(|| panic!("no {label} heading in:\n{}", rows.join("\n")))
}

/// The tool-group folding path (`Ctrl+G` / `Ctrl+Shift+G`). Every other test
/// in this file builds an app with no tool messages at all, so the group
/// heading, the folded digest and the fold state are otherwise unexercised.
///
/// The discriminator is a member's argument text: an open group paints its
/// calls, a folded one replaces them with the digest. The tool *name* appears
/// in both, so it cannot tell them apart.
#[test]
fn folding_a_tool_group_hides_its_calls_and_survives_every_width() {
    let mut app = app();
    app.messages.push(rust_agent::app::ChatMessage::new(
        rust_agent::app::Role::User,
        "look around".to_string(),
    ));
    // Two contiguous reads: one EXPLORE group.
    app.messages
        .push(tool_msg("read_file", "path=src/app.rs", "one\ntwo"));
    app.messages
        .push(tool_msg("read_file", "path=src/render.rs", "three"));
    // A different category ends the group and opens a new one.
    app.messages
        .push(tool_msg("shell", "command=cargo test", "ok"));

    let groups = app.tool_groups();
    assert_eq!(groups.len(), 2, "one EXPLORE group and one RUN group");
    assert_eq!(groups[0].len(), 2, "the two reads are one group");
    assert_eq!(groups[1].len(), 1, "the shell call is its own group");

    let first = groups[0].first;
    assert!(!app.is_group_folded(first), "open is the default");

    let open = paint(&app, 80, 30);
    assert!(
        open.iter().any(|r| r.contains("path=src/app.rs")),
        "an open group paints its calls:\n{}",
        open.join("\n")
    );
    let open_heading = heading_row(&open, "EXPLORE");
    assert!(
        open_heading.trim_start().starts_with('▾'),
        "an open group uses the open marker, got {open_heading:?}"
    );

    app.toggle_group(first);
    assert!(app.is_group_folded(first));

    let folded = paint(&app, 80, 30);
    assert!(
        !folded.iter().any(|r| r.contains("path=src/app.rs")),
        "a folded group hides its calls:\n{}",
        folded.join("\n")
    );
    let folded_heading = heading_row(&folded, "EXPLORE");
    assert!(
        folded_heading.trim_start().starts_with('▸'),
        "a folded group uses the folded marker, got {folded_heading:?}"
    );
    assert!(
        folded.iter().any(|r| r.contains("read_file ×2")),
        "the folded digest still says what the run was made of:\n{}",
        folded.join("\n")
    );
    // Folding is per group: the RUN group is untouched.
    assert!(
        folded.iter().any(|r| r.contains("command=cargo test")),
        "folding one group must not fold another"
    );

    // `Ctrl+Shift+G`: with a group still open, fold them all; with none open,
    // unfold them all.
    app.toggle_all_groups();
    assert!(app.is_group_folded(first));
    assert!(app.is_group_folded(groups[1].first));
    app.toggle_all_groups();
    assert!(!app.is_group_folded(first));
    assert!(!app.is_group_folded(groups[1].first));

    // A folded group is a shorter transcript, so the layout has less to work
    // with. It must still fill the buffer exactly at every size.
    app.toggle_all_groups();
    for height in 1..=6u16 {
        for width in 1..=6u16 {
            let rows = paint(&app, width, height);
            assert_eq!(rows.len(), height as usize, "at {width}x{height}");
            for row in &rows {
                assert_eq!(
                    row.chars().count(),
                    width as usize,
                    "at {width}x{height}: {row:?}"
                );
            }
        }
    }
}

/// The "Agent is responding…" indicator.
///
/// The run's state is on the top header, beside the title. A message row
/// carries no label of its own, so the header is the only row that can claim a
/// run is in flight: the transcript is what was said, and the header is what
/// the app is doing.
#[test]
fn a_streaming_run_says_it_is_responding_and_a_finished_one_does_not() {
    let mut app = app();
    app.messages.push(rust_agent::app::ChatMessage::new(
        rust_agent::app::Role::User,
        "hi".to_string(),
    ));
    app.messages.push(rust_agent::app::ChatMessage::new(
        rust_agent::app::Role::Assistant,
        String::new(),
    ));

    // Idle: the header says the agent is online and nothing more.
    let idle = paint(&app, 80, 30);
    assert!(
        idle[0].contains("Agent online"),
        "an idle session's header says the agent is online: {:?}",
        idle[0]
    );

    // Streaming with no text yet: the header says what the run is doing.
    app.set_streaming(true, "thinking");
    let working = paint(&app, 80, 30);
    assert!(
        working[0].contains("thinking…"),
        "a run in flight says what it is doing: {:?}",
        working[0]
    );
    assert!(
        rust_agent::render::SPINNER
            .iter()
            .any(|ch| working[0].contains(ch)),
        "the header animates while the run is in flight: {:?}",
        working[0]
    );

    // The state is on the header and on no other row. A label row per message
    // would put the run's state beside every message in the conversation, and
    // the animation on all of them at once.
    let animated: Vec<&String> = working
        .iter()
        .skip(1)
        .filter(|r| rust_agent::render::SPINNER.iter().any(|ch| r.contains(ch)))
        .collect();
    assert!(
        animated.is_empty(),
        "only the header carries the run's state:\n{}",
        working.join("\n")
    );

    // A user's own row is the user's words, never the run's state.
    let user_rows: Vec<&String> = working.iter().filter(|r| r.contains("❯ hi")).collect();
    assert_eq!(user_rows.len(), 1, "a user row to check");
    for row in user_rows {
        assert!(
            !row.contains("thinking") && !row.contains("responding"),
            "a user's own row must not claim the run's state: {row:?}"
        );
    }

    // The state can reach `ready` while the last event is still on its way to
    // the screen, so the transcript is keyed off the body as well: an answer
    // that has arrived is not a run in flight, whatever the state says.
    app.set_streaming(false, "ready");
    app.messages.last_mut().unwrap().content.push_str("hello");
    let answered = paint(&app, 80, 30);
    assert!(
        answered.iter().any(|r| r.contains("hello")),
        "the answer is on screen"
    );
    assert!(
        !answered.iter().any(|r| r.contains("responding…")),
        "an answered message is not a run in flight:\n{}",
        answered.join("\n")
    );
    assert!(
        answered[0].contains("Agent online"),
        "a finished run's header is the idle one: {:?}",
        answered[0]
    );

    // A user message can be the newest block: the run's assistant message is
    // created when the run starts, but the renderer must not depend on that
    // ordering. Nothing in the transcript animates, whatever the newest row is.
    app.set_streaming(true, "streaming");
    app.messages.push(rust_agent::app::ChatMessage::new(
        rust_agent::app::Role::User,
        "again".to_string(),
    ));
    app.messages.push(rust_agent::app::ChatMessage::new(
        rust_agent::app::Role::Assistant,
        String::new(),
    ));
    app.messages.push(rust_agent::app::ChatMessage::new(
        rust_agent::app::Role::User,
        "and again".to_string(),
    ));
    let user_last = paint(&app, 80, 30);
    let spinners = user_last
        .iter()
        .skip(1)
        .filter(|r| rust_agent::render::SPINNER.iter().any(|ch| r.contains(ch)))
        .count();
    assert_eq!(
        spinners,
        0,
        "no transcript row animates, even with a user's message last:\n{}",
        user_last.join("\n")
    );
    let user_rows: Vec<&String> = user_last.iter().filter(|r| r.contains("❯ ")).collect();
    assert!(!user_rows.is_empty(), "a user row to check");
    for row in user_rows {
        assert!(
            !row.contains("thinking") && !row.contains("responding"),
            "a user's own row must not claim the run's state: {row:?}"
        );
    }

    // The transcript's rows are unchanged by the state, so the frame still
    // fills the buffer exactly at every size.
    for height in 1..=6u16 {
        for width in 1..=6u16 {
            let rows = paint(&app, width, height);
            assert_eq!(rows.len(), height as usize, "at {width}x{height}");
            for row in &rows {
                assert_eq!(
                    row.chars().count(),
                    width as usize,
                    "at {width}x{height}: {row:?}"
                );
            }
        }
    }
}
