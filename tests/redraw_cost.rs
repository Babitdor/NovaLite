//! What a redraw of a long transcript costs.
//!
//! A transcript is redrawn on every tick, and the tick is ten a second while a
//! run is in flight, so the cost of one frame is multiplied by ten for as long
//! as the run lasts. The thing that grows is the conversation, so the cost of
//! a frame has to stop growing with it.
//!
//! The guard here is a count, not a clock. A test cannot time a frame
//! reliably: a debug build is twenty times slower than a release one, and a
//! loaded machine is slower again, so any threshold that passes here would
//! either be meaningless or flake. What can be asserted exactly is the work:
//! a redraw of a conversation that has not changed must render nothing.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use rust_agent::app::{App, ChatMessage, Role};

mod common;

fn app_with_messages(n: usize) -> App {
    let model =
        rig::test_utils::MockCompletionModel::from_turns([rig::test_utils::MockTurn::text("done")]);
    let rt = common::session_at_crate_root();
    let agent = rig::agent::AgentBuilder::new(model.clone()).build();
    let summariser = rig::agent::AgentBuilder::new(model).build();
    let (mut app, _rx) = App::new(
        rust_agent::agent::Agents {
            main: agent,
            summariser,
        },
        rt,
    );

    // Answers of the shape a real one has: prose, a heading, a fenced block,
    // a list. The parser's cost depends on the structure, not just the length.
    let answer = "Here is what I found.\n\n## The cause\n\nThe check ran before the \
                  change, so it proves nothing about it.\n\n```rust\nfn main() {\n    \
                  println!(\"hello\");\n}\n```\n\n- one\n- two\n- three\n\nThat is all.\n";
    for i in 0..n {
        app.messages
            .push(ChatMessage::new(Role::User, format!("question {i}")));
        app.messages.push(ChatMessage::new(Role::Assistant, answer));
    }
    app
}

fn draw_once(app: &App, width: u16, height: u16) {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|f| rust_agent::render::draw(f, app)).unwrap();
}

fn renders(app: &App) -> u64 {
    app.bodies_rendered()
}

#[test]
fn a_redraw_of_an_unchanged_conversation_renders_nothing() {
    let app = app_with_messages(50);

    draw_once(&app, 120, 40);
    let after_first = renders(&app);
    assert!(
        after_first > 0,
        "the first frame has to render the bodies it paints"
    );

    for _ in 0..5 {
        draw_once(&app, 120, 40);
    }
    assert_eq!(
        renders(&app),
        after_first,
        "five redraws of an unchanged conversation re-rendered a body"
    );
}

#[test]
fn a_body_that_changed_is_rendered_again() {
    // The message being streamed into changes on every frame, and it is the
    // one on screen, so a cache that never invalidated would paint the first
    // fragment of an answer forever.
    let mut app = app_with_messages(3);
    draw_once(&app, 120, 40);
    let before = renders(&app);

    let last = app.messages.len() - 1;
    app.messages[last]
        .content
        .push_str("\n\nand one more thing.\n");
    draw_once(&app, 120, 40);

    assert_eq!(
        renders(&app),
        before + 1,
        "exactly the changed body should have been rendered again"
    );
}

#[test]
fn a_body_wrapped_to_a_new_width_is_rendered_again() {
    // A resize re-wraps every body, so every body is stale. The cache is
    // keyed by width as well as by text for this reason: a body wrapped to
    // the old width would be painted at the new one and the indent would
    // drift.
    //
    // The widths are far enough apart that the transcript column really does
    // change: it is the whole column now, so it follows the terminal's width.
    let app = app_with_messages(3);
    draw_once(&app, 120, 40);
    let before = renders(&app);

    draw_once(&app, 200, 40);

    assert_eq!(
        renders(&app),
        before + 6,
        "all six bodies should have been wrapped again"
    );
}

#[test]
fn the_cache_does_not_outlive_the_conversation_it_describes() {
    // A cleared session must not keep the lines of the conversation it had.
    // The cache is a view of the conversation, so it is pruned against it.
    let mut app = app_with_messages(20);
    draw_once(&app, 120, 40);
    assert_eq!(app.cached_bodies(), 40);

    app.messages.clear();
    draw_once(&app, 120, 40);

    assert_eq!(
        app.cached_bodies(),
        0,
        "the cache kept entries for messages that no longer exist"
    );
}

/// The measurement the cap in `render.rs` is set from.
///
/// Ignored because it asserts nothing: it prints, and a printed number is not
/// a check. Run it by hand when the renderer changes, in release, and update
/// the doc comment on `MAX_TRANSCRIPT_LINES` if the numbers have moved.
///
/// ```text
/// cargo test --release --test redraw_cost -- --ignored --nocapture
/// ```
#[test]
#[ignore = "a measurement, not a check"]
fn a_redraw_of_a_long_transcript_is_measured() {
    for n in [1, 10, 25, 50, 200] {
        let app = app_with_messages(n);
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal
            .draw(|f| rust_agent::render::draw(f, &app))
            .unwrap();
        let start = std::time::Instant::now();
        for _ in 0..20 {
            terminal
                .draw(|f| rust_agent::render::draw(f, &app))
                .unwrap();
        }
        println!(
            "{n:>4} answers: {:.3} ms/frame",
            start.elapsed().as_secs_f64() * 1000.0 / 20.0
        );
    }
}

/// The frame that is actually expensive: the one drawing an answer that is
/// still being written.
///
/// Every other body is cached, so a streaming frame re-parses exactly one
/// body, and that body grows with every chunk. This measures the growth, which
/// is what decides whether caching the parsed prefix is worth its complexity.
///
/// Measured on this machine, release, 120x40:
///
/// ```text
///   250 tokens:   17 frames, 4.987 ms/frame
///   500 tokens:   33 frames, 4.340 ms/frame
///  1000 tokens:   65 frames, 6.778 ms/frame
///  2000 tokens:  130 frames, 5.316 ms/frame
///  4000 tokens:  259 frames, 8.790 ms/frame
/// ```
///
/// The answer grows sixteenfold and the frame grows by less than twofold, and
/// the spread between adjacent sizes is wider than the trend: at these sizes
/// the re-parse is not measurable above the noise floor. A frame is drawn
/// twelve times a second, so 9 ms is a tenth of one core against an 80 ms
/// tick. That is why there is no prefix cache: incremental markdown is hard
/// (a half-written fence, list or table changes how the text before it is
/// read) and it would buy nothing that can be measured.
///
/// ```text
/// cargo test --release --test redraw_cost -- --ignored --nocapture
/// ```
#[test]
#[ignore = "a measurement, not a check"]
fn a_streaming_frame_is_measured() {
    // A chunk of the shape a model emits: a few words at a time.
    let chunk = "the check ran before the change so it proves nothing about it ";
    for tokens in [250, 500, 1_000, 2_000, 4_000] {
        let mut app = app_with_messages(20);
        let target = tokens * 4; // ~4 characters per token
        let mut written = 0;
        let mut frames = 0u32;
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        let start = std::time::Instant::now();
        while written < target {
            let last = app.messages.len() - 1;
            app.messages[last].content.push_str(chunk);
            written += chunk.len();
            terminal
                .draw(|f| rust_agent::render::draw(f, &app))
                .unwrap();
            frames += 1;
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{tokens:>5} tokens: {frames:>4} frames, {:.3} ms/frame, {:.1} ms total",
            elapsed / f64::from(frames),
            elapsed
        );
    }
}
