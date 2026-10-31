//! Geometry and layout guards for the NOVA-Lite banner.
//!
//! The art is a fixed-width block face, so the properties that matter are
//! mechanical: every row of a wordmark must occupy the same number of cells
//! (or the letters shear), and the art must fit the panel it is drawn into
//! (or it wraps and the alignment collapses). Both are asserted here rather
//! than eyeballed, because a hand-edited glyph row is exactly how this kind of
//! art breaks.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use rig::test_utils::MockCompletionModel;

use rust_agent::app::App;
use rust_agent::banner;
use rust_agent::render;

mod common;

/// Cells a string occupies. Every glyph in this art is single-width, so the
/// code-point count is the cell count; the assertion below pins that.
fn cells(s: &str) -> usize {
    s.chars().count()
}

fn app() -> App {
    let model =
        MockCompletionModel::from_stream_turns(Vec::<Vec<rig::test_utils::MockStreamEvent>>::new());
    let rt = common::session();
    App::new(common::agents(model, &rt), rt).0
}

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

#[test]
fn every_wordmark_row_is_the_same_width() {
    for (name, art, expected) in [
        ("NOVA", &banner::NOVA, banner::ART_WIDTH),
        ("LITE", &banner::LITE, banner::LITE_WIDTH),
    ] {
        let widths: Vec<usize> = art.iter().map(|r| cells(r)).collect();
        assert_eq!(
            widths
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            1,
            "{name} rows are ragged, so the letters will shear: {widths:?}"
        );
        assert_eq!(
            widths[0], expected,
            "{name} width disagrees with the declared constant"
        );
    }
}

#[test]
fn the_art_contains_no_double_width_or_combining_glyphs() {
    // A combining mark or a wide glyph would occupy a different number of
    // cells than code points, shifting its whole row. The block and box
    // drawing ranges used here are all single-width.
    for art in [&banner::NOVA, &banner::LITE] {
        for row in art.iter() {
            for ch in row.chars() {
                assert!(
                    !('\u{0300}'..='\u{036f}').contains(&ch),
                    "combining mark in art row: {row:?}"
                );
                assert!(
                    !('\u{1100}'..='\u{115f}').contains(&ch),
                    "wide glyph in art row: {row:?}"
                );
            }
        }
    }
}

#[test]
fn the_full_banner_fits_the_panel_it_is_drawn_into() {
    // The widest row plus the minimum indent must fit, or the paragraph wraps
    // and the wordmark tears.
    let width = (banner::ART_WIDTH + 4) as u16;
    let lines = banner::lines(width, banner::ART_HEIGHT as u16);

    for line in &lines {
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(
            cells(&text) <= width as usize,
            "banner line overflows a {width}-cell panel: {text:?}"
        );
    }
}

#[test]
fn a_narrow_panel_falls_back_to_the_compact_mark() {
    // Below the art's width the full wordmark cannot be legible, so the
    // compact mark is used instead of a clipped one.
    let lines = banner::lines(20, 10);
    let text: String = lines
        .iter()
        .flat_map(|l| l.spans.iter())
        .map(|s| s.content.as_ref())
        .collect();

    assert!(
        text.contains("NOVA-Lite"),
        "a narrow panel should show the compact mark, got: {text:?}"
    );
    assert!(
        !text.contains('█'),
        "a narrow panel should not attempt the block art: {text:?}"
    );
}

#[test]
fn a_short_panel_falls_back_to_the_compact_mark() {
    // Height matters too: the full banner is 15 rows, so a 6-row panel cannot
    // hold it even at full width.
    let lines = banner::lines(120, 6);
    let text: String = lines
        .iter()
        .flat_map(|l| l.spans.iter())
        .map(|s| s.content.as_ref())
        .collect();

    assert!(
        text.contains("NOVA-Lite"),
        "a short panel should show the compact mark, got: {text:?}"
    );
}

#[test]
fn the_banner_is_painted_in_an_empty_conversation() {
    let app = app();
    let painted = paint(&app, 100, 30).join("\n");

    assert!(
        painted.contains("NOVA-Lite") || painted.contains('█'),
        "the banner should be painted on an empty conversation:\n{painted}"
    );
    assert!(
        painted.contains("type /help"),
        "the tagline should be painted:\n{painted}"
    );
}

#[test]
fn the_banner_does_not_overflow_the_transcript_panel() {
    // The real path: draw the whole app and check the art stays inside the
    // conversation panel's inner width at a range of sizes.
    for (w, h) in [(100u16, 30u16), (120, 40), (80, 24), (200, 50)] {
        let app = app();
        let rows = paint(&app, w, h);

        // The panel's right border is the last column; the art must not reach
        // it. Measure in cells, not bytes: `rfind` returns a byte offset and
        // these glyphs are three bytes each.
        for row in rows.iter().filter(|r| r.contains('█')) {
            let last = row
                .char_indices()
                .filter(|(_, c)| *c == '█')
                .map(|(i, _)| row[..i].chars().count())
                .max()
                .unwrap();
            assert!(
                last < w as usize - 1,
                "art reaches the panel edge at {w}x{h}: {row:?}"
            );
        }
    }
}

#[test]
fn the_banner_disappears_once_the_conversation_starts() {
    // The banner is an empty-state affordance, not a permanent header.
    let (mut app, mut rx) = {
        let model = MockCompletionModel::from_stream_turns(vec![vec![
            rig::test_utils::MockStreamEvent::Text("hello".to_string()),
            rig::test_utils::MockStreamEvent::FinalResponse(rig::test_utils::mock_final(
                rig::completion::Usage::new(),
            )),
        ]]);
        let rt = common::session();
        App::new(common::agents(model, &rt), rt)
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        app.send_message("hi".to_string());
        while let Some(event) = rx.recv().await {
            let done = matches!(event, rust_agent::event::Event::Finished(_));
            app.handle_event(event);
            if done {
                break;
            }
        }
    });

    let painted = paint(&app, 100, 30).join("\n");
    assert!(
        !painted.contains('█'),
        "the banner should be gone once a message exists:\n{painted}"
    );
    assert!(painted.contains("hello"), "the reply should be painted");
}
