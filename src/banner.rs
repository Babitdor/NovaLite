//! The NOVA-Lite wordmark shown in an empty conversation.
//!
//! The art is a fixed-width block face: every row of a wordmark occupies the
//! same number of cells, which is what keeps the letters from shearing. The
//! guards in `tests/banner.rs` assert that, and assert the art fits the panel
//! it is drawn into, so a later edit cannot quietly break the alignment.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// "NOVA" in ANSI-shadow block letters. Every row is [`ART_WIDTH`] cells.
pub const NOVA: [&str; 6] = [
    "███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ ",
    "████╗  ██║██╔═══██╗██║   ██║██╔══██╗",
    "██╔██╗ ██║██║   ██║██║   ██║███████║",
    "██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══██║",
    "██║ ╚████║╚██████╔╝ ╚████╔╝ ██║  ██║",
    "╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝  ╚═╝",
];

/// "LITE" in the same face, set under the wordmark.
pub const LITE: [&str; 6] = [
    "██╗     ██╗████████╗███████╗",
    "██║     ██║╚══██╔══╝██╔════╝",
    "██║     ██║   ██║   █████╗  ",
    "██║     ██║   ██║   ██╔══╝  ",
    "███████╗██║   ██║   ███████╗",
    "╚══════╝╚═╝   ╚═╝   ╚══════╝",
];

/// Width of the widest wordmark row.
pub const ART_WIDTH: usize = 36;
/// Width of the `LITE` wordmark, which is narrower than `NOVA`.
pub const LITE_WIDTH: usize = 28;
/// Rows the full banner occupies: a blank, both wordmarks, a blank, the tagline.
pub const ART_HEIGHT: usize = 15;

/// Smallest indent, so the art never touches the panel border.
const MIN_INDENT: usize = 2;
/// `LITE` is centred under the wider `NOVA`.
const LITE_INDENT: usize = (ART_WIDTH - LITE_WIDTH) / 2;

const TAGLINE: &str = "a small coding agent · type /help for the tool list";
const TAGLINE_SHORT: &str = "type /help for the tool list";
const COMPACT_MARK: &str = "NOVA-Lite";

const ACCENT: Color = Color::Rgb(122, 196, 255);
const VIOLET: Color = Color::Rgb(170, 150, 255);
const STEEL: Color = Color::Rgb(140, 160, 200);
const DIM: Color = Color::Rgb(112, 120, 138);

/// The banner for a panel of `width` x `height` cells.
///
/// The full wordmark needs room to be legible; below that the compact mark is
/// used instead, so a small terminal gets a readable name rather than a
/// clipped one.
pub fn lines(width: u16, height: u16) -> Vec<Line<'static>> {
    let width = width as usize;
    let height = height as usize;

    if width >= ART_WIDTH + MIN_INDENT * 2 && height >= ART_HEIGHT {
        full(width)
    } else {
        compact(width)
    }
}

fn full(width: usize) -> Vec<Line<'static>> {
    let indent = (width.saturating_sub(ART_WIDTH) / 2).max(MIN_INDENT);
    let pad = " ".repeat(indent);
    let tagline = tagline_for(width);
    let mut out = Vec::with_capacity(ART_HEIGHT);

    out.push(Line::from(""));

    // A vertical gradient across the wordmark, so the block face reads as one
    // object rather than six unrelated rows.
    let last = (NOVA.len() - 1) as f32;
    for (i, row) in NOVA.iter().enumerate() {
        out.push(Line::from(Span::styled(
            format!("{pad}{row}"),
            Style::default().fg(lerp(ACCENT, VIOLET, i as f32 / last)),
        )));
    }

    let lite_pad = " ".repeat(indent + LITE_INDENT);
    for row in LITE.iter() {
        out.push(Line::from(Span::styled(
            format!("{lite_pad}{row}"),
            Style::default().fg(STEEL),
        )));
    }

    out.push(Line::from(""));
    out.push(Line::from(Span::styled(
        format!("{}{}", " ".repeat(centered_indent(tagline, width)), tagline),
        Style::default().fg(DIM),
    )));

    out
}

/// The longest tagline that fits `width`, so a narrow panel gets a shorter
/// line rather than a wrapped one.
fn tagline_for(width: usize) -> &'static str {
    if width >= TAGLINE.chars().count() + MIN_INDENT * 2 {
        TAGLINE
    } else {
        TAGLINE_SHORT
    }
}

fn compact(width: usize) -> Vec<Line<'static>> {
    let tagline = tagline_for(width);
    vec![
        Line::from(""),
        Line::from(Span::styled(
            format!(
                "{}{COMPACT_MARK}",
                " ".repeat(centered_indent(COMPACT_MARK, width))
            ),
            Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("{}{tagline}", " ".repeat(MIN_INDENT)),
            Style::default().fg(DIM),
        )),
    ]
}

/// Indent that centres `text`, never closer to the edge than [`MIN_INDENT`].
fn centered_indent(text: &str, width: usize) -> usize {
    (width.saturating_sub(text.chars().count()) / 2).max(MIN_INDENT)
}

/// Blend two RGB colours. Non-RGB inputs are returned unchanged.
fn lerp(from: Color, to: Color, t: f32) -> Color {
    let (Color::Rgb(fr, fg, fb), Color::Rgb(tr, tg, tb)) = (from, to) else {
        return from;
    };
    let mix = |a: u8, b: u8| (a as f32 + (b as f32 - a as f32) * t).round() as u8;
    Color::Rgb(mix(fr, tr), mix(fg, tg), mix(fb, tb))
}
