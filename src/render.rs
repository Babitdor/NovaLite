//! Rendering for the agent TUI.
//!
//! Layout: one column, nothing fixed beside it. The terminal's own background
//! shows through, so the UI is chrome around the conversation rather than a
//! painted window on top of it.
//!
//! ```text
//!   title                                      ● ready
//! ────────────────────────────────────────────────────────
//!   the conversation (scrollable, borderless)
//!   ❯ a prompt
//!   the answer, as markdown
//!     ⚙ list_dir  path=.                       12ms
//! ────────────────────────────────────────────────────────
//!   ╭──────────────────────────────────────────────────╮
//!   │ ❯ input                                          │
//!   ╰──────────────────────────────────────────────────╯
//!   title · 12.3k/32k · calls 4 · 1.2k in · 340 out
//! ```
//!
//! The transcript is the conversation and the run's timeline at once: tool
//! calls, their arguments and their results are rows in it, so there is no
//! second place to look for what the run is doing. A message carries no label
//! row of its own — a user's turn opens with the prompt glyph, and an answer
//! is the markdown under it. Only overlays — the model picker and the
//! approval prompt — paint a surface of their own.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, BorderType, Borders, Clear, Padding, Paragraph, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Wrap,
    },
    Frame,
};

use crate::app::{App, ChatMessage, Role, ToolGroup};
use crate::tools::ToolCategory;

// ---------------------------------------------------------------- palette

/// The page behind everything: the terminal's own background.
///
/// Left unpainted on purpose. A TUI that fills the screen with its own colour
/// has to guess at the user's terminal theme; leaving it alone means the UI
/// matches whatever the rest of their shell uses.
const BG: Color = Color::Reset;
/// Panels and the input box.
const PANEL: Color = Color::Rgb(17, 22, 31);
/// Hairlines: panel edges, card borders, rules.
const BORDER: Color = Color::Rgb(31, 41, 55);
/// A border that wants to be noticed, such as the idle input box.
const BORDER_HI: Color = Color::Rgb(48, 62, 84);
const TEXT: Color = Color::Rgb(230, 237, 243);
const DIM: Color = Color::Rgb(139, 148, 158);
/// Section labels and metadata: present, but never competing with content.
const FAINT: Color = Color::Rgb(90, 100, 115);
const ACCENT: Color = Color::Rgb(91, 140, 255);
const VIOLET: Color = Color::Rgb(163, 113, 247);
const WARN: Color = Color::Rgb(210, 153, 34);
const ERROR: Color = Color::Rgb(248, 81, 73);
const OK: Color = Color::Rgb(63, 185, 80);
const CODE: Color = Color::Rgb(160, 200, 240);

// ------------------------------------------------------------- markdown

/// Markdown styling for assistant text.
///
/// The defaults are tuned for a light terminal: headings are cyan-on-cyan and
/// code blocks are delimited by literal ``` fences. Both read badly on this
/// dark palette, so the fence is dropped (the code style already sets the
/// block apart) and the heading colours are replaced with the panel's own.
#[derive(Clone, Debug, Default)]
struct MarkdownStyle;

impl tui_markdown::StyleSheet for MarkdownStyle {
    fn heading(&self, level: u8) -> Style {
        match level {
            1 => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            2 => Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
            3 => Style::default().fg(VIOLET).add_modifier(Modifier::BOLD),
            _ => Style::default().fg(VIOLET).add_modifier(Modifier::ITALIC),
        }
    }

    fn heading_marker(&self, _level: u8) -> &str {
        ""
    }

    fn code(&self) -> Style {
        Style::default().fg(CODE)
    }

    /// No fence lines: the code style alone marks the block, and a literal
    /// ``` in the transcript reads as a rendering bug.
    fn code_block_fence(&self) -> &str {
        ""
    }

    fn link(&self) -> Style {
        Style::default()
            .fg(ACCENT)
            .add_modifier(Modifier::UNDERLINED)
    }

    fn blockquote(&self) -> Style {
        Style::default().fg(DIM).add_modifier(Modifier::ITALIC)
    }

    fn html(&self) -> Style {
        Style::default().fg(DIM)
    }
}

/// Render assistant markdown with this app's stylesheet.
fn markdown(input: &str) -> ratatui::text::Text<'_> {
    tui_markdown::from_str_with_options(input, &tui_markdown::Options::new(MarkdownStyle))
}

/// A message body as it was last rendered.
///
/// Held by [`crate::app::App`] between frames. The two fields that are not the
/// lines are the whole of the invalidation rule: a message whose text has
/// changed is a message that is still being written, and a body wrapped to a
/// different width is a body that has to be wrapped again. Everything else
/// about a message is fixed once it exists, so nothing else can make this
/// stale.
#[derive(Debug, Clone)]
pub struct CachedBody {
    /// The text the lines were rendered from.
    content: String,
    /// The width the lines were wrapped to.
    width: u16,
    /// The rendered lines, owned so they outlive the borrow of the message.
    lines: Vec<Line<'static>>,
}

/// Message bodies as they were last rendered, kept between frames.
///
/// Parsing markdown is the expensive part of drawing an answer, and the
/// transcript is redrawn on every tick: ten times a second while a run is in
/// flight. Re-parsing every answer in a long conversation ten times a second
/// produces the same lines every time, and it is why a long session gets
/// slower to watch than a short one.
#[derive(Debug, Default)]
pub struct BodyCache {
    entries: std::collections::HashMap<usize, CachedBody>,
    /// How many bodies have been rendered since the app started.
    ///
    /// A cache is only worth its complexity if it is hit, and the property
    /// that matters is not "the cache is not empty" but "a redraw of an
    /// unchanged conversation renders nothing". Counting renders is what makes
    /// that assertable: a test cannot time a frame reliably, but it can count
    /// the work.
    renders: u64,
}

impl BodyCache {
    /// How many bodies have been rendered.
    pub fn renders(&self) -> u64 {
        self.renders
    }

    /// How many bodies are held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// The same lines with their text owned.
///
/// The markdown renderer borrows from the string it parses, and the cache
/// outlives any one frame's borrow of the conversation, so the text has to be
/// copied out. The copy happens once per revision rather than once per frame,
/// which is the whole point of the cache.
fn owned(lines: Vec<Line<'_>>) -> Vec<Line<'static>> {
    lines
        .into_iter()
        .map(|line| Line {
            style: line.style,
            alignment: line.alignment,
            spans: line
                .spans
                .into_iter()
                .map(|span| Span::styled(span.content.into_owned(), span.style))
                .collect(),
        })
        .collect()
}

/// The body of message `i`, wrapped to `width`, from the cache when it has not
/// changed since the last frame.
///
/// The cache is keyed by position and validated by text, which is what makes
/// the position safe as a key: a position that has shifted onto a different
/// message fails the text check and is rendered again. A key that was trusted
/// on its own would hand back one answer's lines under another answer's
/// header.
fn cached_body(app: &App, i: usize, msg: &ChatMessage, width: u16) -> Vec<Line<'static>> {
    let mut cache = app.body_cache.borrow_mut();
    if let Some(hit) = cache.entries.get(&i) {
        if hit.content == msg.content && hit.width == width {
            return hit.lines.clone();
        }
    }

    let lines = match msg.role {
        // An answer is markdown.
        Role::Assistant => owned(wrap_prefixed(
            markdown(&msg.content).lines,
            width as usize,
            BODY_INDENT,
            Style::default(),
        )),
        // The user's message is the one row with a prompt glyph on it, so it
        // has its own indent shape and is not wrapped like a body.
        Role::User => user_lines(&msg.content, width as usize),
        // System notes are the app talking about itself: dim and italic, so
        // they never read as something a person said.
        Role::System => {
            let style = Style::default().fg(DIM).add_modifier(Modifier::ITALIC);
            owned(wrap_prefixed(
                msg.content
                    .split('\n')
                    .map(|raw| Line::from(Span::styled(raw.to_string(), style)))
                    .collect(),
                width as usize,
                BODY_INDENT,
                Style::default(),
            ))
        }
        // Unreachable from the transcript, which paints tool messages as
        // blocks; kept so a reordering costs a plain row instead of a session.
        Role::Tool => owned(wrap_prefixed(
            msg.content
                .split('\n')
                .map(|raw| Line::from(Span::styled(raw.to_string(), Style::default().fg(TEXT))))
                .collect(),
            width as usize,
            BODY_INDENT,
            Style::default(),
        )),
    };

    cache.renders += 1;
    cache.entries.insert(
        i,
        CachedBody {
            content: msg.content.clone(),
            width,
            lines: lines.clone(),
        },
    );
    lines
}

/// Drop cache entries that no longer name a message.
///
/// The conversation is the only thing that can shrink, and it shrinks from the
/// front, so the entries to drop are the ones past the end. Without this a
/// session that was cleared and then used again would keep every line of the
/// conversation it had before.
fn prune_body_cache(app: &App) {
    let len = app.messages.len();
    let mut cache = app.body_cache.borrow_mut();
    if cache.entries.keys().any(|i| *i >= len) {
        cache.entries.retain(|i, _| *i < len);
    }
}

/// The caret that marks where the next token will land.
///
/// Two frames, not ten: this is a heartbeat, and a block that comes and goes
/// reads as "still writing" without competing with the header's spinner for
/// the same glance. It is appended to the body rather than rendered into it,
/// so it never touches the markdown cache and a blink costs no re-parse.
pub const CARET: [&str; 2] = ["▌", " "];

/// How many ticks one caret frame lasts.
///
/// The tick is the animation clock for the whole UI, and at its rate a caret
/// would strobe. Six ticks is about half a second, which is the rate a
/// terminal cursor blinks at and so the one the eye reads as a cursor rather
/// than as an error.
const CARET_TICKS: u64 = 6;

/// The spinner frames, newest first.
///
/// `pub` because the tests count them on a rendered header: a test that
/// restated the list would pass even if the frames stopped advancing.
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Columns a message body is indented by, so a wrapped continuation lines up
/// under its message's label rather than at the edge of the column.
const BODY_INDENT: &str = "   ";

/// The glyph a user's message and the input box both open with.
const PROMPT: &str = "❯ ";

/// Two columns of continuation indent under the prompt, so a wrapped message
/// lines up under its own text rather than under the glyph.
const USER_INDENT: &str = "  ";

// ------------------------------------------------------------------ draw

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // The cache is a view of the conversation, so it is pruned against the
    // conversation once a frame rather than at each of the places the
    // conversation can shrink.
    prune_body_cache(app);

    // One column, no rail: the whole width belongs to the conversation.
    //
    // The input box grows with the prompt, but never past a third of the
    // column: the conversation is what the user is reading, and a pasted
    // file must not push it off the screen. It is the only interactive part
    // of the screen, so it is the last thing to be squeezed: it keeps its
    // three rows (two borders and a line of text) whenever the terminal can
    // afford them, and below that it takes whatever is left.
    const HEADER_ROWS: u16 = 2; // the title, then a rule
    const GAP_ROWS: u16 = 1; // so the panels do not touch the input
    const FOOTER_ROWS: u16 = 1;
    const CHROME_ROWS: u16 = HEADER_ROWS + GAP_ROWS + FOOTER_ROWS;

    let room = area.height.saturating_sub(CHROME_ROWS);
    let input_h = app
        .input_height()
        .min((area.height / 3).max(3))
        .min(room.max(1));

    // On a terminal too short for the chrome the fixed rows would sum to more
    // than the height, and ratatui's solver then drops rows arbitrarily: the
    // same height can show the input box on one run and a blank gap on the
    // next. Giving the chrome up in a fixed order keeps the constraints an
    // exact solution at every height, and the order is the order of
    // usefulness: the box the user types into, then the status line, then the
    // header, then the gap.
    let chrome = area.height.saturating_sub(input_h);
    let footer_h = FOOTER_ROWS.min(chrome);
    let header_h = HEADER_ROWS.min(chrome.saturating_sub(footer_h));
    let gap_h = GAP_ROWS.min(chrome.saturating_sub(footer_h + header_h));
    let body_h = chrome.saturating_sub(footer_h + header_h + gap_h);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header_h),
            Constraint::Length(body_h),
            Constraint::Length(gap_h),
            Constraint::Length(input_h),
            Constraint::Length(footer_h),
        ])
        .split(area);

    render_header(frame, app, rows[0]);

    let transcript_area = rows[1];
    render_transcript(frame, app, transcript_area);

    render_input(frame, app, rows[3]);
    render_slash_suggestions(frame, app, rows[3]);
    render_footer(frame, app, rows[4]);

    // The picker floats over the conversation, so it does not reflow the
    // layout underneath it, and it is centred on the column it covers.
    if app.picker.is_some() {
        render_model_picker(frame, app, transcript_area);
    }

    // The approval prompt is drawn last and over everything, including the
    // picker: it is the one thing on screen the run is blocked on, and a
    // question that can be covered by another panel is one the user can
    // answer without having read it.
    if app.approval.is_some() {
        render_approval(frame, app, area);
    }
}

// -------------------------------------------------------------- approval

/// The approval overlay: what the action is, what it will do, and the two
/// answers.
///
/// The preview is the point of the panel. A prompt that only named the file
/// would be answered by reflex; the diff is what makes the question "is this
/// the change I meant" rather than "do I trust you", and the second question
/// is one the user has already answered by trusting the directory.
fn render_approval(frame: &mut Frame, app: &App, area: Rect) {
    let Some(pending) = app.approval.as_ref() else {
        return;
    };
    let request = pending.request();

    let hint = " y approve · n / Esc deny ";
    let width = area.width.saturating_sub(8).clamp(30, 100);
    let rows = area.height.saturating_sub(4).clamp(6, 24);
    let popup = centered(area, width, rows);

    frame.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(WARN))
        .title(Span::styled(
            format!(" {}? ", request.kind.label()),
            Style::default().fg(WARN).add_modifier(Modifier::BOLD),
        ))
        .style(Style::default().bg(PANEL));

    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let mut lines: Vec<Line> = vec![
        Line::from(Span::styled(
            format!("  {}", request.title),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("  {}", request.detail),
            Style::default().fg(DIM),
        )),
        Line::from(""),
    ];

    // The preview gets whatever rows are left, and says how many it dropped.
    // The panel is bounded, so a 400-line diff cannot push the question off
    // the top of the screen; the count is what tells the user there is more
    // than they are being shown.
    let room = inner.height.saturating_sub(lines.len() as u16 + 1) as usize;
    let preview: Vec<&str> = request.preview.lines().collect();
    for line in preview.iter().take(room) {
        let style = if line.starts_with('+') {
            Style::default().fg(OK)
        } else if line.starts_with('-') {
            Style::default().fg(ERROR)
        } else {
            Style::default().fg(DIM)
        };
        lines.push(Line::from(Span::styled(format!("  {line}"), style)));
    }
    if preview.len() > room {
        lines.push(Line::from(Span::styled(
            format!("  ... [{} more line(s)] ...", preview.len() - room),
            Style::default().fg(DIM),
        )));
    }

    frame.render_widget(Paragraph::new(lines), inner);

    if popup.height >= 2 {
        let hint_area = Rect {
            x: popup.x + 1,
            y: popup.y + popup.height - 1,
            width: popup.width.saturating_sub(2),
            height: 1,
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(hint, Style::default().fg(DIM)))),
            hint_area,
        );
    }
}

// ---------------------------------------------------------- model picker

/// Most rows the `/model` overlay will occupy, so a long list scrolls instead
/// of covering the whole conversation.
const MAX_PICKER_ROWS: u16 = 14;

/// The `/model` overlay: the models Ollama reports, with the current one
/// highlighted.
fn render_model_picker(frame: &mut Frame, app: &App, area: Rect) {
    let Some(picker) = app.picker.as_ref() else {
        return;
    };

    // Size to the content, but never wider or taller than the body. The
    // widest row is not always a model name: the "(current)" suffix and the
    // key hint both have to fit, or they are clipped by the border.
    let hint = " ↑↓ choose · Enter switch · Esc cancel ";
    let widest = picker
        .models
        .iter()
        .map(|m| m.chars().count() + "❯ ".chars().count() + "  (current)".chars().count())
        .max()
        .unwrap_or(0)
        .max("select a model".chars().count())
        .max(hint.chars().count());
    let width = ((widest + 2) as u16)
        .min(area.width.saturating_sub(4))
        .max(24);
    // Cap the height so a long list reads as an overlay rather than a
    // full-screen list; the rest scrolls.
    let rows = (picker.models.len() as u16 + 4)
        .min(MAX_PICKER_ROWS)
        .min(area.height.saturating_sub(2))
        .max(5);
    let popup = centered(area, width, rows);

    // Dim what is behind it, so the overlay reads as modal.
    frame.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(ACCENT))
        .title(Span::styled(
            " select a model ",
            Style::default().fg(ACCENT),
        ))
        .style(Style::default().bg(PANEL));

    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let mut lines: Vec<Line> = Vec::new();

    if picker.loading {
        lines.push(Line::from(Span::styled(
            "  loading models…",
            Style::default().fg(DIM).add_modifier(Modifier::ITALIC),
        )));
    } else if let Some(err) = &picker.error {
        lines.push(Line::from(Span::styled(
            "  could not list models",
            Style::default().fg(ERROR).add_modifier(Modifier::BOLD),
        )));
        for raw in err.lines().take(3) {
            lines.push(Line::from(Span::styled(
                format!("  {raw}"),
                Style::default().fg(DIM),
            )));
        }
    } else if picker.models.is_empty() {
        lines.push(Line::from(Span::styled(
            "  no models found",
            Style::default().fg(DIM),
        )));
    } else {
        // Scroll the list so the highlight stays visible in a short popup.
        let view = inner.height as usize;
        let start = picker
            .selected
            .saturating_sub(view.saturating_sub(1))
            .min(picker.models.len().saturating_sub(view));

        for (i, model) in picker.models.iter().enumerate().skip(start).take(view) {
            let selected = i == picker.selected;
            let current = *model == app.model_name;
            let (marker, style) = if selected {
                (
                    "❯ ",
                    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD),
                )
            } else {
                ("  ", Style::default().fg(TEXT))
            };
            let mut spans = vec![
                Span::styled(marker, style),
                Span::styled(model.clone(), style),
            ];
            if current {
                spans.push(Span::styled("  (current)", Style::default().fg(OK)));
            }
            lines.push(Line::from(spans));
        }
    }

    frame.render_widget(Paragraph::new(lines), inner);

    // The hint sits on the bottom border, where it does not consume a row.
    let hint = Line::from(Span::styled(hint, Style::default().fg(DIM)));
    if popup.height >= 2 {
        let hint_area = Rect {
            x: popup.x + 1,
            y: popup.y + popup.height - 1,
            width: popup.width.saturating_sub(2),
            height: 1,
        };
        frame.render_widget(Paragraph::new(hint), hint_area);
    }
}

/// A rectangle of `width` x `height` centred in `area`.
fn centered(area: Rect, width: u16, height: u16) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + (area.width - width) / 2,
        y: area.y + (area.height - height) / 2,
        width,
        height,
    }
}

// ---------------------------------------------------------------- readouts

/// A token count in the shortest form that keeps three significant figures.
fn short_count(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    }
}

/// The context indicator: the last request's prompt length against the window
/// the daemon was asked for. This is what makes the compaction threshold
/// visible before it fires.
///
/// The provider's figure is exact and describes the call that has just been
/// answered. The estimate beside it is the budget hook's measurement of the
/// request it sent, and it is marked with a `~` because it is characters
/// divided by a conservative average token length rather than a token count.
/// They are shown together because they answer different questions: how full
/// the window was, and how close the next request is to the budget.
fn context_indicator(app: &App) -> (String, Color) {
    let window = app.config().context.window as u64;
    let threshold = app.config().context.compact_at;
    if app.context_tokens == 0 || window == 0 {
        return ("ctx —".to_string(), FAINT);
    }
    // Kept as `u64` and clamped before formatting: narrowing a percentage
    // above 255 to `u8` would wrap, so a context far over the window would
    // paint a smaller number than one just over it.
    let pct = (app.context_tokens * 100 / window).min(999);
    let color = if pct >= 90 {
        ERROR
    } else if pct >= threshold as u64 {
        WARN
    } else {
        DIM
    };
    let estimate = if app.estimated_tokens > 0 {
        format!(" ~{}", short_count(app.estimated_tokens))
    } else {
        String::new()
    };
    (format!("ctx {pct}%{estimate}"), color)
}

/// The run's state as a dot, a colour and a word.
fn status_of(app: &App) -> (&'static str, Color, String) {
    // A run blocked on the user is the one state the user has to act on, so it
    // outranks the spinner: the run is not working, it is waiting, and a
    // spinner would say the opposite.
    if app.is_awaiting_approval() {
        return ("?", WARN, "awaiting approval".to_string());
    }
    if app.is_streaming() {
        let ch = SPINNER[(app.tick as usize) % SPINNER.len()];
        (ch, WARN, format!("{}…", app.status))
    } else {
        match app.status.as_str() {
            "error" => ("●", ERROR, app.status.clone()),
            "ready" => ("●", OK, "Agent online".to_string()),
            other => ("●", DIM, other.to_string()),
        }
    }
}

/// Append the caret to the body being written.
///
/// The write point, which the spinner cannot show: the spinner says the run
/// is working, and this says where the next token goes. It is put on the last
/// line so it follows the text, and on its own line when there is no text
/// yet, which is the moment the model has been called and has said nothing.
fn append_caret(lines: &mut Vec<Line<'static>>, tick: u64) {
    let frame = CARET[((tick / CARET_TICKS) as usize) % CARET.len()];
    let caret = Span::styled(frame.to_string(), Style::default().fg(VIOLET));
    match lines.last_mut() {
        Some(line) => line.spans.push(caret),
        None => lines.push(Line::from(caret)),
    }
}

/// The word a run's tool group says while the model is working.
///
/// A run whose only output so far is tool calls has no answer to carry the
/// state, so the group heading is where the word goes: the rows underneath
/// name the calls, and this says the run is still going.
///
/// The top header's status carries the same word, which is what keeps the
/// state visible when a long transcript pushes the rows off the top.
fn responding_label(app: &App) -> Option<&'static str> {
    if !app.is_streaming() {
        return None;
    }
    match app.status.as_str() {
        "thinking" => Some("thinking"),
        "reasoning" => Some("reasoning"),
        "compacting" => Some("compacting"),
        "cancelling" => Some("cancelling"),
        "streaming" => Some("responding"),
        // The tool states carry their own word on the status line, and the
        // tool rows carry the detail. The generic word is what they all
        // share, and the ellipsis is doing the animation.
        "searching" | "applying changes" | "running checks" | "correcting" => Some("responding"),
        s if s.starts_with("running ") || s.starts_with("calling ") => Some("responding"),
        // `ready` and `error` are not working states. The catch-all must not
        // claim them, or a finished run keeps saying "responding…" forever.
        _ => None,
    }
}

/// A horizontal hairline across `width` cells.
fn rule(width: usize) -> Line<'static> {
    Line::from(Span::styled("─".repeat(width), Style::default().fg(BORDER)))
}

/// `s` cut to `width` cells, with an ellipsis when anything was dropped.
fn truncate(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let mut out: String = s.chars().take(width - 1).collect();
    out.push('…');
    out
}

/// One row with `left` at the start and `right` flush to the end, padded
/// between. The left side is cut if the two would collide, so a long title
/// never pushes the status off the row.
fn spread<'a>(left: Vec<Span<'a>>, right: Vec<Span<'a>>, width: usize) -> Line<'a> {
    let r: usize = right.iter().map(|s| s.content.chars().count()).sum();
    let mut spans: Vec<Span<'a>> = Vec::new();

    // No room for both: the left side is the content, so it wins.
    let budget = if r + 1 > width { width } else { width - r - 1 };

    let mut used = 0;
    for s in left {
        if used >= budget {
            break;
        }
        let take = (budget - used).min(s.content.chars().count());
        let text: String = s.content.chars().take(take).collect();
        used += text.chars().count();
        spans.push(Span::styled(text, s.style));
    }

    if r < width {
        spans.push(Span::raw(" ".repeat(width - r - used)));
        spans.extend(right);
    }
    Line::from(spans)
}

// ---------------------------------------------------------------- header

/// The title row and the rule under it.
///
/// The title is the prompt the run is answering, so a conversation scrolled to
/// the bottom still says what it is about.
fn render_header(frame: &mut Frame, app: &App, area: Rect) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);

    let (dot, color, label) = status_of(app);
    let right = vec![
        Span::styled(format!("{dot} "), Style::default().fg(color)),
        Span::styled(label, Style::default().fg(color)),
        Span::raw("  "),
    ];
    let used: usize = right.iter().map(|s| s.content.chars().count()).sum();

    let title = app
        .messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User)
        .map(|m| m.content.lines().next().unwrap_or("").to_string())
        .unwrap_or_else(|| "new session".to_string());
    let title = truncate(&title, (rows[0].width as usize).saturating_sub(used + 2));

    let left = vec![
        Span::raw("  "),
        Span::styled(
            title,
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
    ];

    frame.render_widget(
        Paragraph::new(spread(left, right, rows[0].width as usize)).style(Style::default().bg(BG)),
        rows[0],
    );
    frame.render_widget(
        Paragraph::new(rule(rows[1].width as usize)).style(Style::default().bg(BG)),
        rows[1],
    );
}

// ------------------------------------------------------------ transcript

fn render_transcript(frame: &mut Frame, app: &App, area: Rect) {
    // No border: the column is the whole screen, and a box around the text
    // would eat columns to draw a frame nobody needs.
    let block = Block::default()
        .padding(Padding::new(2, 2, 0, 0))
        .style(Style::default().bg(BG));

    let inner = block.inner(area);
    // One column is held back for the scrollbar, so a long answer never runs
    // underneath it. The text is wrapped to this width here rather than left
    // to `Paragraph`, because the indent has to survive the wrap.
    let text_width = inner.width.saturating_sub(1).max(1);
    let lines = transcript_lines(app, text_width, inner.height);

    let paragraph = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(TEXT));

    // `line_count` is the wrapped height, so the scroll math matches what is
    // actually painted rather than the logical line count. The block has no
    // borders and no vertical padding, so it adds nothing to the count.
    let total = paragraph.line_count(text_width);
    let view = inner.height as usize;
    let max_scroll = total.saturating_sub(view);
    let scroll = if app.auto_scroll {
        max_scroll
    } else {
        (app.scroll as usize).min(max_scroll)
    };

    // Report the height back, so a key press can be clamped against the real
    // end of the transcript rather than against the `u16` ceiling. Without
    // this, holding PageDown past the end would leave the offset far above
    // the last row and the next PageUp would appear to do nothing.
    app.set_scroll_max(max_scroll.min(u16::MAX as usize) as u16);

    // `Paragraph::scroll` takes a `u16`, so the offset is clamped to that
    // rather than cast: a cast would wrap and paint a region from the middle
    // of a very long transcript. `MAX_TRANSCRIPT_LINES` keeps the wrapped
    // height well inside the range in practice; this is the backstop.
    let scroll = scroll.min(u16::MAX as usize) as u16;

    frame.render_widget(paragraph.scroll((scroll, 0)), area);

    if max_scroll > 0 {
        let mut state = ScrollbarState::new(max_scroll).position(scroll as usize);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .style(Style::default().fg(BORDER))
                .begin_symbol(None)
                .end_symbol(None),
            // The last column of the column, inside the right padding, so the
            // bar sits beside the text rather than on it.
            Rect {
                x: area.right().saturating_sub(1),
                y: area.y,
                width: 1,
                height: area.height,
            },
            &mut state,
        );
    }
}

/// The conversation as styled lines.
///
/// Each message is a block of its own: a user's words hang off the `❯ `
/// prompt glyph, and an answer is plain markdown in the conversation column —
/// no role label, because the glyph and the shape say who is speaking. Tool
/// calls are collapsible rows.
///
/// An empty conversation shows the NOVA-Lite banner instead of a bare prompt,
/// sized to the column it is drawn into.
fn transcript_lines(app: &App, width: u16, height: u16) -> Vec<Line<'_>> {
    let mut lines: Vec<Line> = Vec::new();

    if app.messages.is_empty() {
        lines.extend(crate::banner::lines(width, height));
        return lines;
    }

    // A blank row at the top, so the first block is not glued to the edge.
    lines.push(Line::from(""));

    // Tool calls are painted a group at a time, so a run of calls of one
    // category is headed by that category and can be folded as a unit. Every
    // other message is painted on its own.
    let groups = app.tool_groups();
    let mut next_group = 0usize;
    let mut i = 0usize;
    // The transcript is a view, so a block can scroll off the top: a long
    // answer pushes the blocks above it out of the buffer. Counting blocks
    // here would then still call the assistant header "newest" and hang a
    // spinner on a row nobody can see. The message index is the same quantity
    // as the block's position from the end, so it can be compared directly.
    let newest_msg = app.messages.len().saturating_sub(1);

    while i < app.messages.len() {
        let msg = &app.messages[i];
        let group = groups.get(next_group).filter(|g| g.first == i);

        // A blank line separates blocks. The calls inside one group are not
        // separated: a run of eight calls is one block, not eight, and eight
        // rows of whitespace would break it into unrelated fragments.
        if i > 0 {
            lines.push(Line::from(""));
        }

        // The state is read before the group is painted, so a run whose only
        // output so far is tool calls says what it is doing rather than
        // name a tool the row underneath has already named.
        let working = responding_label(app);

        if let Some(group) = group {
            next_group += 1;
            // A run whose only output so far is tool calls has no header row of
            // its own, so the group heading is the only place the state can
            // appear. The rows underneath carry the detail; this is the word.
            if let Some(word) = working {
                lines.push(Line::from(Span::styled(
                    format!("{word}…"),
                    Style::default().fg(FAINT).add_modifier(Modifier::ITALIC),
                )));
            }
            render_tool_group(&mut lines, app, group, width as usize);
            i = group.last + 1;
            continue;
        }

        if msg.role == Role::Tool {
            // Unreachable: `tool_groups` walks these same messages in this
            // same order, so every tool message opens a group. Painting the
            // call on its own beats dropping it.
            render_tool_block(&mut lines, app, i, msg, width as usize);
            i += 1;
            continue;
        }

        // A message carries no header row of its own. The run's state is the
        // top header's job and the tool rows' job; a per-message label row
        // would say "You" and "Agent" once per turn to no one's benefit, and
        // the blank line above already separates one message from the next.

        let mut body = cached_body(app, i, msg, width);
        // Only the answer being written, and only while it is being written:
        // a caret on a finished answer would read as an input box.
        if msg.role == Role::Assistant && i == newest_msg && app.is_streaming() {
            append_caret(&mut body, app.tick);
        }
        lines.extend(body);

        // The run's live reasoning trails the message it belongs to. There is
        // no side panel to put it in, and it is cleared when the run ends, so
        // it is a tail rather than history: the last few lines are what the
        // model is weighing right now.
        if msg.role == Role::Assistant && i == newest_msg && !app.reasoning.is_empty() {
            for line in reasoning_tail(&app.reasoning, width as usize) {
                lines.push(line);
            }
        }
        i += 1;
    }

    // The transcript is a *view* of the conversation, not the conversation:
    // the model's history is `App::history` and is untouched by this. Bounding
    // the view is what keeps a long session's frame cost flat, and it keeps
    // the wrapped height inside the `u16` that `Paragraph::scroll` takes.
    if lines.len() > MAX_TRANSCRIPT_LINES {
        let dropped = lines.len() - MAX_TRANSCRIPT_LINES;
        let mut tail: Vec<Line> = Vec::with_capacity(MAX_TRANSCRIPT_LINES + 1);
        tail.push(Line::from(Span::styled(
            format!("  … {dropped} earlier lines not shown"),
            Style::default().fg(FAINT).add_modifier(Modifier::ITALIC),
        )));
        tail.extend(lines.split_off(lines.len() - MAX_TRANSCRIPT_LINES));
        return tail;
    }

    lines
}
/// Most logical lines the transcript will paint.
///
/// A long session with large tool results reaches tens of thousands of lines,
/// and every one of them is wrapped and measured on every frame. Past this the
/// oldest lines are dropped from the view and a marker says how many.
///
/// The value is set from measurement, not taste. A release build paints a
/// 120x40 frame in 0.25 ms at one answer, 1.2 ms at 25, and 4.7 ms at 200
/// (which is 4 000 lines, this cap exactly). 4.7 ms is 200 fps, and the
/// slowest a frame may be before typing starts to feel lagged is about 20 ms,
/// so this is the largest cap that stays well inside the budget.
///
/// Those numbers are with [`CachedBody`] in place. Without it the same 4 000
/// lines cost 31 ms, which is over the budget: the cap alone was never what
/// kept the frame cheap, and the cache is what makes it true.
const MAX_TRANSCRIPT_LINES: usize = 4_000;

/// Rows of live reasoning the transcript keeps.
///
/// Reasoning is not history — it is cleared when the run ends — and it is
/// verbose enough to bury the answer if all of it is painted. The tail is what
/// the model is weighing right now.
const REASONING_TAIL_ROWS: usize = 6;

/// The last few lines of `reasoning`, dim and indented under the answer it
/// belongs to.
fn reasoning_tail<'a>(reasoning: &str, width: usize) -> Vec<Line<'a>> {
    let style = Style::default().fg(FAINT).add_modifier(Modifier::ITALIC);
    let text: Vec<Line<'a>> = reasoning
        .lines()
        .map(|line| Line::from(Span::styled(line.to_string(), style)))
        .collect();
    let wrapped = wrap_prefixed(text, width, BODY_INDENT, style);
    let start = wrapped.len().saturating_sub(REASONING_TAIL_ROWS);
    wrapped.into_iter().skip(start).collect()
}

/// Wrap styled lines to `width`, putting `prefix` on every visual line.
///
/// `Paragraph`'s own wrapping cannot do this: a wrapped continuation starts at
/// column 0, so a gutter applied to the first line of a paragraph leaves the
/// rest of it flush against the panel edge. Wrapping here keeps the gutter on
/// every visual line, which is what makes a long answer read as one block
/// instead of as a paragraph that fell out of its own margin.
fn wrap_prefixed<'a>(
    lines: Vec<Line<'a>>,
    width: usize,
    prefix: &'static str,
    prefix_style: Style,
) -> Vec<Line<'a>> {
    let avail = width.saturating_sub(prefix.chars().count()).max(1);
    let mut out: Vec<Line<'a>> = Vec::new();

    for line in lines {
        // The line as (char, style) pairs, so a break can fall anywhere
        // without losing the styling on either side of it.
        let mut cells: Vec<(char, Style)> = Vec::new();
        for span in &line.spans {
            for ch in span.content.chars() {
                cells.push((ch, span.style));
            }
        }

        if cells.is_empty() {
            out.push(Line::from(Span::styled(prefix, prefix_style)));
            continue;
        }

        let mut start = 0;
        while start < cells.len() {
            let end = (start + avail).min(cells.len());
            let mut cut = end;
            if end < cells.len() {
                // Break at the last space in the window, so words survive.
                if let Some(pos) = cells[start..end].iter().rposition(|(c, _)| *c == ' ') {
                    if pos > 0 {
                        cut = start + pos;
                    }
                }
            }
            if cut <= start {
                cut = end;
            }

            let mut spans = vec![Span::styled(prefix, prefix_style)];
            spans.extend(styled_runs(&cells[start..cut]));
            out.push(Line::from(spans));

            start = cut;
            // The space the break consumed is not part of the next line.
            while start < cells.len() && cells[start].0 == ' ' {
                start += 1;
            }
        }
    }

    out
}

/// A user's message: `❯ ` opens it, and every wrapped line under it is
/// indented two columns so the continuation lines up under the text rather
/// than under the glyph.
///
/// The continuation indent is applied to the first line too, then its two
/// cells are replaced by the prompt: both prefixes are two columns wide, so
/// the wrap is unchanged and the text of the first line does not move.
fn user_lines(text: &str, width: usize) -> Vec<Line<'static>> {
    let style = Style::default().fg(ACCENT);
    let body: Vec<Line> = text
        .split('\n')
        .map(|raw| Line::from(Span::styled(raw.to_string(), style)))
        .collect();
    let mut lines = owned(wrap_prefixed(body, width, USER_INDENT, style));
    if let Some(span) = lines.first_mut().and_then(|line| line.spans.first_mut()) {
        span.content = PROMPT.into();
    }
    lines
}

/// Group styled characters into spans, one per run of equal style.
fn styled_runs<'a>(cells: &[(char, Style)]) -> Vec<Span<'a>> {
    let mut spans: Vec<Span<'a>> = Vec::new();
    let mut buf = String::new();
    let mut style: Option<Style> = None;

    for (ch, s) in cells {
        match style {
            Some(prev) if prev == *s => buf.push(*ch),
            Some(prev) => {
                spans.push(Span::styled(std::mem::take(&mut buf), prev));
                style = Some(*s);
                buf.push(*ch);
            }
            None => {
                style = Some(*s);
                buf.push(*ch);
            }
        }
    }
    if let Some(prev) = style {
        spans.push(Span::styled(buf, prev));
    }
    spans
}

/// Columns a group's calls are indented by, so they sit under their heading
/// rather than beside it.
const GROUP_INDENT: usize = 2;

/// The colour a category's heading is painted in.
///
/// Explore and Run are the two the eye has to tell apart at a glance: one
/// changed nothing, the other ran something. Edit is the one that matters
/// most, so it gets the loudest colour of the three.
fn category_color(category: ToolCategory) -> Color {
    match category {
        ToolCategory::Explore => ACCENT,
        ToolCategory::Edit => VIOLET,
        ToolCategory::Run => WARN,
        ToolCategory::Other => FAINT,
    }
}

/// A run of tool calls of one category: a heading, then the calls under it.
///
/// The heading names the category and the run's size, so a long exploration
/// reads as `EXPLORE · 8 calls` with the calls beneath it. Folding the group
/// replaces the calls with a one-line digest of what they were, which is the
/// point of grouping them: the run can be got out of the way without losing
/// the fact that it happened.
fn render_tool_group(lines: &mut Vec<Line<'_>>, app: &App, group: &ToolGroup, width: usize) {
    let folded = app.is_group_folded(group.first);
    let n = group.len();

    let head = vec![
        Span::styled(
            if folded { "▸ " } else { "▾ " },
            Style::default().fg(if folded { FAINT } else { ACCENT }),
        ),
        Span::styled(
            group.category.label(),
            Style::default()
                .fg(category_color(group.category))
                .add_modifier(Modifier::BOLD),
        ),
    ];

    // The tail is the run's outcome, flush right so a column of groups can be
    // scanned for the one that failed. It is built as segments so it can shed
    // them from the right: the heading is the content, and a heading cut to
    // `EXPLO` to make room for a line count is worse than no line count.
    let mut segments: Vec<Vec<Span>> = Vec::new();
    let errors = group.errors(&app.messages);
    if errors > 0 {
        segments.push(vec![Span::styled(
            format!("{errors} failed"),
            Style::default().fg(ERROR).add_modifier(Modifier::BOLD),
        )]);
    }
    segments.push(vec![Span::styled(
        format!("{n} call{}", if n == 1 { "" } else { "s" }),
        Style::default().fg(FAINT),
    )]);
    let result_lines = group.result_lines(&app.messages);
    if result_lines > 0 {
        segments.push(vec![Span::styled(
            format!(
                "{result_lines} line{}",
                if result_lines == 1 { "" } else { "s" }
            ),
            Style::default().fg(FAINT),
        )]);
    }

    let label_w: usize = head.iter().map(|s| s.content.chars().count()).sum();
    while !segments.is_empty() {
        let tail_w: usize = segments
            .iter()
            .flatten()
            .map(|s| s.content.chars().count())
            .sum();
        // " · " between each pair, and one column of air before the tail.
        let seps = segments.len().saturating_sub(1) * 3;
        if label_w + tail_w + seps < width {
            break;
        }
        segments.pop();
    }

    let mut tail: Vec<Span> = Vec::new();
    for (i, segment) in segments.into_iter().enumerate() {
        if i > 0 {
            tail.push(Span::styled(" · ", Style::default().fg(BORDER)));
        }
        tail.extend(segment);
    }

    lines.push(spread(head, tail, width));

    if folded {
        // The digest: what the run was made of, so a folded group is not just
        // a count.
        let breakdown = group.breakdown(&app.messages);
        if !breakdown.is_empty() {
            lines.extend(wrap_prefixed(
                vec![Line::from(Span::styled(
                    breakdown,
                    Style::default().fg(DIM),
                ))],
                width,
                "  ",
                Style::default(),
            ));
        }
        return;
    }

    for idx in &group.members {
        let Some(msg) = app.messages.get(*idx) else {
            continue;
        };
        // Painted into a scratch buffer so the whole call, card and all, can
        // be shifted under the heading in one step.
        let mut block: Vec<Line> = Vec::new();
        render_tool_block(
            &mut block,
            app,
            *idx,
            msg,
            width.saturating_sub(GROUP_INDENT),
        );
        lines.extend(block.into_iter().map(|l| indent_line(l, GROUP_INDENT)));
    }
}

/// `line` shifted right by `n` columns.
fn indent_line<'a>(line: Line<'a>, n: usize) -> Line<'a> {
    if n == 0 {
        return line;
    }
    let mut spans = vec![Span::raw(" ".repeat(n))];
    spans.extend(line.spans);
    Line::from(spans)
}

/// A tool call as a collapsible row.
///
/// Collapsed, it is one row: the disclosure marker, the tool, its arguments,
/// and how much came back. Expanded, the same row keeps its place and the
/// result hangs off a `▏ ` gutter beneath it, so the output is tied to the
/// call that produced it without a rule drawn down the column beside it.
/// Collapsed is the default because a run with several calls would otherwise
/// bury the answer under raw output.
///
/// Called once per member of a [`ToolGroup`], with the width already reduced
/// by the group's indent.
fn render_tool_block(
    lines: &mut Vec<Line<'_>>,
    app: &App,
    idx: usize,
    msg: &ChatMessage,
    width: usize,
) {
    let Some(block) = msg.tool.as_ref() else {
        return;
    };

    let expanded = app.is_expanded(idx);
    // The marker is the whole of what a user reads at a glance, so a refusal
    // and a cancellation get their own: a denial is the user's own decision
    // coming back to them, and painting it with the cross of a broken build
    // would tell them their agent is broken when it is doing what they said.
    let (marker, color) = match block.outcome {
        crate::app::ToolOutcome::Failed => ("✗", ERROR),
        crate::app::ToolOutcome::Denied => ("⊘", WARN),
        crate::app::ToolOutcome::Cancelled => ("⊘", FAINT),
        crate::app::ToolOutcome::Ok => ("✓", OK),
        crate::app::ToolOutcome::Pending => ("◐", WARN),
    };

    // The disclosure marker leads the row: it is the control, so it belongs
    // where the eye starts rather than after a variable-length argument
    // summary that pushes it around.
    let mut head = vec![
        Span::styled(
            if expanded { "▾ " } else { "▸ " },
            Style::default().fg(if expanded { ACCENT } else { FAINT }),
        ),
        Span::styled(
            format!("{marker} "),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            block.name.clone(),
            Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
        ),
    ];
    if !block.args.is_empty() {
        head.push(Span::styled("  ", Style::default()));
        head.push(Span::styled(block.args.clone(), Style::default().fg(DIM)));
    }

    // The tail is the call's outcome, flush right so a column of calls can be
    // scanned for the one that returned nothing.
    let mut tail: Vec<Span> = Vec::new();
    if block.streaming {
        tail.push(Span::styled(
            "streaming…",
            Style::default().fg(WARN).add_modifier(Modifier::ITALIC),
        ));
    } else if block.done {
        let n = block.result.lines().count().max(1);
        tail.push(Span::styled(
            format!("{n} line{}", if n == 1 { "" } else { "s" }),
            Style::default().fg(FAINT),
        ));
        // A result the context budget cut is not the whole result, and the
        // model was told so in the text. The user is told here: a truncated
        // result that reads as complete is the one way it can mislead. The
        // word is kept short because the tail shares its row with the call's
        // arguments, and the arguments are what identify the call.
        if block.truncated {
            tail.push(Span::styled(" (cut)", Style::default().fg(WARN)));
        }
    }

    if !expanded {
        lines.push(spread(head, tail, width));
        return;
    }

    // Expanded: the call keeps the row it had, and its result hangs off a
    // `▏ ` gutter under it. Each result line is ruled rather than the whole
    // block being boxed: a box draws a rule the height of the transcript
    // beside every tool call, and the output is the part of the block the
    // reader has not already seen.
    lines.push(spread(head, tail, width));

    let mut body: Vec<Line> = if block.result.is_empty() {
        vec![Line::from(Span::styled(
            "(no output yet)".to_string(),
            Style::default().fg(FAINT).add_modifier(Modifier::ITALIC),
        ))]
    } else {
        let style = if block.is_error() {
            Style::default().fg(ERROR)
        } else {
            Style::default().fg(CODE)
        };
        block
            .result
            .lines()
            .map(|raw| Line::from(Span::styled(raw.to_string(), style)))
            .collect()
    };

    // The verdict, on its own row. The collapsed row carries it as a glyph,
    // which is all a glance needs; expanding a block is how a user finds out
    // what happened, and a word is what they are looking for. It is a row of
    // its own rather than more of the header because the header shares its
    // width with the call's arguments, and the arguments are what identify
    // the call.
    if block.done && block.outcome != crate::app::ToolOutcome::Ok {
        let color = if block.is_error() { ERROR } else { WARN };
        body.push(Line::from(Span::styled(
            block.outcome.label().to_string(),
            Style::default().fg(color),
        )));
    }

    for line in wrap_prefixed(body, width, "▏ ", Style::default().fg(BORDER)) {
        lines.push(line);
    }
}

// ----------------------------------------------------------------- input

fn render_slash_suggestions(frame: &mut Frame, app: &App, anchor: Rect) {
    let suggestions = app.command_suggestions();
    if suggestions.is_empty() || anchor.y == 0 {
        return;
    }
    // Build one line per candidate; keep it compact.
    let width: u16 = 20.min(anchor.width.saturating_sub(2).max(10));
    let popup_h: u16 = (suggestions.len() as u16 + 2).min(anchor.y.max(1));
    if popup_h < 3 {
        return;
    }
    let area = Rect {
        x: anchor.x + 1,
        y: anchor.y - popup_h,
        width,
        height: popup_h,
    };
    frame.render_widget(Clear, area);
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(BORDER))
        .style(Style::default().bg(PANEL));
    let lines: Vec<Line> = suggestions
        .iter()
        .map(|s| Line::from(Span::styled(format!("  {s}"), Style::default().fg(TEXT))))
        .collect();
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .style(Style::default().bg(PANEL)),
        area,
    );
}

fn render_input(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if app.is_streaming() {
            BORDER
        } else {
            BORDER_HI
        }))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(BG));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // The prompt is the whole affordance: Enter sends, and there is no send
    // button taking a bite out of the line the user is typing on.
    let text_w = inner.width.max(1);

    // The prompt marks the first line; continuation lines are indented under
    // it, so a pasted block reads as one message rather than as several.
    let prompt = PROMPT;
    let indent = USER_INDENT;
    let prompt_w = prompt.chars().count();

    let lines: Vec<&str> = app.input().split('\n').collect();
    let caret_line = app.caret_line();
    let caret_in_line = app.caret_in_line();

    // The window of lines on show. The caret's line is always inside it, so
    // typing at the end of a long prompt scrolls the box rather than the
    // caret leaving the box.
    let rows = inner.height as usize;
    let start = if lines.len() <= rows {
        0
    } else {
        caret_line
            .saturating_sub(rows.saturating_sub(1))
            .min(lines.len() - rows)
    };

    // One horizontal offset for the whole box, taken from the caret's line:
    // the caret stays visible and the lines stay aligned with each other.
    // Computed in `usize` and clamped at the end, because `Paragraph::scroll`
    // takes a `u16` and a pasted line can be longer than that.
    let caret_col = prompt_w + caret_in_line;
    let offset = caret_col
        .saturating_sub(text_w.saturating_sub(1) as usize)
        .min(u16::MAX as usize) as u16;

    let mut rendered: Vec<Line> = Vec::new();
    if app.input().is_empty() {
        rendered.push(Line::from(vec![
            Span::styled(prompt, Style::default().fg(ACCENT)),
            Span::styled(
                "Ask anything, or /help",
                Style::default().fg(FAINT).add_modifier(Modifier::ITALIC),
            ),
        ]));
    } else {
        for (i, text) in lines.iter().enumerate().skip(start).take(rows) {
            let prefix = if i == 0 { prompt } else { indent };
            rendered.push(Line::from(vec![
                Span::styled(prefix, Style::default().fg(ACCENT)),
                Span::styled((*text).to_string(), Style::default().fg(TEXT)),
            ]));
        }
    }

    frame.render_widget(
        Paragraph::new(rendered).scroll((0, offset)),
        Rect {
            x: inner.x,
            y: inner.y,
            width: text_w,
            height: inner.height,
        },
    );

    if inner.width > 0 && inner.height > 0 {
        let row = (caret_line - start) as u16;
        // Measured from the unclamped column, so a line longer than the
        // offset can express still puts the caret on the last visible column
        // rather than back at the start of the box.
        let col = caret_col
            .saturating_sub(offset as usize)
            .min(text_w.saturating_sub(1) as usize) as u16;
        frame.set_cursor_position((inner.x + col, inner.y + row));
    }
}

// ---------------------------------------------------------------- footer

/// What the footer says about where the transcript is.
///
/// "following" is the resting state. Once the user scrolls up, the useful
/// number is not "you are scrolled" but how far from the end they are: a
/// static hint leaves a user who scrolled up during a run unable to tell
/// whether the run is still producing anything, which is the one thing they
/// scrolled up to find out.
fn follow_label(app: &App) -> String {
    if app.auto_scroll {
        return "following".to_string();
    }
    let below = app.scroll_max().saturating_sub(app.scroll);
    if below == 0 {
        // Scrolled, but the end is on screen: there is nothing below to
        // count, so the hint is the only thing worth saying.
        return "scrolled (Ctrl+End to follow)".to_string();
    }
    format!(
        "{below} line{} below · Ctrl+End to follow",
        if below == 1 { "" } else { "s" }
    )
}

/// The colour of [`follow_label`].
///
/// Warn only while a run is in flight: that is when output is arriving that
/// the user is not seeing. A user who scrolled up in a finished conversation
/// is reading, not missing anything, and a warning colour there would cry wolf
/// every time they looked back.
fn follow_color(app: &App) -> Color {
    if !app.auto_scroll && app.is_streaming() {
        WARN
    } else {
        DIM
    }
}

fn render_footer(frame: &mut Frame, app: &App, area: Rect) {
    let tokens = if app.usage_in == 0 && app.usage_out == 0 {
        "no usage yet".to_string()
    } else {
        format!("{} in · {} out", app.usage_in, app.usage_out)
    };

    let (ctx, ctx_color) = context_indicator(app);

    // The footer is one row, so a narrow terminal has to drop something. The
    // segments are ordered by what the user needs: the run's accounting
    // first, the key hints last, so the hints go rather than a number being
    // clipped mid-word.
    let mut segments: Vec<Vec<Span>> = vec![
        vec![
            Span::styled("  calls ", Style::default().fg(FAINT)),
            Span::styled(app.calls.to_string(), Style::default().fg(TEXT)),
        ],
        vec![
            Span::styled(" · tokens ", Style::default().fg(FAINT)),
            Span::styled(tokens, Style::default().fg(TEXT)),
        ],
        vec![
            Span::styled(" · ", Style::default().fg(BORDER)),
            Span::styled(ctx, Style::default().fg(ctx_color)),
        ],
        vec![
            Span::styled(" · ", Style::default().fg(BORDER)),
            Span::styled(follow_label(app), Style::default().fg(follow_color(app))),
        ],
        vec![
            Span::styled(" · ", Style::default().fg(BORDER)),
            Span::styled("Enter send", Style::default().fg(DIM)),
        ],
        vec![
            Span::styled(" · ", Style::default().fg(BORDER)),
            // `PgUp/PgDn`, not `PgUp/PgDn scroll`: the verb costs seven
            // columns, and at 140 the row then sheds the `Esc stop` hint,
            // which is the one a user needs when a run misbehaves. The
            // `following` / `scrolled` segment already says the transcript
            // scrolls, so the keys alone are unambiguous.
            Span::styled("PgUp/PgDn", Style::default().fg(DIM)),
        ],
        // One segment for both fold keys, not two: a second segment cost the
        // quit hint at 140 columns, and the footer's job is to say that
        // folding exists, not to be the reference. `/help` is the reference.
        vec![
            Span::styled(" · ", Style::default().fg(BORDER)),
            Span::styled("Ctrl+T/G fold", Style::default().fg(DIM)),
        ],
        vec![
            Span::styled(" · ", Style::default().fg(BORDER)),
            Span::styled("Esc stop", Style::default().fg(DIM)),
        ],
    ];

    // Drop from the right until the row fits. The call count is the last
    // thing to go, because a footer that fits is worth more than one that
    // names every key.
    while segments.len() > 1 {
        let used: usize = segments
            .iter()
            .flatten()
            .map(|s| s.content.chars().count())
            .sum();
        if used <= area.width as usize {
            break;
        }
        segments.pop();
    }

    // The model needs a wider row than the numbers do. It is not a segment of
    // its own in the drop order, because it is the one thing here that is not
    // about this run: it is on the header, and `/model` changes it. So it is
    // shown when the row has room, and skipped when it does not, rather than
    // pushing out the count of what is below the view.
    let name = truncate(&app.model_name, (area.width as usize / 3).max(8));
    let model: Vec<Span> = vec![
        Span::styled("  ⬡ ", Style::default().fg(FAINT)),
        Span::styled(name, Style::default().fg(TEXT)),
    ];
    let used: usize = segments
        .iter()
        .flatten()
        .map(|s| s.content.chars().count())
        .sum();
    let model_w: usize = model.iter().map(|s| s.content.chars().count()).sum();
    let mut spans: Vec<Span> = Vec::new();
    if used + model_w <= area.width as usize {
        spans.extend(model);
    }
    spans.extend(segments.into_iter().flatten());

    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(BG)),
        area,
    );
}
