//! Terminal lifecycle, input handling, and the event loop.

use anyhow::Result;
use crossterm::event::{
    DisableBracketedPaste, EnableBracketedPaste, Event, EventStream, KeyCode, KeyEvent,
    KeyEventKind, KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io;
use std::time::Duration;
use tokio::sync::mpsc;

use crate::app::App;
use crate::event::Event as AppEvent;
use crate::render;

/// How often the UI repaints while the agent is streaming, so the spinner
/// animates and partial output appears without waiting for the next event.
const TICK: Duration = Duration::from_millis(80);

/// How often an idle screen is repainted.
///
/// The tick animates the spinner, which is only on screen while a run is in
/// flight, and keeps the message ages fresh, which change at most once a
/// second. Repainting an idle screen on every tick is twelve full redraws a
/// second for a frame that has not changed, which is battery the tool has no
/// reason to spend.
const IDLE_REDRAW: Duration = Duration::from_secs(1);

/// How long the event loop waits for the background tasks to stop on the way
/// out, before giving the terminal back.
///
/// Long enough for a killed command to be reaped and its pipes to close,
/// short enough that quitting never feels like a hang. A task that outlives
/// it is aborted and reported.
const DRAIN_ON_EXIT: Duration = Duration::from_secs(3);

/// Whether a tick should repaint the screen.
///
/// Split out from the event loop so the rule can be tested: the loop itself
/// needs a real terminal.
///
/// Three reasons to paint. A run in flight animates, and both the spinner and
/// the caret are functions of the tick, so a skipped frame freezes them.
/// Something changed and was not painted when it did, which is what `dirty`
/// carries over from the stream. Or the screen is simply old: the ages on the
/// messages drift, and one repaint a second keeps them honest.
fn tick_repaints(streaming: bool, dirty: bool, since_last_redraw: Duration) -> bool {
    streaming || dirty || since_last_redraw >= IDLE_REDRAW
}

pub enum InputResult {
    Quit,
    Send(String),
    Continue,
}

pub fn handle_input(app: &mut App, key: KeyEvent) -> InputResult {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

    // The approval prompt is modal, and it is checked before the picker: the
    // run is blocked on it, so it is the topmost thing on screen. While it is
    // up the keys answer it and nothing else, or a `y` typed into the prompt
    // box would be read as an answer to a question the user has not read yet.
    if app.is_awaiting_approval() {
        return handle_approval_input(app, key);
    }

    // The model picker is modal: while it is open the keys drive the list, not
    // the prompt, or typing a model name would also type into the input box.
    if app.picker.is_some() {
        return handle_picker_input(app, key);
    }

    match key.code {
        // Shift+Enter inserts a newline; plain Enter sends.
        KeyCode::Enter if shift => {
            app.insert_char('\n');
            InputResult::Continue
        }
        KeyCode::Enter => {
            let text = app.input().to_string();
            app.set_input(String::new());
            InputResult::Send(text)
        }

        // Esc and Ctrl+C stop the run in flight; with nothing running they
        // quit. The second press always quits, so a run that will not stop
        // cannot trap the user in the app.
        KeyCode::Esc => {
            if app.is_streaming() && !app.is_cancelling() {
                app.cancel_run();
                InputResult::Continue
            } else {
                InputResult::Quit
            }
        }
        KeyCode::Char('c') if ctrl => {
            if app.is_streaming() && !app.is_cancelling() {
                app.cancel_run();
                InputResult::Continue
            } else {
                InputResult::Quit
            }
        }

        // Tool blocks: Ctrl+T folds or unfolds the newest one, Ctrl+Shift+T
        // all of them. These must carry a modifier. A bare `t` is a letter the
        // user needs to type, and an arm here runs before the input box ever
        // sees the key, so binding it would make `t` untypeable.
        KeyCode::Char(c) if ctrl && c.eq_ignore_ascii_case(&'t') => {
            if shift {
                app.toggle_all_expanded();
            } else if let Some(idx) = app.last_tool_index() {
                app.toggle_expanded(idx);
            }
            InputResult::Continue
        }

        // Tool groups: Ctrl+G folds or unfolds the newest category group,
        // Ctrl+Shift+G all of them. Same modifier rule as Ctrl+T above.
        KeyCode::Char(c) if ctrl && c.eq_ignore_ascii_case(&'g') => {
            if shift {
                app.toggle_all_groups();
            } else if let Some(first) = app.last_group_first() {
                app.toggle_group(first);
            }
            InputResult::Continue
        }

        // Editing the prompt. The readline bindings are the ones a user
        // already has in their fingers, and every one of them is a modifier
        // combination so no letter becomes untypeable.
        KeyCode::Char('a') if ctrl => {
            app.caret_line_start();
            InputResult::Continue
        }
        KeyCode::Char('e') if ctrl => {
            app.caret_line_end();
            InputResult::Continue
        }
        KeyCode::Char('u') if ctrl => {
            app.kill_to_line_start();
            InputResult::Continue
        }
        KeyCode::Char('k') if ctrl => {
            app.kill_to_line_end();
            InputResult::Continue
        }
        KeyCode::Char('w') if ctrl => {
            app.delete_word_before();
            InputResult::Continue
        }
        KeyCode::Left if ctrl => {
            app.move_caret_word(-1);
            InputResult::Continue
        }
        KeyCode::Right if ctrl => {
            app.move_caret_word(1);
            InputResult::Continue
        }
        KeyCode::Left => {
            app.move_caret(-1);
            InputResult::Continue
        }
        KeyCode::Right => {
            app.move_caret(1);
            InputResult::Continue
        }
        KeyCode::Delete => {
            app.delete_forward();
            InputResult::Continue
        }
        // Home/End are the caret's line, which is what a text box does. The
        // transcript keeps Ctrl+Home/Ctrl+End for its own ends.
        KeyCode::Home if ctrl => {
            app.scroll_by(i32::MIN / 2);
            InputResult::Continue
        }
        KeyCode::End if ctrl => {
            app.scroll_to_bottom();
            InputResult::Continue
        }
        KeyCode::Home => {
            app.caret_line_start();
            InputResult::Continue
        }
        KeyCode::End => {
            app.caret_line_end();
            InputResult::Continue
        }

        // Transcript scrolling. Any manual move leaves follow mode. Up/Down
        // are the prompt history, as in a shell, so the transcript is moved
        // with the page keys and Ctrl+Up/Ctrl+Down.
        KeyCode::Up if ctrl => {
            app.scroll_by(-1);
            InputResult::Continue
        }
        KeyCode::Down if ctrl => {
            app.scroll_by(1);
            InputResult::Continue
        }
        KeyCode::Up => {
            app.history_prev();
            InputResult::Continue
        }
        KeyCode::Down => {
            app.history_next();
            InputResult::Continue
        }
        KeyCode::PageUp => {
            app.scroll_by(-10);
            InputResult::Continue
        }
        KeyCode::PageDown => {
            app.scroll_by(10);
            InputResult::Continue
        }

        KeyCode::Char(c) => {
            app.insert_char(c);
            InputResult::Continue
        }
        KeyCode::Backspace => {
            app.backspace();
            InputResult::Continue
        }
        KeyCode::Tab => {
            // Autocomplete a slash-command name while the user is editing
            // the command token; a Tab anywhere else still indents.
            if app.input().starts_with('/') && !app.input().contains(char::is_whitespace) {
                app.complete_slash_command();
                InputResult::Continue
            } else {
                app.insert_str("    ");
                InputResult::Continue
            }
        }
        _ => InputResult::Continue,
    }
}

/// Keys while an approval prompt is up.
///
/// Only an explicit yes approves. Enter, Escape, Ctrl+C and every other key
/// deny, because the failure this prompt exists to prevent is a write the
/// user did not mean to allow, and a key that means "yes" by accident is
/// exactly that. Denying does not stop the run: the model is told the action
/// was refused and can propose something else. A second Escape, once the
/// prompt is gone, stops the run the usual way.
fn handle_approval_input(app: &mut App, key: KeyEvent) -> InputResult {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match key.code {
        KeyCode::Char('y') | KeyCode::Char('Y') => app.answer_approval(true),
        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Enter | KeyCode::Esc => {
            app.answer_approval(false)
        }
        KeyCode::Char('c') if ctrl => app.answer_approval(false),
        _ => {}
    }
    InputResult::Continue
}

/// Keys while the `/model` picker is open.
fn handle_picker_input(app: &mut App, key: KeyEvent) -> InputResult {
    match key.code {
        KeyCode::Esc => {
            app.close_picker();
            InputResult::Continue
        }
        KeyCode::Enter => {
            app.confirm_model();
            InputResult::Continue
        }
        KeyCode::Up => {
            if let Some(p) = app.picker.as_mut() {
                p.move_by(-1);
            }
            InputResult::Continue
        }
        KeyCode::Down => {
            if let Some(p) = app.picker.as_mut() {
                p.move_by(1);
            }
            InputResult::Continue
        }
        KeyCode::Home => {
            if let Some(p) = app.picker.as_mut() {
                p.selected = 0;
            }
            InputResult::Continue
        }
        KeyCode::End => {
            if let Some(p) = app.picker.as_mut() {
                p.selected = p.models.len().saturating_sub(1);
            }
            InputResult::Continue
        }
        _ => InputResult::Continue,
    }
}

/// Take over the terminal, giving it back if any step fails.
///
/// The steps are not atomic. Raw mode can be on while the alternate screen is
/// not, and the `Terminal` can fail to build after both have succeeded. A
/// failure halfway through would otherwise leave the shell in raw mode with
/// no alternate screen: a terminal the user cannot type into, and one that
/// `leave_terminal` is never called for because the caller never got a
/// terminal to leave.
pub fn enter_terminal() -> Result<Terminal<CrosstermBackend<io::Stdout>>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    if let Err(error) = execute!(stdout, EnterAlternateScreen, EnableBracketedPaste) {
        let _ = disable_raw_mode();
        return Err(error.into());
    }
    match Terminal::new(CrosstermBackend::new(stdout)) {
        Ok(terminal) => Ok(terminal),
        Err(error) => {
            let _ = leave_terminal();
            Err(error.into())
        }
    }
}

/// Give the terminal back.
///
/// Best-effort and idempotent. Every step is attempted even when an earlier
/// one failed, because a terminal left in raw mode is worse than a failed
/// escape sequence, and the first error is returned so a caller that cares
/// can report it. It is also called from the panic hook, where it must not
/// panic itself: a panic inside a panic hook aborts the process and leaves
/// the terminal exactly as it was.
pub fn leave_terminal() -> Result<()> {
    let mut first: Option<io::Error> = None;
    if let Err(error) = disable_raw_mode() {
        first = Some(error);
    }
    if let Err(error) = execute!(io::stdout(), DisableBracketedPaste, LeaveAlternateScreen) {
        first = first.or(Some(error));
    }
    match first {
        Some(error) => Err(error.into()),
        None => Ok(()),
    }
}

/// Whether the TUI can run at all.
///
/// Split out from the loop so the rule can be tested: a test process's own
/// stdout is captured rather than a terminal, so the loop itself cannot be
/// driven here.
///
/// Without this the loop starts anyway when stdout is redirected, writes the
/// alternate-screen escape codes into the file, and then waits for a key that
/// can never arrive: `nova-lite > out.txt` hangs with no output and no error,
/// which is worse than refusing.
fn terminal_available(stdin_is_tty: bool, stdout_is_tty: bool) -> Result<(), String> {
    if stdin_is_tty && stdout_is_tty {
        return Ok(());
    }
    Err(
        "nova-lite is an interactive TUI and needs a terminal, but stdin or stdout is \
         redirected. Run it directly, or use `cargo run --example live -- \"prompt\"` for a \
         non-interactive run."
            .to_string(),
    )
}

pub async fn run_event_loop(app: &mut App, mut rx: mpsc::Receiver<AppEvent>) -> Result<()> {
    use std::io::IsTerminal;
    terminal_available(
        std::io::stdin().is_terminal(),
        std::io::stdout().is_terminal(),
    )
    .map_err(anyhow::Error::msg)?;

    let mut terminal = enter_terminal()?;

    // The loop body can fail (a draw error, a dead channel). Restore the
    // terminal on the way out whatever the reason, or the shell is left in
    // raw mode on the alternate screen.
    let result = event_loop(app, &mut rx, &mut terminal).await;

    // Stop the run and wait for the background tasks *before* the terminal
    // goes back. A task still running holds a child process, and a child that
    // outlives the session keeps writing to a screen that is no longer ours.
    // Stopping first is what makes the wait short: the tasks watch the same
    // signal, so they are already on their way out.
    let unfinished = app.settle(DRAIN_ON_EXIT).await;
    if unfinished > 0 {
        // Said out loud rather than swallowed. A task that would not stop is
        // a child process that may still be running, and the user is the only
        // one who can do anything about it.
        eprintln!(
            "nova-lite: {unfinished} background task(s) did not stop within {:?}; \
             a command they started may still be running.",
            DRAIN_ON_EXIT
        );
    }

    let restored = leave_terminal();
    result.and(restored)
}

async fn event_loop(
    app: &mut App,
    rx: &mut mpsc::Receiver<AppEvent>,
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
) -> Result<()> {
    let mut input_events = EventStream::new();
    let mut ticker = tokio::time::interval(TICK);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // Updated after every draw, so the idle path repaints at most once a
    // second however often the ticker fires.
    let mut last_redraw = std::time::Instant::now();
    // Taken out of the app so the loop can select on it. `None` in a test
    // that drives the app without a terminal: the branch below then never
    // completes, which is the truth, rather than being an error.
    let mut approvals = app.take_approval_channel();
    // A real SIGINT, alongside the keyboard's Ctrl+C. Raw mode turns off
    // `ISIG`, so on Unix the key arrives as an event and never as a signal;
    // on Windows a console control event can still be delivered, and a
    // `kill` from another process is a signal on both. Without this the
    // process would die with the terminal still in raw mode.
    let mut interrupt = Box::pin(tokio::signal::ctrl_c());
    // Something has changed that the screen does not show yet. Set by the
    // stream and cleared by a draw: the tick is what turns it into a frame.
    let mut dirty = false;

    loop {
        // Painted at the end of this iteration, rather than coalesced into
        // the next tick. A keystroke and an approval prompt are painted at
        // once, because the user is waiting on the screen to answer them. A
        // stream's events are not: a local model delivers tokens faster than
        // any terminal can usefully repaint, and a frame per token spends the
        // machine on re-wrapping text that is about to change again — on the
        // same machine that is running the model.
        let mut paint_now = false;

        tokio::select! {
            // Keyboard first: a run that streams thousands of deltas must
            // not make the UI stop responding to Esc or to typing.
            biased;

            maybe_key = input_events.next() => {
                match maybe_key {
                    Some(Ok(Event::Key(key))) => {
                        if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) {
                            match handle_input(app, key) {
                                InputResult::Quit => break,
                                InputResult::Send(text) => app.send_message(text),
                                InputResult::Continue => {}
                            }
                        }
                    }
                    // A paste arrives as one event; insert it verbatim so
                    // newlines in pasted text do not submit the prompt.
                    Some(Ok(Event::Paste(text))) => app.insert_str(&text),
                    Some(Ok(_)) => {}
                    Some(Err(_)) => {}
                    None => break,
                }
                paint_now = true;
            }

            maybe_ev = rx.recv() => {
                match maybe_ev {
                    Some(event) => {
                        app.handle_event(event);
                        dirty = true;
                    }
                    None => break,
                }
            }

            // A tool is asking to do something. The request is put on screen
            // and the tool stays blocked on its reply, so the run is paused
            // here rather than racing ahead: nothing else can be approved
            // while this one is unanswered, because the approver serializes
            // the prompts.
            maybe_approval = async {
                match approvals.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                paint_now = true;
                match maybe_approval {
                    Some(pending) => app.approval = Some(pending),
                    // The sender is gone: the runtime was dropped, so no
                    // further request can arrive. Stop watching rather than
                    // spinning on a closed channel.
                    None => approvals = None,
                }
            }

            _ = ticker.tick() => {
                app.tick = app.tick.wrapping_add(1);
                paint_now = tick_repaints(app.is_streaming(), dirty, last_redraw.elapsed());
            }

            // A signal, not a key. Stop the run if one is in flight, and
            // leave either way: the user asked the process to end, and the
            // cleanup below is what gives the terminal back.
            _ = &mut interrupt => {
                app.cancel_run();
                break;
            }
        }

        // The last frame before the screen goes back is worth painting: the
        // quit may have been the answer to something still on it.
        if !paint_now && !app.should_quit {
            continue;
        }

        terminal.draw(|f| render::draw(f, app))?;
        dirty = false;
        last_redraw = std::time::Instant::now();

        if app.should_quit {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_streaming_tick_always_repaints() {
        // The spinner is the whole point of the tick while a run is in
        // flight, so it must not be rate-limited.
        assert!(tick_repaints(true, false, Duration::ZERO));
        assert!(tick_repaints(true, false, TICK));
    }

    #[test]
    fn a_tick_paints_what_the_stream_changed() {
        // The stream does not paint its own events: a fast model would spend
        // the machine on a frame per token. The tick is what turns them into
        // a frame, so a tick with something waiting must always paint, even
        // on the idle path where the run has just finished.
        assert!(
            tick_repaints(false, true, Duration::ZERO),
            "an event that arrived since the last frame has to reach the screen"
        );
    }

    #[test]
    fn an_idle_tick_repaints_at_most_once_a_second() {
        // The ages on the messages are the only thing that changes on an idle
        // screen, and they change at most once a second.
        assert!(
            !tick_repaints(false, false, Duration::ZERO),
            "a tick right after a redraw must not repaint again"
        );
        assert!(
            !tick_repaints(false, false, TICK),
            "an idle tick must not repaint at the streaming rate"
        );
        assert!(
            !tick_repaints(false, false, IDLE_REDRAW - Duration::from_millis(1)),
            "the boundary is exclusive"
        );
        assert!(
            tick_repaints(false, false, IDLE_REDRAW),
            "an idle screen must still refresh its ages"
        );
    }

    #[test]
    fn a_redirected_stream_refuses_to_start_the_tui() {
        // Both streams have to be terminals: a redirected stdout cannot paint
        // and a redirected stdin cannot deliver a key, and either one leaves
        // the loop waiting forever.
        assert!(terminal_available(true, true).is_ok());
        for (stdin_is_tty, stdout_is_tty) in [(false, true), (true, false), (false, false)] {
            let err = terminal_available(stdin_is_tty, stdout_is_tty)
                .expect_err("a redirected stream must be refused");
            assert!(
                err.contains("needs a terminal"),
                "the error should say what is wrong, got: {err}"
            );
        }
    }
}
