//! The run's event channel, under a reader that cannot keep up.
//!
//! The channel used to be unbounded, so a fast model and a slow terminal grew
//! the queue without limit: every streamed fragment was a message, and a long
//! answer is thousands of them. Bounding it is only half the fix. The other
//! half is that a fragment which cannot be sent must not be *dropped*, and must
//! not make the model's output arrive in lumps either.
//!
//! These tests drive `session::drive` directly, with a channel far smaller
//! than the stream it is fed, and a reader that sleeps between messages. What
//! they assert is the pair of properties that matter: every character survives,
//! and the number of messages is a small fraction of the number of fragments.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig::completion::Usage;
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent};

use rust_agent::event::Event;
use rust_agent::runtime::CancelToken;

mod common;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// A turn that streams `text` one character at a time, then finishes.
fn dribbled(text: &str) -> Vec<MockStreamEvent> {
    let mut turn: Vec<MockStreamEvent> = text
        .chars()
        .map(|c| MockStreamEvent::Text(c.to_string()))
        .collect();
    turn.push(MockStreamEvent::FinalResponse(mock_final(
        Usage::new().input_tokens(10).output_tokens(5),
    )));
    turn
}

/// What a reader saw, in the order it saw it.
#[derive(Default)]
struct Seen {
    text: String,
    text_messages: usize,
    /// The text received before the first tool call, and after it.
    before_call: String,
    after_call: String,
    calls: usize,
    finished: bool,
}

/// Drain `rx` until the producer closes it, sleeping between messages.
///
/// The sleep is the point: it makes the reader slower than the producer, which
/// is the condition the coalescing exists for. Without it the queue would
/// never fill and the test would pass whatever the producer did.
async fn drain(rx: &mut tokio::sync::mpsc::Receiver<Event>, per_message: Duration) -> Seen {
    let mut seen = Seen::default();
    while let Some(event) = rx.recv().await {
        match event {
            Event::TextDelta(chunk) => {
                seen.text_messages += 1;
                if seen.calls == 0 {
                    seen.before_call.push_str(&chunk);
                } else {
                    seen.after_call.push_str(&chunk);
                }
                seen.text.push_str(&chunk);
            }
            Event::ToolCallStarted(_) => seen.calls += 1,
            Event::Finished(_) => seen.finished = true,
            _ => {}
        }
        tokio::time::sleep(per_message).await;
    }
    seen
}

#[test]
fn a_slow_reader_gets_fewer_larger_messages_and_all_of_the_text() {
    // 20,000 one-character fragments into a queue of eight, read one message
    // per millisecond. Uncoalesced this is 20,000 messages and twenty seconds
    // of reading; coalesced it is a handful.
    let text: String = "abcdefghij".repeat(2_000);
    assert_eq!(text.chars().count(), 20_000);

    runtime().block_on(async {
        let model = MockCompletionModel::from_stream_turns(vec![dribbled(&text)]);
        let session = common::session();
        let agent = common::agents(model, &session).main;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let history = Arc::new(Mutex::new(Vec::new()));
        let producer = tokio::spawn(rust_agent::session::drive(
            agent,
            history,
            "go".to_string(),
            CancelToken::new(),
            tx,
            session.clone(),
        ));

        let seen = drain(&mut rx, Duration::from_millis(1)).await;
        producer.await.expect("the run task finishes");

        assert_eq!(
            seen.text, text,
            "coalescing must not lose or reorder a character"
        );
        assert!(seen.finished, "the run still reports that it finished");
        assert!(
            seen.text_messages < 200,
            "20,000 fragments should arrive as far fewer messages, got {}",
            seen.text_messages
        );
    });
}

#[test]
fn a_control_event_is_not_lost_behind_buffered_text() {
    // Text, then a tool call, then more text. The call has to arrive after the
    // first text and before the second: a control event that overtook the text
    // buffered before it would put the transcript out of order, and one that
    // was dropped would leave the call unaccounted for.
    runtime().block_on(async {
        let model = MockCompletionModel::from_stream_turns(vec![
            {
                let mut turn = dribbled("before");
                turn.insert(
                    0,
                    MockStreamEvent::tool_call(
                        "call_1",
                        "read_file",
                        serde_json::json!({ "path": "Cargo.toml" }),
                    ),
                );
                turn
            },
            dribbled("after"),
        ]);
        let session = common::session_over_a_crate();
        let agent = common::agents(model, &session).main;

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let history = Arc::new(Mutex::new(Vec::new()));
        let producer = tokio::spawn(rust_agent::session::drive(
            agent,
            history,
            "go".to_string(),
            CancelToken::new(),
            tx,
            session.clone(),
        ));

        let seen = drain(&mut rx, Duration::from_millis(1)).await;
        producer.await.expect("the run task finishes");

        assert_eq!(seen.calls, 1, "the call is reported once");
        assert_eq!(
            seen.before_call, "before",
            "the text streamed before the call arrives before it"
        );
        assert_eq!(
            seen.after_call, "after",
            "the text streamed after the call arrives after it"
        );
    });
}

#[test]
fn a_cancelled_run_still_reports_the_cancellation() {
    // The terminal event is sent with a wait, after everything buffered. A
    // cancellation that was itself dropped for want of room would leave the
    // app streaming forever, which is the failure the bounded channel could
    // most easily have introduced.
    runtime().block_on(async {
        let text: String = "x".repeat(50_000);
        let model = MockCompletionModel::from_stream_turns(vec![dribbled(&text)]);
        let session = common::session();
        let agent = common::agents(model, &session).main;

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let history = Arc::new(Mutex::new(Vec::new()));
        let cancel = CancelToken::new();
        let producer = tokio::spawn(rust_agent::session::drive(
            agent,
            history,
            "go".to_string(),
            cancel.clone(),
            tx,
            session.clone(),
        ));

        // Stop the run while the producer is still pushing text into a queue
        // nobody is draining.
        tokio::time::sleep(Duration::from_millis(20)).await;
        cancel.stop();

        let mut cancelled = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while let Ok(Some(event)) = tokio::time::timeout_at(deadline, rx.recv()).await {
            if matches!(event, Event::Cancelled) {
                cancelled = true;
            }
        }
        producer.await.expect("the run task finishes");

        assert!(cancelled, "the cancellation reaches the app");
    });
}
