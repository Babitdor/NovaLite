//! Tool-call repair: the argument aliases, the salvage, and the caps.
//!
//! A small model gets a tool call *nearly* right far more often than it gets
//! it exactly right. It writes `readFile` for `read_file`, or `file_path` for
//! `path`, or wraps the arguments in a fenced block, or double-encodes them as
//! a JSON string. Every one of those is a call the model meant to make, and
//! every one of them costs two model calls to recover from if it is refused.
//!
//! These tests pin the two layers that stop that: the lenient deserializers in
//! `tools.rs`, which widen what is accepted without changing what is
//! advertised, and the `RepairToolCalls` hook, which renames a call to a tool
//! that is registered and hands back salvaged arguments.
//!
//! The model is scripted rather than live: the thing under test is what the
//! agent does with a malformed call, not what a model chooses to emit. Where a
//! repair is claimed to have *worked*, the assertion is on a side effect (a
//! file on disk) or on what the model was sent, never on the repair counter
//! alone: a counter that increments while the call still fails would pass.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rig::completion::Usage;
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent};

use rust_agent::app::App;
use rust_agent::event::Event;
use rust_agent::tools::{EditFileArgs, GrepArgs, ReadFileArgs, ShellArgs, WriteFileArgs};

mod common;

type Rx = tokio::sync::mpsc::Receiver<Event>;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

/// A turn that answers with `text`.
fn turn(text: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::Text(text.to_string()),
        MockStreamEvent::FinalResponse(mock_final(Usage::new().input_tokens(100).output_tokens(5))),
    ]
}

/// A turn that calls one tool and then ends.
fn call(id: &str, name: &str, arguments: serde_json::Value) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call(id, name, arguments),
        MockStreamEvent::FinalResponse(mock_final(Usage::new().input_tokens(100).output_tokens(5))),
    ]
}

/// Pump events until the app settles.
///
/// "Settled" is idle for a short grace period rather than a particular event:
/// a run that fails starts a compaction and then a second run, so the end of
/// the work is not the end of the stream.
async fn pump(app: &mut App, rx: &mut Rx) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        let idle = !app.is_streaming() && !app.compacting;
        let wait = if idle {
            std::time::Duration::from_millis(200)
        } else {
            std::time::Duration::from_secs(20)
        };
        match tokio::time::timeout(wait, rx.recv()).await {
            Ok(Some(event)) => app.handle_event(event),
            Ok(None) => break,
            Err(_) if idle => break,
            Err(_) => panic!("the app did not settle within 20s"),
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "the app did not settle within 20s"
        );
    }
}

/// Send `prompt` and pump until the app settles.
fn run_to_idle(app: &mut App, rx: &mut Rx, prompt: &str) {
    runtime().block_on(async {
        app.send_message(prompt.to_string());
        pump(app, rx).await;
    });
}

/// Everything the model was sent, across every call, as one string.
///
/// The `Debug` form rather than a walk of the message enum: what is being
/// asserted is that a particular sentence reached the model, and the shape of
/// the message carrying it is rig's business, not this test's.
fn sent(model: &MockCompletionModel) -> String {
    model
        .requests()
        .iter()
        .flat_map(|request| request.chat_history.iter())
        .map(|message| format!("{message:?}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A JSON object from `(key, value)` pairs, for driving a deserializer with a
/// key that is not a literal.
fn object(fields: &[(&str, serde_json::Value)]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (key, value) in fields {
        map.insert((*key).to_string(), value.clone());
    }
    serde_json::Value::Object(map)
}

/// Deserialize `T` from `(key, value)` pairs, naming the failure.
fn from_object<T: serde::de::DeserializeOwned>(fields: &[(&str, serde_json::Value)]) -> T {
    let value = object(fields);
    serde_json::from_value(value.clone())
        .unwrap_or_else(|e| panic!("{value} should deserialize: {e}"))
}

// ---------------------------------------------------------------------------
// The aliases
// ---------------------------------------------------------------------------

#[test]
fn every_spelling_of_a_path_names_the_path() {
    // `file_path` is the one a model trained on OpenAI's schema reaches for,
    // and it is the single most common argument failure there is.
    for alias in [
        "path",
        "file_path",
        "filepath",
        "filename",
        "file",
        "dir",
        "directory",
    ] {
        let args: ReadFileArgs = from_object(&[(alias, serde_json::json!("src/lib.rs"))]);
        assert_eq!(args.path, "src/lib.rs", "`{alias}` should name the path");

        let args: WriteFileArgs = from_object(&[
            (alias, serde_json::json!("src/lib.rs")),
            ("content", serde_json::json!("x")),
        ]);
        assert_eq!(args.path, "src/lib.rs", "`{alias}` should name the path");
    }
}

#[test]
fn every_spelling_of_the_other_arguments_deserializes() {
    for alias in ["content", "contents", "text", "data"] {
        let args: WriteFileArgs = from_object(&[
            ("path", serde_json::json!("a.txt")),
            (alias, serde_json::json!("hello")),
        ]);
        assert_eq!(args.content, "hello", "`{alias}` should be the content");
    }

    for alias in ["command", "cmd", "script", "shell_command"] {
        let args: ShellArgs = from_object(&[(alias, serde_json::json!("cargo test"))]);
        assert_eq!(
            args.command, "cargo test",
            "`{alias}` should be the command"
        );
    }

    for alias in ["pattern", "query", "regex", "search", "glob"] {
        let args: GrepArgs = from_object(&[(alias, serde_json::json!("fn main"))]);
        assert_eq!(args.pattern, "fn main", "`{alias}` should be the pattern");
    }

    for alias in ["start_line", "from_line", "start"] {
        let args: ReadFileArgs = from_object(&[
            ("path", serde_json::json!("a.rs")),
            (alias, serde_json::json!(3)),
        ]);
        assert_eq!(
            args.start_line,
            Some(3),
            "`{alias}` should be the first line"
        );
    }

    for alias in ["end_line", "to_line", "end"] {
        let args: ReadFileArgs = from_object(&[
            ("path", serde_json::json!("a.rs")),
            (alias, serde_json::json!(9)),
        ]);
        assert_eq!(args.end_line, Some(9), "`{alias}` should be the last line");
    }
}

#[test]
fn every_spelling_of_an_edit_block_deserializes() {
    for alias in ["old_string", "old", "old_str", "search", "find"] {
        let args: EditFileArgs = from_object(&[
            ("path", serde_json::json!("a.rs")),
            (alias, serde_json::json!("before")),
            ("new_string", serde_json::json!("after")),
        ]);
        assert_eq!(args.edits.len(), 1);
        assert_eq!(
            args.edits[0].old_string, "before",
            "`{alias}` should be the search text"
        );
    }

    for alias in ["new_string", "new", "new_str", "replace", "replacement"] {
        let args: EditFileArgs = from_object(&[
            ("path", serde_json::json!("a.rs")),
            ("old_string", serde_json::json!("before")),
            (alias, serde_json::json!("after")),
        ]);
        assert_eq!(args.edits.len(), 1);
        assert_eq!(
            args.edits[0].new_string, "after",
            "`{alias}` should be the replacement"
        );
    }
}

// ---------------------------------------------------------------------------
// The four `edits` shapes
// ---------------------------------------------------------------------------

#[test]
fn all_four_edits_shapes_deserialize() {
    let block = serde_json::json!({ "old_string": "before", "new_string": "after" });

    let shapes: [(&str, serde_json::Value); 4] = [
        // 1. The schema: an array of blocks.
        ("an array of blocks", serde_json::json!([block])),
        // 2. A single bare block, with no array around it.
        ("a single bare block", block.clone()),
        // 3. A JSON string holding either of the above, double-encoded.
        (
            "a double-encoded string",
            serde_json::json!(block.to_string()),
        ),
        // 4. A flattened single edit, with the two strings beside `path`.
        (
            "a flattened edit",
            serde_json::json!({ "old_string": "before", "new_string": "after" }),
        ),
    ];

    for (name, edits) in shapes {
        let args: EditFileArgs =
            from_object(&[("path", serde_json::json!("a.rs")), ("edits", edits)]);
        assert_eq!(args.edits.len(), 1, "{name} should be one edit");
        assert_eq!(args.edits[0].old_string, "before", "{name}");
        assert_eq!(args.edits[0].new_string, "after", "{name}");
    }
}

#[test]
fn a_double_encoded_array_of_blocks_deserializes() {
    // The shape a model produces when it serializes its own tool call: the
    // array, then the array as a string.
    let inner = serde_json::json!([
        { "old_string": "one", "new_string": "1" },
        { "old_string": "two", "new_string": "2" },
    ]);
    let args: EditFileArgs = from_object(&[
        ("path", serde_json::json!("a.rs")),
        ("edits", serde_json::json!(inner.to_string())),
    ]);
    assert_eq!(args.edits.len(), 2);
    assert_eq!(args.edits[1].old_string, "two");
}

#[test]
fn edits_that_are_not_an_edit_block_are_refused_with_a_reason() {
    // Widening what is accepted must not mean accepting anything: a number is
    // not an edit, and the error has to say what was expected.
    let value = object(&[
        ("path", serde_json::json!("a.rs")),
        ("edits", serde_json::json!(7)),
    ]);
    let error = serde_json::from_value::<EditFileArgs>(value)
        .expect_err("a number is not an edit")
        .to_string();
    assert!(
        error.contains("array of edit blocks"),
        "the error should say what was expected: {error}"
    );
}

// ---------------------------------------------------------------------------
// The hook
// ---------------------------------------------------------------------------

#[test]
fn a_misspelled_tool_name_is_repaired_rather_than_refused() {
    // `writeFile` is `write_file` with the separator dropped and the case
    // changed. Normalizing both sides is what catches it, with no table.
    //
    // The assertion is the file on disk, not the counter: a repair that
    // renamed the call to something that still failed would leave the counter
    // at one and the file absent.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "writeFile",
            serde_json::json!({ "path": "made.txt", "content": "hello" }),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model, &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "write the file");

    assert_eq!(rt.repair_report().tool_name_repairs, 1);
    let written = std::fs::read_to_string(rt.workspace().root().join("made.txt"))
        .expect("the repaired call should have written the file");
    assert_eq!(written, "hello");
}

#[test]
fn a_tool_named_by_a_different_word_is_repaired() {
    // `cat` is not a misspelling of `read_file`; it is a different word for it.
    // No amount of normalization finds that, which is what the alias table is
    // for. The observable is the file's contents coming back to the model.
    let model = MockCompletionModel::from_stream_turns(vec![
        call("call_1", "cat", serde_json::json!({ "path": "Cargo.toml" })),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read the manifest");

    assert_eq!(rt.repair_report().tool_name_repairs, 1);
    assert!(
        sent(&model).contains("fixture"),
        "the repaired call should have read the manifest back to the model"
    );
}

#[test]
fn a_tool_that_is_not_registered_at_all_is_reported_to_the_model() {
    // Nothing in the alias table means `frobnicate`, so the model is told what
    // the tools are rather than the run ending. The second turn is the model
    // taking the correction.
    let model = MockCompletionModel::from_stream_turns(vec![
        call("call_1", "frobnicate", serde_json::json!({})),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "do something");

    assert_eq!(
        rt.repair_report().tool_name_repairs,
        0,
        "an unrepairable name is not a repair"
    );
    assert_eq!(
        app.calls, 2,
        "the model should have been given a second turn"
    );
    assert!(
        sent(&model).contains("read_file"),
        "the model should be told which tools exist"
    );
}

#[test]
fn arguments_wrapped_in_a_fence_are_salvaged() {
    // The model wrote the object correctly and then wrapped it in a fenced
    // block. The object is in there, so the hook hands it back and the model
    // re-sends the call. The assertion is the manifest coming back to the
    // model on the third turn: the salvage has to have produced arguments the
    // tool could actually run with, not merely a non-zero counter.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!("```json\n{\"path\": \"Cargo.toml\"}\n```"),
        ),
        call(
            "call_2",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read the manifest");

    assert_eq!(rt.repair_report().argument_salvages, 1);
    assert!(
        sent(&model).contains("fixture"),
        "the salvaged call should have read the manifest back to the model"
    );
}

#[test]
fn arguments_with_prose_around_the_object_are_salvaged() {
    // The same failure with a sentence in front of it, which is what a model
    // that has been asked to explain itself produces.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!("Here is the call: {\"path\": \"Cargo.toml\"} — running it now."),
        ),
        call(
            "call_2",
            "read_file",
            serde_json::json!({ "path": "Cargo.toml" }),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read the manifest");

    assert_eq!(rt.repair_report().argument_salvages, 1);
    assert!(
        sent(&model).contains("fixture"),
        "the salvaged call should have read the manifest back to the model"
    );
}

#[test]
fn a_string_encoded_object_needs_no_repair() {
    // The boundary of the salvage, pinned so a later change to it is a
    // deliberate one. rig's `ToolFunction::parse` already unwraps a string
    // holding an object, and unwraps a string holding *that* as well, so both
    // of these shapes are dispatched without the hook being consulted at all.
    // The salvage's own unwrapping is therefore only reachable through the
    // streaming path, where a cut-off argument buffer bypasses `parse`; that
    // path is covered by the unit tests beside the function.
    for (name, arguments) in [
        (
            "a string holding the object",
            serde_json::Value::String(serde_json::json!({ "path": "Cargo.toml" }).to_string()),
        ),
        (
            "a string holding a string holding the object",
            serde_json::Value::String(
                serde_json::Value::String(serde_json::json!({ "path": "Cargo.toml" }).to_string())
                    .to_string(),
            ),
        ),
    ] {
        let model = MockCompletionModel::from_stream_turns(vec![
            call("call_1", "read_file", arguments),
            turn("done"),
        ]);
        let rt = common::session_over_a_crate();
        let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

        run_to_idle(&mut app, &mut rx, "read the manifest");

        assert_eq!(rt.repair_report().argument_salvages, 0, "{name}");
        assert!(
            sent(&model).contains("fixture"),
            "{name} should have run without any repair"
        );
    }
}

#[test]
fn arguments_that_cannot_be_salvaged_are_reported_to_the_model() {
    // Prose with no object in it. There is nothing to hand back, so the model
    // is told what the tool needs instead of the run ending.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "read_file",
            serde_json::json!("the manifest, please"),
        ),
        turn("done"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model.clone(), &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read the manifest");

    assert_eq!(rt.repair_report().argument_salvages, 0);
    assert_eq!(
        app.calls, 2,
        "the model should have been given a second turn"
    );
    assert!(
        sent(&model).contains("JSON object"),
        "the model should be told what the arguments have to be"
    );
}

#[test]
fn the_repair_cap_stops_a_model_that_keeps_misnaming_tools() {
    // Six bad calls, each naming a different file so the app's own
    // repeated-call guard does not fire first. The hook repairs the first four
    // and then stops, so the run ends with a message rather than spending the
    // whole turn budget on the same mistake.
    let paths = [
        "Cargo.toml",
        "src/lib.rs",
        "Cargo.toml",
        "src/lib.rs",
        "Cargo.toml",
        "src/lib.rs",
    ];
    let mut turns: Vec<Vec<MockStreamEvent>> = paths
        .iter()
        .enumerate()
        .map(|(i, path)| {
            call(
                &format!("call_{i}"),
                "readFile",
                serde_json::json!({ "path": path }),
            )
        })
        .collect();
    turns.push(turn("done"));
    let model = MockCompletionModel::from_stream_turns(turns);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model, &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "read the manifest");

    assert_eq!(
        rt.repair_report().tool_name_repairs,
        4,
        "the cap is four repairs per run"
    );
    assert!(
        app.calls < 8,
        "the run should have ended rather than replaying every bad call: {} calls",
        app.calls
    );
}

#[test]
fn the_repair_budget_is_per_run_not_per_session() {
    // The counters live in the hook's scratchpad, which rig scopes to the run.
    // A second prompt therefore gets a fresh budget: a cap that leaked across
    // prompts would make the second prompt of a session behave differently
    // from the first, which is the kind of bug that only shows up in use.
    let model = MockCompletionModel::from_stream_turns(vec![
        call(
            "call_1",
            "writeFile",
            serde_json::json!({ "path": "one.txt", "content": "1" }),
        ),
        turn("first"),
        call(
            "call_2",
            "writeFile",
            serde_json::json!({ "path": "two.txt", "content": "2" }),
        ),
        turn("second"),
    ]);
    let rt = common::session_over_a_crate();
    let (mut app, mut rx) = App::new(common::agents(model, &rt), rt.clone());

    run_to_idle(&mut app, &mut rx, "write one");
    run_to_idle(&mut app, &mut rx, "write two");

    assert_eq!(
        rt.repair_report().tool_name_repairs,
        2,
        "each prompt should get its own repair budget"
    );
    assert!(rt.workspace().root().join("one.txt").exists());
    assert!(rt.workspace().root().join("two.txt").exists());
}
