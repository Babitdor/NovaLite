//! Score a model against a set of tasks.
//!
//! `examples/live.rs` drives one prompt and prints what happened. This is that
//! file generalized: it runs every task in `evals/tasks.json`, scores each one
//! against what the task says a passing run looks like, and prints a row per
//! task plus one JSON line per task so two runs can be diffed.
//!
//! It exists because the questions this program's changes raise are empirical.
//! Whether a terse system prompt beats the full one, whether temperature 0.1
//! beats 0.0, whether a given 7B model is usable at all: none of those is
//! settled by reading the code, and all of them are settled by running the same
//! tasks twice and comparing the numbers.
//!
//! Each task gets its own workspace, copied from its fixture, and its own
//! `Runtime`. The runtime is per task rather than per sweep because the repair
//! counters are session-scoped: a counter that carried over from the previous
//! task would attribute one task's failures to the next.
//!
//! ```text
//! cargo run --example eval
//! LLM_PROVIDER=ollama LLM_MODEL=qwen2.5-coder:7b LLM_PROMPT_STYLE=full \
//!   cargo run --example eval
//! ```
// The lint restrictions in `Cargo.toml` are for production code; in an example
// a panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rust_agent::app::{ActivityKind, App};
use rust_agent::config::Config;
use rust_agent::event::Event;
use rust_agent::runtime::Runtime;
use rust_agent::workspace::Workspace;
use serde::Deserialize;

/// How long one task may take before it is abandoned.
///
/// Generous, because a local model on a CPU is slow and a task that is making
/// progress should be allowed to finish. It is a backstop for the harness
/// itself, not a measurement: the idle timeout in the session is what catches
/// a stalled provider, and this only catches a task that never ends.
const TASK_DEADLINE: Duration = Duration::from_secs(600);

/// One task, as `evals/tasks.json` describes it.
#[derive(Debug, Deserialize)]
struct Task {
    /// The name the row and the JSON line are labelled with.
    name: String,
    /// What the model is asked to do.
    prompt: String,
    /// A directory copied into the task's workspace before the run.
    ///
    /// Relative to the repository root. `None` means the task runs in an empty
    /// directory, which is what a task that only writes new files wants.
    #[serde(default)]
    fixture: Option<String>,
    /// What a passing run looks like.
    #[serde(default)]
    expect: Expect,
}

/// The conditions a run has to meet to pass.
#[derive(Debug, Default, Deserialize)]
struct Expect {
    /// For each path, substrings that must appear in the file afterwards.
    ///
    /// Relative to the task's workspace. A file that does not exist fails the
    /// check, which is the point: a task that asked for an edit and got none
    /// should not pass because there was nothing to contradict.
    #[serde(default)]
    files_contain: BTreeMap<String, Vec<String>>,
    /// Substrings that must appear in a tool result.
    ///
    /// A tool result, not the model's message: a task that asks for the tests
    /// to be run is proved by the test output, and a model that never ran them
    /// can still claim they passed. Searching the prose would score that claim
    /// as a pass.
    #[serde(default)]
    output_contains: Vec<String>,
}

/// How a run ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The model finished its turn.
    Completed,
    /// The session stopped the run: a repeated call, a repeated failure, an
    /// exhausted correction budget.
    Stopped,
    /// The provider or the harness failed.
    Errored,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Outcome::Completed => "completed",
            Outcome::Stopped => "stopped",
            Outcome::Errored => "errored",
        }
    }
}

/// What one task produced.
struct Report {
    name: String,
    outcome: Outcome,
    calls: usize,
    input_tokens: u64,
    output_tokens: u64,
    tool_errors: usize,
    repairs: rust_agent::runtime::RepairReport,
    /// Why the task failed, empty when it passed.
    failures: Vec<String>,
    /// What the model said last, for a run that failed.
    answer: String,
}

impl Report {
    fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

fn main() -> anyhow::Result<()> {
    // The same logging setup as the binary, so a run that goes wrong leaves
    // the rig INFO lines in the log file rather than on the terminal.
    let _log_guard = rust_agent::logging::init_logging()?;

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "evals/tasks.json".to_string());
    let tasks = load_tasks(Path::new(&path))?;
    let config = Config::from_env();

    // The settings the numbers are only comparable within. Two sweeps with
    // different models or prompts are two different experiments, and a row
    // that did not say which one it came from would be unreadable a day later.
    println!(
        "model={} provider={} prompt={:?} temperature={} parallel_tool_calls={}",
        config.model,
        config.provider,
        config.prompt_style,
        config.temperature,
        config.parallel_tool_calls,
    );
    println!(
        "window={} compact_at={} max_turns={} idle_timeout={:?}",
        config.context.window,
        config.context.compact_at,
        config.limits.max_turns,
        config.stream_idle_timeout,
    );
    // The endpoint is read by the provider arm rather than resolved into the
    // configuration, so it is reported from the environment. A sweep against
    // the wrong daemon is otherwise indistinguishable from a sweep against a
    // broken one.
    println!(
        "ollama_base_url={} llm_base_url={}",
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://localhost:11434".to_string()),
        std::env::var("LLM_BASE_URL").unwrap_or_else(|_| "(unset)".to_string()),
    );
    println!();

    let mut reports = Vec::new();
    for task in &tasks {
        let report = run_task(task, &config);
        print_row(&report);
        reports.push(report);
    }

    println!();
    for report in &reports {
        println!("{}", json_line(report));
    }

    // The score says a task failed; only the text says why. Printed after the
    // machine-readable lines so a sweep can still be diffed.
    for report in reports.iter().filter(|r| !r.passed()) {
        println!();
        println!("--- {} ---", report.name);
        for failure in &report.failures {
            println!("  {failure}");
        }
        if !report.answer.is_empty() {
            println!("  the model said: {}", report.answer);
        }
    }

    let passed = reports.iter().filter(|r| r.passed()).count();
    println!();
    println!("{passed}/{} tasks passed", reports.len());

    // A non-zero exit is what makes this usable from a script: a sweep that
    // regressed should fail the command, not just print a different number.
    if passed < reports.len() {
        std::process::exit(1);
    }
    Ok(())
}

/// Read and parse the task file.
fn load_tasks(path: &Path) -> anyhow::Result<Vec<Task>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("could not read {}: {e}", path.display()))?;
    let tasks: Vec<Task> = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("could not parse {}: {e}", path.display()))?;
    if tasks.is_empty() {
        anyhow::bail!("{} lists no tasks", path.display());
    }
    Ok(tasks)
}

/// Run one task in its own workspace and score it.
fn run_task(task: &Task, config: &Config) -> Report {
    let dir = match tempfile::TempDir::new() {
        Ok(dir) => dir,
        Err(e) => return failed(task, format!("could not make a workspace: {e}")),
    };

    if let Some(fixture) = &task.fixture {
        let from = PathBuf::from(fixture);
        if let Err(e) = copy_tree(&from, dir.path()) {
            return failed(task, format!("could not copy {}: {e}", from.display()));
        }
    }

    let workspace = match Workspace::new(dir.path()) {
        Ok(workspace) => workspace,
        Err(e) => return failed(task, format!("the workspace is not valid: {e}")),
    };

    // The same two gates the binary goes through, in the same order. A harness
    // that built the agent directly would be a second, unguarded way into the
    // tools, and the point of the harness is that it measures the path the
    // binary takes.
    let rt = Runtime::new(config.clone(), workspace);
    rt.set_approver(Arc::new(rust_agent::approval::AutoApprover));
    let agents = match rust_agent::agent::create_agents(&rt) {
        Ok(agents) => agents,
        Err(e) => return failed(task, format!("could not build the agents: {e}")),
    };
    let (mut app, mut rx) = App::new(agents, rt.clone());

    let outcome = drive(&mut app, &mut rx, &task.prompt);

    let mut failures = Vec::new();
    if outcome == Outcome::Errored {
        failures.push("the run errored".to_string());
    }
    check_files(dir.path(), &task.expect, &mut failures);
    check_output(&app, &task.expect, &mut failures);

    Report {
        name: task.name.clone(),
        outcome,
        calls: app.calls,
        input_tokens: app.usage_in,
        output_tokens: app.usage_out,
        // A retry is recorded as an error activity, and it is not one: the
        // model was asked again and the run continued. Counting it here would
        // make a model that needed two retries look like a model whose tools
        // failed, which is the opposite conclusion.
        tool_errors: app
            .activity
            .iter()
            .filter(|a| a.is_error && a.kind != ActivityKind::Retry)
            .count(),
        repairs: rt.repair_report(),
        failures,
        answer: last_answer(&app),
    }
}

/// What the model said last, for a run that failed.
///
/// A score alone does not say whether the model refused, answered from
/// memory, or wrote a tool call into its message. Those call for different
/// fixes, and the difference is only visible in the text.
///
/// A run the session stopped has no assistant message to show: the reason is
/// pushed as a system row, and the terminal `Cancelled` event pushes a generic
/// "Run cancelled." after it. Reporting only the last row would therefore say
/// "Run cancelled." and hide the guard that fired, which is the one thing worth
/// knowing, so the last few system rows are reported together.
fn last_answer(app: &App) -> String {
    let assistant = app
        .messages
        .iter()
        .rev()
        .find(|m| m.role == rust_agent::app::Role::Assistant)
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let text = if assistant.trim().is_empty() {
        let system: Vec<&str> = app
            .messages
            .iter()
            .filter(|m| m.role == rust_agent::app::Role::System)
            .map(|m| m.content.as_str())
            .collect();
        system[system.len().saturating_sub(3)..].join(" | ")
    } else {
        assistant
    };
    let text = text.trim();
    if text.chars().count() <= 300 {
        return text.to_string();
    }
    let head: String = text.chars().take(300).collect();
    format!("{head}…")
}

/// A report for a task that could not be started at all.
fn failed(task: &Task, why: String) -> Report {
    Report {
        name: task.name.clone(),
        outcome: Outcome::Errored,
        calls: 0,
        input_tokens: 0,
        output_tokens: 0,
        tool_errors: 0,
        repairs: rust_agent::runtime::RepairReport::default(),
        failures: vec![why],
        answer: String::new(),
    }
}

/// Send the prompt and pump the event channel until the run ends.
fn drive(app: &mut App, rx: &mut tokio::sync::mpsc::Receiver<Event>, prompt: &str) -> Outcome {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("a tokio runtime");

    rt.block_on(async {
        app.send_message(prompt.to_string());
        let deadline = tokio::time::Instant::now() + TASK_DEADLINE;
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    // Every way the run can end, not just the happy one. A run
                    // the session stopped ends as `Cancelled`, and waiting for
                    // a `Finished` that will never come would report a timeout
                    // for a run that stopped cleanly.
                    let outcome = match &event {
                        Event::Finished(_) => Some(Outcome::Completed),
                        Event::Cancelled => Some(Outcome::Stopped),
                        Event::Error(text) => {
                            eprintln!("error: {text}");
                            Some(Outcome::Errored)
                        }
                        Event::StreamEnded => {
                            eprintln!("error: the stream ended without a response");
                            Some(Outcome::Errored)
                        }
                        Event::TaskFailed(text) => {
                            eprintln!("error: {text}");
                            Some(Outcome::Errored)
                        }
                        _ => None,
                    };
                    app.handle_event(event);
                    if let Some(outcome) = outcome {
                        return outcome;
                    }
                }
                Ok(None) => return Outcome::Errored,
                Err(_) => {
                    eprintln!("{}: no terminal event within {TASK_DEADLINE:?}", app.status);
                    return Outcome::Errored;
                }
            }
        }
    })
}

/// Check every `files_contain` entry against the workspace.
fn check_files(root: &Path, expect: &Expect, failures: &mut Vec<String>) {
    for (path, wanted) in &expect.files_contain {
        let full = root.join(path);
        let text = match std::fs::read_to_string(&full) {
            Ok(text) => text,
            Err(e) => {
                failures.push(format!("{path} could not be read: {e}"));
                continue;
            }
        };
        for needle in wanted {
            if !text.contains(needle.as_str()) {
                failures.push(format!("{path} does not contain {needle:?}"));
            }
        }
    }
}

/// Check every `output_contains` entry against the tool results.
///
/// Tool results only, never the model's prose. The check exists to prove the
/// work was *done*, and a message cannot prove that: a run that called no tool
/// at all can still write "test result: ok" into its reply, and a check that
/// read the message bodies scored exactly that as a pass. Every task here is
/// proved by a tool result, so that is what is searched.
///
/// The narrower check is also the one the tasks were written for: the output of
/// `cargo test` is a tool result, not something the model wrote, so a run that
/// fixed the bug and ran the tests still passes.
fn check_output(app: &App, expect: &Expect, failures: &mut Vec<String>) {
    if expect.output_contains.is_empty() {
        return;
    }
    let evidence = app
        .messages
        .iter()
        .filter_map(|m| m.tool.as_ref().map(|block| block.result.as_str()))
        .collect::<Vec<_>>()
        .join("\n");
    for needle in &expect.output_contains {
        if !evidence.contains(needle.as_str()) {
            failures.push(format!("no tool result contains {needle:?}"));
        }
    }
}

/// Copy the tree at `from` into `to`, creating directories as needed.
fn copy_tree(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// One aligned row per task.
fn print_row(report: &Report) {
    let repairs = report.repairs;
    println!(
        "{:<18} {:<10} calls={:<3} in={:<7} out={:<6} errors={:<3} \
         repairs={}/{}/{}/{} {}",
        report.name,
        report.outcome.as_str(),
        report.calls,
        report.input_tokens,
        report.output_tokens,
        report.tool_errors,
        repairs.tool_name_repairs,
        repairs.argument_salvages,
        repairs.prose_tool_call_retries,
        repairs.stream_timeouts,
        if report.passed() { "PASS" } else { "FAIL" },
    );
    for failure in &report.failures {
        println!("{:18}   - {failure}", "");
    }
}

/// One JSON object per task, so two sweeps diff cleanly.
fn json_line(report: &Report) -> String {
    let repairs = report.repairs;
    let value = serde_json::json!({
        "task": report.name,
        "outcome": report.outcome.as_str(),
        "calls": report.calls,
        "input_tokens": report.input_tokens,
        "output_tokens": report.output_tokens,
        "tool_errors": report.tool_errors,
        "tool_name_repairs": repairs.tool_name_repairs,
        "argument_salvages": repairs.argument_salvages,
        "prose_tool_call_retries": repairs.prose_tool_call_retries,
        "stream_timeouts": repairs.stream_timeouts,
        "passed": report.passed(),
        "failures": report.failures,
    });
    value.to_string()
}
