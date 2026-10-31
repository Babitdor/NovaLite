# Rust-Based-Agents

A small coding agent that runs in the terminal, built on
[Rig](https://crates.io/crates/rig) with a
[ratatui](https://github.com/ratatui/ratatui) front end.

The binary is `nova-lite`. It reads and writes files, runs shell commands, and
streams the model's answer into a scrollable transcript, all inside one
directory that you approve on first run.

## Quick start

```bash
# Build
cargo build --release

# Run it in the current directory
./target/release/nova-lite

# Or name the directory the agent may touch
./target/release/nova-lite path/to/project
```

The first run in a directory asks whether the agent may work there, before
anything is read or written. The answer is recorded per directory, so it is
asked once. That is separate from the per-action approval, which is asked every
time: see [Approval](#approval).

## Command line

```text
nova-lite [OPTIONS] [DIRECTORY]

  -h, --help      print the usage and exit
  -V, --version   print the version and exit
  --auto-approve  do not ask before each write or command (alias: --yolo)
  --              end the options, so a directory starting with `-` can be named
```

An unrecognised option is an error rather than a directory name, and naming two
directories is an error too. `nova-lite --help` prints the full usage, including
the key map and the environment.

`nova-lite` is an interactive TUI and needs a terminal. With stdin or stdout
redirected it refuses to start rather than painting escape codes into the file
and then waiting for a key that can never arrive:

```text
$ nova-lite > out.txt
Error: nova-lite is an interactive TUI and needs a terminal, but stdin or stdout is redirected.
```

For a non-interactive run, use `cargo run --example live -- "prompt"`, which
drives one prompt and prints the stream.

## Keys

| Key | What it does |
| --- | --- |
| `Enter` | send the prompt |
| `Shift+Enter` | insert a newline |
| `Esc`, `Ctrl+C` | stop the run in flight; again to quit |
| `Up`, `Down` | previous prompts |
| `PageUp`, `PageDown` | scroll the transcript (`Ctrl+Up`/`Ctrl+Down` one line) |
| `Ctrl+Home`, `Ctrl+End` | the transcript's top / back to following the tail |
| `Left`, `Right` | move the caret (`Ctrl+Left`/`Ctrl+Right` by word) |
| `Home`, `End` | the caret's line |
| `Delete` | remove the character under the caret |
| `Ctrl+A`, `Ctrl+E` | line start / line end |
| `Ctrl+U`, `Ctrl+K` | kill to line start / line end |
| `Ctrl+W` | delete the word before the caret |
| `Ctrl+T`, `Ctrl+Shift+T` | fold/unfold the newest tool block, or all of them |
| `Ctrl+G`, `Ctrl+Shift+G` | fold/unfold the newest tool group, or all of them |
| `Tab` | complete a `/command` |
| `/help` | the full list, including the tools |

A prompt sent while a run is in flight is queued rather than dropped, and the
queue is shown in the transcript.

## Commands

| Command | What it does |
| --- | --- |
| `/help` | the full help, including the tool list |
| `/model` | pick a model from the local Ollama daemon |
| `/compact` | summarise the conversation and continue from the summary |
| `/clear` | empty the transcript |
| `/exit`, `/quit` | leave |

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama`, `anthropic`, or `openai` |
| `LLM_MODEL` | `deepseek-v4.1-flash:cloud` | the model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | the daemon to talk to |
| `LLM_BASE_URL` | unset | OpenAI-compatible endpoint, e.g. `http://localhost:8080/v1` |
| `LLM_NUM_CTX` | provider default | context window to request, in tokens |
| `LLM_TEMPERATURE` | `0.1` | sampling temperature; `0.0`–`2.0` |
| `LLM_PARALLEL_TOOL_CALLS` | `false` | let the model emit several tool calls in one turn; ignored by `ollama` |
| `LLM_PROMPT_STYLE` | `terse` | `terse` or `full` system prompt |
| `LLM_STREAM_IDLE_TIMEOUT_SECS` | `300` | how long a model stream may go quiet before it is abandoned |
| `LLM_MAX_TURNS` | `100` | model calls allowed per prompt |
| `LLM_COMPACT_AT` | `80` | context percentage that triggers compaction |
| `LLM_COMPACT_TAIL_CHARS` | window / 8 × 4 | characters of recent turns kept verbatim across a compaction |
| `LLM_TOOL_RESULT_CHARS` | window / 8 × 4 | character budget for one tool result |
| `LLM_EVICT_CHARS` | window / 4 × 4 | character budget for the tool results in one request |
| `LLM_OUTPUT_RESERVATION` | window / 8 | tokens held back for the model's own answer — the request's `max_tokens`, taken out of the window as well |
| `LLM_SAFETY_MARGIN` | `1024` | tokens held back for the provider's framing |
| `NOVA_APPROVAL` | `prompt` | `auto` to skip the per-action prompt, as `--auto-approve` |
| `NOVA_COMMAND_TIMEOUT_SECS` | `120` | how long a command may run before its tree is killed |
| `RUST_LOG` | `info` | log filter; logs go to a file, never the terminal |

`LLM_PROVIDER=anthropic` needs `ANTHROPIC_API_KEY`; `openai` needs
`OPENAI_API_KEY`. A missing or blank key is an error naming the variable, not a
panic, and an unrecognised provider is an error rather than a silent fallback to
Ollama.

`LLM_BASE_URL` points the `openai` provider at a local server that speaks the
same wire format, which is how llama.cpp, vLLM and LM Studio are used:

```bash
LLM_PROVIDER=openai LLM_BASE_URL=http://localhost:8080/v1 \
  LLM_MODEL=qwen2.5-coder-7b nova-lite
```

Those servers take no credential, so `OPENAI_API_KEY` becomes optional once a
base URL is set and a placeholder is sent in its place. Without a base URL the
key stays mandatory, because the request really is going to api.openai.com. The
`ollama` provider is unaffected: it uses Ollama's own `/api/chat` route, which
is what makes `LLM_NUM_CTX` mean anything there.

`LLM_STREAM_IDLE_TIMEOUT_SECS` bounds the gap *between* stream items, not the
length of a run, so a slow model is left alone and a silent one is abandoned
with an error rather than leaving the UI on a spinner. It is deliberately well
above `NOVA_COMMAND_TIMEOUT_SECS`: a tool runs inside the stream, so a command
that runs to its own limit is that many seconds of stream silence, and a
timeout at the same figure would race it.

`LLM_PARALLEL_TOOL_CALLS` is not sent to Ollama. Its native `/api/chat` route
has no such field, and rig refuses a request carrying an option the provider
cannot express rather than dropping it, so setting it there fails every request
outright. The daemon decides for itself in that case, which is what it did
before the knob existed. The setting is honoured for `openai` and `anthropic`.

The three budget variables that default to a fraction of the window are
resolved from `LLM_NUM_CTX`, so raising the window raises them with it. The
`× 4` is the characters-per-token estimate: it is a conservative fallback, not
an exact count, and the safety margin is what covers the difference.

`LLM_MAX_TURNS` counts *model calls*, not tool calls: a tool round-trip costs
two, one to emit the call and one to consume the result. The default of 100 is
about fifty tool calls, which is where a task that explores, edits, runs the
tests and then fixes what they report finishes. A run that reaches the ceiling
is abandoned mid-task, so the budget is set where a legitimate task ends rather
than where a pathological one is caught; the guards in "Stopping a run that is
not converging" are what catch the latter.

`LLM_TEMPERATURE` is the knob that decides whether a small model emits
parseable tool-call JSON at all. Left unset, the daemon's own default applies,
and Ollama's is 0.8: at that temperature a 7B model will happily produce a
plausible-looking but unparseable argument object, and every such turn costs
two model calls to recover. `0.1` is the default here because coding work wants
the most likely token, not a creative one. Raise it only if a model is looping
on the same wrong answer.

`LLM_PARALLEL_TOOL_CALLS` is off by default. The UI pairs results by call id, so
serial calls cost only a little speed, and small models are markedly worse at
emitting several well-formed calls in one turn. Turn it on for a large model
where the round-trip latency dominates.

`LLM_NUM_CTX` matters on a small GPU. The Ollama daemon picks its own default
from the card's VRAM (4096 tokens on 6 GB), which a multi-step run overruns; the
agent asks for 256k explicitly on the daemon's native `/api/chat` route, which is
the only route that reads the option. The window is what every other budget is a
fraction of, so a small one costs the agent its memory of the run; a local model
on a small card can be held to a smaller window with `LLM_NUM_CTX`.

Two budgets keep a long run inside that window. A single tool result is elided
to `LLM_TOOL_RESULT_CHARS` (its head, a marker, and its tail), and the results
in one request are bounded by `LLM_EVICT_CHARS`: the history is re-sent on every
model call, so a run that makes many calls grows the prompt by a result per
call, and once the budget is full the older results are replaced by a
placeholder that names the tool and says to call it again. The result of the
call just made is always sent whole, and the full results stay in the
transcript.

```bash
# Local daemon
LLM_MODEL=qwen3-coder:30b nova-lite

# Anthropic
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... LLM_MODEL=claude-sonnet-4-5 nova-lite
```

## Tools

| Tool | Description |
| --- | --- |
| `read_file` | read a file, or a line range of one |
| `write_file` | create a file, or replace one whole |
| `edit_file` | apply ordered search-and-replace blocks to a file |
| `list_dir` | list the entries in a directory |
| `grep` | search file contents recursively |
| `glob` | find files matching a pattern |
| `shell` | run a shell command and return stdout and stderr |

`edit_file` takes a list of `{search, replace}` blocks and applies them in
order to an in-memory copy of the file. Every block is validated before
anything is written, so a block that fails at the end leaves the file exactly
as it was. A search string must be non-empty and must match exactly once: a
missing match and an ambiguous one are both errors that name the problem,
because silently replacing the first of several matches is how an edit lands
in the wrong place. The file's version is checked against the one the model
last read, so an edit computed from a stale read is refused rather than
applied to content that has since changed.

`grep` and `glob` share one walker, which reads `.gitignore`, `.ignore` and the
global ignore file, and skips hidden files by default. Both take `hidden` and
`ignored` flags to override that. The walk is bounded: at most 20,000 files
visited, 1,000 matches, 4 MiB retained, 2 MiB per file, and 30 seconds. When a
bound is reached the result says so, so a truncated answer is never mistaken
for a complete one.

### Limits

A tool call is bounded so that one of them cannot take the process down:

- `read_file` refuses a directory, and reads at most the first 4 MiB of a file,
  saying so when it truncates. The result is elided to the model's budget
  anyway, so pulling a multi-gigabyte file into memory only to discard it is
  pure risk: the allocation is what fails, and an allocation failure aborts the
  process rather than failing the tool. `start_line` and `end_line` read a
  range, which is how the model gets back a part of a file that was elided out
  of an earlier result.
- `edit_file` refuses an empty search string. Every string contains the empty
  string, so the replacement would be prepended to the file rather than
  replacing anything.
- `shell` keeps at most 1 MiB of a command's output, and kills the whole
  process tree if the command is still running after 120 seconds. Killing only
  the shell would leave the command it started holding the pipes open. Both
  ends of the output are kept, because a compiler prints its diagnostics last
  and a head-only bound would show the progress lines and hide the errors. The
  notice left where the middle was cut out says how much was dropped.
- `write_file` and `edit_file` write through a sibling temp file and rename it
  over the target, so a failure partway through leaves the original rather than
  a truncated file. There is no fallback to a plain write: if the rename is
  refused, the original is left alone and the failure is reported. A write that
  silently degrades to truncating the file it was supposed to replace is worse
  than a write that fails.

## Approval

Two different questions, asked at two different times, and it is worth keeping
them apart:

- **Trust** is asked once per directory, on the first run in it, and the answer
  is recorded outside the workspace. It says "this directory is one I work in".
- **Approval** is asked before every write, every destructive action and every
  terminal command, and the answer is never recorded. It says "do this, now".

Before an action runs, the prompt shows what it will do: the operation and the
workspace-relative paths, a diff for a change to an existing file or a preview
for a new one, and for a command the exact command line, its working directory
and its timeout. Only an explicit `y` or `yes` approves. Enter, Escape,
end-of-input, a closed channel and a cancellation all deny, so a prompt nobody
answers leaves the gate shut rather than open.

The approval is bound to the exact bytes that were previewed. If the file
changes between the preview and the write, the approval no longer applies and a
new preview is required, which is what stops a preview from being a licence to
write something else.

`--auto-approve` (alias `--yolo`) skips the question. It does not skip the
workspace boundary, the command timeout, the output bounds or any other limit:
it skips the prompt, not the boundary. A denied action comes back to the model
as a result it can read, and the model is told not to repeat it.

## Cancellation

Escape or Ctrl+C stops the run in flight. What that means is defined around the
commit point of each action, because that is the only place where "stopped" and
"done" are different things:

- A command that is running is killed **with its descendants**. Killing the
  shell alone would leave the compiler it started holding the pipes open, and
  the run would hang on a process nobody is waiting for. On Windows the child
  is put in a Job Object that kills the whole tree when it closes; on Unix it
  gets its own process group and the group is signalled.
- A write that has not committed yet is abandoned. The cancellation is checked
  before the file is touched, so there is no half-written file to clean up and
  no temp file left behind.
- A write that has committed is recorded. The journal is written as each action
  completes, not at the end of the run, so an interrupted run still knows what
  it already did. That is what stops a resumed run from repeating an edit it
  has already made.
- The terminal is restored, and the background tasks are given a bounded grace
  period to finish before the process exits.

A cancelled run is reported as cancelled, not as a failure and not as a
success. A tool result that was cut short is marked as such in the transcript,
so a truncated answer is not read as a complete one.

## Stopping a run that is not converging

A model can spend a whole turn budget without getting anywhere, and the
transcript is the only place that shows it. Four bounds stop that, and each one
says which it was:

- **The turn budget.** `LLM_MAX_TURNS` (default 100) caps the model calls in one
  prompt. It is the backstop, and it is deliberately generous: a run that
  explores a tree before it answers is not a loop.
- **A repeated call.** The same tool with the same arguments twice in a row
  stops the run. A tool's result is a function of its arguments, so the second
  call cannot tell the model anything the first did not, and a call that
  *succeeds* is invisible to every other check here. This is the guard for the
  loop that looks like progress.
- **A repeated failure.** The same error text twice in a row stops the run. The
  second identical failure is the evidence that the first correction did not
  address the cause.
- **The correction budget.** Three changes made after a failed check, and the
  run stops. A change made before any check is the first attempt, not a
  correction, and does not spend the budget.

A run stopped by one of these is reported as cancelled, with the reason in the
transcript, and whatever it changed is still on disk and still in the journal.

## Confinement

Every path a tool is given is resolved against the workspace root, and a path
that lands outside is refused before the filesystem is touched. A relative path
is joined to the root, so the tools do not depend on the process's working
directory; an absolute path is accepted only if it already points inside.

The check is not merely lexical. A path that exists is resolved to its real
filesystem target, so a symlink or a Windows junction pointing out of the
workspace is refused even though its name is inside. A path that does not exist
yet is resolved through its nearest existing ancestor, so a new file cannot be
created through a link either. Path forms that have no single meaning are
rejected outright rather than guessed at: drive-relative paths (`C:foo`),
UNC and device paths, and alternate data streams.

This is a guardrail, not a sandbox, and the difference matters. `shell` runs
with the root as its working directory but is otherwise free to name any path
it likes, so a model that wanted to read `~/.ssh/id_rsa` could do it with
`shell("type %USERPROFILE%\.ssh\id_rsa")`. Closing the symlink case while
`shell` stays open is not security theatre, because the file tools are the ones
the model reaches for by default and the ones whose arguments come from the
conversation; but it is not a boundary against a hostile model either. A real
boundary means sandboxing `shell` too, which is a different piece of work.

The trust store lives outside the workspace, in `%APPDATA%\nova-lite\trusted.json`
on Windows and the equivalent config directory elsewhere, so the tools it gates
cannot rewrite it.

## Examples

```bash
# Render a conversation as text, with no terminal and no model call
cargo run --example preview

# Drive a live provider and print the stream
cargo run --example live -- "list the files in src"
```

## Library

The binary is a thin shell over the library, so the same stream → event → render
path the binary runs is available to a caller:

```rust,no_run
use rust_agent::agent::create_agents;
use rust_agent::app::App;
use rust_agent::config::Config;
use rust_agent::runtime::Runtime;
use rust_agent::workspace::Workspace;

fn main() -> anyhow::Result<()> {
    // The workspace is a value, not a process-global: two of these in one
    // process do not see each other, which is what makes the tests
    // order-independent.
    let workspace = Workspace::new(std::env::current_dir()?)?;
    let rt = Runtime::new(Config::from_env(), workspace);
    let agents = create_agents(&rt)?;
    let (app, rx) = App::new(agents, rt);
    let _ = (app, rx);
    Ok(())
}
```

`Runtime` is the shared execution service: the workspace, the approval gate,
the process runner, the cancellation token and the action journal all live
behind it, and every tool takes one. There is deliberately no constructor that
builds a tool without one, so a caller cannot get an unapproved write path by
forgetting to wire something up.

`Config::defaults()` is the same configuration with no environment consulted,
which is what a test builds when it wants budgets it chose.

## Project structure

```text
src/
├── agent.rs      # agent builder, provider dispatch, the request-budget hook
├── app.rs        # application state: transcript, input, commands, runs
├── approval.rs   # the per-action gate: what is asked, and how it is answered
├── banner.rs     # the boot banner
├── cli.rs        # command-line parsing
├── config.rs     # every budget, timeout and limit, resolved once
├── context.rs    # request-size accounting and the pruning that keeps it in budget
├── error.rs      # the typed errors the library returns
├── event.rs      # the event type the stream is turned into
├── filesystem.rs # root-scoped file access: atomic replace, exclusive temp files
├── lib.rs        # crate root
├── logging.rs    # tracing setup (writes to a file, not the terminal)
├── main.rs       # entry point: arguments, trust, terminal, event loop
├── patch.rs      # ordered search-and-replace blocks, validated before commit
├── process.rs    # the one process runner: deadlines, output bounds, tree kill
├── prompts.rs    # the system and compaction prompts
├── render.rs     # every frame the TUI draws
├── runtime.rs    # the shared services: workspace, approval, process, journal
├── search.rs     # the bounded, ignore-aware walker behind grep and glob
├── session.rs    # what one run did: stage, changes, checks, outcome
├── tools.rs      # the seven tools
├── trust.rs      # the per-directory approval store
├── tui.rs        # terminal lifecycle, key handling, the event loop
└── workspace.rs  # the one directory the agent may touch

examples/
├── live.rs       # drive a live provider and print the stream
└── preview.rs    # render a conversation as text
```

## Development

```bash
cargo fmt --all -- --check
cargo check --all-targets --all-features --locked
cargo test --all-targets --all-features --locked
cargo test --doc --all-features --locked
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo build --release --locked
```

These are the gates CI runs, on Windows, Linux and macOS, plus a job that
builds with the minimum Rust version read out of `Cargo.toml`. The tests need
no credentials and no network: the provider, stream and render paths are driven
by `rig`'s mock model. The tests that do need a live provider are opt-in.

`unwrap`, `expect` and `panic` are denied in production code by the lint
configuration in `Cargo.toml`. A panic in a program that owns the terminal
leaves the user's shell in raw mode on the alternate screen, with no echo and
no cursor, so a new one cannot be merged by accident. The test modules opt out
explicitly, because in a test a panic *is* the assertion.

## Requirements

- Rust 1.95+ (edition 2021). The floor is not a guess: it is the highest
  `rust-version` declared anywhere in `Cargo.lock`, which is `rig`'s. It is
  declared in `Cargo.toml` as well, so an older toolchain is refused by cargo
  with a message naming the crate rather than failing somewhere inside a
  dependency.
- A model: a local [Ollama](https://ollama.com) daemon, or an Anthropic or
  OpenAI API key

## License

MIT
