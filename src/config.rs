//! The configuration, read once and then immutable.
//!
//! Every value the agent runs on is resolved here, at startup, from the
//! environment. Nothing downstream reads the environment again: a run's
//! budgets, timeouts and limits are fixed for the life of the session, so two
//! runs cannot disagree about them and a test can build a configuration
//! without touching process-global state.
//!
//! The pure resolvers (`parse_*`) are split out from the readers (`num_ctx`,
//! `max_turns`, …) for the same reason: `std::env::set_var` is process-global
//! and tests run in parallel, so the tests drive the parser directly rather
//! than the environment.

use std::time::Duration;

// ---------------------------------------------------------------------------
// Model-call budget
// ---------------------------------------------------------------------------

/// The default model-call budget for one prompt.
///
/// This is rig's `max_turns`, which counts *every* model call, not tool calls:
/// a tool round-trip costs two (one to emit the call, one to consume the
/// result). The old value of 10 was therefore about five tool calls, which a
/// model that starts by exploring the tree exhausts before it answers.
///
/// 40 was still only about twenty tool calls, and a real task overruns that:
/// explore, edit, run the tests, fix what they report, run them again. A run
/// that reaches the ceiling is abandoned mid-task, so the budget is set where
/// a legitimate task finishes rather than where a pathological one is caught.
/// Catching a pathological run is the job of the repeated-call and
/// repeated-failure guards, which stop a model going in circles; a model that
/// is working is a different case, and it should not be stopped for working.
pub const DEFAULT_MAX_TURNS: usize = 100;

/// The model-call budget for one prompt, from `LLM_MAX_TURNS`.
pub fn max_turns() -> usize {
    parse_max_turns(std::env::var("LLM_MAX_TURNS").ok().as_deref())
}

/// Resolve a raw `LLM_MAX_TURNS` value.
///
/// A value that is absent, unparseable, or zero falls back to
/// [`DEFAULT_MAX_TURNS`]: a zero budget would emit no model call at all and
/// fail every prompt, which is never what the user meant to ask for.
pub fn parse_max_turns(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_MAX_TURNS)
}

/// Whether `message` is a run that used up its model-call budget.
///
/// rig reports this as `PromptError::MaxTurns`, whose text is "reached the max
/// turns limit of N". It is matched on that phrasing rather than on a type
/// because the failure reaches the UI as a string: `Event` is `Clone`, so it
/// cannot carry the error itself.
///
/// Recognising it is worth doing because the raw text is a dead end. It names
/// the number but not the knob, so a user who hits it cannot raise the budget
/// without reading the README. The caller turns this into a message that names
/// `LLM_MAX_TURNS`.
pub fn is_turn_limit(message: &str) -> bool {
    message.to_ascii_lowercase().contains("max turns limit")
}

// ---------------------------------------------------------------------------
// Context window
// ---------------------------------------------------------------------------

/// The default context window, in tokens, asked of a local daemon.
///
/// The daemon's own default is chosen from the GPU's VRAM (`OLLAMA_CONTEXT_LENGTH`,
/// documented as "4k/32k/256k based on VRAM"), so a 6 GB card serves 4096
/// tokens. A coding agent blows through that in a handful of tool
/// round-trips, and the daemon then rejects the request outright with
/// "exceeds the available context size". Asking for a window explicitly is
/// what makes a multi-step run possible at all.
///
/// 256k because the window is what the whole session is budgeted against: the
/// compaction threshold, the per-result elision and the per-request eviction
/// are all fractions of it, so a small window makes the agent forget work it
/// has already done. The cost is that `num_ctx` is an allocation request
/// rather than a hint, so a local model on a small card pays for the larger
/// window in speed; `LLM_NUM_CTX` holds one to a smaller window.
///
/// Measured on a 6 GB card: a 0.8B model served 262144 entirely on the GPU,
/// and a 5.2 GB 8B model clamped the request to its own 128000 maximum and
/// served it with 43% of the work on the CPU. A large window on a small card
/// therefore costs speed, not correctness, and a model with a smaller maximum
/// simply gets its own.
///
/// This is the local default only; see [`default_window`] for the hosted
/// providers, whose windows are far larger and cost nothing to declare.
pub const DEFAULT_NUM_CTX: u32 = 262_144;

/// The context window a provider serves when `LLM_NUM_CTX` says nothing.
///
/// One global default was wrong at both ends. A hosted model with a 200k
/// window was budgeted at 32k, so every percentage the user was shown was
/// computed against a window six times too small, and compaction fired at a
/// sixth of the real limit. The local default is now the same 256k, because
/// the window is what every other budget is a fraction of and a small one
/// costs the agent its memory of the run. Hosted providers ignore the option
/// entirely, so declaring a large window there costs nothing.
///
/// A per-model-name table would be more precise and would go stale; the
/// provider default plus the `LLM_NUM_CTX` override covers the real cases.
pub fn default_window(provider: &str) -> u32 {
    match provider {
        "anthropic" => 200_000,
        "openai" => 128_000,
        _ => DEFAULT_NUM_CTX,
    }
}

/// Resolve a raw `LLM_NUM_CTX` value against `provider`'s default.
///
/// Absent, unparseable, or zero falls back to [`default_window`]: a zero
/// window would make every request fail, which is never what the user meant.
pub fn parse_num_ctx(raw: Option<&str>, provider: &str) -> u32 {
    raw.and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| default_window(provider))
}

/// The default context-usage percentage at which the conversation is
/// compacted.
///
/// 80%: the conversation is kept whole for as long as the window allows, and
/// the summarisation call still fits. That call sends the very history it is
/// summarising, so it needs headroom left in the window: at 80% of the 256k
/// default it sends ~209k and has ~52k left for the summary, which is ample
/// for the sub-400-word summary the prompt asks for. The fraction is what
/// matters, so the headroom scales with the window rather than being a fixed
/// number of tokens.
///
/// The cost of the higher figure is mid-run headroom. The threshold is only
/// checked once a run has settled, so a run that starts just under it and
/// then makes several large tool calls can still overrun the window before
/// compaction gets a chance to fire. `LLM_TOOL_RESULT_CHARS` (and the
/// window-derived default behind it) is what bounds that growth; lower this
/// value if a model still overruns.
pub const DEFAULT_COMPACT_AT: u8 = 80;

/// Resolve a raw `LLM_COMPACT_AT` value.
///
/// Absent, unparseable, or outside `10..=95` falls back to
/// [`DEFAULT_COMPACT_AT`]. Below 10 the conversation would be compacted almost
/// every turn, which costs a model call each time; above 95 there is no
/// headroom left for the summarisation call, which has to send the history it
/// is summarising.
pub fn parse_compact_at(raw: Option<&str>) -> u8 {
    raw.and_then(|v| v.trim().parse::<u8>().ok())
        .filter(|n| (10..=95).contains(n))
        .unwrap_or(DEFAULT_COMPACT_AT)
}

/// The fraction of the context window a single tool result may occupy.
///
/// A tool result is one message. Letting one take more than an eighth of the
/// window means a handful of calls fill it, and the daemon then rejects the
/// next request outright. Compaction cannot rescue that: it runs after a run
/// has settled, so a run that overruns mid-flight dies before it fires.
const TOOL_RESULT_WINDOW_FRACTION: u32 = 8;

/// Characters per token, for converting a window (tokens) into a budget
/// (characters). The usual figure for English prose and source code.
///
/// This is an estimate, not accounting: it is used to derive *budgets*, which
/// are deliberately conservative, and never to report a token count to the
/// user. See [`crate::context`] for the request-size accounting.
pub const CHARS_PER_TOKEN: usize = 4;

/// The character budget for one tool result, derived from the window.
///
/// A fixed cap would be wrong at both ends: 8,000 characters is 6% of a 32k
/// window but half of a 4,096 one, and 4,096 is what a 6 GB card serves by
/// default. Deriving it from the window keeps the fraction constant.
pub fn tool_result_budget_chars(window: u32) -> usize {
    (window as usize / TOOL_RESULT_WINDOW_FRACTION as usize) * CHARS_PER_TOKEN
}

/// Resolve a raw `LLM_TOOL_RESULT_CHARS` value.
///
/// `None` means "not configured, use the window-derived default". A zero or
/// unparseable value resolves the same way: a zero budget would elide every
/// result to nothing, which is never what the user meant to ask for.
pub fn parse_tool_result_chars(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// The fraction of the context window the tail kept across a compaction may
/// occupy.
///
/// Compaction replaces the evicted turns with a summary, but the most recent
/// turns are what the work is resting on: the request in flight, the file
/// just read, the error just seen. Summarising those away is what makes a
/// compacted agent ask the user to repeat themselves. A quarter of the window
/// holds a turn or two of real work and still leaves the summary, and the run
/// that follows it, room to grow.
const TAIL_WINDOW_FRACTION: u32 = 4;

/// The character budget for the tail kept across a compaction, derived from
/// the window.
pub fn tail_budget_chars(window: u32) -> usize {
    (window as usize / TAIL_WINDOW_FRACTION as usize) * CHARS_PER_TOKEN
}

/// Resolve a raw `LLM_COMPACT_TAIL_CHARS` value.
///
/// `None` means "not configured, use the window-derived default". A zero or
/// unparseable value resolves the same way: a zero budget would drop the tail
/// of every compaction, which is never what the user meant to ask for.
pub fn parse_tail_chars(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// The fraction of the context window the tool results in one request may
/// occupy before the older ones are elided.
///
/// A tool result is the largest thing a turn adds, and a run that makes many
/// calls accumulates them all: the history is re-sent on every call, so the
/// prompt grows by a result per call. Compaction cannot rescue that, because
/// it runs after a run has settled and a run that overruns dies before it
/// fires. Bounding the results per request is what keeps a long run alive.
const EVICTION_WINDOW_FRACTION: u32 = 4;

/// The character budget for the tool results in one request, derived from the
/// window.
pub fn eviction_budget_chars(window: u32) -> usize {
    (window as usize / EVICTION_WINDOW_FRACTION as usize) * CHARS_PER_TOKEN
}

/// Resolve a raw `LLM_EVICT_CHARS` value.
///
/// `None` means "not configured, use the window-derived default". A zero or
/// unparseable value resolves the same way: a zero budget would elide every
/// result the moment it was made, which is never what the user meant to ask
/// for.
pub fn parse_evict_chars(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// The model to use, read once from the environment so the status bar and
/// the agent construction can never disagree.
pub fn model_name() -> String {
    std::env::var("LLM_MODEL").unwrap_or_else(|_| "deepseek-v4.1-flash:cloud".to_string())
}

/// The provider to use, from `LLM_PROVIDER` (default `ollama`).
pub fn provider_name() -> String {
    parse_provider(std::env::var("LLM_PROVIDER").ok().as_deref())
}

/// Resolve a raw `LLM_PROVIDER` value.
///
/// Absent or blank falls back to `ollama`, and the name is lowercased so
/// `LLM_PROVIDER=OpenAI` is not a typo the user has to discover.
pub fn parse_provider(raw: Option<&str>) -> String {
    raw.map(str::trim)
        .filter(|v| !v.is_empty())
        .unwrap_or("ollama")
        .to_ascii_lowercase()
}

// ---------------------------------------------------------------------------
// The assembled configuration
// ---------------------------------------------------------------------------

/// How the agent asks before it changes anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalMode {
    /// Ask before every write, destructive action, or terminal command.
    Prompt,
    /// Run without asking. Path validation, timeouts and every other limit
    /// still apply: this skips the question, not the boundary.
    Auto,
}

impl ApprovalMode {
    /// Resolve a raw `NOVA_APPROVAL` value.
    ///
    /// Anything that is not an explicit `auto`/`yolo` is [`ApprovalMode::Prompt`]:
    /// the gate is the default, so a typo leaves it shut.
    pub fn parse(raw: Option<&str>) -> Self {
        match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("auto") | Some("yolo") => Self::Auto,
            _ => Self::Prompt,
        }
    }

    /// Whether the mode skips the prompt.
    pub fn is_auto(self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// The context budget: what the window is, and what is reserved out of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextConfig {
    /// The window asked of the provider, in tokens.
    pub window: u32,
    /// The character budget for one tool result.
    pub tool_result_chars: usize,
    /// The character budget for the tool results in one request.
    pub evict_chars: usize,
    /// The percentage of the window at which the conversation is compacted.
    pub compact_at: u8,
    /// The character budget for the tail kept across a compaction.
    pub tail_chars: usize,
    /// Tokens held back for the model's own reply.
    pub output_reservation: usize,
    /// Tokens held back as a safety margin, on top of the reservation.
    pub safety_margin: usize,
}

impl ContextConfig {
    /// The tokens a request may occupy, after the reservation and the margin.
    ///
    /// Saturating: a window smaller than the reservation leaves zero, which
    /// [`crate::context`] reports as an actionable error rather than sending
    /// a request the provider will reject.
    pub fn request_budget(&self) -> usize {
        (self.window as usize)
            .saturating_sub(self.output_reservation)
            .saturating_sub(self.safety_margin)
    }
}

/// The resource limits every tool runs under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// The model-call budget for one prompt.
    pub max_turns: usize,
    /// The most of a file `read_file` will pull into memory.
    pub max_read_bytes: u64,
    /// The most of a command's output that is kept.
    pub max_shell_bytes: usize,
    /// The most files one search will visit.
    pub max_search_files: usize,
    /// The most matches one search will collect.
    pub max_search_matches: usize,
    /// The most bytes one search will retain across all its matches.
    pub max_search_bytes: usize,
    /// The largest file one search will read.
    pub max_search_file_bytes: u64,
    /// How long one search may run.
    pub search_timeout: Duration,
}

/// Everything the agent runs on, resolved once.
///
/// Not `Eq`: [`Config::temperature`] is an `f64`, and two temperatures that
/// differ in the last bit are still the same configuration for every purpose
/// this type serves.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// The provider name, lowercased.
    pub provider: String,
    /// The model name.
    pub model: String,
    /// The context budget.
    pub context: ContextConfig,
    /// Whether the agent asks before it acts.
    pub approval: ApprovalMode,
    /// How long one shell command may run before it is killed.
    pub command_timeout: Duration,
    /// How long a model stream may go without yielding an item before it is
    /// abandoned.
    pub stream_idle_timeout: Duration,
    /// The resource limits.
    pub limits: Limits,
    /// The sampling temperature sent to the provider.
    pub temperature: f64,
    /// Whether the model may emit several tool calls in one turn.
    pub parallel_tool_calls: bool,
    /// Which system prompt the working agent is built with.
    pub prompt_style: PromptStyle,
}

/// Which system prompt the working agent is built with.
///
/// A knob rather than a constant because the answer is empirical: the short
/// prompt costs a small model less of its window, and the long one tells a
/// large model more about how to work. Which wins is measured, not argued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PromptStyle {
    /// The short, imperative prompt. The default: the models this program is
    /// for are the ones with the least room to spare.
    #[default]
    Terse,
    /// The full six-stage process description.
    Full,
}

/// Resolve a raw `LLM_PROMPT_STYLE` value.
///
/// Anything unrecognised falls back to the default rather than erroring: the
/// value is a preference, and a typo in it should not stop the program from
/// starting.
pub fn parse_prompt_style(raw: Option<&str>) -> PromptStyle {
    match raw.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("full") => PromptStyle::Full,
        _ => PromptStyle::Terse,
    }
}

/// The default command timeout, in seconds.
///
/// Without a bound a command that waits on input (or simply hangs) stalls the
/// whole agent run, because the tool future never resolves.
pub const DEFAULT_COMMAND_TIMEOUT_SECS: u64 = 120;

/// The default idle timeout on a model stream, in seconds.
///
/// This bounds the gap *between* items, not the length of the run. A local
/// daemon cold-loading weights can take a while to produce its first token,
/// and a CPU-bound model is slow throughout, so the figure is generous: it is
/// there to catch a stall, not slowness. Without it a dropped connection
/// leaves the UI on a spinner until the user gives up and presses Ctrl-C,
/// which is the one failure the user cannot distinguish from the model
/// thinking.
///
/// It is deliberately well above [`DEFAULT_COMMAND_TIMEOUT_SECS`]. A tool runs
/// *inside* the stream, so the gap between two items includes the time the
/// tool took: a command that runs to its own 120-second limit is 120 seconds
/// of stream silence, and a timeout at that figure would race it and kill a
/// run that was working. Two and a half times the command cap leaves room for
/// a turn's tools without letting a dead daemon go unreported.
pub const DEFAULT_STREAM_IDLE_TIMEOUT_SECS: u64 = 300;

/// Resolve a raw `LLM_STREAM_IDLE_TIMEOUT_SECS` value.
///
/// Absent, unparseable, or zero falls back to
/// [`DEFAULT_STREAM_IDLE_TIMEOUT_SECS`]: a zero timeout would abandon every
/// stream before its first token, which is never what the user meant.
pub fn parse_stream_idle_timeout(raw: Option<&str>) -> Duration {
    Duration::from_secs(
        parse_positive(raw).unwrap_or(DEFAULT_STREAM_IDLE_TIMEOUT_SECS as usize) as u64,
    )
}

/// The default cap on what `read_file` pulls into memory.
///
/// The result is elided to the model's budget (a few tens of kilobytes)
/// anyway, so reading a whole multi-gigabyte file only to discard almost all
/// of it is pure risk: the allocation is what fails, and an allocation
/// failure aborts the process rather than failing the tool.
pub const DEFAULT_MAX_READ_BYTES: u64 = 4 * 1024 * 1024;

/// The default cap on what one command's output may keep.
pub const DEFAULT_MAX_SHELL_BYTES: usize = 1024 * 1024;

/// The default cap on the files one search visits.
pub const DEFAULT_MAX_SEARCH_FILES: usize = 20_000;

/// The default cap on the matches one search collects.
pub const DEFAULT_MAX_SEARCH_MATCHES: usize = 1_000;

/// The default cap on the bytes one search retains.
pub const DEFAULT_MAX_SEARCH_BYTES: usize = 4 * 1024 * 1024;

/// The default cap on the size of a file one search will read.
pub const DEFAULT_MAX_SEARCH_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// The default cap on how long one search may run.
pub const DEFAULT_SEARCH_TIMEOUT_SECS: u64 = 30;

/// The default tokens held back for the model's reply.
pub const DEFAULT_OUTPUT_RESERVATION: usize = 4096;

/// The default tokens held back as a safety margin.
pub const DEFAULT_SAFETY_MARGIN: usize = 1024;

/// The default sampling temperature sent to the provider.
///
/// Ollama's own default is 0.8, which is the single largest cause of
/// malformed tool-call JSON from small models: at that temperature a 7B model
/// will happily emit a plausible-looking but unparseable argument object.
/// Coding work wants the most likely token, not a creative one.
pub const DEFAULT_TEMPERATURE: f64 = 0.1;

/// Whether the model is asked to emit several tool calls in one turn.
///
/// Off by default. The UI pairs results by call id, so serial calls cost only
/// a little speed, and small models are markedly worse at emitting several
/// well-formed calls at once.
pub const DEFAULT_PARALLEL_TOOL_CALLS: bool = false;

/// Resolve a raw `LLM_TEMPERATURE` value.
///
/// Absent, unparseable, or outside `0.0..=2.0` falls back to
/// [`DEFAULT_TEMPERATURE`]. No provider accepts a temperature above 2.0, and a
/// negative one is not a temperature at all.
pub fn parse_temperature(raw: Option<&str>) -> f64 {
    raw.and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|t| (0.0..=2.0).contains(t))
        .unwrap_or(DEFAULT_TEMPERATURE)
}

/// Resolve a raw `LLM_PARALLEL_TOOL_CALLS` value.
///
/// Only an explicit `true`/`1`/`yes`/`on` enables it. The default is off, so a
/// typo leaves it off rather than turning it on.
pub fn parse_parallel_tool_calls(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("true") | Some("1") | Some("yes") | Some("on")
    )
}

impl Config {
    /// The configuration in force, read from the environment.
    ///
    /// This is the only place the environment is read for configuration. A
    /// caller that wants a different value builds a [`Config`] and passes it,
    /// rather than setting a variable and hoping every reader agrees.
    pub fn from_env() -> Self {
        let provider = provider_name();
        // The window is resolved once, here, and every budget derived from it
        // is derived from *this* value. Each of the derived readers used to
        // call `num_ctx()` for itself, so a `Config` built with
        // `with_window` had a window that disagreed with its own budgets.
        let window = parse_num_ctx(std::env::var("LLM_NUM_CTX").ok().as_deref(), &provider);
        Self {
            provider,
            model: model_name(),
            context: ContextConfig {
                window,
                tool_result_chars: parse_tool_result_chars(
                    std::env::var("LLM_TOOL_RESULT_CHARS").ok().as_deref(),
                )
                .unwrap_or_else(|| tool_result_budget_chars(window)),
                evict_chars: parse_evict_chars(std::env::var("LLM_EVICT_CHARS").ok().as_deref())
                    .unwrap_or_else(|| eviction_budget_chars(window)),
                compact_at: parse_compact_at(std::env::var("LLM_COMPACT_AT").ok().as_deref()),
                tail_chars: parse_tail_chars(
                    std::env::var("LLM_COMPACT_TAIL_CHARS").ok().as_deref(),
                )
                .unwrap_or_else(|| tail_budget_chars(window)),
                output_reservation: parse_positive(
                    std::env::var("LLM_OUTPUT_RESERVATION").ok().as_deref(),
                )
                .unwrap_or(DEFAULT_OUTPUT_RESERVATION),
                safety_margin: parse_positive(std::env::var("LLM_SAFETY_MARGIN").ok().as_deref())
                    .unwrap_or(DEFAULT_SAFETY_MARGIN),
            },
            approval: ApprovalMode::parse(std::env::var("NOVA_APPROVAL").ok().as_deref()),
            command_timeout: Duration::from_secs(
                parse_positive(std::env::var("NOVA_COMMAND_TIMEOUT_SECS").ok().as_deref())
                    .unwrap_or(DEFAULT_COMMAND_TIMEOUT_SECS as usize) as u64,
            ),
            stream_idle_timeout: parse_stream_idle_timeout(
                std::env::var("LLM_STREAM_IDLE_TIMEOUT_SECS")
                    .ok()
                    .as_deref(),
            ),
            limits: Limits {
                max_turns: max_turns(),
                max_read_bytes: DEFAULT_MAX_READ_BYTES,
                max_shell_bytes: DEFAULT_MAX_SHELL_BYTES,
                max_search_files: DEFAULT_MAX_SEARCH_FILES,
                max_search_matches: DEFAULT_MAX_SEARCH_MATCHES,
                max_search_bytes: DEFAULT_MAX_SEARCH_BYTES,
                max_search_file_bytes: DEFAULT_MAX_SEARCH_FILE_BYTES,
                search_timeout: Duration::from_secs(DEFAULT_SEARCH_TIMEOUT_SECS),
            },
            temperature: parse_temperature(std::env::var("LLM_TEMPERATURE").ok().as_deref()),
            parallel_tool_calls: parse_parallel_tool_calls(
                std::env::var("LLM_PARALLEL_TOOL_CALLS").ok().as_deref(),
            ),
            prompt_style: parse_prompt_style(std::env::var("LLM_PROMPT_STYLE").ok().as_deref()),
        }
    }

    /// The configuration with every default and no environment consulted.
    ///
    /// What a test builds when it wants a run whose budgets it chose.
    pub fn defaults() -> Self {
        let window = default_window("ollama");
        Self {
            provider: "ollama".to_string(),
            model: "deepseek-v4.1-flash:cloud".to_string(),
            context: ContextConfig {
                window,
                tool_result_chars: tool_result_budget_chars(window),
                evict_chars: eviction_budget_chars(window),
                compact_at: DEFAULT_COMPACT_AT,
                tail_chars: tail_budget_chars(window),
                output_reservation: DEFAULT_OUTPUT_RESERVATION,
                safety_margin: DEFAULT_SAFETY_MARGIN,
            },
            approval: ApprovalMode::Prompt,
            command_timeout: Duration::from_secs(DEFAULT_COMMAND_TIMEOUT_SECS),
            stream_idle_timeout: Duration::from_secs(DEFAULT_STREAM_IDLE_TIMEOUT_SECS),
            limits: Limits {
                max_turns: DEFAULT_MAX_TURNS,
                max_read_bytes: DEFAULT_MAX_READ_BYTES,
                max_shell_bytes: DEFAULT_MAX_SHELL_BYTES,
                max_search_files: DEFAULT_MAX_SEARCH_FILES,
                max_search_matches: DEFAULT_MAX_SEARCH_MATCHES,
                max_search_bytes: DEFAULT_MAX_SEARCH_BYTES,
                max_search_file_bytes: DEFAULT_MAX_SEARCH_FILE_BYTES,
                search_timeout: Duration::from_secs(DEFAULT_SEARCH_TIMEOUT_SECS),
            },
            temperature: DEFAULT_TEMPERATURE,
            parallel_tool_calls: DEFAULT_PARALLEL_TOOL_CALLS,
            prompt_style: PromptStyle::default(),
        }
    }

    /// The same configuration with a different approval mode.
    ///
    /// The command line is parsed before the configuration is built, so the
    /// flag is applied here rather than read from the environment.
    pub fn with_approval(mut self, approval: ApprovalMode) -> Self {
        self.approval = approval;
        self
    }

    /// The same configuration with a different context window.
    pub fn with_window(mut self, window: u32) -> Self {
        self.context.window = window;
        self.context.tool_result_chars = tool_result_budget_chars(window);
        self.context.evict_chars = eviction_budget_chars(window);
        self.context.tail_chars = tail_budget_chars(window);
        self
    }

    /// The same configuration with a different model-call budget.
    ///
    /// The run's turn budget belongs to the session, not to the process. A
    /// test that wants a short run sets it here; setting `LLM_MAX_TURNS`
    /// instead would be process-global and would leak into every other test in
    /// the binary.
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.limits.max_turns = max_turns;
        self
    }
}

/// A positive integer from a raw environment value.
///
/// Zero and unparseable both read as "not configured": a zero budget would
/// disable the thing it bounds, which is never what the user meant.
fn parse_positive(raw: Option<&str>) -> Option<usize> {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_are_self_consistent() {
        let config = Config::defaults();
        assert_eq!(config.context.window, DEFAULT_NUM_CTX);
        assert_eq!(
            config.context.tool_result_chars,
            tool_result_budget_chars(DEFAULT_NUM_CTX)
        );
        assert_eq!(config.limits.max_turns, DEFAULT_MAX_TURNS);
        assert_eq!(config.approval, ApprovalMode::Prompt);
    }

    #[test]
    fn the_request_budget_leaves_the_reservation_and_the_margin() {
        let config = Config::defaults();
        assert_eq!(
            config.context.request_budget(),
            DEFAULT_NUM_CTX as usize - DEFAULT_OUTPUT_RESERVATION - DEFAULT_SAFETY_MARGIN
        );
    }

    #[test]
    fn a_window_smaller_than_the_reservation_leaves_no_budget() {
        let config = Config::defaults().with_window(1024);
        assert_eq!(config.context.request_budget(), 0);
    }

    #[test]
    fn only_an_explicit_auto_skips_the_gate() {
        assert_eq!(ApprovalMode::parse(Some("auto")), ApprovalMode::Auto);
        assert_eq!(ApprovalMode::parse(Some(" YOLO ")), ApprovalMode::Auto);
        assert_eq!(ApprovalMode::parse(Some("prompt")), ApprovalMode::Prompt);
        assert_eq!(ApprovalMode::parse(Some("yes")), ApprovalMode::Prompt);
        assert_eq!(ApprovalMode::parse(None), ApprovalMode::Prompt);
    }

    #[test]
    fn a_zero_or_unparseable_override_is_not_configured() {
        assert_eq!(parse_positive(Some("0")), None);
        assert_eq!(parse_positive(Some("nonsense")), None);
        assert_eq!(parse_positive(Some(" 12 ")), Some(12));
    }
}
