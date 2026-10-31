use crate::context::{self, Estimate, Heuristic, Parts};
use crate::runtime::Runtime;
use crate::tools;
use anyhow::Result;
use rig::agent::{
    AgentBuilder, AgentHook, CompletionCallAction, CompletionCallEvent, DispatchAction,
    DispatchEvent, HookContext, InvalidToolCallAction, InvalidToolCallContext,
    InvalidToolCallReason, ModelTurnAction, ModelTurnFinished, RequestPatch,
};
use rig::core::effect::EffectKind;
use rig::message::{
    AssistantContent, Message as RigMessage, Text, ToolName, ToolResult, ToolResultContent,
    UserContent,
};
use rig::providers::anthropic::Anthropic;
use rig::providers::ollama::extension::OllamaOptions;
use rig::providers::ollama::OllamaConfig;
use rig::providers::openai::wire::Route;
use rig::providers::openai::OpenAIConfig;
use std::sync::Arc;

// The instructions the agents are built with live in `crate::prompts`, beside
// the stage names the session tracks, so the two can be checked against each
// other. They are re-exported here because this module is where the agent's
// instructions have always been named, and both the app and the tests reach
// for them by that path.
pub use crate::prompts::{
    COMPACT_PROMPT, COMPACT_REQUEST, SUMMARY_PREFIX, SYSTEM_PROMPT, TERSE_PROMPT,
};

// The budgets and their resolvers live in `crate::config`, which is the one
// place the environment is read. They are re-exported here because this
// module is where the agent's budgets have always been named, and both the
// app and the tests reach for them by that path.
pub use crate::config::{
    default_window, eviction_budget_chars, is_turn_limit, max_turns, model_name, parse_compact_at,
    parse_evict_chars, parse_max_turns, parse_num_ctx, parse_prompt_style, parse_provider,
    parse_tail_chars, parse_tool_result_chars, provider_name, tail_budget_chars,
    tool_result_budget_chars, PromptStyle, CHARS_PER_TOKEN, DEFAULT_COMPACT_AT, DEFAULT_MAX_TURNS,
    DEFAULT_NUM_CTX,
};

// The request-size accounting lives in `crate::context`, which is where the
// budget it is measured against is defined. It is re-exported here because
// this module is where the history's size has always been named, and both the
// app and the tests reach for it by that path.
pub use crate::context::{history_chars, message_chars};

/// Whether `context_tokens` has reached `pct` percent of `window`.
///
/// Inclusive at the boundary: the threshold is the point at which compaction
/// is wanted, not the point after which it is. A zero window means the
/// provider reported no context size, and nothing is compacted on a figure
/// that is not known.
pub fn should_compact(context_tokens: u64, window: u32, pct: u8) -> bool {
    if window == 0 {
        return false;
    }
    context_tokens >= (window as u64) * (pct as u64) / 100
}

/// Whether `message` is a point the retained tail may start at.
///
/// A tail has to begin at a user turn that is not a tool result. A tool
/// result with no call before it is a request the provider rejects, and a
/// tail that began with an assistant turn would be one too: a conversation
/// starts with the user.
fn is_tail_boundary(message: &RigMessage) -> bool {
    context::is_exchange_start(message)
}

/// The index the tail kept across a compaction starts at.
///
/// The tail is the most recent turns, kept verbatim so the work in flight
/// survives the compaction. It starts at a user turn that is not a tool
/// result, so the tail is a well-formed conversation on its own, and it is
/// the earliest such turn whose tail still fits `budget`.
///
/// Two rules bound it. The tail never starts at index 0: a compaction that
/// evicted nothing would send the summariser an empty conversation. And when
/// no turn fits the budget the most recent one is returned anyway, for
/// [`rebuild_history`] to elide down; `history.len()` means there is no turn
/// to keep, and the summary is then the whole of the model's memory.
pub fn tail_start(history: &[RigMessage], budget: usize) -> usize {
    let mut size = 0usize;
    let mut earliest_fitting = None;
    let mut most_recent = None;

    // Index 0 is skipped: the prefix has to keep something to summarise.
    for index in (1..history.len()).rev() {
        size += message_chars(&history[index]);
        if !is_tail_boundary(&history[index]) {
            continue;
        }
        most_recent.get_or_insert(index);
        if size <= budget {
            earliest_fitting = Some(index);
        }
    }

    // The earliest boundary that fits is the longest tail the budget allows.
    earliest_fitting.or(most_recent).unwrap_or(history.len())
}

/// The tool results in `message`, mutably.
fn tool_results(message: &mut RigMessage) -> Vec<&mut ToolResult> {
    let RigMessage::User { content } = message else {
        return Vec::new();
    };
    content
        .iter_mut()
        .filter_map(|item| match item {
            UserContent::ToolResult(result) => Some(result),
            _ => None,
        })
        .collect()
}

/// The text blocks of the tool results in `message`.
fn tool_result_texts(message: &mut RigMessage) -> Vec<&mut Text> {
    tool_results(message)
        .into_iter()
        .flat_map(|result| result.content.iter_mut())
        .filter_map(|part| match part {
            ToolResultContent::Text(text) => Some(text),
            _ => None,
        })
        .collect()
}

/// Elide the tool results in `tail` so the whole tail fits `budget`.
///
/// What makes a turn large is almost always a tool result: a file read, a
/// build log. Eliding those keeps the turn's shape, what was asked, what was
/// called and what was answered, while giving the window back the bulk. The
/// elision marker says the result was cut and how to get the rest, so nothing
/// is lost silently.
///
/// Returns whether the tail now fits. A tail that still does not fit is
/// dropped by the caller: the point of compacting is to leave the window
/// room, and a tail that overran it would only make the next run compact
/// again.
fn shrink_tail(tail: &mut [RigMessage], budget: usize) -> bool {
    let total = history_chars(tail);
    if total <= budget {
        return true;
    }

    let mut count = 0usize;
    let mut result_chars = 0usize;
    for message in tail.iter_mut() {
        for text in tool_result_texts(message) {
            count += 1;
            result_chars += text.text.chars().count();
        }
    }
    if count == 0 {
        return false;
    }

    // What the turn's own words take comes off the top; the rest is shared
    // evenly between its results.
    let per_result = budget.saturating_sub(total - result_chars) / count;
    for message in tail.iter_mut() {
        for text in tool_result_texts(message) {
            text.text = tools::elide(&text.text, per_result);
        }
    }

    history_chars(tail) <= budget
}

/// The history the model carries forward from a compaction: the summary of
/// what was evicted, then the tail that was kept verbatim.
///
/// `tail` is what [`tail_start`] selected. It is elided down to `budget`, and
/// dropped entirely if it cannot be made to fit, so the history a compaction
/// leaves behind is always bounded by the summary plus the tail budget.
///
/// An assistant acknowledgement is pushed between the two. `tail[0]` is a
/// non-tool-result user turn by construction ([`crate::context::is_exchange_start`]),
/// so without it the rebuilt history opens with two consecutive user messages.
/// Providers that enforce strict alternation reject that outright, and the
/// local OpenAI-compatible servers this is aimed at are the strictest of all.
/// The acknowledgement is one line and carries no information: it exists to
/// make the transcript a shape every provider accepts.
pub fn rebuild_history(mut tail: Vec<RigMessage>, summary: &str, budget: usize) -> Vec<RigMessage> {
    if !shrink_tail(&mut tail, budget) {
        tail.clear();
    }

    let mut rebuilt = Vec::with_capacity(tail.len() + 2);
    rebuilt.push(RigMessage::user(format!("{SUMMARY_PREFIX}\n\n{summary}")));
    if !tail.is_empty() {
        rebuilt.push(RigMessage::assistant(ACKNOWLEDGEMENT));
    }
    rebuilt.extend(tail);
    rebuilt
}

/// The line [`rebuild_history`] puts between the summary and the kept tail.
///
/// It says only that the summary was read, because that is all it is for: the
/// alternation rule needs an assistant turn there, and inventing content for
/// it would put words in the model's mouth that it never said.
pub const ACKNOWLEDGEMENT: &str = "Understood. Continuing from where the work left off.";

/// The placeholder an elided tool result is replaced with.
///
/// It names the tool and the size of what was dropped, so the model can tell
/// a result it has already read from one that was taken away, and knows to
/// call the tool again rather than answer from a gap it cannot see.
fn elided_result(name: &ToolName, chars: usize) -> String {
    format!(
        "[result elided: {chars} characters from `{}`; call the tool again if you need it]",
        name.as_str()
    )
}

/// Elide the tool results in `history` that the conversation has moved past.
///
/// The most recent results are what the work is resting on, so the budget is
/// filled from the recent end: a result is kept while it fits, and every
/// older one is replaced by a placeholder. The message itself is kept, so the
/// call/result pairing a provider requires is untouched; only the text inside
/// it changes, which is what makes this safe to do on every request.
///
/// Returns whether anything was elided, so the caller can leave a request
/// alone rather than send a patch that changes nothing.
pub fn evict_stale_results(history: &mut [RigMessage], budget: usize) -> bool {
    let mut kept = 0usize;
    let mut elided = false;

    for message in history.iter_mut().rev() {
        for result in tool_results(message) {
            let chars: usize = result
                .content
                .iter()
                .filter_map(ToolResultContent::as_text)
                .map(|text| text.chars().count())
                .sum();
            // An image or structured result has no text to elide: there is
            // nothing here to shrink, so it is left alone.
            if chars == 0 {
                continue;
            }
            if kept + chars <= budget {
                kept += chars;
                continue;
            }

            let placeholder = elided_result(&result.name, chars);
            let mut first = true;
            for part in result.content.iter_mut() {
                if let ToolResultContent::Text(text) = part {
                    if first {
                        text.text.clone_from(&placeholder);
                        first = false;
                    } else {
                        text.text.clear();
                    }
                }
            }
            elided = true;
        }
    }

    elided
}

/// The hook that keeps every request inside the model's context window.
///
/// It runs before every model call, so a run that makes many tool calls does
/// not fill the window mid-flight. That is the one case compaction cannot
/// rescue: compaction runs after a run has settled, and a run that overruns
/// dies before it fires.
///
/// Two bounds, in order. The tool results in the request are elided to the
/// eviction budget, which is what keeps a long run alive. Then the whole
/// request, the system instructions and the tool schemas included, is
/// measured against the window, and the oldest completed exchanges are dropped
/// if it does not fit. A request that still does not fit is refused with a
/// size the user can act on, rather than sent to be rejected.
///
/// The prompt of a call is the result of the tool call the model just made,
/// and eliding it would take away the answer to the question the model just
/// asked; it is bounded by the tool-result budget instead, which caps any
/// single result. It is never dropped either: it is the current intent.
///
/// The rewrite is per request. The run's own transcript, and the history the
/// app carries between runs, keep the full results, so nothing is lost from
/// the record; only what the model is sent is bounded.
pub struct EnforceRequestBudget {
    /// The character budget for the tool results in one request.
    evict_chars: usize,
    /// The characters the system instructions and the tool schemas take.
    fixed_chars: usize,
    /// The session, so the estimate of the request just made is visible to
    /// the app without a second accounting of the same thing.
    rt: Arc<Runtime>,
    /// Whether to publish the measurement to the session.
    ///
    /// The working agent's hook does, because the app shows the figure. The
    /// summariser's does not: its request is the evicted prefix, not the
    /// conversation, and publishing it would replace the number the footer
    /// is about with one that describes a different request.
    record: bool,
}

impl EnforceRequestBudget {
    /// A hook bounding a request to the session's context budget.
    ///
    /// `fixed_chars` is the system instructions plus the tool schemas: the
    /// part of a request that is the same on every call, and so has to be
    /// counted before the history is measured against what is left.
    pub fn new(evict_chars: usize, fixed_chars: usize, rt: Arc<Runtime>, record: bool) -> Self {
        Self {
            evict_chars,
            fixed_chars,
            rt,
            record,
        }
    }
}

impl AgentHook for EnforceRequestBudget {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let mut history = event.history.to_vec();
        let elided = evict_stale_results(&mut history, self.evict_chars);

        let parts = Parts {
            system: self.fixed_chars,
            history: history_chars(&history),
            prompt: message_chars(event.prompt),
        };
        let estimate = Estimate::measure(parts, &self.rt.config().context, &Heuristic);

        if !estimate.over_budget() {
            if self.record {
                self.rt.record_estimate(estimate);
            }
            return if elided {
                CompletionCallAction::patch(RequestPatch::new().history(history))
            } else {
                CompletionCallAction::continue_run()
            };
        }

        // The history is what may be dropped, so the budget it is allowed is
        // what the fixed parts and the prompt leave. The prompt is not in the
        // history, so the current intent survives by construction.
        context::prune_to_budget(&mut history, estimate.history_chars());

        // Pruning the history is not the same as making the request fit. A
        // prompt that is over the window on its own leaves no room for any
        // history at all, and an empty history "fits" a zero budget trivially,
        // so the request is measured again as it would now be sent. What is
        // recorded is that second figure: it is the request that goes out, and
        // the first one describes a request that was never made.
        let after = Estimate::measure(
            Parts {
                history: history_chars(&history),
                ..parts
            },
            &self.rt.config().context,
            &Heuristic,
        );
        if self.record {
            self.rt.record_estimate(after);
        }
        if after.over_budget() {
            // Nothing left to drop and it still does not fit: the prompt alone
            // is over the window, and no amount of history would help. The
            // model is not called, and the user is told the size.
            return CompletionCallAction::stop(after.too_large().to_string());
        }

        CompletionCallAction::patch(RequestPatch::new().history(history))
    }
}

/// The most tool calls one run may have repaired or salvaged.
///
/// A model that keeps emitting the same unusable call would otherwise loop
/// until the turn budget runs out, spending every call on the same mistake.
/// Past the cap the hook stops intervening and lets rig's own handling take
/// over, which ends the run with a message the user can act on.
const MAX_TOOL_CALL_REPAIRS: usize = 4;

/// The most tool-free turns one run may be retried for carrying a tool call as
/// prose.
///
/// rig imposes no separate limit on a retried model turn, and each retry spends
/// the run's `max_turns` budget, so this cap is what keeps a model that will not
/// use the tool-call channel from consuming the whole run.
const MAX_PROSE_TOOL_CALL_RETRIES: usize = 2;

/// Names models use for the registered tools, and the tool each one means.
///
/// Only consulted when normalization finds no match, so an entry here is a
/// different word for a tool rather than a misspelling of its name. Keys are
/// already normalized (lowercase, alphanumerics only), because that is the form
/// the lookup compares against. Every right-hand side must be a name in
/// [`tools::TOOL_NAMES`]; a test pins that.
const TOOL_ALIASES: &[(&str, &str)] = &[
    ("cat", "read_file"),
    ("read", "read_file"),
    ("open", "read_file"),
    ("ls", "list_dir"),
    ("dir", "list_dir"),
    ("bash", "shell"),
    ("sh", "shell"),
    ("run", "shell"),
    ("exec", "shell"),
    ("terminal", "shell"),
    ("find", "grep"),
    ("rg", "grep"),
    ("ripgrep", "grep"),
    ("applypatch", "edit_file"),
    ("strreplace", "edit_file"),
    ("patch", "edit_file"),
    ("createfile", "write_file"),
];

/// Run-scoped repair counters, kept in the hook scratchpad.
///
/// The scratchpad is rig's documented place for state that lives as long as the
/// run, which is exactly the lifetime these caps need: a fresh run starts with a
/// fresh budget, and nothing has to be reset between runs.
#[derive(Debug, Default, Clone, Copy)]
struct RepairCounts {
    /// Tool calls repaired or salvaged so far.
    repairs: usize,
    /// Tool-free turns retried for carrying a tool call as prose.
    prose_retries: usize,
    /// The last tool call dispatched, hashed, and how many times in a row it
    /// has been dispatched.
    ///
    /// Hashed rather than kept: a `write_file` carries a whole file, and this
    /// is compared on every call the run makes. The hash is of the raw
    /// argument JSON, so two spellings of the same object read as two calls
    /// and the skip does not fire; the session's own guard still stops the
    /// run, and the cheap comparison is what this is for.
    last_call: Option<(u64, usize)>,
}

/// Repairs the tool calls a small model gets wrong, and the turns it does not
/// use the tool-call channel for at all.
///
/// Two failures dominate with a 7B model, and neither is a failure of reasoning:
///
/// * the call is *nearly* right: `readFile` for `read_file`, or arguments that
///   are a JSON string holding the object rather than the object. rig rejects
///   it, the model is told, and two model calls are spent on a spelling mistake.
/// * the call is not in the tool-call channel at all: the model writes a fenced
///   ```json block or a `<tool_call>` tag into its message. rig sees a tool-free
///   turn, treats it as the final answer, and the run ends having done nothing.
///
/// The hook answers both, and never invents a call the model did not make: a
/// repair only renames a call to a tool that is already registered, and a prose
/// retry only asks the model to say the same thing again through the right
/// channel.
pub struct RepairToolCalls {
    /// The session, so the harness can see how often each repair fired.
    rt: Arc<Runtime>,
}

impl RepairToolCalls {
    /// A hook repairing the tool calls of the run it is attached to.
    pub fn new(rt: Arc<Runtime>) -> Self {
        Self { rt }
    }
}

impl AgentHook for RepairToolCalls {
    async fn on_invalid_tool_call(
        &self,
        ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        // Past the cap, stop intervening. rig's own handling ends the run with a
        // message the user can act on, where another repair would spend another
        // model call on the same mistake.
        if ctx
            .scratchpad()
            .update::<RepairCounts, _>(|counts| counts.repairs)
            >= MAX_TOOL_CALL_REPAIRS
        {
            return None;
        }

        let action = match &event.reason {
            InvalidToolCallReason::UnknownTool => {
                match match_tool_name(&event.tool_name, &event.available_tools) {
                    Some(name) => {
                        self.rt.record_tool_name_repair();
                        InvalidToolCallAction::repair(name)
                    }
                    None => InvalidToolCallAction::retry(unknown_tool_feedback(
                        &event.tool_name,
                        &event.available_tools,
                    )),
                }
            }
            InvalidToolCallReason::MalformedArguments { error } => {
                let raw = event.args.as_deref().unwrap_or_default();
                match salvage_arguments(raw) {
                    Some(json) => {
                        self.rt.record_argument_salvage();
                        InvalidToolCallAction::retry(salvaged_arguments_feedback(
                            &event.tool_name,
                            &json,
                        ))
                    }
                    None => InvalidToolCallAction::retry(malformed_arguments_feedback(
                        &event.tool_name,
                        error,
                    )),
                }
            }
            // A tool choice that excludes the call is the caller's decision, not
            // a model mistake, and rig handles it. The wildcard also covers any
            // reason a later rig adds.
            _ => return None,
        };

        ctx.scratchpad()
            .update::<RepairCounts, _>(|counts| counts.repairs += 1);
        Some(action)
    }

    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        // A turn that used the tool-call channel is exactly what was wanted.
        if event
            .content
            .iter()
            .any(|part| matches!(part, AssistantContent::ToolCall(_)))
        {
            return ModelTurnAction::Continue;
        }

        let Some(tool) = prose_tool_call(event.content) else {
            return ModelTurnAction::Continue;
        };

        // rig imposes no separate limit on a retried turn, and each retry spends
        // the run's `max_turns` budget, so the cap is what keeps a model that
        // will not use the tool-call channel from consuming the run.
        if ctx
            .scratchpad()
            .update::<RepairCounts, _>(|counts| counts.prose_retries)
            >= MAX_PROSE_TOOL_CALL_RETRIES
        {
            return ModelTurnAction::Continue;
        }
        ctx.scratchpad()
            .update::<RepairCounts, _>(|counts| counts.prose_retries += 1);
        self.rt.record_prose_tool_call_retry();
        ModelTurnAction::retry_with_feedback(prose_tool_call_feedback(&tool))
    }

    async fn on_dispatch(&self, ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        // The loop that looks like work: the same call, again, both times
        // succeeding. A tool's result is a function of its arguments, so the
        // second call returns what the first one returned. It is skipped and
        // the model told what it already holds, which is the one thing that
        // can end the loop; stopping the run, which is what the session's
        // guard does on the third, cannot.
        let EffectKind::ToolCall { name, args } = event.kind else {
            return DispatchAction::Proceed;
        };
        let fingerprint = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            name.hash(&mut hasher);
            args.hash(&mut hasher);
            hasher.finish()
        };
        // Consecutive, not cumulative: a model that reads a file, edits it and
        // reads it again has made the same call twice, and the second read is
        // the point of the edit rather than a loop.
        let count = ctx.scratchpad().update::<RepairCounts, _>(|counts| {
            let count = match counts.last_call {
                Some((last, count)) if last == fingerprint => count + 1,
                _ => 1,
            };
            counts.last_call = Some((fingerprint, count));
            count
        });
        if count == crate::session::REPEAT_CALL_LIMIT {
            return DispatchAction::skip(repeated_call_feedback(name));
        }
        DispatchAction::Proceed
    }
}

/// A tool name reduced to what it means: lowercase, alphanumerics only.
///
/// `readFile`, `read-file` and `Read_File` all reduce to `readfile`, which is
/// what makes the first pass of [`match_tool_name`] work without a table.
fn normalize_tool_name(name: &str) -> String {
    name.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The registered tool a model's spelling of a name most likely meant.
///
/// Two passes. First the name is normalized and compared against the same
/// normalization of the registered names, which catches `readFile`,
/// `read-file` and `Read_File` without a table. Then [`TOOL_ALIASES`] catches the
/// names models invent outright, where no amount of normalization helps: `cat`
/// is not a misspelling of `read_file`, it is a different word for it.
fn match_tool_name(emitted: &str, available: &[String]) -> Option<String> {
    let wanted = normalize_tool_name(emitted);
    if let Some(name) = available
        .iter()
        .find(|name| normalize_tool_name(name) == wanted)
    {
        return Some(name.clone());
    }
    let canonical = TOOL_ALIASES
        .iter()
        .find(|(alias, _)| *alias == wanted)
        .map(|(_, canonical)| *canonical)?;
    available
        .iter()
        .find(|name| name.as_str() == canonical)
        .cloned()
}

/// What the model is told when it named a tool that does not exist.
fn unknown_tool_feedback(emitted: &str, available: &[String]) -> String {
    format!(
        "There is no tool called `{emitted}`. The tools available are: {}. Call one of them by \
         its exact name.",
        available.join(", ")
    )
}

/// What the model is told when its arguments were salvageable.
///
/// The corrected JSON is quoted verbatim so the model can copy it rather than
/// reconstruct it: a model that produced the object once can produce it again,
/// and the failure was in the framing, not the content.
fn salvaged_arguments_feedback(tool: &str, json: &str) -> String {
    format!(
        "The arguments you sent for `{tool}` were not a JSON object, but they contained one. Send \
         the call again with exactly these arguments:\n{json}"
    )
}

/// What the model is told when its arguments could not be salvaged.
fn malformed_arguments_feedback(tool: &str, error: &str) -> String {
    format!(
        "The arguments you sent for `{tool}` were not a JSON object ({error}). Send the call \
         again with a single JSON object as its arguments."
    )
}

/// What the model is told when the same call was skipped for repeating.
///
/// Three things, because a model told only that the call was skipped repeats
/// it: that the call was not run, that the result it wants is already in the
/// conversation, and what to do instead. The last one is the part that ends
/// the loop.
fn repeated_call_feedback(tool: &str) -> String {
    format!(
        "`{tool}` was already called with these exact arguments, so it was not run again: its \
         result is already in this conversation, just above. Read that result and use it. If it \
         is not what you need, call a different tool, or `{tool}` with different arguments. If \
         it is, answer."
    )
}

/// What the model is told when it wrote a tool call into its message.
fn prose_tool_call_feedback(tool: &str) -> String {
    format!(
        "You wrote a call to `{tool}` in your message. A tool call written as text is not \
         executed: it is read as your final answer, and the run ends. Call `{tool}` through the \
         tool-call interface instead, with the same arguments."
    )
}

/// The registered tool a tool-free turn asked for in its text.
///
/// A small model that has not learned the tool-call channel writes the call into
/// the message body instead. Only a name that is actually registered counts: a
/// fenced block of example JSON in an explanation is not a call, and retrying on
/// one would spend the run's budget on a turn that was already the answer.
fn prose_tool_call(content: &[AssistantContent]) -> Option<String> {
    content.iter().find_map(|part| match part {
        AssistantContent::Text(text) => {
            tool_call_documents(&text.text)
                .into_iter()
                .find_map(|document| {
                    let value: serde_json::Value = serde_json::from_str(document).ok()?;
                    named_tool(&value)
                })
        }
        _ => None,
    })
}

/// The JSON documents a model may have written a tool call into.
///
/// Three spellings cover what small models actually emit: a fenced code block, a
/// `<tool_call>…</tool_call>` span, and a bare object with neither.
///
/// The bare object is the one that matters most and was the one missing. A 7B
/// model asked to read a file answers with
/// `{"name": "read_file", "arguments": {"path": "src/lib.rs"}}` on its own line,
/// sometimes several in a row, and requiring a fence or a tag missed every one
/// of them: the eval harness reported `prose_tool_call_retries=0` on three tasks
/// that all failed for exactly this reason.
fn tool_call_documents(text: &str) -> Vec<&str> {
    let mut documents = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("```") {
        let after = &rest[open + 3..];
        let Some(close) = after.find("```") else {
            break;
        };
        let body = &after[..close];
        // Drop the info string (`json`, `JSON`, …) on the opening line.
        documents.push(match body.find('\n') {
            Some(newline) => body[newline + 1..].trim(),
            None => body.trim(),
        });
        rest = &after[close + 3..];
    }
    let mut rest = text;
    while let Some(open) = rest.find("<tool_call>") {
        let after = &rest[open + "<tool_call>".len()..];
        let Some(close) = after.find("</tool_call>") else {
            break;
        };
        documents.push(after[..close].trim());
        rest = &after[close + "</tool_call>".len()..];
    }
    // A fenced or tagged object is found again here. That costs a second parse
    // and nothing else, and it keeps the three spellings from having to know
    // about each other.
    for (from, to) in balanced_object_spans(text) {
        documents.push(&text[from..=to]);
    }
    documents
}

/// The registered tool named by a parsed tool-call document.
///
/// Accepts the three spellings models use for the name: `{"name": …}`,
/// `{"tool": …}`, and the OpenAI shape `{"function": {"name": …}}`.
fn named_tool(value: &serde_json::Value) -> Option<String> {
    let name = value
        .get("name")
        .or_else(|| value.get("tool"))
        .or_else(|| {
            value
                .get("function")
                .and_then(|function| function.get("name"))
        })
        .and_then(serde_json::Value::as_str)?;
    let wanted = normalize_tool_name(name);
    tools::TOOL_NAMES
        .split(", ")
        .find(|registered| normalize_tool_name(registered) == wanted)
        .map(str::to_string)
}

/// The JSON object hidden in a model's malformed argument text, if there is one.
///
/// Small models wrap arguments in a fenced block, prefix them with prose, or
/// double-encode them as a JSON string. Each of those is one parse away from
/// being correct, and handing the model its own corrected text back is far
/// cheaper than asking it to produce the call again from scratch.
fn salvage_arguments(raw: &str) -> Option<String> {
    let text = strip_code_fence(raw);
    // A double-encoded object: the whole text is a JSON string holding JSON.
    if let Ok(serde_json::Value::String(inner)) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(object) = first_json_object(&inner) {
            return Some(object);
        }
    }
    first_json_object(text)
}

/// The text inside a fenced code block, or the text unchanged when there is no
/// fence.
fn strip_code_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    // Drop the info string (`json`, `JSON`, …) that follows the opening fence.
    let body = match rest.find('\n') {
        Some(newline) => &rest[newline + 1..],
        None => rest,
    };
    body.strip_suffix("```").unwrap_or(body).trim()
}

/// The spans of every balanced `{…}` in `text`, as byte ranges.
///
/// Brace counting has to respect string literals, or a `}` inside a string
/// argument would close the object early and the scan would fail on exactly the
/// arguments that most need it.
fn balanced_object_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in text.bytes().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' => {
                if depth == 0 {
                    start = Some(index);
                }
                depth += 1;
            }
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    if let Some(from) = start {
                        spans.push((from, index));
                    }
                    start = None;
                }
            }
            _ => {}
        }
    }
    spans
}

/// The first balanced `{…}` in `text` that parses as a JSON object, returned as
/// its canonical serialization.
fn first_json_object(text: &str) -> Option<String> {
    balanced_object_spans(text)
        .into_iter()
        .find_map(|(from, to)| {
            let value: serde_json::Value = serde_json::from_str(&text[from..=to]).ok()?;
            value.is_object().then(|| value.to_string())
        })
}

/// Fold `$body` over every tool the agent registers, in display order.
///
/// The seven tools used to be enumerated twice and independently: once to
/// count their schemas for the request budget, once to register them with the
/// agent. Adding a tool to one list and not the other silently mis-measured
/// every request, and nothing would have failed to say so. One list, folded
/// twice, is what makes that impossible.
///
/// A fold rather than a chain because the two uses have different shapes: one
/// sums a number, the other threads a builder. Threading is also what lets the
/// builder's type change on the first `.tool()` call, which a chain written
/// out by hand does implicitly and a loop cannot do at all.
///
/// `$acc` is rebound on every step, so `$body` sees the previous step's value
/// and returns the next one. `$rt` is evaluated once, before the first step.
macro_rules! for_each_tool {
    ($rt:expr, $seed:expr, |$acc:ident, $tool:ident| $body:expr) => {{
        let __rt = $rt;
        let $acc = $seed;
        let $acc = {
            let $tool = tools::ReadFile::new(__rt.clone());
            $body
        };
        let $acc = {
            let $tool = tools::WriteFile::new(__rt.clone());
            $body
        };
        let $acc = {
            let $tool = tools::EditFile::new(__rt.clone());
            $body
        };
        let $acc = {
            let $tool = tools::Shell::new(__rt.clone());
            $body
        };
        let $acc = {
            let $tool = tools::ListDir::new(__rt.clone());
            $body
        };
        let $acc = {
            let $tool = tools::Grep::new(__rt.clone());
            $body
        };
        let $acc = {
            let $tool = tools::Glob::new(__rt.clone());
            $body
        };
        $acc
    }};
}

/// The characters one tool's schema costs in every request.
///
/// A trait rather than a free function because [`rig::tool::Tool::NAME`] is an
/// associated constant: naming it needs the tool's type, and the type is not
/// nameable at the call site, where all that is in hand is a value.
trait SchemaChars {
    /// The name, description and parameter schema, as the provider sees them.
    fn schema_chars(&self) -> usize;
}

impl<T: rig::tool::Tool> SchemaChars for T {
    fn schema_chars(&self) -> usize {
        T::NAME.chars().count()
            + self.description().chars().count()
            + self.parameters().to_string().chars().count()
    }
}

/// The characters the tool schemas take in a request.
///
/// The schemas are sent on every call and are the same every time, so they are
/// counted once, when the agent is built, and added to the system instructions
/// to give the budget hook the fixed part of a request. Counting them here
/// rather than at each call is what makes the per-request measurement cheap
/// enough to run before every model call.
///
/// The count is of the JSON the schema serialises to, which is what goes on
/// the wire, plus the name and the description. It is an estimate of the
/// provider's own framing, not a reproduction of it: the budget it feeds keeps
/// a safety margin for exactly that.
fn tool_schema_chars(rt: &Arc<Runtime>) -> usize {
    for_each_tool!(rt, 0usize, |total, tool| total + tool.schema_chars())
}

/// Build the agent from any model that converts into `DynModel<Completion>`.
/// Taking `impl Into<...>` lets each provider arm keep its own concrete model
/// type instead of forcing them to unify into one boxed type up front.
///
/// Whether `provider`'s wire format has a field for `parallel_tool_calls`.
///
/// rig refuses a request that carries an option the provider cannot express
/// rather than dropping it, which is the right call: silently ignoring the
/// setting would leave the user believing it was in force. The consequence is
/// that the caller has to know which providers have the field, and this is
/// where that knowledge lives.
///
/// Ollama's native `/api/chat` route does not. Setting the option there fails
/// every request outright, which is how this was found: the eval harness
/// reported `calls=0` for every task against a real daemon.
fn supports_parallel_tool_calls(provider: &str) -> bool {
    provider != "ollama"
}

/// Build the working agent and the summariser over `model`.
///
/// The runtime is a parameter rather than something read here: the tools are
/// built from it, so the agent a caller gets is bound to the same workspace,
/// approval gate, process runner and budgets as the session that asked for
/// it. There is no way to build an agent whose tools reach for global state,
/// because there is no global state left to reach for.
pub fn build_agent(
    model: impl Into<rig::DynModel<rig::operation::Completion>>,
    rt: &Arc<Runtime>,
) -> Result<rig::agent::Agent> {
    // The instructions, chosen by `LLM_PROMPT_STYLE`. The terse prompt is the
    // default because the models this program is for are the ones with the
    // least window to spare; the full one is kept so the eval harness can
    // measure the two against the same tasks rather than argue about them.
    let preamble = match rt.config().prompt_style {
        PromptStyle::Terse => TERSE_PROMPT,
        PromptStyle::Full => SYSTEM_PROMPT,
    };

    // The fixed part of every request: the instructions, and the schemas of
    // the tools below. Counted before the builder consumes the tools.
    let fixed_chars = preamble.chars().count() + tool_schema_chars(rt);

    let builder = AgentBuilder::new(model)
        .preamble(preamble)
        // The context window is a model parameter, so it only reaches the
        // daemon on the native `/api/chat` route. Storing it here is harmless
        // for the other providers: options are keyed by provider, and only
        // the Ollama wire reads this entry.
        .provider_option(OllamaOptions::default().num_ctx(rt.config().context.window))
        // Sampling is a model parameter too, and the one that decides whether
        // a small model emits parseable tool-call JSON at all. Left unset, the
        // daemon's own default applies, which for Ollama is 0.8.
        .temperature(rt.config().temperature)
        // The budget already reserves this much room for the answer
        // (`ContextConfig::request_budget`), but the reservation only means
        // anything if the model is told about it. Without this the model is
        // free to run past the reservation and overrun the window.
        .max_tokens(rt.config().context.output_reservation as u64)
        // Local OpenAI-compatible servers (llama.cpp, vLLM, LM Studio) emit
        // finish reasons outside the spec. Refusing them would fail a run
        // that actually succeeded.
        .accept_unknown_finish_reasons(true);

    // Serial tool calls by default: the UI pairs results by call id, so
    // nothing is lost but a little speed, and small models are markedly worse
    // at emitting several well-formed calls in one turn.
    //
    // Only where the provider has the field. rig refuses a request carrying an
    // option the provider cannot express rather than dropping it, which is the
    // right call, and the consequence is that the caller has to know: Ollama's
    // native `/api/chat` has no such field, so setting it there fails every
    // request outright. The daemon decides for itself in that case, which is
    // what it did before this knob existed.
    let builder = if supports_parallel_tool_calls(&rt.config().provider) {
        builder.parallel_tool_calls(rt.config().parallel_tool_calls)
    } else {
        builder
    };

    // The same list `tool_schema_chars` measured a moment ago, so the schemas
    // the budget accounts for are the schemas that are actually sent.
    let builder = for_each_tool!(rt, builder, |builder, tool| builder.tool(tool));

    Ok(builder
        // The history is re-sent on every call, so a run that makes many tool
        // calls grows the prompt by a result per call. This bounds what the
        // model is sent without touching the record the app keeps, and refuses
        // a request that cannot be made to fit rather than sending it.
        .add_hook(EnforceRequestBudget::new(
            rt.config().context.evict_chars,
            fixed_chars,
            rt.clone(),
            true,
        ))
        // Registered after the budget hook, and independent of it: this one
        // repairs what the model got wrong, that one bounds what it is sent.
        // `add_hook` stacks, so both run.
        .add_hook(RepairToolCalls::new(rt.clone()))
        .build())
}

/// Build the agent that summarises a conversation, with no tools at all.
///
/// The absence of tools is the point, and it is why this is a separate agent
/// rather than the working one with a prompt that asks it not to act. An
/// instruction is not a boundary: a model that called a tool anyway would have
/// rig execute it, and the call that exists to shrink the conversation would
/// be able to write a file. Here there is nothing to call, so there is nothing
/// to execute.
///
/// It carries the same context window option as the working agent, because the
/// summarisation request is a request like any other and is measured against
/// the same window. It carries the budget hook for the same reason: the
/// evicted prefix is most of a conversation that has already grown past the
/// threshold, so the request that summarises it needs bounding too. The hook
/// elides the tool results in it first, which is where the bulk is, and only
/// drops the oldest exchanges if that was not enough.
pub fn build_summariser(
    model: impl Into<rig::DynModel<rig::operation::Completion>>,
    rt: &Arc<Runtime>,
) -> Result<rig::agent::Agent> {
    // The summariser's own fixed part: its instructions, and no schemas,
    // because it advertises no tools.
    let fixed_chars = COMPACT_PROMPT.chars().count();

    Ok(AgentBuilder::new(model)
        .preamble(COMPACT_PROMPT)
        .provider_option(OllamaOptions::default().num_ctx(rt.config().context.window))
        // A creative summariser is not wanted either: the summary is a
        // compression of what happened, and the same low temperature that
        // keeps tool-call JSON parseable keeps the summary faithful.
        .temperature(rt.config().temperature)
        .max_tokens(rt.config().context.output_reservation as u64)
        .add_hook(EnforceRequestBudget::new(
            rt.config().context.evict_chars,
            fixed_chars,
            rt.clone(),
            false,
        ))
        .build())
}

/// The two agents a session runs.
///
/// They are built together, from one model, so a session cannot end up with a
/// summariser that can act: the compaction call is the one place the app hands
/// a whole conversation to a model, and it must not be able to change anything
/// while it does.
pub struct Agents {
    /// The agent that does the work. It has the tools.
    pub main: rig::agent::Agent,
    /// The agent that summarises. It has none.
    pub summariser: rig::agent::Agent,
}

impl Agents {
    /// Both agents over one model.
    ///
    /// The model is erased once and the handle cloned, so the two agents share
    /// one connection pool and one set of provider options rather than
    /// building a second client.
    pub fn build(
        model: impl Into<rig::DynModel<rig::operation::Completion>>,
        rt: &Arc<Runtime>,
    ) -> Result<Self> {
        let model: rig::DynModel<rig::operation::Completion> = model.into();
        Ok(Self {
            main: build_agent(model.clone(), rt)?,
            summariser: build_summariser(model, rt)?,
        })
    }
}

/// The models the local Ollama daemon has, newest first as `ollama list`
/// reports them.
///
/// This shells out rather than calling the HTTP API because `ollama list` is
/// the command the user already knows, and it reports the same set the CLI
/// does, including cloud models that have no local weights.
///
/// It goes through the session's [`crate::process::ProcessRunner`] like every
/// other command, so it has a deadline and an output bound. It used to be a
/// bare `std::process::Command::output()`, which waits forever and buffers
/// without limit: a daemon that hung would hang the picker, and a daemon that
/// printed a great deal would be read into memory in full.
pub async fn list_ollama_models(rt: &Runtime) -> Result<Vec<String>> {
    let out = rt.process().run_program("ollama", &["list"]).await?;
    if !out.succeeded() {
        anyhow::bail!(
            "`ollama list` failed: {}",
            out.render(rt.process().max_output())
        );
    }
    Ok(parse_ollama_list(&out.stdout))
}

/// Pull the model names out of `ollama list` output.
///
/// The output is a fixed-width table whose first line is a header. The name is
/// the first column, and a name never contains whitespace, so the first token
/// of each remaining line is the model.
pub fn parse_ollama_list(stdout: &str) -> Vec<String> {
    stdout
        .lines()
        .skip(1) // the NAME/ID/SIZE/MODIFIED header
        .filter_map(|line| line.split_whitespace().next())
        .map(|name| name.to_string())
        .collect()
}

/// Build both agents for a specific model, using the configured provider.
pub fn create_agents_with_model(model: &str, rt: &Arc<Runtime>) -> Result<Agents> {
    build_for_provider(&provider_name(), model, |var| std::env::var(var).ok(), rt)
}

/// Build both agents for a named provider, reading credentials through
/// `credential`.
///
/// Split out from [`create_agents_with_model`] so the dispatch, the missing
/// credential, and the unknown provider can all be tested without touching
/// the process environment, which is global and shared by parallel tests.
///
/// An unrecognised provider is an error rather than a fallback to Ollama: a
/// typo would otherwise send the prompt to a different provider than the one
/// the user asked for, and the only sign of it would be the answer.
pub fn build_for_provider(
    provider: &str,
    model: &str,
    credential: impl Fn(&str) -> Option<String>,
    rt: &Arc<Runtime>,
) -> Result<Agents> {
    // A blank value is as good as absent: `set OPENAI_API_KEY=` is a common
    // way to end up with an empty key and a confusing 401.
    let key = |var: &str| credential(var).filter(|v| !v.trim().is_empty());

    match provider {
        "anthropic" => {
            let api_key = key("ANTHROPIC_API_KEY")
                .ok_or_else(|| missing_credential("ANTHROPIC_API_KEY", provider))?;
            Agents::build(Anthropic::new(api_key).completion(model), rt)
        }
        "openai" => {
            // `LLM_BASE_URL` points the OpenAI-compatible client at a local
            // server: llama.cpp, vLLM, LM Studio, or anything else that speaks
            // the same wire format. Those servers take no credential, so the
            // key becomes optional when a base URL is given and stays
            // mandatory when it is not, which is the case where the request
            // really is going to api.openai.com.
            let base_url = key("LLM_BASE_URL");
            let api_key = match key("OPENAI_API_KEY") {
                Some(api_key) => api_key,
                None if base_url.is_some() => {
                    // A placeholder rather than an empty string: the client
                    // sends it as a bearer token, and a local server that
                    // checks the header at all is happier with a value than
                    // with nothing.
                    "not-needed".to_string()
                }
                None => return Err(missing_credential("OPENAI_API_KEY", provider)),
            };

            let config = OpenAIConfig::new(api_key).with_route(Route::Chat);
            let config = match base_url {
                Some(url) => config.with_base_url(url),
                None => config,
            };
            Agents::build(config.client().completion(model), rt)
        }
        "ollama" => {
            // The daemon's native `/api/chat`, not its OpenAI-compatible
            // `/v1`: only the native route reads model `options`, and
            // `num_ctx` is the one that matters. `/v1` silently ignores it
            // and serves the daemon's VRAM-derived default (4096 on a 6 GB
            // card), which a multi-step run overruns.
            let base_url = credential("OLLAMA_BASE_URL")
                .filter(|v| !v.trim().is_empty())
                .unwrap_or_else(|| "http://localhost:11434".to_string());
            Agents::build(
                OllamaConfig::new()
                    .with_base_url(base_url)
                    .client()
                    .native_completion(model),
                rt,
            )
        }
        other => anyhow::bail!(
            "unknown LLM_PROVIDER {other:?}: expected one of ollama, anthropic, openai"
        ),
    }
}

/// The error for a provider whose credential is not in the environment.
fn missing_credential(var: &str, provider: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{var} is not set, and LLM_PROVIDER={provider} needs it. \
         Set it, or use LLM_PROVIDER=ollama for the local daemon."
    )
}

pub fn create_agents(rt: &Arc<Runtime>) -> Result<Agents> {
    create_agents_with_model(&model_name(), rt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every alias in the table names a tool that is actually registered.
    ///
    /// A typo here would be invisible in use: the repair would find no match
    /// and fall through to the "no such tool" feedback, which looks exactly
    /// like the model having invented a name.
    #[test]
    fn every_alias_names_a_registered_tool() {
        let registered: Vec<&str> = crate::tools::TOOL_NAMES.split(", ").collect();
        for (alias, canonical) in TOOL_ALIASES {
            assert!(
                registered.contains(canonical),
                "`{alias}` maps to `{canonical}`, which is not a registered tool"
            );
            assert_eq!(
                *alias,
                normalize_tool_name(alias),
                "`{alias}` is not in the normalized form the lookup compares against"
            );
        }
    }

    /// The alias table is only consulted when normalization finds no match, so
    /// an entry that normalization *would* have found is dead weight.
    #[test]
    fn no_alias_is_reachable_by_normalization_alone() {
        let registered: Vec<String> = crate::tools::TOOL_NAMES
            .split(", ")
            .map(normalize_tool_name)
            .collect();
        for (alias, _) in TOOL_ALIASES {
            assert!(
                !registered.contains(&normalize_tool_name(alias)),
                "`{alias}` would already be found by normalizing the registered names"
            );
        }
    }

    #[test]
    fn a_name_is_matched_however_it_is_spelled() {
        let available: Vec<String> = crate::tools::TOOL_NAMES
            .split(", ")
            .map(str::to_string)
            .collect();
        for spelling in [
            "read_file",
            "readFile",
            "read-file",
            "Read_File",
            "READFILE",
        ] {
            assert_eq!(
                match_tool_name(spelling, &available).as_deref(),
                Some("read_file"),
                "`{spelling}` should be read as `read_file`"
            );
        }
        for (spelling, expected) in [
            ("cat", "read_file"),
            ("ls", "list_dir"),
            ("bash", "shell"),
            ("rg", "grep"),
            ("apply_patch", "edit_file"),
            ("create_file", "write_file"),
        ] {
            assert_eq!(
                match_tool_name(spelling, &available).as_deref(),
                Some(expected),
                "`{spelling}` should be read as `{expected}`"
            );
        }
        assert_eq!(match_tool_name("frobnicate", &available), None);
    }

    #[test]
    fn a_fenced_object_is_salvaged() {
        let salvaged = salvage_arguments("```json\n{\"path\": \"a.rs\"}\n```")
            .expect("the object is in there");
        assert_eq!(salvaged, "{\"path\":\"a.rs\"}");
    }

    #[test]
    fn an_object_with_prose_around_it_is_salvaged() {
        let salvaged = salvage_arguments("Sure! {\"path\": \"a.rs\"} — running it now.")
            .expect("the object is in there");
        assert_eq!(salvaged, "{\"path\":\"a.rs\"}");
    }

    /// The shape rig's own `parse` filters out, and the reason this branch
    /// exists: a string holding a string holding the object. It is reachable
    /// only through the streaming path, where a cut-off argument buffer
    /// bypasses `parse`, so it cannot be driven from a test through the agent.
    #[test]
    fn a_string_holding_a_string_holding_an_object_is_salvaged() {
        let inner = serde_json::json!({ "path": "a.rs" }).to_string();
        let quoted = serde_json::Value::String(inner).to_string();
        let salvaged = salvage_arguments(&quoted).expect("the object is two unwraps down");
        assert_eq!(salvaged, "{\"path\":\"a.rs\"}");
    }

    #[test]
    fn prose_with_no_object_in_it_is_not_salvaged() {
        assert_eq!(salvage_arguments("the manifest, please"), None);
        assert_eq!(salvage_arguments(""), None);
        // A truncated object is not a balanced one, and guessing where it
        // should have ended is how a repair invents an argument.
        assert_eq!(salvage_arguments("{\"path\": \"a.rs\""), None);
    }

    /// A brace inside a string is not a brace. Counting it as one would cut
    /// the object short at the wrong place.
    #[test]
    fn a_brace_inside_a_string_does_not_end_the_object() {
        let salvaged = salvage_arguments("{\"content\": \"a } b\", \"path\": \"a.rs\"}")
            .expect("the object is balanced");
        assert_eq!(salvaged, "{\"content\":\"a } b\",\"path\":\"a.rs\"}");
    }

    #[test]
    fn a_tool_call_written_as_prose_is_found() {
        let text = "I'll read it.\n\n```json\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n```";
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new(text))]).as_deref(),
            Some("read_file")
        );

        let text =
            "<tool_call>{\"tool\": \"shell\", \"arguments\": {\"command\": \"ls\"}}</tool_call>";
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new(text))]).as_deref(),
            Some("shell")
        );

        let text = "```json\n{\"function\": {\"name\": \"grep\"}}\n```";
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new(text))]).as_deref(),
            Some("grep")
        );
    }

    /// The spelling a 7B model actually uses: a bare object, no fence and no
    /// tag. Requiring one of those missed every call this model made, which the
    /// eval harness showed as three failed tasks with
    /// `prose_tool_call_retries=0`.
    #[test]
    fn a_bare_object_is_a_tool_call() {
        let text = "{\"name\": \"read_file\", \"arguments\": {\"path\": \"src/lib.rs\"}}";
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new(text))]).as_deref(),
            Some("read_file")
        );

        // Several in a row, which is how a model plans a multi-step task.
        let text = "{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n\
                    {\"name\": \"shell\", \"arguments\": {\"command\": \"cargo test\"}}";
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new(text))]).as_deref(),
            Some("read_file")
        );

        // A `}` inside a string argument must not close the object early.
        let text = "{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.rs\", \
                    \"contents\": \"fn main() {}\"}}";
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new(text))]).as_deref(),
            Some("write_file")
        );
    }

    /// A bare object that names no registered tool is prose, not a call.
    #[test]
    fn a_bare_object_that_names_no_tool_is_not_a_call() {
        let text = "The manifest is {\"name\": \"fixture\", \"version\": \"0.0.0\"}.";
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new(text))]),
            None
        );
    }

    /// A fenced block of example JSON in an explanation is not a call. Retrying
    /// on one would spend the run's budget on a turn that was already the
    /// answer.
    #[test]
    fn a_fenced_block_that_names_no_tool_is_not_a_call() {
        let text = "The config looks like:\n\n```json\n{\"name\": \"my-project\", \"version\": \"1.0\"}\n```";
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new(text))]),
            None
        );
        assert_eq!(
            prose_tool_call(&[AssistantContent::Text(Text::new("no code here"))]),
            None
        );
    }

    /// A turn that used the tool-call channel is not a prose call, whatever
    /// its text says.
    #[test]
    fn a_turn_with_a_real_tool_call_is_left_alone() {
        let content = [
            AssistantContent::Text(Text::new("```json\n{\"name\": \"read_file\"}\n```")),
            AssistantContent::ToolCall(rig::message::ToolCall::new(
                rig::message::CallId::from_wire("call_1"),
                rig::message::ToolFunction::new(
                    rig::message::ToolName::new("read_file").unwrap(),
                    serde_json::json!({}),
                ),
            )),
        ];
        assert!(
            content
                .iter()
                .any(|part| matches!(part, AssistantContent::ToolCall(_))),
            "the hook's first check is what makes this a no-op"
        );
    }

    /// The parallel-tool-calls knob is only set where the wire has the field.
    ///
    /// rig refuses a request carrying an option the provider cannot express,
    /// so this is not a preference: getting it wrong fails every request. The
    /// failure is invisible to a test that never sends one, which is why the
    /// guard is asserted here rather than left to the harness to find again.
    #[test]
    fn the_parallel_tool_calls_knob_is_only_set_where_the_wire_has_it() {
        assert!(
            !supports_parallel_tool_calls("ollama"),
            "Ollama's /api/chat has no parallel_tool_calls field"
        );
        assert!(supports_parallel_tool_calls("openai"));
        assert!(supports_parallel_tool_calls("anthropic"));
    }
}
