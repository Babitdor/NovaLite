//! The request budget: what one model request may occupy, and what it does.
//!
//! Every model call re-sends the whole conversation, so the size of a request
//! is the size of the session rather than the size of the prompt. The window
//! is a hard limit: a request over it is rejected outright, and a run that is
//! rejected mid-flight has already made its edits.
//!
//! Two things bound a request. The tool results in it are elided to a budget
//! of their own ([`crate::agent::evict_stale_results`]), which is what keeps a
//! long run alive. On top of that the whole request is measured against the
//! window before it is sent, and the oldest completed exchanges are dropped
//! when it does not fit. What is left is measured again, and a request that
//! still does not fit is refused with a size the user can act on rather than
//! sent to be rejected.
//!
//! The measurement is an estimate. rig exposes no token counter, so the figure
//! is characters divided by a conservative average token length, rounded up,
//! and it is labelled as an estimate wherever it is shown. The provider's own
//! usage report is the exact figure for the call that has *already* happened,
//! and it is kept separate: it is what the next request starts from, not what
//! this one will cost.

use crate::config::{ContextConfig, CHARS_PER_TOKEN};
use crate::error::{Error, Result};
use rig::message::{Message as RigMessage, UserContent};

/// The parts of a request, in characters.
///
/// Named separately because they are bounded differently. The system
/// instructions and the tool schemas are fixed for the life of the agent; the
/// history is what pruning may drop; the prompt is the result of the call the
/// model just made, and is the one part that must survive whole.
///
/// The tool schemas are counted into `system` rather than given a field of
/// their own. They are fixed for the life of the agent exactly as the
/// instructions are, and the two are measured together when the agent is
/// built; a separate field was never written to, so it read as zero and made
/// the parts fail to add up to the total.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Parts {
    /// The system instructions, including the tool schemas.
    pub system: usize,
    /// The conversation carried into the request.
    pub history: usize,
    /// The prompt the request is answering.
    pub prompt: usize,
}

impl Parts {
    /// The whole request, in characters.
    pub fn total(&self) -> usize {
        self.system + self.history + self.prompt
    }
}

/// The tokens `chars` is expected to occupy.
///
/// Rounded up, so a request is never under-counted by the rounding: the
/// estimate is used to decide whether a request fits, and an estimate that
/// rounds down would let a request through that the provider then rejects.
pub fn tokens_for(chars: usize) -> usize {
    chars.div_ceil(CHARS_PER_TOKEN)
}

/// How a request's size is measured.
///
/// rig exposes no token counter, so the only implementation is the estimate.
/// The trait is the seam a provider-specific counter would be added behind:
/// the accounting, the pruning and the error all take a `&dyn Counter`, so
/// adding one is a new implementation rather than a change to the budget.
pub trait Counter: Send + Sync {
    /// The tokens `parts` is expected to occupy.
    fn tokens(&self, parts: &Parts) -> usize;

    /// Whether the figure is exact rather than an estimate.
    fn is_exact(&self) -> bool;
}

/// Characters divided by a conservative average token length, rounded up.
///
/// The usual figure for English prose and source code. It is deliberately not
/// presented as accounting: a request of mostly punctuation or mostly CJK
/// costs more per character than this, which is why the budget it is measured
/// against keeps a safety margin.
#[derive(Debug, Clone, Copy, Default)]
pub struct Heuristic;

impl Counter for Heuristic {
    fn tokens(&self, parts: &Parts) -> usize {
        tokens_for(parts.total())
    }

    fn is_exact(&self) -> bool {
        false
    }
}

/// What a request is expected to occupy, against what it may.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Estimate {
    /// The request's parts, in characters.
    pub parts: Parts,
    /// The request's size, in tokens.
    pub tokens: usize,
    /// The tokens the request may occupy.
    pub budget: usize,
    /// Whether `tokens` came from a token counter rather than the estimate.
    pub exact: bool,
}

impl Estimate {
    /// Measure `parts` against `config`'s request budget.
    pub fn measure(parts: Parts, config: &ContextConfig, counter: &dyn Counter) -> Self {
        Self {
            parts,
            tokens: counter.tokens(&parts),
            budget: config.request_budget(),
            exact: counter.is_exact(),
        }
    }

    /// Whether the request is over the budget.
    pub fn over_budget(&self) -> bool {
        self.tokens > self.budget
    }

    /// How the figure should be labelled wherever it is shown.
    pub fn label(&self) -> &'static str {
        if self.exact {
            "exact"
        } else {
            "estimated"
        }
    }

    /// The request's size as a percentage of the budget, clamped to 999.
    ///
    /// Clamped before narrowing: a request far over the budget would otherwise
    /// wrap in a `u8` and paint a smaller number than one just over it.
    pub fn percent(&self) -> u16 {
        if self.budget == 0 {
            return 999;
        }
        (self.tokens as u64 * 100 / self.budget as u64).min(999) as u16
    }

    /// The characters the history may occupy and still fit the budget.
    ///
    /// The other parts are fixed for this call: the system instructions and
    /// the tool schemas are the same on every request, and the prompt is the
    /// result of the call the model just made, which is what the request is
    /// for. What is left over is what the history may take.
    pub fn history_chars(&self) -> usize {
        let fixed = self.parts.system + self.parts.prompt;
        self.budget
            .saturating_mul(CHARS_PER_TOKEN)
            .saturating_sub(fixed)
    }

    /// The error for a request that cannot be made to fit.
    pub fn too_large(&self) -> Error {
        Error::ContextTooLarge {
            needed: self.tokens,
            budget: self.budget,
            prompt: tokens_for(self.parts.prompt),
        }
    }
}

/// Whether `message` starts a fresh exchange.
///
/// A prune may only cut here. A tool result with no call before it is a request
/// the provider rejects, so a cut in the middle of an exchange would turn a
/// request that is merely too large into one that is malformed.
pub fn is_exchange_start(message: &RigMessage) -> bool {
    match message {
        RigMessage::User { content } => !content
            .iter()
            .any(|item| matches!(item, UserContent::ToolResult(_))),
        _ => false,
    }
}

/// The characters `history` contributes to the context.
///
/// Counted in `char`s, like [`crate::tools::elide`], so a multi-byte glyph
/// costs what it looks like rather than what it occupies.
pub fn history_chars(history: &[RigMessage]) -> usize {
    history.iter().map(message_chars).sum()
}

/// The characters `message` contributes to the context.
pub fn message_chars(message: &RigMessage) -> usize {
    match message {
        RigMessage::System { content } => content.chars().count(),
        RigMessage::User { content } => content
            .iter()
            .map(|item| match item {
                UserContent::Text(text) => text.text.chars().count(),
                UserContent::ToolResult(result) => result
                    .content
                    .iter()
                    .filter_map(rig::message::ToolResultContent::as_text)
                    .map(|text| text.chars().count())
                    .sum(),
                _ => 0,
            })
            .sum(),
        RigMessage::Assistant(turn) => turn
            .content
            .iter()
            .map(|part| match part {
                rig::message::AssistantContent::Text(text) => text.text.chars().count(),
                // A tool call is not free, and counting it as free is the
                // accounting bug that matters most: a `write_file` call
                // carrying four kilobytes of file content is exactly the turn
                // that overruns the window, and it used to measure as nothing.
                // The name is counted too, because it is re-sent on every
                // subsequent request like everything else here.
                rig::message::AssistantContent::ToolCall(call) => {
                    // `arguments` is a map, and serializing it is the only way
                    // to measure it as the wire will. A map of JSON values
                    // cannot fail to serialize, so the fallback is unreachable
                    // and exists only because the signature says it can.
                    let arguments = serde_json::to_string(&call.function.arguments)
                        .map(|text| text.chars().count())
                        .unwrap_or(0);
                    call.function.name.as_str().chars().count() + arguments
                }
                _ => 0,
            })
            .sum(),
    }
}

/// The index the oldest exchange that fits `budget` starts at.
///
/// The most recent exchange is never dropped: it is what the work in flight
/// rests on, and it is where a failure the model is still correcting shows up.
/// `history.len()` means even that does not fit, and the caller reports the
/// request as too large rather than sending it.
pub fn prune_start(history: &[RigMessage], budget: usize) -> usize {
    let mut size = 0usize;
    let mut earliest_fitting = None;
    let mut most_recent = None;

    for index in (0..history.len()).rev() {
        size += message_chars(&history[index]);
        if !is_exchange_start(&history[index]) {
            continue;
        }
        most_recent.get_or_insert(index);
        if size <= budget {
            earliest_fitting = Some(index);
        }
    }

    // The earliest boundary that fits is the longest history the budget
    // allows; failing that, the most recent exchange is kept and the caller
    // finds out it still does not fit.
    earliest_fitting.or(most_recent).unwrap_or(history.len())
}

/// What a prune did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pruned {
    /// How many messages were dropped from the front.
    pub dropped: usize,
    /// Whether what is left fits the budget.
    pub fits: bool,
}

/// Drop the oldest completed exchanges from `history` until it fits `budget`.
///
/// Only whole exchanges go, so every tool result keeps the call it answers and
/// the request stays well-formed. The system instructions are not part of the
/// history and are never touched; the prompt is not in the history either, so
/// the current intent survives by construction.
pub fn prune_to_budget(history: &mut Vec<RigMessage>, budget: usize) -> Pruned {
    let start = prune_start(history, budget);
    if start > 0 {
        history.drain(..start);
    }
    Pruned {
        dropped: start,
        fits: history_chars(history) <= budget,
    }
}

/// Whether `message` is a provider refusing a request for its size.
///
/// Matched on the phrasings the providers actually use rather than on a status
/// code, because the rejection arrives as a stream error with the body in its
/// text. A false negative costs the user a clear error; a false positive would
/// spend the one recovery attempt on a failure that compaction cannot fix, so
/// the patterns are kept narrow.
pub fn is_context_rejection(message: &str) -> bool {
    let text = message.to_ascii_lowercase();
    const PATTERNS: [&str; 5] = [
        "exceeds the available context size",
        "context length exceeded",
        "context_length_exceeded",
        "maximum context length",
        "reduce the length of the messages",
    ];
    PATTERNS.iter().any(|pattern| text.contains(pattern))
}

/// [`Result`] for a request that must fit.
pub fn fits(estimate: &Estimate) -> Result<()> {
    if estimate.over_budget() {
        Err(estimate.too_large())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::message::{CallId, ToolName};

    fn config(window: u32) -> ContextConfig {
        ContextConfig {
            window,
            tool_result_chars: 1_000,
            evict_chars: 1_000,
            compact_at: 80,
            tail_chars: 1_000,
            output_reservation: 1_000,
            safety_margin: 500,
        }
    }

    fn result(text: &str) -> RigMessage {
        RigMessage::tool_result(
            CallId::from_wire("call_1"),
            ToolName::new("read_file").unwrap(),
            text,
        )
    }

    #[test]
    fn the_estimate_rounds_up_rather_than_down() {
        // Rounding down would let a request through that the provider then
        // rejects, which is the failure the budget exists to prevent.
        assert_eq!(tokens_for(0), 0);
        assert_eq!(tokens_for(1), 1);
        assert_eq!(tokens_for(4), 1);
        assert_eq!(tokens_for(5), 2);
    }

    #[test]
    fn the_budget_is_the_window_less_the_reservation_and_the_margin() {
        let estimate = Estimate::measure(Parts::default(), &config(10_000), &Heuristic);
        assert_eq!(estimate.budget, 8_500);
        assert!(!estimate.exact, "the heuristic is not exact accounting");
        assert_eq!(estimate.label(), "estimated");
    }

    #[test]
    fn a_window_smaller_than_the_reservation_leaves_no_budget() {
        let estimate = Estimate::measure(Parts::default(), &config(100), &Heuristic);
        assert_eq!(estimate.budget, 0);
        assert_eq!(estimate.percent(), 999, "a zero budget must not divide");

        // Nothing fits a zero budget. The smallest request there is, a prompt
        // of one character, is already over it. An empty request measures to
        // zero and so is not "over" anything, which is why the hook refuses on
        // the prompt rather than on the history: a request with no prompt is
        // not a request.
        let smallest = Estimate::measure(
            Parts {
                prompt: 1,
                ..Parts::default()
            },
            &config(100),
            &Heuristic,
        );
        assert!(smallest.over_budget(), "nothing fits a zero budget");
    }

    #[test]
    fn the_estimate_accounts_for_every_part_of_the_request() {
        let parts = Parts {
            system: 400,
            history: 400,
            prompt: 400,
        };
        assert_eq!(parts.total(), 1_200);
        let estimate = Estimate::measure(parts, &config(10_000), &Heuristic);
        assert_eq!(estimate.tokens, 300);
    }

    #[test]
    fn the_history_budget_is_what_the_other_parts_leave() {
        let parts = Parts {
            system: 1_000,
            history: 0,
            prompt: 2_000,
        };
        let estimate = Estimate::measure(parts, &config(10_000), &Heuristic);
        // 8,500 tokens of budget is 34,000 characters, less the 3,000 the
        // fixed parts take.
        assert_eq!(estimate.history_chars(), 31_000);
    }

    #[test]
    fn a_request_over_the_budget_names_both_figures() {
        let parts = Parts {
            system: 0,
            history: 0,
            prompt: 40_000,
        };
        let estimate = Estimate::measure(parts, &config(10_000), &Heuristic);
        let err = estimate.too_large();
        let text = err.to_string();
        assert!(text.contains("8500"), "the budget should be named: {text}");
        assert!(text.contains("10000"), "the size should be named: {text}");
        assert!(
            text.contains("LLM_NUM_CTX"),
            "the error should say what to do: {text}"
        );
    }

    #[test]
    fn a_prune_cuts_only_at_an_exchange_start() {
        // A cut in the middle of an exchange would leave a tool result with no
        // call before it, which the provider rejects.
        let history = vec![
            RigMessage::user("first"),
            RigMessage::assistant("reading"),
            result("a"),
            RigMessage::user("second"),
            RigMessage::assistant("reading"),
            result("b"),
        ];
        assert_eq!(prune_start(&history, 0), 3, "the last exchange is kept");
    }

    #[test]
    fn a_prune_keeps_the_longest_history_that_fits() {
        let history = vec![
            RigMessage::user("first"),
            RigMessage::assistant("reading"),
            result(&"a".repeat(100)),
            RigMessage::user("second"),
            RigMessage::assistant("reading"),
            result(&"b".repeat(100)),
        ];
        // Room for the second exchange and the first user turn, but not the
        // first exchange's result.
        assert_eq!(prune_start(&history, 200), 3);
        // Room for everything.
        assert_eq!(prune_start(&history, 10_000), 0);
    }

    #[test]
    fn a_prune_that_cannot_fit_says_so() {
        let mut history = vec![
            RigMessage::user("first"),
            RigMessage::assistant("reading"),
            result(&"a".repeat(1_000)),
        ];
        let pruned = prune_to_budget(&mut history, 10);
        assert_eq!(pruned.dropped, 0, "the only exchange is kept");
        assert!(!pruned.fits, "and the caller is told it still does not fit");
    }

    #[test]
    fn a_prune_drops_whole_exchanges_and_reports_how_many() {
        let mut history = vec![
            RigMessage::user("first"),
            RigMessage::assistant("reading"),
            result(&"a".repeat(1_000)),
            RigMessage::user("second"),
            RigMessage::assistant("reading"),
            result("b"),
        ];
        let pruned = prune_to_budget(&mut history, 100);
        assert_eq!(pruned.dropped, 3);
        assert!(pruned.fits);
        assert_eq!(history.len(), 3);
        assert!(is_exchange_start(&history[0]));
    }

    #[test]
    fn a_history_with_no_exchange_start_is_dropped_entirely() {
        // There is no point to cut at that leaves a well-formed request: a
        // tool result with no call before it is one the provider rejects, so
        // the whole history goes and the request is the prompt alone.
        let mut history = vec![RigMessage::assistant("reading"), result("a")];
        let pruned = prune_to_budget(&mut history, 0);
        assert_eq!(pruned.dropped, 2);
        assert!(pruned.fits);
        assert!(history.is_empty());
    }

    #[test]
    fn a_context_rejection_is_recognised_and_a_plain_failure_is_not() {
        assert!(is_context_rejection(
            "request (9000 tokens) exceeds the available context size (4096 tokens)"
        ));
        assert!(is_context_rejection(
            "This model's maximum context length is 8192"
        ));
        assert!(!is_context_rejection("connection refused"));
        assert!(!is_context_rejection("the daemon went away"));
    }
}
