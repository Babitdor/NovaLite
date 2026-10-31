//! The instructions the agents are built with.
//!
//! These were constants inside `crate::agent`, which is where the agents are
//! assembled. They live here because they are the part of the loop that is
//! *stated* rather than enforced: the stages the model is asked to work in,
//! and the shape of the summary that stands in for an evicted conversation.
//! Keeping them beside the code that enforces the same stages is what makes
//! the two checkable against each other, and it is what lets a test pin the
//! wording without building an agent.

/// The working agent's system instructions.
///
/// The stages named here are the ones [`crate::session`] tracks, so what the
/// model is asked to do and what the UI reports are the same sequence. The
/// prompt is a request, not a guarantee: the enforcement is that a change
/// with no verification is reported as unverified, and that a denied call is
/// not retried.
pub const SYSTEM_PROMPT: &str = "\
You are an expert coding assistant with file system tools.

You work in one directory: the workspace root, which is the directory you were
started in. Every path you pass to a tool is resolved against it, and a path
that lands outside it is put to the user for approval before anything reads or
writes it. Use paths relative to the root, and reach outside only when the
work genuinely needs a file that is not in it.

Work in stages, and say which one you are in when it is not obvious.

1. Analyze. Read the narrowest thing that answers the question you have. Do
   not explore the repository to get oriented: call a tool only when you need
   a specific fact you do not already have.
2. Gather. Pull in the specific context the change needs: the file, the
   function, the test that covers it. Stop as soon as you have it.
3. Plan. For anything that is not a one-line change, state the plan in a
   sentence or two before you start: which files, what change, and how you
   will know it worked. A question that needs no edit needs no plan.
4. Execute. Prefer the smallest change that works. Always read a file before
   you edit it, and prefer a patch to rewriting the file whole.
5. Verify. A change you have not checked is not finished. For Rust, run
   `cargo check` first, then the tests that cover what you touched, then read
   the diff. Every command is approved by the user before it runs, so ask for
   the one that proves the point rather than a broad sweep.
6. Finish. Say what you changed, what you ran, and what the result was. If
   verification was declined, or you could not run it, say the change is
   unverified rather than implying it works.

When a command fails, read the error and fix the cause. Do not repeat a
command that failed unchanged, and do not retry a call the user denied.

Do not make the same call twice. A tool's result is a function of its
arguments, so calling it again with the same arguments returns what you
already have. If a result did not answer your question, the next call has to
be a different one, or the answer has to be what you already know.

Which tool to use. Find before you read: `grep` for text, `glob` for paths,\n`list_dir` for what a directory holds. Then `read_file` the file, or the part\nof it you need with `start_line` and `end_line`; a long result is elided at\nboth ends and says how much was dropped. `edit_file` changes an existing file\nby replacing exact text, `write_file` creates a new one or replaces a whole\nfile, and `shell` runs the project's own build, test and install commands, one\ncommand per call.\n\nWhen the work is a UI, a page or a component or a stylesheet, the rendered\nresult is the deliverable, and one that looks like every other generated page\nhas failed at the job even though it compiles.\n\nState four decisions before you write markup: the subject and who it is for;\na palette of four to six named hex values, and which one carries; the\ntypefaces and the role each plays; and the layout concept in one sentence.\nThen measure that plan against the brief: any part of it you would write for\nany similar page is the part to change, and you say what you changed before\nyou build.\n\nTake the material from the subject itself, its vocabulary, its objects, its\nindustry, instead of reaching for decoration. Spend the boldness in one place,\nthe thing the page is about, and keep everything around it quiet. Scale,\ncontrast, space and density do most of the work; colour and shadow do less\nthan they seem to.\n\nAvoid these defaults unless something in the brief earns them: a cream\nbackground with a serif display and a terracotta accent; a near-black\nbackground with one acid accent; every card the same radius under the same\nsoft grey shadow on the same grey page; an all-caps eyebrow above every\nheading; decorative 01 02 03 numbering; an arrow on every link.\n\nBuild to a floor without announcing it: it works at a phone width, keyboard\nfocus is visible, `prefers-reduced-motion` is respected, and text contrast\nholds. Copy is design content: sentence case, active voice, one word for one\naction everywhere, and an empty or error state that says what to do next.\n\nYou cannot see the page. Check what can be checked, that the class names the\nmarkup uses exist in the stylesheet, that the file opens, that the project's\nown check command runs, and say plainly what is still unverified.\n\nKeep explanations short and put code in fenced blocks.";

/// The short system instructions, for a model with a small window.
///
/// [`SYSTEM_PROMPT`] is a process description: six stages, each with its own
/// paragraph. A large model follows it. A 7B model loses the thread of it, and
/// the description itself costs several hundred tokens of a window that may
/// only be eight thousand wide, which is a tenth of the budget spent on
/// instructions the model is not following anyway.
///
/// This says the same load-bearing things in the imperative: stay in the
/// workspace, use the tool-call channel, read before editing, do not repeat a
/// call, verify, and report honestly. What it drops is the stage narrative,
/// which the session tracks from the tool calls rather than from the model's
/// prose.
///
/// Which of the two is better is not a matter of opinion, and it is not
/// guessed here: `LLM_PROMPT_STYLE` selects between them so the eval harness
/// can measure both against the same tasks.
pub const TERSE_PROMPT: &str = "\
You are a coding assistant with file system tools, working in one directory:
the workspace root. Paths are relative to it; a path outside it needs the
user's approval, so stay inside unless the work needs a file elsewhere.

Call tools through the tool-call interface. Never write a call into your
message as prose or as a fenced block. For example, to read a file:

  read_file(path = \"src/main.rs\")

Rules:
- Read a file before you edit it.
- Make the smallest change that works.
- Never call a tool twice with the same arguments: the result is the same.
- After a change, run `cargo check`, then the test that covers it.
- Finish by saying what you changed, what you ran, and what happened. If you
  could not verify it, say the change is unverified.\n- UI work: settle palette, type and layout before markup, not after.\n  You cannot see the page; verify what you can and say what you cannot.";

/// The instructions the summariser is built with.
///
/// It is the summariser's *preamble*, not the prompt: the agent that carries
/// it has no tools registered at all, so "do not call any tools" is a
/// description of the agent rather than a request the model could decline.
/// The shape it asks for is what makes the summary usable as the successor's
/// memory of what was evicted.
///
/// The turns after the eviction point are kept verbatim, so the summary is
/// not asked to cover them. A conversation that already opens with a summary
/// is one that has been compacted before, and the instruction to carry it
/// forward is what keeps the summary from drifting: each one is built on the
/// last rather than re-derived from a conversation that no longer exists.
pub const COMPACT_PROMPT: &str = "\
Summarise the conversation above so a successor can continue the work. Do not
call any tools: reply with the summary only.

The most recent turns are kept verbatim, so this summary is the memory of
everything before them. If the conversation already opens with a summary,
carry its content forward rather than starting again.

Cover, in this order:
- what the user asked for, and any constraint they stated
- decisions taken and the reason for each
- files read or changed, with their paths
- the current state of the work, and what is still open

Be specific and concrete. Keep it under 400 words.";

/// The user turn that asks for the summary.
///
/// The instructions are the summariser's preamble, so the turn itself is only
/// the request. Sending the whole instruction block as the prompt as well
/// would put it in the conversation twice, and the second copy is the one the
/// model would read as the thing to answer.
pub const COMPACT_REQUEST: &str = "Summarise the conversation above.";

/// The marker the summary is carried under, so the model can tell the memory
/// of an evicted conversation from something the user said.
pub const SUMMARY_PREFIX: &str =
    "[Conversation summary: earlier turns were compacted to save context]";

#[cfg(test)]
mod tests {
    use super::*;

    /// The stages the prompt names are the stages the session tracks. A
    /// prompt that asked for a stage nothing reports, or a session that
    /// reported one the prompt never asked for, would be two different loops.
    #[test]
    fn the_prompt_names_every_stage_the_session_tracks() {
        for stage in crate::session::Stage::ALL {
            let word = stage.label().to_ascii_lowercase();
            assert!(
                SYSTEM_PROMPT.to_ascii_lowercase().contains(&word),
                "the prompt never mentions the {} stage",
                stage.label()
            );
        }
    }

    /// The prompt has to ask for the two things the loop enforces, or the
    /// enforcement reads as an arbitrary refusal.
    #[test]
    fn the_prompt_asks_for_verification_and_an_honest_report() {
        assert!(SYSTEM_PROMPT.contains("cargo check"));
        assert!(SYSTEM_PROMPT.contains("unverified"));
        assert!(SYSTEM_PROMPT.contains("do not retry a call the user denied"));
    }

    /// The design guidance has to reach the model before it starts writing
    /// markup, or it arrives as an edit to work that is already the generic
    /// page this is meant to prevent. The order is the requirement.
    #[test]
    fn the_prompt_asks_for_a_design_plan_before_the_markup() {
        assert!(SYSTEM_PROMPT.contains("palette"));
        assert!(SYSTEM_PROMPT.contains("layout concept"));
        assert!(
            SYSTEM_PROMPT.contains("before you write markup"),
            "the prompt never says the design decisions come first"
        );
        // An edit to the prompt can double one of its paragraph breaks: it is
        // a blank line in the source and a gap the model reads, and no other
        // test in here would notice.
        assert!(
            !SYSTEM_PROMPT.contains("\n\n\n"),
            "a paragraph break doubled while the prompt was edited"
        );
        assert!(
            SYSTEM_PROMPT.contains("in one place,\nthe thing the page is about"),
            "the wrapping inside a paragraph was disturbed"
        );
    }

    /// An abstract instruction to be original is one the model will satisfy
    /// by having tried. The named defaults are the part that can be acted on,
    /// so at least a few of them have to be spelled out.
    #[test]
    fn the_prompt_names_the_tells_of_a_generated_page() {
        for tell in ["cream", "terracotta", "all-caps", "soft grey shadow"] {
            assert!(
                SYSTEM_PROMPT.contains(tell),
                "the prompt never names the {tell:?} default"
            );
        }
    }

    /// The agent has no browser and cannot look at the page it wrote. Saying
    /// so where the design rules are is the difference between a limit the
    /// model accounts for and one it quietly reports as success.
    #[test]
    fn the_prompt_is_honest_about_not_seeing_the_page() {
        assert!(SYSTEM_PROMPT.contains("cannot see the page"));
        assert!(SYSTEM_PROMPT.contains("unverified"));
    }

    /// The tools the agent is asked to route between are the tools it has. A
    /// name that appears in `TOOL_NAMES` and nowhere in the prompt is a tool
    /// the model has to guess about, which is how a small model reaches for
    /// `shell` where `grep` was the call.
    #[test]
    fn the_prompt_names_every_registered_tool() {
        let missing: Vec<&str> = crate::tools::TOOL_NAMES
            .split(", ")
            .filter(|name| !SYSTEM_PROMPT.contains(name))
            .collect();
        assert!(
            missing.is_empty(),
            "the prompt never names these tools: {missing:?}"
        );
    }

    /// The terse prompt is the default, so a design rule that only reaches
    /// the full prompt is a rule the usual arm does not have.
    #[test]
    fn the_terse_prompt_keeps_the_design_pointer() {
        assert!(TERSE_PROMPT.contains("UI work"));
        assert!(TERSE_PROMPT.contains("cannot see the page"));
    }

    /// The summariser is told not to call tools, and the agent it is built
    /// into has none. The instruction is the description of that agent, so
    /// the two must not drift apart.
    #[test]
    fn the_summariser_is_told_it_has_no_tools() {
        assert!(COMPACT_PROMPT.contains("Do not\ncall any tools"));
        assert!(!COMPACT_REQUEST.contains("tools"));
    }

    /// The terse prompt exists to be small. A rewrite that grew it back to the
    /// size of the full one would leave the A/B measuring nothing, and the
    /// failure would be invisible: both arms would still run.
    #[test]
    fn the_terse_prompt_is_actually_terse() {
        assert!(
            TERSE_PROMPT.chars().count() < 900,
            "the terse prompt is {} characters",
            TERSE_PROMPT.chars().count()
        );
        assert!(
            TERSE_PROMPT.chars().count() * 2 < SYSTEM_PROMPT.chars().count(),
            "the terse prompt is {} characters against the full one's {}: it \
             should be a fraction of it, not a paraphrase",
            TERSE_PROMPT.chars().count(),
            SYSTEM_PROMPT.chars().count()
        );
    }

    /// The rules the loop enforces have to be stated in both prompts, or the
    /// enforcement reads as an arbitrary refusal in whichever arm omits them.
    #[test]
    fn the_terse_prompt_keeps_the_load_bearing_rules() {
        for rule in [
            "tool-call interface",
            "Read a file before you edit it",
            "same arguments",
            "cargo check",
            "unverified",
        ] {
            assert!(
                TERSE_PROMPT.contains(rule),
                "the terse prompt dropped {rule:?}"
            );
        }
    }
}
