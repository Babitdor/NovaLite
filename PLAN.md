# PLAN — a frontend-design prompt and a clearer tool contract

## Goal

Make the agent good at UI/frontend work and precise about which tool to reach
for, by condensing the project's `SKILL.md` (the frontend-design skill) into the
system prompt and sharpening the seven tool descriptions — no new tool, no new
dependency, and no growth of the default `terse` budget beyond two lines.

## What exists now (read this session)

**Prompts — `src/prompts.rs` (207 lines).** Three constants. `SYSTEM_PROMPT`
(the `full` style) walks six numbered stages, `Analyze` to `Finish`.
`TERSE_PROMPT` is the default (`LLM_PROMPT_STYLE`, `config.rs:409`) and is 752
characters. Five unit tests pin the text: every `Stage::ALL` label appears in
`SYSTEM_PROMPT`; `SYSTEM_PROMPT` contains `cargo check`, `unverified` and
`do not retry a call the user denied`; terse is under 900 characters and under
half the full prompt; terse keeps `tool-call interface`, `Read a file before you
edit it`, `same arguments`, `cargo check`, `unverified`.

**Tools — `src/tools.rs`.** Seven `Tool` impls (`read_file` 330, `write_file`
448, `edit_file` 627, `shell` 715, `list_dir` 789, `grep` 881, `glob` 981).
Registered and schema-counted by `for_each_tool!` in `src/agent.rs` over
`tools::TOOL_NAMES`, and the schema characters are added to every request's
budget (`fixed_chars`, `agent.rs:994`). No test pins a description's text, so
the descriptions are free to improve but are paid for on every call.

**The gap.** `frontend` appears nowhere under `src/`. The design guidance the
user wants the agent to have lives only in the repo-root `SKILL.md` (~5.5k
characters), which the model never sees. The tool descriptions state what each
tool does but not which one to reach for, so tool choice is left to the model.

**Already-handled, not in scope.** `RepairToolCalls` in `src/agent.rs` repairs
wrong tool names, salvages malformed arguments and retries a tool call written
as prose. Tool errors already carry the message meant for the model
(`ToolError::into_tool_error`). Neither needs changing.

## Steps

### 1. Pin the new prompt requirements as failing tests

Add to `src/prompts.rs`'s `mod tests`:

- `the_prompt_asks_for_a_design_plan_before_the_markup` — `SYSTEM_PROMPT`
  contains `palette`, `layout`, and the ordering claim (the plan is stated
  before markup is written).
- `the_prompt_names_the_tells_of_a_generated_page` — `SYSTEM_PROMPT` names at
  least three specific defaults to avoid (cream/serif/terracotta, identical
  rounded cards, all-caps eyebrows, decorative numbering).
- `the_prompt_is_honest_about_not_seeing_the_page` — `SYSTEM_PROMPT` contains
  `cannot see the page` and `unverified`: the agent has no browser, so the
  honest limit has to be stated where the design rules are.
- `the_prompt_names_every_registered_tool` — walks
  `crate::tools::TOOL_NAMES.split(", ")` and asserts each name appears in
  `SYSTEM_PROMPT`. This is the tool-use map's test, the same shape as the
  existing stage test: a tool the prompt never routes to is a tool the model
  has to guess about.
- `the_terse_prompt_keeps_the_design_pointer` — `TERSE_PROMPT` contains
  `UI work` and `cannot see the page`.

**Check:** `$env:CARGO_TARGET_DIR="B:\cargo-target-rust-agent"; cargo test --lib prompts::tests`
— the five new tests fail, the five existing ones pass. The
`the_prompt_names_every_registered_tool` failure message names exactly which
tools the prompt never mentions, which is what step 3 then fixes.

### 2. Write the frontend-design section into `SYSTEM_PROMPT`

One section, condensed from `SKILL.md` and written in the prompt's existing
voice, covering: the rendered result is the deliverable; state the four
decisions (subject and audience, four-to-six named hex palette with the one
that carries, typefaces and their roles, the layout concept) before writing
markup; take the material from the subject rather than decoration; spend the
boldness in one place; the named tells of a generated page; the accessibility
floor (phone width, visible focus, `prefers-reduced-motion`, contrast); copy as
design content; and the honest verification limit.

**Check:** `cargo test --lib prompts::tests` — the design tests from step 1
pass, and every pre-existing prompt test still passes (the stage labels, the
`cargo check` / `unverified` / denial wording, and the terse ratio: `SYSTEM_PROMPT`
grew, so terse×2 stays comfortably under it).

### 3. Add the tool-use map to `SYSTEM_PROMPT`

One short block naming each registered tool and the job it is for: find before
reading (`grep` for text, `glob` for paths, `list_dir` to orient); read the
part of a long file you need with `start_line`/`end_line`; `edit_file` for a
change to an existing file, `write_file` for a new one or a whole-file
replacement; `shell` for the project's own build/test command, one command per
call. For UI work specifically: grep the markup for the class names about to
change, read the stylesheet before adding a rule.

**Check:** `cargo test --lib prompts::tests` —
`the_prompt_names_every_registered_tool` passes, and the prompt tests that were
green stay green. If that test already passed in step 1, the current
`SYSTEM_PROMPT` names every tool and this step is dropped rather than padded:
the map is written only for the tools the failure message named.

### 4. Add the two-line design pointer to `TERSE_PROMPT`

Two lines (~140 characters) naming the design discipline and the honest limit,
inside the 148 characters of headroom the 900-character cap leaves.

**Check:** `cargo test --lib prompts::tests` —
`the_terse_prompt_keeps_the_design_pointer`,
`the_terse_prompt_is_actually_terse` and
`the_terse_prompt_keeps_the_load_bearing_rules` all pass. If the cap fails, the
pointer is trimmed rather than the cap raised: the cap is what keeps the
`terse`/`full` comparison meaningful.

### 5. Sharpen the seven tool descriptions in `src/tools.rs`

Each description gains the one thing it is missing — when to reach for this
tool over its neighbour, and what to do with what comes back — without
repeating the parameter schema:

- `read_file`: locate first with `grep`/`glob`; use the line range for a long
  file; a long result is elided at both ends and says how much was dropped.
- `write_file`: for a new file or a whole-file replacement; for a small change
  to an existing file, `edit_file` is the tool.
- `edit_file`: keep the existing text; add that a refusal for a changed file
  means re-reading it (already stated in `read_file`) — one copy only.
- `list_dir`: for orienting in an unfamiliar directory; `glob` finds files by
  name, `grep` finds text.
- `grep`: a regular expression, over a file or a tree; results read
  `path:line: content`; the cap is reported when it bites; `ignored` reaches
  into `.gitignore`d paths.
- `glob`: a pattern from the workspace root, for finding files rather than text.
- `shell`: the platform's shell; one command per call; the file tools for
  reading and writing, this tool for build and test commands.

Add a unit test in `src/tools.rs`'s `tests` module pinning the budget those
descriptions spend, since they are counted into every request: each description
is non-empty and under 500 characters, and the seven together stay under 2500.

**Check:** `cargo test --lib tools::tests` passes, including the new budget
test, and the diff of the seven descriptions is read once against the list
above. Prose has no behavioural test; the schema-budget test is the part of it
that can fail.

### 6. Full verification

**Check:** `cargo test` (all targets), `cargo clippy --all-targets` and
`cargo fmt --check` clean. Two `tests/stream_render.rs` failures
(`messages_are_separated_by_blank_lines`,
`text_stream_paints_into_the_transcript`) pre-date this work from the
interrupted UI restyle and are reported as baseline, not as a regression, only
after reproducing them on the same revision without this diff.

## Not doing

- **No new tool.** A `design_guide` tool would return static text that a prompt
  can carry for free in the `full` style; the user chose prompt-only.
- **No browser or screenshot tool.** The agent cannot see a rendered page and
  no tool for it exists in this build; step 2 states that limit rather than
  pretending to verify visually.
- **`SKILL.md` is left alone.** It is the reference the prompt is condensed
  from, not a target.
- **No change to `LLM_PROMPT_STYLE`'s default.** `terse` stays the default; the
  design pointer is what keeps UI work from being unprompted in that arm.

## Outcome

Executed. Evidence:

| Step | What landed | Check and result |
|------|-------------|------------------|
| 1–2 | `SYSTEM_PROMPT` gained a "Which tool to use." paragraph and the frontend-design section condensed from `SKILL.md`; `TERSE_PROMPT` gained the two-line UI pointer | `cargo test --lib prompts::tests` → 10 passed, 0 failed |
| 2 | Prompt-integrity guards added: tool coverage, design-before-markup, generated-page tells, "cannot see the page", no tripled newline, wrapped-sentence line break, terse pointer | same run; the tells test failed first on purpose, then passed |
| 3 | Rendered text of both constants dumped and read once to check the escaped source actually reads as intended (temporary test, removed after) | read |
| 4–5 | 4 of the 7 tool descriptions rewritten: `read_file`, `write_file`, `list_dir`, `glob` | `cargo test --lib tools::tests` → 25 passed; new `the_tool_descriptions_stay_inside_their_budget` passes |
| 6 | Whole suite | `cargo test` → 0 failed across all targets, `tests/stream_render` 48/48 |

Deviations from the plan as written:

- **4 tool descriptions changed, not 7.** `edit_file`, `grep` and `shell` already
  said what the new prompt paragraph says — the edit/whole-file division, the
  ignore rules and the capped-result note, the shell name and "prefer the file
  tools". Rewriting them would have been churn.
- **Budget caps are 600 per description and 2600 for the seven, not 500/2500.**
  2500 left no room for a legitimate sentence; the point of the test is to catch
  a runaway paragraph, not to hold a ceiling that invites shaving accuracy.
- **The two `stream_render` failures were not a baseline.** An earlier run during
  the interrupted restyle failed `messages_are_separated_by_blank_lines` and
  `text_stream_paints_into_the_transcript`; this run is 48/48 green, so no
  exclusion is needed and none was written into the plan.
- **No prompt mechanism claim was recorded to memory.** The rendered text is
  verified by assertion and by the one-time dump; an explanation of the source
  file's escaping is not something this work established, so it is not written
  down as fact.
