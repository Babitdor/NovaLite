# PLAN — streaming identity, approval targets, tool-row clarity

## Goal

Fix the confirmed defects in nova-lite's tool streaming and path approval, and
make a failed or refused tool call readable without expanding it, while keeping
the existing architecture, safety properties, and test suite intact.

## Confirmed findings

Read from `src/session.rs`, `src/event.rs`, `src/app.rs`, `src/runtime.rs`,
`src/workspace.rs`, `src/render.rs`, and rig 0.44's `StreamEvent`:

1. **Argument streaming loses part identity.** `StreamEvent::Arguments { part,
   json }` is translated to `Event::ToolArgsDelta(String)`. `Emitter` appends
   every part's fragments into one `args: String`, and `App` appends them into
   one `pending_args` and writes the result to `tool_idx` (the newest block).
   Two calls whose arguments stream at the same time corrupt each other's live
   display and activity row. The committed call overwrites `args` later, so the
   final transcript hides the corruption; it is visible only while streaming.
2. **A symlink escape asks about the wrong directory.** `Runtime::resolve_asking`
   catches `OutsideWorkspace` from `resolve_existing` (which canonicalises and
   finds the real target outside), then calls `Workspace::target_of`, which is
   lexical. For `link/secret.txt` where `link` points outside, the prompt shows
   a path that looks inside and grants `root/link`. `Workspace::grant` records
   that lexical path, but the real target is outside it, so the re-resolution
   fails again. The user approves and is still refused; the question repeats on
   every access.
3. **A failed or refused call hides its reason.** The collapsed tool row shows
   only the line count and `failed` / `denied`; the first line of the result,
   which says why, requires expanding the card.
4. **History on cancellation is a known design limit, not a regression.** rig's
   `PromptResponse.messages` is only available on `FinalResponse`, and
   `drive()` deliberately appends history only then. A user cancellation drops
   the stream before rig can return `PromptError::Cancelled { chat_history }`.
   The app keeps the user prompt and the runtime journal keeps the effects.
   Reconstructing a partial assistant/tool history is a separate design change;
   it is out of scope for this plan.

## Steps

### 1. Add a failing end-to-end test for concurrent argument streaming

Add `concurrent_tool_arguments_stay_with_their_own_call` to
`tests/stream_render.rs`. Script two calls with interleaved
`ToolCallNameDelta` / `ToolCallArgumentsDelta` fragments (`read_file` part 0 and
`grep` part 1), sample the app after each event, and assert that while both
placeholders are streaming each block holds only its own arguments and each
`ToolCall` activity row names its own call.

**Check:** `cargo test --test stream_render concurrent_tool_arguments` fails on
the current code because the first block's `args` stays empty while the second
receives both calls' fragments.

**Status: done.** The test failed for the expected reason:
`the grep block holds arguments that are not its own: "{\"path\":\"Cargo.toml\"}"`.
The first block stayed empty; the second held the first call's fragment.

### 2. Carry part identity through argument streaming

- `Emitter` keeps one buffer per part (`BTreeMap<usize, String>`) and flushes /
  offers each part separately.
- `Event::ToolArgsDelta` becomes `{ part: usize, json: String }`.
- `ToolBlock` keeps the raw JSON being streamed so the display can be rebuilt;
  `App` routes a delta to the block with `streaming && part == Some(part)`,
  renders `rig::streaming::parse_partial_arguments` through the existing
  `summarize_args`, and updates the activity row carrying that part.
- `Activity` gains `part: Option<usize>`, set only by `ToolCallStreaming`;
  `ToolCallStarted` replaces the matching row by part, falling back to the
  existing title lookup when the provider never opened a part.
- Remove the global `pending_args` field and its clears.

**Check:** `cargo test --test stream_render concurrent_tool_arguments` passes,
and `cargo test --test stream_render` stays green.

**Status: done.** `concurrent_tool_arguments_stay_with_their_own_call` passes,
and all 48 `stream_render` tests pass. The emitter buffers by part, the event
carries the part, the block rebuilds its summary from its own raw JSON, and the
activity row is matched by part before the title fallback.

### 3. Add a failing test for approval through a symlink out of the workspace

Add `a_path_through_a_link_outside_is_asked_about_by_its_real_target` to
`tests/approval.rs`. Create a workspace root, a sibling `outside` directory
with `secret.txt`, and a directory link `project/link -> outside` using
`common::make_dir_link` (return early if the platform refuses the link).
Approve everything, read `link/secret.txt`, and assert:

- the read succeeds after one access question;
- the access prompt's preview and detail name the **real** outside directory;
- `workspace.reach_of("cat link/secret.txt")` reports the real outside file.

**Check:** `cargo test --test approval through_a_link` fails on the current
code (the read is refused after approval, and the prompt names the link).

**Status: done.** The test failed for the expected reason: after the access
question was approved, `read link/secret.txt` still returned
`'link/secret.txt' is outside the workspace`.

### 4. Resolve approvals against the real target

- Add `Workspace::real_target_of(&self, path)` that resolves lexically and then
  canonicalises the deepest existing ancestor, **without** a containment check.
  Reuse the existing walk in `real_target` by splitting it into a pure
  canonical-ancestor step plus the `allows` test.
- `Runtime::resolve_asking` uses `real_target_of` for the prompt and the grant,
  so the user sees and grants the directory the path really reaches.
- `Workspace::target_of_word` canonicalises a command word only when the whole
  path exists; a nonexistent target keeps its lexical form. This makes the
  shell reach scan see an existing link's real target without changing the
  display of paths that do not exist.

**Check:** `cargo test --test approval through_a_link` passes;
`cargo test --test approval` and `cargo test workspace` stay green.

### 5. Show the reason on a collapsed non-OK tool row

In `render_tool_block`, after the collapsed header, add one indented line with
the first non-empty line of the result when `block.done` and the outcome is
`Failed`, `Denied`, or `Cancelled`. Color it with the outcome's existing color.
Successful calls keep the one-row form.

**Check:** a new test in `tests/stream_render.rs` feeds a failed result and
asserts the collapsed transcript contains the error's first line without the
block being expanded; `cargo test --test stream_render` stays green.

### 6. Run the full gate and a live smoke test

**Check:**

- `cargo fmt --check`
- `cargo test --all-targets --locked`
- `cargo clippy --all-targets --all-features --locked -- -D warnings`
- `$env:LLM_MODEL='qwen3.5:0.8b'; "y" | cargo run --quiet --example live -- "list the files in src, read Cargo.toml, and tell me the package name"`

The live run must finish with a final answer and at least two tool results.

## Critique (fresh pass)

Questions from `ds-plan-loop`:

- **Missing step?** No migration or config change is needed. Formatting and
  docs are covered by the comments in each step and the gate in step 6. The
  history limit is explicitly excluded in finding 4 rather than silently
  dropped.
- **Unchecked assumption?** The end-to-end test assumes rig's mock emits one
  `Start` per tool part and preserves part indices under interleaving. That is
  how `StreamEvent::Start { part, .. }` and the mock's `call_index` work in rig
  0.44; step 1 will confirm it by failing for the expected reason before any
  production change. `parse_partial_arguments` is re-exported at
  `rig::streaming::parse_partial_arguments` (rig-core `json_utils` re-export),
  so no new dependency is needed.
- **Simpler path?** Yes for one part: the shell-scan canonicalisation could be
  dropped. It is kept because the same link that defeats `read_file` also makes
  `gate_reach` report nothing, and it is one `canonicalize().unwrap_or(...)` on
  an existing path. It is still bounded to the approval layer, not a sandbox
  claim.

Revisions after critique:

- Step 1 now says the test must fail **for the expected reason** (first block
  empty), not merely fail.
- Step 3 asserts the prompt names the real directory, not just that the read
  eventually succeeds, so the test pins the UX defect as well as the grant.
- Step 4 splits the existing `real_target` walk rather than duplicating it, so
  the containment check and the display path cannot drift.
- Step 5 is limited to non-OK outcomes; a successful call's one-row form and
  the existing collapsed-by-default test are untouched.

---

# PLAN 2 — OpenCode-style TUI

## Goal

nova-lite's TUI adopts OpenCode's layout language: terminal-default background,
no left rail, no live activity side panel, `❯`-prefixed user prompts, plain
assistant text, dim tool rows, and a model/context footer bar — keeping the
existing architecture, folding, scroll behaviour and safety properties intact.

## Decisions

- **TUI render only.** `App.activity` / `Activity` / `ActivityKind` keep being
  maintained by the event pipeline; the render path stops drawing them. They
  stay read by `examples/eval.rs`, `examples/live.rs` and
  `examples/wifi_scanner.rs`, so `src/app.rs` is not edited beyond stale
  comments. Note: for the TUI the activity vector becomes write-only state.
- Grouping/folding, the caret, the top-header spinner, scroll clamping and
  `BodyCache` all stay as they are.
- The rail's TOOLS list is dropped (decorative; the transcript and `/help`
  carry it). Its numbers (model, ctx%, calls, tokens) move to the footer.
- The input box's `➤` send button is dropped (OpenCode has none); Enter still
  sends, the `❯` prompt and placeholder stay.
- `BG` becomes `Color::Reset` (terminal default). Safe: no test asserts a
  background, and the only `.fg(BG)` chip is the `➤` button this plan deletes.

## Steps

### 1. Layout rewrite: single column, no rail, no activity panel, no painted page

One coherent rewrite of `draw` and the palette, plus the cleanup it orphans:

- `draw` becomes single-column at every width: delete `render_activity`,
  `activity_style`, `activity_width`, `ACTIVITY_MIN_COLUMN`/`_MIN_WIDTH`/
  `_MAX_WIDTH`, `render_rail`, `RAIL_WIDTH`, `RAIL_MIN_SCREEN`, and the `RAIL`
  colour.
- `BG` → `Color::Reset` at every background site (page, transcript, header,
  footer); the input box keeps its rounded border but loses its PANEL fill.
- Footer takes over the rail's numbers: model name chip at the front, then the
  existing ctx% indicator, calls and tokens — reuse `short_count`, which today
  has only rail callers. Keep the segment-dropping logic.
- Live reasoning moves into the transcript as a dim italic tail (last ~6
  lines) under the streaming assistant block, extracted from
  `render_activity`'s tail code. It re-wraps each frame and deliberately does
  not go through `BodyCache` (it changes every tick).
- Cleanup required by `-D warnings`: trim the `use crate::app::{Activity,
  ActivityKind, …}` import, and keep or delete `section` (render.rs:868)
  according to whether a surviving call site (the group heading) still uses
  it.
- Stale docs updated in the same diff: the module doc + ASCII diagram
  (render.rs:2-10) and the rail/panel/header comments at render.rs:633-635,
  835, 931-933, 1053-1056, 1639-1644; `src/app.rs:297`, `:733-737`, `:2733`;
  the rail reference in `Cargo.toml:13`; the activity-panel premise in
  `tests/redraw_cost.rs:107-110`.

**Check:** `cargo test --test stream_render` passes after replacing
`tool_call_and_result_appear_in_the_activity_timeline` with a transcript-row
test, replacing
`narrow_terminal_drops_the_activity_panel_without_panicking` with a
narrow-terminal-renders-the-transcript test, deleting
`the_rail_and_the_activity_panel_are_framed_by_rules`, deleting the now-unused
`activity_column` helper, and dropping
`the_activity_panel_does_not_grow_with_the_terminal`; `cargo test --test
model_picker` passes after removing the two `.expect("the activity panel…")`
assertions in `the_picker_stays_inside_the_conversation_panel`; `cargo
clippy --all-targets -- -D warnings` is clean (this is what catches the
orphans); `cargo run --example preview` output has no rail column.

### 2. Message rows go OpenCode-style

- User: `❯ content` with continuations indented two spaces, in ACCENT. This is
  new code, not a `wrap_prefixed` call — `wrap_prefixed` applies one prefix to
  every line. Reuse the hand-rolled `❯ ` + `  ` shape already implemented in
  `render_input`; leave the shared `BODY_INDENT` (`"   "`, used by tool bodies
  and group digests) alone.
- Assistant: the per-message header row (marker, label, age, header spinner)
  is removed; the markdown body stands alone. Working state stays in the top
  header row via `status_of`; the caret stays on the body.
- System: dim italic body, no header.
- Delete `spinner_frame` and `age` once orphaned. The dim "responding…" word
  above a streaming tool group stays.

**Check:** stream_render passes with `text_stream_paints_into_the_transcript`
and `messages_are_separated_by_blank_lines` rewritten (the `You`/`Agent`
labels no longer exist; `status_of`'s "Agent online" string does and is
asserted as the top-header status instead), plus a new test asserting a user
message renders `❯ ` and no `You` label; `cargo test --test
render_robustness` passes after `a_streaming_run_says_it_is_responding_and_a_finished_one_does_not`
drops its `a user header` expectations, its spinner count is retargeted to the
top header, and its `starts_with('◆')` probe is repointed.

### 3. Tool rows restyled; expanded output loses the box

Collapsed rows keep their one-line form (disclosure, outcome glyph, name,
args, line-count tail). Expanded: replace the `╭─╮ / │ │ / ╰─╯` card
(`card_edge`, `card_row`, `pad`) with OpenCode-style output — every result
line prefixed `▏ ` in the border colour, verdict line unchanged, no top or
bottom rules. Group headings and folding are untouched.

**Check:** `expanding_a_tool_block_reveals_its_result` and the related card
tests updated and green (`cargo test --test stream_render`); no `╭`/`│`/`╰`
glyphs remain in transcript assertions.

### 4. Banner, overlays, full gate

Banner draws on the terminal-default background; picker, approval and slash
suggestions keep their PANEL fill so they occlude. Then the full gate.

**Check:**

- `cargo test --test banner` passes
- `cargo fmt --check`
- `cargo clippy --all-targets --all-features --locked -- -D warnings`
- `cargo test --all-targets --locked`
- `cargo run --example preview` (default, `empty`, `expanded`, `folded`)
  printed for an eyeball diff of the new layout

## Critique (fresh pass)

A separate reviewer read PLAN 2 against `src/render.rs`, `src/app.rs` and the
test suites. Findings that changed the plan:

- **Step 6 could not have passed.** `tests/model_picker.rs:301`
  `the_picker_stays_inside_the_conversation_panel` hard-expects the activity
  panel at `:317` and `:326`. Folded into step 1.
- **A cleanup step was missing.** Deleting the rail/panel orphans `section`,
  `RAIL`, `short_count` and the `Activity`/`ActivityKind` import under the
  crate's `-D warnings` gate, plus the stale docs listed in step 1. Now
  explicit.
- **Step 3's premise was wrong.** `wrap_prefixed` (render.rs:1245) repeats one
  prefix on every line, so the `❯ ` / two-space continuation is new code;
  `render_input` already implements that shape. Corrected.
- **Tests the plan had not named:** `render_robustness.rs:350` (user-header
  expects, spinner count, `◆` probe), `stream_render.rs:153` and `:785`
  (`You`/`Agent` assertions), and the orphaned `activity_column` helper at
  `:1194`. All listed now.
- **Verified safe:** nothing outside `render.rs` references the deleted render
  helpers; `Activity` is still read by three examples, so `app.rs` stays
  compile-clean; no test asserts a background; `card_edge`/`card_row`/`pad`
  have no callers outside `render_tool_block`.
- **Simplification taken:** steps 1, 2 and 5 of the first draft all rewrote
  `draw` and would have required three separate test rounds on one function;
  they are now one step, with the orphan cleanup attached instead of implied.
