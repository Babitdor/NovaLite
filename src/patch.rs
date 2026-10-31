//! Applying search-and-replace blocks to one file.
//!
//! The whole module is a pure function over text. Nothing here reads or writes
//! a file, which is what makes the rules testable on their own and what makes
//! "every block is validated before anything is committed" true by
//! construction: [`apply`] either returns the complete new text or returns a
//! conflict, and the caller has nothing to write in the second case.
//!
//! # Why not a line-based diff format
//!
//! A unified diff carries line numbers, and line numbers are the one thing
//! that goes stale the moment anything above the hunk changes. A search string
//! is anchored to the text itself, so it either matches or it does not, and
//! "it does not" is a fact the model can act on. The cost is that the search
//! string has to be unique, which is why an ambiguous match is an error rather
//! than a guess.

use std::fmt;

/// One replacement: find `old`, put `new` in its place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    /// The exact text to find. Must not be empty.
    pub old: String,
    /// What to put in its place. May be empty, which deletes.
    pub new: String,
    /// Replace every occurrence instead of requiring exactly one.
    ///
    /// Off by default. A rename inside one file is the case this is for, and
    /// it is the case where "every occurrence" is what the caller means; for
    /// everything else, a second match is a sign the search string was not
    /// specific enough, and guessing which one was meant is how an edit lands
    /// in the wrong place.
    pub replace_all: bool,
}

impl Block {
    /// A block that must match exactly once.
    pub fn new(old: impl Into<String>, new: impl Into<String>) -> Self {
        Self {
            old: old.into(),
            new: new.into(),
            replace_all: false,
        }
    }

    /// A block that replaces every occurrence.
    pub fn all(old: impl Into<String>, new: impl Into<String>) -> Self {
        Self {
            old: old.into(),
            new: new.into(),
            replace_all: true,
        }
    }
}

/// Why a set of blocks could not be applied.
///
/// Every variant names the block that failed, because a patch of five blocks
/// that fails on the third has to say *which* third: the model's next move is
/// to fix that one block, and it cannot do that from "the patch failed".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conflict {
    /// The search text was empty.
    EmptySearch { index: usize },
    /// The search text was not in the file.
    NotFound { index: usize, old: String },
    /// The search text was in the file more than once.
    Ambiguous {
        index: usize,
        old: String,
        matches: usize,
        /// The 1-based lines the first few matches start on, so the model can
        /// add the surrounding context that tells them apart.
        lines: Vec<usize>,
    },
    /// Every block applied, and the file is byte-for-byte what it was.
    NoChange,
}

/// How many match lines an [`Conflict::Ambiguous`] reports.
const REPORTED_MATCHES: usize = 5;

/// How much of a search string an error message quotes.
const SNIPPET_CHARS: usize = 60;

/// The first line of `text`, shortened, for an error message.
///
/// A search string can be a whole function. Quoting all of it would bury the
/// part of the message that says what went wrong, and the model already has
/// the string it sent.
fn snippet(text: &str) -> String {
    let first = text.lines().next().unwrap_or("");
    let mut out: String = first.chars().take(SNIPPET_CHARS).collect();
    if first.chars().count() > SNIPPET_CHARS || text.lines().count() > 1 {
        out.push('…');
    }
    out
}

/// The 1-based line each occurrence of `needle` in `haystack` starts on.
fn match_lines(haystack: &str, needle: &str) -> Vec<usize> {
    let mut lines = Vec::new();
    let mut from = 0;
    while let Some(at) = haystack[from..].find(needle) {
        let absolute = from + at;
        lines.push(haystack[..absolute].matches('\n').count() + 1);
        // Step past the start of this match. Stepping past its *end* would
        // miss an occurrence that overlaps this one, and the count is what
        // decides whether the block is ambiguous.
        from = absolute + needle.len().max(1);
        if from > haystack.len() {
            break;
        }
    }
    lines
}

/// Apply `blocks` to `original`, in order, against a working copy.
///
/// Each block sees the result of the ones before it, so a later block can
/// match text an earlier one introduced. Nothing is returned unless every
/// block applies: a patch of five blocks that fails on the third leaves the
/// caller with the original text and a conflict, never with a half-applied
/// file.
///
/// The text is treated as an opaque string. Line endings, a byte-order mark, a
/// missing final newline and every byte the blocks do not touch come through
/// unchanged, because nothing here normalises anything.
pub fn apply(original: &str, blocks: &[Block]) -> Result<String, Conflict> {
    let mut working = original.to_string();

    for (index, block) in blocks.iter().enumerate() {
        if block.old.is_empty() {
            return Err(Conflict::EmptySearch { index });
        }

        let lines = match_lines(&working, &block.old);
        if lines.is_empty() {
            return Err(Conflict::NotFound {
                index,
                old: snippet(&block.old),
            });
        }
        if lines.len() > 1 && !block.replace_all {
            return Err(Conflict::Ambiguous {
                index,
                old: snippet(&block.old),
                matches: lines.len(),
                lines: lines.into_iter().take(REPORTED_MATCHES).collect(),
            });
        }

        working = if block.replace_all {
            working.replace(&block.old, &block.new)
        } else {
            working.replacen(&block.old, &block.new, 1)
        };
    }

    if working == original {
        return Err(Conflict::NoChange);
    }
    Ok(working)
}

impl fmt::Display for Conflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySearch { index } => write!(
                f,
                "edit {}: the search text is empty. Give the exact text to replace.",
                index + 1
            ),
            Self::NotFound { index, old } => write!(
                f,
                "edit {}: {old:?} was not found. The text must match the file exactly, \
                 including indentation; read the file again and copy the text from it.",
                index + 1
            ),
            Self::Ambiguous {
                index,
                old,
                matches,
                lines,
            } => {
                write!(
                    f,
                    "edit {}: {old:?} matches {matches} places (lines {}). Include more \
                     surrounding context so it matches once, or set replace_all to change \
                     every one.",
                    index + 1,
                    lines
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Self::NoChange => write!(
                f,
                "the patch would leave the file exactly as it is; there is nothing to apply"
            ),
        }
    }
}

impl std::error::Error for Conflict {}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(old: &str, new: &str) -> Vec<Block> {
        vec![Block::new(old, new)]
    }

    #[test]
    fn a_single_match_is_replaced() {
        let out = apply("let a = 1;\nlet b = 2;\n", &one("let b = 2;", "let b = 3;")).unwrap();
        assert_eq!(out, "let a = 1;\nlet b = 3;\n");
    }

    #[test]
    fn an_empty_search_is_refused() {
        // An empty search matches at every position, so "exactly one match" is
        // not a rule it can satisfy. Refusing is the only honest answer.
        let err = apply("abc", &one("", "x")).unwrap_err();
        assert_eq!(err, Conflict::EmptySearch { index: 0 });
    }

    #[test]
    fn a_missing_search_is_refused_and_names_the_block() {
        let blocks = vec![Block::new("a", "A"), Block::new("nope", "x")];
        let err = apply("abc", &blocks).unwrap_err();
        match err {
            Conflict::NotFound { index, old } => {
                assert_eq!(index, 1, "the second block is the one that failed");
                assert_eq!(old, "nope");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn an_ambiguous_search_is_refused_and_reports_the_lines() {
        let text = "one\ntwo\none\nthree\none\n";
        let err = apply(text, &one("one", "1")).unwrap_err();
        match err {
            Conflict::Ambiguous {
                index,
                matches,
                lines,
                ..
            } => {
                assert_eq!(index, 0);
                assert_eq!(matches, 3);
                assert_eq!(lines, vec![1, 3, 5]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn replace_all_changes_every_occurrence() {
        let text = "one\ntwo\none\n";
        let out = apply(text, &[Block::all("one", "1")]).unwrap();
        assert_eq!(out, "1\ntwo\n1\n");
    }

    #[test]
    fn replace_all_still_refuses_a_search_that_is_not_there() {
        let err = apply("abc", &[Block::all("zzz", "x")]).unwrap_err();
        assert!(matches!(err, Conflict::NotFound { .. }), "{err:?}");
    }

    #[test]
    fn blocks_are_applied_in_order_against_the_working_copy() {
        // The second block matches text the first one introduced. That is the
        // point of applying them in sequence rather than all against the
        // original: a patch can build a change up in steps.
        let blocks = vec![Block::new("a", "b"), Block::new("b", "c")];
        let out = apply("a\n", &blocks).unwrap();
        assert_eq!(out, "c\n");
    }

    #[test]
    fn a_later_block_that_no_longer_matches_fails_the_whole_patch() {
        // The first block rewrites the text the second was looking for. The
        // patch is refused, and the caller still holds the original.
        let blocks = vec![Block::new("a", "z"), Block::new("a", "y")];
        let err = apply("a\n", &blocks).unwrap_err();
        assert!(
            matches!(err, Conflict::NotFound { index: 1, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_patch_that_changes_nothing_is_refused() {
        let err = apply("abc", &one("abc", "abc")).unwrap_err();
        assert_eq!(err, Conflict::NoChange);
    }

    #[test]
    fn a_patch_that_changes_something_and_changes_it_back_is_refused() {
        // The net effect is nothing, so there is no diff to show and nothing
        // to approve.
        let blocks = vec![Block::new("a", "b"), Block::new("b", "a")];
        let err = apply("a\n", &blocks).unwrap_err();
        assert_eq!(err, Conflict::NoChange);
    }

    #[test]
    fn an_empty_replacement_deletes() {
        let out = apply("keep\ndrop\nkeep\n", &one("drop\n", "")).unwrap();
        assert_eq!(out, "keep\nkeep\n");
    }

    #[test]
    fn crlf_line_endings_survive_untouched() {
        let text = "one\r\ntwo\r\nthree\r\n";
        let out = apply(text, &one("two", "TWO")).unwrap();
        assert_eq!(out, "one\r\nTWO\r\nthree\r\n");
        assert!(!out.contains("\n\n"), "no line ending was rewritten");
    }

    #[test]
    fn a_missing_final_newline_survives() {
        let out = apply("a\nb", &one("b", "c")).unwrap();
        assert_eq!(out, "a\nc", "the file must not gain a trailing newline");
    }

    #[test]
    fn a_byte_order_mark_survives() {
        let text = "\u{feff}fn main() {}\n";
        let out = apply(text, &one("main", "start")).unwrap();
        assert!(out.starts_with('\u{feff}'), "the mark was dropped");
    }

    #[test]
    fn unicode_is_matched_and_replaced_whole() {
        let text = "let s = \"héllo wörld\";\n";
        let out = apply(text, &one("héllo wörld", "goodbye")).unwrap();
        assert_eq!(out, "let s = \"goodbye\";\n");
    }

    #[test]
    fn a_multibyte_search_that_is_absent_is_reported_without_panicking() {
        // The line count walks the text by byte offsets, so a multi-byte
        // needle is the case where an off-by-one would panic on a char
        // boundary.
        let err = apply("héllo\n", &one("wörld", "x")).unwrap_err();
        assert!(matches!(err, Conflict::NotFound { .. }), "{err:?}");
    }

    #[test]
    fn the_match_count_agrees_with_what_the_replacement_does() {
        // "aa" occurs at 0 and at 1 in "aaa" if occurrences may overlap, and
        // only at 0 if they may not. The count has to be the second kind,
        // because `str::replace` is the second kind: a count of 2 would make
        // the block look ambiguous and refuse an edit that has exactly one
        // unambiguous answer.
        let out = apply("aaa", &one("aa", "b")).unwrap();
        assert_eq!(out, "ba");
    }

    #[test]
    fn a_search_that_really_does_occur_twice_is_still_ambiguous() {
        // The other side of the rule above: two occurrences that do not
        // overlap are two, and the block is refused.
        let err = apply("aa aa", &one("aa", "b")).unwrap_err();
        match err {
            Conflict::Ambiguous { matches, .. } => assert_eq!(matches, 2),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn the_ambiguity_report_is_capped() {
        let text = "x\n".repeat(50);
        let err = apply(&text, &one("x", "y")).unwrap_err();
        match err {
            Conflict::Ambiguous { matches, lines, .. } => {
                assert_eq!(matches, 50);
                assert_eq!(lines.len(), REPORTED_MATCHES);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn a_long_search_string_is_quoted_only_in_part() {
        let long = "a".repeat(500);
        let err = apply("nothing here", &one(&long, "x")).unwrap_err();
        let message = err.to_string();
        assert!(
            message.len() < 300,
            "the message should not quote the whole search string: {message}"
        );
        assert!(message.contains('…'), "the quote should be marked as cut");
    }

    #[test]
    fn an_empty_block_list_is_a_no_change() {
        let err = apply("abc", &[]).unwrap_err();
        assert_eq!(err, Conflict::NoChange);
    }
}
