//! Bounded, ignore-aware search.
//!
//! Two tools walk the workspace: `grep` and `glob`. They used to walk it two
//! different ways, each with its own idea of what to skip, and neither with a
//! bound that held: `grep` read every file whole and collected every match
//! before truncating, and `glob` collected every hit. This module is the one
//! walker.
//!
//! Three properties matter more than speed.
//!
//! * **The walk is bounded.** Files visited, matches collected, bytes
//!   retained, file size and wall-clock time all have a ceiling, and reaching
//!   one stops the walk and says so. A search that quietly returns the first
//!   thousand matches of a million reads as "there are a thousand", which is a
//!   wrong answer rather than a partial one.
//! * **The walk is interruptible.** Traversal and matching are CPU-bound and
//!   have no await point to yield at, so they run on a blocking worker and the
//!   stop flag is checked per entry. A cancelled search ends within one file
//!   rather than at the end of the tree.
//! * **The walk is confined.** Links are never followed, so a link out of the
//!   workspace is offered as an entry and not read through. The caller has
//!   already resolved the root against the real filesystem; this module never
//!   widens that.
//!
//! Ignore rules come from the `ignore` crate, which is the walker ripgrep
//! uses. Reimplementing `.gitignore` semantics (negation, directory-only
//! rules, nested files, precedence) is a project in itself, and getting it
//! subtly wrong is worse than not having it.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};

/// Whether the caller has asked a search to stop.
///
/// A trait rather than the runtime's concrete token, so this module does not
/// depend on the runtime that owns the cancellation and a test can stop a walk
/// without building a session.
pub trait Stop: Send + Sync {
    /// Whether a stop has been asked for.
    fn stopped(&self) -> bool;
}

impl Stop for crate::runtime::CancelToken {
    fn stopped(&self) -> bool {
        self.is_stopped()
    }
}

/// A stop that is never set, for a caller with nothing to cancel.
#[derive(Debug, Default, Clone, Copy)]
pub struct NeverStop;

impl Stop for NeverStop {
    fn stopped(&self) -> bool {
        false
    }
}

/// What one search is allowed to do.
///
/// Every field is a ceiling that stops the walk rather than a target it aims
/// at. The defaults come from [`crate::config::Limits`], so a session's
/// configuration is the single place they are decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Budget {
    /// The most files the walk will open.
    pub max_files: usize,
    /// The most matches it will collect.
    pub max_matches: usize,
    /// The most bytes of result it will retain.
    pub max_bytes: usize,
    /// The largest file it will read.
    pub max_file_bytes: u64,
    /// How long it may run.
    pub timeout: Duration,
}

impl Budget {
    /// The budget a session's limits describe.
    pub fn from_limits(limits: &crate::config::Limits) -> Self {
        Self {
            max_files: limits.max_search_files,
            max_matches: limits.max_search_matches,
            max_bytes: limits.max_search_bytes,
            max_file_bytes: limits.max_search_file_bytes,
            timeout: limits.search_timeout,
        }
    }
}

/// What the walk is allowed to look at.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Options {
    /// Include dot-files and dot-directories.
    ///
    /// Off by default: a `.git` directory is not what "search the workspace"
    /// means, and a model that wants it can say so.
    pub hidden: bool,
    /// Include paths the ignore files exclude.
    ///
    /// Off by default, and the reason this module exists: a build directory
    /// with a hundred thousand generated files in it is the difference between
    /// a search that answers and one that times out.
    pub ignored: bool,
    /// How deep to descend. `None` is the whole tree.
    pub max_depth: Option<usize>,
}

/// Why a search stopped before it had finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Incomplete {
    /// The file limit was reached.
    Files,
    /// The match limit was reached.
    Matches,
    /// The output limit was reached.
    Bytes,
    /// The time limit was reached.
    Time,
    /// The run was cancelled.
    Cancelled,
}

impl Incomplete {
    /// The reason, as the model is told it.
    pub fn reason(self) -> &'static str {
        match self {
            Self::Files => "the file limit was reached",
            Self::Matches => "the match limit was reached",
            Self::Bytes => "the output limit was reached",
            Self::Time => "the time limit was reached",
            Self::Cancelled => "the search was cancelled",
        }
    }
}

/// What a search found.
#[derive(Debug, Clone, Default)]
pub struct Outcome {
    /// The result lines, already formatted.
    pub lines: Vec<String>,
    /// How many files the walk opened.
    pub files: usize,
    /// How many entries the walk saw, files and directories together.
    pub entries: usize,
    /// How many files were dropped because they are not text.
    pub binary: usize,
    /// Paths the walk could not use, with the reason, kept short.
    pub skipped: Vec<String>,
    /// Why the search stopped early, if it did.
    pub incomplete: Option<Incomplete>,
}

impl Outcome {
    /// The result as the model sees it.
    ///
    /// The incomplete marker is not decoration. A search that returns the
    /// first thousand matches of a million reads as "there are a thousand",
    /// which is a wrong answer rather than a partial one; saying so is what
    /// makes it a partial answer, and it is what tells the model to narrow the
    /// search instead of concluding the rest does not exist.
    pub fn render(&self, nothing: &str) -> String {
        let mut out = String::new();
        if self.lines.is_empty() {
            out.push_str(nothing);
            out.push('\n');
        } else {
            out.push_str(&self.lines.join("\n"));
            out.push('\n');
        }
        if let Some(incomplete) = self.incomplete {
            out.push_str(&format!(
                "\n[incomplete: {} after {} file(s) and {} entr{}]\n",
                incomplete.reason(),
                self.files,
                self.entries,
                if self.entries == 1 { "y" } else { "ies" },
            ));
        }
        if self.binary > 0 {
            out.push_str(&format!("\n[{} binary file(s) skipped]\n", self.binary));
        }
        if !self.skipped.is_empty() {
            out.push_str("\n[skipped: ");
            out.push_str(&self.skipped.join("; "));
            out.push_str("]\n");
        }
        out
    }
}

/// `path` as the model should see it: relative to `root`, with `/`
/// separators.
///
/// A result is read by a model that will put the path back into another tool
/// call, so it has to be a path that call will accept. An absolute path is not:
/// the tools resolve against the root, and an absolute path is refused as
/// outside it. Forward slashes are accepted on Windows and are what a pattern
/// is written with, so a result reads the same on every platform.
fn relative(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.to_string_lossy().replace('\\', "/")
}

/// The one walker.
///
/// `grep` and `glob` both go through this, so "what is in the workspace" has
/// one answer rather than two.
fn walker(root: &Path, opts: &Options) -> ignore::Walk {
    let mut builder = ignore::WalkBuilder::new(root);
    builder
        // `hidden(true)` means "skip hidden", so the flag is inverted here.
        .hidden(!opts.hidden)
        // `ignore(true)` means "respect the ignore files", likewise.
        .ignore(!opts.ignored)
        .git_ignore(!opts.ignored)
        .git_global(!opts.ignored)
        .git_exclude(!opts.ignored)
        .parents(!opts.ignored)
        // A `.gitignore` applies whether or not the directory is a checkout.
        // The workspace is often not one, and a rule the user wrote should not
        // stop applying because of that.
        .require_git(false)
        // Never follow a link. The walker cannot tell where one leads without
        // following it, and following it is the escape the workspace check
        // exists to prevent. A link is offered as an entry and not descended
        // into, so the caller can report it without reading through it.
        .follow_links(false);
    if let Some(depth) = opts.max_depth {
        builder.max_depth(Some(depth));
    }
    builder.build()
}

/// A walk with its bounds and its bookkeeping.
struct Walk<'a> {
    inner: ignore::Walk,
    root: &'a Path,
    budget: &'a Budget,
    stop: &'a dyn Stop,
    started: Instant,
    files: usize,
    entries: usize,
    skipped: Vec<String>,
    incomplete: Option<Incomplete>,
}

impl<'a> Walk<'a> {
    fn new(root: &'a Path, opts: &Options, budget: &'a Budget, stop: &'a dyn Stop) -> Self {
        Self {
            inner: walker(root, opts),
            root,
            budget,
            stop,
            started: Instant::now(),
            files: 0,
            entries: 0,
            skipped: Vec::new(),
            incomplete: None,
        }
    }

    /// The next entry, or `None` when the walk is over or a bound has been
    /// reached.
    ///
    /// The bounds are checked here rather than at the end so that reaching one
    /// *stops* the walk. A search that collected a million matches and then
    /// reported the first thousand would have paid for all million, which is
    /// the cost the bound exists to avoid.
    fn next(&mut self) -> Option<ignore::DirEntry> {
        if self.incomplete.is_some() {
            return None;
        }
        loop {
            if self.stop.stopped() {
                self.incomplete = Some(Incomplete::Cancelled);
                return None;
            }
            if self.started.elapsed() >= self.budget.timeout {
                self.incomplete = Some(Incomplete::Time);
                return None;
            }
            let entry = match self.inner.next()? {
                Ok(entry) => entry,
                Err(err) => {
                    self.note(&err.to_string());
                    continue;
                }
            };
            self.entries += 1;
            // A link is offered but never read through: the walker does not
            // follow it, and neither does the caller.
            if entry.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
                continue;
            }
            if entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                if self.files >= self.budget.max_files {
                    self.incomplete = Some(Incomplete::Files);
                    return None;
                }
                self.files += 1;
            }
            return Some(entry);
        }
    }

    /// Record a path the walk could not use, keeping the list short.
    ///
    /// A tree with a thousand unreadable files would otherwise produce a
    /// thousand lines of explanation, which is the same failure as returning
    /// too many matches.
    fn note(&mut self, message: &str) {
        const MOST: usize = 5;
        if self.skipped.len() < MOST {
            self.skipped.push(message.to_string());
        } else if self.skipped.len() == MOST {
            self.skipped.push("... and more".to_string());
        }
    }

    /// The path as the model should see it.
    fn relative(&self, path: &Path) -> String {
        relative(self.root, path)
    }

    /// The outcome, with the walk's bookkeeping folded in.
    fn finish(self, mut out: Outcome) -> Outcome {
        out.files = self.files;
        out.entries = self.entries;
        out.skipped = self.skipped;
        out.incomplete = self.incomplete;
        out
    }
}
/// Which bound stopped a collection, given what it holds.
fn which_bound(lines: usize, bytes: usize, budget: &Budget) -> Incomplete {
    if lines >= budget.max_matches {
        Incomplete::Matches
    } else if bytes >= budget.max_bytes {
        Incomplete::Bytes
    } else {
        Incomplete::Files
    }
}

/// Search file contents under `root` for `pattern`.
///
/// The pattern is a regular expression, matched line by line. Files are
/// streamed through the matcher rather than read whole, so a two-gigabyte log
/// costs a buffer rather than two gigabytes, and a file with a NUL byte in it
/// is dropped as binary rather than searched.
pub fn grep(
    root: &Path,
    pattern: &str,
    opts: &Options,
    budget: &Budget,
    stop: &dyn Stop,
) -> Result<Outcome> {
    let matcher = grep_regex::RegexMatcherBuilder::new()
        .build(pattern)
        .map_err(|e| Error::invalid(format!("invalid regex '{pattern}': {e}")))?;
    let mut searcher = grep_searcher::SearcherBuilder::new()
        .line_number(true)
        // The default is to search binary files as if they were text, which
        // for a `.png` or a `.rlib` means a page of mojibake in the result and
        // a match count that means nothing. Quitting at the first NUL byte
        // marks the file instead, and the caller drops it whole.
        .binary_detection(grep_searcher::BinaryDetection::quit(b'\x00'))
        .build();

    let mut walk = Walk::new(root, opts, budget, stop);
    let mut out = Outcome::default();
    let mut bytes = 0usize;

    while let Some(entry) = walk.next() {
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        // A file over the cap is skipped rather than read. The cap is on the
        // read, and a match in the first kilobyte of a two-gigabyte file is
        // not worth the two gigabytes.
        match std::fs::metadata(path) {
            Ok(meta) if meta.len() > budget.max_file_bytes => {
                let rel = walk.relative(path);
                walk.note(&format!(
                    "{rel}: {} bytes, over the search limit",
                    meta.len()
                ));
                continue;
            }
            Ok(_) => {}
            Err(err) => {
                let rel = walk.relative(path);
                walk.note(&format!("{rel}: {err}"));
                continue;
            }
        }

        let mut sink = Collector::new(walk.relative(path), budget, out.lines.len(), bytes);
        let searched = searcher.search_path(&matcher, path, &mut sink);
        if let Err(err) = searched {
            let rel = walk.relative(path);
            walk.note(&format!("{rel}: {err}"));
            continue;
        }
        // A file that turned out to be binary is dropped whole. The searcher
        // stops at the first NUL byte and reports what it matched before it,
        // so merging as it went would leave half a binary file in the result.
        if sink.binary {
            out.binary += 1;
            continue;
        }
        bytes += sink.bytes;
        out.lines.extend(sink.lines);
        if sink.full {
            walk.incomplete = Some(which_bound(out.lines.len(), bytes, budget));
            break;
        }
    }

    Ok(walk.finish(out))
}

/// Collects matches from one file, up to what is left of the budget.
///
/// The lines are held per file rather than pushed straight into the result,
/// because a file can turn out to be binary *after* it has matched.
struct Collector {
    /// The path as the model should see it, prefixed to every line.
    prefix: String,
    /// The lines found in this file.
    lines: Vec<String>,
    /// The bytes those lines occupy.
    bytes: usize,
    /// The whole search's ceilings, and what it has already spent.
    max_matches: usize,
    max_bytes: usize,
    spent_matches: usize,
    spent_bytes: usize,
    /// Set when a bound stopped the collection, so the caller can stop too.
    full: bool,
    /// Set when the searcher found a NUL byte and gave up on the file.
    binary: bool,
}

impl Collector {
    fn new(prefix: String, budget: &Budget, spent_matches: usize, spent_bytes: usize) -> Self {
        Self {
            prefix,
            lines: Vec::new(),
            bytes: 0,
            max_matches: budget.max_matches,
            max_bytes: budget.max_bytes,
            spent_matches,
            spent_bytes,
            full: false,
            binary: false,
        }
    }
}

impl grep_searcher::Sink for Collector {
    type Error = std::io::Error;

    fn matched(
        &mut self,
        _searcher: &grep_searcher::Searcher,
        mat: &grep_searcher::SinkMatch<'_>,
    ) -> std::result::Result<bool, std::io::Error> {
        let first = mat.line_number().unwrap_or(0);
        for (offset, line) in mat.lines().enumerate() {
            if self.spent_matches + self.lines.len() >= self.max_matches
                || self.spent_bytes + self.bytes >= self.max_bytes
            {
                self.full = true;
                // `false` stops the search of this file; the caller stops the
                // walk.
                return Ok(false);
            }
            let text = String::from_utf8_lossy(line);
            let text = text.trim_end_matches(['\n', '\r']);
            let rendered = format!("{}:{}: {}", self.prefix, first + offset as u64, text);
            self.bytes += rendered.len() + 1;
            self.lines.push(rendered);
        }
        Ok(true)
    }

    fn finish(
        &mut self,
        _searcher: &grep_searcher::Searcher,
        finish: &grep_searcher::SinkFinish,
    ) -> std::result::Result<(), std::io::Error> {
        self.binary = finish.binary_byte_offset().is_some();
        Ok(())
    }
}

/// Refuse a pattern that cannot be satisfied inside the workspace.
///
/// The pattern is matched against paths relative to the root, so a pattern
/// that climbs out or names an absolute location matches nothing at all. That
/// is safe but silent: the model gets "no files matched" for a pattern that
/// plainly means "look outside", and concludes the file does not exist. Saying
/// why is the difference between a wrong answer and a correction.
fn check_pattern(pattern: &str) -> Result<()> {
    let normalised = pattern.replace('\\', "/");
    let absolute = normalised.starts_with('/') || normalised.as_bytes().get(1) == Some(&b':');
    if absolute {
        return Err(Error::invalid(format!(
            "'{pattern}' is outside the workspace: patterns are matched against paths \
             relative to the workspace root, so a leading drive or slash cannot be used"
        )));
    }
    if normalised.split('/').any(|part| part == "..") {
        return Err(Error::invalid(format!(
            "'{pattern}' is outside the workspace: '..' climbs out of the root, and \
             patterns are matched against paths inside it"
        )));
    }
    Ok(())
}

/// Find files and directories under `root` whose path matches `pattern`.
///
/// The pattern is matched against the path relative to the root, which is what
/// the model wrote it against. `globset` normalises the separators, so
/// `src/*.rs` matches on Windows as well as on Unix.
pub fn glob(
    root: &Path,
    pattern: &str,
    opts: &Options,
    budget: &Budget,
    stop: &dyn Stop,
) -> Result<Outcome> {
    check_pattern(pattern)?;
    let compiled = globset::GlobBuilder::new(pattern)
        .build()
        .map_err(|e| Error::invalid(format!("invalid glob '{pattern}': {e}")))?;
    let set = globset::GlobSetBuilder::new()
        .add(compiled)
        .build()
        .map_err(|e| Error::invalid(format!("invalid glob '{pattern}': {e}")))?;

    let mut walk = Walk::new(root, opts, budget, stop);
    let mut out = Outcome::default();
    let mut bytes = 0usize;

    while let Some(entry) = walk.next() {
        let rel = walk.relative(entry.path());
        if !set.is_match(&rel) {
            continue;
        }
        if out.lines.len() >= budget.max_matches || bytes >= budget.max_bytes {
            walk.incomplete = Some(which_bound(out.lines.len(), bytes, budget));
            break;
        }
        bytes += rel.len() + 1;
        out.lines.push(rel);
    }

    out.lines.sort();
    Ok(walk.finish(out))
}

/// The immediate contents of one directory, bounded.
///
/// Deliberately *not* the shared walker. An explicit `list_dir` is a request
/// for what is in the directory the caller named, and hiding an ignored or
/// hidden entry from it would make the tool lie about that directory. The
/// ignore rules are what "search the workspace" means, not what "list this
/// directory" means. What it does share is the bound: a directory with a
/// hundred thousand entries stops at the limit and says so, rather than
/// building a hundred thousand strings to throw almost all of them away.
pub fn list(dir: &Path, budget: &Budget, stop: &dyn Stop) -> Result<Outcome> {
    let mut out = Outcome::default();
    let mut bytes = 0usize;
    let started = Instant::now();
    let rd = std::fs::read_dir(dir).map_err(|e| Error::io("reading", dir, e))?;

    for entry in rd {
        if stop.stopped() {
            out.incomplete = Some(Incomplete::Cancelled);
            break;
        }
        if started.elapsed() >= budget.timeout {
            out.incomplete = Some(Incomplete::Time);
            break;
        }
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                out.skipped.push(err.to_string());
                continue;
            }
        };
        if out.lines.len() >= budget.max_matches || bytes >= budget.max_bytes {
            out.incomplete = Some(which_bound(out.lines.len(), bytes, budget));
            break;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let ft = entry.file_type();
        // A link is marked as one rather than as the directory it points at:
        // the listing is of this directory, and a name that leads somewhere
        // else is not a name in it.
        let is_dir = ft.as_ref().map(|t| t.is_dir()).unwrap_or(false);
        let is_link = ft.as_ref().map(|t| t.is_symlink()).unwrap_or(false);
        let rendered = match (is_dir, is_link) {
            (_, true) => format!("{name}@"),
            (true, _) => format!("{name}/"),
            _ => name,
        };
        bytes += rendered.len() + 1;
        out.lines.push(rendered);
    }

    out.lines.sort();
    Ok(out)
}

/// Run a search on a blocking worker.
///
/// Traversal and matching are CPU-bound and have no await point to yield at,
/// so running them on the async runtime's worker would stall every other task
/// for the length of the walk. `spawn_blocking` puts them on a thread that is
/// allowed to be busy.
pub async fn on_worker<T, F>(work: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(work).await {
        Ok(result) => result,
        // The worker panicked or the runtime is shutting down. Either way the
        // search did not finish, and saying so is better than a panic
        // propagating into the tool result.
        Err(err) => Err(Error::invalid(format!("the search worker stopped: {err}"))),
    }
}

/// The path a search should start from, given the root and a relative path.
///
/// The caller has already resolved `rel` against the workspace; this only
/// turns the root into an owned value the blocking worker can carry.
pub fn root_of(root: &Path) -> PathBuf {
    root.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tree to search, with the shapes the ignore rules care about.
    fn tree() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::create_dir_all(p.join("src")).unwrap();
        std::fs::create_dir_all(p.join("target/debug")).unwrap();
        std::fs::create_dir_all(p.join(".hidden")).unwrap();
        std::fs::write(p.join("src/main.rs"), "fn main() {}\n// needle\n").unwrap();
        std::fs::write(p.join("src/lib.rs"), "pub fn needle() {}\n").unwrap();
        std::fs::write(p.join("README.md"), "needle in a readme\n").unwrap();
        std::fs::write(p.join("target/debug/out.rs"), "needle in a build\n").unwrap();
        std::fs::write(p.join(".hidden/secret.rs"), "needle in a dotfile\n").unwrap();
        std::fs::write(p.join(".gitignore"), "target/\n").unwrap();
        dir
    }

    fn budget() -> Budget {
        Budget {
            max_files: 1_000,
            max_matches: 1_000,
            max_bytes: 1_000_000,
            max_file_bytes: 1_000_000,
            timeout: Duration::from_secs(30),
        }
    }

    fn opts() -> Options {
        Options::default()
    }

    fn grep_here(root: &Path, pattern: &str) -> Outcome {
        grep(root, pattern, &opts(), &budget(), &NeverStop).unwrap()
    }

    // ------------------------------------------------------------ the walker

    #[test]
    fn an_ignored_directory_is_not_searched() {
        let dir = tree();
        let out = grep_here(dir.path(), "needle");
        let text = out.render("none");
        assert!(text.contains("src/main.rs"), "got: {text}");
        assert!(text.contains("README.md"), "got: {text}");
        assert!(
            !text.contains("target/debug/out.rs"),
            "a .gitignore'd directory must not be searched: {text}"
        );
    }

    #[test]
    fn a_hidden_directory_is_not_searched() {
        let dir = tree();
        let out = grep_here(dir.path(), "needle");
        assert!(
            !out.render("none").contains(".hidden/secret.rs"),
            "a dot-directory must not be searched by default"
        );
    }

    #[test]
    fn asking_for_ignored_files_includes_them() {
        let dir = tree();
        let opts = Options {
            ignored: true,
            ..Options::default()
        };
        let out = grep(dir.path(), "needle", &opts, &budget(), &NeverStop).unwrap();
        assert!(
            out.render("none").contains("target/debug/out.rs"),
            "the override must work"
        );
    }

    #[test]
    fn asking_for_hidden_files_includes_them() {
        let dir = tree();
        let opts = Options {
            hidden: true,
            ..Options::default()
        };
        let out = grep(dir.path(), "needle", &opts, &budget(), &NeverStop).unwrap();
        assert!(
            out.render("none").contains(".hidden/secret.rs"),
            "the override must work"
        );
    }

    #[test]
    fn a_nested_ignore_file_applies_to_its_own_directory() {
        let dir = tree();
        std::fs::create_dir_all(dir.path().join("src/gen")).unwrap();
        std::fs::write(dir.path().join("src/gen/out.rs"), "needle generated\n").unwrap();
        std::fs::write(dir.path().join("src/gen/.gitignore"), "out.rs\n").unwrap();

        let out = grep_here(dir.path(), "needle");
        let text = out.render("none");
        assert!(text.contains("src/lib.rs"), "got: {text}");
        assert!(
            !text.contains("src/gen/out.rs"),
            "a nested .gitignore must apply: {text}"
        );
    }

    #[test]
    fn a_negated_rule_brings_a_file_back() {
        // `*.md` excludes both, and `!README.md` re-includes one. Note that a
        // negation cannot re-include a file whose *parent directory* is
        // excluded: `target/` plus `!target/keep.rs` still excludes the file,
        // which is git's rule and therefore the walker's.
        let dir = tree();
        std::fs::write(dir.path().join("NOTES.md"), "needle in notes\n").unwrap();
        std::fs::write(dir.path().join(".gitignore"), "*.md\n!README.md\n").unwrap();

        let out = grep_here(dir.path(), "needle");
        let text = out.render("none");
        assert!(
            text.contains("README.md"),
            "a negation must re-include the file: {text}"
        );
        assert!(
            !text.contains("NOTES.md"),
            "the rule it negates must still exclude the rest: {text}"
        );
    }

    #[test]
    fn a_dot_ignore_file_is_honoured_too() {
        let dir = tree();
        std::fs::write(dir.path().join(".ignore"), "README.md\n").unwrap();
        let out = grep_here(dir.path(), "needle");
        assert!(
            !out.render("none").contains("README.md"),
            ".ignore must be honoured"
        );
    }

    // ------------------------------------------------------------- the results

    #[test]
    fn a_match_names_the_path_and_the_line() {
        let dir = tree();
        let out = grep_here(dir.path(), "needle");
        assert!(
            out.lines.iter().any(|l| l == "src/main.rs:2: // needle"),
            "got: {:?}",
            out.lines
        );
    }

    #[test]
    fn a_path_is_relative_and_uses_forward_slashes() {
        let dir = tree();
        let out = grep_here(dir.path(), "needle");
        for line in &out.lines {
            assert!(!line.contains('\\'), "a backslash leaked: {line}");
            assert!(
                !line.starts_with(&dir.path().display().to_string()),
                "an absolute path leaked: {line}"
            );
        }
    }

    #[test]
    fn a_binary_file_is_skipped_rather_than_searched() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.bin"), b"needle\x00\x01\x02needle\n").unwrap();
        std::fs::write(dir.path().join("a.txt"), "needle\n").unwrap();

        let out = grep_here(dir.path(), "needle");
        assert_eq!(out.lines, vec!["a.txt:1: needle".to_string()]);
    }

    #[test]
    fn a_file_over_the_size_cap_is_skipped_and_named() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("big.txt"), "needle\n".repeat(100)).unwrap();
        let budget = Budget {
            max_file_bytes: 16,
            ..budget()
        };

        let out = grep(dir.path(), "needle", &opts(), &budget, &NeverStop).unwrap();
        assert!(out.lines.is_empty(), "got: {:?}", out.lines);
        assert!(
            out.skipped
                .iter()
                .any(|s| s.contains("over the search limit")),
            "the skip must be reported: {:?}",
            out.skipped
        );
    }

    #[test]
    fn the_match_limit_stops_the_walk_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..50 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "needle\n").unwrap();
        }
        let budget = Budget {
            max_matches: 5,
            ..budget()
        };

        let out = grep(dir.path(), "needle", &opts(), &budget, &NeverStop).unwrap();
        assert_eq!(out.lines.len(), 5);
        assert_eq!(out.incomplete, Some(Incomplete::Matches));
        assert!(
            out.render("none").contains("[incomplete:"),
            "the model must be told the result is partial"
        );
    }

    #[test]
    fn the_file_limit_stops_the_walk_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..20 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "nothing here\n").unwrap();
        }
        let budget = Budget {
            max_files: 3,
            ..budget()
        };

        let out = grep(dir.path(), "needle", &opts(), &budget, &NeverStop).unwrap();
        assert_eq!(out.incomplete, Some(Incomplete::Files));
        assert_eq!(out.files, 3);
    }

    #[test]
    fn the_byte_limit_stops_the_walk_and_says_so() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..50 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "needle\n").unwrap();
        }
        let budget = Budget {
            max_bytes: 40,
            ..budget()
        };

        let out = grep(dir.path(), "needle", &opts(), &budget, &NeverStop).unwrap();
        assert_eq!(out.incomplete, Some(Incomplete::Bytes));
        assert!(out.lines.len() < 50, "got: {}", out.lines.len());
    }

    #[test]
    fn a_stopped_search_ends_early_and_says_so() {
        struct StopAfter(std::sync::atomic::AtomicUsize);
        impl Stop for StopAfter {
            fn stopped(&self) -> bool {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 2
            }
        }

        let dir = tempfile::tempdir().unwrap();
        for i in 0..50 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "needle\n").unwrap();
        }

        let out = grep(
            dir.path(),
            "needle",
            &opts(),
            &budget(),
            &StopAfter(0.into()),
        )
        .unwrap();
        assert_eq!(out.incomplete, Some(Incomplete::Cancelled));
        assert!(out.lines.len() < 50, "got: {}", out.lines.len());
    }

    #[test]
    fn a_bad_pattern_is_an_error_not_a_panic() {
        let dir = tree();
        let err = grep(dir.path(), "([", &opts(), &budget(), &NeverStop).unwrap_err();
        assert!(err.to_string().contains("invalid regex"), "got: {err}");
    }

    #[test]
    fn a_search_that_finds_nothing_says_so() {
        let dir = tree();
        let out = grep_here(dir.path(), "no-such-token-anywhere");
        assert!(out.lines.is_empty());
        assert!(out.incomplete.is_none());
        assert!(out
            .render("No matches found.")
            .contains("No matches found."));
    }

    // ------------------------------------------------------------------ glob

    #[test]
    fn glob_finds_files_by_pattern() {
        let dir = tree();
        let out = glob(dir.path(), "**/*.rs", &opts(), &budget(), &NeverStop).unwrap();
        assert!(
            out.lines.contains(&"src/main.rs".to_string()),
            "{:?}",
            out.lines
        );
        assert!(
            out.lines.contains(&"src/lib.rs".to_string()),
            "{:?}",
            out.lines
        );
        assert!(
            !out.lines.contains(&"target/debug/out.rs".to_string()),
            "an ignored path must not match: {:?}",
            out.lines
        );
    }

    #[test]
    fn glob_matches_a_top_level_file_with_a_double_star() {
        // `**/` has to match zero directories as well as one, or `**/*.rs`
        // would miss a file at the root and the model would conclude it is not
        // there.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main() {}\n").unwrap();
        let out = glob(dir.path(), "**/*.rs", &opts(), &budget(), &NeverStop).unwrap();
        assert_eq!(out.lines, vec!["main.rs".to_string()]);
    }

    #[test]
    fn glob_reports_directories_too() {
        let dir = tree();
        let out = glob(dir.path(), "src/*", &opts(), &budget(), &NeverStop).unwrap();
        assert!(
            out.lines.contains(&"src/main.rs".to_string()),
            "{:?}",
            out.lines
        );
    }

    #[test]
    fn glob_is_bounded_too() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..50 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "x\n").unwrap();
        }
        let budget = Budget {
            max_matches: 4,
            ..budget()
        };
        let out = glob(dir.path(), "*.txt", &opts(), &budget, &NeverStop).unwrap();
        assert_eq!(out.lines.len(), 4);
        assert_eq!(out.incomplete, Some(Incomplete::Matches));
    }

    #[test]
    fn a_bad_glob_is_an_error_not_a_panic() {
        let dir = tree();
        let err = glob(dir.path(), "[", &opts(), &budget(), &NeverStop).unwrap_err();
        assert!(err.to_string().contains("invalid glob"), "got: {err}");
    }

    // ------------------------------------------------------------------ list

    #[test]
    fn a_listing_marks_directories_and_links() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("a.txt"), "x\n").unwrap();

        let out = list(dir.path(), &budget(), &NeverStop).unwrap();
        assert_eq!(out.lines, vec!["a.txt".to_string(), "sub/".to_string()]);
    }

    #[test]
    fn a_listing_shows_what_the_walker_would_hide() {
        // `list_dir` is a request for what is in the directory the caller
        // named. Hiding an ignored entry from it would make the tool lie about
        // that directory.
        let dir = tree();
        let out = list(&dir.path().join("target"), &budget(), &NeverStop).unwrap();
        assert_eq!(out.lines, vec!["debug/".to_string()]);
    }

    #[test]
    fn a_listing_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..50 {
            std::fs::write(dir.path().join(format!("f{i}.txt")), "x\n").unwrap();
        }
        let budget = Budget {
            max_matches: 3,
            ..budget()
        };
        let out = list(dir.path(), &budget, &NeverStop).unwrap();
        assert_eq!(out.lines.len(), 3);
        assert_eq!(out.incomplete, Some(Incomplete::Matches));
    }

    #[test]
    fn a_missing_directory_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = list(&dir.path().join("nope"), &budget(), &NeverStop).unwrap_err();
        assert!(err.to_string().contains("reading"), "got: {err}");
    }

    // ------------------------------------------------------------- the budget

    #[test]
    fn the_budget_comes_from_the_session_limits() {
        let limits = crate::config::Limits {
            max_turns: 1,
            max_read_bytes: 1,
            max_shell_bytes: 2,
            max_search_files: 3,
            max_search_matches: 4,
            max_search_bytes: 5,
            max_search_file_bytes: 6,
            search_timeout: Duration::from_secs(7),
        };
        let budget = Budget::from_limits(&limits);
        assert_eq!(budget.max_files, 3);
        assert_eq!(budget.max_matches, 4);
        assert_eq!(budget.max_bytes, 5);
        assert_eq!(budget.max_file_bytes, 6);
        assert_eq!(budget.timeout, Duration::from_secs(7));
    }

    #[test]
    fn a_relative_path_is_what_the_model_gets() {
        let root = Path::new("C:/work/proj");
        assert_eq!(relative(root, &root.join("src/main.rs")), "src/main.rs");
        // A path outside the root is not stripped, and is not silently
        // rewritten into something that looks inside it.
        assert_eq!(relative(root, Path::new("C:/other/a.rs")), "C:/other/a.rs");
    }
}
