//! The one directory a session is allowed to touch.
//!
//! The root is fixed once, when the session starts, from the directory the
//! program was launched in (or the one named on the command line). Every path
//! a tool is given is resolved against it, and a path that lands outside is
//! refused before the filesystem is touched.
//!
//! The root is a **value** ([`Workspace`]), not a process global. It used to
//! be a `OnceLock`, which made two sessions in one process impossible and
//! made unit tests order-dependent: a test that fixed the root changed what
//! every later test in the same binary resolved against. The binary builds
//! one and hands it to the runtime; a test builds one per case.
//!
//! # Two checks, not one
//!
//! [`Workspace::resolve`] is **lexical**: it resolves `.` and `..` and
//! compares the result against the root, so `../secrets` and `C:\Windows` are
//! both refused. That is not enough on its own, because a path can be
//! lexically inside the root and still *reach* outside it: `link/secret.txt`
//! where `link` is a symlink, or a Windows junction, or a directory reparse
//! point, resolves to whatever the link points at.
//!
//! [`Workspace::resolve_existing`] is the second check. It canonicalises the
//! deepest ancestor of the path that actually exists, which follows every
//! link in the chain, and re-tests containment against the real target. A
//! path that arrives through a link out of the root is refused there.
//!
//! # What is still not closed
//!
//! The two checks are separated by a window: between the canonicalisation and
//! the access, the link could be swapped. For a **write** the window is
//! harmless, and not by luck: the temp file is a sibling of the target, so if
//! the parent directory is swapped for a link the rename looks for the temp
//! file in the new directory, does not find it, and fails. The write cannot
//! land outside the root. For a **read** the window is real but narrow, and
//! what it exposes is the contents of a file the model is shown, not a
//! mutation. Closing it properly needs handle-relative opens (`openat2` with
//! `RESOLVE_BENEATH`, or `NtCreateFile` against a directory handle), which is
//! platform-specific unsafe code; it is not attempted here, and this comment
//! is the honest statement of that.
//!
//! # Reaching outside, on purpose
//!
//! A path outside the root is not simply refused: it is *asked about*. A
//! reference crate beside the project, or a config in the user's home, is a
//! normal thing to need, and a session that can only ever refuse leaves the
//! user editing the path by hand. [`Workspace::grant`] records a directory
//! the user has approved, and every check here treats a granted directory
//! exactly as it treats the root. The asking is
//! [`crate::runtime::Runtime`]'s, because that is where the approval gate
//! lives; the remembering is here, because this is what the answer changes.
//!
//! A grant lives as long as the session and is never written to disk, so the
//! next session starts with the root alone. It widens *where* a tool may
//! look; it does not widen *what* a tool may do, because a write outside the
//! root still goes through the same per-write approval as a write inside it.
//!
//! # `shell`, and what a string can tell you
//!
//! An approved command runs with the root as its working directory and is
//! otherwise free to name any path it likes. A working directory is not a
//! sandbox, and neither is a Windows Job Object. Confining it properly needs
//! an OS facility — Landlock, a seccomp filter, an AppContainer — each of
//! which exists on one platform and not the others.
//!
//! What is possible without one is to stop the gate running *blind*.
//! [`Workspace::reach_of`] reads the command as a string, resolves the words
//! in it that are shaped like paths, and reports the ones that land outside,
//! along with the words whose target cannot be known from a string at all.
//! The caller then gates the first ([`crate::runtime::Runtime`] asks for the
//! directory, exactly as a `read_file` would have to) and shows the second to
//! whoever is answering.
//!
//! This is a heuristic, and the honest statement of it is this: `eval`, a
//! variable, a command substitution, a path read out of a file, or a script
//! the command invokes all defeat it. It closes the accidental case, makes
//! the deliberate one visible, and is not a sandbox. Nothing here should be
//! read as one.

use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};

/// Why a path was refused.
///
/// The two cases need different answers, so they are not collapsed into one
/// message. "Outside the workspace" tells the model to stay inside; "a
/// verbatim path is not normalised" tells it to write the path plainly.
/// Collapsing them would leave the model retrying a path that can never work.
///
/// The pure functions here return this rather than an [`Error`] because they
/// have no root to name beyond the one they were given; the [`Workspace`]
/// methods that own a root turn it into the typed error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The path's shape is not one this session can check.
    Form(String),
    /// The path resolves outside the root.
    Outside,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Form(reason) => write!(f, "{reason}"),
            Self::Outside => write!(f, "the path is outside the workspace"),
        }
    }
}

impl Refusal {
    /// The typed error for this refusal, naming `path` and `root`.
    fn into_error(self, path: &str, root: &Path) -> Error {
        match self {
            Self::Form(reason) => Error::path_form(path, reason),
            Self::Outside => Error::outside(path, root),
        }
    }
}

/// The one directory a session may touch.
///
/// Cheap to clone (one `PathBuf` and one `Arc`), so it is passed by value
/// into the tools rather than shared behind a lock.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
    /// The directories outside the root the user has granted this session.
    ///
    /// Shared between clones, and shared on purpose: every tool holds its own
    /// `Workspace`, and a grant the user has just given has to be visible to
    /// all of them. A `Vec` rather than a set because it holds the handful of
    /// directories one session reaches into, and the check walks it on every
    /// resolution; a set would be the same walk with a hash in front of it.
    granted: Arc<RwLock<Vec<PathBuf>>>,
}

impl Workspace {
    /// Fix `path` as the root, resolving it to its real filesystem target.
    ///
    /// Canonicalising here is what makes the containment check meaningful: a
    /// root reached through a symlink is stored as the directory it really
    /// is, so a path that arrives by either name compares equal.
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let canonical = path.canonicalize().map_err(|source| Error::WorkspaceRoot {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(Self {
            root: strip_verbatim(canonical),
            granted: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// The root, as fixed at construction.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The comparison form of the root: see [`key`].
    pub fn key(&self) -> String {
        key(&self.root)
    }

    /// Whether `candidate` is the root itself or sits under it.
    ///
    /// The root only: a granted directory is somewhere this session may
    /// *reach*, not somewhere that belongs to the project, and the callers of
    /// this (the banner, the trust record) are asking about the project.
    pub fn contains(&self, candidate: &Path) -> bool {
        contains(&self.root, candidate)
    }

    /// Allow this session to reach `dir` and everything under it.
    ///
    /// Called by [`crate::runtime::Runtime`] once the user has approved the
    /// directory, and never from a tool: a tool that could grant its own
    /// access would make the question it was asked pointless.
    ///
    /// A grant that is already covered by one the user gave earlier is
    /// dropped, and a grant that covers earlier ones replaces them, so the
    /// list stays the set of directories the user actually answered for
    /// rather than one entry per path that was asked about.
    pub fn grant(&self, dir: PathBuf) {
        let mut granted = self
            .granted
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if granted.iter().any(|g| contains(g, &dir)) {
            return;
        }
        granted.retain(|g| !contains(&dir, g));
        granted.push(dir);
    }

    /// Whether this session may touch `candidate` at all: it is inside the
    /// root, or inside a directory the user has granted.
    ///
    /// The question a caller asks when it holds a resolved path rather than a
    /// tool argument — the shell scan, which resolves words out of a command
    /// line.
    pub fn may_touch(&self, candidate: &Path) -> bool {
        self.allows(candidate)
    }

    /// What `command` appears to reach, as far as its text can tell.
    ///
    /// The words that are shaped like paths are resolved against the root and
    /// tested like any other path, so a granted directory is not reported and
    /// an ordinary `cargo test` reports nothing. See [`Reach`] for what this
    /// cannot see; the module docs say why it is still worth having.
    pub fn reach_of(&self, command: &str) -> Reach {
        let mut outside: Vec<PathBuf> = Vec::new();
        for word in words(command) {
            let Some(target) = self.target_of_word(word) else {
                continue;
            };
            if self.allows(&target) || outside.contains(&target) {
                continue;
            }
            outside.push(target);
        }
        Reach {
            outside,
            unresolvable: unresolvable(command),
        }
    }

    /// The path a command-line word names, if it names one.
    ///
    /// `None` for a word that is not shaped like a path, which is most of
    /// them: a subcommand, a flag, a number, a regex with no slash in it.
    fn target_of_word(&self, word: &str) -> Option<PathBuf> {
        let value = argument_value(word);
        if !looks_like_path(value) {
            return None;
        }
        let raw = if value.starts_with('~') {
            // No home directory to resolve against: the word cannot be
            // placed, so it is left to `unresolvable` rather than guessed at.
            expand_home(value)?
        } else {
            PathBuf::from(value)
        };
        let joined = if raw.is_absolute() {
            raw
        } else {
            self.root.join(raw)
        };
        let normalized = normalize(&joined);
        // A path that exists is canonicalised so the scan sees where a link
        // really leads. A path that does not exist keeps its lexical form:
        // there is nothing to follow, and the lexical form is the clearer
        // thing to show.
        Some(match normalized.canonicalize() {
            Ok(real) => strip_verbatim(real),
            Err(_) => normalized,
        })
    }

    /// The directories granted for this session, for the banner and the tests.
    pub fn granted(&self) -> Vec<PathBuf> {
        self.granted
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Whether `candidate` is somewhere this session may touch: inside the
    /// root, or inside a directory the user has granted.
    fn allows(&self, candidate: &Path) -> bool {
        contains(&self.root, candidate)
            || self
                .granted
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .iter()
                .any(|dir| contains(dir, candidate))
    }

    /// The path `path` names, resolved lexically but **not** checked for
    /// containment.
    ///
    /// The lexical form: `..` is folded, but a link is left as the name the
    /// path uses. A path whose *shape* cannot be checked is still refused,
    /// because such a path cannot be judged either.
    pub fn target_of(&self, path: &str) -> Result<PathBuf> {
        check_form(path).map_err(|r| r.into_error(path, &self.root))?;
        let raw = Path::new(path);
        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            self.root.join(raw)
        };
        Ok(normalize(&joined))
    }

    /// The path `path` really names, with links followed, but **not** checked
    /// for containment.
    ///
    /// This is the path the approval prompt shows, because the question the
    /// user is being asked is which directory is being reached into, and only
    /// the real form answers it: `../../etc/passwd`, `/etc/passwd` and
    /// `link/secret.txt` where `link` points at `/etc` are the same question.
    /// It is also what a grant must record: granting the lexical name of a
    /// link would grant a directory the access never uses.
    pub fn real_target_of(&self, path: &str) -> Result<PathBuf> {
        let lexical = self.target_of(path)?;
        canonical_ancestor(&lexical).map_err(|_| Error::outside(path, &self.root))
    }

    /// Resolve a tool's `path` argument against the root, refusing anything
    /// outside.
    ///
    /// Lexical only: see the module docs for why that is not the whole check.
    ///
    /// A granted directory passes here exactly as the root does. The grant is
    /// the user's answer to this same question asked once already, and
    /// refusing it a second time would make the answer worthless.
    pub fn resolve(&self, path: &str) -> Result<PathBuf> {
        let target = self.target_of(path)?;
        if !self.allows(&target) {
            return Err(Error::outside(path, &self.root));
        }
        Ok(target)
    }

    /// Resolve a path that is expected to exist, following links to their
    /// real targets.
    ///
    /// This is the check that closes the symlink and junction escape: the
    /// deepest ancestor of the path that exists is canonicalised, which
    /// follows every link in the chain, and the result is tested against the
    /// root. `link/secret.txt`, where `link` points outside, is refused here
    /// even though it is lexically inside.
    ///
    /// The returned path is the real one, with any not-yet-existing tail
    /// re-appended, so a caller that goes on to open it opens the file that
    /// was checked rather than the name that was checked.
    pub fn resolve_existing(&self, path: &str) -> Result<PathBuf> {
        let lexical = self.resolve(path)?;
        self.real_target(&lexical)
            .map_err(|_| Error::outside(path, &self.root))
    }
    /// Resolve a path that is about to be created or replaced.
    ///
    /// The target itself may not exist, so the check is on its **existing
    /// ancestors**: every directory between the root and the target is
    /// canonicalised and tested, which is what stops `link/new.txt` from
    /// creating a file outside the root. The target's own name is not
    /// followed, because a name that does not exist has nothing to follow.
    pub fn resolve_for_write(&self, path: &str) -> Result<PathBuf> {
        let lexical = self.resolve(path)?;
        let parent = match lexical.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => return Err(Error::outside(path, &self.root)),
        };
        let real_parent = self
            .real_target(parent)
            .map_err(|_| Error::outside(path, &self.root))?;
        let name = lexical
            .file_name()
            .ok_or_else(|| Error::outside(path, &self.root))?;
        Ok(real_parent.join(name))
    }

    /// Anchor a glob pattern to the root.
    pub fn anchor_pattern(&self, pattern: &str) -> Result<String> {
        anchor_pattern_in(&self.root, pattern).map_err(|r| r.into_error(pattern, &self.root))
    }

    /// The root's directory name, which is what the user calls this project.
    pub fn name(&self) -> String {
        self.root
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    }
}

/// What a command appears to reach, as far as its text can tell.
///
/// Two lists rather than a verdict, because the two need different answers. A
/// resolved path that lands outside is a question with a directory in it, and
/// the user can answer it. A command substitution is not a question at all:
/// nothing can say where it will point until it has run, so it is reported as
/// what it is and shown to whoever decides.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reach {
    /// Resolved paths outside everything this session may touch, in the order
    /// they appear in the command, without duplicates.
    pub outside: Vec<PathBuf>,
    /// The kinds of indirection the command uses, named for the prompt. A
    /// non-empty list means the paths above are what *could* be read, not
    /// everything the command will touch.
    pub unresolvable: Vec<String>,
}

impl Reach {
    /// Whether the command reads as staying inside the workspace.
    pub fn is_contained(&self) -> bool {
        self.outside.is_empty() && self.unresolvable.is_empty()
    }
}

/// The words of a command, for the path scan.
///
/// Split on whitespace, on quotes and on the shell's own operators, with no
/// attempt to interpret any of them. A quoted path with a space in it arrives
/// as two words, and `sh -c "cat ../x"` arrives with `../x` as a word of its
/// own — which is the point: treating the quoted string as one token would
/// hide exactly the path worth seeing.
fn words(command: &str) -> impl Iterator<Item = &str> {
    command
        .split(|c: char| c.is_whitespace() || "\"'|&;<>()".contains(c))
        .filter(|word| !word.is_empty())
}

/// The part of a word that names a path: what follows the first `=`, so
/// `--manifest-path=../other/Cargo.toml` is scanned as the path it carries.
fn argument_value(word: &str) -> &str {
    match word.split_once('=') {
        Some((_, value)) => value,
        None => word,
    }
}

/// Whether a word is shaped like a path rather than a flag or a bare word.
///
/// A separator or a leading `~`. Deliberately generous: `s/a/b/` is scanned
/// as a path, resolves inside the root and is reported as nothing, which
/// costs a comparison. The other direction costs a missed path.
///
/// The one exception is a Windows flag. `cmd /c`, `for /L` and `robocopy /E`
/// are flags written with the separator a Unix path starts with, and a path
/// on Windows needs a drive or a UNC prefix rather than a leading slash. A
/// second separator is what tells the two apart, so `/c/Users/me/x` — the
/// shape a path has under Git Bash — is still seen.
fn looks_like_path(word: &str) -> bool {
    if cfg!(windows) && word.starts_with('/') && !word[1..].contains('/') {
        return false;
    }
    // A bare `..` is the one path with no separator in it, and `cd ..` is how
    // a command leaves the workspace without naming anywhere.
    word == ".." || word.starts_with('~') || word.contains('/') || word.contains('\\')
}

/// `~`, `~/x` and `~\x` as the home directory and what follows it.
///
/// `None` when there is no home directory to resolve against. `~user/x` is
/// resolved as if it were `~/user/x`, which names the wrong directory but the
/// right answer: it is outside the workspace either way.
fn expand_home(word: &str) -> Option<PathBuf> {
    let rest = word.strip_prefix('~')?.trim_start_matches(['/', '\\']);
    let home = std::env::home_dir()?;
    Some(if rest.is_empty() {
        home
    } else {
        home.join(rest)
    })
}

/// The kinds of indirection in `command` whose target no scan can resolve.
///
/// Each condition can fire only once, so the list needs no de-duplication.
fn unresolvable(command: &str) -> Vec<String> {
    let mut found = Vec::new();
    if command.contains("$(") {
        found.push("$(...) command substitution".to_string());
    }
    if command.contains('`') {
        found.push("`...` command substitution".to_string());
    }
    if command.contains("${") {
        found.push("${...} variable".to_string());
    }
    if has_dollar_name(command) {
        found.push("$NAME variable".to_string());
    }
    if cfg!(windows) && has_percent_name(command) {
        found.push("%NAME% variable".to_string());
    }
    if words(command).any(|word| word == "eval") {
        found.push("eval".to_string());
    }
    found
}

/// Whether `command` contains a `$NAME` variable reference.
///
/// A `$` followed by a letter or an underscore. A `$` before anything else is
/// an ordinary character — the end-of-line anchor in a regex, most often,
/// which is not a variable and must not read as one.
fn has_dollar_name(command: &str) -> bool {
    let bytes = command.as_bytes();
    bytes.iter().enumerate().any(|(at, byte)| {
        *byte == b'$'
            && bytes
                .get(at + 1)
                .is_some_and(|next| next.is_ascii_alphabetic() || *next == b'_')
    })
}

/// Whether `command` contains a `%NAME%` variable reference, the Windows
/// form.
fn has_percent_name(command: &str) -> bool {
    let mut rest = command;
    while let Some(open) = rest.find('%') {
        rest = &rest[open + 1..];
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() && rest[name.len()..].starts_with('%') {
            return true;
        }
    }
    false
}

/// Resolve `lexical` to the path it really names, following links, and refuse
/// it when that path is outside everything this session may touch.
impl Workspace {
    fn real_target(&self, lexical: &Path) -> std::result::Result<PathBuf, Refusal> {
        let real = canonical_ancestor(lexical)?;
        if !self.allows(&real) {
            return Err(Refusal::Outside);
        }
        Ok(real)
    }
}

/// Canonicalise the deepest ancestor of `lexical` that exists, following
/// links, and re-append the tail that does not exist yet.
///
/// This is the filesystem half of the confinement check: it answers where a
/// path really leads without judging whether that is allowed. The callers
/// decide: [`Workspace::real_target`] tests containment, and
/// [`Workspace::real_target_of`] hands the same answer to the approval
/// prompt. The verbatim prefix Windows adds is dropped so the result compares
/// and displays like every other path here.
///
/// Walking up rather than canonicalising the whole path is what makes this
/// work for a file that is about to be created: `canonicalize` fails on a
/// missing path, and a `write_file` to a new file is a normal thing to do.
/// The components below the deepest existing ancestor are re-appended
/// unchanged: they do not exist, so there is nothing there to follow, and
/// they were already checked lexically by the caller.
fn canonical_ancestor(lexical: &Path) -> std::result::Result<PathBuf, Refusal> {
    let mut existing = lexical.to_path_buf();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match existing.canonicalize() {
            Ok(real) => {
                let mut out = strip_verbatim(real);
                for part in tail.iter().rev() {
                    out.push(part);
                }
                return Ok(out);
            }
            Err(_) => {
                // Nothing here. Step up and try the parent, remembering
                // the name so the path can be rebuilt.
                let name = existing.file_name().ok_or(Refusal::Outside)?.to_os_string();
                tail.push(name);
                if !existing.pop() {
                    return Err(Refusal::Outside);
                }
            }
        }
    }
}

/// The forms of path a tool argument may not use, whatever they resolve to.
///
/// These are refused on their shape rather than on where they land, because
/// each of them defeats the resolution that follows:
///
/// * A **verbatim** path (`\\?\C:\work`) is not normalised by the OS, so `.`
///   and `..` inside it stay literal and the lexical check sees a path the
///   filesystem will not.
/// * A **device** path (`\\.\C:`) names a device, not a file.
/// * A **UNC** path (`\\server\share`) is never inside a local root.
/// * A **drive-relative** path (`C:foo`) means "foo under the current
///   directory on drive C", which is not the workspace and is not knowable
///   from the string.
/// * An **alternate data stream** (`a.txt:stream`) is a second file attached
///   to the first, which the containment check does not see.
///
/// The Windows-only forms are gated on `cfg!(windows)`: a colon is an
/// ordinary character in a Unix filename, and refusing it there would refuse
/// real files.
pub fn check_form(path: &str) -> std::result::Result<(), Refusal> {
    if path.is_empty() {
        return Err(Refusal::Form("the path is empty".to_string()));
    }
    if path.contains('\0') {
        return Err(Refusal::Form(format!("'{path}' contains a NUL byte")));
    }

    let raw = path.replace('\\', "/");

    if cfg!(windows) {
        if raw.starts_with("//?/") || raw.starts_with("//./") {
            return Err(Refusal::Form(format!(
                "'{path}' is a verbatim or device path, which is not normalised and cannot be \
                 checked against the workspace"
            )));
        }
        if raw.starts_with("//") {
            return Err(Refusal::Form(format!(
                "'{path}' is a UNC path, which is never inside the workspace"
            )));
        }
        let bytes = raw.as_bytes();
        let drive_prefixed = bytes.len() >= 2 && bytes[1] == b':' && bytes[0].is_ascii_alphabetic();
        if drive_prefixed && bytes.get(2).is_some_and(|c| *c != b'/') {
            return Err(Refusal::Form(format!(
                "'{path}' is relative to a drive's current directory, not to the workspace"
            )));
        }
        // A colon after the drive prefix is a stream separator. A colon at
        // index 1 is the drive prefix itself.
        let rest = if drive_prefixed { &raw[2..] } else { &raw[..] };
        if rest.contains(':') {
            return Err(Refusal::Form(format!(
                "'{path}' names an alternate data stream, which is a second file the workspace \
                 check does not see"
            )));
        }
    }

    Ok(())
}

/// The comparison form of a path: forward slashes, no trailing separator, and
/// lower case on Windows, where the filesystem is case-insensitive.
///
/// Two paths are the same directory exactly when their keys are equal, so
/// every containment check goes through this rather than comparing `Path`s
/// directly.
pub fn key(path: &Path) -> String {
    // The verbatim prefix is dropped first: `canonicalize` adds it on
    // Windows, and a path that arrives through it (`\\?\C:\work\src`) must
    // compare equal to the same path written plainly (`C:\work\src`).
    let text = path.to_string_lossy();
    let text = match text.strip_prefix(r"\\?\") {
        Some(rest) => match rest.strip_prefix("UNC\\") {
            Some(unc) => format!(r"\\{unc}"),
            None => rest.to_string(),
        },
        None => text.into_owned(),
    };
    let slashed = text.replace('\\', "/");
    let trimmed = slashed.trim_end_matches('/');
    let trimmed = if trimmed.is_empty() { "/" } else { trimmed };
    if cfg!(windows) {
        trimmed.to_lowercase()
    } else {
        trimmed.to_string()
    }
}

/// Whether `candidate` is `root` itself or sits under it.
pub fn contains(root: &Path, candidate: &Path) -> bool {
    is_under(&key(root), &key(candidate))
}

/// Whether `candidate_key` is `root_key` or a descendant of it.
///
/// Compared component-wise, not as a string prefix: `C:/work/proj-other` must
/// not count as being inside `C:/work/proj`.
fn is_under(root_key: &str, candidate_key: &str) -> bool {
    if candidate_key == root_key {
        return true;
    }
    let prefix = if root_key.ends_with('/') {
        root_key.to_string()
    } else {
        format!("{root_key}/")
    };
    candidate_key.starts_with(&prefix)
}

/// Resolve a tool's `path` argument against `root`, refusing anything outside.
///
/// A relative path is joined to the root, so the tools are independent of the
/// process's working directory; an absolute path is accepted only if it
/// already points inside.
///
/// The error is a [`Refusal`] rather than an [`Error`] because this is the
/// pure lexical core: it has no root to name beyond the one it was given, and
/// the caller that owns a [`Workspace`] rebuilds the typed error from it.
pub fn resolve_in(root: &Path, path: &str) -> std::result::Result<PathBuf, Refusal> {
    check_form(path)?;
    let raw = Path::new(path);
    let joined = if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        root.join(raw)
    };
    let normalized = normalize(&joined);
    if !contains(root, &normalized) {
        return Err(Refusal::Outside);
    }
    Ok(normalized)
}

/// Anchor a glob pattern to the workspace root.
///
/// A relative pattern is joined to the root, so `**/*.rs` searches the
/// workspace rather than whatever directory the process happens to be in. An
/// absolute pattern is accepted only if it already points inside the root.
///
/// The result is a pattern string, not a path: the glob crate reads `/` as the
/// separator on every platform, so the root is rendered with forward slashes
/// rather than round-tripped through `PathBuf`.
pub fn anchor_pattern_in(root: &Path, pattern: &str) -> std::result::Result<String, Refusal> {
    check_form(pattern)?;
    let root_key = key(root);
    // The pattern is built from the real path, not the comparison key: the
    // key is lower-cased on Windows, and a pattern is handed to the filesystem
    // rather than compared.
    let root_base = root.to_string_lossy().replace('\\', "/");
    let root_base = root_base.trim_end_matches('/').to_string();
    let raw = pattern.replace('\\', "/");

    // A `..` in a pattern cannot be resolved lexically, because the segments
    // around it may be wildcards. Refusing it outright is the honest rule: a
    // pattern that needs to climb out is not a pattern this session can run.
    if raw.split('/').any(|segment| segment == "..") {
        return Err(Refusal::Outside);
    }

    let absolute = raw.starts_with('/') || raw.as_bytes().get(1) == Some(&b':');
    if absolute {
        let folded = if cfg!(windows) {
            raw.to_lowercase()
        } else {
            raw.clone()
        };
        if !is_under(&root_key, &folded) {
            return Err(Refusal::Outside);
        }
        Ok(raw)
    } else {
        Ok(format!("{root_base}/{}", raw.trim_start_matches("./")))
    }
}

/// Resolve `.` and `..` without touching the filesystem.
///
/// `canonicalize` cannot be used here: the target of a `write_file` does not
/// exist yet, and it would also follow a symlink out of the root. Popping past
/// the root is a no-op, so `../../..` from the root stays at the root and is
/// then refused by the containment check rather than escaping it.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Drop the `\\?\` prefix `canonicalize` adds on Windows.
///
/// Without this the root would be `\\?\C:\work` while a path the model writes
/// as `C:\work\src` is `C:\work\src`, and the two would never compare equal.
fn strip_verbatim(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    let Some(rest) = text.strip_prefix(r"\\?\") else {
        return path;
    };
    match rest.strip_prefix("UNC\\") {
        Some(unc) => PathBuf::from(format!(r"\\{unc}")),
        None => PathBuf::from(rest),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A root that does not have to exist: every function here is lexical.
    fn root() -> PathBuf {
        std::env::temp_dir().join("nova_lite_workspace_unit")
    }

    #[test]
    fn a_relative_path_resolves_under_the_root() {
        let resolved = resolve_in(&root(), "src/main.rs").expect("inside the root");
        assert_eq!(resolved, root().join("src/main.rs"));
    }

    #[test]
    fn a_parent_escape_is_refused() {
        let err = resolve_in(&root(), "../secrets.txt").expect_err("must be refused");
        assert!(
            err.to_string().contains("outside the workspace"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn a_deep_parent_escape_is_refused() {
        assert!(resolve_in(&root(), "../../../../etc/passwd").is_err());
    }

    #[test]
    fn an_absolute_path_outside_is_refused() {
        let elsewhere = std::env::temp_dir().join("nova_lite_elsewhere");
        let err =
            resolve_in(&root(), &elsewhere.display().to_string()).expect_err("must be refused");
        assert!(
            err.to_string().contains("outside the workspace"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn an_absolute_path_inside_is_accepted() {
        let inside = root().join("src/lib.rs");
        let resolved = resolve_in(&root(), &inside.display().to_string())
            .expect("an absolute path inside the root is fine");
        assert_eq!(resolved, inside);
    }

    #[test]
    fn a_dot_segment_is_normalized_away() {
        let resolved = resolve_in(&root(), "./src/./main.rs").expect("inside the root");
        assert_eq!(resolved, root().join("src/main.rs"));
    }

    #[test]
    fn an_escape_that_returns_inside_is_accepted() {
        // `src/../lib.rs` never leaves the root, so it is not an escape.
        let resolved = resolve_in(&root(), "src/../lib.rs").expect("inside the root");
        assert_eq!(resolved, root().join("lib.rs"));
    }

    #[test]
    fn the_root_itself_is_inside() {
        assert!(contains(&root(), &root()));
    }

    #[test]
    fn a_sibling_with_a_shared_prefix_is_outside() {
        // The check is component-wise: `proj-other` is not inside `proj`, and
        // a string-prefix test would say it was.
        let parent = std::env::temp_dir();
        assert!(!contains(&parent.join("proj"), &parent.join("proj-other")));
    }

    #[test]
    fn a_relative_glob_is_anchored_to_the_root() {
        let anchored = anchor_pattern_in(&root(), "**/*.rs").expect("a relative pattern is fine");
        assert!(
            anchored.to_lowercase().starts_with(&key(&root())),
            "the pattern should be anchored to the root: {anchored}"
        );
        assert!(anchored.ends_with("/**/*.rs"), "unexpected: {anchored}");
    }

    #[test]
    fn a_relative_glob_that_climbs_out_is_refused() {
        let err = anchor_pattern_in(&root(), "../**/*.rs").expect_err("must be refused");
        assert!(
            err.to_string().contains("outside the workspace"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn an_absolute_glob_outside_is_refused() {
        let elsewhere = std::env::temp_dir().join("nova_lite_elsewhere/**/*.rs");
        let err = anchor_pattern_in(&root(), &elsewhere.display().to_string())
            .expect_err("must be refused");
        assert!(
            err.to_string().contains("outside the workspace"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn an_absolute_glob_inside_is_kept() {
        let inside = root().join("**/*.rs");
        let anchored = anchor_pattern_in(&root(), &inside.display().to_string())
            .expect("an absolute pattern inside the root is fine");
        assert!(anchored.contains("**/*.rs"), "unexpected: {anchored}");
    }

    #[test]
    fn an_absolute_glob_that_climbs_out_is_refused() {
        // The prefix check alone would accept this: `root/../elsewhere` starts
        // with the root. The `..` rule is what refuses it.
        let climbing = root().join("..").join("elsewhere/**/*.rs");
        let err = anchor_pattern_in(&root(), &climbing.display().to_string())
            .expect_err("must be refused");
        assert!(
            err.to_string().contains("outside the workspace"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn a_workspace_resolves_against_its_own_root() {
        let dir = std::env::temp_dir();
        let ws = Workspace::new(&dir).expect("the temp dir is a valid root");
        let resolved = ws.resolve("src/main.rs").expect("inside the root");
        assert_eq!(resolved, ws.root().join("src/main.rs"));
        assert!(ws.resolve("../escape").is_err());
    }

    // ------------------------------------------------------ the shell scan

    /// A workspace over a real directory, for the tests that need one.
    fn real() -> Workspace {
        Workspace::new(std::env::temp_dir()).expect("the temp dir is a valid root")
    }

    #[test]
    fn an_ordinary_command_reaches_nothing() {
        let ws = real();
        for command in [
            "cargo check",
            "cargo test -- --nocapture",
            "git log --oneline -n 5",
            "sed -i 's/a/b/' src/lib.rs",
            "grep -rn \"foo\" src/",
        ] {
            let reach = ws.reach_of(command);
            assert!(reach.is_contained(), "{command}: {reach:?}");
        }
    }

    #[test]
    #[cfg(windows)]
    fn a_windows_flag_is_not_a_path() {
        // `for /L` and `cmd /c` are flags that happen to start with the
        // separator a Unix path starts with. Reading them as paths would
        // report every `cmd` invocation as leaving the workspace.
        let ws = real();
        for command in [
            "for /L %i in (1,1,10) do @echo x",
            "cmd /c echo hi",
            "robocopy src dst /E",
        ] {
            assert!(ws.reach_of(command).outside.is_empty(), "{command}");
        }
    }

    #[test]
    fn a_command_that_climbs_out_is_reported() {
        let ws = real();
        let outside = ws
            .root()
            .parent()
            .expect("the temp dir has a parent")
            .join("secrets.txt");
        for command in [
            "cat ../secrets.txt",
            "sh -c \"cat ../secrets.txt\"",
            "cp src/lib.rs ../secrets.txt",
        ] {
            let reach = ws.reach_of(command);
            assert_eq!(reach.outside, vec![outside.clone()], "{command}");
        }
    }

    #[test]
    fn climbing_out_without_naming_anywhere_is_reported() {
        let ws = real();
        let reach = ws.reach_of("cd .. && cat secret.txt");
        assert_eq!(
            reach.outside,
            vec![ws.root().parent().expect("a parent").to_path_buf()],
            "{reach:?}"
        );
    }

    #[test]
    fn a_path_in_a_flag_is_reported() {
        // `--flag=path` is one word to the shell and two to the reader. The
        // path is what matters, so the scan takes what follows the `=`.
        let ws = real();
        let reach = ws.reach_of("cargo test --manifest-path=../other/Cargo.toml");
        assert_eq!(reach.outside.len(), 1, "{reach:?}");
        assert!(reach.outside[0].ends_with("other/Cargo.toml"), "{reach:?}");
    }

    #[test]
    fn the_same_path_twice_is_reported_once() {
        let ws = real();
        let reach = ws.reach_of("diff ../a.txt ../a.txt");
        assert_eq!(reach.outside.len(), 1, "{reach:?}");
    }

    #[test]
    fn a_granted_directory_is_not_outside() {
        // The user already answered for this directory; asking again, or
        // warning about it again, would make the answer worthless.
        let ws = real();
        let elsewhere = ws.root().parent().expect("a parent").to_path_buf();
        assert!(!ws.reach_of("cat ../a.txt").is_contained());
        ws.grant(elsewhere);
        assert!(ws.reach_of("cat ../a.txt").is_contained());
    }

    #[test]
    fn indirection_is_named_rather_than_resolved() {
        let ws = real();
        for (command, expected) in [
            ("cat $(cat list.txt)", "$(...)"),
            ("cat `cat list.txt`", "`...`"),
            ("cat ${HOME}/x", "${...}"),
            ("cat $HOME/x", "$NAME"),
            ("eval cat x", "eval"),
        ] {
            let reach = ws.reach_of(command);
            assert!(
                reach.unresolvable.iter().any(|u| u.contains(expected)),
                "{command}: {reach:?}"
            );
        }
    }

    #[test]
    fn a_regex_anchor_is_not_a_variable() {
        // `$` before anything but a name is an ordinary character, and a
        // scan that read `grep "foo$"` as indirection would refuse half the
        // commands a real session runs.
        let ws = real();
        let reach = ws.reach_of("grep -n \"foo$\" src/lib.rs");
        assert!(reach.is_contained(), "{reach:?}");
    }

    #[test]
    fn a_home_relative_path_is_outside() {
        // The one path shape that is outside the workspace without looking
        // like it: PowerShell expands `~`, so the scan has to as well.
        let Some(home) = std::env::home_dir() else {
            return;
        };
        let ws = real();
        if ws.may_touch(&home) {
            // The temp directory is inside the home directory on some
            // systems, and then there is nothing to report.
            return;
        }
        let reach = ws.reach_of("cat ~/.ssh/id_rsa");
        assert_eq!(reach.outside, vec![home.join(".ssh/id_rsa")], "{reach:?}");
    }

    #[test]
    fn two_workspaces_do_not_share_a_root() {
        // The property the process-global root could not have: two roots in
        // one process, each resolving against itself.
        let a = Workspace::new(std::env::temp_dir()).expect("a valid root");
        let b = Workspace::new(std::env::temp_dir().join("..")).expect("a valid root");
        assert_eq!(a.resolve("x").expect("inside"), a.root().join("x"));
        assert_eq!(b.resolve("x").expect("inside"), b.root().join("x"));
    }
}
