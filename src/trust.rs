//! First-run approval for the directory the agent is allowed to touch.
//!
//! The approval is recorded outside the workspace, in the user's own config
//! directory. A marker inside the workspace would be writable by the very
//! tools it is meant to gate, so the agent could grant itself access.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use crate::workspace;

/// The record of which directories have been approved.
pub struct TrustStore {
    path: PathBuf,
}

impl TrustStore {
    /// A store backed by a specific file. Tests use this; the binary uses
    /// [`TrustStore::default_location`].
    pub fn at(path: PathBuf) -> Self {
        Self { path }
    }

    /// The store in the user's config directory.
    pub fn default_location() -> Self {
        Self::at(default_store_path())
    }

    /// The file the approvals are written to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The approved directories, in [`workspace::key`] form.
    ///
    /// A missing or unreadable file reads as "nothing approved", which is the
    /// safe direction: the worst case is being asked again.
    pub fn entries(&self) -> Vec<String> {
        let Ok(text) = std::fs::read_to_string(&self.path) else {
            return Vec::new();
        };
        serde_json::from_str::<Vec<String>>(&text).unwrap_or_default()
    }

    /// Whether `root` has already been approved.
    pub fn is_trusted(&self, root: &Path) -> bool {
        let key = workspace::key(root);
        self.entries().iter().any(|entry| entry == &key)
    }

    /// Record `root` as approved.
    pub fn trust(&self, root: &Path) -> Result<(), String> {
        let key = workspace::key(root);
        let mut entries = self.entries();
        if entries.iter().any(|entry| entry == &key) {
            return Ok(());
        }
        entries.push(key);

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("creating '{}': {e}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(&entries)
            .map_err(|e| format!("encoding the trust list: {e}"))?;
        std::fs::write(&self.path, text)
            .map_err(|e| format!("writing '{}': {e}", self.path.display()))
    }
}

/// Where the trust list lives: the user's config directory, never the
/// workspace.
pub fn default_store_path() -> PathBuf {
    #[cfg(windows)]
    let base = std::env::var_os("APPDATA").map(PathBuf::from);
    #[cfg(not(windows))]
    let base = std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"));

    base.unwrap_or_else(std::env::temp_dir)
        .join("nova-lite")
        .join("trusted.json")
}

/// Whether a line of input is an approval.
///
/// Anything that is not an explicit yes is a no. The prompt is a gate, so a
/// stray newline, an EOF, or a typo must all leave it shut.
pub fn parse_answer(line: &str) -> bool {
    matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

/// Ask the user to approve `root`, reading one line from `reader`.
///
/// Split from [`ensure_trusted`] so the question can be tested without a
/// terminal: the reader and the writer are both injected.
pub fn prompt(
    root: &Path,
    reader: &mut impl BufRead,
    writer: &mut impl Write,
) -> std::io::Result<bool> {
    writeln!(writer, "nova-lite wants to work in:")?;
    writeln!(writer, "  {}", root.display())?;
    writeln!(writer)?;
    writeln!(
        writer,
        "Its file tools can read and write inside that directory only, and every\n\
         write, edit and command is shown to you before it runs."
    )?;
    writeln!(
        writer,
        "A command you approve runs with that directory as its working directory,\n\
         but a working directory is not a sandbox: the command itself can name any\n\
         path it likes. Approve commands you have read."
    )?;
    write!(writer, "Approve? [y/N] ")?;
    writer.flush()?;

    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(parse_answer(&line))
}

/// Ensure `root` is approved, asking the first time it is seen.
///
/// Returns `false` when the user declined, which the caller turns into an
/// exit: nothing has been read or written at that point.
pub fn ensure_trusted(
    root: &Path,
    store: &TrustStore,
    reader: &mut impl BufRead,
    writer: &mut impl Write,
) -> std::io::Result<bool> {
    if store.is_trusted(root) {
        return Ok(true);
    }
    if !prompt(root, reader, writer)? {
        return Ok(false);
    }
    if let Err(e) = store.trust(root) {
        // Not fatal: the approval holds for this run, it just will not be
        // remembered, so the question comes back next time.
        writeln!(
            writer,
            "Could not record the approval ({e}); it will be asked again next time."
        )?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_explicit_yes_approves() {
        for yes in ["y", "Y", "yes", "YES", " Yes \n"] {
            assert!(parse_answer(yes), "'{yes}' should approve");
        }
        for no in ["", "\n", "n", "no", "maybe", "yep", "1"] {
            assert!(!parse_answer(no), "'{no}' should not approve");
        }
    }
}
