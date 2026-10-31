//! Reading and replacing a file without losing it.
//!
//! Two things live here that the tool path needs and that are easy to get
//! subtly wrong in a tool body:
//!
//! * **A version.** An approval is bound to the exact bytes that were
//!   previewed, so the file has to be re-checked between the preview and the
//!   commit. [`Version`] is that check: it is what the file held, and
//!   comparing it to what the file holds now is what turns "the user approved
//!   this diff" into "the user approved this diff *of this file*".
//! * **An atomic replace.** `fs::write` truncates before it writes, so a
//!   failure partway through leaves a truncated file: the original is gone
//!   and the new content is incomplete. Writing a sibling temp file and
//!   renaming it over the target means the file is either the old content or
//!   the new one, never a prefix of either.
//!
//! The confinement rules (what a path may resolve to) are in
//! [`crate::workspace`]; this module is about the bytes once the path is
//! settled.

use std::path::Path;

/// What a file held when it was read.
///
/// The point of naming it is that a write can be bound to it: the approval
/// previews a diff against one version, and the commit refuses if the file is
/// no longer that version. Without it, a file edited by the user (or by
/// another tool call) between the preview and the commit would be overwritten
/// with a change computed against text that is no longer there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Version {
    /// The file did not exist.
    Absent,
    /// The file held exactly this text.
    Text(String),
}

impl Version {
    /// The version of `path` right now.
    ///
    /// A missing file is [`Version::Absent`], which is a fact rather than an
    /// error: creating a file is a normal thing to do. Any other read failure
    /// is returned, because "I could not read it" is not the same as "it is
    /// not there", and treating the two alike would let a write proceed over
    /// a file whose contents were never previewed.
    pub async fn of(path: &Path) -> std::io::Result<Self> {
        match tokio::fs::read_to_string(path).await {
            Ok(text) => Ok(Self::Text(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::Absent),
            Err(e) => Err(e),
        }
    }

    /// The text, if the file existed.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Absent => None,
        }
    }

    /// Whether the file is still what it was.
    pub fn is(&self, other: &Self) -> bool {
        self == other
    }

    /// A digest of the content, for "has this changed since I looked?".
    ///
    /// The point of a digest rather than the text is that a session can keep
    /// one per file it has read without holding every file it has read in
    /// memory. It is not a cryptographic hash and it is not stable across
    /// builds; it is only ever compared against another digest taken in the
    /// same process, which is the only comparison that is made.
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        match self {
            Self::Absent => 0u8.hash(&mut hasher),
            Self::Text(text) => {
                1u8.hash(&mut hasher);
                text.hash(&mut hasher);
            }
        }
        hasher.finish()
    }
}

/// Distinguishes concurrent temp files, so two writes to one path in the same
/// turn cannot collide.
static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Write `content` to `path`, replacing it atomically where the platform
/// allows.
///
/// The temp file is a sibling because a rename across volumes is a copy,
/// which is not atomic. Every failure before the rename takes the temp file
/// with it: a stray `.name.nova-*.tmp` in the user's source tree would be
/// picked up by glob, grep and list_dir and read as part of the project.
///
/// The temp file is created **exclusively** (`create_new`), so a name that
/// already exists is an error rather than something to write through. The
/// name carries the process id and a counter, which makes a collision
/// unlikely; `create_new` is what makes it harmless, because a symlink
/// planted at the temp path is refused instead of followed.
///
/// The rename is not always available: on Windows it fails where the target
/// is held open by another process without a delete share. That case is
/// reported rather than worked around. The old code fell back to a plain
/// truncating write, which is exactly the failure this function exists to
/// prevent: a file that is neither the old content nor the new one.
///
/// # Why the sibling temp file also closes the swap race
///
/// The caller checks that the target is inside the workspace, and then this
/// function writes. Between the two, a directory in the chain could be
/// replaced by a link pointing outside. That does not let the write escape:
/// the temp file was created in the *old* directory, so the rename looks for
/// it under the new one, does not find it, and fails. The file is left
/// untouched and the failure is reported.
pub async fn write_atomic(path: &Path, content: &str) -> std::io::Result<()> {
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let tmp = dir.join(format!(".{name}.nova-{}-{seq}.tmp", std::process::id()));

    if let Err(e) = write_new(&tmp, content).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    // The rename installs a new file, so the target's permissions have to be
    // carried over: a script that was executable must stay executable.
    if let Ok(meta) = tokio::fs::metadata(path).await {
        let _ = tokio::fs::set_permissions(&tmp, meta.permissions()).await;
    }
    if let Err(e) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    Ok(())
}

/// Create `path` and write `content` to it, refusing if it already exists.
///
/// `create_new` is the whole point: it is the one open mode that fails rather
/// than following a symlink or truncating an existing file, so a name planted
/// at the temp path cannot be used to redirect the write.
async fn write_new(path: &Path, content: &str) -> std::io::Result<()> {
    use tokio::io::AsyncWriteExt;

    let mut file = tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .await?;
    file.write_all(content.as_bytes()).await?;
    // Flush before the rename: a rename that lands before the bytes do would
    // leave the target pointing at a file whose contents are not yet on disk.
    file.flush().await?;
    file.sync_all().await?;
    Ok(())
}

/// Whether `path` is a regular file, following links.
///
/// A device, a FIFO, a socket or a directory is not something a file tool
/// should read or replace: reading a device can block forever, and replacing
/// one is not a thing that means anything. The check is on the real target,
/// so a link to a device is refused too.
pub async fn is_regular_file(path: &Path) -> std::io::Result<bool> {
    Ok(tokio::fs::metadata(path).await?.is_file())
}

/// The first `limit` bytes of a file, as text.
///
/// The cut can land inside a multi-byte character, so the bytes are decoded
/// lossily: a replacement character at the very end of a truncated read is
/// better than failing the read outright.
pub async fn read_head(path: &Path, limit: usize) -> std::io::Result<String> {
    use tokio::io::AsyncReadExt;

    let mut file = tokio::fs::File::open(path).await?;
    let mut buf = vec![0u8; limit];
    let mut filled = 0;
    while filled < limit {
        let n = file.read(&mut buf[filled..]).await?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    buf.truncate(filled);
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_missing_file_is_absent_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let version = Version::of(&dir.path().join("nope.txt")).await.unwrap();
        assert_eq!(version, Version::Absent);
        assert!(version.text().is_none());
    }

    #[tokio::test]
    async fn a_present_file_carries_its_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "hello\n").await.unwrap();
        let version = Version::of(&path).await.unwrap();
        assert_eq!(version, Version::Text("hello\n".to_string()));
        assert_eq!(version.text(), Some("hello\n"));
    }

    #[tokio::test]
    async fn a_directory_is_an_error_rather_than_absent() {
        // "I could not read it" is not "it is not there": treating the two
        // alike would let a write proceed over something never previewed.
        let dir = tempfile::tempdir().unwrap();
        assert!(Version::of(dir.path()).await.is_err());
    }

    #[tokio::test]
    async fn a_write_replaces_the_content_and_leaves_no_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.txt");
        tokio::fs::write(&path, "old\n").await.unwrap();
        write_atomic(&path, "new\n").await.unwrap();
        assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), "new\n");

        let mut names: Vec<String> = Vec::new();
        let mut rd = tokio::fs::read_dir(dir.path()).await.unwrap();
        while let Some(entry) = rd.next_entry().await.unwrap() {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        assert_eq!(
            names,
            vec!["a.txt".to_string()],
            "a temp file was left behind"
        );
    }

    #[tokio::test]
    async fn a_write_creates_a_file_that_was_not_there() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.txt");
        write_atomic(&path, "content\n").await.unwrap();
        assert_eq!(tokio::fs::read_to_string(&path).await.unwrap(), "content\n");
    }
}
