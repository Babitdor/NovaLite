//! Logging setup.
//!
//! The TUI owns the terminal: it is in raw mode on the alternate screen, so a
//! write to stdout or stderr lands wherever the cursor happens to be and
//! corrupts the frame. In practice the damage shows up in the input box, which
//! is where the cursor sits while the agent runs.
//!
//! Logs therefore go to a file, never to the terminal.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing_appender::non_blocking::WorkerGuard;

/// Where logs are written unless the caller overrides it.
pub fn default_log_path() -> PathBuf {
    std::env::temp_dir().join("nova-lite.log")
}

/// Install the global subscriber, writing to `path`.
///
/// The returned guard must be kept alive for the process lifetime: dropping it
/// stops the writer thread and discards anything still buffered.
pub fn init_logging_at(path: &Path) -> Result<WorkerGuard> {
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path.file_name().context("log path has no file name")?;

    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating log directory {}", dir.display()))?;

    let appender = tracing_appender::rolling::never(dir, name);
    let (writer, guard) = tracing_appender::non_blocking(appender);

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(writer)
        // A file should not carry terminal escape codes.
        .with_ansi(false)
        .try_init()
        .map_err(|e| anyhow::anyhow!("installing the tracing subscriber: {e}"))?;

    Ok(guard)
}

/// Install the subscriber at [`default_log_path`].
pub fn init_logging() -> Result<WorkerGuard> {
    init_logging_at(&default_log_path())
}
