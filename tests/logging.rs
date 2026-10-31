//! Logging must never touch the terminal.
//!
//! The TUI runs in raw mode on the alternate screen, so a write to stdout or
//! stderr lands at the cursor and corrupts the frame (it shows up in the input
//! box). These tests cover the path and directory handling.
//!
//! The test that actually installs the subscriber lives in
//! `logging_subscriber.rs`: `tracing` allows one global subscriber per process,
//! and tests within a binary share a process.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

fn tmp_dir(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("rust_agent_log_{}_{}", std::process::id(), name));
    p
}

#[test]
fn the_default_log_path_is_an_absolute_file_path() {
    // A guard against regressing to `with_writer(std::io::stderr)`.
    let path = rust_agent::logging::default_log_path();
    assert!(
        path.is_absolute(),
        "the default log path should be absolute, got {}",
        path.display()
    );
    assert!(
        path.parent().is_some(),
        "the default log path should have a parent directory"
    );
    assert!(
        path.file_name().is_some(),
        "the default log path should name a file"
    );
}

#[test]
fn init_logging_creates_a_missing_parent_directory() {
    // Directory handling happens before `try_init`, so this is safe to run
    // without owning the global subscriber.
    let dir = tmp_dir("nested");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("deeper").join("agent.log");

    let _ = rust_agent::logging::init_logging_at(&path);

    assert!(
        dir.join("deeper").exists(),
        "the parent directory should have been created"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_log_path_with_no_parent_is_accepted() {
    // `Path::parent()` returns `Some("")` for a bare file name; the directory
    // step must not try to create an empty path.
    let _ = rust_agent::logging::init_logging_at(std::path::Path::new("bare-name.log"));
    let _ = std::fs::remove_file("bare-name.log");
}
