//! The subscriber test, alone in its own binary.
//!
//! `tracing` allows the global subscriber to be installed only once per
//! process, and every test in a binary shares that process. Keeping this test
//! by itself means it always owns the global, so it cannot fail for a reason
//! unrelated to what it checks.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

fn tmp_log(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "rust_agent_log_sub_{}_{}",
        std::process::id(),
        name
    ));
    p
}

/// Wait for the non-blocking writer thread to flush `needle` into `path`.
fn wait_for(path: &std::path::Path, needle: &str) -> String {
    for _ in 0..100 {
        let contents = std::fs::read_to_string(path).unwrap_or_default();
        if contents.contains(needle) {
            return contents;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    std::fs::read_to_string(path).unwrap_or_default()
}

#[test]
fn events_are_written_to_the_log_file_and_not_the_terminal() {
    let path = tmp_log("main.log");
    let _ = std::fs::remove_file(&path);

    let _guard = rust_agent::logging::init_logging_at(&path).expect("init logging");

    tracing::info!(target: "rust_agent_test", "hello_from_the_agent");
    tracing::warn!(target: "rust_agent_test", "plain_text_only");

    // Wait for the *last* event, not the first. The writer is asynchronous and
    // flushes in batches, so the first line appearing says nothing about the
    // second: waiting on the first and then asserting on the second is a race
    // that passes on an idle machine and fails under a full parallel run.
    let contents = wait_for(&path, "plain_text_only");

    // The event reached the file.
    assert!(
        contents.contains("hello_from_the_agent"),
        "the event should be in the log file, got: {contents:?}"
    );
    assert!(
        contents.contains("plain_text_only"),
        "every level should be written, got: {contents:?}"
    );

    // A file should not carry terminal escape codes: their presence would mean
    // the writer is still configured for a terminal.
    assert!(
        !contents.contains('\u{1b}'),
        "log file should not contain escape codes: {contents:?}"
    );

    let _ = std::fs::remove_file(&path);
}
