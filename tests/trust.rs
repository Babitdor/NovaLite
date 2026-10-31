//! First-run approval: the question, the answer, and what is remembered.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rust_agent::trust::{self, TrustStore};
use std::io::Cursor;
use std::path::PathBuf;

/// A store file of this test's own, so nothing here touches the real one.
fn store_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "nova_lite_trust_{}_{}.json",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_file(&path);
    path
}

/// A directory to approve. It does not have to exist: the store records the
/// path, it does not inspect it.
fn dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "nova_lite_trust_dir_{}_{}",
        std::process::id(),
        name
    ))
}

#[test]
fn the_prompt_names_the_directory_and_reads_one_line() {
    let root = dir("prompted");
    let mut reader = Cursor::new("y\n");
    let mut writer: Vec<u8> = Vec::new();

    let approved = trust::prompt(&root, &mut reader, &mut writer).expect("prompting");

    assert!(approved, "`y` should approve");
    let shown = String::from_utf8(writer).unwrap();
    assert!(
        shown.contains(&root.display().to_string()),
        "the prompt should name the directory it is asking about: {shown}"
    );
}

#[test]
fn a_declined_prompt_returns_false() {
    for answer in ["\n", "n\n", "no\n", "maybe\n"] {
        let mut reader = Cursor::new(answer);
        let mut writer: Vec<u8> = Vec::new();
        let approved = trust::prompt(&dir("declined"), &mut reader, &mut writer).unwrap();
        assert!(!approved, "'{answer}' should not approve");
    }
}

#[test]
fn an_end_of_input_is_not_an_approval() {
    // A piped or closed stdin must leave the gate shut rather than hang or
    // approve by default.
    let mut reader = Cursor::new("");
    let mut writer: Vec<u8> = Vec::new();
    let approved = trust::prompt(&dir("eof"), &mut reader, &mut writer).unwrap();
    assert!(!approved, "an empty stdin must not approve");
}

#[test]
fn a_missing_store_file_reads_as_nothing_approved() {
    let store = TrustStore::at(store_path("missing"));
    assert!(store.entries().is_empty());
    assert!(!store.is_trusted(&dir("anything")));
}

#[test]
fn a_corrupt_store_file_reads_as_nothing_approved() {
    let path = store_path("corrupt");
    std::fs::write(&path, "not json at all").unwrap();
    let store = TrustStore::at(path.clone());
    assert!(
        store.entries().is_empty(),
        "a corrupt store must read as empty, not panic"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_approved_directory_is_remembered() {
    let path = store_path("remembered");
    let store = TrustStore::at(path.clone());
    let root = dir("remembered");

    assert!(!store.is_trusted(&root), "nothing is approved yet");
    store.trust(&root).expect("recording the approval");
    assert!(store.is_trusted(&root), "the approval should be remembered");

    // A second store over the same file sees it too, so it survives the
    // process that wrote it.
    assert!(TrustStore::at(path.clone()).is_trusted(&root));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_different_directory_is_not_trusted() {
    let path = store_path("sibling");
    let store = TrustStore::at(path.clone());
    store.trust(&dir("approved")).unwrap();

    assert!(
        !store.is_trusted(&dir("not_approved")),
        "approving one directory must not approve another"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn an_already_trusted_directory_is_not_asked_again() {
    let path = store_path("silent");
    let store = TrustStore::at(path.clone());
    let root = dir("silent");
    store.trust(&root).unwrap();

    // An empty reader: if the prompt ran, it would read EOF and decline.
    let mut reader = Cursor::new("");
    let mut writer: Vec<u8> = Vec::new();
    let approved = trust::ensure_trusted(&root, &store, &mut reader, &mut writer).unwrap();

    assert!(approved, "an approved directory should be let through");
    assert!(
        writer.is_empty(),
        "nothing should have been asked: {}",
        String::from_utf8_lossy(&writer)
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_first_run_is_asked_and_then_recorded() {
    let path = store_path("first_run");
    let store = TrustStore::at(path.clone());
    let root = dir("first_run");

    let mut reader = Cursor::new("yes\n");
    let mut writer: Vec<u8> = Vec::new();
    let approved = trust::ensure_trusted(&root, &store, &mut reader, &mut writer).unwrap();

    assert!(approved, "`yes` should approve");
    assert!(store.is_trusted(&root), "the approval should be recorded");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_declined_first_run_records_nothing() {
    let path = store_path("declined_first_run");
    let store = TrustStore::at(path.clone());
    let root = dir("declined_first_run");

    let mut reader = Cursor::new("n\n");
    let mut writer: Vec<u8> = Vec::new();
    let approved = trust::ensure_trusted(&root, &store, &mut reader, &mut writer).unwrap();

    assert!(!approved, "`n` should decline");
    assert!(
        !store.is_trusted(&root),
        "a declined directory must not be recorded"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_default_store_lives_outside_any_workspace() {
    // The store must not be writable by the tools it gates, so it cannot sit
    // inside the directory being approved.
    let path = trust::default_store_path();
    assert!(
        path.to_string_lossy().contains("nova-lite"),
        "unexpected store path: {}",
        path.display()
    );
    assert!(
        path.is_absolute(),
        "the store path should be absolute: {}",
        path.display()
    );
}
