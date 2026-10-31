//! The process runner: what a command reports, and what it leaves behind.
//!
//! Every case here is driven by a *fixture process* rather than by a shell
//! built-in, because the shell built-ins differ per platform: `ping -n 30` is
//! a Windows spelling, `sleep 30` is a Unix one, and a test that uses either
//! is a test that only runs on one machine. The fixture is this test binary
//! re-run with `--exact fixture`, so it is the same executable on every
//! platform and needs nothing installed.
//!
//! The fixture reads `NOVA_FIXTURE` to decide what to do. When the variable is
//! absent, which is the ordinary `cargo test` run, the test body returns
//! immediately and does nothing.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use rust_agent::process::{ProcessRunner, Termination};
use rust_agent::runtime::CancelToken;

/// The environment variable that turns this binary into a fixture.
const MODE: &str = "NOVA_FIXTURE";

/// The environment variable naming the file the grandchild writes.
const MARKER: &str = "NOVA_FIXTURE_MARKER";

/// The fixture process.
///
/// Reached only when `NOVA_FIXTURE` is set, which the tests below do when they
/// spawn `current_exe()`. In an ordinary run the variable is absent and this
/// is a test that passes without doing anything.
///
/// The grandchild is deliberately never waited for: leaving a descendant
/// behind is the whole point of that mode, and the test that uses it asserts
/// the runner's kill reaches the descendant anyway.
#[allow(clippy::zombie_processes)]
#[test]
fn fixture() {
    let Ok(mode) = std::env::var(MODE) else {
        return;
    };
    match mode.as_str() {
        // A nonzero exit with output on both streams. The old result reported
        // the status only when the output was empty, so this is the case that
        // made a failing command that printed anything look like a success.
        "exit-with-output" => {
            println!("stdout-before-the-failure");
            eprintln!("stderr-before-the-failure");
            std::process::exit(7);
        }

        // More output than any bound will keep, on stdout.
        "flood" => {
            for _ in 0..200_000 {
                println!("0123456789012345678901234567890123456789");
            }
        }

        // Both pipes filled at once. A runner that reads one to end-of-stream
        // before touching the other deadlocks here, because the child blocks
        // writing to the pipe nobody is draining.
        "both-pipes" => {
            let out = std::thread::spawn(|| {
                for _ in 0..200_000 {
                    println!("out");
                }
            });
            let err = std::thread::spawn(|| {
                for _ in 0..200_000 {
                    eprintln!("err");
                }
            });
            let _ = out.join();
            let _ = err.join();
        }

        // Runs far longer than any deadline a test will set.
        "hang" => {
            std::thread::sleep(Duration::from_secs(600));
        }

        // Prints the directory it was started in, so a test can check that a
        // path containing a space survived the trip.
        "print-cwd" => {
            println!("cwd={}", std::env::current_dir().unwrap().display());
        }

        // Prints every argument it was given, one per line, so a test can check
        // that an argument containing a space and a shell metacharacter arrived
        // as one argument rather than being split or interpreted.
        //
        // The arguments are read from `args()`, not from a filter, so the
        // metacharacter the test appends is visible here even though libtest
        // treats it as a filter that matches nothing.
        "print-args" => {
            for arg in std::env::args().skip(1) {
                println!("{arg}");
            }
        }

        // Spawns a grandchild that outlives it and then hangs. The grandchild
        // writes a marker file after a delay, so a test can tell whether the
        // kill reached the whole tree or only the process it was pointed at.
        "spawn-grandchild-then-hang" => {
            let marker = std::env::var(MARKER).expect("the marker path");
            let exe = std::env::current_exe().expect("the fixture's own path");
            let _grandchild = std::process::Command::new(exe)
                .args(["--exact", "fixture", "--nocapture"])
                .env(MODE, "write-marker-later")
                .env(MARKER, &marker)
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .spawn()
                .expect("the grandchild should start");
            std::thread::sleep(Duration::from_secs(600));
        }

        // The grandchild: waits, then writes the marker. If the runner's kill
        // reached it, the file never appears.
        "write-marker-later" => {
            let marker = std::env::var(MARKER).expect("the marker path");
            std::thread::sleep(Duration::from_millis(1500));
            std::fs::write(marker, "the grandchild survived the kill\n")
                .expect("the marker should be writable");
        }

        other => panic!("unknown fixture mode: {other}"),
    }
}

/// A runner over a fresh temp directory, with a generous deadline.
fn runner() -> (tempfile::TempDir, ProcessRunner) {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let runner = ProcessRunner::new(dir.path(), Duration::from_secs(30), 64 * 1024);
    (dir, runner)
}

/// Run the fixture in `mode`, in `runner`'s directory.
async fn fixture_in(
    runner: &ProcessRunner,
    mode: &str,
) -> rust_agent::error::Result<rust_agent::process::CommandOutput> {
    let exe = std::env::current_exe()
        .expect("the test binary's own path")
        .to_string_lossy()
        .into_owned();
    runner
        .run_program_with_env(
            &exe,
            &["--exact", "fixture", "--nocapture"],
            &[(MODE, mode)],
        )
        .await
}

// ------------------------------------------------------------- what it reports

#[tokio::test]
async fn a_nonzero_exit_is_reported_alongside_the_output() {
    let (_dir, runner) = runner();
    let out = fixture_in(&runner, "exit-with-output")
        .await
        .expect("a nonzero exit is a result, not a spawn failure");

    assert_eq!(out.termination, Termination::Exited(7));
    assert_eq!(out.termination.exit_code(), Some(7));
    assert!(!out.succeeded());
    assert!(
        out.stdout.contains("stdout-before-the-failure"),
        "stdout was lost: {:?}",
        out.stdout
    );
    assert!(
        out.stderr.contains("stderr-before-the-failure"),
        "stderr was lost: {:?}",
        out.stderr
    );
    // The status is in the rendered result even though there *is* output.
    let rendered = out.render(runner.max_output());
    assert!(
        rendered.contains("exit code 7"),
        "the status must survive into the result: {rendered}"
    );
    assert!(out.read_error.is_none(), "{:?}", out.read_error);
}

#[tokio::test]
async fn the_command_identity_is_carried_with_the_result() {
    // A result has to be attributable to the call that produced it without
    // the caller re-deriving the command line.
    let (_dir, runner) = runner();
    let out = fixture_in(&runner, "print-cwd")
        .await
        .expect("it should run");
    assert!(
        out.command.contains("--exact"),
        "the command line should be recorded: {}",
        out.command
    );
    assert!(out.duration > Duration::ZERO);
}

/// An argument containing a space and a shell metacharacter arrives whole.
///
/// This is the whole point of the structured form: through a shell, `a b & c`
/// is two arguments and a background job. As an array it is one argument, and
/// the program sees exactly what the caller wrote.
///
/// The program is the fixture, not `echo`, because there is no `echo` that
/// exists on both Windows and Unix: a test that names one is a test that only
/// runs on one machine, and the claim here is about the runner rather than
/// about the program.
#[tokio::test]
async fn an_argument_with_a_space_and_a_metacharacter_arrives_whole() {
    const TRICKY: &str = "a b & c";
    let (_dir, runner) = runner();
    let exe = std::env::current_exe()
        .expect("the test binary's own path")
        .to_string_lossy()
        .into_owned();
    let out = runner
        .run_program_with_env(
            &exe,
            &["--exact", "fixture", "--nocapture", TRICKY],
            &[(MODE, "print-args")],
        )
        .await
        .expect("it should run");

    assert!(out.succeeded(), "{:?}", out.render(runner.max_output()));
    // The fixture prints one argument per line, so the argument is whole
    // exactly when a line equals it. A `contains` would also pass if the
    // runner had split it and the pieces happened to be adjacent.
    assert!(
        out.stdout.lines().any(|line| line == TRICKY),
        "the argument must arrive as one argument, got: {:?}",
        out.stdout
    );
}

#[tokio::test]
async fn both_pipes_are_drained_concurrently() {
    // Reading one pipe to end-of-stream before touching the other deadlocks
    // as soon as the child fills the second. The deadline is what turns that
    // deadlock into a failure rather than a hang.
    let (_dir, runner) = runner();
    let started = Instant::now();
    let out = fixture_in(&runner, "both-pipes")
        .await
        .expect("it should run");

    assert_eq!(out.termination, Termination::Exited(0));
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "the two pipes should be read at the same time, took {:?}",
        started.elapsed()
    );
    assert!(out.stdout.contains("out"), "stdout was lost");
    assert!(out.stderr.contains("err"), "stderr was lost");
}

#[tokio::test]
async fn output_past_the_bound_is_dropped_and_counted() {
    let (_dir, runner) = runner();
    let out = fixture_in(&runner, "flood").await.expect("it should run");

    assert_eq!(out.termination, Termination::Exited(0));
    assert!(
        out.dropped_bytes > 0,
        "the flood should have exceeded the bound"
    );
    // The fixture runs under the test harness, so the harness's own lines are
    // interleaved with the flood and neither end of the stream is the flood's
    // own first or last line. What can be asserted is the shape: the kept
    // bytes are near the bound, and they hold the command's real output.
    assert!(
        out.stdout.contains("0123456789"),
        "the command's own output should be kept"
    );
    // Both ends are kept, not just the front. A head-only bound would keep
    // half the bound; keeping the tail as well is what makes the total
    // approach the whole of it. A compiler prints its diagnostics last, so a
    // head-only bound would show the progress lines and hide the errors.
    assert!(
        out.stdout.len() > runner.max_output(),
        "both ends should be kept, so the total should exceed half the bound; got {} bytes",
        out.stdout.len()
    );
    assert!(
        out.stdout.len() <= runner.max_output() + 1024,
        "the kept output should be near the bound, got {} bytes",
        out.stdout.len()
    );
    let rendered = out.render(runner.max_output());
    assert!(
        rendered.contains("bytes of output were dropped"),
        "the truncation must be reported: {}",
        &rendered[..rendered.len().min(200)]
    );
}

// ------------------------------------------------------------- the deadline

#[tokio::test]
async fn a_command_that_outlives_its_deadline_is_killed_and_its_output_kept() {
    let dir = tempfile::tempdir().unwrap();
    let runner = ProcessRunner::new(dir.path(), Duration::from_millis(400), 64 * 1024);

    let started = Instant::now();
    let out = fixture_in(&runner, "hang").await.expect("it should run");

    assert_eq!(out.termination, Termination::TimedOut);
    assert!(!out.succeeded());
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the deadline should have fired promptly, took {:?}",
        started.elapsed()
    );
    // The kill closed the pipes, so the readers saw end-of-stream rather than
    // waiting out the fixture's ten minutes.
    assert!(out.read_error.is_none(), "{:?}", out.read_error);
}

#[tokio::test]
async fn a_killed_command_leaves_no_descendant_behind() {
    // The case the whole job-object / process-group machinery exists for. The
    // fixture spawns a grandchild that outlives it and writes a marker file
    // after a delay; if the kill reached only the process it was pointed at,
    // the grandchild writes the file and the test fails.
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("survived.txt");
    let runner = ProcessRunner::new(dir.path(), Duration::from_millis(400), 64 * 1024);

    let exe = std::env::current_exe()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    let out = runner
        .run_program_with_env(
            &exe,
            &["--exact", "fixture", "--nocapture"],
            &[
                (MODE, "spawn-grandchild-then-hang"),
                (MARKER, &marker.to_string_lossy()),
            ],
        )
        .await
        .expect("it should run");

    assert_eq!(out.termination, Termination::TimedOut);

    // The grandchild writes at 1.5s. Wait past that, then look.
    tokio::time::sleep(Duration::from_millis(2500)).await;
    assert!(
        !marker.exists(),
        "a descendant outlived the kill and wrote {}",
        marker.display()
    );
}

// ------------------------------------------------------------- cancellation

#[tokio::test]
async fn a_cancelled_command_is_killed_and_reports_that_it_was() {
    let dir = tempfile::tempdir().unwrap();
    // A deadline far beyond the test, so the only thing that can stop it is
    // the cancel token.
    let runner = ProcessRunner::new(dir.path(), Duration::from_secs(600), 64 * 1024);
    let cancel = CancelToken::new();

    let stopper = cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        stopper.stop();
    });

    let started = Instant::now();
    let out = fixture_in_until(&runner, "hang", &cancel)
        .await
        .expect("a cancelled command is a result, not a spawn failure");

    assert_eq!(out.termination, Termination::Cancelled);
    assert!(!out.succeeded());
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "the cancel should have fired promptly, took {:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_command_that_finishes_before_the_cancel_is_not_marked_cancelled() {
    // The token is watched, not obeyed: a command that exits on its own keeps
    // its own exit code even if the run is stopped a moment later.
    let (_dir, runner) = runner();
    let cancel = CancelToken::new();
    let out = fixture_in_until(&runner, "print-cwd", &cancel)
        .await
        .expect("it should run");
    assert_eq!(out.termination, Termination::Exited(0));
    cancel.stop();
}

// ------------------------------------------------------------- paths

#[tokio::test]
async fn a_working_directory_containing_a_space_is_used_as_given() {
    // The directory is passed to the child as one value, not as a string a
    // shell re-splits, so a space in it is a space in the name.
    let base = tempfile::tempdir().unwrap();
    let spaced = base.path().join("a directory with spaces");
    std::fs::create_dir(&spaced).unwrap();
    let runner = ProcessRunner::new(&spaced, Duration::from_secs(30), 64 * 1024);

    let out = fixture_in(&runner, "print-cwd")
        .await
        .expect("it should run");
    assert_eq!(out.termination, Termination::Exited(0));
    assert!(
        out.stdout.contains("a directory with spaces"),
        "the child should have started in the spaced directory: {:?}",
        out.stdout
    );
}

#[tokio::test]
async fn a_program_that_does_not_exist_is_a_spawn_failure() {
    // Not a result with a nonzero exit: there was no process to exit. The
    // distinction matters, because the model should not be told a command
    // "failed" when it never ran.
    let (_dir, runner) = runner();
    let err = runner
        .run_program("nova-lite-no-such-program-anywhere", &[])
        .await
        .expect_err("a missing program cannot be spawned");
    assert!(
        err.to_string().contains("spawning"),
        "the error should say what failed: {err}"
    );
}

#[tokio::test]
async fn the_displayed_command_line_quotes_an_argument_with_a_space() {
    // The approval prompt shows this string, so it has to be the command that
    // will run rather than a paraphrase that reads as two arguments.
    let line = rust_agent::process::program_line("cargo", &["test", "--", "a b"]);
    assert_eq!(line, r#"cargo test -- "a b""#);
    let plain = rust_agent::process::program_line("git", &["diff", "--stat"]);
    assert_eq!(plain, "git diff --stat");
}

// ------------------------------------------------------------- helpers

/// Run the fixture in `mode`, watched by `cancel`.
async fn fixture_in_until(
    runner: &ProcessRunner,
    mode: &str,
    cancel: &CancelToken,
) -> rust_agent::error::Result<rust_agent::process::CommandOutput> {
    let exe = std::env::current_exe()
        .expect("the test binary's own path")
        .to_string_lossy()
        .into_owned();
    runner
        .run_program_until_with_env(
            &exe,
            &["--exact", "fixture", "--nocapture"],
            &[(MODE, mode)],
            cancel,
        )
        .await
}
