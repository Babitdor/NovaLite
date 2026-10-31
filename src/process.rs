//! One place a child process is spawned.
//!
//! Every command the agent runs goes through [`ProcessRunner`], so the
//! deadline, the output bound, the working directory and the kill path are
//! decided once rather than per call site. The runner is built from the
//! session's [`crate::config::Config`] and the session's
//! [`crate::workspace::Workspace`], so a command runs in the directory the
//! session was opened on and under the timeout the session was configured
//! with.
//!
//! The runner is not a sandbox. It sets the working directory, which keeps a
//! *relative* path inside the workspace, and it bounds how long a command may
//! run and how much output it may produce. It does not inspect the command
//! string, so an absolute path or a `cd ..` still reaches outside. That is
//! documented rather than papered over: an approved arbitrary command is not
//! a filesystem sandbox, and pretending otherwise would be worse than saying
//! so.
//!
//! # What a caller gets back
//!
//! A command that runs and fails is a *successful call* with a nonzero exit
//! code, because the model needs the output either way. A command that cannot
//! be spawned at all is an error. A command that outlives its deadline, or
//! that is cancelled, is also a successful call: it carries the output it
//! produced before it was killed, and a [`Termination`] that says why it
//! stopped. Discarding that output was the old behaviour, and it threw away
//! exactly the diagnostics a failing build had already printed.
//!
//! # Killing a tree
//!
//! Killing the shell is not enough. `cmd /C` and `sh -c` run the real command
//! as a *child* of the shell, and that child holds the pipes open, so a
//! reader waiting for end-of-stream waits for the grandchild to finish on its
//! own. On Windows the runner puts the child in a Job Object with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, so terminating the job takes the
//! whole tree with it. On Unix the child is put in a session of its own with
//! `setsid`, so its process group id is its pid and `killpg` reaches every
//! descendant that did not create a group of its own.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::runtime::CancelToken;

/// How long the runner waits for the output readers after the child is dead.
///
/// The child's death closes its pipes, so the readers finish on their own.
/// This bound is for the case where they do not: a grandchild that inherited
/// the pipe and that the kill did not reach. Waiting forever there would hang
/// the session, and whatever was read by then is worth keeping.
const DRAIN_GRACE: Duration = Duration::from_secs(2);

/// How a command ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// The process exited on its own with this code.
    Exited(i32),
    /// The process was killed by a signal, or the platform reported no exit
    /// code. On Windows a process killed by `TerminateProcess` has no exit
    /// code of its own, so a hard kill also lands here.
    Signalled,
    /// The runner killed it for outliving its deadline.
    TimedOut,
    /// The run was cancelled while the command was running.
    Cancelled,
}

impl Termination {
    /// The exit code, when the process exited on its own.
    pub fn exit_code(self) -> Option<i32> {
        match self {
            Self::Exited(code) => Some(code),
            _ => None,
        }
    }

    /// A short phrase naming how it ended.
    pub fn describe(self) -> String {
        match self {
            Self::Exited(code) => format!("exit code {code}"),
            Self::Signalled => "killed by a signal".to_string(),
            Self::TimedOut => "killed: it outlived its deadline".to_string(),
            Self::Cancelled => "killed: the run was cancelled".to_string(),
        }
    }
}

/// What a finished command produced.
///
/// The exit status is carried separately from the text so a caller can tell
/// "the command failed" from "the command printed nothing", which the old
/// single-string result could not: it reported the status only when the
/// output was empty, so a failing command that printed anything looked like a
/// success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandOutput {
    /// The command line, as it was run. The identity of the command, so a
    /// result can be attributed to a call without the caller re-deriving it.
    pub command: String,
    /// How it ended.
    pub termination: Termination,
    /// The command's standard output, bounded, with the middle elided.
    pub stdout: String,
    /// The command's standard error, bounded, with the middle elided.
    pub stderr: String,
    /// How many bytes of output were dropped to stay inside the bound.
    pub dropped_bytes: usize,
    /// How long the command ran.
    pub duration: Duration,
    /// A failure while reading the command's output, if there was one.
    ///
    /// Reported rather than swallowed: the old reader treated a read error as
    /// end-of-stream, so a command whose output could not be read looked like
    /// a command that printed nothing.
    pub read_error: Option<String>,
}

impl CommandOutput {
    /// Whether the command exited zero.
    pub fn succeeded(&self) -> bool {
        self.termination == Termination::Exited(0)
    }

    /// The first line of what the command printed, for a one-line summary.
    ///
    /// Standard error is preferred when there is any: a command that failed
    /// usually says why on stderr, and a summary that quoted the progress
    /// chatter on stdout instead would name the wrong thing.
    pub fn first_line(&self) -> String {
        let source = if self.stderr.trim().is_empty() {
            &self.stdout
        } else {
            &self.stderr
        };
        source
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or("")
            .to_string()
    }

    /// The result as the model sees it.
    ///
    /// `limit` is the byte bound the runner kept, and is only used to word the
    /// truncation notice.
    pub fn render(&self, limit: usize) -> String {
        let mut result = String::new();
        if !self.stdout.is_empty() {
            result.push_str(&self.stdout);
        }
        if !self.stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("[stderr] ");
            result.push_str(&self.stderr);
        }
        if result.is_empty() {
            result = format!("(no output, {})", self.termination.describe());
        } else if !self.succeeded() {
            // The status is reported even when there is output. Reporting it
            // only when the output was empty is what made a failing command
            // that printed anything look like a success.
            result.push_str(&format!("\n[{}]", self.termination.describe()));
        }
        if self.dropped_bytes > 0 {
            result.push_str(&format!(
                "\n\n... [{} bytes of output were dropped; the first and last {limit} bytes are \
                 shown. Narrow the command (head, findstr, Select-Object -First) to see the \
                 rest.] ...",
                self.dropped_bytes
            ));
        }
        if let Some(error) = &self.read_error {
            result.push_str(&format!(
                "\n\n... [reading the command's output failed: {error}] ..."
            ));
        }
        result
    }
}

/// A byte buffer that keeps the beginning *and* the end of a stream.
///
/// Keeping only the beginning loses the part that matters most: a compiler
/// prints its diagnostics last, so a truncated `cargo check` would show the
/// progress lines and hide the errors. Keeping only the end loses the command
/// echo and the first error. So both ends are kept and the middle is dropped,
/// with a count of what went.
#[derive(Debug)]
struct Bounded {
    head: Vec<u8>,
    tail: VecDeque<u8>,
    head_cap: usize,
    tail_cap: usize,
    dropped: usize,
}

impl Bounded {
    /// A buffer that keeps at most `limit` bytes, half at each end.
    fn new(limit: usize) -> Self {
        let head_cap = limit / 2;
        Self {
            head: Vec::new(),
            tail: VecDeque::new(),
            head_cap,
            tail_cap: limit - head_cap,
            dropped: 0,
        }
    }

    /// Add `bytes`, evicting the oldest of the tail when it is full.
    fn push(&mut self, bytes: &[u8]) {
        let mut rest = bytes;
        if self.head.len() < self.head_cap {
            let room = self.head_cap - self.head.len();
            let take = room.min(rest.len());
            self.head.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
        }
        for &byte in rest {
            if self.tail.len() == self.tail_cap {
                self.tail.pop_front();
                self.dropped += 1;
            }
            self.tail.push_back(byte);
        }
    }

    /// The kept bytes, with a marker where the middle was dropped, and how
    /// many bytes went.
    fn finish(&self) -> (Vec<u8>, usize) {
        let mut out = self.head.clone();
        if self.dropped > 0 {
            out.extend_from_slice(
                format!("\n... [{} bytes dropped] ...\n", self.dropped).as_bytes(),
            );
        }
        out.extend(self.tail.iter().copied());
        (out, self.dropped)
    }
}

/// What to run.
#[derive(Debug, Clone)]
enum Spec {
    /// A command line handed to the platform shell.
    Shell(String),
    /// An executable and its arguments, with no shell in between.
    ///
    /// The shell is what makes a command line ambiguous: quoting, globbing and
    /// redirection are all the shell's, and a path with a space in it is a
    /// different command depending on which shell reads it. A program and an
    /// argument array have none of that, so the commands the agent runs on its
    /// own behalf (`cargo check`, `cargo test`, `git diff`) go through here.
    Program {
        program: String,
        args: Vec<String>,
        /// Variables to set for this command only, on top of the inherited
        /// environment. A command that needs one is a command that needs it
        /// for itself, not a reason to change the session's environment.
        env: Vec<(String, String)>,
    },
}

impl Spec {
    /// The command line, as a person would write it.
    fn display(&self) -> String {
        match self {
            Self::Shell(command) => command.clone(),
            Self::Program { program, args, .. } => {
                let mut line = program.clone();
                for arg in args {
                    line.push(' ');
                    if arg.contains(' ') || arg.is_empty() {
                        line.push_str(&format!("{arg:?}"));
                    } else {
                        line.push_str(arg);
                    }
                }
                line
            }
        }
    }
}

/// Runs commands under a deadline, in a fixed directory, with bounded output.
#[derive(Debug, Clone)]
pub struct ProcessRunner {
    cwd: PathBuf,
    timeout: Duration,
    max_output: usize,
}

impl ProcessRunner {
    /// A runner that starts commands in `cwd`, kills them after `timeout`,
    /// and keeps at most `max_output` bytes of each stream.
    pub fn new(cwd: impl Into<PathBuf>, timeout: Duration, max_output: usize) -> Self {
        Self {
            cwd: cwd.into(),
            timeout,
            max_output,
        }
    }

    /// The directory commands start in.
    pub fn cwd(&self) -> &std::path::Path {
        &self.cwd
    }

    /// How long a command may run.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// The most of each stream that is kept.
    pub fn max_output(&self) -> usize {
        self.max_output
    }

    /// Run `command` through the platform shell.
    ///
    /// The escape hatch: the shell is what makes a pipeline, a redirect or a
    /// glob possible, and the agent needs those. It is also what makes the
    /// command line ambiguous, so it is the form that has to be approved as
    /// written rather than assembled from parts.
    pub async fn run_shell(&self, command: &str) -> Result<CommandOutput> {
        self.run(Spec::Shell(command.to_string()), None).await
    }

    /// Run `command` through the platform shell, stopping if the run is
    /// cancelled.
    pub async fn run_shell_until(
        &self,
        command: &str,
        cancel: &CancelToken,
    ) -> Result<CommandOutput> {
        self.run(Spec::Shell(command.to_string()), Some(cancel))
            .await
    }

    /// Run `program` with `args`, with no shell in between.
    ///
    /// The arguments are passed as an array, so a path containing a space, a
    /// quote or a `&` is one argument rather than several, and nothing in the
    /// argument is interpreted by a shell.
    pub async fn run_program(&self, program: &str, args: &[&str]) -> Result<CommandOutput> {
        self.run(program_spec(program, args), None).await
    }

    /// Run `program` with `args`, stopping if the run is cancelled.
    pub async fn run_program_until(
        &self,
        program: &str,
        args: &[&str],
        cancel: &CancelToken,
    ) -> Result<CommandOutput> {
        self.run(program_spec(program, args), Some(cancel)).await
    }

    /// Run `program` with `args` and `env`, with no shell in between.
    ///
    /// The variables are set for this command only. A command that needs one
    /// needs it for itself; changing the session's environment to suit it
    /// would leak into every later command.
    pub async fn run_program_with_env(
        &self,
        program: &str,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<CommandOutput> {
        self.run(program_spec_env(program, args, env), None).await
    }

    /// Run `program` with `args` and `env`, stopping if the run is cancelled.
    pub async fn run_program_until_with_env(
        &self,
        program: &str,
        args: &[&str],
        env: &[(&str, &str)],
        cancel: &CancelToken,
    ) -> Result<CommandOutput> {
        self.run(program_spec_env(program, args, env), Some(cancel))
            .await
    }

    /// The one place a child is spawned, waited for, and killed.
    async fn run(&self, spec: Spec, cancel: Option<&CancelToken>) -> Result<CommandOutput> {
        let started = Instant::now();
        let mut child = self.spawn(&spec)?;
        let pid = child.id();

        // The thing that makes killing the command kill what it spawned. It
        // is built before the pipes are taken so that a failure to attach is
        // still a command that runs.
        let guard = ProcessGuard::attach(&child, pid);

        let stdout_buf = Arc::new(Mutex::new(Bounded::new(self.max_output)));
        let stderr_buf = Arc::new(Mutex::new(Bounded::new(self.max_output)));

        // Both pipes are drained concurrently, on their own tasks: reading one
        // to EOF while the other fills would deadlock a command that writes to
        // both, and a task is what lets the buffers be read after a kill
        // instead of being lost with the future.
        let mut readers = Vec::new();
        if let Some(pipe) = child.stdout.take() {
            readers.push(tokio::spawn(drain_into(pipe, stdout_buf.clone())));
        }
        if let Some(pipe) = child.stderr.take() {
            readers.push(tokio::spawn(drain_into(pipe, stderr_buf.clone())));
        }

        let outcome = tokio::select! {
            status = child.wait() => Outcome::Exited(status),
            _ = tokio::time::sleep(self.timeout) => Outcome::TimedOut,
            _ = wait_for_cancel(cancel) => Outcome::Cancelled,
        };

        let termination = match outcome {
            Outcome::Exited(Ok(status)) => match status.code() {
                Some(code) => Termination::Exited(code),
                None => Termination::Signalled,
            },
            Outcome::Exited(Err(source)) => {
                guard.kill();
                return Err(Error::Io {
                    operation: "waiting for",
                    path: PathBuf::from(spec.display()),
                    source,
                });
            }
            Outcome::TimedOut => {
                guard.kill();
                Termination::TimedOut
            }
            Outcome::Cancelled => {
                guard.kill();
                Termination::Cancelled
            }
        };

        // The child is dead, so its pipes close and the readers finish. Wait
        // for them, but not forever: a grandchild that inherited the pipe and
        // that the kill did not reach would otherwise hang the session.
        let mut read_error = None;
        for mut reader in readers {
            match tokio::time::timeout(DRAIN_GRACE, &mut reader).await {
                Ok(Ok(Some(error))) => {
                    if read_error.is_none() {
                        read_error = Some(error.to_string());
                    }
                }
                Ok(Ok(None)) | Ok(Err(_)) => {}
                Err(_) => reader.abort(),
            }
        }

        let (stdout, stdout_dropped) = take(&stdout_buf);
        let (stderr, stderr_dropped) = take(&stderr_buf);

        Ok(CommandOutput {
            command: spec.display(),
            termination,
            stdout: String::from_utf8_lossy(&stdout).into_owned(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
            dropped_bytes: stdout_dropped + stderr_dropped,
            duration: started.elapsed(),
            read_error,
        })
    }

    /// Spawn the command, with its stdin closed and its own process group.
    fn spawn(&self, spec: &Spec) -> Result<tokio::process::Child> {
        let mut cmd = match spec {
            #[cfg(windows)]
            Spec::Shell(command) => {
                let mut c = tokio::process::Command::new("cmd");
                // `raw_arg`, not `arg`. `cmd.exe` parses the raw command line
                // itself and does not use the MSVC quoting rules `Command::arg`
                // applies, so `arg` mangles anything containing a quote:
                // `echo "a b"` arrives as `echo \"a b\"` and prints the
                // backslashes. `raw_arg` appends the text verbatim, which is
                // what `cmd /C` expects.
                c.raw_arg("/C");
                c.raw_arg(command);
                c
            }
            #[cfg(not(windows))]
            Spec::Shell(command) => {
                let mut c = tokio::process::Command::new("sh");
                c.arg("-c").arg(command);
                c
            }
            Spec::Program { program, args, env } => {
                let mut c = tokio::process::Command::new(program);
                c.args(args);
                for (key, value) in env {
                    c.env(key, value);
                }
                c
            }
        };

        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            // A command that reads stdin would block on a terminal the agent
            // does not own, and would take the user's keystrokes with it. A
            // closed stdin makes the read fail immediately instead.
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true)
            .current_dir(&self.cwd);

        // A session of its own, so the child leads its own process group and
        // `killpg` reaches the whole tree. A failure here is not fatal: the
        // kill then reaches only the shell, which is what happened before
        // this existed.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // SAFETY: `setsid` is async-signal-safe and touches no memory the
            // parent shares; it is the documented use of `pre_exec`.
            unsafe {
                cmd.pre_exec(|| {
                    libc::setsid();
                    Ok(())
                });
            }
        }

        cmd.spawn().map_err(|source| Error::Io {
            operation: "spawning",
            path: PathBuf::from(spec.display()),
            source,
        })
    }
}

/// A [`Spec::Program`] from a program name and a slice of arguments.
fn program_spec(program: &str, args: &[&str]) -> Spec {
    program_spec_env(program, args, &[])
}

/// A [`Spec::Program`] with variables set for this command only.
fn program_spec_env(program: &str, args: &[&str], env: &[(&str, &str)]) -> Spec {
    Spec::Program {
        program: program.to_string(),
        args: args.iter().map(|arg| arg.to_string()).collect(),
        env: env
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
    }
}

/// A program and its arguments as a person would write the command line.
///
/// Public so a caller can show the user the exact command before it runs,
/// which is what the approval prompt needs: the preview has to be the command
/// that will be executed, not a paraphrase of it. The quoting is the same
/// rule the display uses, so what is previewed and what is recorded are the
/// same string.
pub fn program_line(program: &str, args: &[&str]) -> String {
    program_spec(program, args).display()
}

/// How the wait ended.
enum Outcome {
    /// The process exited, or waiting for it failed.
    Exited(std::io::Result<std::process::ExitStatus>),
    /// The deadline passed.
    TimedOut,
    /// The run was cancelled.
    Cancelled,
}

/// A future that resolves when the run is cancelled, or never when there is
/// no token to watch.
async fn wait_for_cancel(cancel: Option<&CancelToken>) {
    match cancel {
        Some(token) => token.notified().await,
        None => std::future::pending().await,
    }
}

/// Read `pipe` to EOF into `buf`, reporting a read failure.
///
/// The error is returned rather than swallowed: a pipe that fails to read is a
/// real failure, and treating it as end-of-stream is what made a command whose
/// output could not be read look like a command that printed nothing.
async fn drain_into<R>(mut pipe: R, buf: Arc<Mutex<Bounded>>) -> Option<std::io::Error>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    use tokio::io::AsyncReadExt;

    let mut chunk = [0u8; 16 * 1024];
    loop {
        match pipe.read(&mut chunk).await {
            Ok(0) => return None,
            Ok(n) => {
                let mut guard = buf.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.push(&chunk[..n]);
            }
            Err(error) => return Some(error),
        }
    }
}

/// The kept bytes of `buf`, and how many were dropped.
fn take(buf: &Arc<Mutex<Bounded>>) -> (Vec<u8>, usize) {
    let guard = buf.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.finish()
}

// ------------------------------------------------------- killing the tree

/// The thing that makes killing the command kill what it spawned.
///
/// On Windows that is a Job Object; on Unix it is the process group the child
/// leads. Both are attached at spawn and used at most once.
#[cfg(windows)]
struct ProcessGuard {
    job: Option<job::Job>,
    pid: Option<u32>,
}

#[cfg(windows)]
impl ProcessGuard {
    /// Put `child` in a job object, falling back to `taskkill` if it cannot be.
    ///
    /// A job object is the only reliable way to kill a whole tree on Windows:
    /// `taskkill /T` enumerates the tree and *then* kills it, so a process
    /// spawned in between survives. If the job cannot be created, or the child
    /// cannot be assigned to it (a nested job on an old Windows, a restricted
    /// token), the fallback is used rather than failing the command: a command
    /// that runs is better than a command that cannot.
    fn attach(child: &tokio::process::Child, pid: Option<u32>) -> Self {
        let job = job::Job::new().ok().filter(|job| job.assign(child).is_ok());
        Self { job, pid }
    }

    /// Kill the command and everything it spawned.
    fn kill(&self) {
        if let Some(job) = &self.job {
            job.terminate();
            return;
        }
        kill_tree_fallback(self.pid);
    }
}

#[cfg(unix)]
struct ProcessGuard {
    pid: Option<u32>,
}

#[cfg(unix)]
impl ProcessGuard {
    /// Nothing to attach: `setsid` in `spawn` already made the process group.
    fn attach(_child: &tokio::process::Child, pid: Option<u32>) -> Self {
        Self { pid }
    }

    /// Kill the command and everything it spawned.
    fn kill(&self) {
        if let Some(pid) = self.pid {
            // The child leads its own process group, so the group id is the
            // child's pid and the signal reaches every descendant that did not
            // create a group of its own.
            // SAFETY: `killpg` takes a pid and a signal number and touches no
            // memory. A pid that has already been reaped makes it fail with
            // `ESRCH`, which is ignored.
            unsafe {
                libc::killpg(pid as libc::pid_t, libc::SIGKILL);
            }
        }
    }
}

/// Kill `pid` and its descendants with the platform's own tool.
///
/// The fallback for a platform where the job object could not be attached.
#[cfg(windows)]
fn kill_tree_fallback(pid: Option<u32>) {
    let Some(pid) = pid else {
        return;
    };
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// A Job Object that kills every process in it when it is closed.
///
/// This is the crate's only `unsafe`, and it is here because there is no safe
/// way to do it: neither the standard library nor `tokio` can put a child in a
/// job object, and a job object is the only thing on Windows that takes a
/// command's whole process tree down with it. Every block below carries a
/// `SAFETY` comment naming the invariant it relies on, and the module is
/// otherwise ordinary safe code.
#[cfg(windows)]
#[allow(unsafe_code)]
mod job {
    use std::io;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    /// An owned job object handle.
    pub struct Job(HANDLE);

    // The handle is owned by this struct and only ever used through `&self`,
    // so moving it between threads is safe. `HANDLE` is a raw pointer, which
    // is why the compiler cannot see that on its own.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        /// A job object that kills its members when the last handle closes.
        pub fn new() -> io::Result<Self> {
            // SAFETY: both arguments are documented as optional and null is
            // the documented "no name, no attributes" value.
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return Err(io::Error::last_os_error());
            }
            let job = Self(handle);

            // SAFETY: `info` is a plain-old-data struct that is fully
            // initialised by `zeroed` and then given a valid flag, and the
            // size passed matches the type.
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    std::ptr::addr_of!(info).cast(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(job)
        }

        /// Put `child` in the job.
        pub fn assign(&self, child: &tokio::process::Child) -> io::Result<()> {
            // `raw_handle` is `None` once the child has been reaped, which
            // cannot happen here: the guard is attached immediately after the
            // spawn and before the wait. A `None` is reported rather than
            // unwrapped, so a race would fall back to `taskkill` instead of
            // panicking.
            let Some(handle) = child.raw_handle() else {
                return Err(io::Error::other("the child has no process handle"));
            };
            // SAFETY: the handle is live for as long as `self`, and the child
            // handle is live for as long as the borrow.
            let ok = unsafe { AssignProcessToJobObject(self.0, handle as HANDLE) };
            if ok == 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        /// Kill every process in the job.
        pub fn terminate(&self) {
            // SAFETY: the handle is live for as long as `self`. The exit code
            // is arbitrary; nothing reads it.
            unsafe {
                TerminateJobObject(self.0, 1);
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: the handle was created by `CreateJobObjectW`, is owned
            // by this struct, and is closed exactly once, here.
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runner() -> ProcessRunner {
        ProcessRunner::new(std::env::temp_dir(), Duration::from_secs(30), 1024 * 1024)
    }

    #[tokio::test]
    async fn a_successful_command_reports_its_output_and_a_zero_exit() {
        let out = runner()
            .run_shell("echo hello")
            .await
            .expect("it should run");
        assert_eq!(out.stdout.trim(), "hello");
        assert_eq!(out.termination, Termination::Exited(0));
        assert!(out.succeeded());
        assert_eq!(out.dropped_bytes, 0);
        assert!(out.read_error.is_none());
    }

    #[tokio::test]
    async fn a_failing_command_reports_its_exit_code_alongside_its_output() {
        // The old result reported the status only when the output was empty,
        // so a failing command that printed anything looked like a success.
        let out = runner()
            .run_shell("echo before-failing & exit 3")
            .await
            .expect("a nonzero exit is not a spawn failure");
        assert_eq!(out.termination, Termination::Exited(3));
        assert!(!out.succeeded());
        let rendered = out.render(1024);
        assert!(
            rendered.contains("before-failing"),
            "the output must survive: {rendered}"
        );
        assert!(
            rendered.contains("exit code 3"),
            "the status must be reported even when there is output: {rendered}"
        );
    }

    #[tokio::test]
    async fn a_command_that_outlives_its_deadline_is_killed_and_keeps_its_output() {
        let quick = ProcessRunner::new(
            std::env::temp_dir(),
            Duration::from_millis(400),
            1024 * 1024,
        );
        let out = quick
            .run_shell("echo started & ping -n 30 127.0.0.1")
            .await
            .expect("a timeout is a result, not a spawn failure");
        assert_eq!(out.termination, Termination::TimedOut);
        assert!(!out.succeeded());
        assert!(
            out.stdout.contains("started"),
            "the output produced before the kill must survive: {:?}",
            out.stdout
        );
        assert!(out.render(1024).contains("outlived its deadline"));
    }

    #[tokio::test]
    async fn a_cancelled_command_is_killed_and_says_so() {
        let cancel = CancelToken::new();
        let token = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            token.stop();
        });
        let out = runner()
            .run_shell_until("ping -n 30 127.0.0.1", &cancel)
            .await
            .expect("a cancellation is a result, not a spawn failure");
        assert_eq!(out.termination, Termination::Cancelled);
        assert!(out.render(1024).contains("cancelled"));
    }

    #[tokio::test]
    async fn output_past_the_bound_keeps_both_ends_and_counts_what_went() {
        let small = ProcessRunner::new(std::env::temp_dir(), Duration::from_secs(30), 64);
        let out = small
            .run_shell("echo FIRST-MARKER & for /L %i in (1,1,200) do @echo 0123456789012345678901234567890123456789 & echo LAST-MARKER")
            .await
            .expect("it should finish");
        assert!(out.dropped_bytes > 0, "the excess should be reported");
        assert!(
            out.stdout.contains("FIRST-MARKER"),
            "the head must be kept: {:?}",
            out.stdout
        );
        assert!(
            out.stdout.contains("LAST-MARKER"),
            "the tail must be kept, or a compiler's diagnostics are lost: {:?}",
            out.stdout
        );
        assert!(out.stdout.contains("bytes dropped"), "{:?}", out.stdout);
        assert!(out.render(64).contains("bytes of output were dropped"));
    }

    #[tokio::test]
    async fn a_program_that_does_not_exist_is_a_spawn_error() {
        let err = runner()
            .run_program("definitely-not-a-real-program-xyz", &[])
            .await
            .expect_err("a missing program is a spawn failure");
        assert!(err.to_string().contains("spawning"), "{err}");
    }

    #[tokio::test]
    async fn a_command_that_reads_stdin_does_not_wait_for_one() {
        // stdin is null, so a read returns end-of-stream at once. With an
        // inherited terminal the command would block on a keypress the agent
        // never sends, and the run would hang until the deadline.
        let quick = ProcessRunner::new(std::env::temp_dir(), Duration::from_secs(5), 1024 * 1024);
        let out = quick
            .run_shell("set /p x= & echo done")
            .await
            .expect("it should finish");
        assert!(
            out.duration < Duration::from_secs(4),
            "it should not have waited for input, took {:?}",
            out.duration
        );
    }

    #[test]
    fn a_bounded_buffer_keeps_both_ends() {
        let mut buf = Bounded::new(10);
        buf.push(b"0123456789ABCDEFGHIJ");
        let (kept, dropped) = buf.finish();
        let text = String::from_utf8_lossy(&kept);
        assert!(text.starts_with("01234"), "{text:?}");
        assert!(text.ends_with("FGHIJ"), "{text:?}");
        assert_eq!(dropped, 10);
    }

    #[test]
    fn a_bounded_buffer_that_never_fills_drops_nothing() {
        let mut buf = Bounded::new(100);
        buf.push(b"short");
        let (kept, dropped) = buf.finish();
        assert_eq!(kept, b"short");
        assert_eq!(dropped, 0);
    }

    #[test]
    fn a_termination_names_itself() {
        assert_eq!(Termination::Exited(0).exit_code(), Some(0));
        assert_eq!(Termination::TimedOut.exit_code(), None);
        assert!(Termination::Cancelled.describe().contains("cancelled"));
    }
}
