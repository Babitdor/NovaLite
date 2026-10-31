use anyhow::{Context, Result};
use rust_agent::app::App;
use rust_agent::cli::{self, Command};
use rust_agent::config::Config;
use rust_agent::runtime::Runtime;
use rust_agent::trust::{self, TrustStore};
use rust_agent::workspace::Workspace;
use std::io::Write;
use std::panic;
use tokio::sync::mpsc;

fn main() -> Result<()> {
    // The command line is read before anything else, so `--help` and
    // `--version` work without a workspace, a trust prompt, or a log file.
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let (requested, auto_approve) = match cli::parse_args(&args) {
        Ok(Command::Help) => {
            print!("{}", cli::usage());
            return Ok(());
        }
        Ok(Command::Version) => {
            println!("{}", cli::version());
            return Ok(());
        }
        Ok(Command::Run { dir, auto_approve }) => match dir {
            Some(dir) => (dir, auto_approve),
            None => (std::env::current_dir()?, auto_approve),
        },
        // A usage error is not a crash: it is the user asking for something
        // the program does not do, so it gets a one-line message and the
        // conventional exit code for a bad invocation.
        Err(message) => {
            eprintln!("nova-lite: {message}");
            std::process::exit(2);
        }
    };

    // Logs go to a file, not the terminal: the TUI owns the screen, and a
    // stray write lands in the input box. The guard keeps the writer alive.
    let _log_guard = rust_agent::logging::init_logging()?;

    // Restore the terminal before panicking so we don't leave it in raw mode.
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        let _ = rust_agent::tui::leave_terminal();
        default_hook(info);
    }));

    // The directory the agent may touch: the one it was started in, or the one
    // named on the command line. Everything the tools do is resolved against
    // it, so it is fixed before the agent is built.
    let root = Workspace::new(&requested).context("establishing the workspace root")?;

    // First run in a directory: ask before the agent gets access to it. The
    // question is asked before the terminal is taken over, so it is an
    // ordinary line of input rather than a modal the TUI has to draw.
    //
    // Trust is not approval. This asks whether the agent may work in the
    // directory at all, once, and records the answer outside the workspace.
    // Whether it may make a particular change is asked separately, every
    // time, by the approval gate the runtime installs.
    let store = TrustStore::default_location();
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut out = std::io::stdout();
    if !trust::ensure_trusted(root.root(), &store, &mut reader, &mut out)? {
        writeln!(
            out,
            "Not approved. Nothing was read or written. Approvals are recorded in {}.",
            store.path().display()
        )?;
        return Ok(());
    }

    // The configuration is read once, here, and is immutable from this point
    // on: nothing downstream consults the environment again, so the budgets
    // the agent runs under cannot change halfway through a session.
    //
    // The flag wins over the environment: it is the more specific statement,
    // and a user who typed `--auto-approve` should not be overruled by a
    // `NOVA_APPROVAL=prompt` left in a shell profile.
    let mut config = Config::from_env();
    if auto_approve {
        config = config.with_approval(rust_agent::config::ApprovalMode::Auto);
    }
    let rt = Runtime::new(config, root);

    // The approval channel is made before the app, because the approver needs
    // the sender and the app needs the receiver. In auto mode the gate is
    // installed here and the channel is never used: the tools still ask, and
    // the answer is always yes, which is what keeps the boundary in one place
    // rather than branching around it.
    let (approval_tx, approval_rx) = mpsc::unbounded_channel();
    if rt.config().approval.is_auto() {
        rt.set_approver(std::sync::Arc::new(rust_agent::approval::AutoApprover));
    } else {
        rt.set_approver(std::sync::Arc::new(
            rust_agent::approval::InteractiveApprover::new(approval_tx),
        ));
    }

    let agents = rust_agent::agent::create_agents(&rt)?;
    let (mut app, rx) = App::new(agents, rt);
    app.set_approval_channel(approval_rx);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(rust_agent::tui::run_event_loop(&mut app, rx))?;

    Ok(())
}
