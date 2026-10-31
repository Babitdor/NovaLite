//! Drive the real agent against a live provider, without a terminal.
//!
//! `cargo run --example live -- "list the files in src"` builds the same
//! agent the binary builds, streams one prompt, and prints the activity
//! timeline and the final transcript. It is the check the mock tests cannot
//! make: that the provider's real stream decodes into the events the UI
//! expects.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use rust_agent::app::App;
use rust_agent::config::Config;
use rust_agent::event::Event;
use rust_agent::runtime::Runtime;
use rust_agent::trust::{self, TrustStore};
use rust_agent::workspace::Workspace;

fn main() -> anyhow::Result<()> {
    // Same logging setup as the binary, so this run also proves the rig INFO
    // lines land in the log file rather than on the terminal.
    let _log_guard = rust_agent::logging::init_logging()?;

    let prompt = std::env::args().nth(1).unwrap_or_else(|| {
        "List the files in the src directory, then say what each one does in one line.".to_string()
    });

    // The same two gates the binary goes through, in the same order. An
    // example that built the agent directly would be a second, unguarded way
    // into the tools, and the point of the example is that it exercises the
    // path the binary takes.
    let root = Workspace::new(std::env::current_dir()?)?;
    let store = TrustStore::default_location();
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut out = std::io::stdout();
    if !trust::ensure_trusted(root.root(), &store, &mut reader, &mut out)? {
        eprintln!("not approved; nothing was read or written");
        return Ok(());
    }

    let rt = Runtime::new(Config::from_env(), root);
    // The example has no terminal to prompt on, so it approves every action
    // and says so. It is not a bypass: the tools still ask, and the answer is
    // always yes, which is the same path the binary takes in auto mode. An
    // example that reached the filesystem without asking would be a second,
    // unguarded way in, and the point of the example is that it exercises the
    // path the binary takes.
    rt.set_approver(std::sync::Arc::new(rust_agent::approval::AutoApprover));
    eprintln!("note: this example auto-approves every write and command");
    let agents = rust_agent::agent::create_agents(&rt)?;
    let (mut app, mut rx) = App::new(agents, rt);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async {
        app.send_message(prompt.clone());
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(180);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
                    // Every way the run can end, not just the happy one. A run
                    // the session stopped (a repeated call, a repeated
                    // failure, an exhausted correction budget) ends as
                    // `Cancelled`, and waiting for a `Finished` that will never
                    // come would report a timeout for a run that stopped
                    // cleanly.
                    let terminal = matches!(
                        event,
                        Event::Finished(_)
                            | Event::Error(_)
                            | Event::Cancelled
                            | Event::StreamEnded
                            | Event::TaskFailed(_)
                    );
                    app.handle_event(event);
                    if terminal {
                        break;
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    eprintln!("timed out after 180s");
                    break;
                }
            }
        }
    });

    println!("=== activity ===");
    for a in &app.activity {
        println!("{:?} | {} | {}", a.kind, a.title, a.detail);
    }
    println!("\n=== transcript ===");
    for m in &app.messages {
        println!("[{:?}] {}", m.role, m.content);
    }
    println!(
        "\n=== totals ===\ncalls={} in={} out={} status={}",
        app.calls, app.usage_in, app.usage_out, app.status
    );

    Ok(())
}
