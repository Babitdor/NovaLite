//! WiFi Network Scanner using the Rust Agent
//!
//! This example demonstrates how to use the Rust Agent to scan for WiFi networks.
//! It requires your agent to be approved to run shell commands.

use rust_agent::app::App;
use rust_agent::config::Config;
use rust_agent::event::Event;
use rust_agent::runtime::Runtime;
use rust_agent::trust::{self, TrustStore};
use rust_agent::workspace::Workspace;

fn main() -> anyhow::Result<()> {
    // Initialize logging
    let _log_guard = rust_agent::logging::init_logging()?;

    // Get the workspace root
    let root = Workspace::new(std::env::current_dir()?)?;

    // Create a trust store for approval
    let store = TrustStore::default_location();

    // Create stdin and stdout for the agent
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut out = std::io::stdout();

    // Check if the agent is approved to run shell commands
    if !trust::ensure_trusted(root.root(), &store, &mut reader, &mut out)? {
        eprintln!("not approved; nothing was read or written");
        return Ok(());
    }

    // Create the runtime with auto-approval (since we're using the agent)
    let rt = Runtime::new(Config::from_env(), root);
    rt.set_approver(std::sync::Arc::new(rust_agent::approval::AutoApprover));

    // Create agents
    let agents = rust_agent::agent::create_agents(&rt)?;
    let (mut app, mut rx) = App::new(agents, rt);

    // The runtime handle is separate from the agent's `Runtime`: `Runtime::new`
    // returns the shared services, not a tokio runtime.
    let tokio_rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    // Run the WiFi scanner
    tokio_rt.block_on(async {
        app.send_message("Scan for WiFi networks in the current directory and list them with their SSIDs and BSSIDs".to_string());

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(300);
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(event)) => {
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
                    eprintln!("timed out after 5 minutes");
                    break;
                }
            }
        }
    });

    println!("\n=== activity ===");
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
