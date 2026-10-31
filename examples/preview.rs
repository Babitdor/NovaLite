//! Print the TUI as text, so the layout can be inspected without a terminal.
//!
//! `cargo run --example preview` renders a representative conversation into an
//! offscreen buffer and writes it to stdout. It is the quickest way to check a
//! layout change: no provider, no raw mode, and the output can be diffed.
//!
//! `cargo run --example preview -- empty [width] [height]` renders the
//! empty-conversation state instead, which is where the NOVA-Lite banner
//! lives. `picker` renders the model overlay, `expanded` renders the
//! conversation with every tool block opened, and `folded` renders it with
//! every tool group folded.
// The lint restrictions in `Cargo.toml` are for production code; in a test a
// panic is the assertion. See the note in `src/lib.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use rig::completion::Usage;
use rig::test_utils::{mock_final, MockCompletionModel, MockStreamEvent};

use rust_agent::app::App;
use rust_agent::event::Event;
use rust_agent::render;
use rust_agent::tools;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let empty = args.first().map(|a| a == "empty").unwrap_or(false);
    let picker = args
        .first()
        .map(|a| a == "picker" || a == "picker-live")
        .unwrap_or(false);
    let expanded = args.first().map(|a| a == "expanded").unwrap_or(false);
    let folded = args.first().map(|a| a == "folded").unwrap_or(false);
    let width: u16 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(100);
    let height: u16 = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(30);

    let model = MockCompletionModel::from_stream_turns(vec![
        vec![
            MockStreamEvent::ReasoningDelta {
                id: "r0".to_string(),
                reasoning: "The user wants the file list, so list_dir is the right tool."
                    .to_string(),
            },
            MockStreamEvent::ToolCall {
                id: "t0".to_string(),
                name: "list_dir".to_string(),
                arguments: serde_json::json!({"path": "src"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(
                Usage::new().input_tokens(412).output_tokens(38),
            )),
        ],
        vec![
            MockStreamEvent::ToolCall {
                id: "t1".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "src/app.rs"}),
                call_id: None,
            },
            MockStreamEvent::ToolCall {
                id: "t2".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "src/render.rs"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(
                Usage::new().input_tokens(1_204).output_tokens(41),
            )),
        ],
        vec![
            MockStreamEvent::ToolCall {
                id: "t3".to_string(),
                name: "shell".to_string(),
                arguments: serde_json::json!({"command": "echo hello"}),
                call_id: None,
            },
            MockStreamEvent::FinalResponse(mock_final(
                Usage::new().input_tokens(1_388).output_tokens(29),
            )),
        ],
        vec![
            MockStreamEvent::Text(
                "Here is what `src` holds:\n\n\
                 - `main.rs` builds the agent and enters the terminal\n\
                 - `app.rs` holds the state and the stream translation\n\
                 - `render.rs` paints the frame\n\n\
                 ```rust\n\
                 let rt = Runtime::new(Config::from_env(), workspace);\n\
                 let (app, rx) = App::new(create_agents(&rt)?, rt);\n\
                 ```\n"
                    .to_string(),
            ),
            MockStreamEvent::FinalResponse(mock_final(
                Usage::new().input_tokens(486).output_tokens(96),
            )),
        ],
    ]);

    // A session over the crate root, approving everything: the preview is a
    // layout tool, and a prompt would be a modal in the middle of a render.
    // The tools are still the real ones, so a scripted call goes through the
    // same workspace check and the same process runner the binary uses.
    let rt = rust_agent::runtime::Runtime::new(
        rust_agent::config::Config::defaults()
            .with_approval(rust_agent::config::ApprovalMode::Auto),
        rust_agent::workspace::Workspace::new(
            std::env::current_dir().expect("the process has a working directory"),
        )
        .expect("the crate root is a valid workspace"),
    );

    let agent = rig::agent::AgentBuilder::new(model.clone())
        .tool(tools::ReadFile::new(rt.clone()))
        .tool(tools::ListDir::new(rt.clone()))
        .tool(tools::Shell::new(rt.clone()))
        .build();
    let summariser = rig::agent::AgentBuilder::new(model).build();
    // Kept for the live picker below, which fetches through the same process
    // runner the binary uses rather than a bare `Command::output()`.
    let picker_rt = rt.clone();
    let (mut app, mut rx) = App::new(
        rust_agent::agent::Agents {
            main: agent,
            summariser,
        },
        rt,
    );

    if !empty {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        rt.block_on(async {
            app.send_message("what is in src?".to_string());
            while let Some(event) = rx.recv().await {
                let finished = matches!(event, Event::Finished(_));
                app.handle_event(event);
                if finished {
                    break;
                }
            }
        });
    }

    if picker {
        // The picker's list comes from `ollama list`, which is not available
        // in every environment, so the preview fills it with a fixed set: the
        // point is to inspect the overlay's layout, not the daemon. Pass
        // `picker-live` to exercise the real fetch instead.
        let live = args.first().map(|a| a == "picker-live").unwrap_or(false);
        let (models, error) = if live {
            let fetched = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("a tokio runtime")
                .block_on(rust_agent::agent::list_ollama_models(&picker_rt));
            match fetched {
                Ok(models) => (models, None),
                Err(e) => (Vec::new(), Some(e.to_string())),
            }
        } else {
            (
                vec![
                    "deepseek-v4.1-flash:cloud".to_string(),
                    "gemma4:e2b".to_string(),
                    "glm-5.3-flash:cloud".to_string(),
                    "kimi-k3:cloud".to_string(),
                    "tev1:4b".to_string(),
                ],
                None,
            )
        };
        app.picker = Some(rust_agent::app::ModelPicker {
            selected: models
                .iter()
                .position(|m| *m == app.model_name)
                .unwrap_or(0),
            models,
            loading: false,
            error,
        });
    }

    if expanded {
        // Tool blocks are collapsed by default, so the expanded state is the
        // one worth inspecting: it is where the result body and its guide are
        // painted.
        app.toggle_all_expanded();
    }

    if folded {
        // Groups are open by default, so the folded state is the one worth
        // inspecting: it is where the heading's digest is painted.
        app.toggle_all_groups();
    }

    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|f| render::draw(f, &app)).unwrap();

    let buffer = terminal.backend().buffer();
    let area = *buffer.area();
    for y in 0..area.height {
        let row: String = (0..area.width)
            .map(|x| buffer.cell((x, y)).map(|c| c.symbol()).unwrap_or(" "))
            .collect();
        println!("{}", row.trim_end());
    }
}
