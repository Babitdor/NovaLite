//! Typed errors for the library boundary.
//!
//! The library returns [`Error`]; the binary wraps it in `anyhow` with the
//! context only the executable has (which directory, which command line).
//! Tools still hand rig a `String`-backed error, because rig's `Tool` trait
//! requires `std::error::Error` and the model reads the message: the typed
//! error is converted at that boundary, and its `Display` is what the model
//! sees. That conversion is the only place a message is built by hand, so a
//! refusal reads the same wherever it is raised.

use std::path::{Path, PathBuf};

/// The prefix [`Error::Denied`] renders with.
///
/// A refusal reaches the transcript as the tool's result text, and the
/// transcript has to tell a refusal apart from a failure: a call the user
/// declined did not fail, and painting it with the same cross as a broken
/// build tells the user their agent is broken when it is doing what they told
/// it to. The text is the only thing that survives the trip from the tool to
/// the screen, so the prefix is a constant rather than a literal in two
/// places, and a test pins it to the `Display` it has to match.
pub const DENIED_PREFIX: &str = "denied: ";

/// The text [`Error::Cancelled`] renders with.
///
/// Read the same way as [`DENIED_PREFIX`]: a cancelled call is not a failed
/// one, and the transcript has to be able to tell them apart from the result
/// text alone, because that is all that reaches it.
pub const CANCELLED_TEXT: &str = "cancelled";

/// Everything the library can fail at.
///
/// Each variant carries the path or value that caused it, so a caller can
/// branch on the kind of failure rather than on the text of a message.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A path resolved outside the workspace root.
    #[error("'{path}' is outside the workspace ({}). This session can only read and write inside that directory.", root.display())]
    OutsideWorkspace { path: String, root: PathBuf },

    /// A path is written in a form this session cannot check.
    ///
    /// Separate from [`Error::OutsideWorkspace`] because the two need
    /// different answers. "Outside the workspace" tells the model to stay
    /// inside; "a verbatim path is not normalised" tells it to write the path
    /// plainly. Collapsing them into one message would leave the model
    /// retrying a path that can never work.
    #[error("'{path}' cannot be used: {reason}")]
    PathForm { path: String, reason: String },

    /// The workspace root could not be established.
    #[error("cannot use '{}' as the workspace: {source}", path.display())]
    WorkspaceRoot {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// A filesystem operation failed.
    #[error("{operation} '{}': {source}", path.display())]
    Io {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The caller asked for something the tool cannot do.
    #[error("{0}")]
    Invalid(String),

    /// The user refused the action.
    #[error("denied: {0}")]
    Denied(String),

    /// The file changed between the preview and the commit.
    ///
    /// The approval was bound to the bytes that were shown, so a file that is
    /// no longer those bytes cannot be written under that decision: the diff
    /// the user agreed to is not the diff that would be applied.
    #[error("'{path}' changed after it was read; read it again and re-apply the change")]
    Stale { path: String },

    /// The file changed since this session last read it.
    ///
    /// Distinct from [`Error::Stale`], which is about the window between the
    /// preview and the commit. This one is about the window between the read
    /// and the edit: the model is working from text it saw earlier, and the
    /// file is no longer that text. The search string may still match, which
    /// is exactly why the check exists: an edit that lands in a file the model
    /// has not seen is a silent wrong edit, not a failed one.
    #[error("'{path}' has changed since you read it; read it again before editing")]
    StaleRead { path: String },

    /// A patch could not be applied to the file as it stands.
    ///
    /// The message is the conflict's own, because it is written for the model:
    /// it names the block that failed and what to do about it.
    #[error("{0}")]
    Conflict(#[from] crate::patch::Conflict),

    /// The run was cancelled while the action was in flight.
    #[error("cancelled")]
    Cancelled,

    /// A request cannot be made to fit the model's context window.
    ///
    /// Raised before the request is sent rather than after the provider
    /// rejects it, and it names both figures so the user can act: the window
    /// is a model parameter, and a prompt that does not fit on its own is not
    /// something dropping history can fix.
    #[error(
        "the request needs about {needed} tokens but the model's window leaves {budget}; \
         the prompt alone is {prompt} tokens. Start a new conversation, or raise \
         LLM_NUM_CTX if the model's window is larger than that."
    )]
    ContextTooLarge {
        /// The request's estimated size, in tokens.
        needed: usize,
        /// The tokens the request may occupy.
        budget: usize,
        /// The prompt's own estimated size, in tokens.
        prompt: usize,
    },
}

impl Error {
    /// An [`Error::Io`] naming the operation and the path it failed on.
    pub fn io(operation: &'static str, path: impl AsRef<Path>, source: std::io::Error) -> Self {
        Self::Io {
            operation,
            path: path.as_ref().to_path_buf(),
            source,
        }
    }

    /// An [`Error::OutsideWorkspace`] for `path` under `root`.
    pub fn outside(path: impl Into<String>, root: impl AsRef<Path>) -> Self {
        Self::OutsideWorkspace {
            path: path.into(),
            root: root.as_ref().to_path_buf(),
        }
    }

    /// An [`Error::Invalid`] for a caller-supplied value the tool refuses.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    /// An [`Error::PathForm`] for a path written in a form that cannot be checked.
    pub fn path_form(path: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::PathForm {
            path: path.into(),
            reason: reason.into(),
        }
    }

    /// Whether this is a refusal the model should not retry unchanged.
    ///
    /// A denial, a cancellation and a stale edit are decisions, not failures:
    /// the model is told what happened and must not re-issue the same call.
    ///
    /// A patch conflict is deliberately *not* one of these. The model can fix
    /// it: a search string that matched twice needs more context, and one that
    /// matched nothing needs to be copied from the file again. Refusing to let
    /// it try would turn a recoverable mistake into a dead end.
    pub fn is_refusal(&self) -> bool {
        matches!(
            self,
            Self::Denied(_) | Self::Cancelled | Self::Stale { .. } | Self::StaleRead { .. }
        )
    }
}

/// The library's result type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_outside_path_names_the_root_and_the_rule() {
        let err = Error::outside("../secrets", Path::new("C:/work/proj"));
        let text = err.to_string();
        assert!(text.contains("outside the workspace"), "unexpected: {text}");
        assert!(text.contains("C:/work/proj"), "unexpected: {text}");
        assert!(text.contains("../secrets"), "unexpected: {text}");
    }

    #[test]
    fn an_io_error_names_the_operation_and_the_path() {
        let err = Error::io(
            "reading",
            Path::new("C:/work/proj/src/main.rs"),
            std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"),
        );
        let text = err.to_string();
        assert!(text.starts_with("reading '"), "unexpected: {text}");
        assert!(text.contains("main.rs"), "unexpected: {text}");
    }

    #[test]
    fn a_denial_and_a_cancellation_are_refusals() {
        assert!(Error::Denied("no".into()).is_refusal());
        assert!(Error::Cancelled.is_refusal());
        assert!(!Error::invalid("bad argument").is_refusal());
    }

    #[test]
    fn a_denial_renders_with_the_prefix_the_transcript_looks_for() {
        // The transcript reads the result text to decide whether to paint a
        // refusal or a failure, and the text is built here. If the two ever
        // drift, a denial is drawn as a broken tool and nothing else fails.
        let text = Error::Denied("not this one".to_string()).to_string();
        assert!(text.starts_with(DENIED_PREFIX), "unexpected: {text}");
        assert!(text.contains("not this one"), "unexpected: {text}");
    }

    #[test]
    fn a_cancellation_renders_with_the_text_the_transcript_looks_for() {
        assert_eq!(Error::Cancelled.to_string(), CANCELLED_TEXT);
    }
}
