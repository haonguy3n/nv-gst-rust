use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid config: field `{field}`: {message}")]
    InvalidConfig {
        field: &'static str,
        message: String,
    },

    #[error("config error: {0}")]
    Config(String),

    /// Two stages disagree on where pixels live or how they are laid out.
    /// The builder refuses at startup instead of inserting a silent copy.
    #[error("memory domain mismatch at stage `{stage}`: expected {expected}, got {got}")]
    DomainMismatch {
        stage: &'static str,
        expected: String,
        got: String,
    },

    #[error("unsupported: {0}")]
    Unsupported(String),

    #[error("platform error: {0}")]
    Platform(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    pub fn invalid(field: &'static str, message: impl Into<String>) -> Self {
        Self::InvalidConfig {
            field,
            message: message.into(),
        }
    }
}
