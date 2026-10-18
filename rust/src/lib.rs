//! hr-rag: multilingual RAG chatbot + evaluator for the Suryodaya Indic HR multi-hop QA benchmark.
//!
//! Library used by the `hr-rag` command-line tool and by the Jupyter notebook (evcxr kernel).
//! Start with [`session::Session`].
pub mod chunk;
pub mod data;
pub mod eval;
pub mod index;
pub mod ollama;
pub mod rag;
pub mod retrieve;
pub mod session;
pub mod text;

pub use rag::{Config, Prediction, Strategy};
pub use retrieve::Mode;
pub use session::{compare, details, Answer, Report, Session};
