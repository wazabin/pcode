//! The vocabulary types shared between a p-code *producer* and a p-code
//! *consumer*.
//!
//! A SLEIGH specification describes memory spaces, processor registers and
//! user-defined p-code operations; an IR built from that specification refers
//! to the same things. Both sides need to agree on how those are identified, so
//! the definitions live here rather than in either crate — which lets the
//! decoder and the IR depend on each other only through this vocabulary.
//!
//! These types are serialization-stable: they appear directly in precompiled
//! specification blobs, so their field layout must not change casually.

pub mod register;
pub mod space;

pub use register::{Register, RegisterId, RegisterMutRef, RegisterRef};
pub use space::{SPACE_CONST, Space, SpaceId, SpaceRef, SpaceStore, SpaceType};

use jstd::Identifier;

/// A stable identifier for a user-defined p-code operation.
///
/// These are the `define pcodeop` names in a SLEIGH specification: operations
/// with no p-code semantics, which a consumer must interpret itself.
#[derive(Identifier)]
pub struct PCodeOpId(usize);
