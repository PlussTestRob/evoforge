//! Rigid-body physics.
//!
//! The public surface is deliberately small — [`World::step`] plus accessors —
//! so that the solver can be replaced wholesale without touching the rest of the
//! simulator. See [`world`] for the method, its limitations, and the intended
//! upgrade path.

pub mod body;
pub mod world;

pub use body::RigidBody;
pub use world::{Joint, TerrainModel, World, WorldParams};
