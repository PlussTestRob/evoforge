//! EvoForge: a headless evolutionary artificial-life laboratory.
//!
//! The pipeline is a straight line, and each stage is a separate module:
//!
//! ```text
//! genome  ->  phenotype  ->  physics  ->  sim  ->  fitness  ->  evolution
//! ```
//!
//! Three properties hold everything together and are worth stating up front,
//! because most of the design follows from them:
//!
//! 1. **Evaluation is pure.** [`sim::evaluate`] depends only on a genome and a
//!    configuration. It touches no globals, no clock and no shared state. That is
//!    what makes evaluation parallel across cores today and across machines
//!    later, and what lets any organism be reconstructed from its genome alone.
//!
//! 2. **Randomness is derived, never ambient.** Every stream comes from
//!    [`rng::derive_seed`] applied to the experiment seed and a stable identity,
//!    so results do not depend on execution order.
//!
//! 3. **Nothing renders.** The simulator writes trajectories for a handful of
//!    selected organisms and stops there. Visualisation is a separate program
//!    reading a separate file.

pub mod brain;
pub mod config;
pub mod evolution;
pub mod fitness;
pub mod genome;
pub mod math;
pub mod phenotype;
pub mod physics;
pub mod record;
pub mod rng;
pub mod sim;
pub mod stats;

pub mod bench;
pub mod runner;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
