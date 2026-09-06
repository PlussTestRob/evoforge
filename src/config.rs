//! Experiment configuration.
//!
//! Everything that defines an experiment lives in a single TOML file. Every
//! field has a default so that a minimal config is legal, but unknown fields are
//! rejected: a typo that silently reverts a setting to its default would quietly
//! invalidate a comparison between runs.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::brain::BrainLayout;
use crate::math::Real;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub experiment: ExperimentCfg,
    pub evolution: EvolutionCfg,
    pub mutation: MutationParams,
    pub body: BodyLimits,
    pub brain: BrainCfg,
    pub simulation: SimulationCfg,
    pub environment: EnvironmentCfg,
    pub fitness: FitnessCfg,
    pub recording: RecordingCfg,
    pub checkpoint: CheckpointCfg,
}

impl Config {
    pub fn from_toml_str(text: &str) -> Result<Config, ConfigError> {
        let cfg: Config = toml::from_str(text).map_err(|e| ConfigError::Parse(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn load(path: &Path) -> Result<Config, ConfigError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::Io(path.display().to_string(), e.to_string()))?;
        Config::from_toml_str(&text)
    }

    pub fn to_toml_string(&self) -> String {
        toml::to_string_pretty(self).expect("config is always serialisable")
    }

    /// The controller shape implied by this configuration.
    pub fn brain_layout(&self) -> BrainLayout {
        BrainLayout::new(self.body.max_parts, self.brain.hidden)
    }

    /// Number of physics steps in one evaluation, including the settling period.
    pub fn total_steps(&self) -> u32 {
        ((self.simulation.settle_time + self.simulation.duration) / self.simulation.timestep)
            .ceil() as u32
    }

    /// Number of physics steps to run before the controller is enabled and
    /// fitness measurement begins.
    pub fn settle_steps(&self) -> u32 {
        (self.simulation.settle_time / self.simulation.timestep).ceil() as u32
    }

    /// Fingerprint of the whole configuration, recorded in a run's manifest.
    pub fn digest(&self) -> u64 {
        fnv1a(self.to_toml_string().as_bytes())
    }

    /// Fingerprint of only those settings that change what evolution *does*.
    ///
    /// Where the results go, how long the run lasts, and what gets recorded do
    /// not affect the trajectory, so they are excluded. That distinction is what
    /// lets a finished run be extended — resume with a larger `generations` and
    /// the checkpoint still matches — while still refusing to resume a run whose
    /// physics, mutation rates or seed have changed underneath it.
    pub fn evolution_digest(&self) -> u64 {
        let mut c = self.clone();
        c.experiment.name = String::new();
        c.experiment.output_dir = PathBuf::new();
        c.evolution.generations = 0;
        c.recording = RecordingCfg::default();
        c.checkpoint = CheckpointCfg::default();
        fnv1a(c.to_toml_string().as_bytes())
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: &str| -> Result<(), ConfigError> { Err(ConfigError::Invalid(m.into())) };
        if self.evolution.population_size < 2 {
            bad("evolution.population_size must be at least 2")?;
        }
        if self.evolution.elite_count >= self.evolution.population_size {
            bad("evolution.elite_count must be smaller than population_size")?;
        }
        if self.evolution.tournament_size < 1 {
            bad("evolution.tournament_size must be at least 1")?;
        }
        if self.body.min_parts < 1 || self.body.max_parts < self.body.min_parts {
            bad("body part limits must satisfy 1 <= min_parts <= max_parts")?;
        }
        if self.body.max_parts > 255 {
            bad("body.max_parts must be at most 255 (slots are u8)")?;
        }
        if self.body.min_half_extent <= 0.0 || self.body.max_half_extent < self.body.min_half_extent
        {
            bad("body half-extent limits must satisfy 0 < min <= max")?;
        }
        if self.brain.hidden == 0 {
            bad("brain.hidden must be at least 1")?;
        }
        if self.simulation.timestep <= 0.0 {
            bad("simulation.timestep must be positive")?;
        }
        if self.simulation.duration <= 0.0 {
            bad("simulation.duration must be positive")?;
        }
        if self.simulation.control_hz <= 0.0 {
            bad("simulation.control_hz must be positive")?;
        }
        if self.simulation.solver_iterations == 0 {
            bad("simulation.solver_iterations must be at least 1")?;
        }
        if !(0.0..=2.0).contains(&self.environment.friction) {
            bad("environment.friction must be within [0, 2]")?;
        }
        if self.recording.record_hz <= 0.0 {
            bad("recording.record_hz must be positive")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ExperimentCfg {
    /// Human-readable name; also used to name the run directory.
    pub name: String,
    /// Root seed. Every other random stream in the experiment is derived from
    /// this plus a stable identity, never from wall-clock time.
    pub seed: u64,
    /// Directory under which run directories are created.
    pub output_dir: PathBuf,
}

impl Default for ExperimentCfg {
    fn default() -> Self {
        ExperimentCfg {
            name: "unnamed".into(),
            seed: 1,
            output_dir: PathBuf::from("runs"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EvolutionCfg {
    pub population_size: usize,
    pub generations: u32,
    /// Top individuals copied unchanged into the next generation.
    pub elite_count: usize,
    /// Tournament size; larger means stronger selection pressure.
    pub tournament_size: usize,
    /// Probability that an offspring is produced by crossover rather than
    /// cloning a single parent.
    pub crossover_rate: Real,
    /// Fraction of each generation replaced by freshly generated random
    /// genomes. A small amount of immigration is cheap insurance against the
    /// population collapsing onto one lineage.
    pub immigrant_rate: Real,
}

impl Default for EvolutionCfg {
    fn default() -> Self {
        EvolutionCfg {
            population_size: 100,
            generations: 100,
            elite_count: 2,
            tournament_size: 4,
            crossover_rate: 0.7,
            immigrant_rate: 0.02,
        }
    }
}

/// Per-gene mutation rates and step sizes.
///
/// Rates are per-gene probabilities, not per-genome, so their effect does not
/// change as organisms grow more parts.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MutationParams {
    pub weight_rate: Real,
    pub weight_sigma: Real,
    /// Probability that a mutated weight is redrawn from scratch instead of
    /// perturbed, which lets the search escape a deep local basin.
    pub weight_reset_rate: Real,
    pub size_rate: Real,
    pub size_sigma: Real,
    pub attach_rate: Real,
    pub attach_sigma: Real,
    pub joint_limit_rate: Real,
    pub joint_limit_sigma: Real,
    pub joint_kind_rate: Real,
    pub joint_axis_rate: Real,
    pub motor_rate: Real,
    pub motor_sigma: Real,
    /// Probability per genome of appending one new part.
    pub add_part_rate: Real,
    /// Probability per genome of deleting one leaf part.
    pub remove_part_rate: Real,
}

impl Default for MutationParams {
    fn default() -> Self {
        MutationParams {
            weight_rate: 0.08,
            weight_sigma: 0.25,
            weight_reset_rate: 0.05,
            size_rate: 0.05,
            size_sigma: 0.05,
            attach_rate: 0.05,
            attach_sigma: 0.15,
            joint_limit_rate: 0.05,
            joint_limit_sigma: 0.2,
            joint_kind_rate: 0.02,
            joint_axis_rate: 0.03,
            motor_rate: 0.05,
            motor_sigma: 0.15,
            add_part_rate: 0.06,
            remove_part_rate: 0.05,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BodyLimits {
    pub min_parts: usize,
    pub max_parts: usize,
    pub min_half_extent: Real,
    pub max_half_extent: Real,
    /// kg/m^3. Deliberately far below water: heavy blocks need implausible
    /// torques to move and make early evolution uninteresting.
    pub density: Real,
    /// Hinge limits are symmetric (+/- limit) in radians.
    pub min_joint_limit: Real,
    pub max_joint_limit: Real,
    pub max_motor_speed: Real,
    pub max_motor_torque: Real,
    /// Probability that a newly generated joint is a hinge rather than fixed.
    pub hinge_probability: Real,
}

impl Default for BodyLimits {
    fn default() -> Self {
        BodyLimits {
            min_parts: 2,
            max_parts: 6,
            min_half_extent: 0.08,
            max_half_extent: 0.35,
            density: 250.0,
            min_joint_limit: 0.3,
            max_joint_limit: 1.4,
            max_motor_speed: 6.0,
            max_motor_torque: 120.0,
            hinge_probability: 0.85,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct BrainCfg {
    pub hidden: usize,
    /// Standard deviation used when drawing fresh weights.
    pub init_sigma: Real,
    /// Weights are clamped to +/- this. Unbounded weights saturate `tanh` and
    /// turn the controller into a constant, which evolution finds embarrassingly
    /// quickly.
    pub weight_limit: Real,
}

impl Default for BrainCfg {
    fn default() -> Self {
        BrainCfg {
            hidden: 10,
            init_sigma: 0.8,
            weight_limit: 8.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct SimulationCfg {
    /// Physics step, seconds.
    pub timestep: Real,
    /// Measured simulation time per evaluation, seconds.
    pub duration: Real,
    /// Controller update rate. Decoupled from the physics rate because the
    /// network only needs to act on the timescale the body can respond to, and
    /// evaluating it less often is free performance.
    pub control_hz: Real,
    pub solver_iterations: u32,
    /// Time the organism falls and settles before measurement begins, so that
    /// the initial drop does not count as locomotion.
    pub settle_time: Real,
}

impl Default for SimulationCfg {
    fn default() -> Self {
        SimulationCfg {
            timestep: 1.0 / 120.0,
            duration: 8.0,
            control_hz: 20.0,
            solver_iterations: 10,
            settle_time: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Terrain {
    Flat,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EnvironmentCfg {
    pub terrain: Terrain,
    /// Downward acceleration magnitude, m/s^2.
    pub gravity: Real,
    pub friction: Real,
    pub restitution: Real,
    pub linear_damping: Real,
    pub angular_damping: Real,
}

impl Default for EnvironmentCfg {
    fn default() -> Self {
        EnvironmentCfg {
            terrain: Terrain::Flat,
            gravity: 9.81,
            friction: 0.8,
            restitution: 0.0,
            linear_damping: 0.02,
            angular_damping: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Horizontal distance from the starting position, any direction.
    Distance,
    /// Signed displacement along +X. Harder than `distance`: the organism must
    /// commit to a direction rather than fall over impressively.
    DistanceX,
    /// Mean horizontal speed over the measured window.
    Speed,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct FitnessCfg {
    pub objective: Objective,
    /// Subtracted per unit of actuation impulse. Zero by default: energy
    /// pressure before locomotion exists just selects for doing nothing.
    pub energy_penalty: Real,
    /// Added per second spent with the root block upright.
    pub upright_bonus: Real,
}

impl Default for FitnessCfg {
    fn default() -> Self {
        FitnessCfg {
            objective: Objective::Distance,
            energy_penalty: 0.0,
            upright_bonus: 0.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct RecordingCfg {
    /// Record trajectories for the top N organisms of a recorded generation.
    pub top_n: usize,
    /// Additionally record this many uniformly sampled organisms, which keeps
    /// some record of what the unsuccessful majority was doing.
    pub random_samples: usize,
    /// Trajectory sample rate. Independent of the physics rate.
    pub record_hz: Real,
    /// Record only every Nth generation (1 = every generation).
    pub every_generations: u32,
    /// Store the genomes of recorded organisms in `genomes.jsonl`, which is what
    /// makes `evo replay` able to re-simulate them at higher fidelity later.
    pub store_genomes: bool,
}

impl Default for RecordingCfg {
    fn default() -> Self {
        RecordingCfg {
            top_n: 1,
            random_samples: 0,
            record_hz: 30.0,
            every_generations: 10,
            store_genomes: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CheckpointCfg {
    /// Write a full-population checkpoint every N generations (0 disables).
    pub every_generations: u32,
    /// Also write a checkpoint when the run finishes.
    pub on_finish: bool,
}

impl Default for CheckpointCfg {
    fn default() -> Self {
        CheckpointCfg {
            every_generations: 25,
            on_finish: true,
        }
    }
}

#[derive(Debug)]
pub enum ConfigError {
    Io(String, String),
    Parse(String),
    Invalid(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(path, e) => write!(f, "could not read config {path}: {e}"),
            ConfigError::Parse(e) => write!(f, "could not parse config: {e}"),
            ConfigError::Invalid(m) => write!(f, "invalid config: {m}"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// FNV-1a, used for configuration and structure fingerprints.
///
/// Not cryptographic; it only needs to be fast, stable across versions and
/// unlikely to collide by accident.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_is_the_default() {
        let cfg = Config::from_toml_str("").unwrap();
        assert_eq!(cfg, Config::default());
    }

    #[test]
    fn partial_config_overrides_only_named_fields() {
        let cfg = Config::from_toml_str(
            r#"
            [experiment]
            name = "walkers"
            seed = 99

            [evolution]
            population_size = 32
            "#,
        )
        .unwrap();
        assert_eq!(cfg.experiment.name, "walkers");
        assert_eq!(cfg.experiment.seed, 99);
        assert_eq!(cfg.evolution.population_size, 32);
        // Untouched fields keep their defaults.
        assert_eq!(cfg.evolution.generations, EvolutionCfg::default().generations);
        assert_eq!(cfg.brain.hidden, BrainCfg::default().hidden);
    }

    #[test]
    fn unknown_fields_are_rejected() {
        let err = Config::from_toml_str(
            r#"
            [evolution]
            populaton_size = 32
            "#,
        );
        assert!(matches!(err, Err(ConfigError::Parse(_))));
    }

    #[test]
    fn roundtrips_through_toml() {
        let cfg = Config::default();
        let text = cfg.to_toml_string();
        let back = Config::from_toml_str(&text).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn validation_catches_bad_values() {
        let bad = Config::from_toml_str(
            r#"
            [evolution]
            population_size = 4
            elite_count = 4
            "#,
        );
        assert!(matches!(bad, Err(ConfigError::Invalid(_))));

        let bad = Config::from_toml_str(
            r#"
            [body]
            min_parts = 5
            max_parts = 2
            "#,
        );
        assert!(matches!(bad, Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn digest_is_sensitive_to_changes() {
        let a = Config::default();
        let mut b = Config::default();
        b.experiment.seed = 2;
        assert_ne!(a.digest(), b.digest());
        assert_eq!(a.digest(), Config::default().digest());
    }

    #[test]
    fn evolution_digest_ignores_bookkeeping_but_not_dynamics() {
        let a = Config::default();

        for tweak in [
            |c: &mut Config| c.evolution.generations = 9999,
            |c: &mut Config| c.experiment.output_dir = "elsewhere".into(),
            |c: &mut Config| c.experiment.name = "renamed".into(),
            |c: &mut Config| c.recording.top_n = 17,
            |c: &mut Config| c.checkpoint.every_generations = 3,
        ] {
            let mut b = Config::default();
            tweak(&mut b);
            assert_eq!(a.evolution_digest(), b.evolution_digest());
        }

        for tweak in [
            |c: &mut Config| c.experiment.seed = 2,
            |c: &mut Config| c.evolution.tournament_size = 9,
            |c: &mut Config| c.mutation.weight_sigma = 0.9,
            |c: &mut Config| c.simulation.timestep = 0.01,
            |c: &mut Config| c.environment.gravity = 3.7,
        ] {
            let mut b = Config::default();
            tweak(&mut b);
            assert_ne!(a.evolution_digest(), b.evolution_digest());
        }
    }

    #[test]
    fn step_counts_are_consistent() {
        let cfg = Config::default();
        assert_eq!(cfg.settle_steps(), 60);
        assert_eq!(cfg.total_steps(), 1020);
    }
}
