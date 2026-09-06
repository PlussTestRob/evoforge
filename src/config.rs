//! Experiment configuration.
//!
//! Everything that defines an experiment lives in a single TOML file. Every
//! field has a default so that a minimal config is legal, but unknown fields are
//! rejected: a typo that silently reverts a setting to its default would quietly
//! invalidate a comparison between runs.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::brain::BrainLayout;
use crate::genome::ShapeKind;
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
        ((self.simulation.settle_time + self.simulation.duration) / self.simulation.timestep).ceil()
            as u32
    }

    /// Number of physics steps to run before the controller is enabled and
    /// fitness measurement begins.
    pub fn settle_steps(&self) -> u32 {
        (self.simulation.settle_time / self.simulation.timestep).ceil() as u32
    }

    /// Fingerprint of the whole configuration, recorded in a run's manifest.
    pub fn digest(&self) -> u64 {
        fingerprint(self, true)
    }

    /// Fingerprint of only those settings that change what evolution *does*.
    ///
    /// Where the results go, how long the run lasts, and what gets recorded do
    /// not affect the trajectory, so they are excluded. That distinction is what
    /// lets a finished run be extended — resume with a larger `generations` and
    /// the checkpoint still matches — while still refusing to resume a run whose
    /// physics, mutation rates or seed have changed underneath it.
    ///
    /// Built from a versioned, field-by-field byte stream rather than pretty
    /// TOML, so a serializer upgrade or a comment cannot silently invalidate
    /// resume.
    pub fn evolution_digest(&self) -> u64 {
        fingerprint(self, false)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |m: &str| -> Result<(), ConfigError> { Err(ConfigError::Invalid(m.into())) };
        let rate = |v: Real, name: &str| -> Result<(), ConfigError> {
            if (0.0..=1.0).contains(&v) {
                Ok(())
            } else {
                bad(&format!("{name} must be within [0, 1]"))
            }
        };

        if self.evolution.population_size < 2 {
            bad("evolution.population_size must be at least 2")?;
        }
        if self.evolution.elite_count >= self.evolution.population_size {
            bad("evolution.elite_count must be smaller than population_size")?;
        }
        if self.evolution.tournament_size < 1 {
            bad("evolution.tournament_size must be at least 1")?;
        }
        rate(self.evolution.crossover_rate, "evolution.crossover_rate")?;
        rate(self.evolution.immigrant_rate, "evolution.immigrant_rate")?;
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
        if self.body.min_joint_limit <= 0.0
            || self.body.max_joint_limit < self.body.min_joint_limit
            || self.body.max_joint_limit >= crate::math::FRAC_PI_2
        {
            bad("body joint limits must satisfy 0 < min <= max < pi/2 (cosine limit test)")?;
        }
        if self.body.max_motor_speed < 0.0 || self.body.max_motor_torque < 0.0 {
            bad("body motor limits must be non-negative")?;
        }
        rate(self.body.hinge_probability, "body.hinge_probability")?;
        if self.body.shapes.is_empty() {
            bad("body.shapes must list at least one shape")?;
        }
        // A taper wider at the top than the base would put its bounding box
        // somewhere other than where `Shape::bounds` says it is, and every
        // attachment and spawn calculation trusts that bound.
        if !(0.0..=1.0).contains(&self.body.taper_top_scale) {
            bad("body.taper_top_scale must be within [0, 1]")?;
        }
        if self.body.density <= 0.0 {
            bad("body.density must be positive")?;
        }
        if self.brain.hidden == 0 {
            bad("brain.hidden must be at least 1")?;
        }
        if self.brain.init_sigma < 0.0 || self.brain.weight_limit <= 0.0 {
            bad("brain.init_sigma must be non-negative and weight_limit positive")?;
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
        if self.simulation.settle_time < 0.0 {
            bad("simulation.settle_time must be non-negative")?;
        }
        if !(0.0..=1.0).contains(&self.simulation.baumgarte) {
            bad("simulation.baumgarte must be within [0, 1]")?;
        }
        if self.simulation.slop < 0.0 {
            bad("simulation.slop must be non-negative")?;
        }
        if self.simulation.max_correction_speed <= 0.0
            || self.simulation.max_linear_speed <= 0.0
            || self.simulation.max_angular_speed <= 0.0
        {
            bad("simulation speed clamps must be positive")?;
        }
        if !(0.0..=2.0).contains(&self.environment.friction) {
            bad("environment.friction must be within [0, 2]")?;
        }
        if self.environment.restitution < 0.0 || self.environment.restitution > 1.0 {
            bad("environment.restitution must be within [0, 1]")?;
        }
        if self.environment.linear_damping < 0.0 || self.environment.angular_damping < 0.0 {
            bad("environment damping must be non-negative")?;
        }
        if self.environment.gravity < 0.0 {
            bad("environment.gravity must be non-negative")?;
        }
        if self.recording.record_hz <= 0.0 {
            bad("recording.record_hz must be positive")?;
        }
        rate(self.mutation.weight_rate, "mutation.weight_rate")?;
        rate(self.mutation.weight_reset_rate, "mutation.weight_reset_rate")?;
        rate(self.mutation.size_rate, "mutation.size_rate")?;
        rate(self.mutation.shape_rate, "mutation.shape_rate")?;
        rate(self.mutation.attach_rate, "mutation.attach_rate")?;
        rate(self.mutation.joint_limit_rate, "mutation.joint_limit_rate")?;
        rate(self.mutation.joint_kind_rate, "mutation.joint_kind_rate")?;
        rate(self.mutation.joint_axis_rate, "mutation.joint_axis_rate")?;
        rate(self.mutation.motor_rate, "mutation.motor_rate")?;
        rate(self.mutation.add_part_rate, "mutation.add_part_rate")?;
        rate(self.mutation.remove_part_rate, "mutation.remove_part_rate")?;
        if self.mutation.weight_sigma < 0.0
            || self.mutation.size_sigma < 0.0
            || self.mutation.attach_sigma < 0.0
            || self.mutation.joint_limit_sigma < 0.0
            || self.mutation.motor_sigma < 0.0
        {
            bad("mutation step sizes must be non-negative")?;
        }
        if self.fitness.energy_penalty < 0.0 || self.fitness.upright_bonus < 0.0 {
            bad("fitness energy_penalty and upright_bonus must be non-negative")?;
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
        ExperimentCfg { name: "unnamed".into(), seed: 1, output_dir: PathBuf::from("runs") }
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
    /// Probability per part of redrawing its shape. Only consulted when
    /// `body.shapes` offers more than one, so a box-only experiment never spends
    /// a draw on it.
    pub shape_rate: Real,
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
            shape_rate: 0.04,
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
    /// Which primitives parts may be carved from, e.g.
    /// `shapes = ["box", "capsule", "sphere"]`.
    ///
    /// A single-entry list means every part is that shape and *no randomness is
    /// spent choosing*, which is what lets a box-only experiment reproduce
    /// results recorded before shapes existed, bit for bit.
    pub shapes: Vec<ShapeKind>,
    /// Cross-section of a taper's far end as a fraction of its base. Fixed per
    /// experiment rather than evolved, because a second size gene would mostly
    /// duplicate what `half_extents` already says.
    pub taper_top_scale: Real,
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
            shapes: vec![ShapeKind::Box],
            taper_top_scale: 0.45,
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
        BrainCfg { hidden: 10, init_sigma: 0.8, weight_limit: 8.0 }
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
    /// the initial drop does not count as locomotion. The controller is held
    /// off during this window.
    pub settle_time: Real,
    /// Fraction of positional error corrected per step (Baumgarte).
    pub baumgarte: Real,
    /// Penetration tolerated before positional correction kicks in.
    pub slop: Real,
    /// Ceiling on Baumgarte-injected velocity.
    pub max_correction_speed: Real,
    /// Hard linear velocity clamp, m/s. Keeps a pathological body finite.
    pub max_linear_speed: Real,
    /// Hard angular velocity clamp, rad/s.
    pub max_angular_speed: Real,
}

impl Default for SimulationCfg {
    fn default() -> Self {
        SimulationCfg {
            timestep: 1.0 / 120.0,
            duration: 8.0,
            control_hz: 20.0,
            solver_iterations: 10,
            settle_time: 0.5,
            baumgarte: 0.2,
            slop: 0.002,
            max_correction_speed: 2.0,
            max_linear_speed: 60.0,
            max_angular_speed: 40.0,
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
        FitnessCfg { objective: Objective::Distance, energy_penalty: 0.0, upright_bonus: 0.0 }
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
        CheckpointCfg { every_generations: 25, on_finish: true }
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

/// Version of the dynamics fingerprint layout. Bump when a field is added,
/// removed or reinterpreted — existing checkpoints will then correctly refuse
/// to resume rather than silently continue under a different hash.
const FINGERPRINT_VERSION: u32 = 1;

/// Canonical, serializer-independent fingerprint.
///
/// New dynamics fields must be appended here. Pretty-printed TOML is not used:
/// field order, comments and crate upgrades must not change the digest.
/// Whether a shape roster asks for anything a pre-shapes build would not have done.
fn uses_shapes(shapes: &[ShapeKind]) -> bool {
    shapes.len() > 1 || (shapes.len() == 1 && shapes[0] != ShapeKind::Box)
}

fn fingerprint(cfg: &Config, include_bookkeeping: bool) -> u64 {
    let mut f = Fingerprint::new();
    f.u32(FINGERPRINT_VERSION);
    f.bool(include_bookkeeping);

    f.tag(b"experiment");
    f.u64(cfg.experiment.seed);
    if include_bookkeeping {
        f.str(&cfg.experiment.name);
        f.str(&cfg.experiment.output_dir.to_string_lossy());
    }

    f.tag(b"evolution");
    f.usize(cfg.evolution.population_size);
    f.usize(cfg.evolution.elite_count);
    f.usize(cfg.evolution.tournament_size);
    f.real(cfg.evolution.crossover_rate);
    f.real(cfg.evolution.immigrant_rate);
    if include_bookkeeping {
        f.u32(cfg.evolution.generations);
    }

    f.tag(b"mutation");
    f.real(cfg.mutation.weight_rate);
    f.real(cfg.mutation.weight_sigma);
    f.real(cfg.mutation.weight_reset_rate);
    f.real(cfg.mutation.size_rate);
    f.real(cfg.mutation.size_sigma);
    f.real(cfg.mutation.attach_rate);
    f.real(cfg.mutation.attach_sigma);
    f.real(cfg.mutation.joint_limit_rate);
    f.real(cfg.mutation.joint_limit_sigma);
    f.real(cfg.mutation.joint_kind_rate);
    f.real(cfg.mutation.joint_axis_rate);
    f.real(cfg.mutation.motor_rate);
    f.real(cfg.mutation.motor_sigma);
    f.real(cfg.mutation.add_part_rate);
    f.real(cfg.mutation.remove_part_rate);

    f.tag(b"body");
    f.usize(cfg.body.min_parts);
    f.usize(cfg.body.max_parts);
    f.real(cfg.body.min_half_extent);
    f.real(cfg.body.max_half_extent);
    f.real(cfg.body.density);
    f.real(cfg.body.min_joint_limit);
    f.real(cfg.body.max_joint_limit);
    f.real(cfg.body.max_motor_speed);
    f.real(cfg.body.max_motor_torque);
    f.real(cfg.body.hinge_probability);
    // Folded in only when the experiment actually uses shapes. A box-only
    // configuration therefore keeps the digest it had before shapes existed,
    // which is what lets a run started before this feature still be resumed.
    if uses_shapes(&cfg.body.shapes) {
        f.tag(b"shapes");
        for kind in &cfg.body.shapes {
            f.u32(*kind as u32);
        }
        f.real(cfg.body.taper_top_scale);
        f.real(cfg.mutation.shape_rate);
    }

    f.tag(b"brain");
    f.usize(cfg.brain.hidden);
    f.real(cfg.brain.init_sigma);
    f.real(cfg.brain.weight_limit);

    f.tag(b"simulation");
    f.real(cfg.simulation.timestep);
    f.real(cfg.simulation.duration);
    f.real(cfg.simulation.control_hz);
    f.u32(cfg.simulation.solver_iterations);
    f.real(cfg.simulation.settle_time);
    f.real(cfg.simulation.baumgarte);
    f.real(cfg.simulation.slop);
    f.real(cfg.simulation.max_correction_speed);
    f.real(cfg.simulation.max_linear_speed);
    f.real(cfg.simulation.max_angular_speed);

    f.tag(b"environment");
    f.u8(match cfg.environment.terrain {
        Terrain::Flat => 0,
    });
    f.real(cfg.environment.gravity);
    f.real(cfg.environment.friction);
    f.real(cfg.environment.restitution);
    f.real(cfg.environment.linear_damping);
    f.real(cfg.environment.angular_damping);

    f.tag(b"fitness");
    f.u8(match cfg.fitness.objective {
        Objective::Distance => 0,
        Objective::DistanceX => 1,
        Objective::Speed => 2,
    });
    f.real(cfg.fitness.energy_penalty);
    f.real(cfg.fitness.upright_bonus);

    if include_bookkeeping {
        f.tag(b"recording");
        f.usize(cfg.recording.top_n);
        f.usize(cfg.recording.random_samples);
        f.real(cfg.recording.record_hz);
        f.u32(cfg.recording.every_generations);
        f.bool(cfg.recording.store_genomes);

        f.tag(b"checkpoint");
        f.u32(cfg.checkpoint.every_generations);
        f.bool(cfg.checkpoint.on_finish);
    }

    f.finish()
}

struct Fingerprint(Vec<u8>);

impl Fingerprint {
    fn new() -> Fingerprint {
        Fingerprint(Vec::with_capacity(512))
    }

    fn tag(&mut self, bytes: &[u8]) {
        self.u32(bytes.len() as u32);
        self.0.extend_from_slice(bytes);
    }

    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }

    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }

    fn real(&mut self, v: Real) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    fn bool(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }

    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }

    fn finish(&self) -> u64 {
        fnv1a(&self.0)
    }
}

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
    fn bundled_experiments_validate() {
        for name in ["first-walkers.toml", "directed-walkers.toml"] {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments").join(name);
            Config::load(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

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
            |c: &mut Config| c.simulation.baumgarte = 0.35,
            |c: &mut Config| c.environment.gravity = 3.7,
            |c: &mut Config| c.fitness.objective = Objective::DistanceX,
        ] {
            let mut b = Config::default();
            tweak(&mut b);
            assert_ne!(a.evolution_digest(), b.evolution_digest());
        }
    }

    #[test]
    fn evolution_digest_is_independent_of_toml_formatting() {
        let compact = Config::from_toml_str("[experiment]\nseed = 99\n").unwrap();
        let padded = Config::from_toml_str(
            "# comment\n\n[experiment]\nseed = 99\n\n[evolution]\ngenerations = 100\n",
        )
        .unwrap();
        assert_eq!(compact.evolution_digest(), padded.evolution_digest());
        assert_eq!(
            compact.digest(),
            Config::from_toml_str("[experiment]\nseed = 99\n").unwrap().digest()
        );
    }

    #[test]
    fn validation_rejects_rates_outside_unit_interval() {
        let err = Config::from_toml_str(
            r#"
            [evolution]
            crossover_rate = 1.5
            "#,
        );
        assert!(matches!(err, Err(ConfigError::Invalid(_))));

        let err = Config::from_toml_str(
            r#"
            [mutation]
            weight_rate = -0.1
            "#,
        );
        assert!(matches!(err, Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn validation_rejects_hinge_limits_outside_the_cosine_range() {
        let err = Config::from_toml_str(
            r#"
            [body]
            min_joint_limit = 0.3
            max_joint_limit = 2.0
            "#,
        );
        assert!(matches!(err, Err(ConfigError::Invalid(_))));
    }

    #[test]
    fn step_counts_are_consistent() {
        let cfg = Config::default();
        assert_eq!(cfg.settle_steps(), 60);
        assert_eq!(cfg.total_steps(), 1020);
    }
}
