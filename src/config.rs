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

/// The speed a terrace wall has to survive being hit at, m/s.
///
/// Not `max_linear_speed`, which is a divergence clamp at 60 m/s rather than a
/// speed anything reaches. Measured honest gaits in this project run at one to
/// two metres a second; three leaves room for a faster one without demanding
/// walls so wide they stop being walls.
const WALL_CROSSING_SPEED: Real = 3.0;

/// How many integration steps a body must take to cross a wall.
///
/// Below about four the contact solver meets the wall as one huge penetration
/// rather than a surface; below one, the body tunnels straight through.
const MIN_WALL_STEPS: Real = 4.0;

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
        BrainLayout::new_with(
            self.body.max_parts,
            self.brain.hidden,
            self.joints_can_break(),
            self.simulation.steer,
        )
    }

    /// Whether this experiment lets joints wear out and limbs detach.
    #[inline]
    pub fn joints_can_break(&self) -> bool {
        self.body.joint_endurance > 0.0
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

    /// Refuse terraces whose walls are too thin for the timestep to resolve.
    ///
    /// A cliff in a height field is only a cliff if a body meets it over
    /// several integration steps. Cross it in one and the solver sees a single
    /// enormous penetration and responds accordingly; cross it in less than one
    /// and the body tunnels through as though it were not there. Neither is
    /// terrain, and both are the kind of thing evolution finds and lives on.
    fn validate_terrace_walls(&self) -> Result<(), ConfigError> {
        if self.environment.terrain != Terrain::Fractal || self.environment.terrain_step <= 0.0 {
            return Ok(());
        }
        // Built with an arbitrary seed: wall width is a property of the band
        // structure, not of which landscape the seed picks out.
        let field = crate::physics::FractalField {
            seed: 1,
            amplitude: self.environment.terrain_amplitude,
            wavelength: self.environment.terrain_wavelength,
            octaves: self.environment.terrain_octaves,
            lacunarity: self.environment.terrain_lacunarity,
            gain: self.environment.terrain_gain,
            warp: self.environment.terrain_warp,
            detail_amplitude: self.environment.terrain_detail_amplitude,
            detail_wavelength: self.environment.terrain_detail_wavelength,
            detail_octaves: self.environment.terrain_detail_octaves,
            modulation: self.environment.terrain_modulation,
            modulation_wavelength: self.environment.terrain_modulation_wavelength,
            step: self.environment.terrain_step,
            riser: self.environment.terrain_riser,
            terrace_mask: self.environment.terrain_terrace_mask,
            ..Default::default()
        };
        let width = field.riser_width();
        let per_step = WALL_CROSSING_SPEED * self.simulation.timestep;
        if width < MIN_WALL_STEPS * per_step {
            return Err(ConfigError::Invalid(format!(
                "environment.terrain_riser is too small for this timestep: the terrace \
                 walls would be {:.0} mm wide, which a body at {WALL_CROSSING_SPEED} m/s \
                 crosses in {:.1} steps. Raise terrain_riser or terrain_step, lower \
                 terrain_amplitude, or lower simulation.timestep, until the walls are at \
                 least {:.0} mm.",
                width * 1000.0,
                width / per_step,
                MIN_WALL_STEPS * per_step * 1000.0,
            )));
        }
        Ok(())
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
        if self.simulation.trials < 1 {
            bad("simulation.trials must be at least 1")?;
        }
        rate(self.simulation.start_jitter, "simulation.start_jitter")?;
        if self.environment.terrain_amplitude < 0.0 {
            bad("environment.terrain_amplitude must not be negative")?;
        }
        if self.environment.terrain_wavelength <= 0.0 {
            bad("environment.terrain_wavelength must be positive")?;
        }
        if self.environment.terrain_octaves < 1
            || self.environment.terrain_octaves > crate::physics::world::MAX_TERRAIN_OCTAVES
        {
            bad("environment.terrain_octaves must be within [1, 8]")?;
        }
        // Below one, an "octave" would be *coarser* than the one before it, and
        // `terrain_wavelength` would stop describing the largest feature.
        if self.environment.terrain_lacunarity < 1.0 {
            bad("environment.terrain_lacunarity must be at least 1")?;
        }
        // At a gain of one every octave contributes its full amplitude and the
        // field is dominated by its finest, which is noise rather than terrain.
        if !(0.0..=1.0).contains(&self.environment.terrain_gain) {
            bad("environment.terrain_gain must be within [0, 1]")?;
        }
        if self.environment.terrain_warp < 0.0 {
            bad("environment.terrain_warp must not be negative")?;
        }
        if self.environment.terrain_detail_amplitude < 0.0 {
            bad("environment.terrain_detail_amplitude must not be negative")?;
        }
        if self.environment.terrain_detail_wavelength <= 0.0 {
            bad("environment.terrain_detail_wavelength must be positive")?;
        }
        if self.environment.terrain_detail_octaves < 1
            || self.environment.terrain_detail_octaves > crate::physics::world::MAX_TERRAIN_OCTAVES
        {
            bad("environment.terrain_detail_octaves must be within [1, 8]")?;
        }
        rate(self.environment.terrain_modulation, "environment.terrain_modulation")?;
        if self.environment.terrain_modulation_wavelength <= 0.0 {
            bad("environment.terrain_modulation_wavelength must be positive")?;
        }
        if self.environment.terrain_step < 0.0 {
            bad("environment.terrain_step must not be negative (0 is smooth ground)")?;
        }
        if !(0.0..=1.0).contains(&self.environment.terrain_riser)
            || (self.environment.terrain_step > 0.0 && self.environment.terrain_riser <= 0.0)
        {
            bad("environment.terrain_riser must be within (0, 1] when terracing is on")?;
        }
        self.validate_terrace_walls()?;
        if self.body.tendon_frequency < 0.0 || self.body.tendon_damping < 0.0 {
            bad("body.tendon_frequency and body.tendon_damping must not be negative")?;
        }
        // Beyond this the explicit spring stops being stable within one step.
        if self.body.tendon_frequency * self.simulation.timestep > 0.5 {
            bad("body.tendon_frequency is too high for this timestep")?;
        }
        if self.body.muscle_stress < 0.0 {
            bad("body.muscle_stress must not be negative (0 disables the cap)")?;
        }
        if self.body.joint_endurance < 0.0 {
            bad("body.joint_endurance must not be negative (0 disables joint damage)")?;
        }
        if !(0.0..=1.0).contains(&self.body.min_drive) {
            bad("body.min_drive must be within [0, 1]")?;
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
        rate(self.mutation.caution_rate, "mutation.caution_rate")?;
        rate(self.mutation.pair_rate, "mutation.pair_rate")?;
        rate(self.mutation.repeat_rate, "mutation.repeat_rate")?;
        rate(self.body.pair_probability, "body.pair_probability")?;
        if self.body.max_repeat < 1 {
            bad("body.max_repeat must be at least 1")?;
        }
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
    /// Probability per genome of perturbing the caution trait, and the size of
    /// that perturbation. Only consulted when `body.joint_endurance` is positive.
    pub caution_rate: Real,
    pub caution_sigma: Real,
    /// Probability per part of flipping whether it is a mirrored pair, or which
    /// way its halves are driven. Only consulted when `body.pair_probability`
    /// is positive.
    pub pair_rate: Real,
    /// Probability per part of redrawing its segment count. Only consulted when
    /// `body.max_repeat` exceeds one.
    pub repeat_rate: Real,
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
            caution_rate: 0.08,
            caution_sigma: 0.12,
            pair_rate: 0.04,
            repeat_rate: 0.03,
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
    /// How much overwork a joint survives, in radians of undelivered rotation.
    ///
    /// A motorised joint takes damage only while its motor is saturated — the
    /// controller is asking for a speed the joint's genetic `motor_torque`
    /// cannot deliver — and the damage is the rotation it fell short by. When
    /// the total reaches this, the joint fails and the limb detaches.
    ///
    /// `0` disables joint damage entirely, which is the default: with it off no
    /// randomness is spent on the caution gene and the controller keeps its
    /// original input count, so an experiment reproduces exactly what it did
    /// before joints could break.
    pub joint_endurance: Real,
    /// Floor on how far the caution gene may throttle motor demand, so a maximally
    /// cautious organism is still able to move.
    pub min_drive: Real,
    /// Muscle strength per unit of joint cross-section. `0` disables the cap and
    /// leaves `motor_torque` as a free gene, which is how every experiment
    /// before this behaved.
    ///
    /// In an animal, the force a muscle can produce scales with its
    /// cross-sectional area, and the torque it exerts with that force times a
    /// moment arm that scales with the limb's width. So the ceiling here goes as
    /// `stress * area^1.5`, and a limb cannot be stronger than its own girth
    /// allows. Without it, `motor_torque` is drawn independently of size and a
    /// matchstick can be as strong as a thigh — which is a large part of why
    /// evolved bodies here look nothing like animals.
    pub muscle_stress: Real,
    /// Natural frequency of every hinge's passive spring, rad/s — a tendon.
    /// Zero leaves joints purely servo-driven, as before.
    ///
    /// A frequency rather than a stiffness so that the spring means the same
    /// thing on a thigh and on a toe. Tendon elasticity is most of why animal
    /// running and hopping are efficient: energy stored on landing comes back on
    /// push-off instead of being paid for again by the muscle.
    pub tendon_frequency: Real,
    /// Damping ratio of that spring. 1 is critically damped.
    pub tendon_damping: Real,
    /// Probability that a newly drawn part is a mirrored pair. `0` disables
    /// bilateral symmetry entirely and spends no randomness on it.
    pub pair_probability: Real,
    /// Longest chain of repeated segments a part may become. `1` disables
    /// segmentation and spends no randomness on it.
    pub max_repeat: u8,
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
            joint_endurance: 0.0,
            min_drive: 0.2,
            muscle_stress: 0.0,
            tendon_frequency: 0.0,
            tendon_damping: 0.5,
            pair_probability: 0.0,
            max_repeat: 1,
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
    /// How many times each organism is evaluated. `1` is a single trial, which
    /// is what every experiment did before this.
    ///
    /// One trial from one pose rewards a stunt as readily as a gait: a single
    /// well-timed lunge scores like walking, and a strategy that works exactly
    /// once cannot be told from one that works. Repeating the trial from varied
    /// starts is the cheapest pressure there is toward behaviour that is
    /// actually repeatable.
    pub trials: usize,
    /// How much the start pose varies between trials, `0` to `1`. Zero means
    /// every trial is identical, which makes repeating them pointless.
    pub start_jitter: Real,
    /// How trials combine into one score.
    pub aggregate: Aggregate,
    /// Whether each trial commands a direction of travel, given to the
    /// controller as an input and scored by `objective = "heading"`.
    pub steer: bool,
    /// Widest angle, radians, that a commanded heading may stray from +X.
    pub steer_spread: Real,
}

/// How an organism's trials combine into the number it is selected on.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Aggregate {
    /// Average. Rewards being good on balance.
    #[default]
    Mean,
    /// The worst trial. Rewards having no bad day at all, which is a much
    /// stronger demand and the one that most favours a robust gait.
    Worst,
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
            trials: 1,
            start_jitter: 0.0,
            aggregate: Aggregate::Mean,
            steer: false,
            steer_spread: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Terrain {
    Flat,
    /// Rolling ground. See [`crate::physics::TerrainModel::Rough`].
    Rough,
    /// Seeded fractal landscape. See [`crate::physics::TerrainModel::Fractal`].
    Fractal,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct EnvironmentCfg {
    pub terrain: Terrain,
    /// Whether an organism's own parts collide with each other.
    ///
    /// Off by default, which is how every experiment before this behaved and the
    /// usual choice in this class of work. Turning it on is what stops a body
    /// being a cloud of overlapping blocks: limbs have to be somewhere the torso
    /// is not, which is the most basic thing that makes an animal an animal.
    pub self_collision: bool,
    /// Scale of the ground's relief, metres. Only consulted when the terrain is
    /// not flat.
    ///
    /// Peak-to-trough is about `2.5 * terrain_amplitude` for both `rough` and
    /// `fractal`. Steepness is set by the *ratio* of amplitude to wavelength,
    /// not by amplitude alone: doubling one and doubling the other leaves the
    /// slope distribution where it was.
    pub terrain_amplitude: Real,
    /// Distance between crests, metres. For `fractal` this is the size of the
    /// *largest* feature; each further octave is `terrain_lacunarity` times
    /// finer.
    pub terrain_wavelength: Real,
    /// Which fractal landscape to generate. `0` derives one from
    /// `experiment.seed`, so two experiments with different seeds get different
    /// ground; any other value names a specific landscape, which is what to use
    /// when comparing two experiments on identical terrain. Only consulted when
    /// `terrain = "fractal"`.
    pub terrain_seed: u64,
    /// How many octaves of noise are summed. One is a single smooth scale; four
    /// spans a factor of eight in feature size, which is about where ground
    /// starts reading as landscape rather than as a pattern.
    pub terrain_octaves: u32,
    /// Frequency step between octaves. Two is the conventional choice — each
    /// octave half the size of the last.
    pub terrain_lacunarity: Real,
    /// Amplitude step between octaves. Below 0.5 the fine detail vanishes;
    /// above it the ground gets rougher at every scale at once.
    pub terrain_gain: Real,
    /// Domain warp strength, in units of `terrain_wavelength`. Zero is plain
    /// fractional Brownian motion.
    ///
    /// Warping bends the field into ridges and basins rather than blobs. It
    /// buys the tail of the slope distribution — at amplitude 0.25 and
    /// wavelength 6 the steepest slope anywhere goes from 19 degrees to 29 —
    /// and it does make the ground more heterogeneous: the spread of mean slope
    /// across 12 m tiles roughly doubles, from 0.04 of its mean to 0.11.
    ///
    /// An earlier version of this comment claimed the opposite, on the strength
    /// of measuring the spread of *relief* per tile rather than of slope. A
    /// tile on the flank of a large hill has enormous relief and can still be
    /// billiard-smooth, so relief answers a different question. What warping
    /// cannot do is produce cliffs; that is `terrain_step`.
    pub terrain_warp: Real,
    /// Amplitude of a second, finer band of noise laid over the landscape,
    /// metres. Zero — the default — leaves the field exactly as it was before
    /// this band existed.
    ///
    /// The landscape band sets how big the hills are; this one sets what the
    /// ground under an organism's feet is like, and they want different
    /// wavelengths. One band cannot do both: fractional Brownian motion has a
    /// single steepness, set by amplitude over wavelength, and it applies it at
    /// every scale at once.
    pub terrain_detail_amplitude: Real,
    pub terrain_detail_wavelength: Real,
    pub terrain_detail_octaves: u32,
    /// How strongly a slow field varies the detail band's amplitude, in
    /// `[0, 1]`. Zero is uniform detail everywhere.
    ///
    /// Warping the domain also varies the ground's character, but only by
    /// rearranging one stationary field; scaling a band's amplitude by a
    /// second, slower field is the direct way to say "calm here, savage
    /// there". Measured as the spread of mean slope across 12 m tiles, the
    /// landscape band alone sits at 0.09 of its mean, this raises it to 0.12,
    /// and with `terrain_terrace_mask` it reaches 0.23.
    pub terrain_modulation: Real,
    /// Size of the calm and savage regions, metres.
    pub terrain_modulation_wavelength: Real,
    /// Terrace height, metres. Zero — the default — is a smooth field.
    ///
    /// Quantising height to terraces is the only thing here that produces a
    /// genuinely sheer face. Scaling the noise up does not: ten metres of
    /// relief still tops out near 48 degrees, and the median slope climbs with
    /// the maximum, which is uniformly steep ground rather than occasional
    /// cliffs. A terrace is flat for most of its span and climbs through the
    /// rest, so the difficulty sits in a small fraction of the area and the
    /// rest stays crossable — measured at 91% of the plane under 40 degrees
    /// with a 99th-percentile slope of 82.
    pub terrain_step: Real,
    /// Fraction of a terrace spent climbing. The riser is steeper than the
    /// underlying slope by exactly `1 / terrain_riser`.
    ///
    /// Its floor is physics, not taste: a body at 3 m/s covers 25 mm per step,
    /// and a wall it crosses in one step is a wall the solver meets as a single
    /// enormous penetration. `validate` refuses a combination that makes the
    /// walls too thin for the timestep.
    pub terrain_riser: Real,
    /// Terrace only where `terrain_modulation` says the ground is savage,
    /// blending back into untouched hills elsewhere. Concentrates the cliffs
    /// rather than tiling the world with them, at the cost of shorter walls.
    pub terrain_terrace_mask: bool,
    /// Whether each trial slides and turns the landscape underneath the
    /// organism.
    ///
    /// Without this every organism in every trial of the whole experiment meets
    /// the same surface, which is memorisable in principle — an organism can be
    /// selected for a gait that suits one particular hill. With it, coping with
    /// ground in general is the only thing that survives.
    pub terrain_per_trial: bool,
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
            self_collision: false,
            terrain_amplitude: 0.06,
            terrain_wavelength: 1.5,
            terrain_seed: 0,
            terrain_octaves: 4,
            terrain_lacunarity: 2.0,
            terrain_gain: 0.5,
            terrain_warp: 0.3,
            terrain_detail_amplitude: 0.0,
            terrain_detail_wavelength: 3.0,
            terrain_detail_octaves: 4,
            terrain_modulation: 0.0,
            terrain_modulation_wavelength: 35.0,
            terrain_step: 0.0,
            terrain_riser: 0.12,
            terrain_terrace_mask: false,
            terrain_per_trial: true,
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
    /// Displacement along the direction the organism was told to go.
    ///
    /// With `simulation.steer` on, that direction changes between trials, so an
    /// organism cannot succeed by committing to one heading and hoping. It has
    /// to be steerable, which is a far stronger demand than being fast — and it
    /// is what forces a controllable body rather than a one-shot launcher.
    Heading,
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
    /// Added per second with no attached part touching the ground.
    ///
    /// This is what makes hopping beat sliding. Zero by default: without it an
    /// organism has no reason ever to leave the ground, and the cheapest way to
    /// travel is to stay on it.
    pub air_bonus: Real,
    /// Added per metre the centre of mass rises above where it started.
    ///
    /// Pairs with `air_bonus`: hang time alone rewards a long low skim, and
    /// height alone rewards a rear-up that never leaves the ground. Together
    /// they ask for a jump.
    pub height_bonus: Real,
}

impl Default for FitnessCfg {
    fn default() -> Self {
        FitnessCfg {
            objective: Objective::Distance,
            energy_penalty: 0.0,
            upright_bonus: 0.0,
            air_bonus: 0.0,
            height_bonus: 0.0,
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
    // As with shapes: folded in only when the feature is enabled, so an
    // experiment that cannot break joints keeps the digest it always had.
    if cfg.body.muscle_stress > 0.0 {
        f.tag(b"muscle");
        f.real(cfg.body.muscle_stress);
    }
    if cfg.body.pair_probability > 0.0 || cfg.body.max_repeat > 1 {
        f.tag(b"bodyplan");
        f.real(cfg.body.pair_probability);
        f.u32(cfg.body.max_repeat as u32);
        f.real(cfg.mutation.pair_rate);
        f.real(cfg.mutation.repeat_rate);
    }
    if cfg.body.tendon_frequency > 0.0 {
        f.tag(b"tendon");
        f.real(cfg.body.tendon_frequency);
        f.real(cfg.body.tendon_damping);
    }
    if cfg.joints_can_break() {
        f.tag(b"joint_health");
        f.real(cfg.body.joint_endurance);
        f.real(cfg.body.min_drive);
        f.real(cfg.mutation.caution_rate);
        f.real(cfg.mutation.caution_sigma);
    }
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

    // Folded in only when the experiment repeats trials, so a single-trial
    // experiment keeps the digest it always had.
    if cfg.simulation.trials > 1 || cfg.simulation.start_jitter > 0.0 {
        f.tag(b"trials");
        f.usize(cfg.simulation.trials);
        f.real(cfg.simulation.start_jitter);
        f.u8(match cfg.simulation.aggregate {
            Aggregate::Mean => 0,
            Aggregate::Worst => 1,
        });
    }
    if cfg.simulation.steer {
        f.tag(b"steer");
        f.real(cfg.simulation.steer_spread);
    }

    f.tag(b"environment");
    f.u8(match cfg.environment.terrain {
        Terrain::Flat => 0,
        Terrain::Rough => 1,
        Terrain::Fractal => 2,
    });
    // Folded in only for the terrain that uses them, so a flat experiment keeps
    // the digest it had before rolling ground existed.
    if cfg.environment.terrain == Terrain::Rough || cfg.environment.terrain == Terrain::Fractal {
        f.real(cfg.environment.terrain_amplitude);
        f.real(cfg.environment.terrain_wavelength);
    }
    // And the fractal knobs only for the fractal, so a `rough` experiment keeps
    // the digest it had before this terrain existed. `experiment.seed` is
    // already part of the digest, so a derived terrain seed needs no extra
    // fold — but an explicit one is not otherwise represented anywhere.
    if cfg.environment.terrain == Terrain::Fractal {
        f.tag(b"fractal");
        f.u64(cfg.environment.terrain_seed);
        f.u32(cfg.environment.terrain_octaves);
        f.real(cfg.environment.terrain_lacunarity);
        f.real(cfg.environment.terrain_gain);
        f.real(cfg.environment.terrain_warp);
        f.bool(cfg.environment.terrain_per_trial);
    }
    // Each later band is folded in only when it is switched on, so a fractal
    // experiment that predates a band keeps the digest it had — the same rule
    // shapes, tendons and breakable joints already follow.
    if cfg.environment.terrain == Terrain::Fractal && cfg.environment.terrain_detail_amplitude > 0.0
    {
        f.tag(b"detail");
        f.real(cfg.environment.terrain_detail_amplitude);
        f.real(cfg.environment.terrain_detail_wavelength);
        f.u32(cfg.environment.terrain_detail_octaves);
    }
    if cfg.environment.terrain == Terrain::Fractal && cfg.environment.terrain_modulation > 0.0 {
        f.tag(b"modulation");
        f.real(cfg.environment.terrain_modulation);
        f.real(cfg.environment.terrain_modulation_wavelength);
    }
    if cfg.environment.terrain == Terrain::Fractal && cfg.environment.terrain_step > 0.0 {
        f.tag(b"terrace");
        f.real(cfg.environment.terrain_step);
        f.real(cfg.environment.terrain_riser);
        f.bool(cfg.environment.terrain_terrace_mask);
    }
    f.real(cfg.environment.gravity);
    f.real(cfg.environment.friction);
    f.real(cfg.environment.restitution);
    f.real(cfg.environment.linear_damping);
    f.real(cfg.environment.angular_damping);
    if cfg.environment.self_collision {
        f.tag(b"selfcollide");
    }

    f.tag(b"fitness");
    f.u8(match cfg.fitness.objective {
        Objective::Distance => 0,
        Objective::DistanceX => 1,
        Objective::Speed => 2,
        Objective::Heading => 3,
    });
    f.real(cfg.fitness.energy_penalty);
    // Guarded like every other opt-in term: an experiment that does not reward
    // leaving the ground keeps the digest it had before jumping was scorable.
    if cfg.fitness.air_bonus != 0.0 || cfg.fitness.height_bonus != 0.0 {
        f.tag(b"jump");
        f.real(cfg.fitness.air_bonus);
        f.real(cfg.fitness.height_bonus);
    }
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
        for name in [
            "first-walkers.toml",
            "directed-walkers.toml",
            "animals.toml",
            "fractal-animals.toml",
            "brittle-walkers.toml",
            "jumpers.toml",
            "shaped-walkers.toml",
        ] {
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

    /// The compatibility rule this repository lives by: a feature that is off
    /// must leave the digest exactly where it was, or every run directory
    /// started before it stops being resumable.
    #[test]
    fn the_fractal_knobs_are_invisible_until_the_fractal_terrain_is_chosen() {
        for terrain in [Terrain::Flat, Terrain::Rough] {
            let mut a = Config::default();
            a.environment.terrain = terrain;
            for tweak in [
                |c: &mut Config| c.environment.terrain_seed = 12345,
                |c: &mut Config| c.environment.terrain_octaves = 7,
                |c: &mut Config| c.environment.terrain_lacunarity = 2.5,
                |c: &mut Config| c.environment.terrain_gain = 0.75,
                |c: &mut Config| c.environment.terrain_warp = 0.0,
                |c: &mut Config| c.environment.terrain_per_trial = false,
            ] {
                let mut b = a.clone();
                tweak(&mut b);
                assert_eq!(a.digest(), b.digest(), "{terrain:?} noticed a fractal knob");
            }
        }

        // And on the fractal terrain every one of them counts.
        let mut a = Config::default();
        a.environment.terrain = Terrain::Fractal;
        for tweak in [
            |c: &mut Config| c.environment.terrain_seed = 12345,
            |c: &mut Config| c.environment.terrain_octaves = 7,
            |c: &mut Config| c.environment.terrain_lacunarity = 2.5,
            |c: &mut Config| c.environment.terrain_gain = 0.75,
            |c: &mut Config| c.environment.terrain_warp = 0.0,
            |c: &mut Config| c.environment.terrain_per_trial = false,
            |c: &mut Config| c.environment.terrain_amplitude = 0.3,
            |c: &mut Config| c.environment.terrain_wavelength = 9.0,
        ] {
            let mut b = a.clone();
            tweak(&mut b);
            assert_ne!(a.digest(), b.digest());
        }

        // The three terrains are three different experiments even at identical
        // amplitude and wavelength.
        let mut rough = Config::default();
        rough.environment.terrain = Terrain::Rough;
        assert_ne!(rough.digest(), a.digest());
        assert_ne!(Config::default().digest(), rough.digest());
    }

    /// Terracing is the one setting whose floor is physics rather than taste: a
    /// wall thinner than a few integration steps is not a cliff, it is a
    /// tunnelling bug waiting for evolution to find it.
    #[test]
    fn validation_measures_terrace_wall_width() {
        let base = "[environment]\nterrain = \"fractal\"\nterrain_amplitude = 3.0\n\
                    terrain_wavelength = 25.0\nterrain_octaves = 5\n\
                    terrain_detail_amplitude = 0.35\nterrain_step = 0.8\n";
        // As shipped: 150 mm walls, six steps at 3 m/s.
        Config::from_toml_str(&format!("{base}terrain_riser = 0.12\n")).unwrap();

        // A quarter of the riser is a quarter of the wall, and below what the
        // contact solver can meet as a surface.
        let err = Config::from_toml_str(&format!("{base}terrain_riser = 0.03\n"))
            .expect_err("a 40 mm wall should be refused");
        assert!(err.to_string().contains("terrain_riser"), "{err}");
        assert!(err.to_string().contains("mm wide"), "{err}");

        // The same walls become fine at a finer timestep, because what the rule
        // is really about is how far a body moves between contacts.
        Config::from_toml_str(&format!(
            "{base}terrain_riser = 0.03\n\n[simulation]\ntimestep = 0.001\n"
        ))
        .unwrap();

        // And with terracing off there are no walls to be too thin.
        Config::from_toml_str("[environment]\nterrain = \"fractal\"\nterrain_riser = 0.001\n")
            .unwrap();
    }

    #[test]
    fn validation_catches_bad_fractal_terrain() {
        for (field, value) in [
            ("terrain_octaves", "0"),
            ("terrain_octaves", "9"),
            ("terrain_lacunarity", "0.5"),
            ("terrain_gain", "1.5"),
            ("terrain_gain", "-0.1"),
            ("terrain_warp", "-1.0"),
        ] {
            let toml = format!("[environment]\nterrain = \"fractal\"\n{field} = {value}\n");
            let err = Config::from_toml_str(&toml).expect_err("{field} = {value} should fail");
            assert!(err.to_string().contains(field), "{field} = {value}: {err}");
        }
        // And the defaults are inside every one of those bounds.
        let mut ok = Config::default();
        ok.environment.terrain = Terrain::Fractal;
        ok.validate().unwrap();
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
