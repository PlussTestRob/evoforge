//! Evaluating one organism.
//!
//! An evaluation is a pure function of `(genome, config)`: build the world, run
//! it for a fixed number of steps driving the joints from the controller, and
//! reduce the behaviour to [`Metrics`]. Nothing is shared between evaluations,
//! nothing is read from the environment, and nothing depends on wall-clock time.
//!
//! That purity is the whole basis of the performance plan. Evaluations can run on
//! every core, in any order, and later on any machine, and the results are
//! identical. It is also what makes `evo replay` possible: a champion from
//! generation 8421 can be reconstructed from its genome alone.

use serde::{Deserialize, Serialize};

use crate::brain::{self, input, BrainScratch};
use crate::config::{Aggregate, Config};
use crate::fitness::{self, Metrics};
use crate::genome::Genome;
use crate::math::{clamp, triangle, vec3, Real, Vec3};
use crate::phenotype::{self, BodySpec, Phenotype};
use crate::rng::{derive_seed, Rng};

/// Frequency of the controller's clock inputs, Hz.
///
/// Two out-of-phase triangle waves give the network a pacemaker to build a gait
/// around, without having to evolve an oscillator from scratch first. Triangle
/// rather than sine so it stays exact for arbitrarily long simulations.
/// Clearance above the terrain, in metres, before an organism counts as
/// airborne.
///
/// A contact only exists once a point is *below* the ground, so "touching
/// nothing" also describes a body hovering imperceptibly above it. Asked for
/// hang time on that basis, evolution produced organisms that spent half the
/// trial airborne while never rising above the grass. A couple of centimetres
/// of margin is the difference between leaving the ground and skimming it.
const AIRBORNE_CLEARANCE: Real = 0.02;

/// Stream tag separating trial perturbations from every other derived stream.
const TRIAL_STREAM: u64 = 0x5452_4941_4c53_0001;

/// Widest start perturbations at `start_jitter = 1`.
const MAX_START_YAW: Real = 0.35;
const MAX_START_TILT: Real = 0.2;
const MAX_START_OFFSET: Real = 0.5;

pub const CLOCK_HZ: Real = 1.0;

/// One recorded instant: body poses at a point in time.
///
/// Poses are stored flat, seven values per body (position xyz, orientation xyzw),
/// which keeps the JSON compact and maps directly onto what a renderer wants.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Frame {
    pub t: Real,
    pub poses: Vec<Real>,
}

/// A recorded trajectory, sufficient to replay an organism's behaviour.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Trace {
    pub bodies: Vec<BodySpec>,
    pub record_hz: Real,
    /// Simulation time at which measurement began; frames before this are the
    /// settling drop.
    pub measure_start_t: Real,
    /// The ground the organism ran on, so a viewer can draw the same surface the
    /// physics used rather than assuming a flat one.
    #[serde(default = "flat_ground")]
    pub terrain: crate::physics::TerrainModel,
    pub frames: Vec<Frame>,
    /// Joints that failed during the run, as `(body detached, time)`. Empty for
    /// any experiment whose joints cannot break, and omitted from the JSON
    /// entirely in that case.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub breaks: Vec<JointBreak>,
}

/// What a replay recorded before terrain was written down: flat ground, which is
/// the only kind that existed then.
fn flat_ground() -> crate::physics::TerrainModel {
    crate::physics::TerrainModel::Flat { height: 0.0 }
}

/// A joint failing mid-run, recorded so a viewer can mark the moment a limb
/// came off rather than leaving it to be inferred from the poses.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct JointBreak {
    /// Index into `Trace::bodies` of the body that came loose.
    pub body: u16,
    /// Simulation time of the failure, on the same clock as `Frame::t`.
    pub t: Real,
}

/// The outcome of one evaluation.
pub struct EvalResult {
    pub fitness: Real,
    pub metrics: Metrics,
    pub trace: Option<Trace>,
}

/// Reusable buffers for evaluating many organisms on one thread.
///
/// Evaluation is short, so per-evaluation allocation would be a measurable
/// fraction of the cost. One of these per worker thread removes it.
pub struct EvalWorkspace {
    scratch: BrainScratch,
}

impl EvalWorkspace {
    pub fn new(cfg: &Config) -> EvalWorkspace {
        EvalWorkspace { scratch: BrainScratch::new(&cfg.brain_layout()) }
    }
}

/// Evaluate `genome` under `cfg`, optionally recording a trajectory.
pub fn evaluate(genome: &Genome, cfg: &Config, record: bool) -> EvalResult {
    let mut ws = EvalWorkspace::new(cfg);
    evaluate_with(genome, cfg, record, &mut ws)
}

/// Evaluate using caller-provided scratch space. This is the form the parallel
/// evaluator uses.
pub fn evaluate_with(
    genome: &Genome,
    cfg: &Config,
    record: bool,
    ws: &mut EvalWorkspace,
) -> EvalResult {
    let trials = cfg.simulation.trials.max(1);
    if trials == 1 && cfg.simulation.start_jitter <= 0.0 {
        return run_trial(genome, cfg, record, ws, None);
    }

    // Every organism in the experiment faces the *same* set of starts, because
    // the perturbations are drawn from the experiment seed and the trial index
    // and nothing else. Common random numbers: two organisms differ in their
    // scores because they differ, not because one drew an easier world.
    let mut total = 0.0;
    let mut worst = Real::INFINITY;
    let mut acc: Option<Metrics> = None;
    let mut first_trace = None;
    for trial in 0..trials {
        let mut rng = Rng::new(derive_seed(&[cfg.experiment.seed, TRIAL_STREAM, trial as u64]));
        let start = perturbation(&mut rng, cfg.simulation.start_jitter);
        let heading = commanded_heading(&mut rng, cfg);
        let r = run_trial_towards(genome, cfg, record && trial == 0, ws, Some(start), heading);
        total += r.fitness;
        worst = worst.min(r.fitness);
        if trial == 0 {
            first_trace = r.trace;
        }
        acc = Some(match acc {
            None => r.metrics,
            Some(a) => accumulate(a, r.metrics),
        });
    }

    let n = trials as Real;
    let mut metrics = acc.unwrap_or_default();
    scale_metrics(&mut metrics, 1.0 / n);
    // An organism is scored on the aggregate of its trials, not on a single
    // lucky one. `Worst` asks for a strategy with no bad day at all.
    let fitness = match cfg.simulation.aggregate {
        Aggregate::Mean => total / n,
        Aggregate::Worst => worst,
    };
    EvalResult { fitness, metrics, trace: first_trace }
}

/// Draw a start pose. `jitter` of 0 leaves the organism exactly where it would
/// have been, so a single-trial experiment is unaffected.
fn perturbation(rng: &mut Rng, jitter: Real) -> phenotype::StartPerturbation {
    phenotype::StartPerturbation {
        yaw: rng.signed() * MAX_START_YAW * jitter,
        tilt: rng.signed() * MAX_START_TILT * jitter,
        offset: vec3(rng.signed(), 0.0, rng.signed()) * (MAX_START_OFFSET * jitter),
    }
}

/// The direction this trial asks the organism to travel.
///
/// Unsteered experiments always command +X, which is what every objective but
/// `Heading` measures anyway.
fn commanded_heading(rng: &mut Rng, cfg: &Config) -> Vec3 {
    if !cfg.simulation.steer {
        return Vec3::X;
    }
    let angle = rng.signed() * cfg.simulation.steer_spread;
    let (s, c) = crate::math::dsincos(angle);
    // Rotation about the vertical: +X turned by `angle` toward +Z.
    vec3(c, 0.0, s)
}

/// Sum two metric sets field by field, for later averaging.
fn accumulate(a: Metrics, b: Metrics) -> Metrics {
    Metrics {
        start: a.start + b.start,
        end: a.end + b.end,
        displacement: a.displacement + b.displacement,
        displacement_x: a.displacement_x + b.displacement_x,
        path_length: a.path_length + b.path_length,
        max_displacement: a.max_displacement + b.max_displacement,
        mean_height: a.mean_height + b.mean_height,
        upright_seconds: a.upright_seconds + b.upright_seconds,
        actuation: a.actuation + b.actuation,
        duration: a.duration + b.duration,
        heading_progress: a.heading_progress + b.heading_progress,
        peak_height: a.peak_height + b.peak_height,
        airborne_seconds: a.airborne_seconds + b.airborne_seconds,
        // Counts and flags describe the whole set rather than averaging.
        steps: a.steps.max(b.steps),
        joints_lost: a.joints_lost.max(b.joints_lost),
        diverged: a.diverged || b.diverged,
    }
}

fn scale_metrics(m: &mut Metrics, k: Real) {
    m.start = m.start * k;
    m.end = m.end * k;
    m.displacement *= k;
    m.displacement_x *= k;
    m.path_length *= k;
    m.max_displacement *= k;
    m.mean_height *= k;
    m.upright_seconds *= k;
    m.actuation *= k;
    m.duration *= k;
    m.heading_progress *= k;
    m.peak_height *= k;
    m.airborne_seconds *= k;
}

/// One trial: build the organism, simulate it, and score it.
fn run_trial(
    genome: &Genome,
    cfg: &Config,
    record: bool,
    ws: &mut EvalWorkspace,
    start: Option<phenotype::StartPerturbation>,
) -> EvalResult {
    run_trial_towards(genome, cfg, record, ws, start, Vec3::X)
}

/// One trial, travelling toward `heading` — a horizontal unit vector.
fn run_trial_towards(
    genome: &Genome,
    cfg: &Config,
    record: bool,
    ws: &mut EvalWorkspace,
    start: Option<phenotype::StartPerturbation>,
    heading: Vec3,
) -> EvalResult {
    let mut pheno = phenotype::build_with_start(genome, cfg, start);
    let layout = cfg.brain_layout();
    debug_assert_eq!(genome.weights.len(), layout.weight_count());

    let dt = cfg.simulation.timestep;
    let settle_steps = cfg.settle_steps();
    let total_steps = cfg.total_steps();
    let control_interval = steps_per(cfg.simulation.control_hz, dt);
    let record_interval = steps_per(cfg.recording.record_hz, dt);

    let mut trace = record.then(|| Trace {
        bodies: pheno.body_specs(),
        record_hz: cfg.recording.record_hz,
        measure_start_t: settle_steps as Real * dt,
        // Plus the closing frame and the measurement-boundary frame.
        frames: Vec::with_capacity((total_steps / record_interval) as usize + 2),
        breaks: Vec::new(),
        terrain: pheno.world.params.terrain,
    });

    // How hard this organism drives its motors. With joint damage off, caution is
    // always 0 and drive is exactly 1, so nothing changes.
    let drive = if cfg.joints_can_break() {
        crate::math::clamp(1.0 - genome.caution, cfg.body.min_drive, 1.0)
    } else {
        1.0
    };

    let mut metrics = Metrics::default();
    let mut previous_com = Vec3::ZERO;
    let mut height_sum = 0.0;
    let mut measured_steps: u32 = 0;
    let mut measurement_began = false;
    // Actuation spent during the settle drop belongs to no measured window: the
    // controller is held off, so it is impulse the organism could not have
    // influenced. Subtracting the settle total keeps every metric on `Metrics`
    // describing the same interval.
    let mut settle_actuation = 0.0;

    for step in 0..total_steps {
        let t = step as Real * dt;

        // Measurement starts from the state the settled organism is *in*, before
        // the first controlled step acts on it. Capturing it after that step
        // would put `Trace::measure_start_t` one step ahead of the pose the
        // window is actually measured from.
        if step == settle_steps {
            metrics.start = pheno.world.centre_of_mass();
            previous_com = metrics.start;
            settle_actuation = pheno.world.actuation_impulse;
            measurement_began = true;
        }

        if should_apply_control(step, settle_steps, control_interval) {
            let measured_t = (step - settle_steps) as Real * dt;
            apply_control(&mut pheno, &layout, &genome.weights, ws, measured_t, drive, heading);
        }

        if let Some(tr) = trace.as_mut() {
            // The measurement boundary always gets a frame, even when it does not
            // fall on the recording grid. Without it a viewer cannot draw the pose
            // that `measure_start_t` names and has to interpolate towards it.
            if step % record_interval == 0 || step == settle_steps {
                tr.frames.push(capture_frame(&pheno, t));
            }
        }

        pheno.world.step(dt);

        if pheno.world.diverged {
            metrics.diverged = true;
            break;
        }

        if step >= settle_steps {
            let com = pheno.world.centre_of_mass();
            let delta = horizontal(com - previous_com);
            metrics.path_length += delta.length();
            previous_com = com;

            let offset = horizontal(com - metrics.start);
            metrics.max_displacement = metrics.max_displacement.max(offset.length());
            height_sum += com.y;

            let up_y = pheno.world.bodies[0].orient.rotate(Vec3::Y).y;
            if up_y > 0.7 {
                metrics.upright_seconds += dt;
            }

            metrics.peak_height = metrics.peak_height.max(com.y);
            // Airborne means the whole organism has cleared the ground by a real
            // margin. Debris is excluded deliberately: a shed limb bouncing
            // along is not the organism flying.
            if pheno.world.ground_clearance() > AIRBORNE_CLEARANCE {
                metrics.airborne_seconds += dt;
            }

            measured_steps += 1;
        }
    }

    metrics.joints_lost = pheno.world.breaks.len() as u32;

    if !metrics.diverged {
        if let Some(tr) = trace.as_mut() {
            tr.frames.push(capture_frame(&pheno, total_steps as Real * dt));
            // The world reports breaks against joint indices; a viewer only knows
            // bodies, and a joint's child body is what visibly comes off.
            tr.breaks = pheno
                .world
                .breaks
                .iter()
                .map(|&(joint, t)| JointBreak {
                    body: pheno.world.joints[joint as usize].body_b,
                    t,
                })
                .collect();
        }
        // Guarded on measurement having begun at all: a configuration whose
        // `settle_time` rounds up to the whole evaluation never sets
        // `metrics.start`, and differencing against a default origin would report
        // the organism's absolute position as displacement.
        if measurement_began {
            let com = pheno.world.centre_of_mass();
            metrics.heading_progress = horizontal(com - metrics.start).dot(heading);
            metrics.end = com;
            let offset = horizontal(com - metrics.start);
            metrics.displacement = offset.length();
            metrics.displacement_x = offset.x;
        }
        metrics.mean_height =
            if measured_steps > 0 { height_sum / measured_steps as Real } else { 0.0 };
    }

    metrics.steps = measured_steps;
    metrics.duration = measured_steps as Real * dt;
    metrics.actuation = pheno.world.actuation_impulse - settle_actuation;

    let fitness = fitness::score(&cfg.fitness, &metrics);
    EvalResult { fitness, metrics, trace: if metrics.diverged { None } else { trace } }
}

/// Gather sensors, run the controller, and write motor targets.
fn apply_control(
    pheno: &mut Phenotype,
    layout: &crate::brain::BrainLayout,
    weights: &[Real],
    ws: &mut EvalWorkspace,
    t: Real,
    drive: Real,
    heading: Vec3,
) {
    let s = &mut ws.scratch;
    s.clear_inputs();

    s.inputs[input::BIAS] = 1.0;
    s.inputs[input::CLOCK_A] = triangle(t * CLOCK_HZ);
    s.inputs[input::CLOCK_B] = triangle(t * CLOCK_HZ + 0.25);

    let root = &pheno.world.bodies[0];
    s.inputs[input::UP_Y] = root.orient.rotate(Vec3::Y).y;
    s.inputs[input::RIGHT_Y] = root.orient.rotate(Vec3::X).y;
    // Velocities and heights are scaled into roughly [-1, 1] so that a freshly
    // initialised network is not immediately saturated.
    s.inputs[input::VEL_X] = clamp(root.lin_vel.x * 0.2, -1.0, 1.0);
    s.inputs[input::VEL_Y] = clamp(root.lin_vel.y * 0.2, -1.0, 1.0);
    s.inputs[input::VEL_Z] = clamp(root.lin_vel.z * 0.2, -1.0, 1.0);
    s.inputs[input::HEIGHT] = clamp(root.pos.y * 0.5, 0.0, 1.0);
    // Where it has been told to go. Absent in an unsteered experiment, where the
    // command is always +X and telling the controller so would be nine tenths of
    // a wasted weight.
    if layout.is_steered() {
        s.inputs[input::COMMAND_X] = heading.x;
        s.inputs[input::COMMAND_Z] = heading.z;
    }

    // A slot can own more than one joint and more than one body: a mirrored
    // pair, or a chain of repeated segments. They share one set of controller
    // weights, so their senses are averaged into one reading and their motors
    // take one command. That sharing is the point — two legs driven by one leg
    // controller move as a pair, which is what a gait is, whereas two
    // independently wired legs mostly flail.
    for (j, &slot) in pheno.joint_slots.iter().enumerate() {
        let (cos_a, sin_a) = pheno.world.hinge_angle_cos_sin(j);
        let base = layout.slot_input_base(slot as usize);
        let n = pheno.slot_joints[slot as usize].max(1) as Real;
        s.inputs[base] += cos_a / n;
        s.inputs[base + 1] += sin_a / n;
    }
    for (b, &slot) in pheno.body_slots.iter().enumerate() {
        let base = layout.slot_input_base(slot as usize);
        let n = pheno.slot_bodies[slot as usize].max(1) as Real;
        if pheno.world.body_in_contact(b) {
            s.inputs[base + 2] += 1.0 / n;
        }
    }
    // The reactive half of the wear trade-off: an organism that can feel a joint
    // failing can ease off it. Present only when the experiment lets joints
    // break, because the input count sets the weight-vector length.
    if layout.senses_health() {
        for (j, &slot) in pheno.joint_slots.iter().enumerate() {
            let base = layout.slot_input_base(slot as usize);
            let n = pheno.slot_joints[slot as usize].max(1) as Real;
            s.inputs[base + 3] += pheno.world.joints[j].health_fraction() / n;
        }
    }

    brain::evaluate(layout, weights, s);

    // `drive` is the standing half: how hard this organism is willing to push
    // regardless of what it senses. Throttling the requested *speed* lowers the
    // velocity error the motor has to close, so a cautious organism saturates
    // its motors less often and wears its joints more slowly — at the cost of
    // being slower.
    for (j, &slot) in pheno.joint_slots.iter().enumerate() {
        let sign = pheno.joint_drive[j];
        let joint = &mut pheno.world.joints[j];
        joint.motor_target = s.outputs[slot as usize] * joint.motor_speed_max * drive * sign;
    }
}

fn capture_frame(pheno: &Phenotype, t: Real) -> Frame {
    let mut poses = Vec::with_capacity(pheno.world.bodies.len() * 7);
    for b in &pheno.world.bodies {
        poses.extend_from_slice(&[
            b.pos.x, b.pos.y, b.pos.z, b.orient.x, b.orient.y, b.orient.z, b.orient.w,
        ]);
    }
    Frame { t, poses }
}

#[inline]
fn horizontal(v: Vec3) -> Vec3 {
    crate::math::vec3(v.x, 0.0, v.z)
}

/// Steps between events occurring at `hz`, at least one.
fn steps_per(hz: Real, dt: Real) -> u32 {
    let interval = (1.0 / hz) / dt;
    (interval.round() as u32).max(1)
}

/// The controller stays off while the organism drops onto the terrain, so
/// settle is a passive fall rather than a powered twitch that measurement
/// then treats as the starting pose.
fn should_apply_control(step: u32, settle_steps: u32, control_interval: u32) -> bool {
    step >= settle_steps && step % control_interval == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn quick_config() -> Config {
        let mut cfg = Config::default();
        cfg.simulation.duration = 2.0;
        cfg.simulation.settle_time = 0.25;
        cfg
    }

    fn random_genome(cfg: &Config, seed: u64) -> Genome {
        Genome::random(&mut Rng::new(seed), &cfg.body, &cfg.brain, &cfg.brain_layout())
    }

    #[test]
    fn evaluation_is_reproducible() {
        let cfg = quick_config();
        let g = random_genome(&cfg, 4);
        let a = evaluate(&g, &cfg, false);
        let b = evaluate(&g, &cfg, false);
        assert_eq!(a.fitness, b.fitness);
        assert_eq!(a.metrics, b.metrics);
    }

    #[test]
    fn recording_does_not_change_the_simulation() {
        let cfg = quick_config();
        let g = random_genome(&cfg, 5);
        let plain = evaluate(&g, &cfg, false);
        let recorded = evaluate(&g, &cfg, true);
        assert_eq!(plain.fitness, recorded.fitness);
        assert_eq!(plain.metrics, recorded.metrics);
        assert!(recorded.trace.is_some());
        assert!(plain.trace.is_none());
    }

    #[test]
    fn trace_shape_matches_the_organism() {
        let cfg = quick_config();
        let g = random_genome(&cfg, 6);
        let r = evaluate(&g, &cfg, true);
        let tr = r.trace.unwrap();
        assert_eq!(tr.bodies.len(), g.part_count());
        assert!(!tr.frames.is_empty());
        for f in &tr.frames {
            assert_eq!(f.poses.len(), g.part_count() * 7);
            assert!(f.poses.iter().all(|v| v.is_finite()));
        }
        // Frames are in time order.
        for w in tr.frames.windows(2) {
            assert!(w[1].t > w[0].t);
        }
    }

    #[test]
    fn metrics_are_self_consistent() {
        let cfg = quick_config();
        for seed in 0..40 {
            let g = random_genome(&cfg, 200 + seed);
            let m = evaluate(&g, &cfg, false).metrics;
            if m.diverged {
                continue;
            }
            assert!(m.displacement >= 0.0);
            // A straight-line displacement can never exceed the path walked.
            assert!(
                m.path_length >= m.displacement - 1e-3,
                "seed {seed}: path {} < displacement {}",
                m.path_length,
                m.displacement
            );
            assert!(m.max_displacement >= m.displacement - 1e-3);
            assert!(m.upright_seconds <= m.duration + 1e-4);
            assert!((m.duration - cfg.simulation.duration).abs() < 0.05);
        }
    }

    #[test]
    fn a_still_organism_scores_near_zero() {
        // Zero weights mean zero motor targets, so nothing should move much
        // after settling. Note this does *not* mean zero actuation: a motor
        // commanded to zero velocity still spends impulse holding the joint
        // against gravity, which is exactly what a real actuator does.
        let cfg = quick_config();
        let mut g = random_genome(&cfg, 7);
        for w in g.weights.iter_mut() {
            *w = 0.0;
        }
        let r = evaluate(&g, &cfg, false);
        assert!(r.metrics.displacement < 0.2, "displacement {}", r.metrics.displacement);
        assert!(r.fitness < 0.2);
    }

    #[test]
    fn an_active_controller_actually_actuates() {
        let cfg = quick_config();
        let mut any_actuation = false;
        for seed in 0..30 {
            let g = random_genome(&cfg, 300 + seed);
            if g.hinge_count() == 0 {
                continue;
            }
            let m = evaluate(&g, &cfg, false).metrics;
            if m.actuation > 0.0 {
                any_actuation = true;
                break;
            }
        }
        assert!(any_actuation, "no random organism moved a joint at all");
    }

    #[test]
    fn most_random_organisms_do_not_diverge() {
        let cfg = quick_config();
        let mut diverged = 0;
        let n = 100;
        for seed in 0..n {
            if evaluate(&random_genome(&cfg, 500 + seed), &cfg, false).metrics.diverged {
                diverged += 1;
            }
        }
        assert!(diverged < n / 10, "{diverged}/{n} organisms diverged");
    }

    /// Shapes are new geometry meeting an old solver. A sphere resting on one
    /// contact point and a cylinder standing on its rim are both cases a box
    /// never produced, so the divergence budget has to be checked against them
    /// specifically rather than inferred from the box result.
    #[test]
    fn most_random_shaped_organisms_do_not_diverge() {
        let mut cfg = quick_config();
        cfg.body.shapes = vec![
            crate::genome::ShapeKind::Box,
            crate::genome::ShapeKind::Taper,
            crate::genome::ShapeKind::Sphere,
            crate::genome::ShapeKind::Capsule,
            crate::genome::ShapeKind::Cylinder,
        ];
        let mut diverged = 0;
        let n = 100;
        for seed in 0..n {
            if evaluate(&random_genome(&cfg, 500 + seed), &cfg, false).metrics.diverged {
                diverged += 1;
            }
        }
        assert!(diverged < n / 10, "{diverged}/{n} shaped organisms diverged");
    }

    fn repeated_config() -> Config {
        let mut cfg = quick_config();
        cfg.simulation.trials = 4;
        cfg.simulation.start_jitter = 0.6;
        cfg
    }

    /// Repeating a trial must not make evaluation any less of a pure function.
    /// The perturbations come from the experiment seed and the trial index, so
    /// the same organism always meets the same four worlds.
    #[test]
    fn repeated_trials_are_deterministic() {
        let cfg = repeated_config();
        for seed in 0..12 {
            let g = random_genome(&cfg, 900 + seed);
            let a = evaluate(&g, &cfg, false);
            let b = evaluate(&g, &cfg, false);
            assert_eq!(a.fitness.to_bits(), b.fitness.to_bits(), "seed {seed}");
        }
    }

    /// Every organism faces the same starts — common random numbers — so a
    /// score difference is a difference between organisms, not between the
    /// worlds they happened to draw. Changing the experiment seed changes the
    /// worlds; changing the organism must not.
    #[test]
    fn every_organism_meets_the_same_worlds() {
        let mut cfg = repeated_config();
        let g = random_genome(&cfg, 4242);
        let first = evaluate(&g, &cfg, false).fitness;
        cfg.experiment.seed = cfg.experiment.seed.wrapping_add(1);
        let moved = evaluate(&g, &cfg, false).fitness;
        assert!(
            first.to_bits() != moved.to_bits(),
            "the trial starts ignored the experiment seed, so they are not varying at all"
        );
    }

    /// The worst trial can never flatter an organism more than the mean of them.
    #[test]
    fn the_worst_trial_never_scores_above_the_mean() {
        let mut cfg = repeated_config();
        for seed in 0..12 {
            let g = random_genome(&cfg, 700 + seed);
            cfg.simulation.aggregate = crate::config::Aggregate::Mean;
            let mean = evaluate(&g, &cfg, false).fitness;
            cfg.simulation.aggregate = crate::config::Aggregate::Worst;
            let worst = evaluate(&g, &cfg, false).fitness;
            assert!(worst <= mean + 1e-4, "seed {seed}: worst {worst} above mean {mean}");
        }
    }

    /// A jittered start actually moves the organism, or repeating the trial is
    /// theatre.
    #[test]
    fn a_jittered_start_actually_varies_the_outcome() {
        let cfg = repeated_config();
        let g = random_genome(&cfg, 31);
        let mut seen = Vec::new();
        for trial in 0..4u64 {
            let mut rng = Rng::new(derive_seed(&[cfg.experiment.seed, TRIAL_STREAM, trial]));
            let start = perturbation(&mut rng, cfg.simulation.start_jitter);
            let mut ws = EvalWorkspace::new(&cfg);
            seen.push(run_trial(&g, &cfg, false, &mut ws, Some(start)).fitness);
        }
        assert!(
            seen.windows(2).any(|w| w[0].to_bits() != w[1].to_bits()),
            "every trial produced an identical score: {seen:?}"
        );
    }

    /// A steered experiment gives the controller the command and scores what it
    /// did with it. Progress is measured along the commanded heading, so an
    /// organism sent one way and travelling another earns nothing.
    #[test]
    fn heading_progress_measures_the_commanded_direction() {
        let mut cfg = quick_config();
        cfg.simulation.steer = true;
        assert!(cfg.brain_layout().is_steered());
        assert!(cfg.brain_layout().weight_count() > quick_config().brain_layout().weight_count());

        let g = random_genome(&cfg, 77);
        let mut ws = EvalWorkspace::new(&cfg);
        // Straight down +X: progress and displacement_x are the same measurement.
        let ahead = run_trial_towards(&g, &cfg, false, &mut ws, None, Vec3::X);
        assert!(
            (ahead.metrics.heading_progress - ahead.metrics.displacement_x).abs() < 1e-4,
            "along +X the two should agree: {} vs {}",
            ahead.metrics.heading_progress,
            ahead.metrics.displacement_x
        );

        // Commanded sideways, progress is what it did along +Z instead.
        let across = run_trial_towards(&g, &cfg, false, &mut ws, None, Vec3::Z);
        let moved = across.metrics.end - across.metrics.start;
        assert!(
            (across.metrics.heading_progress - moved.z).abs() < 1e-4,
            "across +Z progress should be the Z displacement: {} vs {}",
            across.metrics.heading_progress,
            moved.z
        );
    }

    /// Commanded headings must actually vary between trials, or steering is
    /// nothing but an extra input.
    #[test]
    fn commanded_headings_vary_between_trials() {
        let mut cfg = quick_config();
        cfg.simulation.steer = true;
        cfg.simulation.steer_spread = 1.0;
        let mut seen = Vec::new();
        for trial in 0..6u64 {
            let mut rng = Rng::new(derive_seed(&[cfg.experiment.seed, TRIAL_STREAM, trial]));
            let _ = perturbation(&mut rng, cfg.simulation.start_jitter);
            let h = commanded_heading(&mut rng, &cfg);
            assert!((h.length() - 1.0).abs() < 1e-4, "heading not a unit vector: {h:?}");
            seen.push(h);
        }
        assert!(
            seen.windows(2).any(|w| (w[0] - w[1]).length() > 1e-3),
            "every trial commanded the same direction"
        );
        // And an unsteered experiment always commands +X.
        let plain = quick_config();
        let mut rng = Rng::new(1);
        assert_eq!(commanded_heading(&mut rng, &plain), Vec3::X);
    }

    #[test]
    fn steps_per_rounds_sensibly() {
        assert_eq!(steps_per(20.0, 1.0 / 120.0), 6);
        assert_eq!(steps_per(30.0, 1.0 / 120.0), 4);
        // Faster than the physics rate still means every step, never zero.
        assert_eq!(steps_per(1000.0, 1.0 / 120.0), 1);
    }

    #[test]
    fn control_stays_off_during_settle() {
        let interval = 6;
        for step in 0..60 {
            assert!(!should_apply_control(step, 60, interval), "step {step}");
        }
        assert!(should_apply_control(60, 60, interval));
        assert!(!should_apply_control(61, 60, interval));
        assert!(should_apply_control(66, 60, interval));
    }

    /// Effort must be charged for the measured window only. The settle drop
    /// spends real motor impulse holding joints against gravity, but the
    /// controller is switched off for it, so including it would make
    /// `energy_penalty` scale with `settle_time` — pricing a fall the organism
    /// could not influence.
    #[test]
    fn settle_actuation_is_not_charged_to_the_measured_window() {
        // Same settle, so the physics up to the measurement boundary is identical
        // and only the window length differs. Shrinking the window towards zero
        // must take reported effort towards zero with it; while the settle total
        // was included, this left a large constant intercept instead.
        let mut brief = quick_config();
        brief.simulation.settle_time = 1.0; // 120 steps of holding impulse
        brief.simulation.duration = 0.025; // 3 measured steps
        let mut full = brief.clone();
        full.simulation.duration = 1.0; // 120 measured steps

        let mut checked = 0;
        for seed in 0..40 {
            let g = random_genome(&full, 700 + seed);
            if g.hinge_count() == 0 {
                continue;
            }
            let a = evaluate(&g, &brief, false).metrics;
            let b = evaluate(&g, &full, false).metrics;
            if a.diverged || b.diverged || b.actuation <= 0.0 {
                continue;
            }
            assert!(
                a.actuation < 0.1 * b.actuation,
                "seed {seed}: {} over 3 measured steps vs {} over 120 — \
                 the settle window is leaking into the total",
                a.actuation,
                b.actuation
            );
            checked += 1;
        }
        assert!(checked > 5, "only {checked} organisms actuated at all");
    }

    #[test]
    fn measurement_starts_where_the_trace_says_it_does() {
        let cfg = quick_config();
        let g = random_genome(&cfg, 12);
        let r = evaluate(&g, &cfg, true);
        let tr = r.trace.unwrap();
        // The frame at `measure_start_t` must be the pose that `metrics.start`
        // was taken from, or a viewer highlights the wrong window.
        let frame = tr
            .frames
            .iter()
            .find(|f| (f.t - tr.measure_start_t).abs() < 1e-6)
            .expect("no frame at the declared measurement start");
        let com = centre_of_mass_of(&frame.poses, &tr.bodies, &cfg);
        assert!(
            (com - r.metrics.start).length() < 1e-4,
            "trace says measurement starts at {:?}, metrics say {:?}",
            com,
            r.metrics.start
        );
    }

    /// Mass-weighted centre of a recorded frame, reconstructed the way a viewer
    /// would have to.
    fn centre_of_mass_of(poses: &[Real], bodies: &[BodySpec], cfg: &Config) -> Vec3 {
        let mut total = 0.0;
        let mut acc = Vec3::ZERO;
        for (i, spec) in bodies.iter().enumerate() {
            let (m, _) = spec.geometry().mass_properties(cfg.body.density);
            let p = crate::math::vec3(poses[i * 7], poses[i * 7 + 1], poses[i * 7 + 2]);
            acc += p * m;
            total += m;
        }
        acc * (1.0 / total)
    }
}
