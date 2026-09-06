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
use crate::config::Config;
use crate::fitness::{self, Metrics};
use crate::genome::Genome;
use crate::math::{clamp, triangle, Real, Vec3};
use crate::phenotype::{self, BodySpec, Phenotype};

/// Frequency of the controller's clock inputs, Hz.
///
/// Two out-of-phase triangle waves give the network a pacemaker to build a gait
/// around, without having to evolve an oscillator from scratch first. Triangle
/// rather than sine so it stays exact for arbitrarily long simulations.
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
    pub frames: Vec<Frame>,
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
        EvalWorkspace {
            scratch: BrainScratch::new(&cfg.brain_layout()),
        }
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
    let mut pheno = phenotype::build(genome, cfg);
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
        frames: Vec::with_capacity((total_steps / record_interval) as usize + 1),
    });

    let mut metrics = Metrics::default();
    let mut previous_com = Vec3::ZERO;
    let mut height_sum = 0.0;
    let mut measured_steps: u32 = 0;

    for step in 0..total_steps {
        let t = step as Real * dt;

        if step % control_interval == 0 {
            let measured_t = (step.saturating_sub(settle_steps)) as Real * dt;
            apply_control(&mut pheno, &layout, &genome.weights, ws, measured_t);
        }

        if let Some(tr) = trace.as_mut() {
            if step % record_interval == 0 {
                tr.frames.push(capture_frame(&pheno, t));
            }
        }

        pheno.world.step(dt);

        if pheno.world.diverged {
            metrics.diverged = true;
            break;
        }

        // Measurement begins once the organism has settled, so that the initial
        // drop onto the terrain is not scored as locomotion.
        if step == settle_steps {
            metrics.start = pheno.world.centre_of_mass();
            previous_com = metrics.start;
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
            measured_steps += 1;
        }
    }

    if !metrics.diverged {
        if let Some(tr) = trace.as_mut() {
            tr.frames.push(capture_frame(&pheno, total_steps as Real * dt));
        }
        let com = pheno.world.centre_of_mass();
        metrics.end = com;
        let offset = horizontal(com - metrics.start);
        metrics.displacement = offset.length();
        metrics.displacement_x = offset.x;
        metrics.mean_height = if measured_steps > 0 {
            height_sum / measured_steps as Real
        } else {
            0.0
        };
    }

    metrics.steps = measured_steps;
    metrics.duration = measured_steps as Real * dt;
    metrics.actuation = pheno.world.actuation_impulse;

    let fitness = fitness::score(&cfg.fitness, &metrics);
    EvalResult {
        fitness,
        metrics,
        trace: if metrics.diverged { None } else { trace },
    }
}

/// Gather sensors, run the controller, and write motor targets.
fn apply_control(
    pheno: &mut Phenotype,
    layout: &crate::brain::BrainLayout,
    weights: &[Real],
    ws: &mut EvalWorkspace,
    t: Real,
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

    for (j, &slot) in pheno.joint_slots.iter().enumerate() {
        let (cos_a, sin_a) = pheno.world.hinge_angle_cos_sin(j);
        let base = layout.slot_input_base(slot as usize);
        s.inputs[base] = cos_a;
        s.inputs[base + 1] = sin_a;
    }
    for (b, &slot) in pheno.body_slots.iter().enumerate() {
        let base = layout.slot_input_base(slot as usize);
        s.inputs[base + 2] = if pheno.world.body_in_contact(b) { 1.0 } else { 0.0 };
    }

    brain::evaluate(layout, weights, s);

    for (j, &slot) in pheno.joint_slots.iter().enumerate() {
        let joint = &mut pheno.world.joints[j];
        joint.motor_target = s.outputs[slot as usize] * joint.motor_speed_max;
    }
}

fn capture_frame(pheno: &Phenotype, t: Real) -> Frame {
    let mut poses = Vec::with_capacity(pheno.world.bodies.len() * 7);
    for b in &pheno.world.bodies {
        poses.extend_from_slice(&[
            b.pos.x,
            b.pos.y,
            b.pos.z,
            b.orient.x,
            b.orient.y,
            b.orient.z,
            b.orient.w,
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

    #[test]
    fn steps_per_rounds_sensibly() {
        assert_eq!(steps_per(20.0, 1.0 / 120.0), 6);
        assert_eq!(steps_per(30.0, 1.0 / 120.0), 4);
        // Faster than the physics rate still means every step, never zero.
        assert_eq!(steps_per(1000.0, 1.0 / 120.0), 1);
    }
}
