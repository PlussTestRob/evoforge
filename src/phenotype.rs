//! Genome expression: turning a [`Genome`] into a simulatable [`World`].
//!
//! This is the only place that knows how a gene becomes geometry. Keeping it
//! separate from the genome means the expression rules can be revised — different
//! attachment geometry, derived masses, symmetry operators — without changing
//! what is stored on disk, and means a genome is meaningful only in the context
//! of the code version and config that expressed it. Both facts are recorded with
//! every result.

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::genome::{Genome, JointKind};
use crate::math::{dcos, vec3, Real, Vec3};
use crate::physics::{Joint, RigidBody, TerrainModel, World, WorldParams};

/// Height above the terrain at which an organism is spawned. Small but nonzero,
/// so the first step resolves a shallow contact rather than a deep overlap.
pub const SPAWN_CLEARANCE: Real = 0.02;

/// One face of a box, in body-local coordinates.
pub struct Face {
    pub normal: Vec3,
    /// The two in-plane directions. A hinge axis is always one of these, which
    /// guarantees every generated hinge is geometrically sensible.
    pub tangents: [Vec3; 2],
}

/// Face order used by [`crate::genome::PartGene::attach_face`].
pub const FACES: [Face; 6] = [
    Face { normal: vec3(1.0, 0.0, 0.0), tangents: [vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)] },
    Face { normal: vec3(-1.0, 0.0, 0.0), tangents: [vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)] },
    Face { normal: vec3(0.0, 1.0, 0.0), tangents: [vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)] },
    Face { normal: vec3(0.0, -1.0, 0.0), tangents: [vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0)] },
    Face { normal: vec3(0.0, 0.0, 1.0), tangents: [vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)] },
    Face { normal: vec3(0.0, 0.0, -1.0), tangents: [vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)] },
];

/// Static description of a body, recorded with a replay so a viewer can draw the
/// organism without re-expressing the genome.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct BodySpec {
    pub half_extents: Vec3,
    pub slot: u8,
}

/// An expressed organism, ready to simulate.
pub struct Phenotype {
    pub world: World,
    /// Controller slot for each body, parallel to `world.bodies`.
    pub body_slots: Vec<u8>,
    /// Controller slot for each joint, parallel to `world.joints`. A joint takes
    /// the slot of its *child* part, so a limb's motor output and its angle
    /// sensor share an index.
    pub joint_slots: Vec<u8>,
    pub total_mass: Real,
}

impl Phenotype {
    pub fn body_specs(&self) -> Vec<BodySpec> {
        self.world
            .bodies
            .iter()
            .zip(&self.body_slots)
            .map(|(b, &slot)| BodySpec { half_extents: b.half_extents, slot })
            .collect()
    }
}

/// Express `genome` into a world according to `cfg`.
///
/// Deterministic and allocation-light: this runs once per evaluation, so it sits
/// on the hot path of the whole experiment.
pub fn build(genome: &Genome, cfg: &Config) -> Phenotype {
    let n = genome.parts.len();
    debug_assert!(n >= 1);

    let density = cfg.body.density;
    let mut centres: Vec<Vec3> = Vec::with_capacity(n);
    let mut bodies: Vec<RigidBody> = Vec::with_capacity(n);
    let mut joints: Vec<Joint> = Vec::with_capacity(n.saturating_sub(1));
    let mut body_slots: Vec<u8> = Vec::with_capacity(n);
    let mut joint_slots: Vec<u8> = Vec::with_capacity(n.saturating_sub(1));

    // Root at the origin; the whole organism is translated onto the terrain once
    // every part has been placed.
    centres.push(Vec3::ZERO);
    bodies.push(RigidBody::box_body(Vec3::ZERO, genome.parts[0].half_extents, density));
    body_slots.push(genome.parts[0].slot);

    for i in 1..n {
        let part = &genome.parts[i];
        let parent_index = part.parent as usize;
        let parent_extents = genome.parts[parent_index].half_extents;
        let face = &FACES[(part.attach_face % 6) as usize];

        // Point on the parent's face, in the parent's local frame.
        let n_abs = face.normal.abs();
        let t0_abs = face.tangents[0].abs();
        let t1_abs = face.tangents[1].abs();
        let anchor_in_parent = face.normal * parent_extents.dot(n_abs)
            + face.tangents[0] * (part.attach_u * parent_extents.dot(t0_abs))
            + face.tangents[1] * (part.attach_v * parent_extents.dot(t1_abs));

        // The child sits just outside that face, touching it.
        let child_offset = face.normal * part.half_extents.dot(n_abs);
        let centre = centres[parent_index] + anchor_in_parent + child_offset;
        centres.push(centre);
        bodies.push(RigidBody::box_body(centre, part.half_extents, density));
        body_slots.push(part.slot);

        // All parts are spawned axis-aligned, so local and world frames coincide
        // at construction and the joint frames are the same vectors in both
        // bodies. This is also why the fixed-joint rest pose is the identity.
        let axis_index = (part.joint.axis & 1) as usize;
        let axis = face.tangents[axis_index];
        let reference = face.tangents[1 - axis_index];

        joints.push(Joint {
            body_a: parent_index as u16,
            body_b: i as u16,
            kind: part.joint.kind,
            anchor_a: anchor_in_parent,
            anchor_b: -child_offset,
            axis_a: axis,
            axis_b: axis,
            ref_a: reference,
            ref_b: reference,
            cos_limit: match part.joint.kind {
                JointKind::Hinge => dcos(part.joint.limit),
                // A fixed joint's angular constraint is handled separately; the
                // limit is never consulted.
                JointKind::Fixed => -1.0,
            },
            motor_speed_max: part.joint.motor_speed,
            motor_torque_max: match part.joint.kind {
                JointKind::Hinge => part.joint.motor_torque,
                JointKind::Fixed => 0.0,
            },
            motor_target: 0.0,
        });
        joint_slots.push(part.slot);
    }

    // Drop the organism onto the terrain: translate straight up until no corner
    // is below the ground beneath *that corner*, plus a little clearance.
    //
    // Sampling the terrain per corner rather than once under the root costs one
    // pass over eight corners per body and is what keeps this correct when
    // `TerrainModel` grows a non-flat variant. For `Flat` it reduces to exactly
    // the same arithmetic.
    let terrain = cfg_terrain(cfg);
    let deepest = bodies
        .iter()
        .flat_map(|b| b.corners())
        .fold(Real::NEG_INFINITY, |m, c| m.max(terrain.height_at(c.x, c.z) - c.y));
    let lift = deepest + SPAWN_CLEARANCE;
    for b in bodies.iter_mut() {
        b.pos.y += lift;
    }

    let total_mass = bodies.iter().map(|b| b.mass()).sum();

    Phenotype {
        world: World::new(bodies, joints, world_params(cfg)),
        body_slots,
        joint_slots,
        total_mass,
    }
}

fn cfg_terrain(cfg: &Config) -> TerrainModel {
    match cfg.environment.terrain {
        crate::config::Terrain::Flat => TerrainModel::Flat { height: 0.0 },
    }
}

pub fn world_params(cfg: &Config) -> WorldParams {
    WorldParams {
        gravity: vec3(0.0, -cfg.environment.gravity, 0.0),
        terrain: cfg_terrain(cfg),
        friction: cfg.environment.friction,
        restitution: cfg.environment.restitution,
        iterations: cfg.simulation.solver_iterations,
        linear_damping: cfg.environment.linear_damping,
        angular_damping: cfg.environment.angular_damping,
        baumgarte: cfg.simulation.baumgarte,
        slop: cfg.simulation.slop,
        max_correction_speed: cfg.simulation.max_correction_speed,
        max_linear_speed: cfg.simulation.max_linear_speed,
        max_angular_speed: cfg.simulation.max_angular_speed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn cfg() -> Config {
        Config::default()
    }

    #[test]
    fn faces_are_orthonormal() {
        for f in FACES.iter() {
            assert!((f.normal.length() - 1.0).abs() < 1e-6);
            for t in f.tangents {
                assert!((t.length() - 1.0).abs() < 1e-6);
                assert!(f.normal.dot(t).abs() < 1e-6);
            }
            assert!(f.tangents[0].dot(f.tangents[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn built_organism_sits_on_the_ground() {
        let cfg = cfg();
        let layout = cfg.brain_layout();
        for seed in 0..100 {
            let mut rng = Rng::new(seed);
            let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            let p = build(&g, &cfg);
            let lowest =
                p.world.bodies.iter().fold(Real::INFINITY, |m, b| m.min(b.lowest_corner_y()));
            assert!(
                (lowest - SPAWN_CLEARANCE).abs() < 1e-4,
                "seed {seed}: lowest corner at {lowest}"
            );
        }
    }

    #[test]
    fn body_and_joint_counts_match_the_genome() {
        let cfg = cfg();
        let layout = cfg.brain_layout();
        let mut rng = Rng::new(3);
        for _ in 0..50 {
            let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            let p = build(&g, &cfg);
            assert_eq!(p.world.bodies.len(), g.part_count());
            assert_eq!(p.world.joints.len(), g.joint_count());
            assert_eq!(p.body_slots.len(), g.part_count());
            assert_eq!(p.joint_slots.len(), g.joint_count());
        }
    }

    #[test]
    fn joint_anchors_start_coincident() {
        let cfg = cfg();
        let layout = cfg.brain_layout();
        let mut rng = Rng::new(9);
        for _ in 0..100 {
            let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            let p = build(&g, &cfg);
            for j in &p.world.joints {
                let a = &p.world.bodies[j.body_a as usize];
                let b = &p.world.bodies[j.body_b as usize];
                let pa = a.pos + a.orient.rotate(j.anchor_a);
                let pb = b.pos + b.orient.rotate(j.anchor_b);
                assert!(
                    (pa - pb).length() < 1e-4,
                    "anchors {} apart at construction",
                    (pa - pb).length()
                );
            }
        }
    }

    #[test]
    fn hinge_axes_are_perpendicular_to_their_reference() {
        let cfg = cfg();
        let layout = cfg.brain_layout();
        let mut rng = Rng::new(15);
        let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        let p = build(&g, &cfg);
        for j in &p.world.joints {
            assert!(j.axis_a.dot(j.ref_a).abs() < 1e-6);
            assert!(j.axis_b.dot(j.ref_b).abs() < 1e-6);
        }
    }

    #[test]
    fn expression_is_deterministic() {
        let cfg = cfg();
        let layout = cfg.brain_layout();
        let g = Genome::random(&mut Rng::new(21), &cfg.body, &cfg.brain, &layout);
        let a = build(&g, &cfg);
        let b = build(&g, &cfg);
        for (x, y) in a.world.bodies.iter().zip(&b.world.bodies) {
            assert_eq!(x.pos, y.pos);
            assert_eq!(x.half_extents, y.half_extents);
        }
        assert_eq!(a.total_mass, b.total_mass);
    }

    #[test]
    fn world_params_carry_solver_knobs_from_config() {
        let mut cfg = cfg();
        cfg.simulation.baumgarte = 0.4;
        cfg.simulation.slop = 0.01;
        cfg.simulation.max_linear_speed = 12.0;
        let p = world_params(&cfg);
        assert!((p.baumgarte - 0.4).abs() < 1e-6);
        assert!((p.slop - 0.01).abs() < 1e-6);
        assert!((p.max_linear_speed - 12.0).abs() < 1e-6);
        assert_eq!(p.iterations, cfg.simulation.solver_iterations);
    }

    #[test]
    fn a_built_organism_settles_without_diverging() {
        let cfg = cfg();
        let layout = cfg.brain_layout();
        for seed in 0..60 {
            let mut rng = Rng::new(1000 + seed);
            let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            let mut p = build(&g, &cfg);
            for _ in 0..240 {
                p.world.step(cfg.simulation.timestep);
            }
            assert!(!p.world.diverged, "seed {seed} diverged while settling");
            for b in &p.world.bodies {
                assert!(b.pos.y > -1.0, "seed {seed} fell through the floor");
            }
        }
    }
}
