//! The constraint solver.
//!
//! # Why not a third-party physics engine?
//!
//! Rapier and friends are good libraries, but they solve a much larger problem
//! than this one: arbitrary geometry, broad-phase acceleration, continuous
//! collision, sleeping, scene graphs. We need a handful of convex primitives on a
//! ground plane connected by hinges, with organisms that do not collide with
//! themselves. Every one of those simplifications removes an entire subsystem —
//! and the last one is what keeps [`super::shape`] small, because it means the
//! only collision query is a shape against the terrain. What is left is small
//! enough to read in one sitting, has no version-drift risk to reproducibility,
//! and has no per-evaluation setup cost worth mentioning — which matters when the
//! workload is millions of very short evaluations rather than one long one.
//!
//! # Method
//!
//! Semi-implicit Euler with sequential-impulse constraint solving in maximal
//! coordinates (each body carries its own 6 degrees of freedom, and joints are
//! constraints rather than a reduced parameterisation), with Baumgarte
//! stabilisation for position error.
//!
//! # The upgrade path
//!
//! Organisms are *trees* with no self-collision. That is precisely the case
//! where a reduced-coordinate articulated-body formulation (Featherstone's ABA)
//! is both faster and dramatically more stable — joints become exactly satisfied
//! by construction rather than approximately satisfied by iteration, so the
//! solver-iteration budget disappears. That is the intended replacement, and
//! this module is deliberately kept behind a narrow surface ([`World::step`],
//! plus accessors) so it can be swapped without touching evolution, fitness or
//! recording. Sequential impulses come first because they are far harder to get
//! *wrong*.
//!
//! # Not implemented
//!
//! Self-collision: an organism's blocks pass through each other. This is the
//! usual choice in this class of experiment (Sims 1994 did the same). It removes
//! the broad phase entirely and avoids the jitter that overlapping
//! newly-mutated limbs would otherwise cause.

use crate::genome::JointKind;
use crate::math::{clamp, dcos, dsin, vec3, Mat3, Real, Vec3, TAU};

use super::body::RigidBody;

/// Ground model.
///
/// An enum rather than a trait object: dispatch is in the innermost loop, and a
/// second variant is cheaper than an abstraction.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TerrainModel {
    Flat {
        height: Real,
    },
    /// Smooth rolling ground: two octaves of a separable sine field.
    ///
    /// Analytic rather than sampled, so the surface and its gradient are exact
    /// everywhere and there is no grid to store, interpolate or get wrong at the
    /// seams. Built from the deterministic [`dsincos`] rather than the standard
    /// library's, because the whole reproducibility argument rests on every
    /// transcendental in the pipeline being ours.
    ///
    /// Why it matters: on flat ground, rolling is optimal and legs are strictly
    /// worse, which is why evolution here keeps rediscovering the wheel. Broken
    /// ground is what makes legs the good answer, without a fitness function
    /// ever mentioning them.
    Rough {
        amplitude: Real,
        /// Distance between crests, metres.
        wavelength: Real,
    },
}

impl TerrainModel {
    #[inline]
    pub fn height_at(&self, x: Real, z: Real) -> Real {
        match *self {
            TerrainModel::Flat { height } => height,
            TerrainModel::Rough { amplitude, wavelength } => {
                let k = TAU / wavelength.max(1e-3);
                amplitude * (dsin(k * x) * dcos(k * z))
                    + 0.5 * amplitude * (dsin(2.0 * k * x + 1.7) * dcos(2.0 * k * z + 0.9))
            }
        }
    }

    #[inline]
    pub fn normal_at(&self, x: Real, z: Real) -> Vec3 {
        match *self {
            TerrainModel::Flat { .. } => Vec3::Y,
            TerrainModel::Rough { amplitude, wavelength } => {
                // Exact gradient of `height_at`, so contacts on a slope get the
                // slope's own normal rather than a finite-difference guess.
                let k = TAU / wavelength.max(1e-3);
                let dhdx = amplitude * k * dcos(k * x) * dcos(k * z)
                    + amplitude * k * dcos(2.0 * k * x + 1.7) * dcos(2.0 * k * z + 0.9);
                let dhdz = -amplitude * k * dsin(k * x) * dsin(k * z)
                    - amplitude * k * dsin(2.0 * k * x + 1.7) * dsin(2.0 * k * z + 0.9);
                vec3(-dhdx, 1.0, -dhdz).normalize_or(Vec3::Y)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WorldParams {
    pub gravity: Vec3,
    pub terrain: TerrainModel,
    pub friction: Real,
    pub restitution: Real,
    pub iterations: u32,
    pub linear_damping: Real,
    pub angular_damping: Real,
    /// Fraction of positional error corrected per step, for joints and contacts.
    pub baumgarte: Real,
    /// Penetration tolerated before positional correction kicks in. Prevents
    /// contacts from jittering against the numerical noise floor.
    pub slop: Real,
    /// Ceiling on Baumgarte-injected velocity, so a deeply penetrating body is
    /// pushed out steadily rather than launched.
    pub max_correction_speed: Real,
    /// Hard velocity clamps. A constraint solver can diverge given a pathological
    /// mutated body; clamping keeps a bad organism merely bad rather than letting
    /// it produce infinities that poison fitness statistics.
    pub max_linear_speed: Real,
    pub max_angular_speed: Real,
    /// Whether an organism's own parts collide with each other.
    pub self_collision: bool,
}

impl Default for WorldParams {
    fn default() -> Self {
        WorldParams {
            gravity: vec3(0.0, -9.81, 0.0),
            terrain: TerrainModel::Flat { height: 0.0 },
            friction: 0.8,
            restitution: 0.0,
            iterations: 10,
            linear_damping: 0.02,
            angular_damping: 0.05,
            baumgarte: 0.2,
            slop: 0.002,
            max_correction_speed: 2.0,
            max_linear_speed: 60.0,
            max_angular_speed: 40.0,
            self_collision: false,
        }
    }
}

/// A constraint between two bodies, derived from a [`crate::genome::JointGene`].
///
/// Anchors, axes and reference vectors are in body-local coordinates and never
/// change; everything derived per step lives in [`JointPrep`].
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    pub body_a: u16,
    pub body_b: u16,
    pub kind: JointKind,
    /// Anchor point in each body's local frame. The joint holds these coincident.
    pub anchor_a: Vec3,
    pub anchor_b: Vec3,
    /// Hinge axis in each body's local frame.
    pub axis_a: Vec3,
    pub axis_b: Vec3,
    /// Perpendicular reference direction in each local frame, coincident at zero
    /// angle. Measuring the hinge angle from these avoids ever calling `atan2`.
    pub ref_a: Vec3,
    pub ref_b: Vec3,
    /// `cos(limit)`. Comparing cosines rather than angles keeps the limit check
    /// to a dot product.
    pub cos_limit: Real,
    pub motor_speed_max: Real,
    pub motor_torque_max: Real,
    /// Target angular velocity about the hinge axis, written by the controller.
    pub motor_target: Real,
    /// Natural frequency of the joint's passive spring, rad/s. Zero for a joint
    /// with no tendon.
    ///
    /// A frequency rather than a torque, because a torque has to be matched to
    /// the limb it acts on: the same N m/rad that gently returns a thigh will
    /// fling a toe. Expressed this way the spring is scale-free — every joint
    /// oscillates at the same rate whatever its inertia — and it is stable for
    /// any `frequency * dt` below about one, which no plausible setting reaches.
    pub tendon_frequency: Real,
    /// Damping ratio of that spring. 1 is critically damped; 0 is a spring that
    /// rings forever.
    pub tendon_damping: Real,
    /// Radians of undelivered rotation this joint can absorb before it fails.
    /// Zero means the joint is indestructible, which is the behaviour every
    /// experiment had before joints could break.
    pub endurance: Real,
    /// Remaining health, counting down from `endurance`.
    pub health: Real,
    /// Set once health reaches zero. A broken joint stops constraining anything,
    /// so whatever hung from it falls away.
    pub broken: bool,
}

impl Joint {
    pub fn fixed(body_a: u16, body_b: u16, anchor_a: Vec3, anchor_b: Vec3) -> Joint {
        Joint {
            body_a,
            body_b,
            kind: JointKind::Fixed,
            anchor_a,
            anchor_b,
            axis_a: Vec3::X,
            axis_b: Vec3::X,
            ref_a: Vec3::Y,
            ref_b: Vec3::Y,
            cos_limit: -1.0,
            motor_speed_max: 0.0,
            motor_torque_max: 0.0,
            motor_target: 0.0,
            tendon_frequency: 0.0,
            tendon_damping: 0.0,
            endurance: 0.0,
            health: 0.0,
            broken: false,
        }
    }

    /// Remaining health as a fraction of capacity: 1 is pristine, 0 is failed.
    /// An indestructible joint always reports 1.
    #[inline]
    pub fn health_fraction(&self) -> Real {
        if self.endurance > 0.0 {
            clamp(self.health / self.endurance, 0.0, 1.0)
        } else {
            1.0
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct JointPrep {
    ra: Vec3,
    rb: Vec3,
    axis_w: Vec3,
    perp1: Vec3,
    perp2: Vec3,
    k_point: Mat3,
    k_ang: Mat3,
    /// Effective angular mass about the hinge axis and the two perpendiculars.
    k_axis: Real,
    k_perp1: Real,
    k_perp2: Real,
    motor_impulse: Real,
}

/// A contact between two of the organism's own parts.
#[derive(Clone, Copy, Debug)]
struct PairContact {
    a: u16,
    b: u16,
    /// Offsets from each body's centre of mass to the shared contact point.
    r_a: Vec3,
    r_b: Vec3,
    /// Points from `a` toward `b`.
    normal: Vec3,
    depth: Real,
    k_n: Real,
    pn: Real,
}

#[derive(Clone, Copy, Debug)]
struct Contact {
    body: u16,
    /// Offset from the body's centre of mass to the contact point, world frame.
    r: Vec3,
    normal: Vec3,
    tangent1: Vec3,
    tangent2: Vec3,
    depth: Real,
    k_n: Real,
    k_t1: Real,
    k_t2: Real,
    /// Accumulated impulses, needed so the friction cone can be clamped against
    /// the normal impulse actually applied.
    pn: Real,
    pt1: Real,
    pt2: Real,
    /// Restitution target captured before solving.
    bounce: Real,
}

/// The simulation world for one organism.
pub struct World {
    pub bodies: Vec<RigidBody>,
    pub joints: Vec<Joint>,
    pub params: WorldParams,
    /// Per-body inverse inertia in world coordinates, refreshed once per step
    /// rather than once per solver iteration.
    inv_inertia: Vec<Mat3>,
    prep: Vec<JointPrep>,
    contacts: Vec<Contact>,
    pair_contacts: Vec<PairContact>,
    /// Row-major `n x n` table of which body pairs are joined by a joint, and so
    /// are meant to touch. Built once, because it never changes.
    jointed: Vec<bool>,
    /// Whether each body has been cut loose from the root by a broken joint.
    /// Detached bodies still fall, tumble and collide with the ground — they are
    /// debris, not deletions — but they stop counting as part of the organism.
    detached: Vec<bool>,
    /// Joints that failed since the last time this was drained, with the time
    /// each failed at, so a recording can note when a limb came off.
    pub breaks: Vec<(u16, Real)>,
    /// Sum of `|motor angular impulse|` applied so far. A cheap, monotone proxy
    /// for actuation effort, used by energy-penalising fitness functions.
    pub actuation_impulse: Real,
    /// Simulation time accumulated by `step`, used only to timestamp breakages.
    elapsed: Real,
    /// Accumulated bookkeeping shift to subtract from the reported centre of
    /// mass. See [`World::centre_of_mass`].
    com_correction: Vec3,
    /// Set once any body leaves the representable range; the evaluation is then
    /// abandoned rather than allowed to produce meaningless fitness.
    pub diverged: bool,
}

impl World {
    pub fn new(bodies: Vec<RigidBody>, joints: Vec<Joint>, params: WorldParams) -> World {
        let n = bodies.len();
        let j = joints.len();
        // Parts joined by a joint are supposed to be in contact; only parts with
        // no joint between them are colliding when they overlap.
        let mut jointed = vec![false; n * n];
        for joint in &joints {
            let (a, b) = (joint.body_a as usize, joint.body_b as usize);
            jointed[a * n + b] = true;
            jointed[b * n + a] = true;
        }
        World {
            bodies,
            joints,
            params,
            inv_inertia: vec![Mat3::ZERO; n],
            prep: vec![JointPrep::default(); j],
            contacts: Vec::with_capacity(n * 8),
            pair_contacts: Vec::new(),
            jointed,
            detached: vec![false; n],
            breaks: Vec::new(),
            actuation_impulse: 0.0,
            elapsed: 0.0,
            com_correction: Vec3::ZERO,
            diverged: false,
        }
    }

    /// Advance the simulation by one fixed step.
    pub fn step(&mut self, dt: Real) {
        if self.diverged {
            return;
        }
        self.integrate_velocities(dt);
        self.refresh_inertia();
        // Tendons are a *force*, not a constraint, so they are applied once per
        // step alongside gravity rather than inside the solver's iteration loop.
        // Applied per iteration they would fire a dozen times a step and pump
        // energy into the organism — which, tried once, produced bodies
        // travelling thirty metres a second.
        self.build_contacts();
        self.build_pair_contacts();
        self.prepare_joints(dt);
        self.apply_tendons(dt);

        for _ in 0..self.params.iterations {
            self.solve_joints(dt);
            self.solve_contacts(dt);
            self.solve_pair_contacts(dt);
        }

        self.integrate_positions(dt);
        self.wear_joints(dt);
        self.check_finite();
    }

    /// Charge each saturated motor for the rotation it failed to deliver.
    ///
    /// A joint is only harmed while its motor is at the torque ceiling its
    /// genome set — that is exactly the state of being asked for more than it
    /// can give. The damage is the shortfall in the rotation actually achieved,
    /// in radians, which makes endurance a quantity with a physical meaning
    /// rather than an arbitrary point score, and makes it independent of the
    /// solver's iteration count.
    fn wear_joints(&mut self, dt: Real) {
        for i in 0..self.joints.len() {
            let j = self.joints[i];
            if j.broken || j.endurance <= 0.0 || j.motor_torque_max <= 0.0 {
                continue;
            }
            // Saturated means the accumulated motor impulse hit its budget.
            let budget = j.motor_torque_max * dt;
            if self.prep[i].motor_impulse.abs() < budget * 0.999 {
                continue;
            }
            let target = clamp(j.motor_target, -j.motor_speed_max, j.motor_speed_max);
            let axis = self.prep[i].axis_w;
            let achieved = (self.bodies[j.body_b as usize].ang_vel
                - self.bodies[j.body_a as usize].ang_vel)
                .dot(axis);
            let shortfall = (target - achieved).abs();
            if shortfall <= 0.0 {
                continue;
            }
            let joint = &mut self.joints[i];
            joint.health -= shortfall * dt;
            if joint.health <= 0.0 {
                joint.health = 0.0;
                joint.broken = true;
                self.breaks.push((i as u16, self.elapsed));
                // Take the shift this detachment causes and cancel it, so
                // shedding a limb neither teleports the organism forward nor
                // drags it back.
                let before = self.attached_centre_of_mass();
                self.refresh_detached();
                let after = self.attached_centre_of_mass();
                self.com_correction += after - before;
            }
        }
        self.elapsed += dt;
    }

    /// Recompute which bodies are still connected to the root.
    ///
    /// Relies on the invariant [`crate::phenotype::build`] maintains: joints are
    /// stored parent-before-child, so one forward pass propagates a break down
    /// the whole subtree hanging off it.
    fn refresh_detached(&mut self) {
        for d in self.detached.iter_mut() {
            *d = false;
        }
        for j in &self.joints {
            let cut = j.broken || self.detached[j.body_a as usize];
            if cut {
                self.detached[j.body_b as usize] = true;
            }
        }
    }

    /// Whether `body` is still part of the organism rather than debris.
    #[inline]
    pub fn is_attached(&self, body: usize) -> bool {
        !self.detached[body]
    }

    /// Centre of mass of the whole organism.
    /// Centre of mass of the organism, counting only what is still attached.
    ///
    /// A shed limb keeps falling and tumbling in the world, but it stops being
    /// part of *you*: fitness should charge an organism for losing a limb's
    /// usefulness, not for where the wreckage happens to land.
    ///
    /// # Why the correction
    ///
    /// Dropping a body out of an average moves that average, instantly and for
    /// free. Shed a limb that trails behind you and the mean of what is left
    /// lurches forward — displacement the organism never travelled. Measured on
    /// a real run, most organisms lost a little distance this way and a few
    /// gained a great deal: one collected 3.34 m, a third of its recorded
    /// distance, by discarding a part at the right moment.
    ///
    /// So the discontinuity is cancelled. At the instant a joint fails the shift
    /// is measured and folded into `com_correction`, which every later reading
    /// subtracts. The reported centre of mass is therefore continuous across a
    /// breakage while still tracking only the attached parts afterwards — the
    /// wreckage stops counting, but detaching it is worth exactly zero metres.
    pub fn centre_of_mass(&self) -> Vec3 {
        self.attached_centre_of_mass() - self.com_correction
    }

    /// The raw mean position of everything still attached, before the
    /// continuity correction. This is the quantity that jumps.
    fn attached_centre_of_mass(&self) -> Vec3 {
        let mut total = 0.0;
        let mut acc = Vec3::ZERO;
        for (i, b) in self.bodies.iter().enumerate() {
            if self.detached[i] {
                continue;
            }
            let m = b.mass();
            total += m;
            acc += b.pos * m;
        }
        if total > 0.0 {
            acc * (1.0 / total)
        } else {
            Vec3::ZERO
        }
    }

    /// `(cos, sin)` of the hinge angle, measured from the reference vectors.
    ///
    /// Returned as a pair rather than an angle: it costs no transcendentals, and
    /// it is a better controller input because it has no discontinuity at the
    /// wrap-around.
    pub fn hinge_angle_cos_sin(&self, joint_index: usize) -> (Real, Real) {
        let j = &self.joints[joint_index];
        let a = &self.bodies[j.body_a as usize];
        let b = &self.bodies[j.body_b as usize];
        let axis = a.orient.rotate(j.axis_a).normalize_or(Vec3::X);
        let ra = project_out(a.orient.rotate(j.ref_a), axis).normalize_or(axis.any_perpendicular());
        let rb = project_out(b.orient.rotate(j.ref_b), axis).normalize_or(ra);
        (clamp(ra.dot(rb), -1.0, 1.0), ra.cross(rb).dot(axis))
    }

    /// Smallest gap between any still-attached part and the terrain below it.
    ///
    /// Negative while something is penetrating, zero while resting, positive
    /// only when the whole organism is genuinely off the ground.
    ///
    /// This exists because "is anything in contact?" is not the same question.
    /// A contact is only generated once a point is *below* the terrain, so a
    /// body skimming a millimetre above it registers no contact at all. Asked
    /// for hang time on that basis, evolution promptly produced organisms that
    /// spent half the trial "airborne" while never rising above the grass.
    pub fn ground_clearance(&self) -> Real {
        let mut gap = Real::INFINITY;
        for (i, body) in self.bodies.iter().enumerate() {
            if self.detached[i] {
                continue;
            }
            let (points, count) = body.ground_points(Vec3::Y);
            for p in &points[..count] {
                gap = gap.min(p.y - self.params.terrain.height_at(p.x, p.z));
            }
        }
        if gap.is_finite() {
            gap
        } else {
            0.0
        }
    }

    /// Whether any corner of `body` is touching the terrain.
    pub fn body_in_contact(&self, body: usize) -> bool {
        self.contacts.iter().any(|c| c.body as usize == body)
    }

    // -----------------------------------------------------------------------
    // Integration
    // -----------------------------------------------------------------------

    fn integrate_velocities(&mut self, dt: Real) {
        let g = self.params.gravity;
        let lin_scale = 1.0 - clamp(self.params.linear_damping * dt, 0.0, 1.0);
        let ang_scale = 1.0 - clamp(self.params.angular_damping * dt, 0.0, 1.0);
        for b in self.bodies.iter_mut() {
            b.lin_vel += g * dt;
            b.lin_vel = b.lin_vel * lin_scale;
            b.ang_vel = b.ang_vel * ang_scale;
        }
    }

    fn integrate_positions(&mut self, dt: Real) {
        let max_v = self.params.max_linear_speed;
        let max_w = self.params.max_angular_speed;
        for b in self.bodies.iter_mut() {
            clamp_speed(&mut b.lin_vel, max_v);
            clamp_speed(&mut b.ang_vel, max_w);
            b.pos += b.lin_vel * dt;
            b.orient = b.orient.integrate(b.ang_vel, dt);
        }
    }

    fn refresh_inertia(&mut self) {
        for (i, b) in self.bodies.iter().enumerate() {
            self.inv_inertia[i] = b.inv_inertia_world();
        }
    }

    fn check_finite(&mut self) {
        for b in &self.bodies {
            if !b.is_finite() || b.pos.length_sq() > 1.0e8 {
                self.diverged = true;
                return;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Self-collision
    // -----------------------------------------------------------------------

    /// Find places where two of the organism's own parts have run into each
    /// other.
    ///
    /// # Why this exists
    ///
    /// Without it, parts pass straight through one another, and a body has no
    /// reason to be a body: limbs can occupy the torso, two legs can share a
    /// space, and a coherent shape with its limbs on the outside has no
    /// advantage over a cloud of overlapping blocks. Occupying a volume is the
    /// most basic thing an animal does, and it is a *constraint*, not a reward.
    ///
    /// # What it costs
    ///
    /// The solver was built on the assumption that this would never exist (see
    /// the module header), so the cheapest honest version is used: parts are
    /// approximated by capsules, pairs joined by a joint are skipped because
    /// they are meant to touch, and only a normal impulse is solved — no
    /// friction between an organism's own parts. Bodies are few, so the pairing
    /// is quadratic and unapologetic, behind a bounding-sphere reject.
    fn build_pair_contacts(&mut self) {
        self.pair_contacts.clear();
        if !self.params.self_collision {
            return;
        }
        let n = self.bodies.len();
        for a in 0..n {
            for b in (a + 1)..n {
                if self.jointed[a * n + b] {
                    continue;
                }
                let (a0, a1, ra) = self.bodies[a].collision_segment();
                let (b0, b1, rb) = self.bodies[b].collision_segment();

                // Cheap reject before the closest-point work.
                let gap = self.bodies[a].pos - self.bodies[b].pos;
                let reach = self.bodies[a].bounding_radius() + self.bodies[b].bounding_radius();
                if gap.length_sq() > reach * reach {
                    continue;
                }

                let (pa, pb) = closest_points_on_segments(a0, a1, b0, b1);
                let delta = pb - pa;
                let distance = delta.length();
                let touching = ra + rb;
                if distance >= touching {
                    continue;
                }
                // Coincident centres give no direction to push apart along;
                // any consistent one will do, and the position correction will
                // separate them over the next few steps.
                let normal = if distance > 1e-6 { delta * (1.0 / distance) } else { Vec3::Y };
                let depth = touching - distance;

                let contact_a = pa + normal * ra;
                let contact_b = pb - normal * rb;
                let midpoint = (contact_a + contact_b) * 0.5;
                let r_a = midpoint - self.bodies[a].pos;
                let r_b = midpoint - self.bodies[b].pos;

                let k = pair_effective_mass(
                    &self.bodies[a],
                    &self.bodies[b],
                    &self.inv_inertia[a],
                    &self.inv_inertia[b],
                    r_a,
                    r_b,
                    normal,
                );
                if k <= 0.0 {
                    continue;
                }
                self.pair_contacts.push(PairContact {
                    a: a as u16,
                    b: b as u16,
                    r_a,
                    r_b,
                    normal,
                    depth,
                    k_n: k,
                    pn: 0.0,
                });
            }
        }
    }

    /// Push interpenetrating parts apart. Normal impulse only: friction between
    /// an organism's own limbs would be a second-order effect on top of a
    /// first-order approximation.
    fn solve_pair_contacts(&mut self, dt: Real) {
        let inv_dt = 1.0 / dt;
        let beta = self.params.baumgarte;
        let slop = self.params.slop;
        let max_corr = self.params.max_correction_speed;

        for i in 0..self.pair_contacts.len() {
            let c = self.pair_contacts[i];
            let ia = c.a as usize;
            let ib = c.b as usize;
            let inv_ia = self.inv_inertia[ia];
            let inv_ib = self.inv_inertia[ib];

            let relative =
                self.bodies[ib].point_velocity(c.r_b) - self.bodies[ia].point_velocity(c.r_a);
            let vn = relative.dot(c.normal);
            let bias = clamp((c.depth - slop).max(0.0) * beta * inv_dt, 0.0, max_corr);

            // The normal points from a to b, so separating means vn > 0.
            let mut lambda = (bias - vn) / c.k_n;
            let old = c.pn;
            let new = (old + lambda).max(0.0);
            lambda = new - old;
            self.pair_contacts[i].pn = new;
            if lambda == 0.0 {
                continue;
            }
            let impulse = c.normal * lambda;
            self.bodies[ia].apply_impulse(c.r_a, -impulse, &inv_ia);
            self.bodies[ib].apply_impulse(c.r_b, impulse, &inv_ib);
        }
    }

    // -----------------------------------------------------------------------
    // Contacts
    // -----------------------------------------------------------------------

    fn build_contacts(&mut self) {
        self.contacts.clear();
        let terrain = self.params.terrain;
        let restitution = self.params.restitution;

        for (bi, body) in self.bodies.iter().enumerate() {
            let inv_i = &self.inv_inertia[bi];
            // Which points of a curved shape are candidates depends on the
            // ground normal, so ask the terrain first. Under the body's centre
            // is close enough: a shape is small relative to any terrain feature
            // we intend to support, and the per-point height below is still
            // sampled exactly.
            let under = terrain.normal_at(body.pos.x, body.pos.z);
            let (points, count) = body.ground_points(under);
            for &corner in &points[..count] {
                let ground = terrain.height_at(corner.x, corner.z);
                let depth = ground - corner.y;
                if depth <= 0.0 {
                    continue;
                }
                let normal = terrain.normal_at(corner.x, corner.z);
                let tangent1 = normal.any_perpendicular();
                let tangent2 = normal.cross(tangent1);
                let r = corner - body.pos;

                let vn = body.point_velocity(r).dot(normal);
                // Only meaningful impacts bounce; otherwise resting contacts
                // would jitter forever.
                let bounce = if vn < -1.0 { -restitution * vn } else { 0.0 };

                self.contacts.push(Contact {
                    body: bi as u16,
                    r,
                    normal,
                    tangent1,
                    tangent2,
                    depth,
                    k_n: effective_mass(body.inv_mass, inv_i, r, normal),
                    k_t1: effective_mass(body.inv_mass, inv_i, r, tangent1),
                    k_t2: effective_mass(body.inv_mass, inv_i, r, tangent2),
                    pn: 0.0,
                    pt1: 0.0,
                    pt2: 0.0,
                    bounce,
                });
            }
        }
    }

    fn solve_contacts(&mut self, dt: Real) {
        let inv_dt = 1.0 / dt;
        let beta = self.params.baumgarte;
        let slop = self.params.slop;
        let max_corr = self.params.max_correction_speed;
        let mu = self.params.friction;

        for ci in 0..self.contacts.len() {
            let c = self.contacts[ci];
            let bi = c.body as usize;
            let inv_i = self.inv_inertia[bi];

            // Normal.
            let correction = clamp(beta * (c.depth - slop).max(0.0) * inv_dt, 0.0, max_corr);
            let target = correction + c.bounce;
            let vn = self.bodies[bi].point_velocity(c.r).dot(c.normal);
            let mut lambda = (target - vn) / c.k_n;
            let new_pn = (c.pn + lambda).max(0.0);
            lambda = new_pn - c.pn;
            self.contacts[ci].pn = new_pn;
            if lambda != 0.0 {
                let p = c.normal * lambda;
                self.bodies[bi].apply_impulse(c.r, p, &inv_i);
            }

            // Friction, clamped to the Coulomb cone around the normal impulse
            // accumulated so far.
            let limit = mu * new_pn;
            for (tangent, k, stored) in [(c.tangent1, c.k_t1, 1usize), (c.tangent2, c.k_t2, 2usize)]
            {
                let old = if stored == 1 { self.contacts[ci].pt1 } else { self.contacts[ci].pt2 };
                let vt = self.bodies[bi].point_velocity(c.r).dot(tangent);
                let new = clamp(old - vt / k, -limit, limit);
                let delta = new - old;
                if stored == 1 {
                    self.contacts[ci].pt1 = new;
                } else {
                    self.contacts[ci].pt2 = new;
                }
                if delta != 0.0 {
                    self.bodies[bi].apply_impulse(c.r, tangent * delta, &inv_i);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Joints
    // -----------------------------------------------------------------------

    fn prepare_joints(&mut self, _dt: Real) {
        for (i, j) in self.joints.iter().enumerate() {
            let a = &self.bodies[j.body_a as usize];
            let b = &self.bodies[j.body_b as usize];
            let inv_ia = self.inv_inertia[j.body_a as usize];
            let inv_ib = self.inv_inertia[j.body_b as usize];

            let ra = a.orient.rotate(j.anchor_a);
            let rb = b.orient.rotate(j.anchor_b);

            // K = (ima + imb) I - [ra] Ia [ra] - [rb] Ib [rb]
            let sa = Mat3::skew(ra);
            let sb = Mat3::skew(rb);
            let mass_term = Mat3::diagonal(Vec3::splat(a.inv_mass + b.inv_mass));
            let k_point = mass_term
                .sub(&sa.mul_mat(&inv_ia).mul_mat(&sa))
                .sub(&sb.mul_mat(&inv_ib).mul_mat(&sb));

            let k_ang = inv_ia.add(&inv_ib);

            let axis_w = a.orient.rotate(j.axis_a).normalize_or(Vec3::X);
            let perp1 = axis_w.any_perpendicular();
            let perp2 = axis_w.cross(perp1);

            self.prep[i] = JointPrep {
                ra,
                rb,
                axis_w,
                perp1,
                perp2,
                k_point,
                k_ang,
                k_axis: axis_w.dot(k_ang.mul_vec(axis_w)),
                k_perp1: perp1.dot(k_ang.mul_vec(perp1)),
                k_perp2: perp2.dot(k_ang.mul_vec(perp2)),
                motor_impulse: 0.0,
            };
        }
    }

    fn solve_joints(&mut self, dt: Real) {
        let inv_dt = 1.0 / dt;
        let beta = self.params.baumgarte;
        let max_corr = self.params.max_correction_speed;

        for i in 0..self.joints.len() {
            let j = self.joints[i];
            if j.broken {
                continue;
            }
            let p = self.prep[i];
            let ia = j.body_a as usize;
            let ib = j.body_b as usize;
            let inv_ia = self.inv_inertia[ia];
            let inv_ib = self.inv_inertia[ib];

            // --- Point-to-point: the anchors must coincide. ---
            let anchor_a = self.bodies[ia].pos + p.ra;
            let anchor_b = self.bodies[ib].pos + p.rb;
            let error = anchor_b - anchor_a;
            let mut bias = error * (-beta * inv_dt);
            clamp_speed(&mut bias, max_corr);

            let v_rel = self.bodies[ib].point_velocity(p.rb) - self.bodies[ia].point_velocity(p.ra);
            let impulse = p.k_point.solve(bias - v_rel);
            self.bodies[ia].apply_impulse(p.ra, -impulse, &inv_ia);
            self.bodies[ib].apply_impulse(p.rb, impulse, &inv_ib);

            // --- Angular. ---
            match j.kind {
                JointKind::Fixed => {
                    // Drive the relative rotation back to the rest pose. Bodies
                    // start axis-aligned, so the rest relative rotation is the
                    // identity and the error is just the relative quaternion.
                    let q_rel = self.bodies[ib].orient.mul(self.bodies[ia].orient.conjugate());
                    let sign = if q_rel.w < 0.0 { -1.0 } else { 1.0 };
                    let err = q_rel.vec() * (2.0 * sign);
                    let mut ang_bias = err * (-beta * inv_dt);
                    clamp_speed(&mut ang_bias, max_corr * 4.0);

                    let w_rel = self.bodies[ib].ang_vel - self.bodies[ia].ang_vel;
                    let ang_impulse = p.k_ang.solve(ang_bias - w_rel);
                    self.bodies[ia].apply_angular_impulse(-ang_impulse, &inv_ia);
                    self.bodies[ib].apply_angular_impulse(ang_impulse, &inv_ib);
                }
                JointKind::Hinge => {
                    // Remove the two rotational degrees of freedom that are not
                    // about the hinge axis, leaving exactly one free.
                    let axis_b_w = self.bodies[ib].orient.rotate(j.axis_b);
                    let misalign = p.axis_w.cross(axis_b_w);
                    for (t, k) in [(p.perp1, p.k_perp1), (p.perp2, p.k_perp2)] {
                        if k <= 0.0 {
                            continue;
                        }
                        let w_rel = self.bodies[ib].ang_vel - self.bodies[ia].ang_vel;
                        let target = clamp(
                            -beta * inv_dt * misalign.dot(t),
                            -max_corr * 4.0,
                            max_corr * 4.0,
                        );
                        let lambda = (target - w_rel.dot(t)) / k;
                        let imp = t * lambda;
                        self.bodies[ia].apply_angular_impulse(-imp, &inv_ia);
                        self.bodies[ib].apply_angular_impulse(imp, &inv_ib);
                    }

                    self.solve_hinge_limit(i, dt);
                    self.solve_hinge_motor(i, dt);
                }
            }
        }
    }

    fn solve_hinge_limit(&mut self, i: usize, dt: Real) {
        let j = self.joints[i];
        let p = self.prep[i];
        if p.k_axis <= 0.0 {
            return;
        }
        let (cos_theta, sin_theta) = self.hinge_angle_cos_sin(i);
        if cos_theta >= j.cos_limit {
            return; // inside the allowed range
        }

        let ia = j.body_a as usize;
        let ib = j.body_b as usize;
        let inv_ia = self.inv_inertia[ia];
        let inv_ib = self.inv_inertia[ib];

        // Sign of the direction in which the joint is over-rotated.
        let dir = if sin_theta >= 0.0 { 1.0 } else { -1.0 };
        // Overshoot measured in cosine rather than radians: monotone in |theta|
        // over the half-turn a hinge limit can occupy, and free of `acos`.
        let overshoot = j.cos_limit - cos_theta;
        let push_back = -clamp(
            self.params.baumgarte * overshoot / dt,
            0.0,
            self.params.max_correction_speed * 4.0,
        );

        // Relative rotation rate in the violating direction.
        let w_rel = self.bodies[ib].ang_vel - self.bodies[ia].ang_vel;
        let rate = dir * w_rel.dot(p.axis_w);
        if rate <= push_back {
            return; // already recovering fast enough
        }
        let lambda = (push_back - rate) / p.k_axis;
        let imp = p.axis_w * (lambda * dir);
        self.bodies[ia].apply_angular_impulse(-imp, &inv_ia);
        self.bodies[ib].apply_angular_impulse(imp, &inv_ib);
    }

    /// A passive spring and damper across the hinge: a tendon.
    ///
    /// Animals do not move by servo. A great deal of what makes running and
    /// hopping efficient is elastic: tendons store energy on landing and return
    /// it on push-off, so the muscle does not have to pay for the whole stride.
    /// Without any passive element, every joule of a gait has to come out of the
    /// motor, which is why evolved gaits here look so unlike animal ones.
    ///
    /// The restoring torque uses `sin(angle)` rather than the angle itself. It
    /// is monotone over the whole legal range — [`crate::config`] refuses a
    /// joint limit at or beyond a quarter turn — costs no `atan2`, and is
    /// already computed for the controller's benefit.
    fn apply_tendons(&mut self, dt: Real) {
        for i in 0..self.joints.len() {
            let j = self.joints[i];
            if j.broken || j.kind != JointKind::Hinge || j.tendon_frequency <= 0.0 {
                continue;
            }
            let k = self.prep[i].k_axis;
            if k <= 0.0 {
                continue;
            }
            let axis = self.prep[i].axis_w;
            let (_, sin_a) = self.hinge_angle_cos_sin(i);
            let ia = j.body_a as usize;
            let ib = j.body_b as usize;
            let rate = (self.bodies[ib].ang_vel - self.bodies[ia].ang_vel).dot(axis);

            // The change in relative rate a spring of this frequency asks for
            // over one step. Working in rate rather than torque is what makes it
            // independent of the limb's inertia, and therefore stable.
            let w = j.tendon_frequency;
            let spring = -w * w * sin_a * dt;
            // The damper may remove the joint's motion but never reverse it,
            // which is the difference between damping and driving.
            let bleed = clamp(2.0 * j.tendon_damping * w * dt, 0.0, 1.0);
            let delta_rate = spring - bleed * rate;

            let lambda = delta_rate / k;
            if lambda == 0.0 {
                continue;
            }
            let inv_ia = self.inv_inertia[ia];
            let inv_ib = self.inv_inertia[ib];
            let imp = axis * lambda;
            self.bodies[ia].apply_angular_impulse(-imp, &inv_ia);
            self.bodies[ib].apply_angular_impulse(imp, &inv_ib);
        }
    }

    fn solve_hinge_motor(&mut self, i: usize, dt: Real) {
        let j = self.joints[i];
        let p = self.prep[i];
        if p.k_axis <= 0.0 || j.motor_torque_max <= 0.0 {
            return;
        }
        let ia = j.body_a as usize;
        let ib = j.body_b as usize;
        let inv_ia = self.inv_inertia[ia];
        let inv_ib = self.inv_inertia[ib];

        let target = clamp(j.motor_target, -j.motor_speed_max, j.motor_speed_max);
        let w_rel = self.bodies[ib].ang_vel - self.bodies[ia].ang_vel;
        let current = w_rel.dot(p.axis_w);
        let desired = (target - current) / p.k_axis;

        // Accumulate so the torque budget applies to the whole step rather than
        // being granted afresh on every solver iteration.
        let max_impulse = j.motor_torque_max * dt;
        let old = p.motor_impulse;
        let new = clamp(old + desired, -max_impulse, max_impulse);
        let lambda = new - old;
        self.prep[i].motor_impulse = new;
        if lambda == 0.0 {
            return;
        }
        let imp = p.axis_w * lambda;
        self.bodies[ia].apply_angular_impulse(-imp, &inv_ia);
        self.bodies[ib].apply_angular_impulse(imp, &inv_ib);
        self.actuation_impulse += lambda.abs();
    }
}

#[inline]
fn project_out(v: Vec3, axis: Vec3) -> Vec3 {
    v - axis * v.dot(axis)
}

#[inline]
fn clamp_speed(v: &mut Vec3, max: Real) {
    let len_sq = v.length_sq();
    if len_sq > max * max {
        *v = *v * (max / len_sq.sqrt());
    }
}

/// Scalar effective mass for a single dynamic body constrained along `dir` at
/// world offset `r`: `1 / (m^-1 + dir . ((I^-1 (r x dir)) x r))`.
#[inline]
/// Effective mass of a contact between two moving bodies along `dir`.
fn pair_effective_mass(
    a: &RigidBody,
    b: &RigidBody,
    inv_ia: &Mat3,
    inv_ib: &Mat3,
    ra: Vec3,
    rb: Vec3,
    dir: Vec3,
) -> Real {
    let ta = ra.cross(dir);
    let tb = rb.cross(dir);
    a.inv_mass + b.inv_mass + ta.dot(inv_ia.mul_vec(ta)) + tb.dot(inv_ib.mul_vec(tb))
}

/// Closest points on two segments, one on each.
///
/// The standard clamped-parameter solution: solve the unconstrained least
/// squares for the two line parameters, then clamp each to its segment and
/// re-solve the other against the clamped value. Degenerate segments — a sphere
/// stands in as a zero-length one — fall out of the same arithmetic.
fn closest_points_on_segments(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> (Vec3, Vec3) {
    let da = a1 - a0;
    let db = b1 - b0;
    let r = a0 - b0;
    let aa = da.dot(da);
    let bb = db.dot(db);
    let f = db.dot(r);

    const EPS: Real = 1e-12;
    let (mut s, mut t);
    if aa <= EPS && bb <= EPS {
        return (a0, b0);
    }
    if aa <= EPS {
        s = 0.0;
        t = clamp(f / bb, 0.0, 1.0);
    } else {
        let c = da.dot(r);
        if bb <= EPS {
            t = 0.0;
            s = clamp(-c / aa, 0.0, 1.0);
        } else {
            let d = da.dot(db);
            let denom = aa * bb - d * d;
            s = if denom > EPS { clamp((d * f - c * bb) / denom, 0.0, 1.0) } else { 0.0 };
            t = (d * s + f) / bb;
            if t < 0.0 {
                t = 0.0;
                s = clamp(-c / aa, 0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = clamp((d - c) / aa, 0.0, 1.0);
            }
        }
    }
    (a0 + da * s, b0 + db * t)
}

fn effective_mass(inv_mass: Real, inv_inertia: &Mat3, r: Vec3, dir: Vec3) -> Real {
    let rn = r.cross(dir);
    let term = inv_inertia.mul_vec(rn).cross(r).dot(dir);
    let k = inv_mass + term;
    if k > 1e-12 {
        k
    } else {
        1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_params() -> WorldParams {
        WorldParams::default()
    }

    fn drop_box(height: Real) -> World {
        let body = RigidBody::box_body(vec3(0.0, height, 0.0), vec3(0.25, 0.25, 0.25), 250.0);
        World::new(vec![body], vec![], flat_params())
    }

    #[test]
    fn a_dropped_box_falls_and_comes_to_rest_on_the_ground() {
        let mut w = drop_box(2.0);
        let dt = 1.0 / 120.0;
        for _ in 0..600 {
            w.step(dt);
        }
        assert!(!w.diverged);
        let b = &w.bodies[0];
        // Resting on a face: centre sits one half-extent above the ground.
        assert!((b.pos.y - 0.25).abs() < 0.02, "settled at y = {}", b.pos.y);
        assert!(b.lin_vel.length() < 0.05, "still moving: {:?}", b.lin_vel);
    }

    #[test]
    fn free_fall_matches_analytic_solution() {
        let mut w = drop_box(100.0);
        w.params = WorldParams { linear_damping: 0.0, ..WorldParams::default() };
        let dt = 1.0 / 240.0;
        let n = 240;
        for _ in 0..n {
            w.step(dt);
        }
        let t = n as Real * dt;
        let expected = 100.0 - 0.5 * 9.81 * t * t;
        // Semi-implicit Euler overshoots by exactly g*dt*t/2; allow for it.
        assert!(
            (w.bodies[0].pos.y - expected).abs() < 0.05,
            "y = {}, expected ~{expected}",
            w.bodies[0].pos.y
        );
    }

    #[test]
    fn friction_stops_a_sliding_box() {
        let mut w = drop_box(0.25);
        w.bodies[0].lin_vel = vec3(4.0, 0.0, 0.0);
        let dt = 1.0 / 120.0;
        for _ in 0..600 {
            w.step(dt);
        }
        assert!(w.bodies[0].lin_vel.x.abs() < 0.1, "vx = {}", w.bodies[0].lin_vel.x);
        assert!(w.bodies[0].pos.x > 0.1, "it should have slid some distance first");
    }

    #[test]
    fn frictionless_box_keeps_sliding() {
        let mut w = drop_box(0.25);
        w.params.friction = 0.0;
        w.params.linear_damping = 0.0;
        w.bodies[0].lin_vel = vec3(4.0, 0.0, 0.0);
        let dt = 1.0 / 120.0;
        for _ in 0..240 {
            w.step(dt);
        }
        assert!(w.bodies[0].lin_vel.x > 3.5, "vx = {}", w.bodies[0].lin_vel.x);
    }

    /// Two boxes welded together must behave as one rigid object.
    #[test]
    fn a_fixed_joint_holds_bodies_together() {
        let a = RigidBody::box_body(vec3(0.0, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        let b = RigidBody::box_body(vec3(0.4, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        let joint = Joint::fixed(0, 1, vec3(0.2, 0.0, 0.0), vec3(-0.2, 0.0, 0.0));
        let mut w = World::new(vec![a, b], vec![joint], flat_params());
        let dt = 1.0 / 120.0;
        for _ in 0..900 {
            w.step(dt);
        }
        assert!(!w.diverged);
        let separation = (w.bodies[1].pos - w.bodies[0].pos).length();
        assert!((separation - 0.4).abs() < 0.02, "separation drifted to {separation}");
        // A weld also holds orientation.
        let q_rel = w.bodies[1].orient.mul(w.bodies[0].orient.conjugate());
        assert!(q_rel.vec().length() < 0.05, "relative rotation {q_rel:?}");
    }

    fn hinge_pair(limit_cos: Real, torque: Real) -> World {
        let a = RigidBody::box_body(vec3(0.0, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        let b = RigidBody::box_body(vec3(0.4, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        let joint = Joint {
            body_a: 0,
            body_b: 1,
            kind: JointKind::Hinge,
            anchor_a: vec3(0.2, 0.0, 0.0),
            anchor_b: vec3(-0.2, 0.0, 0.0),
            axis_a: Vec3::Z,
            axis_b: Vec3::Z,
            ref_a: Vec3::Y,
            ref_b: Vec3::Y,
            cos_limit: limit_cos,
            motor_speed_max: 4.0,
            motor_torque_max: torque,
            motor_target: 0.0,
            tendon_frequency: 0.0,
            tendon_damping: 0.0,
            endurance: 0.0,
            health: 0.0,
            broken: false,
        };
        let mut p = flat_params();
        p.gravity = Vec3::ZERO; // isolate joint behaviour from falling
        World::new(vec![a, b], vec![joint], p)
    }

    #[test]
    fn a_hinge_motor_rotates_the_child() {
        let mut w = hinge_pair(-1.0, 200.0);
        w.joints[0].motor_target = 3.0;
        let dt = 1.0 / 240.0;
        for _ in 0..240 {
            w.step(dt);
        }
        assert!(!w.diverged);
        let (cos_t, _) = w.hinge_angle_cos_sin(0);
        assert!(cos_t < 0.9, "hinge barely moved, cos = {cos_t}");
        // The anchors must still coincide.
        let anchor_a = w.bodies[0].pos + w.bodies[0].orient.rotate(vec3(0.2, 0.0, 0.0));
        let anchor_b = w.bodies[1].pos + w.bodies[1].orient.rotate(vec3(-0.2, 0.0, 0.0));
        assert!((anchor_a - anchor_b).length() < 0.02);
    }

    /// A velocity-target motor has to brake as well as drive.
    ///
    /// This exists because an evolved organism was found riding a wheel that
    /// turned at 9.8 rad/s across a joint whose motor was capped at 6.0 — which
    /// is legitimate only if the *ground* is spinning the wheel and the motor is
    /// merely losing the argument. If instead the motor were one-directional,
    /// any joint could be spun up for free and every fast organism in the
    /// repository would be an artefact. With no contacts and no gravity there is
    /// nothing to sustain the overspeed, so the motor must pull it back to
    /// target on its own.
    #[test]
    fn a_hinge_motor_brakes_a_joint_spun_past_its_target() {
        let mut w = hinge_pair(-1.0, 200.0);
        w.joints[0].motor_target = 2.0;

        // Spin the child far beyond what the motor would ever drive.
        let overspeed = 20.0;
        w.bodies[1].ang_vel = Vec3::Z * overspeed;

        let dt = 1.0 / 240.0;
        for _ in 0..480 {
            w.step(dt);
        }
        assert!(!w.diverged);

        let rel = (w.bodies[1].ang_vel - w.bodies[0].ang_vel).dot(Vec3::Z);
        assert!(
            rel < 2.5,
            "motor did not brake an overspeeding joint: {rel} rad/s against a target of 2.0"
        );
        // And it brakes *to* the target rather than through it to a standstill.
        assert!(rel > 1.5, "motor overshot its target and stalled the joint: {rel} rad/s");
    }

    /// The other half: a motor may not drive a free joint past its own cap, so
    /// the speed limit means something in the absence of outside help.
    #[test]
    fn a_hinge_motor_does_not_exceed_its_speed_cap() {
        let mut w = hinge_pair(-1.0, 200.0);
        // `motor_speed_max` is 4.0 in the fixture; ask for far more.
        w.joints[0].motor_target = 50.0;
        let dt = 1.0 / 240.0;
        for _ in 0..480 {
            w.step(dt);
        }
        assert!(!w.diverged);
        let rel = (w.bodies[1].ang_vel - w.bodies[0].ang_vel).dot(Vec3::Z);
        assert!(rel <= 4.5, "motor drove past its own speed cap: {rel} rad/s against 4.0");
    }

    /// A motor asked for more than its torque can deliver wears its joint out,
    /// and when the joint fails the limb stops being part of the organism.
    #[test]
    fn an_overworked_joint_breaks_and_sheds_its_limb() {
        let mut w = hinge_pair(-1.0, 0.5); // a very weak motor
        w.joints[0].endurance = 1.0;
        w.joints[0].health = 1.0;
        w.joints[0].motor_target = 4.0; // far beyond what 0.5 N m can achieve
        assert!(w.is_attached(1));

        let dt = 1.0 / 240.0;
        for _ in 0..240 {
            w.step(dt);
        }
        assert!(!w.diverged);
        assert!(w.joints[0].broken, "joint survived with health {}", w.joints[0].health);
        assert!(!w.is_attached(1), "the limb is still counted as part of the organism");
        assert_eq!(w.breaks.len(), 1);

        // Fitness measures only what is still attached, so where the wreckage
        // goes is no longer any of its business. (The reported centre of mass is
        // not the root's raw position: the shift caused by dropping the limb out
        // of the average is deliberately cancelled — see `centre_of_mass`.)
        let before = w.centre_of_mass();
        w.bodies[1].pos = vec3(500.0, -400.0, 300.0);
        let after = w.centre_of_mass();
        assert!(
            (after - before).length() < 1e-5,
            "debris still moves the organism's measured position: {before:?} -> {after:?}"
        );
    }

    /// The same joint, driven just as hard, is indestructible when the
    /// experiment has not enabled wear. This is the switch every other
    /// experiment in the repository is sitting on.
    #[test]
    fn a_joint_with_no_endurance_never_wears_out() {
        let mut w = hinge_pair(-1.0, 0.5);
        w.joints[0].motor_target = 4.0;
        let dt = 1.0 / 240.0;
        for _ in 0..480 {
            w.step(dt);
        }
        assert!(!w.joints[0].broken);
        assert!(w.is_attached(1));
        assert!(w.breaks.is_empty());
    }

    /// A motor working within its means costs its joint nothing, so wear is a
    /// charge for overreach rather than for being used at all.
    #[test]
    fn a_joint_driven_within_its_torque_takes_no_damage() {
        let mut w = hinge_pair(-1.0, 400.0); // plenty of torque
        w.joints[0].endurance = 1.0;
        w.joints[0].health = 1.0;
        w.joints[0].motor_target = 1.0;
        let dt = 1.0 / 240.0;
        for _ in 0..480 {
            w.step(dt);
        }
        assert!(!w.joints[0].broken);
        assert!(
            w.joints[0].health > 0.99,
            "an unstressed joint lost health: {}",
            w.joints[0].health
        );
    }

    /// Shedding a limb must be worth exactly zero metres.
    ///
    /// Dropping a body out of an average moves that average for free, and
    /// `distance_x` is measured from that average. Without the correction in
    /// [`World::centre_of_mass`] an organism could collect real fitness by
    /// discarding a trailing part — which is what a run measured before this
    /// test existed actually did, to the tune of a third of one organism's
    /// recorded distance.
    #[test]
    fn detaching_a_limb_does_not_move_the_measured_centre_of_mass() {
        let mut w = hinge_pair(-1.0, 0.5);
        // Put the limb well to one side, so dropping it would shift the mean a
        // long way if the shift were not cancelled.
        w.bodies[1].pos = vec3(4.0, 3.0, 0.0);
        w.joints[0].endurance = 1.0;
        w.joints[0].health = 1.0;
        w.joints[0].motor_target = 4.0;

        let dt = 1.0 / 240.0;
        let mut previous = w.centre_of_mass();
        let mut worst_step = 0.0;
        let mut broke = false;
        for _ in 0..240 {
            w.step(dt);
            let com = w.centre_of_mass();
            worst_step = (com - previous).length().max(worst_step);
            previous = com;
            broke |= w.joints[0].broken;
        }
        assert!(broke, "the joint never failed, so nothing was tested");
        // Bodies move a little each step under their own momentum; a teleport
        // would be an order of magnitude larger than that.
        assert!(
            worst_step < 0.05,
            "the centre of mass jumped {worst_step} m in one step when the limb came off"
        );
    }

    /// Breaking one joint has to cut loose everything hanging below it, not just
    /// the body immediately attached.
    #[test]
    fn breaking_a_joint_detaches_the_whole_subtree() {
        let mut w = hinge_pair(-1.0, 0.5);
        // Extend the chain: a third body welded to the second.
        let c = RigidBody::box_body(vec3(0.8, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        w.bodies.push(c);
        w.joints.push(Joint::fixed(1, 2, vec3(0.2, 0.0, 0.0), vec3(-0.2, 0.0, 0.0)));
        let mut rebuilt = World::new(w.bodies.clone(), w.joints.clone(), w.params);
        rebuilt.joints[0].endurance = 1.0;
        rebuilt.joints[0].health = 1.0;
        rebuilt.joints[0].motor_target = 4.0;

        let dt = 1.0 / 240.0;
        for _ in 0..240 {
            rebuilt.step(dt);
        }
        assert!(rebuilt.joints[0].broken);
        assert!(!rebuilt.is_attached(1), "the limb is still attached");
        assert!(!rebuilt.is_attached(2), "the limb's own child is still attached");
        assert!(rebuilt.is_attached(0), "the root can never detach");
    }

    /// Two unjointed parts placed on top of each other must push apart.
    #[test]
    fn overlapping_parts_separate_when_self_collision_is_on() {
        let mut p = flat_params();
        p.gravity = Vec3::ZERO;
        p.self_collision = true;
        let a = RigidBody::box_body(vec3(0.0, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        // Deliberately overlapping, and with no joint between them.
        let b = RigidBody::box_body(vec3(0.12, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        let mut w = World::new(vec![a, b], vec![], p);

        let before = (w.bodies[0].pos - w.bodies[1].pos).length();
        for _ in 0..240 {
            w.step(1.0 / 240.0);
        }
        assert!(!w.diverged);
        let after = (w.bodies[0].pos - w.bodies[1].pos).length();
        assert!(after > before + 0.05, "parts did not separate: {before} -> {after}");
    }

    /// With it off they pass straight through, which is the behaviour every
    /// experiment before this relied on.
    #[test]
    fn overlapping_parts_are_ignored_when_self_collision_is_off() {
        let mut p = flat_params();
        p.gravity = Vec3::ZERO;
        let a = RigidBody::box_body(vec3(0.0, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        let b = RigidBody::box_body(vec3(0.12, 3.0, 0.0), vec3(0.2, 0.2, 0.2), 250.0);
        let mut w = World::new(vec![a, b], vec![], p);
        let before = (w.bodies[0].pos - w.bodies[1].pos).length();
        for _ in 0..240 {
            w.step(1.0 / 240.0);
        }
        let after = (w.bodies[0].pos - w.bodies[1].pos).length();
        assert!((after - before).abs() < 1e-4, "parts moved: {before} -> {after}");
    }

    /// Parts joined by a joint are *meant* to touch. If self-collision fought
    /// the joint holding them together, every organism would tear itself apart.
    #[test]
    fn jointed_parts_do_not_collide_with_each_other() {
        let mut w = hinge_pair(-1.0, 0.0);
        w.params.self_collision = true;
        let dt = 1.0 / 240.0;
        for _ in 0..240 {
            w.step(dt);
        }
        assert!(!w.diverged);
        // The anchors must still coincide: nothing pushed them apart.
        let anchor_a = w.bodies[0].pos + w.bodies[0].orient.rotate(vec3(0.2, 0.0, 0.0));
        let anchor_b = w.bodies[1].pos + w.bodies[1].orient.rotate(vec3(-0.2, 0.0, 0.0));
        assert!(
            (anchor_a - anchor_b).length() < 0.02,
            "self-collision pulled a joint apart by {}",
            (anchor_a - anchor_b).length()
        );
    }

    /// The closest-point routine underpins every self-collision test above, and
    /// its clamping is exactly the part that is easy to get wrong.
    #[test]
    fn closest_points_handles_parallel_crossing_and_degenerate_segments() {
        // Parallel, overlapping in their shared direction.
        let (a, b) = closest_points_on_segments(
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.25, 1.0, 0.0),
            vec3(0.75, 1.0, 0.0),
        );
        assert!((a.y - b.y).abs() > 0.9 && (a - b).length() - 1.0 < 1e-5);

        // Crossing at right angles: the closest points are where they cross.
        let (a, b) = closest_points_on_segments(
            vec3(-1.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 0.5, -1.0),
            vec3(0.0, 0.5, 1.0),
        );
        assert!(a.length() < 1e-5, "{a:?}");
        assert!((b - vec3(0.0, 0.5, 0.0)).length() < 1e-5, "{b:?}");

        // A point against a segment, and two points: a sphere is a segment of
        // zero length, so both have to work.
        let (a, b) = closest_points_on_segments(
            vec3(0.4, 2.0, 0.0),
            vec3(0.4, 2.0, 0.0),
            vec3(0.0, 0.0, 0.0),
            vec3(1.0, 0.0, 0.0),
        );
        assert!((a - vec3(0.4, 2.0, 0.0)).length() < 1e-5);
        assert!((b - vec3(0.4, 0.0, 0.0)).length() < 1e-5, "{b:?}");

        let (a, b) = closest_points_on_segments(
            vec3(1.0, 1.0, 1.0),
            vec3(1.0, 1.0, 1.0),
            vec3(-2.0, 0.0, 0.0),
            vec3(-2.0, 0.0, 0.0),
        );
        assert!((a - vec3(1.0, 1.0, 1.0)).length() < 1e-6);
        assert!((b - vec3(-2.0, 0.0, 0.0)).length() < 1e-6);
    }

    /// A tendon is passive: it may store and return energy, and it may lose it,
    /// but it must never create any.
    ///
    /// This exists because the first version did. Applied as an explicit torque
    /// impulse, the damping term inverted for light limbs — `damping * dt /
    /// inertia` above 2 amplifies instead of damping — and evolution found it
    /// within a dozen generations, producing organisms crossing a hundred metres
    /// in eight seconds. Expressing the spring as a frequency rather than a
    /// stiffness is what makes it independent of the limb it acts on.
    #[test]
    fn a_tendon_never_adds_energy() {
        for freq in [2.0, 6.0, 20.0, 55.0] {
            for damping in [0.0, 0.5, 1.0] {
                let mut w = hinge_pair(-1.0, 0.0); // no motor at all
                w.joints[0].tendon_frequency = freq;
                w.joints[0].tendon_damping = damping;
                // Set it swinging, then leave it alone.
                w.bodies[1].ang_vel = Vec3::Z * 3.0;

                let dt = 1.0 / 120.0;
                let energy = |w: &World| -> Real {
                    w.bodies
                        .iter()
                        .map(|b| {
                            let i = 1.0 / b.inv_inertia_local.z;
                            0.5 * b.mass() * b.lin_vel.length_sq() + 0.5 * i * b.ang_vel.length_sq()
                        })
                        .sum()
                };
                let start = energy(&w);
                let mut peak: Real = start;
                for _ in 0..600 {
                    w.step(dt);
                    peak = peak.max(energy(&w));
                }
                assert!(!w.diverged, "freq {freq} damping {damping}: diverged");
                // A spring converts kinetic energy to potential and back, so the
                // kinetic peak may exceed the start a little; it may not run away.
                assert!(
                    peak < start * 3.0,
                    "freq {freq} damping {damping}: energy grew from {start} to {peak}"
                );
            }
        }
    }

    /// And a damped tendon actually settles the joint rather than leaving it
    /// ringing, which is the half of the behaviour that makes it useful.
    #[test]
    fn a_damped_tendon_brings_a_joint_to_rest() {
        let mut w = hinge_pair(-1.0, 0.0);
        w.joints[0].tendon_frequency = 8.0;
        w.joints[0].tendon_damping = 1.0;
        w.bodies[1].ang_vel = Vec3::Z * 3.0;
        for _ in 0..1200 {
            w.step(1.0 / 120.0);
        }
        let rate = (w.bodies[1].ang_vel - w.bodies[0].ang_vel).length();
        assert!(rate < 0.3, "joint still swinging at {rate} rad/s");
    }

    #[test]
    fn a_hinge_limit_stops_rotation() {
        // cos(0.5 rad) ~ 0.8776
        let mut w = hinge_pair(crate::math::dcos(0.5), 200.0);
        w.joints[0].motor_target = 4.0;
        let dt = 1.0 / 240.0;
        for _ in 0..600 {
            w.step(dt);
        }
        assert!(!w.diverged);
        let (cos_t, _) = w.hinge_angle_cos_sin(0);
        // Allow a little overshoot from the soft constraint, but nothing close to
        // a free spin.
        assert!(cos_t > crate::math::dcos(0.75), "limit breached, cos = {cos_t}");
    }

    #[test]
    fn a_hinge_without_a_motor_stays_where_it_is_put() {
        let mut w = hinge_pair(-1.0, 0.0);
        let dt = 1.0 / 240.0;
        for _ in 0..240 {
            w.step(dt);
        }
        let (cos_t, _) = w.hinge_angle_cos_sin(0);
        assert!(cos_t > 0.999, "drifted to cos = {cos_t}");
    }

    #[test]
    fn stepping_is_bitwise_reproducible() {
        let run = || {
            let mut w = hinge_pair(0.5, 150.0);
            w.params.gravity = vec3(0.0, -9.81, 0.0);
            w.joints[0].motor_target = 2.5;
            for i in 0..500 {
                w.joints[0].motor_target = if i % 100 < 50 { 2.5 } else { -2.5 };
                w.step(1.0 / 120.0);
            }
            (w.bodies[0].pos, w.bodies[1].pos, w.actuation_impulse)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn motor_effort_is_recorded() {
        let mut w = hinge_pair(-1.0, 100.0);
        assert_eq!(w.actuation_impulse, 0.0);
        w.joints[0].motor_target = 3.0;
        for _ in 0..120 {
            w.step(1.0 / 120.0);
        }
        assert!(w.actuation_impulse > 0.0);
    }

    #[test]
    fn divergence_is_detected_rather_than_propagated() {
        let mut w = drop_box(1.0);
        w.bodies[0].lin_vel = vec3(Real::NAN, 0.0, 0.0);
        w.step(1.0 / 120.0);
        assert!(w.diverged);
        // Once diverged, stepping is a no-op rather than a source of further
        // garbage.
        let before = w.bodies[0].pos;
        w.step(1.0 / 120.0);
        assert!(before.x.is_nan() || before.x == w.bodies[0].pos.x);
    }
}
