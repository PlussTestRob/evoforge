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
use crate::math::{clamp, vec3, Mat3, Real, Vec3};

use super::body::RigidBody;

/// Ground model.
///
/// An enum rather than a trait object: there is one variant today, dispatch is
/// in the innermost loop, and a heightfield variant will be a second variant
/// rather than a new abstraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TerrainModel {
    Flat { height: Real },
}

impl TerrainModel {
    #[inline]
    pub fn height_at(&self, _x: Real, _z: Real) -> Real {
        match self {
            TerrainModel::Flat { height } => *height,
        }
    }

    #[inline]
    pub fn normal_at(&self, _x: Real, _z: Real) -> Vec3 {
        match self {
            TerrainModel::Flat { .. } => Vec3::Y,
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
    /// Sum of `|motor angular impulse|` applied so far. A cheap, monotone proxy
    /// for actuation effort, used by energy-penalising fitness functions.
    pub actuation_impulse: Real,
    /// Set once any body leaves the representable range; the evaluation is then
    /// abandoned rather than allowed to produce meaningless fitness.
    pub diverged: bool,
}

impl World {
    pub fn new(bodies: Vec<RigidBody>, joints: Vec<Joint>, params: WorldParams) -> World {
        let n = bodies.len();
        let j = joints.len();
        World {
            bodies,
            joints,
            params,
            inv_inertia: vec![Mat3::ZERO; n],
            prep: vec![JointPrep::default(); j],
            contacts: Vec::with_capacity(n * 8),
            actuation_impulse: 0.0,
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
        self.build_contacts();
        self.prepare_joints(dt);

        for _ in 0..self.params.iterations {
            self.solve_joints(dt);
            self.solve_contacts(dt);
        }

        self.integrate_positions(dt);
        self.check_finite();
    }

    /// Centre of mass of the whole organism.
    pub fn centre_of_mass(&self) -> Vec3 {
        let mut total = 0.0;
        let mut acc = Vec3::ZERO;
        for b in &self.bodies {
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
