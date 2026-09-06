//! Rigid bodies.
//!
//! A body is a single convex primitive — see [`super::shape::Shape`] for which
//! ones and why the set is small. The body itself is deliberately ignorant of
//! geometry beyond delegating three questions to its shape: what it weighs, what
//! could be touching the ground, and what box it fits in.

use crate::math::{vec3, Mat3, Quat, Real, Vec3};

use super::shape::{Shape, MAX_GROUND_POINTS};

/// A rigid body in maximal coordinates.
///
/// Stored flat in [`super::World::bodies`] and referenced by index. There are no
/// back-references, no parent pointers and no `Rc`: the whole world is a couple
/// of contiguous `Vec`s, which is what makes an evaluation cheap to construct,
/// cheap to iterate and trivially `Send`.
#[derive(Clone, Copy, Debug)]
pub struct RigidBody {
    /// World-space centre of mass. For a shape whose centre of mass is not its
    /// geometric centre — a taper — the two differ by
    /// [`Shape::com_offset`], and this is the former.
    pub pos: Vec3,
    pub orient: Quat,
    pub lin_vel: Vec3,
    pub ang_vel: Vec3,
    /// Collision geometry in the body frame.
    pub shape: Shape,
    pub inv_mass: Real,
    /// Diagonal of the inverse inertia tensor in the body frame. Every shape we
    /// admit is symmetric enough for its inertia tensor to be diagonal in its own
    /// frame, so there is no reason to store nine numbers.
    pub inv_inertia_local: Vec3,
}

impl RigidBody {
    /// Create a dynamic body of the given shape and density.
    pub fn new(pos: Vec3, shape: Shape, density: Real) -> RigidBody {
        let (mass, i) = shape.mass_properties(density);
        debug_assert!(mass > 0.0);
        RigidBody {
            pos,
            orient: Quat::IDENTITY,
            lin_vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            shape,
            inv_mass: 1.0 / mass,
            inv_inertia_local: vec3(1.0 / i.x, 1.0 / i.y, 1.0 / i.z),
        }
    }

    /// Create a dynamic box. Kept as a named constructor because boxes are still
    /// what most of the tests want.
    pub fn box_body(pos: Vec3, half_extents: Vec3, density: Real) -> RigidBody {
        RigidBody::new(pos, Shape::Box { half_extents }, density)
    }

    /// Half-extents of the box bounding this body, about its geometric centre.
    #[inline]
    pub fn bounds(&self) -> Vec3 {
        self.shape.bounds()
    }

    #[inline]
    pub fn mass(&self) -> Real {
        1.0 / self.inv_mass
    }

    /// Inverse inertia tensor in world coordinates: `R * I^-1 * R^T`.
    #[inline]
    pub fn inv_inertia_world(&self) -> Mat3 {
        let r = self.orient.to_mat3();
        let rt = r.transpose();
        let scaled = Mat3 {
            cols: [
                rt.cols[0].mul_elem(self.inv_inertia_local),
                rt.cols[1].mul_elem(self.inv_inertia_local),
                rt.cols[2].mul_elem(self.inv_inertia_local),
            ],
        };
        // `scaled` is I^-1 * R^T stored column-wise, so R * that is the result.
        r.mul_mat(&scaled)
    }

    /// Velocity of the material point at world offset `r` from the centre of mass.
    #[inline]
    pub fn point_velocity(&self, r: Vec3) -> Vec3 {
        self.lin_vel + self.ang_vel.cross(r)
    }

    #[inline]
    pub fn apply_impulse(&mut self, r: Vec3, impulse: Vec3, inv_inertia: &Mat3) {
        self.lin_vel += impulse * self.inv_mass;
        self.ang_vel += inv_inertia.mul_vec(r.cross(impulse));
    }

    #[inline]
    pub fn apply_angular_impulse(&mut self, impulse: Vec3, inv_inertia: &Mat3) {
        self.ang_vel += inv_inertia.mul_vec(impulse);
    }

    /// World-space points that may be touching ground whose normal is `normal`.
    pub fn ground_points(&self, normal: Vec3) -> ([Vec3; MAX_GROUND_POINTS], usize) {
        self.shape.ground_points(self.pos, self.orient, normal)
    }

    /// Lowest world-space point of the body. Used to place an organism on the
    /// ground at spawn.
    pub fn lowest_point_y(&self) -> Real {
        let (pts, n) = self.ground_points(Vec3::Y);
        pts[..n].iter().fold(Real::INFINITY, |m, c| m.min(c.y))
    }

    pub fn is_finite(&self) -> bool {
        self.pos.is_finite()
            && self.orient.is_finite()
            && self.lin_vel.is_finite()
            && self.ang_vel.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_and_inertia_are_sane() {
        let b = RigidBody::box_body(Vec3::ZERO, vec3(0.5, 0.5, 0.5), 1000.0);
        // 1 m cube of water.
        assert!((b.mass() - 1000.0).abs() < 1e-3);
        // Uniform cube: I = m/6 * side^2 = 1000/6 on every axis.
        let expect = 1000.0 / 6.0;
        for axis in [b.inv_inertia_local.x, b.inv_inertia_local.y, b.inv_inertia_local.z] {
            assert!((1.0 / axis - expect).abs() < 1e-1);
        }
    }

    #[test]
    fn inv_inertia_world_is_symmetric_and_matches_local_at_identity() {
        let b = RigidBody::box_body(Vec3::ZERO, vec3(0.2, 0.4, 0.1), 250.0);
        let m = b.inv_inertia_world();
        assert!((m.cols[0].x - b.inv_inertia_local.x).abs() < 1e-6);
        assert!((m.cols[1].y - b.inv_inertia_local.y).abs() < 1e-6);
        assert!((m.cols[2].z - b.inv_inertia_local.z).abs() < 1e-6);

        let mut rotated = b;
        rotated.orient = Quat { x: 0.3, y: 0.2, z: -0.1, w: 0.9 }.normalize();
        let r = rotated.inv_inertia_world();
        // Symmetry is the property that catches a transposed rotation.
        assert!((r.cols[0].y - r.cols[1].x).abs() < 1e-6);
        assert!((r.cols[0].z - r.cols[2].x).abs() < 1e-6);
        assert!((r.cols[1].z - r.cols[2].y).abs() < 1e-6);
    }

    #[test]
    fn corners_bound_the_box() {
        let b = RigidBody::box_body(vec3(1.0, 2.0, 3.0), vec3(0.5, 0.25, 0.75), 100.0);
        let (cs, n) = b.ground_points(Vec3::Y);
        assert_eq!(n, 8);
        for c in &cs[..n] {
            let d = *c - b.pos;
            assert!(d.x.abs() <= 0.5 + 1e-6);
            assert!(d.y.abs() <= 0.25 + 1e-6);
            assert!(d.z.abs() <= 0.75 + 1e-6);
        }
        assert!((b.lowest_point_y() - 1.75).abs() < 1e-6);
    }

    #[test]
    fn impulse_conserves_expected_momentum() {
        let mut b = RigidBody::box_body(Vec3::ZERO, vec3(0.5, 0.5, 0.5), 8.0);
        let inv_i = b.inv_inertia_world();
        let m = b.mass();
        b.apply_impulse(vec3(0.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), &inv_i);
        assert!((b.lin_vel.x * m - 1.0).abs() < 1e-5);
        // r x p is around -z for r=+y, p=+x.
        assert!(b.ang_vel.z < 0.0);
    }
}
