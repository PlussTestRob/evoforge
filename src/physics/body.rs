//! Rigid bodies.
//!
//! Every body is a box. That is not a placeholder for a general collision
//! system: boxes give us mass properties, ground contact and a visual identity
//! for free, and the whole point of the initial experiment is to see whether
//! evolution finds locomotion, not to support arbitrary geometry.

use crate::math::{vec3, Mat3, Quat, Real, Vec3};

/// A box-shaped rigid body in maximal coordinates.
///
/// Stored flat in [`super::World::bodies`] and referenced by index. There are no
/// back-references, no parent pointers and no `Rc`: the whole world is a couple
/// of contiguous `Vec`s, which is what makes an evaluation cheap to construct,
/// cheap to iterate and trivially `Send`.
#[derive(Clone, Copy, Debug)]
pub struct RigidBody {
    /// World-space centre of mass.
    pub pos: Vec3,
    pub orient: Quat,
    pub lin_vel: Vec3,
    pub ang_vel: Vec3,
    /// Box half-extents in the body frame.
    pub half_extents: Vec3,
    pub inv_mass: Real,
    /// Diagonal of the inverse inertia tensor in the body frame. A box's inertia
    /// tensor is diagonal in its own frame, so there is no reason to store nine
    /// numbers.
    pub inv_inertia_local: Vec3,
}

impl RigidBody {
    /// Create a dynamic box of the given density.
    pub fn box_body(pos: Vec3, half_extents: Vec3, density: Real) -> RigidBody {
        let h = half_extents;
        let mass = 8.0 * h.x * h.y * h.z * density;
        debug_assert!(mass > 0.0);
        // For a box, I_xx = m/12 * (height^2 + depth^2) with full extents,
        // which reduces to m/3 * (hy^2 + hz^2) in half-extents.
        let i = vec3(
            mass / 3.0 * (h.y * h.y + h.z * h.z),
            mass / 3.0 * (h.x * h.x + h.z * h.z),
            mass / 3.0 * (h.x * h.x + h.y * h.y),
        );
        RigidBody {
            pos,
            orient: Quat::IDENTITY,
            lin_vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            half_extents: h,
            inv_mass: 1.0 / mass,
            inv_inertia_local: vec3(1.0 / i.x, 1.0 / i.y, 1.0 / i.z),
        }
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

    /// The eight corners of the box in world space.
    pub fn corners(&self) -> [Vec3; 8] {
        let h = self.half_extents;
        let mut out = [Vec3::ZERO; 8];
        let mut n = 0;
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    let local: Vec3 = vec3(sx * h.x, sy * h.y, sz * h.z);
                    out[n] = self.pos + self.orient.rotate(local);
                    n += 1;
                }
            }
        }
        out
    }

    /// Lowest world-space corner height. Used to place an organism on the ground
    /// at spawn.
    pub fn lowest_corner_y(&self) -> Real {
        self.corners().iter().fold(Real::INFINITY, |m, c| m.min(c.y))
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
        let cs = b.corners();
        assert_eq!(cs.len(), 8);
        for c in cs {
            let d = c - b.pos;
            assert!(d.x.abs() <= 0.5 + 1e-6);
            assert!(d.y.abs() <= 0.25 + 1e-6);
            assert!(d.z.abs() <= 0.75 + 1e-6);
        }
        assert!((b.lowest_corner_y() - 1.75).abs() < 1e-6);
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
