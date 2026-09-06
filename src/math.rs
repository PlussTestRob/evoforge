//! Minimal 3D math for the simulator.
//!
//! # Why the hand-rolled transcendentals?
//!
//! Reproducibility is a first-class requirement. `f32::sin`, `f32::ln` and friends
//! delegate to the platform libm, which is *not* required to produce identical
//! results across operating systems, CPUs or libc versions. Basic arithmetic and
//! `sqrt` are exactly specified by IEEE-754 and are safe.
//!
//! So: every transcendental the simulator needs is implemented here with a fixed
//! polynomial. The results are bitwise identical everywhere, and the accuracy
//! (~1e-7 relative) is far beyond what an evolutionary experiment needs.

use serde::{Deserialize, Serialize};

/// The scalar type used throughout the simulator.
///
/// `f32` is deliberate: bodies are few, so the bottleneck is arithmetic
/// throughput and cache behaviour rather than precision. Switching to `f64` is a
/// one-line change if the constraint solver ever proves too soft.
pub type Real = f32;

pub const PI: Real = std::f32::consts::PI;
pub const TAU: Real = std::f32::consts::TAU;
pub const FRAC_PI_2: Real = std::f32::consts::FRAC_PI_2;
const LN_2: Real = std::f32::consts::LN_2;

// ---------------------------------------------------------------------------
// Deterministic transcendentals
// ---------------------------------------------------------------------------

/// sin(r) for |r| <= pi/4. Taylor to r^9; error < 3e-8 on the interval.
#[inline]
fn poly_sin(r: Real) -> Real {
    let r2 = r * r;
    r * (1.0
        + r2 * (-1.0 / 6.0
            + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0 + r2 * (1.0 / 362_880.0)))))
}

/// cos(r) for |r| <= pi/4. Taylor to r^8; error < 2e-9 on the interval.
#[inline]
fn poly_cos(r: Real) -> Real {
    let r2 = r * r;
    1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 + r2 * (1.0 / 40320.0))))
}

/// Deterministic `(sin(x), cos(x))` via quadrant reduction.
///
/// Accurate for the modest arguments the simulator uses (joint limits, phase
/// offsets). Precision degrades for very large `|x|` because the reduction is
/// done in `Real`; that is fine and is never hit in practice.
#[inline]
pub fn dsincos(x: Real) -> (Real, Real) {
    let q = (x * (2.0 / PI)).round();
    let r = x - q * FRAC_PI_2;
    let (s, c) = (poly_sin(r), poly_cos(r));
    match (q as i64).rem_euclid(4) {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

#[inline]
pub fn dsin(x: Real) -> Real {
    dsincos(x).0
}

#[inline]
pub fn dcos(x: Real) -> Real {
    dsincos(x).1
}

/// Deterministic natural logarithm. Returns `-inf` for zero, `NaN` for negatives.
///
/// Splits off the binary exponent, then uses the fast-converging
/// `ln(m) = 2 * atanh((m-1)/(m+1))` series on the reduced mantissa.
pub fn dln(x: Real) -> Real {
    if x <= 0.0 {
        return if x == 0.0 { Real::NEG_INFINITY } else { Real::NAN };
    }
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mut m = Real::from_bits((bits & 0x007f_ffff) | (127 << 23));
    // Subnormals: normalise by scaling into range. Never occurs in practice but
    // keeps the function total.
    if exp == -127 {
        return dln(x * 16_777_216.0) - 16.0 * LN_2;
    }
    if m > std::f32::consts::SQRT_2 {
        m *= 0.5;
        exp += 1;
    }
    let t = (m - 1.0) / (m + 1.0);
    let t2 = t * t;
    let series = 2.0
        * t
        * (1.0 + t2 * (1.0 / 3.0 + t2 * (0.2 + t2 * (1.0 / 7.0 + t2 * (1.0 / 9.0)))));
    series + (exp as Real) * LN_2
}

/// Deterministic `tanh`, the neural-network activation.
///
/// Padé [5/4] approximant, `x(945 + 105x^2 + x^4) / (945 + 420x^2 + 15x^4)`,
/// clamped to +/-1 beyond +/-3.5 where it would otherwise exceed 1. Max error
/// about 2e-3, monotone, and cheaper than a true `tanh` — one divide and a
/// handful of multiplies, with no library call and therefore no cross-platform
/// variation.
///
/// The lower-order `x(27 + x^2) / (27 + 9x^2)` form often used for this is
/// tempting but is only accurate to about 2.4e-2, which is visible as a
/// systematic flattening of the activation curve.
#[inline]
pub fn tanh_approx(x: Real) -> Real {
    const CLAMP: Real = 3.5;
    if x >= CLAMP {
        return 1.0;
    }
    if x <= -CLAMP {
        return -1.0;
    }
    let x2 = x * x;
    let x4 = x2 * x2;
    x * (945.0 + 105.0 * x2 + x4) / (945.0 + 420.0 * x2 + 15.0 * x4)
}

/// Triangle wave in [-1, 1] with unit period, used for controller clock inputs.
///
/// A sine would work equally well but this is exact, branch-light and free of
/// any range-reduction error as simulation time grows.
#[inline]
pub fn triangle(t: Real) -> Real {
    let f = t - t.floor(); // [0, 1)
    if f < 0.5 {
        4.0 * f - 1.0
    } else {
        3.0 - 4.0 * f
    }
}

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: Real,
    pub y: Real,
    pub z: Real,
}

pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

#[inline]
pub const fn vec3(x: Real, y: Real, z: Real) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = vec3(0.0, 0.0, 0.0);
    pub const X: Vec3 = vec3(1.0, 0.0, 0.0);
    pub const Y: Vec3 = vec3(0.0, 1.0, 0.0);
    pub const Z: Vec3 = vec3(0.0, 0.0, 1.0);

    #[inline]
    pub fn splat(v: Real) -> Vec3 {
        vec3(v, v, v)
    }

    #[inline]
    pub fn dot(self, o: Vec3) -> Real {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Vec3) -> Vec3 {
        vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    #[inline]
    pub fn length_sq(self) -> Real {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> Real {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalize_or(self, fallback: Vec3) -> Vec3 {
        let len_sq = self.length_sq();
        if len_sq > 1e-20 {
            self * (1.0 / len_sq.sqrt())
        } else {
            fallback
        }
    }

    #[inline]
    pub fn abs(self) -> Vec3 {
        vec3(self.x.abs(), self.y.abs(), self.z.abs())
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Component-wise multiply.
    #[inline]
    pub fn mul_elem(self, o: Vec3) -> Vec3 {
        vec3(self.x * o.x, self.y * o.y, self.z * o.z)
    }

    /// Any unit vector perpendicular to `self` (which must be normalised).
    #[inline]
    pub fn any_perpendicular(self) -> Vec3 {
        let a = if self.x.abs() < 0.7 { Vec3::X } else { Vec3::Y };
        self.cross(a).normalize_or(Vec3::Z)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, o: Vec3) -> Vec3 {
        vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl std::ops::Mul<Real> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, s: Real) -> Vec3 {
        vec3(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    #[inline]
    fn neg(self) -> Vec3 {
        vec3(-self.x, -self.y, -self.z)
    }
}

impl std::ops::AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}

impl std::ops::SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, o: Vec3) {
        *self = *self - o;
    }
}

// ---------------------------------------------------------------------------
// Quat
// ---------------------------------------------------------------------------

/// Unit quaternion, `w` scalar last.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    pub x: Real,
    pub y: Real,
    pub z: Real,
    pub w: Real,
}

impl Default for Quat {
    fn default() -> Self {
        Quat::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    #[inline]
    pub fn vec(self) -> Vec3 {
        vec3(self.x, self.y, self.z)
    }

    #[inline]
    pub fn conjugate(self) -> Quat {
        Quat { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }

    /// Quaternion composition. Named `mul` because that is what it is called
    /// everywhere quaternions are discussed; an operator overload would read
    /// worse at the call sites, which are all in the solver.
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn mul(self, o: Quat) -> Quat {
        Quat {
            w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        }
    }

    /// Rotate a vector by this quaternion.
    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = self.vec();
        let t = u.cross(v) * 2.0;
        v + t * self.w + u.cross(t)
    }

    /// Rotate a vector by the inverse of this quaternion.
    #[inline]
    pub fn inv_rotate(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    #[inline]
    pub fn normalize(self) -> Quat {
        let n_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if n_sq > 1e-20 {
            let inv = 1.0 / n_sq.sqrt();
            Quat { x: self.x * inv, y: self.y * inv, z: self.z * inv, w: self.w * inv }
        } else {
            Quat::IDENTITY
        }
    }

    /// First-order integration of angular velocity `omega` over `dt`.
    ///
    /// Uses only multiplication and a `sqrt` in the renormalisation, so it is
    /// bitwise reproducible.
    #[inline]
    pub fn integrate(self, omega: Vec3, dt: Real) -> Quat {
        let dq = Quat { x: omega.x, y: omega.y, z: omega.z, w: 0.0 }.mul(self);
        let h = 0.5 * dt;
        Quat {
            x: self.x + dq.x * h,
            y: self.y + dq.y * h,
            z: self.z + dq.z * h,
            w: self.w + dq.w * h,
        }
        .normalize()
    }

    /// Rotation matrix for this quaternion.
    pub fn to_mat3(self) -> Mat3 {
        let (x, y, z, w) = (self.x, self.y, self.z, self.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        Mat3 {
            cols: [
                vec3(1.0 - (yy + zz), xy + wz, xz - wy),
                vec3(xy - wz, 1.0 - (xx + zz), yz + wx),
                vec3(xz + wy, yz - wx, 1.0 - (xx + yy)),
            ],
        }
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

// ---------------------------------------------------------------------------
// Mat3
// ---------------------------------------------------------------------------

/// Column-major 3x3 matrix. Used for inertia tensors and constraint effective-mass
/// matrices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub cols: [Vec3; 3],
}

impl Default for Mat3 {
    fn default() -> Self {
        Mat3::ZERO
    }
}

impl Mat3 {
    pub const ZERO: Mat3 = Mat3 { cols: [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO] };
    pub const IDENTITY: Mat3 = Mat3 { cols: [Vec3::X, Vec3::Y, Vec3::Z] };

    #[inline]
    pub fn diagonal(d: Vec3) -> Mat3 {
        Mat3 {
            cols: [vec3(d.x, 0.0, 0.0), vec3(0.0, d.y, 0.0), vec3(0.0, 0.0, d.z)],
        }
    }

    /// Skew-symmetric matrix such that `skew(a) * b == a.cross(b)`.
    #[inline]
    pub fn skew(a: Vec3) -> Mat3 {
        Mat3 {
            cols: [vec3(0.0, a.z, -a.y), vec3(-a.z, 0.0, a.x), vec3(a.y, -a.x, 0.0)],
        }
    }

    #[inline]
    pub fn mul_vec(&self, v: Vec3) -> Vec3 {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }

    #[inline]
    pub fn mul_mat(&self, o: &Mat3) -> Mat3 {
        Mat3 {
            cols: [self.mul_vec(o.cols[0]), self.mul_vec(o.cols[1]), self.mul_vec(o.cols[2])],
        }
    }

    pub fn transpose(&self) -> Mat3 {
        let c = &self.cols;
        Mat3 {
            cols: [
                vec3(c[0].x, c[1].x, c[2].x),
                vec3(c[0].y, c[1].y, c[2].y),
                vec3(c[0].z, c[1].z, c[2].z),
            ],
        }
    }

    pub fn sub(&self, o: &Mat3) -> Mat3 {
        Mat3 {
            cols: [
                self.cols[0] - o.cols[0],
                self.cols[1] - o.cols[1],
                self.cols[2] - o.cols[2],
            ],
        }
    }

    pub fn add(&self, o: &Mat3) -> Mat3 {
        Mat3 {
            cols: [
                self.cols[0] + o.cols[0],
                self.cols[1] + o.cols[1],
                self.cols[2] + o.cols[2],
            ],
        }
    }

    pub fn determinant(&self) -> Real {
        self.cols[0].dot(self.cols[1].cross(self.cols[2]))
    }

    /// Inverse, or `None` if singular.
    pub fn inverse(&self) -> Option<Mat3> {
        let c = &self.cols;
        let r0 = c[1].cross(c[2]);
        let r1 = c[2].cross(c[0]);
        let r2 = c[0].cross(c[1]);
        let det = c[0].dot(r0);
        if det.abs() < 1e-20 {
            return None;
        }
        let inv_det = 1.0 / det;
        // Rows of the adjugate become columns of the inverse after transposing.
        Some(Mat3 {
            cols: [
                vec3(r0.x, r1.x, r2.x) * inv_det,
                vec3(r0.y, r1.y, r2.y) * inv_det,
                vec3(r0.z, r1.z, r2.z) * inv_det,
            ],
        })
    }

    /// Solve `self * x = b`, falling back to the zero vector when singular.
    #[inline]
    pub fn solve(&self, b: Vec3) -> Vec3 {
        match self.inverse() {
            Some(inv) => inv.mul_vec(b),
            None => Vec3::ZERO,
        }
    }
}

#[inline]
pub fn clamp(v: Real, lo: Real, hi: Real) -> Real {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Real, b: Real, tol: Real) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn dsincos_matches_std() {
        let mut x = -20.0f32;
        while x < 20.0 {
            let (s, c) = dsincos(x);
            assert!(close(s, x.sin(), 2e-5), "sin({x}) = {s} vs {}", x.sin());
            assert!(close(c, x.cos(), 2e-5), "cos({x}) = {c} vs {}", x.cos());
            x += 0.037;
        }
    }

    #[test]
    fn dln_matches_std() {
        for i in 1..2000u32 {
            let x = i as Real * 0.01;
            assert!(close(dln(x), x.ln(), 1e-5), "ln({x})");
        }
        assert!(dln(0.0).is_infinite());
        assert!(dln(-1.0).is_nan());
    }

    #[test]
    fn tanh_approx_is_bounded_and_monotone() {
        let mut prev = -2.0;
        let mut x = -6.0f32;
        while x < 6.0 {
            let t = tanh_approx(x);
            assert!((-1.0..=1.0).contains(&t));
            assert!(t >= prev - 1e-7, "not monotone at {x}");
            assert!(close(t, x.tanh(), 3e-3), "tanh({x}) = {t} vs {}", x.tanh());
            prev = t;
            x += 0.01;
        }
    }

    #[test]
    fn triangle_wave_shape() {
        assert!(close(triangle(0.0), -1.0, 1e-6));
        assert!(close(triangle(0.25), 0.0, 1e-6));
        assert!(close(triangle(0.5), 1.0, 1e-6));
        assert!(close(triangle(0.75), 0.0, 1e-6));
        // Periodic.
        assert!(close(triangle(3.3), triangle(0.3), 1e-6));
    }

    #[test]
    fn quat_rotation_roundtrip() {
        let q = Quat { x: 0.2, y: -0.4, z: 0.1, w: 0.9 }.normalize();
        let v = vec3(1.0, 2.0, -3.0);
        let back = q.inv_rotate(q.rotate(v));
        assert!((back - v).length() < 1e-5);
        // Rotation preserves length.
        assert!(close(q.rotate(v).length(), v.length(), 1e-5));
    }

    #[test]
    fn quat_matrix_agrees_with_rotate() {
        let q = Quat { x: -0.3, y: 0.5, z: 0.2, w: 0.7 }.normalize();
        let m = q.to_mat3();
        for v in [Vec3::X, Vec3::Y, Vec3::Z, vec3(1.0, -2.0, 0.5)] {
            assert!((m.mul_vec(v) - q.rotate(v)).length() < 1e-5);
        }
    }

    #[test]
    fn quat_integration_stays_normalised() {
        let mut q = Quat::IDENTITY;
        let w = vec3(3.0, -1.0, 2.0);
        for _ in 0..10_000 {
            q = q.integrate(w, 1.0 / 240.0);
        }
        let n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
        assert!(close(n, 1.0, 1e-5));
    }

    #[test]
    fn mat3_inverse() {
        let m = Mat3 {
            cols: [vec3(2.0, 0.3, -0.1), vec3(0.3, 1.5, 0.2), vec3(-0.1, 0.2, 3.0)],
        };
        let inv = m.inverse().unwrap();
        let id = m.mul_mat(&inv);
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                let got = match j {
                    0 => id.cols[i].x,
                    1 => id.cols[i].y,
                    _ => id.cols[i].z,
                };
                assert!(close(got, expect, 1e-5));
            }
        }
        assert!(Mat3::ZERO.inverse().is_none());
    }

    #[test]
    fn skew_matches_cross() {
        let a = vec3(1.0, -2.0, 0.5);
        let b = vec3(-0.3, 0.7, 2.0);
        assert!((Mat3::skew(a).mul_vec(b) - a.cross(b)).length() < 1e-6);
    }
}
