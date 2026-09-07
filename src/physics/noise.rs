//! Deterministic gradient noise, with exact derivatives.
//!
//! Perlin's construction, with two departures from the usual implementation:
//!
//! 1. **Gradients are hashed, not tabulated.** The classic version indexes a
//!    256-entry permutation table. A table would have to live inside
//!    [`TerrainModel`](super::world::TerrainModel), which is `Copy` and is
//!    copied into every `World` — once per trial per organism. Hashing the
//!    lattice coordinates with [`splitmix64`] instead costs a few integer
//!    operations and leaves the model a handful of scalars.
//! 2. **The value and its gradient are returned together.** The two share
//!    nearly all of their arithmetic, and the terrain needs both at almost
//!    every call site.
//!
//! There is not a transcendental anywhere in here: integer hashing and
//! polynomial arithmetic only. That makes this field *more* defensibly
//! reproducible than the sine terrain it was written to replace, rather than
//! merely as reproducible.

use crate::math::Real;
use crate::rng::splitmix64;

/// The eight unit gradients: four axes and four diagonals.
///
/// Perlin's original 2D set. Sixteen would reduce the faint axis alignment
/// visible in a single octave, but summing over octaves and warping the domain
/// both hide it, and eight keeps the selection a three-bit shift.
const SQRT_HALF: Real = std::f32::consts::FRAC_1_SQRT_2;
const GRADIENTS: [(Real, Real); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (SQRT_HALF, SQRT_HALF),
    (-SQRT_HALF, SQRT_HALF),
    (SQRT_HALF, -SQRT_HALF),
    (-SQRT_HALF, -SQRT_HALF),
];

/// Perlin noise with unit gradients is bounded by `sqrt(n)/2` in `n` dimensions,
/// so in 2D it lands in `[-1/sqrt(2), 1/sqrt(2)]`. Scaling by `sqrt(2)` puts the
/// output in `[-1, 1]`, which is what makes an amplitude parameter mean
/// something.
const PERLIN_SCALE: Real = std::f32::consts::SQRT_2;

/// Odd multipliers, so that sign-extended lattice coordinates map injectively
/// before mixing. A hash built by adding or XOR-ing the raw coordinates is
/// symmetric about zero, which shows up as a landscape mirrored through the
/// origin — exactly where every organism spawns.
const HASH_X: u64 = 0x9E37_79B9_7F4A_7C15;
const HASH_Z: u64 = 0xC2B2_AE3D_27D4_EB4F;

#[inline]
fn gradient(seed: u64, i: i32, j: i32) -> (Real, Real) {
    let mut s =
        seed ^ (i as i64 as u64).wrapping_mul(HASH_X) ^ (j as i64 as u64).wrapping_mul(HASH_Z);
    GRADIENTS[(splitmix64(&mut s) >> 61) as usize]
}

/// Quintic fade `6t^5 - 15t^4 + 10t^3` and its derivative `30t^2(t-1)^2`.
///
/// Quintic rather than Hermite because its *second* derivative also vanishes at
/// the ends, so the surface has no visible crease along the lattice — and
/// because a polynomial fade is what makes the analytic gradient available at
/// all.
#[inline]
fn fade(t: Real) -> (Real, Real) {
    let t2 = t * t;
    let t3 = t2 * t;
    let d = t - 1.0;
    (t3 * (t * (t * 6.0 - 15.0) + 10.0), 30.0 * t2 * d * d)
}

/// Where the field stops.
///
/// Past 2^22 an `f32` has under a quarter of a unit of resolution left, so the
/// lattice interpolation has nothing to interpolate; further out the corner
/// offsets grow until the dot products overflow to infinity and the normalised
/// normal comes back `NaN`. A `NaN` surface normal fed to the contact solver is
/// the worst possible failure — silent, and it poisons every statistic
/// downstream — so the field is simply zero beyond here, and beyond any `NaN`
/// coordinate too. Nothing in a simulation reaches it: speed is clamped, and a
/// trial lasts seconds.
const FIELD_LIMIT: Real = 4_194_304.0;

/// `floor` as an integer. Only ever called on a coordinate already inside
/// [`FIELD_LIMIT`], so the cast cannot saturate and `+ 1` cannot overflow.
#[inline]
fn floor_i(v: Real) -> i32 {
    v.floor() as i32
}

/// Gradient noise at `(x, z)`, returning `(value, dv/dx, dv/dz)`.
///
/// The value is in `[-1, 1]`. The derivative is exact — carried through the dot
/// products and both lerps by the product rule — which is what lets the terrain
/// hand the solver a true surface normal rather than a finite-difference guess.
#[inline]
pub fn perlin_d(seed: u64, x: Real, z: Real) -> (Real, Real, Real) {
    // Written as a positive test so that a `NaN` coordinate fails it too.
    if !(x.abs() < FIELD_LIMIT && z.abs() < FIELD_LIMIT) {
        return (0.0, 0.0, 0.0);
    }
    let xi = floor_i(x);
    let zi = floor_i(z);
    let u = x - xi as Real;
    let v = z - zi as Real;
    let u1 = u - 1.0;
    let v1 = v - 1.0;

    let g00 = gradient(seed, xi, zi);
    let g10 = gradient(seed, xi + 1, zi);
    let g01 = gradient(seed, xi, zi + 1);
    let g11 = gradient(seed, xi + 1, zi + 1);

    // Each corner's gradient dotted with the offset from that corner.
    let n00 = g00.0 * u + g00.1 * v;
    let n10 = g10.0 * u1 + g10.1 * v;
    let n01 = g01.0 * u + g01.1 * v1;
    let n11 = g11.0 * u1 + g11.1 * v1;

    let (su, dsu) = fade(u);
    let (sv, dsv) = fade(v);

    let nx0 = n00 + su * (n10 - n00);
    let nx1 = n01 + su * (n11 - n01);
    let n = nx0 + sv * (nx1 - nx0);

    // d/du of each lerp: the interpolated gradient component, plus the fade's
    // own rate times the difference it is interpolating.
    let dnx0_du = g00.0 + su * (g10.0 - g00.0) + dsu * (n10 - n00);
    let dnx1_du = g01.0 + su * (g11.0 - g01.0) + dsu * (n11 - n01);
    // d/dv of the same two: no fade term, because `su` does not depend on `v`.
    let dnx0_dv = g00.1 + su * (g10.1 - g00.1);
    let dnx1_dv = g01.1 + su * (g11.1 - g01.1);

    let dn_du = dnx0_du + sv * (dnx1_du - dnx0_du);
    let dn_dv = dnx0_dv + sv * (dnx1_dv - dnx0_dv) + dsv * (nx1 - nx0);

    (n * PERLIN_SCALE, dn_du * PERLIN_SCALE, dn_dv * PERLIN_SCALE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gradient is the part that will be wrong, and it cannot be validated
    /// against itself. Central differences are an independent computation.
    #[test]
    fn the_analytic_gradient_matches_central_differences() {
        let h = 1e-3;
        let mut worst = 0.0f64;
        for seed in [0u64, 1, 0xDEAD_BEEF, u64::MAX] {
            // Deliberately awkward steps: sampling on a rational grid would keep
            // landing on lattice points and miss most of each cell.
            for a in -60..60 {
                for b in -60..60 {
                    let x = a as Real * 0.137;
                    let z = b as Real * 0.211;
                    let (_, dx, dz) = perlin_d(seed, x, z);
                    let fdx = (perlin_d(seed, x + h, z).0 - perlin_d(seed, x - h, z).0) / (2.0 * h);
                    let fdz = (perlin_d(seed, x, z + h).0 - perlin_d(seed, x, z - h).0) / (2.0 * h);
                    worst = worst.max((dx - fdx).abs() as f64);
                    worst = worst.max((dz - fdz).abs() as f64);
                }
            }
        }
        // The residual is dominated by f32 cancellation in the difference
        // quotient, not by the derivative.
        assert!(worst < 5e-3, "worst gradient error {worst}");
    }

    #[test]
    fn values_stay_within_the_unit_bound() {
        let mut worst = 0.0f32;
        for seed in [0u64, 7, 0x5EED] {
            for a in -200..200 {
                for b in -200..200 {
                    let n = perlin_d(seed, a as Real * 0.0731, b as Real * 0.0917).0;
                    assert!(n.is_finite());
                    worst = worst.max(n.abs());
                }
            }
        }
        assert!(worst <= 1.0, "exceeded the unit bound: {worst}");
        // And it does get somewhere near it, or the scaling is wrong.
        assert!(worst > 0.5, "suspiciously flat: {worst}");
    }

    #[test]
    fn lattice_points_are_zero_and_finite() {
        for i in -3..=3 {
            for j in -3..=3 {
                let (n, dx, dz) = perlin_d(12345, i as Real, j as Real);
                assert_eq!(n, 0.0, "Perlin is zero at every lattice point");
                assert!(dx.is_finite() && dz.is_finite());
            }
        }
    }

    /// A hash built symmetrically about zero mirrors the whole field through the
    /// origin. Negative coordinates must be their own place.
    #[test]
    fn negative_coordinates_are_not_a_mirror_of_positive_ones() {
        let mut differs = 0;
        for a in 1..50 {
            let x = a as Real * 0.31;
            if perlin_d(3, x, x).0 != perlin_d(3, -x, -x).0 {
                differs += 1;
            }
        }
        assert!(differs > 40, "field looks mirrored through the origin");
    }

    #[test]
    fn the_same_input_always_gives_the_same_bits() {
        for a in 0..200 {
            let x = a as Real * 0.37;
            let z = a as Real * -0.19;
            assert_eq!(perlin_d(99, x, z), perlin_d(99, x, z));
        }
    }

    /// A `NaN` surface normal reaching the contact solver is the failure mode
    /// this guard exists for: silent, and it poisons every statistic downstream.
    #[test]
    fn nothing_outside_the_field_produces_a_nan() {
        let far =
            [FIELD_LIMIT, -FIELD_LIMIT, 1e12, -1e12, Real::INFINITY, Real::NEG_INFINITY, Real::NAN];
        for v in far {
            for (x, z) in [(v, 0.0), (0.0, v), (v, v)] {
                assert_eq!(perlin_d(5, x, z), (0.0, 0.0, 0.0), "at {x}, {z}");
            }
        }
        // And it is still live right up to the edge.
        assert_ne!(perlin_d(5, FIELD_LIMIT - 0.5, 0.25), (0.0, 0.0, 0.0));
    }

    #[test]
    fn different_seeds_give_different_fields() {
        let differs = (0..200).filter(|a| {
            let x = *a as Real * 0.37;
            perlin_d(1, x, 0.5) != perlin_d(2, x, 0.5)
        });
        assert!(differs.count() > 150);
    }
}
