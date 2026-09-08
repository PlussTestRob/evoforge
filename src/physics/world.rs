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
use crate::math::{clamp, dcos, dsin, dsincos, vec3, Mat3, Real, Vec3, TAU};

use super::body::RigidBody;
use super::noise;

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
    /// Seeded fractal landscape: octaves of hashed gradient noise over a warped
    /// domain.
    ///
    /// What [`Rough`](TerrainModel::Rough) is not: it repeats every wavelength,
    /// it has no seed, and it has features at one scale only, so every organism
    /// in every trial meets the same memorisable ripple. This field is
    /// aperiodic, keyed on a seed, moved per trial, and built at four or more
    /// scales at once — which is what makes ground look and behave like ground.
    ///
    /// Still analytic, still exactly differentiable, and now with no
    /// transcendental in it at all: integer hashing and polynomial arithmetic
    /// only. See [`noise`](super::noise) for the construction and
    /// [`Self::sample`] for the chain rule through the warp.
    ///
    /// One honest limit, measured rather than assumed:
    ///
    /// * **It is still a height field**, so it is single-valued and smooth: no
    ///   overhangs, no vertical walls, no gaps. Smooth undulation is exactly
    ///   what a wheel is good at, and raising `amplitude` makes the ground
    ///   *steeper* rather than a different kind of problem. What defeats a
    ///   wheel is a discontinuity at or above its own radius, and that needs
    ///   discrete obstacles, not a better height field.
    Fractal(FractalField),
}

/// The parameters of a fractal landscape.
///
/// A struct rather than a pile of enum fields because there are now four bands
/// of them, and because a struct can be built with `..Default::default()` —
/// which is what lets every new band default to *off* and keeps a config that
/// does not mention them meaning exactly what it meant before.
///
/// Serialised flat into the `fractal` variant, so a trace still reads
/// `{"kind": "fractal", "amplitude": ..., ...}`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FractalField {
    /// Field identity. Two seeds give unrelated landscapes.
    ///
    /// Written to JSON as a decimal *string*: JavaScript's only number is a
    /// double, so a bare `u64` above 2^53 comes back off by a few — and a seed
    /// off by a few is a different landscape entirely. The viewer has to
    /// reproduce this exactly.
    #[serde(with = "seed_as_string")]
    pub seed: u64,

    // --- Band 1: the landscape itself. -----------------------------------
    /// Scale of the largest band, metres. Relief runs to about 1.8x this.
    pub amplitude: Real,
    /// Size of the largest feature, metres.
    pub wavelength: Real,
    /// How many octaves are summed. Each one is `lacunarity` times finer and
    /// `gain` times shallower than the last.
    pub octaves: u32,
    pub lacunarity: Real,
    pub gain: Real,
    /// Domain warp strength, in units of `wavelength`.
    ///
    /// Bends the field into ridges and basins rather than blobs. It buys the
    /// tail of the slope distribution, and it roughly doubles how much the
    /// ground's character varies from place to place. What it cannot do is
    /// produce a cliff — that is `step` — or concentrate the difficulty, which
    /// is `terrace_mask`.
    pub warp: Real,

    // --- Band 2: detail at organism scale. --------------------------------
    /// Amplitude of a second, finer band laid over the landscape, metres.
    /// Zero leaves the field exactly as it was before this band existed.
    pub detail_amplitude: Real,
    pub detail_wavelength: Real,
    pub detail_octaves: u32,

    // --- Band 3: where the ground is calm and where it is savage. ---------
    /// How strongly a slow field modulates the detail band's amplitude, in
    /// `[0, 1]`. Zero is uniform detail everywhere.
    pub modulation: Real,
    /// Size of the calm and savage regions, metres.
    pub modulation_wavelength: Real,

    // --- Band 4: cliffs. ---------------------------------------------------
    /// Terrace height, metres. Zero is a smooth field.
    ///
    /// Quantising the height to steps is the only thing here that produces a
    /// genuinely sheer face: fractional Brownian motion has one steepness and
    /// applies it everywhere, so scaling it up gives uniformly steep ground
    /// rather than occasional cliffs. A terrace is flat for most of its span
    /// and climbs through the rest, which puts the difficulty in a small
    /// fraction of the area and leaves the rest crossable.
    pub step: Real,
    /// Fraction of a terrace spent on the riser. The riser is steeper than the
    /// underlying slope by exactly `1 / riser`, so this is the cliff knob.
    ///
    /// It has a floor that is physics rather than taste: a wall must be several
    /// integration steps wide or a body crosses it in one and meets it as a
    /// single enormous penetration. `Config::validate` enforces that.
    pub riser: Real,
    /// Terrace only the savage regions, blending smoothly back into untouched
    /// hills elsewhere. Concentrates the cliffs instead of tiling the world
    /// with them.
    pub terrace_mask: bool,

    // --- Per-trial placement. ---------------------------------------------
    /// Rigid motion of the field under the world, so a trial can be run on a
    /// different piece of the same landscape. Translation is in units of
    /// `wavelength`; rotation is stored as its sine and cosine so that no
    /// trigonometry happens per sample. Identity is `(0, 0, 0, 1)`.
    pub offset_x: Real,
    pub offset_z: Real,
    pub rot_sin: Real,
    pub rot_cos: Real,
}

impl Default for FractalField {
    /// Every band but the first is off, so a `FractalField` built with
    /// `..Default::default()` is the field as it was before the bands existed.
    fn default() -> FractalField {
        FractalField {
            seed: 0,
            amplitude: 0.25,
            wavelength: 6.0,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            warp: 0.3,
            detail_amplitude: 0.0,
            detail_wavelength: 3.0,
            detail_octaves: 4,
            modulation: 0.0,
            modulation_wavelength: 35.0,
            step: 0.0,
            riser: 0.12,
            terrace_mask: false,
            offset_x: 0.0,
            offset_z: 0.0,
            rot_sin: 0.0,
            rot_cos: 1.0,
        }
    }
}

impl FractalField {
    /// Height and its exact gradient, `(h, dh/dx, dh/dz)`, in metres.
    ///
    /// Composed as bands rather than one noise call, because that is what makes
    /// each piece independently measurable — see `examples/terrain_probe.rs`.
    /// The gradient is carried through every band by the product and chain
    /// rules; the places it goes wrong are the warp Jacobian, the `dm` term
    /// where modulation multiplies the detail band, and the `dm * gap` term
    /// where the mask blends terraced ground into smooth. All three are checked
    /// against central differences in the tests.
    pub fn height_and_gradient(&self, x: Real, z: Real) -> (Real, Real, Real) {
        let wavelength = self.wavelength.max(1e-3);
        let inv_w = 1.0 / wavelength;

        // World space to field space, in *metres*: rotate about the origin,
        // then translate by the per-trial offset. Working in metres rather than
        // in units of the base wavelength is what lets the bands below each
        // divide by their own wavelength and still move together per trial.
        let mx = x * self.rot_cos - z * self.rot_sin + self.offset_x * wavelength;
        let mz = x * self.rot_sin + z * self.rot_cos + self.offset_z * wavelength;

        // Domain warp: displace the sample point by a coarse vector field
        // before evaluating anything. Applied in metres, so every band is
        // warped by the same displacement and they stay registered.
        let (gx, gz, jxx, jxz, jzx, jzz) = if self.warp != 0.0 {
            let (px, pz) = (mx * inv_w, mz * inv_w);
            let (wx, wxu, wxv) = noise::perlin_d(
                self.seed ^ WARP_SEED_X,
                px * WARP_FREQUENCY + WARP_OFFSET_X,
                pz * WARP_FREQUENCY + WARP_OFFSET_Z,
            );
            let (wz, wzu, wzv) = noise::perlin_d(
                self.seed ^ WARP_SEED_Z,
                px * WARP_FREQUENCY + WARP_OFFSET_Z,
                pz * WARP_FREQUENCY + WARP_OFFSET_X,
            );
            let g = self.warp * WARP_FREQUENCY;
            // Jacobian of the warped position with respect to the unwarped one.
            // The `wavelength` factors cancel: the displacement is
            // `warp * wavelength * w`, and `w`'s derivative carries `inv_w`.
            (
                mx + self.warp * wavelength * wx,
                mz + self.warp * wavelength * wz,
                1.0 + g * wxu,
                g * wxv,
                g * wzu,
                1.0 + g * wzv,
            )
        } else {
            (mx, mz, 1.0, 0.0, 0.0, 1.0)
        };

        // --- Band 1: the landscape. ---
        let (base, base_dx, base_dz) =
            fbm(self.seed, gx * inv_w, gz * inv_w, self.octaves, self.lacunarity, self.gain);
        let mut h = self.amplitude * base;
        let mut dx = self.amplitude * base_dx * inv_w;
        let mut dz = self.amplitude * base_dz * inv_w;

        // --- Band 3: how savage the ground is here. ---
        // `m` runs 0 (calm) to 1 (savage) and is slow. Computed before the
        // detail band because it scales it, and before the terracing because it
        // can also mask that.
        let (m, mdx, mdz) = if self.modulation > 0.0 {
            let mw = 1.0 / self.modulation_wavelength.max(1e-3);
            let (v, vdx, vdz) = noise::perlin_d(
                self.seed ^ MODULATION_SEED,
                gx * mw + MODULATION_OFFSET_X,
                gz * mw + MODULATION_OFFSET_Z,
            );
            // `smootherstep` of the noise, so the transition between calm and
            // savage is gradual and its derivative vanishes at both ends.
            let (s, ds) = smootherstep(0.5 + 0.5 * v);
            let k = self.modulation;
            (1.0 - k + k * s, k * ds * 0.5 * vdx * mw, k * ds * 0.5 * vdz * mw)
        } else {
            (1.0, 0.0, 0.0)
        };

        // --- Band 2: detail at organism scale, scaled by the modulation. ---
        if self.detail_amplitude > 0.0 {
            let dw = 1.0 / self.detail_wavelength.max(1e-3);
            let (det, det_dx, det_dz) = fbm(
                self.seed ^ DETAIL_SEED,
                gx * dw,
                gz * dw,
                self.detail_octaves,
                self.lacunarity,
                self.gain,
            );
            h += self.detail_amplitude * m * det;
            dx += self.detail_amplitude * (mdx * det + m * det_dx * dw);
            dz += self.detail_amplitude * (mdz * det + m * det_dz * dw);
        }

        // --- Band 4: cliffs. ---
        if self.step > 0.0 {
            let riser = self.riser.clamp(1e-3, 1.0);
            let t = h / self.step;
            let floor = t.floor();
            let (s, ds) = smootherstep((t - floor - 0.5) / riser + 0.5);
            let scale = ds / riser;
            let (terraced, tdx, tdz) = ((floor + s) * self.step, dx * scale, dz * scale);
            if self.terrace_mask {
                // Blend hills into badlands, weighted by the same slow field
                // that drives the detail: terraced where `m` is high, smooth
                // where it is low. `gap * dm` is the blend weight's own
                // contribution, and dropping it is the easiest way to get a
                // normal that does not match the surface.
                let gap = terraced - h;
                dx = dx + m * (tdx - dx) + mdx * gap;
                dz = dz + m * (tdz - dz) + mdz * gap;
                h += m * gap;
            } else {
                h = terraced;
                dx = tdx;
                dz = tdz;
            }
        }

        // Out through the warp and the rotation. `jxx..jzz` is transposed here
        // because a gradient is a covector.
        let wx = dx * jxx + dz * jzx;
        let wz = dx * jxz + dz * jzz;
        (h, wx * self.rot_cos + wz * self.rot_sin, wz * self.rot_cos - wx * self.rot_sin)
    }

    /// The largest height this field can produce, in metres.
    ///
    /// Exact rather than measured: each octave of gradient noise is bounded by
    /// one, so each band is bounded by the sum of its octave weights.
    /// Terracing cannot exceed it either — quantising a value moves it by less
    /// than one step, and the bound already covers that.
    pub fn height_bound(&self) -> Real {
        let band = |octaves: u32, gain: Real| {
            let mut total = 0.0;
            let mut weight = 1.0;
            for _ in 0..octaves.min(MAX_TERRAIN_OCTAVES) {
                total += weight;
                weight *= gain.abs();
            }
            total
        };
        let g = self.gain.abs();
        self.amplitude.abs() * band(self.octaves, g)
            + self.detail_amplitude.abs() * band(self.detail_octaves, g)
            + self.step.abs()
    }

    /// The gradient of this field with terracing switched off, at the 90th
    /// percentile of a sample of it.
    ///
    /// Ninetieth rather than median because that is where the thin walls are:
    /// a riser compresses a terrace's whole height change into `riser` of its
    /// span, so it is steepest, and therefore narrowest, exactly where the
    /// underlying ground was already steep. Measured rather than derived — an
    /// analytic estimate summing the octaves' contributions came out five times
    /// wrong against the field it was estimating, because the gradient of
    /// fractional Brownian motion is dominated by its finest octave in a way
    /// that is easy to get backwards.
    ///
    /// Sampled on a coarse, deliberately awkward grid. This runs once per config
    /// load, not per step.
    pub fn smooth_gradient(&self) -> Real {
        let smooth = FractalField { step: 0.0, ..*self };
        let mut g = Vec::with_capacity(21 * 21);
        for a in -10..11 {
            for b in -10..11 {
                let (_, dx, dz) =
                    smooth.height_and_gradient(a as Real * 3.7 + 0.31, b as Real * 4.3 - 0.17);
                g.push((dx * dx + dz * dz).sqrt());
            }
        }
        g.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        g[(g.len() * 9) / 10]
    }

    /// How wide this field's terrace risers are, in metres.
    ///
    /// This is the number that decides whether a cliff is physics or a bug. A
    /// body moving at `v` covers `v * dt` per step, and a wall it crosses in one
    /// step is a wall the solver meets as a single enormous penetration — or
    /// tunnels through entirely. `Config::validate` uses this.
    ///
    /// The riser occupies `step * riser` of height, so on ground of gradient `g`
    /// it occupies `step * riser / g` of horizontal distance. Checked against
    /// `examples/terrain_probe.rs`, which measures the walls the field actually
    /// contains: this predicts 150 mm for the shipped settings and the probe
    /// finds a median of 160.
    pub fn riser_width(&self) -> Real {
        if self.step <= 0.0 {
            return Real::INFINITY;
        }
        self.step * self.riser / self.smooth_gradient().max(1e-3)
    }
}

/// Fractional Brownian motion: octaves of gradient noise, each finer and
/// shallower than the last. Returns the value and its gradient in the
/// coordinates it was given.
#[inline]
fn fbm(
    seed: u64,
    x: Real,
    z: Real,
    octaves: u32,
    lacunarity: Real,
    gain: Real,
) -> (Real, Real, Real) {
    let mut sum = 0.0;
    let mut dx = 0.0;
    let mut dz = 0.0;
    let mut frequency = 1.0;
    let mut weight = 1.0;
    for o in 0..octaves.min(MAX_TERRAIN_OCTAVES) {
        let step = (o + 1) as Real;
        let (n, nu, nv) = noise::perlin_d(
            seed.wrapping_add((o as u64).wrapping_mul(OCTAVE_SEED_STRIDE)),
            x * frequency + step * OCTAVE_OFFSET_X,
            z * frequency + step * OCTAVE_OFFSET_Z,
        );
        sum += weight * n;
        dx += weight * frequency * nu;
        dz += weight * frequency * nv;
        frequency *= lacunarity;
        weight *= gain;
    }
    (sum, dx, dz)
}

/// `6t^5 - 15t^4 + 10t^3` clamped to `[0, 1]`, and its derivative.
///
/// Zero first derivative at both ends, so anything built by blending with it
/// stays continuously differentiable where the pieces meet — which is what
/// keeps the terrace risers and the badlands mask from putting creases in the
/// surface normal.
#[inline]
fn smootherstep(t: Real) -> (Real, Real) {
    if t <= 0.0 {
        return (0.0, 0.0);
    }
    if t >= 1.0 {
        return (1.0, 0.0);
    }
    let t2 = t * t;
    (t2 * t * (t * (t * 6.0 - 15.0) + 10.0), 30.0 * t2 * (t - 1.0) * (t - 1.0))
}

/// A `u64` that survives a round trip through JavaScript. See
/// [`FractalField::seed`].
mod seed_as_string {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &u64, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(v)
    }

    /// Accepts a number as well as a string, so that a trace hand-edited into
    /// the obvious shape still loads.
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Either {
            Text(String),
            Number(u64),
        }
        match Either::deserialize(d)? {
            Either::Number(n) => Ok(n),
            Either::Text(s) => s.parse().map_err(serde::de::Error::custom),
        }
    }
}

/// Hardest limit on the octave loop, and therefore on the cost of one sample.
/// Beyond about six octaves the finest is far below the size of any body part
/// and only costs time. `Config::validate` enforces the same bound.
pub const MAX_TERRAIN_OCTAVES: u32 = 8;

/// Per-octave displacement in field space.
///
/// Perlin noise is exactly zero at every lattice point, and with an integer
/// `lacunarity` every octave shares a lattice point at the origin — which is
/// where every organism spawns. Without these offsets the spawn point would sit
/// in a dead flat dimple whatever the seed, and the flatness would not show up
/// in any aggregate statistic.
const OCTAVE_OFFSET_X: Real = 0.513_7;
const OCTAVE_OFFSET_Z: Real = 0.942_1;

/// The warp field is sampled at half the base frequency: it has to be coarser
/// than what it is warping, or it merely adds noise instead of shaping it.
const WARP_FREQUENCY: Real = 0.5;
const WARP_OFFSET_X: Real = 3.311;
const WARP_OFFSET_Z: Real = -1.749;

/// Sub-seeds for the two warp components, mixed with the field seed so that
/// changing the seed moves the warp too.
const WARP_SEED_X: u64 = 0x5741_5250_5f58_0001;
const WARP_SEED_Z: u64 = 0x5741_5250_5f5a_0001;
/// Sub-seeds for the bands that were added after the first. Distinct from the
/// base seed so that a band is not a rescaled copy of the landscape under it.
const DETAIL_SEED: u64 = 0x4445_5441_494c_0001;
const MODULATION_SEED: u64 = 0x4d4f_4455_4c41_5445;
/// Offsets keeping the modulation field off the lattice at the origin, for the
/// same reason as [`OCTAVE_OFFSET_X`].
const MODULATION_OFFSET_X: Real = 7.13;
const MODULATION_OFFSET_Z: Real = -2.71;
/// Separation between octaves, so no two octaves are the same field rescaled.
const OCTAVE_SEED_STRIDE: u64 = 0x4f43_5441_5645_0001;

/// Spacing between samples along a sensor ray, metres.
///
/// The resolution of every range sensor in the simulator, and the width of the
/// narrowest feature one can see: terrain that rises above the ray and drops
/// back within a stride is stepped over. 100 mm is chosen against the terrace
/// risers the fractal field produces, which measure about 152 mm at the shipped
/// settings and are the single most important thing for an organism to notice.
const MARCH_STRIDE: Real = 0.1;
/// Ceiling on samples per ray, so a large `range` cannot make evaluation
/// arbitrarily expensive.
const MAX_MARCH_STEPS: u32 = 128;
/// Halvings used to refine the crossing once a stride containing it is found.
const BISECTIONS: u32 = 4;

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
            TerrainModel::Fractal { .. } => self.sample(x, z).0,
        }
    }

    /// Distance from `origin` along `dir` to the first point where the ray meets
    /// the ground, or `None` if it does not within `range`.
    ///
    /// `dir` must be normalised. An origin already below the surface returns
    /// `Some(0.0)`: a sensor buried in a hillside sees the hillside, which is
    /// both true and the reading that keeps a controller's input bounded.
    ///
    /// # Why a fixed march rather than a root find
    ///
    /// The height field is cheap and everywhere-defined but not Lipschitz-bounded
    /// in any form the solver knows, so sphere tracing has nothing to step by.
    /// Marching at a fixed stride until `ray.y - h(ray.x, ray.z)` changes sign and
    /// then bisecting is simple, has no failure mode worse than missing a feature
    /// narrower than the stride, and — the part that matters here — costs the
    /// *same* number of samples for every ray.
    ///
    /// That last property is not an optimisation. A loop that stopped early would
    /// make evaluation cost a function of what evolved and of where an organism
    /// happened to be standing, which turns a benchmark into a measurement of the
    /// population. It also keeps the work identical on every thread, which is
    /// what the determinism contract needs.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, range: Real) -> Option<Real> {
        let above = |p: Vec3| p.y - self.height_at(p.x, p.z);
        if above(origin) <= 0.0 {
            return Some(0.0);
        }

        // Steps follow from the stride, not the other way round. A fixed *count*
        // would make the sensor's resolution depend on its range, so a
        // longer-sighted experiment would quietly become blind to terrace
        // risers — which at the shipped settings are 152 mm wide and are
        // precisely the feature worth seeing. Fixing the stride instead makes
        // resolution a stated physical property, and keeps the sample count
        // identical for every ray in an experiment.
        let steps = ((range / MARCH_STRIDE).ceil() as u32).clamp(1, MAX_MARCH_STEPS);
        let stride = range / steps as Real;
        let mut near = 0.0;
        let mut hit = false;
        for i in 1..=steps {
            let far = i as Real * stride;
            if above(origin + dir * far) <= 0.0 {
                hit = true;
                break;
            }
            near = far;
        }
        if !hit {
            return None;
        }

        // The crossing is somewhere in `(near, near + stride]`. Bisection rather
        // than a secant step because it cannot diverge on a discontinuous-looking
        // terrace riser, and four halvings of a stride this size are already
        // finer than the contact solver resolves.
        let mut lo = near;
        let mut hi = near + stride;
        for _ in 0..BISECTIONS {
            let mid = 0.5 * (lo + hi);
            if above(origin + dir * mid) <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        Some(0.5 * (lo + hi))
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
            TerrainModel::Fractal { .. } => self.sample(x, z).1,
        }
    }

    /// Height and surface normal at one point, together.
    ///
    /// Every contact needs both, and for all three models they share nearly all
    /// of their arithmetic — so asking for them separately does the work twice
    /// on the hottest path in the simulator. Callers with both in hand should
    /// prefer this to a `height_at` followed by a `normal_at`.
    ///
    /// Bit-identical to calling the two separately, for every model. That is
    /// what keeps the goldens where they are.
    #[inline]
    pub fn sample(&self, x: Real, z: Real) -> (Real, Vec3) {
        match *self {
            TerrainModel::Flat { height } => (height, Vec3::Y),
            TerrainModel::Rough { amplitude, wavelength } => {
                // `dsin` and `dcos` are both projections of `dsincos`, so taking
                // the pairs once is the same arithmetic in the same order — and
                // half the trig.
                let k = TAU / wavelength.max(1e-3);
                let (sx, cx) = dsincos(k * x);
                let (sz, cz) = dsincos(k * z);
                let (sx2, cx2) = dsincos(2.0 * k * x + 1.7);
                let (sz2, cz2) = dsincos(2.0 * k * z + 0.9);
                let h = amplitude * (sx * cz) + 0.5 * amplitude * (sx2 * cz2);
                let dhdx = amplitude * k * cx * cz + amplitude * k * cx2 * cz2;
                let dhdz = -amplitude * k * sx * sz - amplitude * k * sx2 * sz2;
                (h, vec3(-dhdx, 1.0, -dhdz).normalize_or(Vec3::Y))
            }
            TerrainModel::Fractal(f) => {
                let (h, dhdx, dhdz) = f.height_and_gradient(x, z);
                (h, vec3(-dhdx, 1.0, -dhdz).normalize_or(Vec3::Y))
            }
        }
    }

    /// How level the ground is within `radius` of `(x, z)`, as the smallest `y`
    /// component of the surface normal found there: 1 is a flat plateau, 0 is a
    /// vertical face.
    ///
    /// Used to choose somewhere fair to set an organism down. On smooth ground
    /// this is a formality; on terraced ground it is not, because an organism
    /// spawned straddling a riser starts half inside a wall, and what the solver
    /// does about that is not a fair test of a gait.
    ///
    /// Nine samples — the centre and a ring of eight — which is enough to catch
    /// a riser crossing the footprint and cheap enough to run a dozen times per
    /// trial.
    pub fn levelness_near(&self, x: Real, z: Real, radius: Real) -> Real {
        const RING: [(Real, Real); 8] = [
            (1.0, 0.0),
            (0.707_106_77, 0.707_106_77),
            (0.0, 1.0),
            (-0.707_106_77, 0.707_106_77),
            (-1.0, 0.0),
            (-0.707_106_77, -0.707_106_77),
            (0.0, -1.0),
            (0.707_106_77, -0.707_106_77),
        ];
        let mut worst = self.normal_at(x, z).y;
        for (dx, dz) in RING {
            worst = worst.min(self.normal_at(x + dx * radius, z + dz * radius).y);
        }
        worst
    }

    /// The largest height this model can produce, in metres.
    ///
    /// Exact rather than measured: each octave of gradient noise is bounded by
    /// one, so the sum is bounded by the sum of the octave weights. Used to
    /// document what an `amplitude` setting actually buys, and asserted against
    /// in the tests.
    pub fn height_bound(&self) -> Real {
        match *self {
            TerrainModel::Flat { height } => height.abs(),
            TerrainModel::Rough { amplitude, .. } => 1.5 * amplitude.abs(),
            TerrainModel::Fractal(f) => f.height_bound(),
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
    /// Accumulated *positional* normal impulse, kept apart from `pn` so the two
    /// halves of the solve each converge against their own history.
    pn_bias: Real,
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
    /// Accumulated positional normal impulse. See [`World::bias_lin`].
    pn_bias: Real,
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
    /// Positional correction, held as a velocity that never becomes momentum.
    ///
    /// Every constraint here has two jobs: stop bodies moving into each other,
    /// and undo the overlap they are already in. The second used to be done by
    /// adding a Baumgarte bias straight into `lin_vel`, which works and is also
    /// a motor: the velocity it injects to separate two bodies is still there
    /// after they have separated. An organism whose own parts kept
    /// re-penetrating collected `max_correction_speed` every step and kept it,
    /// and evolution found that long before it found walking — champions that
    /// crossed twenty-six metres with their motors switched off.
    ///
    /// So the correction is accumulated here instead, used only to displace
    /// bodies in [`Self::integrate_positions`], and discarded at the end of the
    /// step. Bodies still separate; separating no longer pays. This is Catto's
    /// split impulse, and it is why `solve_*` below each have a velocity half
    /// and a position half that look almost the same.
    bias_lin: Vec<Vec3>,
    bias_ang: Vec<Vec3>,
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
            bias_lin: vec![Vec3::ZERO; n],
            bias_ang: vec![Vec3::ZERO; n],
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

        // The positional correction starts each step from nothing. It is a
        // property of the overlap that exists right now, not a quantity a body
        // is allowed to carry from one step to the next — carrying it is
        // exactly what made it a motor. See [`Self::bias_lin`].
        for i in 0..self.bodies.len() {
            self.bias_lin[i] = Vec3::ZERO;
            self.bias_ang[i] = Vec3::ZERO;
        }

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

    /// Move bodies by their velocity *plus* the step's positional correction.
    ///
    /// The correction displaces and is then dropped: it never reaches
    /// `lin_vel`, so a body that has been pushed out of an overlap is left
    /// exactly as fast as it was before. It is clamped on the same terms as the
    /// individual constraint biases that produced it, because a pathological
    /// mutated body can pile up dozens of overlapping contacts and their
    /// corrections all point the same way.
    fn integrate_positions(&mut self, dt: Real) {
        let max_v = self.params.max_linear_speed;
        let max_w = self.params.max_angular_speed;
        let max_corr = self.params.max_correction_speed;
        for (i, b) in self.bodies.iter_mut().enumerate() {
            clamp_speed(&mut b.lin_vel, max_v);
            clamp_speed(&mut b.ang_vel, max_w);
            let mut bias_lin = self.bias_lin[i];
            let mut bias_ang = self.bias_ang[i];
            clamp_speed(&mut bias_lin, max_corr);
            clamp_speed(&mut bias_ang, max_corr * 4.0);
            b.pos += (b.lin_vel + bias_lin) * dt;
            b.orient = b.orient.integrate(b.ang_vel + bias_ang, dt);
        }
    }

    /// Velocity of a body-fixed point under the positional correction alone.
    #[inline]
    fn bias_point_velocity(&self, i: usize, r: Vec3) -> Vec3 {
        self.bias_lin[i] + self.bias_ang[i].cross(r)
    }

    /// The positional-correction counterpart of [`RigidBody::apply_impulse`].
    #[inline]
    fn apply_bias_impulse(&mut self, i: usize, r: Vec3, impulse: Vec3) {
        self.bias_lin[i] += impulse * self.bodies[i].inv_mass;
        self.bias_ang[i] += self.inv_inertia[i].mul_vec(r.cross(impulse));
    }

    #[inline]
    fn apply_bias_angular_impulse(&mut self, i: usize, impulse: Vec3) {
        self.bias_ang[i] += self.inv_inertia[i].mul_vec(impulse);
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
                    pn_bias: 0.0,
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

            // Velocity half: stop the parts driving further into each other.
            // The normal points from a to b, so separating means vn > 0.
            let relative =
                self.bodies[ib].point_velocity(c.r_b) - self.bodies[ia].point_velocity(c.r_a);
            let vn = relative.dot(c.normal);
            let mut lambda = -vn / c.k_n;
            let old = c.pn;
            let new = (old + lambda).max(0.0);
            lambda = new - old;
            self.pair_contacts[i].pn = new;
            if lambda != 0.0 {
                let impulse = c.normal * lambda;
                self.bodies[ia].apply_impulse(c.r_a, -impulse, &inv_ia);
                self.bodies[ib].apply_impulse(c.r_b, impulse, &inv_ib);
            }

            // Position half. This is the one that mattered: shoving overlapping
            // limbs apart at up to `max_correction_speed` and *keeping* the
            // velocity was a rocket any sprawling body could fire, and the
            // whole population found it.
            let bias = clamp((c.depth - slop).max(0.0) * beta * inv_dt, 0.0, max_corr);
            if bias > 0.0 {
                let vb = (self.bias_point_velocity(ib, c.r_b)
                    - self.bias_point_velocity(ia, c.r_a))
                .dot(c.normal);
                let mut lb = (bias - vb) / c.k_n;
                let old_b = c.pn_bias;
                let new_b = (old_b + lb).max(0.0);
                lb = new_b - old_b;
                self.pair_contacts[i].pn_bias = new_b;
                if lb != 0.0 {
                    let impulse = c.normal * lb;
                    self.apply_bias_impulse(ia, c.r_a, -impulse);
                    self.apply_bias_impulse(ib, c.r_b, impulse);
                }
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
                // One sample for both: height and normal share nearly all of
                // their arithmetic, and this is the innermost loop in the
                // simulator — up to eight points per body per step.
                let (ground, normal) = terrain.sample(corner.x, corner.z);
                let drop = ground - corner.y;
                if drop <= 0.0 {
                    continue;
                }
                // `drop` is how far the point is below the surface *vertically*,
                // but the contact is resolved along the surface normal, and on a
                // slope those are not the same distance: the vertical measure
                // overstates the true perpendicular penetration by 1/cos(slope)
                // — 1.4x at 45 degrees, 7x at 82. Since the positional
                // correction is proportional to depth, leaving it uncorrected
                // makes every steep face push about seven times too hard, and
                // the steeper the ground the harder it shoves. `normal.y` is
                // exactly that cosine, and the conversion is exact for a
                // locally flat surface.
                let depth = drop * normal.y;
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
                    pn_bias: 0.0,
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

            // Normal, velocity half: stop the body moving into the ground, and
            // bounce it if the impact was hard enough. No positional term —
            // that is the job of the bias half below, and mixing the two is
            // what used to make the ground a motor.
            let vn = self.bodies[bi].point_velocity(c.r).dot(c.normal);
            let mut lambda = (c.bounce - vn) / c.k_n;
            let new_pn = (c.pn + lambda).max(0.0);
            lambda = new_pn - c.pn;
            self.contacts[ci].pn = new_pn;
            if lambda != 0.0 {
                let p = c.normal * lambda;
                self.bodies[bi].apply_impulse(c.r, p, &inv_i);
            }

            // Normal, position half: separate what is already overlapping,
            // into a velocity that only ever displaces.
            let correction = clamp(beta * (c.depth - slop).max(0.0) * inv_dt, 0.0, max_corr);
            if correction > 0.0 {
                let vb = self.bias_point_velocity(bi, c.r).dot(c.normal);
                let mut lb = (correction - vb) / c.k_n;
                let new_pb = (c.pn_bias + lb).max(0.0);
                lb = new_pb - c.pn_bias;
                self.contacts[ci].pn_bias = new_pb;
                if lb != 0.0 {
                    self.apply_bias_impulse(bi, c.r, c.normal * lb);
                }
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
            // Velocity half: hold the anchors moving together.
            let v_rel = self.bodies[ib].point_velocity(p.rb) - self.bodies[ia].point_velocity(p.ra);
            let impulse = p.k_point.solve(-v_rel);
            self.bodies[ia].apply_impulse(p.ra, -impulse, &inv_ia);
            self.bodies[ib].apply_impulse(p.rb, impulse, &inv_ib);

            // Position half: pull apart anchors that have already drifted. A
            // joint stretched every step by a limb it cannot hold would
            // otherwise be a motor in exactly the way self-collision was.
            let anchor_a = self.bodies[ia].pos + p.ra;
            let anchor_b = self.bodies[ib].pos + p.rb;
            let error = anchor_b - anchor_a;
            let mut bias = error * (-beta * inv_dt);
            clamp_speed(&mut bias, max_corr);
            let vb_rel = self.bias_point_velocity(ib, p.rb) - self.bias_point_velocity(ia, p.ra);
            let bias_impulse = p.k_point.solve(bias - vb_rel);
            self.apply_bias_impulse(ia, p.ra, -bias_impulse);
            self.apply_bias_impulse(ib, p.rb, bias_impulse);

            // --- Angular. ---
            match j.kind {
                JointKind::Fixed => {
                    // Drive the relative rotation back to the rest pose. Bodies
                    // start axis-aligned, so the rest relative rotation is the
                    // identity and the error is just the relative quaternion.
                    let w_rel = self.bodies[ib].ang_vel - self.bodies[ia].ang_vel;
                    let ang_impulse = p.k_ang.solve(-w_rel);
                    self.bodies[ia].apply_angular_impulse(-ang_impulse, &inv_ia);
                    self.bodies[ib].apply_angular_impulse(ang_impulse, &inv_ib);

                    let q_rel = self.bodies[ib].orient.mul(self.bodies[ia].orient.conjugate());
                    let sign = if q_rel.w < 0.0 { -1.0 } else { 1.0 };
                    let err = q_rel.vec() * (2.0 * sign);
                    let mut ang_bias = err * (-beta * inv_dt);
                    clamp_speed(&mut ang_bias, max_corr * 4.0);
                    let wb_rel = self.bias_ang[ib] - self.bias_ang[ia];
                    let bias_impulse = p.k_ang.solve(ang_bias - wb_rel);
                    self.apply_bias_angular_impulse(ia, -bias_impulse);
                    self.apply_bias_angular_impulse(ib, bias_impulse);
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
                        let lambda = -w_rel.dot(t) / k;
                        let imp = t * lambda;
                        self.bodies[ia].apply_angular_impulse(-imp, &inv_ia);
                        self.bodies[ib].apply_angular_impulse(imp, &inv_ib);

                        let target = clamp(
                            -beta * inv_dt * misalign.dot(t),
                            -max_corr * 4.0,
                            max_corr * 4.0,
                        );
                        let wb_rel = self.bias_ang[ib] - self.bias_ang[ia];
                        let lb = (target - wb_rel.dot(t)) / k;
                        let imp_b = t * lb;
                        self.apply_bias_angular_impulse(ia, -imp_b);
                        self.apply_bias_angular_impulse(ib, imp_b);
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

        // Velocity half: stop the joint rotating further past its limit. It is
        // allowed to come back on its own; it is not allowed to keep going.
        let w_rel = self.bodies[ib].ang_vel - self.bodies[ia].ang_vel;
        let rate = dir * w_rel.dot(p.axis_w);
        if rate > 0.0 {
            let lambda = -rate / p.k_axis;
            let imp = p.axis_w * (lambda * dir);
            self.bodies[ia].apply_angular_impulse(-imp, &inv_ia);
            self.bodies[ib].apply_angular_impulse(imp, &inv_ib);
        }

        // Position half: unwind the overshoot that already happened. Overshoot
        // is measured in cosine rather than radians — monotone in |theta| over
        // the half-turn a hinge limit can occupy, and free of `acos`.
        let overshoot = j.cos_limit - cos_theta;
        let push_back = -clamp(
            self.params.baumgarte * overshoot / dt,
            0.0,
            self.params.max_correction_speed * 4.0,
        );
        let wb_rel = self.bias_ang[ib] - self.bias_ang[ia];
        let bias_rate = dir * wb_rel.dot(p.axis_w);
        if bias_rate > push_back {
            let lb = (push_back - bias_rate) / p.k_axis;
            let imp = p.axis_w * (lb * dir);
            self.apply_bias_angular_impulse(ia, -imp);
            self.apply_bias_angular_impulse(ib, imp);
        }
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

    /// The ray against an independent brute-force march, which is how the
    /// terrain gradient was validated and for the same reason: a root find that
    /// agrees with itself proves nothing.
    #[test]
    fn a_ray_finds_the_same_ground_a_dense_march_does() {
        let fields = [
            TerrainModel::Flat { height: 0.0 },
            TerrainModel::Rough { amplitude: 0.25, wavelength: 2.0 },
            TerrainModel::Fractal(FractalField {
                seed: 7,
                amplitude: 3.0,
                wavelength: 25.0,
                octaves: 5,
                lacunarity: 2.0,
                gain: 0.5,
                warp: 0.6,
                detail_amplitude: 0.35,
                detail_wavelength: 3.0,
                detail_octaves: 4,
                modulation: 0.9,
                modulation_wavelength: 35.0,
                step: 0.8,
                riser: 0.12,
                terrace_mask: true,
                ..Default::default()
            }),
        ];
        let range = 6.0;
        // A stride, refined by four bisections, is all the march can resolve.
        let tol = 0.1 / 16.0 + 1e-3;
        let mut checked = 0;
        let mut missed = 0;
        for terrain in fields {
            for i in 0..400 {
                // Deliberately awkward origins and bearings: on a lattice point
                // every octave of gradient noise is exactly zero, so round
                // numbers would agree between two different implementations.
                let a = i as Real * 0.37;
                let x = (a * 1.7) % 23.0 - 11.5;
                let z = (a * 2.9) % 19.0 - 9.5;
                let origin = vec3(x, terrain.height_at(x, z) + 0.6, z);
                let dir = vec3(
                    ((a * 0.61) % 2.0) - 1.0,
                    ((a * 0.29) % 1.4) - 1.2,
                    ((a * 0.83) % 2.0) - 1.0,
                )
                .normalize_or(vec3(0.0, -1.0, 0.0));

                let got = terrain.raycast(origin, dir, range);

                // Brute force: step finely and take the first crossing.
                const FINE: u32 = 4000;
                let mut expected = None;
                for k in 1..=FINE {
                    let t = k as Real * (range / FINE as Real);
                    let p = origin + dir * t;
                    if p.y - terrain.height_at(p.x, p.z) <= 0.0 {
                        expected = Some(t);
                        break;
                    }
                }

                match (got, expected) {
                    (Some(g), Some(e)) if (g - e).abs() < tol => checked += 1,
                    (None, None) => checked += 1,
                    // Terrain that rises above the ray and drops back within one
                    // stride is invisible by construction, and the march then
                    // either finds a later crossing or none. That is the stated
                    // resolution limit rather than a bug, so it is counted and
                    // bounded rather than forgiven case by case.
                    (Some(_), Some(_)) | (None, Some(_)) => missed += 1,
                    (Some(g), None) => panic!("ray {i}: reported a hit at {g} that is not there"),
                }
            }
        }
        // Sub-stride features are missed by construction; the contract is that
        // they are rare. A regression that broke the march outright, or coarsened
        // the stride, shows up here immediately.
        assert!(
            missed * 50 < checked,
            "{missed} of {} rays disagreed with a dense march; the resolution limit              should account for far fewer than 2%",
            checked + missed
        );
        assert!(checked > 1000, "only {checked} rays actually agreed");
    }

    /// A sensor already under the surface sees the surface, rather than
    /// reporting nothing and letting a buried organism believe it is in clear
    /// air.
    #[test]
    fn a_ray_from_below_the_ground_hits_immediately() {
        let t = TerrainModel::Rough { amplitude: 0.3, wavelength: 2.0 };
        let below = vec3(0.4, t.height_at(0.4, 0.2) - 0.1, 0.2);
        assert_eq!(t.raycast(below, vec3(1.0, 0.0, 0.0), 4.0), Some(0.0));
    }

    /// A ray pointing away from the ground finds nothing, and says so.
    #[test]
    fn a_ray_into_the_sky_finds_nothing() {
        let t = TerrainModel::Flat { height: 0.0 };
        assert_eq!(t.raycast(vec3(0.0, 0.5, 0.0), Vec3::Y, 10.0), None);
    }

    /// Straight down onto flat ground is the one case with an exact answer.
    #[test]
    fn a_vertical_ray_measures_its_own_height() {
        let t = TerrainModel::Flat { height: 0.0 };
        let d = t.raycast(vec3(1.0, 2.0, -3.0), vec3(0.0, -1.0, 0.0), 8.0).expect("hit");
        // Resolved to a stride refined by `BISECTIONS` halvings, and no finer;
        // the march does not claim more precision than that.
        let tol = MARCH_STRIDE / (1 << BISECTIONS) as Real;
        assert!((d - 2.0).abs() <= tol, "expected 2 m within {tol}, got {d}");
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

    /// Internal forces cannot move a centre of mass.
    ///
    /// A motor and a joint exchange momentum between an organism's own parts.
    /// They can spin it, fold it and tear it apart, but with no gravity and
    /// nothing to touch, the mass-weighted mean of its parts must stay exactly
    /// where it started. Anything else is the solver inventing momentum, and
    /// evolution finds invented momentum faster than it finds walking.
    ///
    /// Driven hard and reversed repeatedly, because that is the regime evolved
    /// controllers actually use and the one where the per-body clamps in
    /// `integrate_positions` could clip one half of an equal-and-opposite pair.
    #[test]
    fn a_motor_cannot_move_the_centre_of_mass() {
        let mut w = hinge_pair(-1.0, 40.0);
        w.params.gravity = Vec3::ZERO;
        // Far above any ground, and the terrain is flat at zero anyway.
        for b in w.bodies.iter_mut() {
            b.pos.y += 50.0;
        }

        let com = |w: &World| {
            let mut acc = Vec3::ZERO;
            let mut total = 0.0;
            for b in &w.bodies {
                let m = b.mass();
                acc += b.pos * m;
                total += m;
            }
            acc * (1.0 / total)
        };

        let start = com(&w);
        let dt = 1.0 / 120.0;
        for step in 0..1200 {
            // Slam the motor from one extreme to the other every few steps.
            w.joints[0].motor_target = if (step / 3) % 2 == 0 { 4.0 } else { -4.0 };
            w.step(dt);
        }
        let drift = (com(&w) - start).length();
        assert!(
            drift < 1e-3,
            "a motor with nothing to push against moved the centre of mass {drift} m"
        );
    }

    /// The same conservation law, with the parts a real organism actually has:
    /// a joint limit to slam into and a tendon pulling back.
    ///
    /// The plain hinge above conserves momentum exactly, so anything that leaks
    /// leaks here — the limit's positional half and the tendon are the two
    /// places a torque is applied that is not obviously equal and opposite.
    #[test]
    fn a_limit_and_a_tendon_cannot_move_the_centre_of_mass() {
        // cos_limit 0.5 is a +/- 60 degree range, so a motor driven flat out
        // hits the stop and stays there.
        let mut w = hinge_pair(0.5, 40.0);
        w.params.gravity = Vec3::ZERO;
        w.joints[0].tendon_frequency = 6.0;
        w.joints[0].tendon_damping = 0.5;
        for b in w.bodies.iter_mut() {
            b.pos.y += 50.0;
        }

        let com = |w: &World| {
            let mut acc = Vec3::ZERO;
            let mut total = 0.0;
            for b in &w.bodies {
                let m = b.mass();
                acc += b.pos * m;
                total += m;
            }
            acc * (1.0 / total)
        };

        let start = com(&w);
        let dt = 1.0 / 120.0;
        for step in 0..1200 {
            w.joints[0].motor_target = if (step / 3) % 2 == 0 { 4.0 } else { -4.0 };
            w.step(dt);
        }
        let drift = (com(&w) - start).length();
        assert!(
            drift < 1e-3,
            "a motor slamming a joint limit moved the centre of mass {drift} m with              nothing to push against"
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

    /// Self-collision must not be a motor.
    ///
    /// This is the shape of the bug that inflated every result in this project
    /// for two months, reduced to forty lines: three parts in a chain, folded
    /// so the two ends overlap. A pair of loose boxes cannot show it — shoved
    /// apart, they separate once and stop. A folded chain is a *cycle*: the
    /// joints pull the ends back into each other every step, self-collision
    /// shoves them apart again, and when the shove was added straight into
    /// `lin_vel` the pair became an engine that never ran down. Evolved
    /// champions crossed twenty-six metres with their motors switched off, and
    /// no test in the suite objected.
    ///
    /// Nothing here is meant to do work: no motor, no tendon, and the
    /// measurement starts after the chain has settled on flat ground, so there
    /// is not even potential energy left to spend. On the solver this replaced,
    /// it crawls 2.74 m per 7.5 s and keeps doing it indefinitely.
    ///
    /// It is not zero now, and the honest reason is that split impulse stops
    /// the correction becoming *momentum* without stopping it becoming
    /// *displacement*: the ground is immovable, so a body in an internal cycle
    /// can still ratchet against it a fraction of a millimetre at a time. The
    /// residue is 0.46 m per 7.5 s here and unmeasurable on real organisms —
    /// the champions that exploited the old behaviour now travel nothing at
    /// all. The bound below is set to catch a regression toward the old
    /// behaviour, not to certify zero.
    #[test]
    fn self_collision_is_not_a_motor() {
        let dt = 1.0 / 120.0;
        let mut worst: Real = 0.0;
        for fold in 1..10 {
            let reach = 0.30 - fold as Real * 0.028;
            let bodies: Vec<RigidBody> = (0..3)
                .map(|i| {
                    let angle = i as Real * 1.9;
                    let (s, c) = dsincos(angle);
                    RigidBody::box_body(
                        vec3(reach * c, 1.0 + i as Real * 0.05, reach * s),
                        vec3(0.12, 0.12, 0.12),
                        1000.0,
                    )
                })
                .collect();
            let joints: Vec<Joint> = (0..2)
                .map(|i| {
                    let mid = (bodies[i].pos + bodies[i + 1].pos) * 0.5;
                    Joint::fixed(
                        i as u16,
                        i as u16 + 1,
                        mid - bodies[i].pos,
                        mid - bodies[i + 1].pos,
                    )
                })
                .collect();
            let params = WorldParams { self_collision: true, ..WorldParams::default() };
            let mut w = World::new(bodies, joints, params);

            // Let it fall and settle; everything after this starts from rest.
            for _ in 0..240 {
                w.step(dt);
            }
            assert!(!w.diverged, "fold {fold}: diverged while settling");
            let (x0, z0) = (w.bodies[0].pos.x, w.bodies[0].pos.z);
            for _ in 0..900 {
                w.step(dt);
            }
            assert!(!w.diverged, "fold {fold}: diverged");
            let (dx, dz) = (w.bodies[0].pos.x - x0, w.bodies[0].pos.z - z0);
            worst = worst.max((dx * dx + dz * dz).sqrt());
        }
        assert!(worst < 1.0, "a motorless folded chain crawled {worst} m in 7.5 s");
    }

    // ------------------------------------------------------------------ terrain

    /// The shipped landscape band with one seed changed. Everything after band
    /// one is off in `FractalField::default()`, so a case that varies one field
    /// with `..` varies exactly that.
    fn frac(seed: u64) -> FractalField {
        FractalField { seed, wavelength: 6.0, ..FractalField::default() }
    }

    fn fractal(seed: u64) -> TerrainModel {
        TerrainModel::Fractal(frac(seed))
    }

    /// `sample` exists to halve the terrain work on the contact path. It is only
    /// allowed to do that if it is the *same* arithmetic, bit for bit — the
    /// goldens depend on that for `Rough`, and the gradient test below depends
    /// on it for `Fractal`.
    #[test]
    fn sample_agrees_bitwise_with_height_and_normal_taken_separately() {
        let models = [
            TerrainModel::Flat { height: 0.0 },
            TerrainModel::Flat { height: -0.7 },
            TerrainModel::Rough { amplitude: 0.05, wavelength: 1.6 },
            TerrainModel::Rough { amplitude: 0.25, wavelength: 6.0 },
            fractal(0),
            fractal(0xABCD_EF01),
            TerrainModel::Fractal(FractalField { warp: 0.0, ..frac(9) }),
            TerrainModel::Fractal(FractalField { octaves: 1, ..frac(9) }),
        ];
        for m in models {
            for a in -40..40 {
                for b in -40..40 {
                    let (x, z) = (a as Real * 0.313, b as Real * 0.271);
                    let (h, n) = m.sample(x, z);
                    assert_eq!(h.to_bits(), m.height_at(x, z).to_bits(), "height at {x},{z}");
                    let want = m.normal_at(x, z);
                    assert_eq!(
                        (n.x.to_bits(), n.y.to_bits(), n.z.to_bits()),
                        (want.x.to_bits(), want.y.to_bits(), want.z.to_bits()),
                        "normal at {x},{z}"
                    );
                }
            }
        }
    }

    /// The normal is the analytic gradient of the height, and the contact solver
    /// trusts it completely. Central differences over the *assembled* field —
    /// warp, octaves, rotation and all — are the independent check that the
    /// chain rule was carried through correctly.
    #[test]
    fn the_fractal_normal_is_the_gradient_of_its_own_height() {
        let h = 5e-3;
        let variants = [
            fractal(0),
            fractal(1),
            fractal(0xDEAD_BEEF),
            TerrainModel::Fractal(FractalField { warp: 0.0, ..frac(2) }),
            TerrainModel::Fractal(FractalField { warp: 0.9, ..frac(3) }),
            TerrainModel::Fractal(FractalField { octaves: 1, ..frac(4) }),
            TerrainModel::Fractal(FractalField { octaves: 6, ..frac(5) }),
            TerrainModel::Fractal(FractalField { lacunarity: 2.7, gain: 0.65, ..frac(6) }),
            TerrainModel::Fractal(FractalField { wavelength: 1.5, amplitude: 0.05, ..frac(7) }),
            TerrainModel::Fractal(FractalField { rot_sin: 0.6, rot_cos: 0.8, ..frac(8) }),
            TerrainModel::Fractal(FractalField { offset_x: 12.5, offset_z: -7.25, ..frac(9) }),
        ];
        let mut worst = 0.0f64;
        for m in variants {
            for a in -25..25 {
                for b in -25..25 {
                    let (x, z) = (a as Real * 0.731, b as Real * 0.917);
                    let n = m.normal_at(x, z);
                    // Recover the gradient the normal encodes.
                    let (dhdx, dhdz) = (-n.x / n.y, -n.z / n.y);
                    let fdx = (m.height_at(x + h, z) - m.height_at(x - h, z)) / (2.0 * h);
                    let fdz = (m.height_at(x, z + h) - m.height_at(x, z - h)) / (2.0 * h);
                    worst = worst.max((dhdx - fdx).abs() as f64);
                    worst = worst.max((dhdz - fdz).abs() as f64);
                }
            }
        }
        // What is left is finite-difference truncation over a field whose finest
        // octave is a fraction of a metre, not a wrong derivative. A sign error
        // or a dropped warp term lands orders of magnitude above this.
        assert!(worst < 2e-2, "worst gradient error {worst}");
    }

    /// The bands that were added after the first, each varied on its own, and
    /// then all at once. Terracing is the one that will break: it multiplies
    /// the gradient by `1/riser`, so a factor dropped there is a normal that
    /// disagrees with the surface by a factor of eight.
    fn banded() -> FractalField {
        FractalField {
            seed: 0x5EED_0001,
            amplitude: 3.0,
            wavelength: 25.0,
            octaves: 5,
            detail_amplitude: 0.35,
            detail_wavelength: 3.0,
            modulation: 0.9,
            step: 0.8,
            riser: 0.12,
            terrace_mask: true,
            ..FractalField::default()
        }
    }

    #[test]
    fn every_band_has_an_exact_gradient() {
        let h = 1e-3;
        let variants: [(&str, FractalField); 9] = [
            ("landscape only", FractalField { detail_amplitude: 0.0, ..banded() }),
            ("detail, unmodulated", FractalField { modulation: 0.0, step: 0.0, ..banded() }),
            ("detail, modulated", FractalField { step: 0.0, ..banded() }),
            ("terraced, unmasked", FractalField { terrace_mask: false, ..banded() }),
            ("terraced, masked", banded()),
            ("terraced, sheer", FractalField { riser: 0.05, ..banded() }),
            ("terraced, shallow", FractalField { riser: 0.9, ..banded() }),
            ("terraced, no warp", FractalField { warp: 0.0, ..banded() }),
            (
                "everything, moved",
                FractalField {
                    offset_x: 3.5,
                    offset_z: -7.25,
                    rot_sin: 0.6,
                    rot_cos: 0.8,
                    ..banded()
                },
            ),
        ];
        for (label, f) in variants {
            let mut worst = 0.0f64;
            for a in -60..60 {
                for b in -60..60 {
                    let (x, z) = (a as Real * 0.317, b as Real * 0.211);
                    let (_, dx, dz) = f.height_and_gradient(x, z);
                    let fdx = (f.height_and_gradient(x + h, z).0
                        - f.height_and_gradient(x - h, z).0)
                        / (2.0 * h);
                    let fdz = (f.height_and_gradient(x, z + h).0
                        - f.height_and_gradient(x, z - h).0)
                        / (2.0 * h);
                    // A riser is a genuinely steep, genuinely narrow feature, so
                    // a central difference straddling one is measuring the
                    // secant of a cliff rather than its tangent. Compare
                    // relative to the local scale instead of absolutely.
                    let scale = 1.0 + dx.abs().max(dz.abs()) as f64;
                    worst = worst.max((dx - fdx).abs() as f64 / scale);
                    worst = worst.max((dz - fdz).abs() as f64 / scale);
                }
            }
            assert!(worst < 0.05, "{label}: worst relative gradient error {worst}");
        }
    }

    /// Terracing exists to make cliffs, and the mask exists to put them
    /// somewhere rather than everywhere. Both claims are measurable.
    #[test]
    fn terracing_makes_cliffs_and_leaves_the_ground_crossable() {
        let smooth = FractalField { step: 0.0, ..banded() };
        let terraced = FractalField { terrace_mask: false, ..banded() };

        let slopes = |f: &FractalField| {
            let mut v: Vec<Real> = Vec::with_capacity(240 * 240);
            for a in -120..120 {
                for b in -120..120 {
                    let (_, dx, dz) = f.height_and_gradient(a as Real * 0.19, b as Real * 0.23);
                    v.push((dx * dx + dz * dz).sqrt().atan().to_degrees());
                }
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v
        };

        let s = slopes(&smooth);
        let t = slopes(&terraced);
        let at = |v: &[Real], q: f64| v[((v.len() - 1) as f64 * q) as usize];

        // Smooth fractional Brownian motion has one steepness and applies it
        // everywhere: scaling it up cannot make a cliff.
        assert!(at(&s, 1.0) < 65.0, "smooth fBm reached {} degrees", at(&s, 1.0));
        // Terracing puts most of the plane flat and the difficulty in the rest.
        assert!(at(&t, 0.5) < 5.0, "terraced median slope {} is not a plateau", at(&t, 0.5));
        assert!(at(&t, 0.99) > 70.0, "terraced p99 slope {} is not a cliff", at(&t, 0.99));
        // And it stays crossable: the ground must not become a wall everywhere.
        let walkable = t.iter().filter(|v| **v < 40.0).count() as Real / t.len() as Real;
        assert!(walkable > 0.80, "only {:.0}% of terraced ground is walkable", walkable * 100.0);
    }

    /// How much the *character* of the ground varies from place to place.
    ///
    /// Measured as the spread of mean slope across 12 m tiles, which is the
    /// metric that matters: a tile on the flank of a big hill has enormous
    /// relief and may still be billiard-smooth, so relief per tile answers a
    /// different question and answers it misleadingly. Measuring relief instead
    /// is what produced the claim — repeated in this file's history, the README
    /// and both plan documents — that domain warping does nothing for
    /// heterogeneity. It does; the metric could not see it.
    ///
    /// Measured on this field, warp 0 to 1 takes the spread from 0.09 to 0.18,
    /// and on the shipped landscape band from 0.04 to 0.11. The mask is still
    /// the strongest single lever and they compose.
    #[test]
    fn the_bands_each_make_the_ground_more_heterogeneous() {
        let spread = |f: &FractalField| {
            let mut tiles = Vec::new();
            for tx in -5..5 {
                for tz in -5..5 {
                    let mut sum = 0.0;
                    for a in 0..24 {
                        for b in 0..24 {
                            let (_, dx, dz) = f.height_and_gradient(
                                tx as Real * 12.0 + a as Real * 0.5,
                                tz as Real * 12.0 + b as Real * 0.5,
                            );
                            sum += (dx * dx + dz * dz).sqrt().atan().to_degrees();
                        }
                    }
                    tiles.push(sum / 576.0);
                }
            }
            let mean = tiles.iter().sum::<Real>() / tiles.len() as Real;
            let var = tiles.iter().map(|t| (t - mean).powi(2)).sum::<Real>() / tiles.len() as Real;
            var.sqrt() / mean
        };

        let plain = spread(&FractalField { modulation: 0.0, step: 0.0, warp: 0.0, ..banded() });
        let warped = spread(&FractalField { modulation: 0.0, step: 0.0, warp: 1.0, ..banded() });
        let modulated = spread(&FractalField { step: 0.0, ..banded() });
        let masked = spread(&banded());

        assert!(warped > plain * 1.5, "warp: {plain} -> {warped}");
        assert!(modulated > plain * 1.2, "modulation: {plain} -> {modulated}");
        assert!(masked > plain * 2.0, "mask: {plain} -> {masked}");
        // The mask is the strongest single lever, which is why it is the one
        // the shipped experiment turns on.
        assert!(masked > warped, "the mask ({masked}) should beat warp ({warped}) alone");
    }

    #[test]
    fn fractal_heights_respect_the_amplitude_bound() {
        for seed in [0u64, 3, 0x1234_5678_9ABC_DEF0] {
            let m = fractal(seed);
            let bound = m.height_bound();
            let mut peak = Real::NEG_INFINITY;
            let mut trough = Real::INFINITY;
            for a in -150..150 {
                for b in -150..150 {
                    let y = m.height_at(a as Real * 0.41, b as Real * 0.37);
                    assert!(y.is_finite(), "not finite");
                    assert!(y.abs() <= bound, "{y} exceeds the bound {bound}");
                    peak = peak.max(y);
                    trough = trough.min(y);
                }
            }
            // Relief runs to roughly 2.5x amplitude in practice, well inside the
            // 3.75x the bound allows. If this ever fails low, the field has gone
            // flat and every organism is on a plain.
            let relief = peak - trough;
            assert!(relief > 0.5 * 0.25, "suspiciously flat: {relief}");
        }
    }

    /// Every octave of Perlin noise is exactly zero at every lattice point, and
    /// with an integer lacunarity they all share one at the origin. Organisms
    /// spawn at the origin, so a dead flat dimple there would be present in
    /// every trial and invisible in every aggregate.
    #[test]
    fn the_origin_is_not_a_flat_spot() {
        for seed in [0u64, 1, 2, 3, 99] {
            let m = fractal(seed);
            let n = m.normal_at(0.0, 0.0);
            assert!(n.y < 0.9999, "origin is flat for seed {seed}: {n:?}");
        }
    }

    #[test]
    fn a_fractal_field_is_deterministic_and_seed_dependent() {
        let a = fractal(11);
        let b = fractal(12);
        let mut differ = 0;
        for i in 0..500 {
            let (x, z) = (i as Real * 0.19, i as Real * -0.07);
            assert_eq!(a.height_at(x, z).to_bits(), a.height_at(x, z).to_bits());
            if a.height_at(x, z) != b.height_at(x, z) {
                differ += 1;
            }
        }
        assert!(differ > 490, "seeds barely differ: {differ}/500");
    }

    /// Layer 2: a per-trial rigid motion has to move the ground the organism
    /// meets, or it is not closing the memorisation hole it exists to close.
    #[test]
    fn a_rigid_motion_moves_the_field() {
        let base = fractal(5);
        let shifted =
            TerrainModel::Fractal(FractalField { offset_x: 3.7, offset_z: -2.1, ..frac(5) });
        let turned = TerrainModel::Fractal(FractalField { rot_sin: 0.6, rot_cos: 0.8, ..frac(5) });
        let mut moved = 0;
        for i in 1..200 {
            let (x, z) = (i as Real * 0.23, i as Real * 0.17);
            if base.height_at(x, z) != shifted.height_at(x, z)
                && base.height_at(x, z) != turned.height_at(x, z)
            {
                moved += 1;
            }
        }
        assert!(moved > 190, "field did not move: {moved}/199");
        // A rotation about the origin leaves the origin where it was.
        assert_eq!(base.height_at(0.0, 0.0), turned.height_at(0.0, 0.0));
    }

    /// Aperiodicity is the whole point of replacing the sine field. Sampling a
    /// full wavelength apart should find different ground.
    #[test]
    fn the_fractal_field_does_not_repeat() {
        let m = fractal(21);
        let mut same = 0;
        for i in 1..300 {
            let x = i as Real * 0.3;
            if (m.height_at(x, 0.0) - m.height_at(x + 6.0, 0.0)).abs() < 1e-4 {
                same += 1;
            }
        }
        assert!(same < 15, "field looks periodic: {same}/299 samples coincide");
    }

    #[test]
    fn a_degenerate_fractal_field_stays_finite() {
        let m = TerrainModel::Fractal(FractalField {
            seed: 0,
            amplitude: 0.0,
            wavelength: 0.0,
            octaves: 0,
            lacunarity: 0.0,
            gain: 0.0,
            warp: 0.0,
            detail_amplitude: 0.0,
            detail_wavelength: 0.0,
            detail_octaves: 0,
            modulation: 0.0,
            modulation_wavelength: 0.0,
            step: 0.0,
            riser: 0.0,
            terrace_mask: true,
            offset_x: 0.0,
            offset_z: 0.0,
            rot_sin: 0.0,
            rot_cos: 0.0,
        });
        let (h, n) = m.sample(1.0, -1.0);
        assert_eq!(h, 0.0);
        assert_eq!(n, Vec3::Y);
        // Nor a long way from the origin, where the lattice index saturates.
        for x in [1e4, -1e4, 1e12, -1e12] {
            let (h, n) = fractal(4).sample(x, x);
            assert!(h.is_finite() && n.y.is_finite(), "blew up at {x}");
        }
    }
}
