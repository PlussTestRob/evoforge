//! Body geometry.
//!
//! # Why this is small
//!
//! Organisms do not collide with each other or with themselves (see
//! [`super::world`]), so the only collision query in the simulator is *shape
//! against the terrain*. That removes every part of a collision system that is
//! normally expensive to write and easy to get wrong: no support mapping against
//! arbitrary opponents, no manifold generation, no broad phase. A shape has to
//! answer exactly three questions:
//!
//! * what does it weigh, and how is that mass distributed ([`Shape::mass_properties`]);
//! * which of its points are candidates for touching the ground
//!   ([`Shape::ground_points`]);
//! * what box does it fit inside ([`Shape::bounds`]), which is what the
//!   attachment rules and the spawn placement work in.
//!
//! # Centres
//!
//! Every shape is described about its *geometric* centre, because that is the
//! frame the attachment rules in [`crate::phenotype`] are written in. A
//! [`super::RigidBody`], though, is positioned by its centre of mass, and for a
//! tapered part the two differ. [`Shape::com_offset`] is the vector between
//! them, and it is the caller's job to apply it — which
//! [`crate::phenotype::build`] does once, at construction.
//!
//! # Axes
//!
//! The shapes that have a long direction carry it as an axis index rather than
//! an orientation, because parts are spawned axis-aligned and a rotation gene
//! would be a second way to express something the attachment face already says.

use serde::{Deserialize, Serialize};

use crate::math::{vec3, Quat, Real, Vec3, PI};

/// Most contact points any one shape can contribute. A box and a taper have
/// eight corners; nothing else comes close.
pub const MAX_GROUND_POINTS: usize = 8;

/// A body's collision geometry, described about its geometric centre.
///
/// Serialised into every replay so that a viewer can draw the organism without
/// re-expressing the genome.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Shape {
    Box {
        half_extents: Vec3,
    },
    /// Rectangular frustum: the cross-section at `+axis` is `top_scale` times
    /// the cross-section at `-axis`, which is the one `half_extents` describes.
    Taper {
        half_extents: Vec3,
        axis: u8,
        top_scale: Real,
    },
    Sphere {
        radius: Real,
    },
    /// Cylinder of `half_length` along `axis`, capped with hemispheres. The
    /// total half-length along the axis is therefore `half_length + radius`.
    Capsule {
        radius: Real,
        half_length: Real,
        axis: u8,
    },
    Cylinder {
        radius: Real,
        half_length: Real,
        axis: u8,
    },
}

impl Shape {
    /// Half-extents of the axis-aligned box that bounds this shape, about the
    /// geometric centre.
    pub fn bounds(&self) -> Vec3 {
        match *self {
            // A taper's base is its widest cross-section, so the bound is the
            // base. This is why `top_scale` is required to be <= 1.
            Shape::Box { half_extents } | Shape::Taper { half_extents, .. } => half_extents,
            Shape::Sphere { radius } => Vec3::splat(radius),
            Shape::Capsule { radius, half_length, axis } => {
                spread(half_length + radius, radius, axis)
            }
            Shape::Cylinder { radius, half_length, axis } => spread(half_length, radius, axis),
        }
    }

    /// Vector from the geometric centre to the centre of mass, in the body frame.
    ///
    /// Zero for everything symmetric, which is everything but a taper.
    pub fn com_offset(&self) -> Vec3 {
        match *self {
            Shape::Taper { half_extents, axis, top_scale } => {
                // With k(u) = 1 + a*u the linear cross-section scale along the
                // axis, the first moment integrates to this. At top_scale = 1 it
                // is zero, and at top_scale = 0 it is -h/2 — a quarter of the
                // way up from the base of a pyramid, which is the known answer.
                let a = top_scale - 1.0;
                let h = component(half_extents, axis);
                let offset = h * (a * (2.0 + a)) / (2.0 * (a * a + 3.0 * a + 3.0));
                axis_vec(axis) * offset
            }
            _ => Vec3::ZERO,
        }
    }

    /// Mass, and the diagonal of the inertia tensor about the centre of mass in
    /// the body frame.
    ///
    /// Every shape here is symmetric enough that its inertia tensor is diagonal
    /// in its own frame, which is why [`super::RigidBody`] stores three numbers
    /// rather than nine.
    pub fn mass_properties(&self, density: Real) -> (Real, Vec3) {
        match *self {
            Shape::Box { half_extents: h } => {
                let mass = 8.0 * h.x * h.y * h.z * density;
                let i = vec3(
                    mass / 3.0 * (h.y * h.y + h.z * h.z),
                    mass / 3.0 * (h.x * h.x + h.z * h.z),
                    mass / 3.0 * (h.x * h.x + h.y * h.y),
                );
                (mass, i)
            }

            Shape::Taper { half_extents, axis, top_scale } => {
                let s = top_scale;
                let ha = component(half_extents, axis);
                let (b_axis, c_axis) = cross_axes(axis);
                let hb = component(half_extents, b_axis);
                let hc = component(half_extents, c_axis);

                // Moments of the cross-section scale k(u) = 1 + (s-1)u over the
                // length of the shape. At s = 1 these are 1/3, 1 and 1/3, and
                // every formula below collapses to the box.
                let a = s - 1.0;
                let k2 = (s * s + s + 1.0) / 3.0;
                let k4 = (s * s * s * s + s * s * s + s * s + s + 1.0) / 5.0;
                let q = (2.0 / 15.0) * a * a + a / 3.0 + 1.0 / 3.0;

                let mass = 8.0 * hb * hc * ha * k2 * density;

                // About the geometric centre first, then shifted onto the centre
                // of mass with the parallel axis theorem.
                let i_axis =
                    density * (8.0 / 3.0) * ha * (hb * hb * hb * hc + hb * hc * hc * hc) * k4;
                let shift = mass * component(self.com_offset(), axis).powi(2);
                let i_b = density
                    * (8.0 * hb * hc * ha * ha * ha * q
                        + (8.0 / 3.0) * hb * hc * hc * hc * ha * k4)
                    - shift;
                let i_c = density
                    * (8.0 * hb * hc * ha * ha * ha * q
                        + (8.0 / 3.0) * hb * hb * hb * hc * ha * k4)
                    - shift;

                let mut i = Vec3::ZERO;
                set_component(&mut i, axis, i_axis);
                set_component(&mut i, b_axis, i_b);
                set_component(&mut i, c_axis, i_c);
                (mass, i)
            }

            Shape::Sphere { radius: r } => {
                let mass = (4.0 / 3.0) * PI * r * r * r * density;
                (mass, Vec3::splat(0.4 * mass * r * r))
            }

            Shape::Capsule { radius: r, half_length: l, axis } => {
                let m_cyl = PI * r * r * (2.0 * l) * density;
                let m_cap = (4.0 / 3.0) * PI * r * r * r * density;
                let i_axis = 0.5 * m_cyl * r * r + 0.4 * m_cap * r * r;
                // The hemispheres' transverse moment about the capsule's centre:
                // their own 2/5 r^2, displaced by the cylinder half-length, plus
                // the cross term from the hemisphere centroid at 3r/8.
                let i_trans = m_cyl * (r * r / 4.0 + l * l / 3.0)
                    + m_cap * (0.4 * r * r + l * l + 0.75 * l * r);
                (m_cyl + m_cap, spread_inertia(i_axis, i_trans, axis))
            }

            Shape::Cylinder { radius: r, half_length: l, axis } => {
                let mass = PI * r * r * (2.0 * l) * density;
                let i_axis = 0.5 * mass * r * r;
                let i_trans = mass * (r * r / 4.0 + l * l / 3.0);
                (mass, spread_inertia(i_axis, i_trans, axis))
            }
        }
    }

    /// World-space points that may be touching the ground, given the body's
    /// centre of mass, its orientation, and the terrain normal beneath it.
    ///
    /// For flat-faced shapes this is every corner, exactly as before shapes
    /// existed. For curved shapes it is the analytically deepest point against
    /// `normal`, which is both cheaper and more accurate than sampling the
    /// surface — a sphere resting on a plane touches it at one point, and that
    /// point is known in closed form.
    pub fn ground_points(
        &self,
        pos: Vec3,
        orient: Quat,
        normal: Vec3,
    ) -> ([Vec3; MAX_GROUND_POINTS], usize) {
        let mut out = [Vec3::ZERO; MAX_GROUND_POINTS];
        // Geometry is described about the geometric centre; the body is
        // positioned by its centre of mass.
        let centre = pos - orient.rotate(self.com_offset());
        let place = |local: Vec3| centre + orient.rotate(local);

        match *self {
            Shape::Box { half_extents: h } => {
                let mut n = 0;
                for sx in [-1.0, 1.0] {
                    for sy in [-1.0, 1.0] {
                        for sz in [-1.0, 1.0] {
                            out[n] = place(vec3(sx * h.x, sy * h.y, sz * h.z));
                            n += 1;
                        }
                    }
                }
                (out, n)
            }

            Shape::Taper { half_extents, axis, top_scale } => {
                let ha = component(half_extents, axis);
                let (b_axis, c_axis) = cross_axes(axis);
                let hb = component(half_extents, b_axis);
                let hc = component(half_extents, c_axis);
                let mut n = 0;
                // Base ring at -axis is full width; the ring at +axis is scaled.
                for (along, scale) in [(-ha, 1.0), (ha, top_scale)] {
                    for sb in [-1.0, 1.0] {
                        for sc in [-1.0, 1.0] {
                            let mut local = Vec3::ZERO;
                            set_component(&mut local, axis, along);
                            set_component(&mut local, b_axis, sb * hb * scale);
                            set_component(&mut local, c_axis, sc * hc * scale);
                            out[n] = place(local);
                            n += 1;
                        }
                    }
                }
                (out, n)
            }

            Shape::Sphere { radius } => {
                out[0] = centre - normal * radius;
                (out, 1)
            }

            Shape::Capsule { radius, half_length, axis } => {
                let a = orient.rotate(axis_vec(axis));
                out[0] = centre + a * half_length - normal * radius;
                out[1] = centre - a * half_length - normal * radius;
                (out, 2)
            }

            Shape::Cylinder { radius, half_length, axis } => {
                let a = orient.rotate(axis_vec(axis));
                // Deepest direction around the rim, and the two points a quarter
                // turn either side of it. One point alone would let a cylinder
                // standing on its end pivot on a needle; three give it a base.
                let down = project_out(-normal, a).normalize_or(a.any_perpendicular());
                let side = a.cross(down);
                let mut n = 0;
                for end in [half_length, -half_length] {
                    let c = centre + a * end;
                    out[n] = c + down * radius;
                    out[n + 1] = c + side * radius;
                    out[n + 2] = c - side * radius;
                    n += 3;
                }
                (out, n)
            }
        }
    }

    /// Whether a point given in the body frame, about the geometric centre, is
    /// inside the shape. Used by the tests that check the closed-form mass
    /// properties above against numerical integration.
    #[cfg(test)]
    pub fn contains(&self, p: Vec3) -> bool {
        match *self {
            Shape::Box { half_extents: h } => {
                p.x.abs() <= h.x && p.y.abs() <= h.y && p.z.abs() <= h.z
            }
            Shape::Taper { half_extents, axis, top_scale } => {
                let ha = component(half_extents, axis);
                let (b_axis, c_axis) = cross_axes(axis);
                let along = component(p, axis);
                if along.abs() > ha {
                    return false;
                }
                let u = (along + ha) / (2.0 * ha);
                let k = 1.0 + (top_scale - 1.0) * u;
                component(p, b_axis).abs() <= component(half_extents, b_axis) * k
                    && component(p, c_axis).abs() <= component(half_extents, c_axis) * k
            }
            Shape::Sphere { radius } => p.length_sq() <= radius * radius,
            Shape::Capsule { radius, half_length, axis } => {
                let along = crate::math::clamp(component(p, axis), -half_length, half_length);
                (p - axis_vec(axis) * along).length_sq() <= radius * radius
            }
            Shape::Cylinder { radius, half_length, axis } => {
                component(p, axis).abs() <= half_length
                    && project_out(p, axis_vec(axis)).length_sq() <= radius * radius
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Axis helpers
// ---------------------------------------------------------------------------

#[inline]
fn axis_vec(axis: u8) -> Vec3 {
    match axis % 3 {
        0 => Vec3::X,
        1 => Vec3::Y,
        _ => Vec3::Z,
    }
}

#[inline]
fn component(v: Vec3, axis: u8) -> Real {
    match axis % 3 {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[inline]
fn set_component(v: &mut Vec3, axis: u8, value: Real) {
    match axis % 3 {
        0 => v.x = value,
        1 => v.y = value,
        _ => v.z = value,
    }
}

/// The two axes that are not `axis`, in ascending order.
#[inline]
fn cross_axes(axis: u8) -> (u8, u8) {
    match axis % 3 {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// A vector with `along` on `axis` and `across` on the other two.
#[inline]
fn spread(along: Real, across: Real, axis: u8) -> Vec3 {
    let mut v = Vec3::splat(across);
    set_component(&mut v, axis, along);
    v
}

#[inline]
fn spread_inertia(i_axis: Real, i_trans: Real, axis: u8) -> Vec3 {
    spread(i_axis, i_trans, axis)
}

/// The component of `v` perpendicular to the unit vector `axis`.
#[inline]
fn project_out(v: Vec3, axis: Vec3) -> Vec3 {
    v - axis * v.dot(axis)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DENSITY: Real = 250.0;

    /// Mass and inertia by brute force: chop the bounding box into cells, keep
    /// the ones inside the shape, and sum. Slow, obviously correct, and entirely
    /// independent of the closed forms it is checking — which is the point, as
    /// the frustum integrals are the kind of algebra that is easy to get subtly
    /// wrong and hard to eyeball.
    fn numeric_properties(shape: &Shape, steps: usize) -> (Real, Vec3, Vec3) {
        let b = shape.bounds();
        let step =
            vec3(2.0 * b.x / steps as Real, 2.0 * b.y / steps as Real, 2.0 * b.z / steps as Real);
        let cell = step.x * step.y * step.z * DENSITY;

        let mut mass = 0.0;
        let mut first = Vec3::ZERO;
        let mut points = Vec::new();
        for ix in 0..steps {
            for iy in 0..steps {
                for iz in 0..steps {
                    let p = vec3(
                        -b.x + (ix as Real + 0.5) * step.x,
                        -b.y + (iy as Real + 0.5) * step.y,
                        -b.z + (iz as Real + 0.5) * step.z,
                    );
                    if shape.contains(p) {
                        mass += cell;
                        first += p * cell;
                        points.push(p);
                    }
                }
            }
        }
        let com = first * (1.0 / mass);

        let mut i = Vec3::ZERO;
        for p in points {
            let d = p - com;
            i.x += cell * (d.y * d.y + d.z * d.z);
            i.y += cell * (d.x * d.x + d.z * d.z);
            i.z += cell * (d.x * d.x + d.y * d.y);
        }
        (mass, com, i)
    }

    fn assert_close(label: &str, got: Real, want: Real, tol: Real) {
        let scale = want.abs().max(1e-6);
        assert!(
            (got - want).abs() / scale < tol,
            "{label}: got {got}, want {want} (relative {})",
            (got - want).abs() / scale
        );
    }

    fn check_against_integration(shape: Shape, tol: Real) {
        let (mass, inertia) = shape.mass_properties(DENSITY);
        let (n_mass, n_com, n_inertia) = numeric_properties(&shape, 60);

        assert_close("mass", mass, n_mass, tol);
        let com = shape.com_offset();
        for (label, got, want) in
            [("com.x", com.x, n_com.x), ("com.y", com.y, n_com.y), ("com.z", com.z, n_com.z)]
        {
            // The offset is small in absolute terms, so compare against the
            // shape's size rather than against the offset itself.
            let scale = shape.bounds().length();
            assert!((got - want).abs() / scale < tol, "{label}: got {got}, want {want}");
        }
        assert_close("inertia.x", inertia.x, n_inertia.x, tol);
        assert_close("inertia.y", inertia.y, n_inertia.y, tol);
        assert_close("inertia.z", inertia.z, n_inertia.z, tol);
    }

    #[test]
    fn box_mass_properties_match_integration() {
        check_against_integration(Shape::Box { half_extents: vec3(0.3, 0.15, 0.22) }, 0.02);
    }

    #[test]
    fn sphere_mass_properties_match_integration() {
        check_against_integration(Shape::Sphere { radius: 0.24 }, 0.02);
    }

    #[test]
    fn capsule_mass_properties_match_integration() {
        for axis in 0..3 {
            check_against_integration(
                Shape::Capsule { radius: 0.12, half_length: 0.2, axis },
                0.02,
            );
        }
    }

    #[test]
    fn cylinder_mass_properties_match_integration() {
        for axis in 0..3 {
            check_against_integration(
                Shape::Cylinder { radius: 0.14, half_length: 0.25, axis },
                0.02,
            );
        }
    }

    #[test]
    fn taper_mass_properties_match_integration() {
        for axis in 0..3 {
            for top_scale in [0.2, 0.45, 0.8] {
                check_against_integration(
                    Shape::Taper { half_extents: vec3(0.3, 0.2, 0.25), axis, top_scale },
                    0.03,
                );
            }
        }
    }

    /// The frustum formulas have to collapse onto the box formulas at
    /// `top_scale == 1`, exactly. This is the cheapest check that the algebra is
    /// not merely close.
    #[test]
    fn a_taper_of_one_is_a_box() {
        let h = vec3(0.31, 0.17, 0.23);
        let (bm, bi) = Shape::Box { half_extents: h }.mass_properties(DENSITY);
        for axis in 0..3 {
            let taper = Shape::Taper { half_extents: h, axis, top_scale: 1.0 };
            let (tm, ti) = taper.mass_properties(DENSITY);
            assert!((tm - bm).abs() < 1e-3, "mass on axis {axis}: {tm} vs {bm}");
            assert!(taper.com_offset().length() < 1e-6);
            for (g, w) in [(ti.x, bi.x), (ti.y, bi.y), (ti.z, bi.z)] {
                assert!((g - w).abs() / w < 1e-4, "inertia on axis {axis}: {g} vs {w}");
            }
        }
    }

    /// A pyramid's centre of mass sits a quarter of the way up from its base.
    #[test]
    fn a_full_taper_has_a_pyramids_centre_of_mass() {
        let h = vec3(0.2, 0.3, 0.2);
        let s = Shape::Taper { half_extents: h, axis: 1, top_scale: 0.0 };
        assert!((s.com_offset().y + 0.15).abs() < 1e-5, "{:?}", s.com_offset());
    }

    #[test]
    fn ground_points_sit_on_the_surface_of_a_curved_shape() {
        let normal = Vec3::Y;
        for shape in [
            Shape::Sphere { radius: 0.2 },
            Shape::Capsule { radius: 0.1, half_length: 0.3, axis: 0 },
            Shape::Cylinder { radius: 0.15, half_length: 0.25, axis: 2 },
        ] {
            let pos = vec3(0.0, 1.0, 0.0);
            let (pts, n) = shape.ground_points(pos, Quat::IDENTITY, normal);
            assert!(n > 0);
            // The lowest candidate must be the true lowest point of the shape,
            // which for these shapes is `bounds().y` below the centre.
            let lowest = pts[..n].iter().fold(Real::INFINITY, |m, p| m.min(p.y));
            assert!(
                (lowest - (pos.y - shape.bounds().y)).abs() < 1e-5,
                "{shape:?}: lowest {lowest}"
            );
        }
    }

    /// However a cylinder is turned, the deepest of its candidate points has to
    /// be the deepest point of the shape — otherwise it would sink through the
    /// ground at some orientations and hover at others.
    #[test]
    fn a_rolling_cylinder_always_finds_its_lowest_point() {
        let shape = Shape::Cylinder { radius: 0.15, half_length: 0.3, axis: 0 };
        let pos = vec3(0.0, 2.0, 0.0);
        for i in 0..32 {
            let angle = i as Real * (crate::math::TAU / 32.0);
            let (s, c) = crate::math::dsincos(angle * 0.5);
            // Rotate about Z, so the cylinder's X axis sweeps through vertical.
            let orient = Quat { x: 0.0, y: 0.0, z: s, w: c }.normalize();
            let (pts, n) = shape.ground_points(pos, orient, Vec3::Y);
            let lowest = pts[..n].iter().fold(Real::INFINITY, |m, p| m.min(p.y));

            // Brute-force the true minimum over the two rims.
            let a = orient.rotate(Vec3::X);
            let mut truth = Real::INFINITY;
            for end in [0.3, -0.3] {
                let c = pos + a * end;
                for j in 0..720 {
                    let t = j as Real * (crate::math::TAU / 720.0);
                    let (sj, cj) = crate::math::dsincos(t);
                    let d = a.any_perpendicular();
                    let e = a.cross(d);
                    let p = c + (d * cj + e * sj) * 0.15;
                    truth = truth.min(p.y);
                }
            }
            assert!(
                (lowest - truth).abs() < 2e-3,
                "orientation {i}: candidates bottom out at {lowest}, true minimum {truth}"
            );
        }
    }
}
