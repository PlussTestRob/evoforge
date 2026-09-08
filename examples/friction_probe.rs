//! Does a resting body get the friction it is owed?
//!
//! Coulomb friction here is clamped to `mu * pn`, where `pn` is the normal
//! impulse accumulated *this step*. Contacts are rebuilt every step with
//! `pn = 0`, so a body already at rest starts each step with no friction budget
//! and earns it back over the solver's iterations. If the budget never catches
//! up, a resting body slides when it should not — and the amount it slides is a
//! function of the iteration count rather than of the physics.
//!
//! A box on a slope shallower than `atan(mu)` must not move. At mu = 0.8 that
//! angle is 38.7 degrees.

use evoforge::math::{vec3, Real};
use evoforge::physics::{RigidBody, Shape, TerrainModel, World, WorldParams};

fn slide(angle_deg: Real, iterations: u32, dt: Real, seconds: Real) -> Real {
    // A plane tilted by `angle` expressed as a sine field so gentle it is
    // effectively a ramp over the region the box occupies.
    let slope = (angle_deg * std::f32::consts::PI / 180.0).tan();
    let amplitude = slope * 40.0;
    let terrain = TerrainModel::Rough { amplitude, wavelength: 250.0 };

    let shape = Shape::Box { half_extents: vec3(0.1, 0.1, 0.1) };
    let x = 0.0;
    let z = 0.0;
    let y = terrain.height_at(x, z) + 0.1;
    let body = RigidBody::new(vec3(x, y, z), shape, 1000.0);

    let mut w =
        World::new(vec![body], vec![], WorldParams { terrain, iterations, ..Default::default() });
    let start = w.bodies[0].pos;
    let steps = (seconds / dt) as u32;
    for _ in 0..steps {
        w.step(dt);
    }
    let d = w.bodies[0].pos - start;
    (d.x * d.x + d.z * d.z).sqrt()
}

fn main() {
    println!("A box on a slope, friction mu = 0.8, so it must hold below 38.7 degrees.");
    println!("Distance slid in 4 s:\n");
    println!("  slope | iters=12 | iters=24 | iters=48 | iters=96 | verdict");
    for angle in [5.0, 10.0, 20.0, 30.0, 35.0] {
        let a = slide(angle, 12, 1.0 / 120.0, 4.0);
        let b = slide(angle, 24, 1.0 / 120.0, 4.0);
        let c = slide(angle, 48, 1.0 / 120.0, 4.0);
        let d = slide(angle, 96, 1.0 / 120.0, 4.0);
        let verdict = if a > 0.05 { "SLIDES when it should hold" } else { "holds" };
        println!("  {angle:5.0} | {a:8.3} | {b:8.3} | {c:8.3} | {d:8.3} | {verdict}");
    }
    println!("\nSame, varying the timestep at the shipped 12 iterations:");
    println!("  slope |  dt=1/120 |  dt=1/240 |  dt=1/480 |  dt=1/960");
    for angle in [10.0, 20.0, 30.0] {
        println!(
            "  {angle:5.0} | {:9.3} | {:9.3} | {:9.3} | {:9.3}",
            slide(angle, 12, 1.0 / 120.0, 4.0),
            slide(angle, 12, 1.0 / 240.0, 4.0),
            slide(angle, 12, 1.0 / 480.0, 4.0),
            slide(angle, 12, 1.0 / 960.0, 4.0),
        );
    }
}
