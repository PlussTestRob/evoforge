//! Does the ground push? Slide a passive body across each terrain and watch it.
//!
//! There is no motor, no joint and no tendon here — nothing that is *supposed*
//! to do work. Gravity is conservative; friction, damping and inelastic contact
//! can only take energy out. A passive body launched across the ground must
//! therefore end slower than it started, and its total mechanical energy must
//! never exceed what it began with. Anything else came from the solver, and
//! anything the solver gives away for free, evolution will find.
//!
//! Dropping a body from rest is not enough to see it: the failure needs speed,
//! because it is depth of penetration per step that the positional correction
//! reacts to.

use evoforge::math::{vec3, Real};
use evoforge::physics::{RigidBody, TerrainModel, World, WorldParams};

const GRAVITY: Real = 9.81;

fn energy(w: &World) -> Real {
    w.bodies
        .iter()
        .map(|b| {
            let m = b.mass();
            let i = 1.0 / b.inv_inertia_local.z;
            0.5 * m * b.lin_vel.length_sq()
                + 0.5 * i * b.ang_vel.length_sq()
                + m * GRAVITY * b.pos.y
        })
        .sum()
}

/// Total mechanical energy the body would have at rest on the ground beneath
/// it. Subtracting this is what stops "it slid downhill" reading as a gain.
fn datum(w: &World, terrain: &TerrainModel) -> Real {
    w.bodies.iter().map(|b| b.mass() * GRAVITY * terrain.height_at(b.pos.x, b.pos.z)).sum()
}

fn probe(label: &str, terrain: TerrainModel, speed: Real) {
    let params = WorldParams { terrain, friction: 0.9, ..WorldParams::default() };
    let dt = 1.0 / 120.0;

    let mut worst_gain = 1.0f32;
    let mut worst_speed_ratio = 0.0f32;
    let mut worst_at = (0.0, 0.0);
    let mut diverged = 0;

    for a in -5..6 {
        for b in -5..6 {
            let (x, z) = (a as Real * 1.31, b as Real * 0.97);
            let y = terrain.height_at(x, z) + 0.15;
            let mut body = RigidBody::box_body(vec3(x, y, z), vec3(0.12, 0.12, 0.12), 1000.0);
            body.lin_vel = vec3(speed, 0.0, 0.0);
            let mut w = World::new(vec![body], vec![], params);

            let start = energy(&w) - datum(&w, &terrain);
            let mut peak = start;
            let mut fastest = speed;
            for _ in 0..480 {
                w.step(dt);
                if w.diverged {
                    break;
                }
                peak = peak.max(energy(&w) - datum(&w, &terrain));
                fastest = fastest.max(w.bodies[0].lin_vel.length());
            }
            if w.diverged {
                diverged += 1;
                continue;
            }
            let gain = peak / start.abs().max(1e-6);
            if gain > worst_gain {
                worst_gain = gain;
                worst_at = (x, z);
            }
            worst_speed_ratio = worst_speed_ratio.max(fastest / speed);
        }
    }
    println!(
        "{label:<40} v0={speed:4.1}  peak energy {worst_gain:6.2}x start \
         at ({:6.2},{:6.2})   fastest {worst_speed_ratio:5.2}x v0{}",
        worst_at.0,
        worst_at.1,
        if diverged > 0 { format!("   [{diverged} diverged]") } else { String::new() },
    );
}

fn fractal(amplitude: Real, wavelength: Real, octaves: u32) -> TerrainModel {
    TerrainModel::Fractal {
        seed: 0x5EED_0001,
        amplitude,
        wavelength,
        octaves,
        lacunarity: 2.0,
        gain: 0.5,
        warp: 0.6,
        offset_x: 0.0,
        offset_z: 0.0,
        rot_sin: 0.0,
        rot_cos: 1.0,
    }
}

fn main() {
    for speed in [1.0, 4.0, 8.0] {
        println!();
        probe("flat", TerrainModel::Flat { height: 0.0 }, speed);
        probe(
            "rough  a=0.05 w=1.6 (animals.toml)",
            TerrainModel::Rough { amplitude: 0.05, wavelength: 1.6 },
            speed,
        );
        probe(
            "rough  a=0.25 w=1.6",
            TerrainModel::Rough { amplitude: 0.25, wavelength: 1.6 },
            speed,
        );
        probe("fractal a=0.05 w=3.0 oct=4", fractal(0.05, 3.0, 4), speed);
        probe("fractal a=0.25 w=3.0 oct=1", fractal(0.25, 3.0, 1), speed);
        probe("fractal a=0.25 w=3.0 oct=4 (shipped)", fractal(0.25, 3.0, 4), speed);
        probe("fractal a=0.25 w=12.0 oct=4", fractal(0.25, 12.0, 4), speed);
    }
}
