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
use evoforge::physics::{FractalField, Joint, RigidBody, TerrainModel, World, WorldParams};

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

// There is deliberately no "height of the ground beneath it" datum here.
//
// An earlier version of this probe subtracted one, reasoning that it would keep
// "it slid downhill" from reading as an energy gain. It does the opposite: a
// body converting a hill's potential energy into speed keeps all that speed in
// the numerator while the datum follows it downhill, so honest acceleration
// reported as a 3.7x gain and sent me looking for a bug that was not there.
//
// Total mechanical energy in a fixed frame needs no correction: gravity is
// conservative, friction and damping only remove, so a passive body's
// `KE + mgy` can never exceed where it started. Anything above 1.0x below is
// the solver inventing energy.

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

            let start = energy(&w);
            let mut peak = start;
            let mut fastest = speed;
            for _ in 0..480 {
                w.step(dt);
                if w.diverged {
                    break;
                }
                peak = peak.max(energy(&w));
                fastest = fastest.max(w.bodies[0].lin_vel.length());
            }
            if w.diverged {
                diverged += 1;
                continue;
            }
            // Heights can be negative, so measure against a floor well below the
            // field rather than against zero.
            let base = 200.0 * GRAVITY * 1000.0 * 0.12 * 0.12 * 0.12 * 8.0;
            let gain = (peak + base) / (start + base);
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
    TerrainModel::Fractal(FractalField {
        seed: 0x5EED_0001,
        amplitude,
        wavelength,
        octaves,
        warp: 0.6,
        ..FractalField::default()
    })
}

/// Two parts sharing a space, which is where the energy actually came from.
///
/// A single box on the ground cannot show the bug: with one body there are no
/// pair contacts. What evolution found was overlapping *parts* — self-collision
/// shoving them apart and the shove being kept as momentum. So: two unjointed
/// boxes overlapping by `overlap`, no motors, no joints, nothing that is meant
/// to do work, and the sweep is over how deeply they start interpenetrating.
fn probe_pair(label: &str, self_collision: bool) {
    let params = WorldParams {
        terrain: TerrainModel::Flat { height: 0.0 },
        friction: 0.9,
        self_collision,
        ..WorldParams::default()
    };
    let dt = 1.0 / 120.0;

    let mut worst = 1.0f32;
    let mut worst_overlap = 0.0;
    let mut worst_travel = 0.0f32;
    for step in 1..12 {
        let overlap = step as Real * 0.02;
        // Offset on two axes so the contact normal is oblique: parts stacked
        // dead square push straight up, which goes nowhere.
        let a = RigidBody::box_body(vec3(0.0, 0.30, 0.0), vec3(0.15, 0.15, 0.15), 1000.0);
        let b = RigidBody::box_body(
            vec3(0.30 - overlap, 0.30 + 0.12, 0.05),
            vec3(0.15, 0.15, 0.15),
            1000.0,
        );
        let mut w = World::new(vec![a, b], vec![], params);
        let start = energy(&w);
        let start_x = 0.5 * (w.bodies[0].pos.x + w.bodies[1].pos.x);
        let mut peak = start;
        for _ in 0..600 {
            w.step(dt);
            if w.diverged {
                break;
            }
            peak = peak.max(energy(&w));
        }
        if w.diverged {
            continue;
        }
        let base = 200.0 * GRAVITY * 1000.0 * 0.15 * 0.15 * 0.15 * 8.0;
        let gain = (peak + base) / (start + base);
        let travel = (0.5 * (w.bodies[0].pos.x + w.bodies[1].pos.x) - start_x).abs();
        if gain > worst {
            worst = gain;
            worst_overlap = overlap;
        }
        worst_travel = worst_travel.max(travel);
    }
    println!(
        "{label:<40} peak energy {worst:6.2}x start (at {worst_overlap:.2} m overlap), \
         drifted {worst_travel:5.2} m"
    );
}

/// A folded limb: three parts in a chain, with the two ends overlapping.
///
/// This is the shape the bug actually needs, and why two loose boxes do not
/// show it. A pair shoved apart separates once and is done. But `A-B-C` with
/// `A` and `C` overlapping is a cycle: the joints pull the ends back into each
/// other every step, self-collision shoves them apart again, and if the shove
/// is kept as momentum the pair is a motor that never runs down. Evolution
/// found this before it found walking.
fn probe_folded(label: &str) {
    let dt = 1.0 / 120.0;
    let mut worst = 1.0f32;
    let mut worst_travel = 0.0f32;
    let mut worst_late = 0.0f32;

    for fold in 1..10 {
        let reach = 0.30 - fold as Real * 0.028;
        let bodies: Vec<RigidBody> = (0..3)
            .map(|i| {
                let angle = i as Real * 1.9;
                RigidBody::box_body(
                    vec3(reach * angle.cos(), 1.0 + i as Real * 0.05, reach * angle.sin()),
                    vec3(0.12, 0.12, 0.12),
                    1000.0,
                )
            })
            .collect();
        // Weld each part to the next, anchored at the midpoint between them, so
        // the chain holds its folded pose against the collision pushing it open.
        let joints: Vec<Joint> = (0..2)
            .map(|i| {
                let mid = (bodies[i].pos + bodies[i + 1].pos) * 0.5;
                Joint::fixed(i as u16, i as u16 + 1, mid - bodies[i].pos, mid - bodies[i + 1].pos)
            })
            .collect();

        let params = WorldParams {
            terrain: TerrainModel::Flat { height: 0.0 },
            friction: 0.9,
            self_collision: true,
            ..WorldParams::default()
        };
        let mut w = World::new(bodies, joints, params);
        // Let it fall and settle first. Everything measured after this point
        // starts from rest on flat ground with no potential energy to spend, so
        // any motion at all is motion the solver invented.
        for _ in 0..240 {
            w.step(dt);
        }
        if w.diverged {
            continue;
        }
        let settled = energy(&w);
        let settled_x = w.bodies[0].pos.x;
        let settled_z = w.bodies[0].pos.z;
        let mut peak = settled;
        let mut mid = (0.0, 0.0);
        for k in 0..1800 {
            w.step(dt);
            if w.diverged {
                break;
            }
            if k == 900 {
                mid = (w.bodies[0].pos.x, w.bodies[0].pos.z);
            }
            peak = peak.max(energy(&w));
        }
        if w.diverged {
            continue;
        }
        let base = 200.0 * GRAVITY * 1000.0 * 0.12 * 0.12 * 0.12 * 8.0;
        worst = worst.max((peak + base) / (settled + base));
        let dx = mid.0 - settled_x;
        let dz = mid.1 - settled_z;
        worst_travel = worst_travel.max((dx * dx + dz * dz).sqrt());
        let lx = w.bodies[0].pos.x - mid.0;
        let lz = w.bodies[0].pos.z - mid.1;
        worst_late = worst_late.max((lx * lx + lz * lz).sqrt());
    }
    println!(
        "{label:<40} after settling: peak energy {worst:6.2}x, \n         crawled {worst_travel:5.2} m in the first 7.5 s, {worst_late:5.2} m in the next"
    );
}

fn main() {
    println!("-- two overlapping parts, nothing driving them");
    probe_pair("self-collision on", true);
    probe_pair("self-collision off", false);

    println!("\n-- a folded chain whose ends overlap: joints in, collision out");
    probe_folded("folded chain, motors off");

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
