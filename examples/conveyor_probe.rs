//! Which ground carries a corpse?
//!
//! Takes one evolved organism, switches its motors fully off, and drops it onto
//! a sweep of terrain settings. Distance covered is distance the world gave
//! away. Isolates *which property of the ground* does it — steepness, feature
//! size, or the number of scales — by varying one at a time.
//!
//!   cargo run --release --example conveyor_probe -- <run-dir>

use std::fs;
use std::path::Path;

use evoforge::config::{Config, Terrain};
use evoforge::genome::Genome;
use evoforge::math::Real;
use evoforge::sim;

fn main() {
    let run = std::env::args().nth(1).expect("usage: conveyor_probe <run-dir>");
    let base = Config::load(&Path::new(&run).join("config.toml")).expect("run config");
    let text = fs::read_to_string(Path::new(&run).join("genomes.jsonl")).expect("stored genomes");

    let mut stored: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("genome record"))
        .collect();
    stored
        .sort_by(|a, b| b["fitness"].as_f64().unwrap().total_cmp(&a["fitness"].as_f64().unwrap()));

    // Three champions, so no single body's quirk is mistaken for the world's.
    let corpses: Vec<Genome> = stored
        .iter()
        .take(3)
        .map(|r| {
            let mut g: Genome = serde_json::from_value(r["genome"].clone()).expect("genome");
            g.caution = 1.0; // motors fully off, with min_drive below
            g
        })
        .collect();

    println!("three champions from {run}, motors off, mean heading progress over 8 s");
    println!("(each is also given the amplitude/wavelength ratio, which sets slope)\n");
    println!("  terrain                                  a/w    mean heading");

    let trial = |label: &str, f: &dyn Fn(&mut Config)| {
        let mut cfg = base.clone();
        cfg.body.min_drive = 0.0;
        f(&mut cfg);
        cfg.validate().expect("valid");
        let ratio = match cfg.environment.terrain {
            Terrain::Flat => 0.0,
            _ => cfg.environment.terrain_amplitude / cfg.environment.terrain_wavelength,
        };
        let total: Real =
            corpses.iter().map(|g| sim::evaluate(g, &cfg, false).metrics.heading_progress).sum();
        println!("  {label:<40} {ratio:5.3}  {:8.2} m", total / corpses.len() as Real);
    };

    trial("flat", &|c| c.environment.terrain = Terrain::Flat);

    for (a, w) in [(0.05, 1.6), (0.25, 1.6), (0.25, 3.0), (0.05, 3.0)] {
        trial(&format!("rough   a={a:.2} w={w:.1}"), &|c| {
            c.environment.terrain = Terrain::Rough;
            c.environment.terrain_amplitude = a;
            c.environment.terrain_wavelength = w;
        });
    }

    for (a, w, oct) in [
        (0.25, 3.0, 4),
        (0.25, 3.0, 1),
        (0.25, 3.0, 2),
        (0.05, 3.0, 4),
        (0.25, 12.0, 4),
        (0.25, 24.0, 4),
        (0.10, 3.0, 4),
    ] {
        trial(&format!("fractal a={a:.2} w={w:<4.1} oct={oct}"), &|c| {
            c.environment.terrain = Terrain::Fractal;
            c.environment.terrain_amplitude = a;
            c.environment.terrain_wavelength = w;
            c.environment.terrain_octaves = oct;
        });
    }

    // Solver settings, on the shipped terrain: if the positional correction is
    // the source, turning it down should turn the conveyor off.
    println!("\n  on the shipped fractal terrain, varying the solver:");
    for speed in [2.0, 1.0, 0.5, 0.25, 0.1] {
        trial(&format!("max_correction_speed = {speed:.2}"), &|c| {
            c.environment.terrain = Terrain::Fractal;
            c.environment.terrain_amplitude = 0.25;
            c.environment.terrain_wavelength = 3.0;
            c.simulation.max_correction_speed = speed;
        });
    }
    for baumgarte in [0.2, 0.05, 0.0] {
        trial(&format!("baumgarte = {baumgarte:.2}"), &|c| {
            c.environment.terrain = Terrain::Fractal;
            c.environment.terrain_amplitude = 0.25;
            c.environment.terrain_wavelength = 3.0;
            c.simulation.baumgarte = baumgarte;
        });
    }
    for slop in [0.002, 0.02] {
        trial(&format!("slop = {slop:.3}"), &|c| {
            c.environment.terrain = Terrain::Fractal;
            c.environment.terrain_amplitude = 0.25;
            c.environment.terrain_wavelength = 3.0;
            c.simulation.slop = slop;
        });
    }
    for dt in [1.0 / 120.0, 1.0 / 480.0] {
        trial(&format!("timestep = 1/{:.0}", 1.0 / dt), &|c| {
            c.environment.terrain = Terrain::Fractal;
            c.environment.terrain_amplitude = 0.25;
            c.environment.terrain_wavelength = 3.0;
            c.simulation.timestep = dt;
        });
    }

    // The other constraint that pushes bodies apart. If the propulsion is parts
    // shoving each other rather than the ground shoving them, this turns it off.
    println!(
        "
  with the terrain held flat, isolating what is pushing:"
    );
    trial("self_collision on  (flat)", &|c| {
        c.environment.terrain = Terrain::Flat;
        c.environment.self_collision = true;
    });
    trial("self_collision off (flat)", &|c| {
        c.environment.terrain = Terrain::Flat;
        c.environment.self_collision = false;
    });
    trial("self_collision off (fractal)", &|c| {
        c.environment.terrain = Terrain::Fractal;
        c.environment.terrain_amplitude = 0.25;
        c.environment.terrain_wavelength = 3.0;
        c.environment.self_collision = false;
    });
    trial("self_collision on, correction 0.25 (flat)", &|c| {
        c.environment.terrain = Terrain::Flat;
        c.environment.self_collision = true;
        c.simulation.max_correction_speed = 0.25;
    });
}
