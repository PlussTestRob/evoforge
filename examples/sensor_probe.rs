//! What do the sensors actually see?
//!
//! A range sensor is only worth its weight if it reports something that varies.
//! Two failure modes are invisible from a fitness curve and look identical to
//! each other: a sensor aimed at the sky, which reads zero forever, and one
//! aimed at the ground under its own body, which reads nearly one forever. Both
//! are a constant, and a constant input is a bias term the organism is carrying
//! on a stalk.
//!
//! This settles an organism, then sweeps it across the terrain it evolved on and
//! reports the readings each sensor produces. Wide spread means the sensor is
//! informative; a flat line means it is ballast.
//!
//!   cargo run --release --example sensor_probe -- <run-dir>

use std::fs;
use std::path::Path;

use evoforge::config::Config;
use evoforge::genome::Genome;
use evoforge::math::{vec3, Real, Vec3};
use evoforge::phenotype;

fn main() {
    let run = std::env::args().nth(1).expect("usage: sensor_probe <run-dir>");
    let cfg = Config::load(&Path::new(&run).join("config.toml")).expect("run config");
    let text = fs::read_to_string(Path::new(&run).join("genomes.jsonl")).expect("stored genomes");

    if !cfg.uses_sensors() {
        println!("{run} has no sensors");
        return;
    }

    let mut stored: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    stored
        .sort_by(|a, b| b["fitness"].as_f64().unwrap().total_cmp(&a["fitness"].as_f64().unwrap()));

    println!("{run}");
    println!(
        "range {} m, {} rays, spread {}\n",
        cfg.sensor.range, cfg.sensor.rays, cfg.sensor.spread
    );
    println!("   id | gen | parts | sensors | aim (world, settled)      | readings: min  mean   max  spread");

    for record in stored.iter().take(8) {
        let genome: Genome = serde_json::from_value(record["genome"].clone()).expect("genome");
        let n_sensors = genome.parts.iter().filter(|p| p.sensor).count();
        if n_sensors == 0 {
            println!(
                " {:4} | {:3} | {:5} | {:7} | (blind)",
                record["id"].as_u64().unwrap_or(0),
                record["generation"].as_u64().unwrap_or(0),
                genome.parts.len(),
                0
            );
            continue;
        }

        // Settle it, so the pose is the one the measured window starts from
        // rather than the drop pose.
        let mut pheno = phenotype::build(&genome, &cfg);
        let dt = cfg.simulation.timestep;
        let settle = (cfg.simulation.settle_time / dt) as u32;
        for _ in 0..settle {
            pheno.world.step(dt);
        }

        // Sweep the organism across the landscape it evolved on and read the
        // sensors at each station. One pose says nothing; the spread over many
        // is what says whether the sensor carries information.
        let mut readings: Vec<Real> = Vec::new();
        let mut aim = Vec3::ZERO;
        let terrain = pheno.world.params.terrain;
        for step in 0..64 {
            let offset = vec3(step as Real * 0.9 - 28.0, 0.0, (step % 7) as Real * 1.3 - 4.0);
            for mount in &pheno.sensors {
                let body = &pheno.world.bodies[mount.body as usize];
                let a = body.orient.rotate(mount.dir);
                aim = a;
                let up = body.orient.rotate(Vec3::Y);
                let perp = (up - a * a.dot(up)).normalize_or(Vec3::Y);
                let origin = body.pos + offset;
                for r in 0..cfg.sensor.rays {
                    let k = r as Real - (cfg.sensor.rays as Real - 1.0) * 0.5;
                    let dir = (a + perp * (k * cfg.sensor.spread)).normalize_or(a);
                    let v = match terrain.raycast(origin, dir, cfg.sensor.range) {
                        Some(d) => 1.0 - (d / cfg.sensor.range).clamp(0.0, 1.0),
                        None => 0.0,
                    };
                    readings.push(v);
                }
            }
        }

        let n = readings.len() as Real;
        let mean = readings.iter().sum::<Real>() / n;
        let min = readings.iter().cloned().fold(Real::INFINITY, Real::min);
        let max = readings.iter().cloned().fold(Real::NEG_INFINITY, Real::max);
        let sd = (readings.iter().map(|v| (v - mean) * (v - mean)).sum::<Real>() / n).sqrt();

        println!(
            " {:4} | {:3} | {:5} | {:7} | {:+5.2} {:+5.2} {:+5.2}       | {:5.2} {:5.2} {:5.2}  {:6.3}{}",
            record["id"].as_u64().unwrap_or(0),
            record["generation"].as_u64().unwrap_or(0),
            genome.parts.len(),
            n_sensors,
            aim.x,
            aim.y,
            aim.z,
            min,
            mean,
            max,
            sd,
            if sd < 0.02 { "   <- constant: ballast" } else { "" },
        );
    }
}
