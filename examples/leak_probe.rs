//! Where does the un-earned travel come from?
//!
//! An evolved champion's distance collapses by ~94% when the solver is refined,
//! so most of it is discretisation rather than physics. Three candidates, ruled
//! in or out here:
//!
//! * internal forces creating momentum — tested in weightlessness with no
//!   contact, where the centre of mass must not move at all;
//! * the Baumgarte positional correction at ground contacts displacing the body;
//! * friction failing to converge.
//!
//!   cargo run --release --example leak_probe -- <run-dir>

use std::fs;
use std::path::Path;

use evoforge::config::{Config, Terrain};
use evoforge::genome::Genome;
use evoforge::math::Real;
use evoforge::sim;

fn main() {
    let run = std::env::args().nth(1).expect("usage: leak_probe <run-dir>");
    let base = Config::load(&Path::new(&run).join("config.toml")).expect("run config");
    let text = fs::read_to_string(Path::new(&run).join("genomes.jsonl")).expect("stored genomes");

    let mut stored: Vec<serde_json::Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    stored
        .sort_by(|a, b| b["fitness"].as_f64().unwrap().total_cmp(&a["fitness"].as_f64().unwrap()));
    let genomes: Vec<Genome> = stored
        .iter()
        .take(5)
        .map(|r| serde_json::from_value(r["genome"].clone()).expect("genome"))
        .collect();
    let mean = |cfg: &Config| -> Real {
        genomes.iter().map(|g| sim::evaluate(g, cfg, false).metrics.displacement).sum::<Real>()
            / genomes.len() as Real
    };

    // --- 1. Internal forces alone, with nothing to push against. -----------
    // Weightless rather than free-falling: flat terrain sits at height 0, so a
    // falling organism lands within a step and the test measures ordinary
    // locomotion instead. With gravity off it hangs where it spawned and never
    // touches anything.
    let mut air = base.clone();
    air.environment.terrain = Terrain::Flat;
    air.environment.gravity = 0.0;
    air.simulation.settle_time = 0.0;
    air.simulation.trials = 1;
    air.simulation.start_jitter = 0.0;
    println!("{run}\n");
    println!("1. weightless, no contact — internal forces cannot move a centre of mass");
    println!("   travel: {:.4} m  (a small residual; not the source)\n", mean(&air));

    // --- 2. On the ground, sweeping the positional correction. -------------
    let mut ground = base.clone();
    ground.simulation.trials = 1;
    ground.simulation.start_jitter = 0.0;
    println!("2. on the ground, sweeping Baumgarte positional correction");
    println!("   baumgarte | iters=12 | iters=48 | iters=96");
    for beta in [base.simulation.baumgarte, 0.1, 0.05, 0.0] {
        let mut row = Vec::new();
        for it in [12u32, 48, 96] {
            let mut c = ground.clone();
            c.simulation.baumgarte = beta;
            c.simulation.solver_iterations = it;
            row.push(mean(&c));
        }
        println!("   {beta:9.2} | {:8.3} | {:8.3} | {:8.3}", row[0], row[1], row[2]);
    }

    // --- 3. Cost of the mitigations. ---------------------------------------
    println!("\n3. what each mitigation costs, and what travel survives it");
    println!("   setting                          | travel | relative cost");
    let t0 = std::time::Instant::now();
    let baseline = mean(&ground);
    let cost0 = t0.elapsed().as_secs_f64();
    println!("   as shipped (dt 1/120, 12 iters)  | {baseline:6.3} | 1.00x");
    for (label, f) in [
        ("48 iterations", 0u8),
        ("96 iterations", 1),
        ("dt 1/240", 2),
        ("dt 1/480", 3),
        ("baumgarte 0.05", 4),
    ] {
        let mut c = ground.clone();
        match f {
            0 => c.simulation.solver_iterations = 48,
            1 => c.simulation.solver_iterations = 96,
            2 => c.simulation.timestep = 1.0 / 240.0,
            3 => c.simulation.timestep = 1.0 / 480.0,
            _ => c.simulation.baumgarte = 0.05,
        }
        let t = std::time::Instant::now();
        let d = mean(&c);
        let cost = t.elapsed().as_secs_f64() / cost0.max(1e-9);
        println!("   {label:32} | {d:6.3} | {cost:.2}x");
    }
}
