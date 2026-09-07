//! How far does a *dead* organism travel?
//!
//! Takes the champions a run actually produced, switches their motors fully off
//! (`caution = 1`, `min_drive = 0`, so every motor target is zero), and
//! evaluates them again on the same ground.
//!
//! A corpse should fall over and stop. Whatever distance it covers instead was
//! not earned by the controller — it was given away by the world, and any
//! locomotion score above that number is the only part that means anything.
//!
//!   cargo run --release --example dead_organism_probe -- <run-dir> [<run-dir>...]

use std::fs;

use evoforge::config::Config;
use evoforge::genome::Genome;
use evoforge::sim;

fn main() {
    let runs: Vec<String> = std::env::args().skip(1).collect();
    if runs.is_empty() {
        eprintln!("usage: dead_organism_probe <run-dir> [<run-dir>...]");
        std::process::exit(2);
    }

    for run in runs {
        let cfg = Config::load(std::path::Path::new(&run).join("config.toml").as_path())
            .expect("run config");
        let text = fs::read_to_string(std::path::Path::new(&run).join("genomes.jsonl"))
            .expect("stored genomes");

        // The best few stored genomes, so this is not one lucky organism.
        let mut stored: Vec<serde_json::Value> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("genome record"))
            .collect();
        stored.sort_by(|a, b| {
            b["fitness"].as_f64().unwrap().total_cmp(&a["fitness"].as_f64().unwrap())
        });

        println!("\n=== {run} ({}) ===", serde_json::to_string(&cfg.environment.terrain).unwrap());
        println!("   id |  alive fit   heading |   dead fit   heading | dead share");

        let mut alive_total = 0.0;
        let mut dead_total = 0.0;
        for record in stored.iter().take(5) {
            let genome: Genome = serde_json::from_value(record["genome"].clone()).expect("genome");

            let alive = sim::evaluate(&genome, &cfg, false);

            // Motors fully off. `caution` is the organism's own throttle and
            // `min_drive` its floor, so this is the existing mechanism turned to
            // zero rather than a special case in the simulator.
            let mut dead_cfg = cfg.clone();
            dead_cfg.body.min_drive = 0.0;
            let mut corpse = genome.clone();
            corpse.caution = 1.0;
            let dead = sim::evaluate(&corpse, &dead_cfg, false);

            alive_total += alive.metrics.heading_progress;
            dead_total += dead.metrics.heading_progress;
            println!(
                " {:4} | {:10.3} {:9.3} | {:10.3} {:9.3} | {:9.0}%",
                record["id"].as_u64().unwrap_or(0),
                alive.fitness,
                alive.metrics.heading_progress,
                dead.fitness,
                dead.metrics.heading_progress,
                100.0 * dead.metrics.heading_progress / alive.metrics.heading_progress.max(1e-6),
            );
        }
        println!(
            "  mean heading: alive {:.2} m, dead {:.2} m — {:.0}% of the distance is free",
            alive_total / 5.0,
            dead_total / 5.0,
            100.0 * dead_total / alive_total.max(1e-6),
        );
    }
}
