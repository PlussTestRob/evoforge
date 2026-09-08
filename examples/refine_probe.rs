//! Which subsystem is stealing the refinement?
//!
//! An evolved champion's travel collapses by 90% when the timestep or the solver
//! iteration count is refined, which means most of it is discretisation rather
//! than physics. This turns the parts off one at a time and re-runs the sweep,
//! so the responsible subsystem names itself.
//!
//!   cargo run --release --example refine_probe -- <run-dir>

use std::fs;
use std::path::Path;

use evoforge::config::Config;
use evoforge::genome::Genome;
use evoforge::math::Real;
use evoforge::sim;

fn main() {
    let run = std::env::args().nth(1).expect("usage: refine_probe <run-dir>");
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

    let run_with = |cfg: &Config, gs: &[Genome]| -> Real {
        gs.iter().map(|g| sim::evaluate(g, cfg, false).metrics.heading_progress).sum::<Real>()
            / gs.len() as Real
    };

    // Motors off, via the organism's own throttle rather than a special case.
    let dead: Vec<Genome> = genomes
        .iter()
        .map(|g| {
            let mut g = g.clone();
            g.caution = 1.0;
            g
        })
        .collect();

    println!("{run}\n");
    println!("  mean heading progress of the top 5, at 12 and 96 solver iterations");
    println!("  a condition whose travel survives refinement is real physics\n");
    println!("  condition                      | iters=12 | iters=96 | retained");

    let report = |label: &str, mut cfg: Config, gs: &[Genome]| {
        cfg.simulation.solver_iterations = 12;
        let coarse = run_with(&cfg, gs);
        cfg.simulation.solver_iterations = 96;
        let fine = run_with(&cfg, gs);
        println!(
            "  {label:30} | {coarse:8.3} | {fine:8.3} | {:7.0}%",
            100.0 * fine / coarse.abs().max(1e-6)
        );
    };

    report("as shipped", base.clone(), &genomes);

    let mut no_self = base.clone();
    no_self.environment.self_collision = false;
    report("self-collision off", no_self, &genomes);

    let mut no_tendon = base.clone();
    no_tendon.body.tendon_frequency = 0.0;
    report("tendons off", no_tendon, &genomes);

    let mut no_both = base.clone();
    no_both.environment.self_collision = false;
    no_both.body.tendon_frequency = 0.0;
    report("self-collision + tendons off", no_both, &genomes);

    let mut dead_cfg = base.clone();
    dead_cfg.body.min_drive = 0.0;
    report("motors off", dead_cfg.clone(), &dead);

    let mut flat = base.clone();
    flat.environment.terrain = evoforge::config::Terrain::Flat;
    report("flat ground", flat, &genomes);

    let mut flat_no_self = base.clone();
    flat_no_self.environment.terrain = evoforge::config::Terrain::Flat;
    flat_no_self.environment.self_collision = false;
    report("flat ground, self-collision off", flat_no_self, &genomes);
}
