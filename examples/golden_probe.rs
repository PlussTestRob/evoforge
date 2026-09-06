//! Prints the golden constants baked into tests/golden.rs. Not part of the suite.
use evoforge::config::Config;
use evoforge::evolution::{self, Population};
use evoforge::genome::Genome;
use evoforge::rng::Rng;
use evoforge::sim;

const GOLDEN_CONFIG: &str = include_str!("../tests/golden.toml");

fn main() {
    let cfg = Config::from_toml_str(GOLDEN_CONFIG).unwrap();
    println!("EVOLUTION_DIGEST = 0x{:016x};", cfg.evolution_digest());
    println!("FULL_DIGEST = 0x{:016x};", cfg.digest());

    let layout = cfg.brain_layout();
    let g = Genome::random(&mut Rng::new(0xC0FFEE), &cfg.body, &cfg.brain, &layout);
    println!("STRUCTURE_HASH = 0x{:016x};", g.structure_hash());
    let r = sim::evaluate(&g, &cfg, false);
    let m = r.metrics;
    println!("ONE_ORGANISM = [");
    for (name, bits) in [
        ("fitness", r.fitness.to_bits()),
        ("displacement", m.displacement.to_bits()),
        ("displacement_x", m.displacement_x.to_bits()),
        ("path_length", m.path_length.to_bits()),
        ("max_displacement", m.max_displacement.to_bits()),
        ("mean_height", m.mean_height.to_bits()),
        ("upright_seconds", m.upright_seconds.to_bits()),
        ("actuation", m.actuation.to_bits()),
    ] {
        println!("    0x{bits:08x}, // {name}");
    }
    println!("];");

    let mut pop = Population::founding(&cfg);
    println!("GENERATION_BESTS = [");
    for _ in 0..4 {
        evolution::evaluate_population_serial(&mut pop, &cfg);
        let b = pop.best();
        println!("    (0x{:08x}, {}), // gen {}", b.fitness.to_bits(), b.id, pop.generation);
        pop = evolution::next_generation(&pop, &cfg);
    }
    println!("];");
}
