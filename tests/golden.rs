//! Golden values: the reproducibility contract, pinned to constants.
//!
//! Every other determinism test in this repository compares two runs *inside the
//! same process on the same binary* — `evo verify`, `parallel_and_serial_evaluation_agree`,
//! `the_same_seed_gives_the_same_results_on_any_number_of_threads`. Those catch
//! order dependence and thread races. None of them can catch the failure the
//! whole design is arranged against: the simulator quietly producing *different
//! numbers* than it used to, on a different platform, a different compiler, or
//! after a refactor.
//!
//! That is what this file is for. `rng::stream_is_stable` does the same job one
//! layer down, and this is its counterpart for the pipeline as a whole. Together
//! with a multi-platform CI matrix it is what turns "results replay identically
//! years from now" from an assertion into a test.
//!
//! # If a test here fails
//!
//! Do not update the constant to make it pass. A failure means one of:
//!
//! * the dynamics changed deliberately — then the constants are stale, and the
//!   change also needs a version bump, because stored genomes will no longer
//!   re-simulate to their recorded fitness;
//! * the dynamics changed accidentally — a refactor was not the no-op it looked
//!   like, and this is the bug report;
//! * the platform is not reproducing the reference values — which is precisely
//!   the thing `math`'s hand-rolled transcendentals exist to prevent, and worth
//!   investigating before trusting any result from that machine.
//!
//! The constants were produced on x86-64 by `cargo run --example golden_probe`.

use evoforge::config::Config;
use evoforge::evolution::{self, Population};
use evoforge::genome::Genome;
use evoforge::math::Real;
use evoforge::rng::Rng;
use evoforge::sim;

/// Frozen alongside the constants below. Held in its own file rather than built
/// from `Config::default()` so that changing a default cannot silently redefine
/// what these values describe.
const GOLDEN_CONFIG: &str = include_str!("golden.toml");

/// Seed for the single organism evaluated below. Arbitrary, and fixed forever.
const ORGANISM_SEED: u64 = 0xC0FFEE;

const EVOLUTION_DIGEST: u64 = 0xc38b_740e_5a77_f1b0;
const FULL_DIGEST: u64 = 0xdf82_cf76_afd7_1a3f;
const STRUCTURE_HASH: u64 = 0x8fa4_6b52_502d_4257;

/// Bit patterns, not values: `assert_eq!` on floats would accept a result that
/// differs in the last place, and the last place is exactly where drift starts.
const ONE_ORGANISM: [(&str, u32); 8] = [
    ("fitness", 0x3d98_96f8),
    ("displacement", 0x3d26_1371),
    ("displacement_x", 0x3be2_2d08),
    ("path_length", 0x3daa_e39e),
    ("max_displacement", 0x3d42_d463),
    ("mean_height", 0x3e6f_40b6),
    ("upright_seconds", 0x3fbf_fff7),
    ("actuation", 0x43aa_84fb),
];

/// `(best fitness bits, best organism id)` for generations 0..4.
const GENERATION_BESTS: [(u32, u64); 4] =
    [(0x3f4d_837a, 4), (0x3f55_2b96, 27), (0x3f84_aff4, 43), (0x3f85_6935, 61)];

fn golden_config() -> Config {
    Config::from_toml_str(GOLDEN_CONFIG).expect("the frozen config must stay valid")
}

/// Guards the constants below against a *different question* being asked.
///
/// If a dynamics field is added, removed or given a different default, this fails
/// first and says so — which is far easier to act on than a bare fitness
/// mismatch further down.
#[test]
fn the_frozen_configuration_still_describes_the_same_experiment() {
    let cfg = golden_config();
    assert_eq!(
        cfg.evolution_digest(),
        EVOLUTION_DIGEST,
        "the dynamics covered by golden.toml changed; the golden values below \
         no longer describe the same experiment"
    );
    assert_eq!(cfg.digest(), FULL_DIGEST, "the full config digest changed");
}

/// The tightest pin available: one genome, one evaluation, every metric.
#[test]
fn one_organism_evaluates_to_its_recorded_bits() {
    let cfg = golden_config();
    let genome =
        Genome::random(&mut Rng::new(ORGANISM_SEED), &cfg.body, &cfg.brain, &cfg.brain_layout());
    assert_eq!(
        genome.structure_hash(),
        STRUCTURE_HASH,
        "genome generation drifted, so the organism being measured is not the \
         one these values describe"
    );

    let result = sim::evaluate(&genome, &cfg, false);
    let m = result.metrics;
    assert!(!m.diverged, "the golden organism must not diverge");

    let got: [(&str, u32); 8] = [
        ("fitness", result.fitness.to_bits()),
        ("displacement", m.displacement.to_bits()),
        ("displacement_x", m.displacement_x.to_bits()),
        ("path_length", m.path_length.to_bits()),
        ("max_displacement", m.max_displacement.to_bits()),
        ("mean_height", m.mean_height.to_bits()),
        ("upright_seconds", m.upright_seconds.to_bits()),
        ("actuation", m.actuation.to_bits()),
    ];
    for ((name, expected), (_, actual)) in ONE_ORGANISM.iter().zip(got) {
        assert_eq!(
            *expected,
            actual,
            "{name}: expected {} (0x{expected:08x}), got {} (0x{actual:08x})",
            Real::from_bits(*expected),
            Real::from_bits(actual)
        );
    }
}

/// Selection, crossover, mutation and immigration are all in the loop here, so
/// this pins the evolutionary machinery as well as the simulator. Ids are checked
/// too: the same fitness reached by a differently-numbered organism means lineage
/// bookkeeping moved.
#[test]
fn four_generations_reproduce_their_recorded_bits() {
    let cfg = golden_config();
    let mut pop = Population::founding(&cfg);
    for (generation, &(fitness, id)) in GENERATION_BESTS.iter().enumerate() {
        evolution::evaluate_population_serial(&mut pop, &cfg);
        let best = pop.best();
        assert_eq!(
            best.fitness.to_bits(),
            fitness,
            "generation {generation}: best fitness {} does not match the recorded {}",
            best.fitness,
            Real::from_bits(fitness)
        );
        assert_eq!(best.id, id, "generation {generation}: best organism id moved");
        pop = evolution::next_generation(&pop, &cfg);
    }
}

/// The golden values must not depend on how the population was evaluated. This is
/// the in-process determinism check pointed at the frozen configuration, so a
/// parallelism regression shows up here rather than only in a fresh run.
#[test]
fn the_golden_run_is_independent_of_thread_count() {
    let cfg = golden_config();
    let pool = evoforge::runner::build_pool(4).unwrap();
    let mut pop = Population::founding(&cfg);
    for &(fitness, id) in GENERATION_BESTS.iter() {
        evolution::evaluate_population(&mut pop, &cfg, &pool);
        let best = pop.best();
        assert_eq!(best.fitness.to_bits(), fitness);
        assert_eq!(best.id, id);
        pop = evolution::next_generation(&pop, &cfg);
    }
}
