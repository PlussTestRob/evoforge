//! Integration tests for the evolutionary pipeline.
//!
//! The unit tests check that each stage does what it says. These check the
//! claims that only make sense end to end:
//!
//! * evolution actually improves fitness, rather than merely running;
//! * the same seed gives the same results regardless of how many cores ran it;
//! * a champion can be reconstructed exactly from its stored genome, which is
//!   what the whole recording strategy depends on.

use std::fs;
use std::path::PathBuf;

use evoforge::config::Config;
use evoforge::evolution::{self, Population};
use evoforge::record::{self, Run};
use evoforge::runner::{self, RunOptions};
use evoforge::sim;
use evoforge::stats;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!(
            "evoforge-it-{tag}-{}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn base_config(dir: &TempDir, tag: &str) -> Config {
    let mut cfg = Config::default();
    cfg.experiment.name = tag.into();
    cfg.experiment.output_dir = dir.0.clone();
    cfg.experiment.seed = 424_242;
    cfg.evolution.population_size = 40;
    cfg.evolution.generations = 12;
    cfg.simulation.duration = 2.0;
    cfg.simulation.settle_time = 0.3;
    cfg.recording.every_generations = 4;
    cfg.checkpoint.every_generations = 6;
    cfg
}

/// The central claim of the whole project: run the loop and organisms get
/// better. Median rather than best, because with elitism the best score cannot
/// go down whether or not anything is being learned — a rising median means the
/// population as a whole is improving.
#[test]
fn evolution_improves_the_population() {
    let tmp = TempDir::new("improves");
    let mut cfg = base_config(&tmp, "improves");
    cfg.evolution.generations = 20;

    let mut pop = Population::founding(&cfg);
    let pool = runner::build_pool(0).unwrap();
    let mut history = Vec::new();

    for _ in 0..cfg.evolution.generations {
        evolution::evaluate_population(&mut pop, &cfg, &pool);
        history.push(stats::summarise(&pop, 1.0, 1.0));
        pop = evolution::next_generation(&pop, &cfg);
    }

    let first = &history[0];
    let last = history.last().unwrap();

    assert!(
        last.best_fitness > first.best_fitness * 1.5,
        "best fitness barely moved: {} -> {}",
        first.best_fitness,
        last.best_fitness
    );
    assert!(
        last.median_fitness > first.median_fitness * 2.0,
        "the population as a whole did not improve: median {} -> {}",
        first.median_fitness,
        last.median_fitness
    );
    // Elitism guarantees this; if it ever fails, selection is broken.
    for w in history.windows(2) {
        assert!(
            w[1].best_fitness >= w[0].best_fitness - 1e-6,
            "best fitness regressed at generation {}",
            w[1].generation
        );
    }
    // A population that has collapsed onto one morphology is not evolving any
    // more, whatever the fitness curve says.
    assert!(
        last.unique_structures > 5,
        "structural diversity collapsed to {}",
        last.unique_structures
    );
    assert_eq!(last.diverged, 0, "the physics gave up on some organisms");
}

#[test]
fn the_same_seed_gives_the_same_results_on_any_number_of_threads() {
    let tmp = TempDir::new("threads");
    let cfg = base_config(&tmp, "threads");

    let evolve = |threads: usize| -> Vec<(u64, u32)> {
        let pool = runner::build_pool(threads).unwrap();
        let mut pop = Population::founding(&cfg);
        let mut out = Vec::new();
        for _ in 0..cfg.evolution.generations {
            evolution::evaluate_population(&mut pop, &cfg, &pool);
            for i in &pop.individuals {
                out.push((i.id, i.fitness.to_bits()));
            }
            pop = evolution::next_generation(&pop, &cfg);
        }
        out
    };

    let single = evolve(1);
    assert_eq!(single.len(), 40 * 12);
    assert_eq!(single, evolve(4));
    assert_eq!(single, evolve(8));
}

#[test]
fn a_different_seed_gives_a_different_experiment() {
    let tmp = TempDir::new("seeds");
    let cfg = base_config(&tmp, "seeds");
    let mut other = cfg.clone();
    other.experiment.seed += 1;

    let run_once = |cfg: &Config| -> f32 {
        let mut pop = Population::founding(cfg);
        for _ in 0..5 {
            evolution::evaluate_population_serial(&mut pop, cfg);
            pop = evolution::next_generation(&pop, cfg);
        }
        evolution::evaluate_population_serial(&mut pop, cfg);
        pop.best().fitness
    };

    assert_ne!(run_once(&cfg), run_once(&other));
}

/// The recording strategy only works if a stored genome can be re-simulated
/// exactly. If this drifts, champions are not reproducible and replays are
/// fiction.
#[test]
fn a_recorded_champion_re_simulates_to_the_same_fitness() {
    let tmp = TempDir::new("champion");
    let cfg = base_config(&tmp, "champion");
    let summary =
        runner::run(&cfg, &RunOptions { threads: 4, quiet: true, ..Default::default() }).unwrap();

    let run = Run::open(&summary.dir).unwrap();
    let stored_text = fs::read_to_string(summary.dir.join(record::GENOMES_FILE)).unwrap();
    let stored: Vec<record::StoredGenome> = stored_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert!(!stored.is_empty(), "no genomes were stored");

    for genome in &stored {
        let again = sim::evaluate(&genome.genome, &cfg, false);
        assert_eq!(
            again.fitness.to_bits(),
            genome.fitness.to_bits(),
            "organism {} re-simulated to {} instead of {}",
            genome.id,
            again.fitness,
            genome.fitness
        );
    }

    // And the same organism must be findable by id, from either store.
    let first = &stored[0];
    let found = run.find_genome(first.id).unwrap().unwrap();
    assert_eq!(found.genome, first.genome);
}

/// Recording at a higher rate must not perturb the physics; the only difference
/// should be how often the pose is sampled.
#[test]
fn replaying_at_higher_fidelity_does_not_change_the_outcome() {
    let tmp = TempDir::new("fidelity");
    let mut cfg = base_config(&tmp, "fidelity");
    cfg.recording.record_hz = 10.0;

    let mut pop = Population::founding(&cfg);
    evolution::evaluate_population_serial(&mut pop, &cfg);
    let champion = pop.best().clone();

    let low = sim::evaluate(&champion.genome, &cfg, true);
    let mut hi_cfg = cfg.clone();
    hi_cfg.recording.record_hz = 120.0;
    let high = sim::evaluate(&champion.genome, &hi_cfg, true);

    assert_eq!(low.fitness.to_bits(), high.fitness.to_bits());
    assert_eq!(low.metrics, high.metrics);
    let low_frames = low.trace.unwrap().frames.len();
    let high_frames = high.trace.unwrap().frames.len();
    assert!(high_frames > low_frames * 5, "{high_frames} frames at 120Hz vs {low_frames} at 10Hz");
}

#[test]
fn a_run_directory_is_self_describing() {
    let tmp = TempDir::new("artefacts");
    let cfg = base_config(&tmp, "artefacts");
    let summary =
        runner::run(&cfg, &RunOptions { threads: 2, quiet: true, ..Default::default() }).unwrap();

    // Everything needed to understand or continue the run is in one directory.
    let run = Run::open(&summary.dir).unwrap();
    assert_eq!(run.config().unwrap(), cfg, "the run stores its resolved config");
    assert_eq!(run.manifest.seed, cfg.experiment.seed);
    assert_eq!(run.manifest.config_digest, cfg.digest());

    let organisms = fs::read_to_string(summary.dir.join(record::ORGANISMS_FILE)).unwrap();
    let records: Vec<record::OrganismRecord> = organisms
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(records.len(), 40 * 12, "every organism is recorded exactly once");

    // Lineage is complete: every non-founder parent is an organism we recorded.
    let known: std::collections::HashSet<u64> = records.iter().map(|r| r.id).collect();
    for r in &records {
        for p in r.parents {
            assert!(p == 0 || known.contains(&p), "organism {} cites unknown parent {p}", r.id);
        }
    }

    assert!(run.latest_checkpoint().unwrap().is_some());
    let replays = fs::read_dir(summary.dir.join(record::REPLAY_DIR)).unwrap().count();
    assert!(replays > 0);
}

/// Travel must survive refining the solver.
///
/// This is the standing guard against the whole family of bugs where an
/// organism is propelled by discretisation rather than by physics. Real
/// locomotion is a property of the equations, so solving them more accurately
/// perturbs a trajectory but does not systematically abolish the distance
/// covered. Anything that *only* works at a coarse step is the integrator
/// moving the organism, and evolution finds it fast: it is what the
/// self-collision conveyor was, and it is what a population of champions was
/// living on when `simulation.baumgarte` still defaulted to 0.2.
///
/// Measured then: refining 12 solver iterations to 96 removed 94% of an evolved
/// champion's travel. Measured now, over unevolved organisms, retention averages
/// around 100% and never collapses. The threshold is set well below that and
/// well above the failure, because individual trajectories genuinely do diverge
/// under refinement — it is the *systematic* loss that indicts.
#[test]
fn travel_survives_refining_the_solver() {
    use evoforge::genome::Genome;
    use evoforge::rng::Rng;

    let mut cfg = Config::load(&PathBuf::from("experiments/animals.toml")).expect("config");
    cfg.simulation.trials = 1;
    let layout = cfg.brain_layout();

    let mut coarse_total = 0.0;
    let mut fine_total = 0.0;
    let mut counted = 0;
    for seed in 0..24u64 {
        let g = Genome::random(&mut Rng::new(seed), &cfg.body, &cfg.brain, &layout);

        let mut coarse = cfg.clone();
        coarse.simulation.solver_iterations = 12;
        let dc = sim::evaluate(&g, &coarse, false).metrics.displacement;
        // Organisms that barely move say nothing about refinement either way,
        // and their ratios are noise.
        if dc < 0.15 {
            continue;
        }

        let mut fine = cfg.clone();
        fine.simulation.solver_iterations = 96;
        let df = sim::evaluate(&g, &fine, false).metrics.displacement;

        coarse_total += dc;
        fine_total += df;
        counted += 1;
    }

    assert!(counted >= 8, "only {counted} organisms moved enough to judge");
    let retained = fine_total / coarse_total;
    assert!(
        retained > 0.5,
        "refining 12 solver iterations to 96 left only {:.0}% of the travel across {counted} \
         organisms. Distance that exists only at a coarse solve is the integrator propelling \
         the organism, not the organism moving.",
        retained * 100.0
    );
}
