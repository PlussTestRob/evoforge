//! Driving an experiment: the generation loop, recording policy and resume.
//!
//! This is the only module that does I/O during evolution, and the only one that
//! reads a clock. Everything it calls is deterministic; the loop's job is to
//! sequence those pure steps, persist their results, and report progress.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};

use crate::config::Config;
use crate::evolution::{self, Population};
use crate::record::{self, Run};
use crate::sim;
use crate::stats::{self, GenerationStats};

#[derive(Clone, Debug, Default)]
pub struct RunOptions {
    /// Worker threads for population evaluation. `0` means one per core.
    pub threads: usize,
    /// Suppress the per-generation table.
    pub quiet: bool,
    /// Resume from the latest checkpoint in this existing run directory.
    pub resume: Option<PathBuf>,
    /// Allow resume when the checkpoint was written by a different evoforge
    /// version. Dynamics may not match; the default is to refuse.
    pub force_resume: bool,
}

#[derive(Debug)]
pub struct RunSummary {
    pub dir: PathBuf,
    pub generations_completed: u32,
    pub organisms_evaluated: u64,
    pub wall_seconds: f64,
    pub final_stats: Option<GenerationStats>,
}

/// Run an experiment to completion.
pub fn run(cfg: &Config, opts: &RunOptions) -> Result<RunSummary> {
    // Validated here as well as in the CLI: this is a public entry point, and a
    // configuration that never went through `Config::load` has never been checked.
    cfg.validate().context("invalid configuration")?;

    let pool = build_pool(opts.threads)?;

    let (run, mut population) = match &opts.resume {
        Some(dir) => resume(dir, cfg, opts)?,
        None => {
            let run = Run::create(cfg).context("creating run directory")?;
            (run, Population::founding(cfg))
        }
    };

    // Continue the wall-clock column rather than restarting it, so a resumed run
    // does not appear to travel backwards in time at the resume boundary.
    let elapsed_before = run.elapsed_seconds_so_far()?;

    if !opts.quiet {
        println!(
            "experiment {}  seed {}  population {}  generations {}  threads {}",
            run.manifest.experiment_id,
            cfg.experiment.seed,
            cfg.evolution.population_size,
            cfg.evolution.generations,
            pool.current_num_threads(),
        );
        println!("output {}", run.dir.display());
        println!("{}", GenerationStats::TABLE_HEADER);
    }

    let started = Instant::now();
    let mut organisms_evaluated: u64 = 0;
    let mut final_stats = None;
    let mut generations_completed = 0;

    let mut last_checkpointed: Option<u32> = None;

    while population.generation < cfg.evolution.generations {
        let eval_started = Instant::now();
        evolution::evaluate_population(&mut population, cfg, &pool);
        let eval_seconds = eval_started.elapsed().as_secs_f64();
        organisms_evaluated += population.len() as u64;

        let elapsed = elapsed_before + started.elapsed().as_secs_f64();
        let summary = stats::summarise(&population, eval_seconds, elapsed);
        if !opts.quiet {
            if summary.generation % 20 == 0 && summary.generation > 0 {
                println!("{}", GenerationStats::TABLE_HEADER);
            }
            println!("{}", summary.to_table_row());
        }
        run.append_stats(&summary)?;
        run.append_organisms(&population)?;
        record_selected(&run, &population, cfg)?;

        let completed = population.generation;
        generations_completed += 1;
        final_stats = Some(summary);
        population = evolution::next_generation(&population, cfg);

        // Checkpoint *after* breeding, so the snapshot holds a generation that has
        // not yet been evaluated or written to disk. Checkpointing the generation
        // just finished instead would make a resume re-evaluate it and append a
        // second copy of its stats row and its organism records — which is exactly
        // what a preempted worker would hit, since it has no on-finish checkpoint
        // to land on. The schedule still counts completed generations; only the
        // population inside the file changed.
        if record::should_checkpoint(completed, cfg) {
            run.write_checkpoint(&population, cfg)?;
            last_checkpointed = Some(population.generation);
        }
    }

    // The loop leaves `population` holding the unevaluated next generation;
    // checkpointing it means a resume picks up exactly where this run stopped.
    // Skipped when the schedule already wrote this very generation.
    if cfg.checkpoint.on_finish && last_checkpointed != Some(population.generation) {
        run.write_checkpoint(&population, cfg)?;
    }

    Ok(RunSummary {
        dir: run.dir,
        generations_completed,
        organisms_evaluated,
        wall_seconds: started.elapsed().as_secs_f64(),
        final_stats,
    })
}

/// Re-simulate the organisms chosen by the recording policy, this time with
/// trajectory capture, and write them out.
///
/// Re-simulating rather than recording every organism speculatively is the whole
/// point of keeping evaluation pure: the extra work is a handful of evaluations
/// per recorded generation, and in exchange the hot loop never allocates a frame
/// buffer it is going to throw away.
fn record_selected(run: &Run, pop: &Population, cfg: &Config) -> Result<()> {
    for id in record::selection_for_recording(pop, cfg) {
        let Some(individual) = pop.find(id) else { continue };
        if cfg.recording.store_genomes {
            run.append_genome(individual)?;
        }
        let result = sim::evaluate(&individual.genome, cfg, true);
        if let Some(trace) = result.trace {
            run.write_replay(individual, trace, cfg)?;
        }
    }
    Ok(())
}

/// Load the newest checkpoint from an existing run directory.
fn resume(dir: &Path, cfg: &Config, opts: &RunOptions) -> Result<(Run, Population)> {
    let force = opts.force_resume;
    let run = Run::open(dir).with_context(|| format!("opening run {}", dir.display()))?;
    let Some(checkpoint) = run.latest_checkpoint()? else {
        bail!("{} has no checkpoints to resume from", dir.display());
    };

    // A checkpoint older than `MIN_RESUMABLE_FORMAT` holds a population that has
    // already been evaluated and written out, so resuming it would duplicate a
    // generation's records. The layout is still readable — `evo inspect` and
    // `evo replay` work fine against it — but continuing the run is not safe.
    if checkpoint.format < record::MIN_RESUMABLE_FORMAT && !force {
        bail!(
            "checkpoint is in format {} and holds an already-evaluated generation; \
             resuming it would append a second copy of generation {}'s records. \
             Start a new run, or pass --force-resume to accept the duplication",
            checkpoint.format,
            checkpoint.population.generation
        );
    }

    // A checkpoint is only meaningful under the configuration that produced it.
    // Refusing loudly is better than silently continuing an experiment whose
    // parameters changed underneath it.
    if checkpoint.config_digest != cfg.evolution_digest() {
        bail!(
            "config does not match the checkpoint (digest {:016x} vs {:016x}); \
             resume with the run's own {} or start a new run",
            cfg.evolution_digest(),
            checkpoint.config_digest,
            record::CONFIG_FILE
        );
    }
    if checkpoint.evoforge_version != crate::VERSION {
        if !force {
            bail!(
                "checkpoint was written by evoforge {}, this is {}; \
                 resume with --force-resume if you intend to continue anyway",
                checkpoint.evoforge_version,
                crate::VERSION
            );
        }
        eprintln!(
            "warning: checkpoint was written by evoforge {}, this is {}; \
             --force-resume accepted, results may not be comparable",
            checkpoint.evoforge_version,
            crate::VERSION
        );
    }

    let population = checkpoint.population;
    if !opts.quiet {
        println!(
            "resuming {} from generation {}",
            run.manifest.experiment_id, population.generation
        );
    }
    Ok((run, population))
}

pub fn build_pool(threads: usize) -> Result<rayon::ThreadPool> {
    let builder = rayon::ThreadPoolBuilder::new().num_threads(threads);
    builder.build().context("building the worker thread pool")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> TempDir {
            let dir = std::env::temp_dir().join(format!(
                "evoforge-runner-{tag}-{:?}-{}",
                std::thread::current().id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
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

    fn tiny_config(dir: &TempDir) -> Config {
        let mut cfg = Config::default();
        cfg.experiment.name = "runner-test".into();
        cfg.experiment.output_dir = dir.0.clone();
        cfg.evolution.population_size = 8;
        cfg.evolution.generations = 4;
        cfg.simulation.duration = 0.4;
        cfg.simulation.settle_time = 0.1;
        cfg.recording.every_generations = 2;
        cfg.checkpoint.every_generations = 2;
        cfg
    }

    #[test]
    fn a_run_produces_the_expected_artefacts() {
        let tmp = TempDir::new("artefacts");
        let cfg = tiny_config(&tmp);
        let summary =
            run(&cfg, &RunOptions { threads: 2, quiet: true, ..Default::default() }).unwrap();

        assert_eq!(summary.generations_completed, 4);
        assert_eq!(summary.organisms_evaluated, 32);

        let stats_text = fs::read_to_string(summary.dir.join(record::STATS_FILE)).unwrap();
        assert_eq!(stats_text.lines().count(), 5, "header plus four generations");

        let organisms = fs::read_to_string(summary.dir.join(record::ORGANISMS_FILE)).unwrap();
        assert_eq!(organisms.lines().count(), 32);

        let replays: Vec<_> = fs::read_dir(summary.dir.join(record::REPLAY_DIR))
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(!replays.is_empty(), "recording policy produced no replays");

        let checkpoints: Vec<_> = fs::read_dir(summary.dir.join(record::CHECKPOINT_DIR))
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert!(!checkpoints.is_empty());
    }

    #[test]
    fn identical_seeds_produce_identical_runs() {
        let tmp = TempDir::new("determinism");
        let cfg = tiny_config(&tmp);

        let read_stats = |dir: &Path| fs::read_to_string(dir.join(record::STATS_FILE)).unwrap();
        let strip_timings = |text: String| -> Vec<String> {
            text.lines().map(|l| l.split(',').take(10).collect::<Vec<_>>().join(",")).collect()
        };

        let a = run(&cfg, &RunOptions { threads: 1, quiet: true, ..Default::default() }).unwrap();
        let b = run(&cfg, &RunOptions { threads: 4, quiet: true, ..Default::default() }).unwrap();

        assert_eq!(
            strip_timings(read_stats(&a.dir)),
            strip_timings(read_stats(&b.dir)),
            "runs differed despite identical seeds"
        );
    }

    #[test]
    fn a_resumed_run_continues_the_same_trajectory() {
        let tmp = TempDir::new("resume");
        let mut short = tiny_config(&tmp);
        short.evolution.generations = 2;
        short.checkpoint.every_generations = 0; // rely on the on-finish checkpoint

        let mut full = short.clone();
        full.evolution.generations = 4;

        // The uninterrupted reference run.
        let reference =
            run(&full, &RunOptions { threads: 2, quiet: true, ..Default::default() }).unwrap();

        // A run stopped after two generations, then extended. Raising
        // `generations` must not invalidate the checkpoint.
        let partial =
            run(&short, &RunOptions { threads: 2, quiet: true, ..Default::default() }).unwrap();
        let resumed = run(
            &full,
            &RunOptions {
                threads: 2,
                quiet: true,
                resume: Some(partial.dir.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(resumed.generations_completed, 2, "should only run the remaining two");

        let fitness_column = |dir: &Path| -> Vec<String> {
            fs::read_to_string(dir.join(record::STATS_FILE))
                .unwrap()
                .lines()
                .skip(1)
                .map(|l| l.split(',').take(6).collect::<Vec<_>>().join(","))
                .collect()
        };

        let reference_rows = fitness_column(&reference.dir);
        let resumed_rows = fitness_column(&resumed.dir);
        assert_eq!(resumed_rows.len(), 4, "resume appends to the existing stats file");
        assert_eq!(reference_rows, resumed_rows);
    }

    /// The preemption path: a run killed between checkpoints has no on-finish
    /// snapshot to land on, so it resumes from a periodic one. That must not
    /// replay a generation whose stats row and organism records are already on
    /// disk.
    #[test]
    fn resume_from_a_periodic_checkpoint_does_not_duplicate_a_generation() {
        let tmp = TempDir::new("periodic");
        let mut short = tiny_config(&tmp);
        short.evolution.generations = 3;
        short.checkpoint.every_generations = 1;
        short.checkpoint.on_finish = false; // as if killed mid-run

        let mut full = short.clone();
        full.evolution.generations = 5;

        let partial =
            run(&short, &RunOptions { threads: 1, quiet: true, ..Default::default() }).unwrap();
        run(
            &full,
            &RunOptions {
                threads: 1,
                quiet: true,
                resume: Some(partial.dir.clone()),
                ..Default::default()
            },
        )
        .unwrap();

        let text = fs::read_to_string(partial.dir.join(record::STATS_FILE)).unwrap();
        let generations: Vec<&str> = text
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split(',').next().unwrap())
            .collect();
        assert_eq!(
            generations,
            ["0", "1", "2", "3", "4"],
            "every generation must appear exactly once in stats.csv"
        );

        let organisms = fs::read_to_string(partial.dir.join(record::ORGANISMS_FILE)).unwrap();
        let mut ids: Vec<u64> = organisms
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<record::OrganismRecord>(l).unwrap().id)
            .collect();
        let total = ids.len();
        assert_eq!(total, 5 * 8, "one record per organism per generation");
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), total, "organisms.jsonl contains duplicate ids");
    }

    /// A checkpoint must hold a generation that has *not* been written out yet,
    /// which is what makes resume idempotent.
    #[test]
    fn a_checkpoint_holds_the_next_unevaluated_generation() {
        let tmp = TempDir::new("unevaluated-checkpoint");
        let mut cfg = tiny_config(&tmp);
        cfg.evolution.generations = 2;
        cfg.checkpoint.every_generations = 1;
        cfg.checkpoint.on_finish = false;

        let summary =
            run(&cfg, &RunOptions { threads: 1, quiet: true, ..Default::default() }).unwrap();
        let opened = Run::open(&summary.dir).unwrap();
        for path in opened.checkpoint_paths().unwrap() {
            let checkpoint: record::Checkpoint = record::read_json(&path).unwrap();
            assert!(
                checkpoint
                    .population
                    .individuals
                    .iter()
                    .all(|i| i.fitness == evolution::UNEVALUATED_FITNESS),
                "{} holds an already-evaluated population",
                path.display()
            );
        }
    }

    #[test]
    fn elapsed_seconds_does_not_go_backwards_across_a_resume() {
        let tmp = TempDir::new("elapsed");
        let mut short = tiny_config(&tmp);
        short.evolution.generations = 2;
        let mut full = short.clone();
        full.evolution.generations = 4;

        let partial =
            run(&short, &RunOptions { threads: 1, quiet: true, ..Default::default() }).unwrap();
        run(
            &full,
            &RunOptions {
                threads: 1,
                quiet: true,
                resume: Some(partial.dir.clone()),
                ..Default::default()
            },
        )
        .unwrap();

        let text = fs::read_to_string(partial.dir.join(record::STATS_FILE)).unwrap();
        let elapsed: Vec<f64> = text
            .lines()
            .skip(1)
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                l.split(',').nth(GenerationStats::ELAPSED_SECONDS_COLUMN).unwrap().parse().unwrap()
            })
            .collect();
        assert_eq!(elapsed.len(), 4);
        for w in elapsed.windows(2) {
            assert!(w[1] >= w[0], "elapsed went backwards: {elapsed:?}");
        }
    }

    /// A pre-v2 checkpoint holds an evaluated generation, so resuming it would
    /// duplicate records. It must be refused by default and only accepted when
    /// the operator says so.
    #[test]
    fn resume_refuses_a_pre_v2_checkpoint_unless_forced() {
        let tmp = TempDir::new("legacy");
        let cfg = tiny_config(&tmp);
        let first =
            run(&cfg, &RunOptions { threads: 1, quiet: true, ..Default::default() }).unwrap();

        let opened = Run::open(&first.dir).unwrap();
        let mut checkpoint = opened.latest_checkpoint().unwrap().unwrap();
        checkpoint.format = 1;
        record::write_json(&opened.checkpoint_path(checkpoint.population.generation), &checkpoint)
            .unwrap();

        let opts = |force| RunOptions {
            threads: 1,
            quiet: true,
            resume: Some(first.dir.clone()),
            force_resume: force,
        };
        let err = run(&cfg, &opts(false)).unwrap_err();
        assert!(err.to_string().contains("already-evaluated"), "{err}");
        assert!(run(&cfg, &opts(true)).is_ok());
    }

    #[test]
    fn run_rejects_an_invalid_configuration() {
        let tmp = TempDir::new("invalid");
        let mut cfg = tiny_config(&tmp);
        cfg.evolution.elite_count = cfg.evolution.population_size;
        let err = run(&cfg, &RunOptions { quiet: true, ..Default::default() }).unwrap_err();
        assert!(err.to_string().contains("invalid configuration"), "{err}");
    }

    #[test]
    fn resume_rejects_a_mismatched_config() {
        let tmp = TempDir::new("mismatch");
        let cfg = tiny_config(&tmp);
        let first =
            run(&cfg, &RunOptions { threads: 1, quiet: true, ..Default::default() }).unwrap();

        let mut changed = cfg.clone();
        changed.evolution.tournament_size += 1;
        let err = run(
            &changed,
            &RunOptions {
                threads: 1,
                quiet: true,
                resume: Some(first.dir.clone()),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");
    }

    #[test]
    fn resume_rejects_a_version_mismatch_unless_forced() {
        let tmp = TempDir::new("version");
        let cfg = tiny_config(&tmp);
        let first =
            run(&cfg, &RunOptions { threads: 1, quiet: true, ..Default::default() }).unwrap();

        let opened = Run::open(&first.dir).unwrap();
        let mut checkpoint = opened.latest_checkpoint().unwrap().unwrap();
        checkpoint.evoforge_version = "0.0.0-test".into();
        record::write_json(&opened.checkpoint_path(checkpoint.population.generation), &checkpoint)
            .unwrap();

        let err = run(
            &cfg,
            &RunOptions {
                threads: 1,
                quiet: true,
                resume: Some(first.dir.clone()),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("written by evoforge"), "{err}");

        let forced = run(
            &cfg,
            &RunOptions {
                threads: 1,
                quiet: true,
                resume: Some(first.dir.clone()),
                force_resume: true,
            },
        );
        assert!(forced.is_ok(), "{forced:?}");
    }
}
