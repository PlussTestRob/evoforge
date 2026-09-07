//! `evo` — the EvoForge command line.
//!
//! Five verbs, no daemon, no state outside the run directory:
//!
//! ```text
//! evo run     experiments/first-walkers.toml   # evolve
//! evo bench   experiments/first-walkers.toml   # measure throughput
//! evo inspect runs/first-walkers-1700000000    # what happened
//! evo replay  runs/... --organism 1837         # re-simulate one organism
//! evo verify  experiments/first-walkers.toml   # prove determinism
//! ```

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};

use evoforge::bench;
use evoforge::config::Config;
use evoforge::evolution::{self, Population};
use evoforge::fitness::{self, Metrics};
use evoforge::math::Real;
use evoforge::record::{self, Run};
use evoforge::runner::{self, RunOptions};
use evoforge::sim;
use evoforge::stats;

#[derive(Parser)]
#[command(
    name = "evo",
    version,
    about = "EvoForge - a headless evolutionary artificial-life simulator"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a headless evolutionary experiment.
    Run(RunArgs),
    /// Measure evaluation throughput and its scaling across cores.
    Bench(BenchArgs),
    /// Summarise a completed or in-progress run.
    Inspect(InspectArgs),
    /// Re-simulate a recorded organism, optionally at higher fidelity.
    Replay(ReplayArgs),
    /// Check that an experiment reproduces exactly across thread counts.
    Verify(VerifyArgs),
    /// Re-score a finished run under different fitness weights, without
    /// re-simulating anything.
    Rescore(RescoreArgs),
}

#[derive(Args)]
struct RunArgs {
    /// Experiment configuration (TOML).
    config: PathBuf,
    /// Worker threads; 0 uses one per core.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Resume from the newest checkpoint in this run directory.
    #[arg(long)]
    resume: Option<PathBuf>,
    /// Override the output directory.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Override the experiment seed.
    #[arg(long)]
    seed: Option<u64>,
    /// Override the number of generations.
    #[arg(long)]
    generations: Option<u32>,
    /// Override the population size.
    #[arg(long)]
    population: Option<usize>,
    /// Override fitness climb_bonus: per metre ended above the settled start.
    #[arg(long)]
    climb_bonus: Option<Real>,
    /// Override fitness descent_penalty: per metre ended below it.
    #[arg(long)]
    descent_penalty: Option<Real>,
    /// Suppress the per-generation table.
    #[arg(long)]
    quiet: bool,
    /// Resume a checkpoint written by a different evoforge version.
    #[arg(long)]
    force_resume: bool,
}

#[derive(Args)]
struct BenchArgs {
    /// Experiment configuration to benchmark. Defaults to built-in defaults.
    config: Option<PathBuf>,
    /// Comma-separated thread counts, e.g. `1,2,4,8`. Defaults to a sweep up to
    /// the core count.
    #[arg(long, value_delimiter = ',')]
    threads: Option<Vec<usize>>,
    /// Measurements per thread count; the fastest is reported.
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// Override the population size used for the measurement.
    #[arg(long)]
    population: Option<usize>,
    /// Price assumption for the cost model, USD per core-hour.
    #[arg(long, default_value_t = 0.01)]
    price: f64,
}

#[derive(Args)]
struct InspectArgs {
    /// Run directory.
    run: PathBuf,
    /// How many of the most recent generations to show.
    #[arg(long, default_value_t = 10)]
    tail: usize,
}

#[derive(Args)]
struct RescoreArgs {
    /// Run directory.
    run: PathBuf,
    /// How many of the most recent generations to show.
    #[arg(long, default_value_t = 10)]
    tail: usize,
    /// Per metre ended above the settled start.
    #[arg(long)]
    climb_bonus: Option<Real>,
    /// Per metre ended below it.
    #[arg(long)]
    descent_penalty: Option<Real>,
    /// Per metre of total (hysteresis-filtered) ascent.
    #[arg(long)]
    cumulative_climb_bonus: Option<Real>,
    /// Per metre of total descent.
    #[arg(long)]
    cumulative_descent_penalty: Option<Real>,
    /// Per second spent upright.
    #[arg(long)]
    upright_bonus: Option<Real>,
    /// Per unit of actuation impulse.
    #[arg(long)]
    energy_penalty: Option<Real>,
    /// Show the organisms each scoring promotes, for this generation.
    #[arg(long)]
    show_generation: Option<u32>,
}

#[derive(Args)]
struct ReplayArgs {
    /// Run directory.
    run: PathBuf,
    /// Organism id to replay.
    #[arg(long)]
    organism: Option<u64>,
    /// Replay the best organism whose genome was stored.
    #[arg(long)]
    best: bool,
    /// Recording rate for the regenerated trajectory.
    #[arg(long)]
    hz: Option<f32>,
    /// Simulate for this many seconds instead of the experiment's duration.
    #[arg(long)]
    duration: Option<f32>,
    /// Where to write the replay. Defaults to the run's `replays/` directory.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct VerifyArgs {
    /// Experiment configuration (TOML).
    config: PathBuf,
    /// Generations to compare.
    #[arg(long, default_value_t = 3)]
    generations: u32,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run(args) => cmd_run(args),
        Command::Bench(args) => cmd_bench(args),
        Command::Inspect(args) => cmd_inspect(args),
        Command::Replay(args) => cmd_replay(args),
        Command::Verify(args) => cmd_verify(args),
        Command::Rescore(args) => cmd_rescore(args),
    }
}

fn cmd_run(args: RunArgs) -> Result<()> {
    let mut cfg = Config::load(&args.config)?;
    if let Some(out) = args.out {
        cfg.experiment.output_dir = out;
    }
    if let Some(seed) = args.seed {
        cfg.experiment.seed = seed;
    }
    if let Some(g) = args.generations {
        cfg.evolution.generations = g;
    }
    if let Some(p) = args.population {
        cfg.evolution.population_size = p;
    }
    // Overrides go through the config before the digest is taken, exactly as the
    // ones above do, so the run directory still records what actually produced it
    // rather than the file it started from.
    if let Some(v) = args.climb_bonus {
        cfg.fitness.climb_bonus = v;
    }
    if let Some(v) = args.descent_penalty {
        cfg.fitness.descent_penalty = v;
    }
    cfg.validate()?;

    let summary = runner::run(
        &cfg,
        &RunOptions {
            threads: args.threads,
            quiet: args.quiet,
            resume: args.resume,
            force_resume: args.force_resume,
        },
    )?;

    println!();
    println!(
        "{} generations, {} organisms evaluated in {:.1}s ({:.0} organisms/s overall)",
        summary.generations_completed,
        summary.organisms_evaluated,
        summary.wall_seconds,
        summary.organisms_evaluated as f64 / summary.wall_seconds.max(1e-9),
    );
    if let Some(s) = &summary.final_stats {
        println!(
            "final generation {}: best {:.3} (organism {}), mean {:.3}, {} distinct structures",
            s.generation, s.best_fitness, s.best_id, s.mean_fitness, s.unique_structures
        );
    }
    println!("results in {}", summary.dir.display());
    Ok(())
}

fn cmd_bench(args: BenchArgs) -> Result<()> {
    let mut cfg = match &args.config {
        Some(path) => Config::load(path)?,
        None => Config::default(),
    };
    if let Some(p) = args.population {
        cfg.evolution.population_size = p;
    }
    cfg.validate()?;

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let thread_counts = args.threads.unwrap_or_else(|| bench::default_thread_counts(cores));

    println!(
        "benchmarking: population {}, {:.1}s simulated per organism at {}Hz ({} steps), \
         control {}Hz, {} solver iterations",
        cfg.evolution.population_size,
        cfg.simulation.duration + cfg.simulation.settle_time,
        (1.0 / cfg.simulation.timestep).round(),
        cfg.total_steps(),
        cfg.simulation.control_hz,
        cfg.simulation.solver_iterations,
    );
    println!(
        "machine reports {cores} cores; best of {} runs per thread count",
        args.repeats.max(1)
    );
    println!();

    let report = bench::run(&cfg, &thread_counts, args.repeats)?;

    println!("mean body: {:.2} parts, {:.2} joints", report.mean_parts, report.mean_joints);
    println!(
        "memory: ~{} B per genome, ~{:.1} KiB for the live population{}",
        report.genome_bytes,
        report.population_bytes as f64 / 1024.0,
        match report.resident_bytes {
            Some(rss) => format!(", {:.1} MiB resident", rss as f64 / (1024.0 * 1024.0)),
            None => String::new(),
        }
    );
    println!();
    println!(
        "threads |   org/s |    steps/s | speedup | efficiency | USD/M evals | gens/USD @ ${:.4}/core-h",
        args.price
    );
    for r in &report.results {
        println!(
            "{:7} | {:7.1} | {:10.3e} | {:6.2}x | {:9.0}% | {:11.4} | {:8.0}",
            r.threads,
            r.organisms_per_second,
            r.steps_per_second,
            r.speedup,
            r.efficiency * 100.0,
            report.usd_per_million_evaluations(args.price, r),
            report.generations_per_usd(args.price, r),
        );
    }

    if let Some(best) = report.best() {
        println!();
        println!(
            "best: {:.0} organisms/s at {} threads ({:.0}% scaling efficiency)",
            best.organisms_per_second,
            best.threads,
            best.efficiency * 100.0
        );
        println!(
            "at ${:.4}/core-hour that is ${:.4} per million evaluations, \
             {:.0} generations of {} per dollar",
            args.price,
            report.usd_per_million_evaluations(args.price, best),
            report.generations_per_usd(args.price, best),
            report.population,
        );
    }
    Ok(())
}

/// One organism as `organisms.jsonl` records it.
#[derive(serde::Deserialize)]
struct OrganismRecord {
    id: u64,
    generation: u32,
    fitness: Real,
    metrics: Metrics,
}

/// Re-score a finished run under different fitness weights.
///
/// This works at all because `fitness::score` is a pure function of
/// `(FitnessCfg, &Metrics)` and full metrics are written for every organism, so
/// asking "what would this population have looked like under a different
/// question" costs a file read rather than a re-simulation. With no weights
/// overridden it is a round trip, and must reproduce the recorded fitness
/// exactly — which is what makes the rest of its output trustworthy.
fn cmd_rescore(args: RescoreArgs) -> Result<()> {
    let run =
        Run::open(&args.run).with_context(|| format!("opening run {}", args.run.display()))?;
    let cfg = run.config()?;

    let mut new = cfg.fitness.clone();
    let mut changed = Vec::new();
    let mut set = |name: &str, slot: &mut Real, v: Option<Real>| {
        if let Some(v) = v {
            if *slot != v {
                changed.push(format!("{name} {slot} -> {v}"));
            }
            *slot = v;
        }
    };
    set("climb_bonus", &mut new.climb_bonus, args.climb_bonus);
    set("descent_penalty", &mut new.descent_penalty, args.descent_penalty);
    set("cumulative_climb_bonus", &mut new.cumulative_climb_bonus, args.cumulative_climb_bonus);
    set(
        "cumulative_descent_penalty",
        &mut new.cumulative_descent_penalty,
        args.cumulative_descent_penalty,
    );
    set("upright_bonus", &mut new.upright_bonus, args.upright_bonus);
    set("energy_penalty", &mut new.energy_penalty, args.energy_penalty);

    let path = args.run.join(record::ORGANISMS_FILE);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    // A kill mid-append leaves a truncated final line, as everywhere else that
    // reads a `.jsonl` here.
    let mut records: Vec<OrganismRecord> =
        text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    if records.is_empty() {
        bail!("{} holds no readable organism records", path.display());
    }

    // Runs that predate the elevation metrics stored `start` and `end` but not
    // the derived pair. Net elevation is recoverable from them, so those runs can
    // still be re-scored on it — approximately, because the stored endpoints are
    // already averaged across trials and the clamp is not linear. Anything with
    // one trial, or that never crosses zero, is exact.
    let mut backfilled = 0usize;
    for r in &mut records {
        if r.metrics.net_gain == 0.0 && r.metrics.net_loss == 0.0 {
            let net = r.metrics.end.y - r.metrics.start.y;
            if net.abs() > 1e-6 {
                r.metrics.net_gain = net.max(0.0);
                r.metrics.net_loss = (-net).max(0.0);
                backfilled += 1;
            }
        }
    }

    // The round trip. Under the run's own weights, re-scoring has to give back
    // exactly what the simulator wrote, or nothing else printed here means
    // anything.
    let mut worst_drift = 0.0f32;
    for r in &records {
        let drift = (fitness::score(&cfg.fitness, &r.metrics) - r.fitness).abs();
        worst_drift = worst_drift.max(drift);
    }

    println!("run           {}", run.manifest.experiment_id);
    println!("objective     {:?}", cfg.fitness.objective);
    println!("organisms     {}", records.len());
    if backfilled > 0 {
        println!(
            "backfilled    {backfilled} records' net elevation from stored start/end \
             (pre-dates the metric; averaged across trials, so approximate)"
        );
    }
    println!(
        "round trip    worst |rescored - recorded| = {worst_drift:.3e}{}",
        if worst_drift > 1e-3 { "   *** MISMATCH ***" } else { "" }
    );
    if changed.is_empty() {
        println!("\nno weights overridden; nothing to compare");
        return Ok(());
    }
    println!("changed       {}", changed.join(", "));
    println!();

    let mut generations: Vec<u32> = records.iter().map(|r| r.generation).collect();
    generations.sort_unstable();
    generations.dedup();

    println!("  gen |   best now   best then |  median now  median then | top-10 kept");
    for &g in generations.iter().skip(generations.len().saturating_sub(args.tail)) {
        let mut gen: Vec<&OrganismRecord> = records.iter().filter(|r| r.generation == g).collect();
        if gen.is_empty() {
            continue;
        }
        let rescored: Vec<Real> = gen.iter().map(|r| fitness::score(&new, &r.metrics)).collect();

        let mut old_sorted: Vec<Real> = gen.iter().map(|r| r.fitness).collect();
        let mut new_sorted = rescored.clone();
        old_sorted.sort_by(|a, b| b.total_cmp(a));
        new_sorted.sort_by(|a, b| b.total_cmp(a));
        let median = |v: &[Real]| v[v.len() / 2];

        // How much the ranking actually moved: of the ten best under the new
        // weights, how many were in the ten best under the old. A scoring that
        // reorders nothing is not asking a new question.
        let top = 10.min(gen.len());
        let mut by_new: Vec<usize> = (0..gen.len()).collect();
        by_new.sort_by(|&a, &b| rescored[b].total_cmp(&rescored[a]));
        let mut by_old: Vec<usize> = (0..gen.len()).collect();
        by_old.sort_by(|&a, &b| gen[b].fitness.total_cmp(&gen[a].fitness));
        let old_top: std::collections::HashSet<u64> =
            by_old[..top].iter().map(|&i| gen[i].id).collect();
        let kept = by_new[..top].iter().filter(|&&i| old_top.contains(&gen[i].id)).count();

        println!(
            "  {g:>4} | {:10.3} {:11.3} | {:11.3} {:12.3} | {kept:>2}/{top}",
            new_sorted[0],
            old_sorted[0],
            median(&new_sorted),
            median(&old_sorted),
        );

        if args.show_generation == Some(g) {
            gen.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            println!("\n    generation {g}: what each scoring promotes");
            println!("      rank | id     | then   | now    | travel | net dy | climb  descent");
            let mut ranked: Vec<(usize, &&OrganismRecord)> = gen.iter().enumerate().collect();
            ranked.sort_by(|a, b| {
                fitness::score(&new, &b.1.metrics).total_cmp(&fitness::score(&new, &a.1.metrics))
            });
            for (rank, (old_rank, r)) in ranked.iter().take(10).enumerate() {
                let m = &r.metrics;
                println!(
                    "      {:>4} | {:6} | {:6.2} | {:6.2} | {:6.2} | {:+6.3} | {:5.2}  {:5.2}   (was #{})",
                    rank + 1,
                    r.id,
                    r.fitness,
                    fitness::score(&new, m),
                    m.displacement,
                    m.end.y - m.start.y,
                    m.climb,
                    m.descent,
                    old_rank + 1,
                );
            }
            println!();
        }
    }
    Ok(())
}

fn cmd_inspect(args: InspectArgs) -> Result<()> {
    let run =
        Run::open(&args.run).with_context(|| format!("opening run {}", args.run.display()))?;
    let cfg = run.config()?;

    println!("experiment    {}", run.manifest.experiment_id);
    println!("name          {}", run.manifest.experiment_name);
    println!("evoforge      {}", run.manifest.evoforge_version);
    println!("seed          {}", run.manifest.seed);
    println!("config digest {:016x}", run.manifest.config_digest);
    println!(
        "population    {}   generations configured  {}",
        cfg.evolution.population_size, cfg.evolution.generations
    );
    println!("objective     {:?}", cfg.fitness.objective);
    println!();

    let stats_path = args.run.join(record::STATS_FILE);
    let text = std::fs::read_to_string(&stats_path)
        .with_context(|| format!("reading {}", stats_path.display()))?;
    let rows: Vec<&str> = text.lines().skip(1).filter(|l| !l.trim().is_empty()).collect();
    if rows.is_empty() {
        println!("no generations recorded yet");
        return Ok(());
    }

    println!("{} generations recorded", rows.len());
    println!("{}", stats::GenerationStats::TABLE_HEADER);
    for row in rows.iter().skip(rows.len().saturating_sub(args.tail)) {
        let f: Vec<&str> = row.split(',').collect();
        if f.len() < stats::GenerationStats::CSV_COLUMNS {
            continue;
        }
        println!(
            "{:>5} | {:>7} {:>8} {:>8} {:>8} | {:>4} {:>6} {:>4} | {:>7} {:>8}s",
            f[0],
            trim(f[2]),
            trim(f[3]),
            trim(f[4]),
            trim(f[5]),
            f[7],
            trim(f[8]),
            f[9],
            trim(f[11]),
            trim(f[12])
        );
    }

    let checkpoints = run.checkpoint_paths()?;
    println!();
    println!(
        "{} checkpoint(s), newest {}",
        checkpoints.len(),
        checkpoints.first().map(|p| p.display().to_string()).unwrap_or_else(|| "none".into())
    );

    if let Some(b) = run.best_stored_genome()? {
        println!(
            "best stored genome: organism {} from generation {}, fitness {:.3}, {} parts",
            b.id,
            b.generation,
            b.fitness,
            b.genome.part_count()
        );
        println!("  evo replay {} --organism {}", args.run.display(), b.id);
    }
    Ok(())
}

fn trim(s: &str) -> String {
    match s.parse::<f64>() {
        Ok(v) => format!("{v:.3}"),
        Err(_) => s.to_string(),
    }
}

fn cmd_replay(args: ReplayArgs) -> Result<()> {
    let run =
        Run::open(&args.run).with_context(|| format!("opening run {}", args.run.display()))?;
    let mut cfg = run.config()?;

    let stored = match (args.organism, args.best) {
        (Some(id), _) => {
            run.find_genome(id)?.with_context(|| format!("no stored genome for organism {id}"))?
        }
        (None, true) => run.best_stored_genome()?.context("no genomes were stored for this run")?,
        (None, false) => bail!("specify --organism <id> or --best"),
    };

    // The run's own dynamics, before any override. `--hz` only changes how often
    // the trajectory is sampled and leaves this untouched, which is the whole
    // point of the command; `--duration` changes the simulation itself.
    let run_digest = cfg.evolution_digest();
    if let Some(hz) = args.hz {
        cfg.recording.record_hz = hz;
    }
    if let Some(d) = args.duration {
        cfg.simulation.duration = d;
    }
    cfg.validate()?;
    let dynamics_changed = cfg.evolution_digest() != run_digest;

    println!(
        "organism {} from generation {} (parents {:?}), recorded fitness {:.4}",
        stored.id, stored.generation, stored.parents, stored.fitness
    );
    describe_genome(&stored.genome);

    let result = sim::evaluate(&stored.genome, &cfg, true);
    println!();
    println!("re-simulated fitness {:.4}", result.fitness);
    if (result.fitness - stored.fitness).abs() > 1e-3 && args.duration.is_none() {
        println!(
            "  note: differs from the recorded fitness by {:.4}; the run's config may \
             have changed since",
            result.fitness - stored.fitness
        );
    }
    let m = &result.metrics;
    println!(
        "  displacement {:.3} m ({:.3} along x), path {:.3} m, mean speed {:.3} m/s",
        m.displacement,
        m.displacement_x,
        m.path_length,
        m.mean_speed()
    );
    println!(
        "  upright {:.2}s of {:.2}s, mean height {:.3} m, actuation {:.1}",
        m.upright_seconds, m.duration, m.mean_height, m.actuation
    );

    let Some(trace) = result.trace else {
        bail!("the organism diverged when re-simulated; nothing to record");
    };
    let frames = trace.frames.len();

    // The re-simulated organism, carrying the fitness it just earned rather than
    // the one it was recorded with.
    let individual = evolution::Individual {
        id: stored.id,
        generation: stored.generation,
        parents: stored.parents,
        genome: stored.genome.clone(),
        fitness: result.fitness,
        metrics: result.metrics,
    };

    let path = match args.out {
        Some(p) => {
            run.write_replay_to(&p, &individual, trace, &cfg)?;
            p
        }
        // Re-simulating under different dynamics must not overwrite the trajectory
        // the run itself recorded: that file is the run's own evidence, and this
        // one was produced by a different experiment.
        None if dynamics_changed => {
            let p = run.variant_replay_path(
                individual.generation,
                individual.id,
                cfg.evolution_digest(),
            );
            run.write_replay_to(&p, &individual, trace, &cfg)?;
            println!();
            println!(
                "note: these dynamics differ from the run's own, so the run's \
                 recording of organism {} was left untouched",
                individual.id
            );
            p
        }
        None => run.write_replay(&individual, trace, &cfg)?,
    };

    println!();
    println!("wrote {frames} frames at {}Hz to {}", cfg.recording.record_hz, path.display());
    Ok(())
}

fn describe_genome(g: &evoforge::genome::Genome) {
    println!(
        "  {} parts, {} joints ({} hinges), {} controller weights",
        g.part_count(),
        g.joint_count(),
        g.hinge_count(),
        g.weights.len()
    );
    for (i, p) in g.parts.iter().enumerate() {
        if i == 0 {
            println!(
                "    part 0 (slot {}): root, {:?} {:.2} x {:.2} x {:.2} m",
                p.slot,
                p.shape,
                p.half_extents.x * 2.0,
                p.half_extents.y * 2.0,
                p.half_extents.z * 2.0
            );
        } else {
            println!(
                "    part {} (slot {}): {:?} {:.2} x {:.2} x {:.2} m, {:?} on face {} of part {}, \
                 limit {:.2} rad, motor {:.1} rad/s / {:.0} N m",
                i,
                p.slot,
                p.shape,
                p.half_extents.x * 2.0,
                p.half_extents.y * 2.0,
                p.half_extents.z * 2.0,
                p.joint.kind,
                p.attach_face,
                p.parent,
                p.joint.limit,
                p.joint.motor_speed,
                p.joint.motor_torque
            );
        }
    }
}

/// Prove the reproducibility claim rather than asserting it: evolve the same
/// experiment single-threaded and multi-threaded and compare every fitness.
fn cmd_verify(args: VerifyArgs) -> Result<()> {
    let mut cfg = Config::load(&args.config)?;
    cfg.evolution.generations = args.generations;
    cfg.validate()?;

    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!(
        "verifying {} generations of {} organisms: 1 thread vs {cores}",
        args.generations, cfg.evolution.population_size
    );

    let trace = |threads: usize| -> Result<Vec<(u64, [u64; 2], u32)>> {
        let pool = runner::build_pool(threads)?;
        let mut pop = Population::founding(&cfg);
        let mut out = Vec::new();
        for _ in 0..cfg.evolution.generations {
            evolution::evaluate_population(&mut pop, &cfg, &pool);
            for i in &pop.individuals {
                // Compare the exact bit patterns; "close enough" would hide
                // precisely the kind of drift this command exists to detect.
                out.push((i.id, i.parents, i.fitness.to_bits()));
            }
            pop = evolution::next_generation(&pop, &cfg);
        }
        Ok(out)
    };

    let single = trace(1)?;
    let multi = trace(cores)?;

    if single == multi {
        println!("identical: {} organism results matched exactly", single.len());
        Ok(())
    } else {
        let first = single
            .iter()
            .zip(&multi)
            .position(|(a, b)| a != b)
            .unwrap_or(single.len().min(multi.len()));
        bail!(
            "results diverged at result {} of {} (organism {:?} vs {:?})",
            first,
            single.len(),
            single.get(first),
            multi.get(first)
        );
    }
}
