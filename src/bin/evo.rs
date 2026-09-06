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
    /// Suppress the per-generation table.
    #[arg(long)]
    quiet: bool,
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
    cfg.validate()?;

    let summary = runner::run(
        &cfg,
        &RunOptions { threads: args.threads, quiet: args.quiet, resume: args.resume },
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
    println!("machine reports {cores} cores; best of {} runs per thread count", args.repeats.max(1));
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

fn cmd_inspect(args: InspectArgs) -> Result<()> {
    let run = Run::open(&args.run)
        .with_context(|| format!("opening run {}", args.run.display()))?;
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
        if f.len() < 13 {
            continue;
        }
        println!(
            "{:>5} | {:>7} {:>8} {:>8} {:>8} | {:>4} {:>6} {:>4} | {:>7} {:>8}s",
            f[0], trim(f[2]), trim(f[3]), trim(f[4]), trim(f[5]), f[7], trim(f[8]), f[9],
            trim(f[11]), trim(f[12])
        );
    }

    let checkpoints = run.checkpoint_paths()?;
    println!();
    println!("{} checkpoint(s), newest {}",
        checkpoints.len(),
        checkpoints.first().map(|p| p.display().to_string()).unwrap_or_else(|| "none".into()));

    let genomes_path = args.run.join(record::GENOMES_FILE);
    if genomes_path.exists() {
        let stored = std::fs::read_to_string(&genomes_path)?;
        let mut best: Option<record::StoredGenome> = None;
        for line in stored.lines().filter(|l| !l.trim().is_empty()) {
            let g: record::StoredGenome = serde_json::from_str(line)?;
            if best.as_ref().is_none_or(|b| g.fitness > b.fitness) {
                best = Some(g);
            }
        }
        if let Some(b) = best {
            println!(
                "best stored genome: organism {} from generation {}, fitness {:.3}, {} parts",
                b.id,
                b.generation,
                b.fitness,
                b.genome.part_count()
            );
            println!("  evo replay {} --organism {}", args.run.display(), b.id);
        }
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
    let run = Run::open(&args.run)
        .with_context(|| format!("opening run {}", args.run.display()))?;
    let mut cfg = run.config()?;

    let stored = match (args.organism, args.best) {
        (Some(id), _) => run
            .find_genome(id)?
            .with_context(|| format!("no stored genome for organism {id}"))?,
        (None, true) => best_stored_genome(&args.run)?,
        (None, false) => bail!("specify --organism <id> or --best"),
    };

    if let Some(hz) = args.hz {
        cfg.recording.record_hz = hz;
    }
    if let Some(d) = args.duration {
        cfg.simulation.duration = d;
    }
    cfg.validate()?;

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

    let path = match args.out {
        Some(p) => {
            let replay = record::Replay {
                experiment_id: run.manifest.experiment_id.clone(),
                organism_id: stored.id,
                generation: stored.generation,
                parents: stored.parents,
                fitness: result.fitness,
                metrics: result.metrics,
                config_digest: cfg.evolution_digest(),
                timestep: cfg.simulation.timestep,
                genome: stored.genome.clone(),
                trace,
            };
            record::write_json(&p, &replay)?;
            p
        }
        None => {
            let individual = evolution::Individual {
                id: stored.id,
                generation: stored.generation,
                parents: stored.parents,
                genome: stored.genome.clone(),
                fitness: result.fitness,
                metrics: result.metrics,
            };
            run.write_replay(&individual, trace, &cfg)?
        }
    };

    println!();
    println!("wrote {frames} frames at {}Hz to {}", cfg.recording.record_hz, path.display());
    Ok(())
}

fn best_stored_genome(dir: &std::path::Path) -> Result<record::StoredGenome> {
    let path = dir.join(record::GENOMES_FILE);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut best: Option<record::StoredGenome> = None;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let g: record::StoredGenome = serde_json::from_str(line)?;
        if best.as_ref().is_none_or(|b| g.fitness > b.fitness) {
            best = Some(g);
        }
    }
    best.context("no genomes were stored for this run")
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
                "    part 0 (slot {}): root, {:.2} x {:.2} x {:.2} m",
                p.slot,
                p.half_extents.x * 2.0,
                p.half_extents.y * 2.0,
                p.half_extents.z * 2.0
            );
        } else {
            println!(
                "    part {} (slot {}): {:.2} x {:.2} x {:.2} m, {:?} on face {} of part {}, \
                 limit {:.2} rad, motor {:.1} rad/s / {:.0} N m",
                i,
                p.slot,
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
