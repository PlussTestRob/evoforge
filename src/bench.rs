//! Throughput measurement.
//!
//! The metric that actually matters for this project is not simulation steps per
//! second — it is **organisms evaluated per second per dollar**. That is what
//! decides how much evolution a fixed budget buys. So the benchmark measures
//! whole-population evaluation at several thread counts, reports scaling
//! efficiency (the thing that determines whether a bigger instance is worth
//! paying for), and converts the result into a cost per million evaluations.
//!
//! It is built in from the start rather than added later because the alternative
//! is optimising against intuition.

use std::time::Instant;

use crate::config::Config;
use crate::evolution::{self, Population};
use crate::genome::PartGene;
use crate::math::Real;
use crate::runner;

/// Result at one thread count.
#[derive(Clone, Debug)]
pub struct ThreadResult {
    pub threads: usize,
    pub seconds: f64,
    pub organisms_per_second: f64,
    pub steps_per_second: f64,
    /// Throughput relative to the first measurement in the sweep.
    pub speedup: f64,
    /// Speedup divided by the *core ratio* over the baseline measurement. This is
    /// the number that decides whether a larger instance is worth its price.
    ///
    /// Normalised against the baseline's thread count rather than against
    /// `threads` outright, so a sweep that does not start at one thread (`--threads
    /// 4,8`) still reports meaningful scaling instead of charging the baseline for
    /// cores it was never compared against.
    pub efficiency: f64,
}

#[derive(Clone, Debug)]
pub struct BenchReport {
    pub population: usize,
    pub steps_per_organism: u32,
    pub simulated_seconds_per_organism: Real,
    pub mean_parts: Real,
    pub mean_joints: Real,
    /// Approximate heap footprint of one genome.
    pub genome_bytes: usize,
    /// Approximate heap footprint of the live population.
    pub population_bytes: usize,
    /// Resident set size, where the platform makes it cheaply available.
    pub resident_bytes: Option<u64>,
    pub results: Vec<ThreadResult>,
}

impl BenchReport {
    pub fn best(&self) -> Option<&ThreadResult> {
        self.results.iter().max_by(|a, b| a.organisms_per_second.total_cmp(&b.organisms_per_second))
    }

    /// USD per million organism evaluations at the given price per core-hour.
    ///
    /// Uses the *measured* throughput at each thread count, so a configuration
    /// that scales badly is correctly penalised: paying for cores that deliver
    /// 60% efficiency costs 1.7x more per evaluation than the core count
    /// suggests.
    pub fn usd_per_million_evaluations(&self, usd_per_core_hour: f64, r: &ThreadResult) -> f64 {
        let core_hours = r.threads as f64 * (1.0e6 / r.organisms_per_second) / 3600.0;
        core_hours * usd_per_core_hour
    }

    /// Generations per dollar at the given price, for this population size.
    pub fn generations_per_usd(&self, usd_per_core_hour: f64, r: &ThreadResult) -> f64 {
        let seconds_per_generation = self.population as f64 / r.organisms_per_second;
        let usd_per_generation =
            r.threads as f64 * usd_per_core_hour * seconds_per_generation / 3600.0;
        if usd_per_generation > 0.0 {
            1.0 / usd_per_generation
        } else {
            f64::INFINITY
        }
    }
}

/// Measure evaluation throughput at each of `thread_counts`.
///
/// Each measurement is the best of `repeats` runs: the fastest observation is
/// the one least contaminated by other activity on the machine.
pub fn run(cfg: &Config, thread_counts: &[usize], repeats: usize) -> anyhow::Result<BenchReport> {
    let reference = Population::founding(cfg);
    let steps_per_organism = cfg.total_steps();

    // Warm up: the first evaluation pays for page faults and branch predictor
    // training that would otherwise be charged to whichever measurement ran first.
    {
        let mut warm = reference.clone();
        warm.individuals.truncate(cfg.evolution.population_size.min(16));
        evolution::evaluate_population_serial(&mut warm, cfg);
    }

    let mut results: Vec<ThreadResult> = Vec::new();
    let mut baseline_rate = 0.0;
    let mut baseline_threads = 1usize;

    for (i, &threads) in thread_counts.iter().enumerate() {
        let pool = runner::build_pool(threads)?;
        let actual_threads = pool.current_num_threads();
        let mut best_seconds = f64::INFINITY;

        for _ in 0..repeats.max(1) {
            let mut pop = reference.clone();
            let started = Instant::now();
            evolution::evaluate_population(&mut pop, cfg, &pool);
            best_seconds = best_seconds.min(started.elapsed().as_secs_f64());
        }

        let organisms_per_second = reference.len() as f64 / best_seconds;
        if i == 0 {
            baseline_rate = organisms_per_second;
            baseline_threads = actual_threads.max(1);
        }
        let speedup = organisms_per_second / baseline_rate;
        let core_ratio = actual_threads as f64 / baseline_threads as f64;
        results.push(ThreadResult {
            threads: actual_threads,
            seconds: best_seconds,
            organisms_per_second,
            steps_per_second: organisms_per_second * steps_per_organism as f64,
            speedup,
            efficiency: speedup / core_ratio,
        });
    }

    let total_parts: usize = reference.individuals.iter().map(|i| i.genome.part_count()).sum();
    let total_joints: usize = reference.individuals.iter().map(|i| i.genome.joint_count()).sum();
    let n = reference.len();
    let genome_bytes = genome_footprint(&reference);

    Ok(BenchReport {
        population: n,
        steps_per_organism,
        simulated_seconds_per_organism: cfg.simulation.duration + cfg.simulation.settle_time,
        mean_parts: total_parts as Real / n as Real,
        mean_joints: total_joints as Real / n as Real,
        genome_bytes: genome_bytes / n,
        population_bytes: genome_bytes,
        resident_bytes: resident_bytes(),
        results,
    })
}

fn genome_footprint(pop: &Population) -> usize {
    pop.individuals
        .iter()
        .map(|i| {
            std::mem::size_of::<crate::evolution::Individual>()
                + i.genome.parts.len() * std::mem::size_of::<PartGene>()
                + i.genome.weights.len() * std::mem::size_of::<Real>()
        })
        .sum()
}

/// Resident set size in bytes.
///
/// Linux only, read straight from `/proc`. Everywhere else this returns `None`
/// rather than pulling in a dependency for a number that is only advisory —
/// the deployment target for long runs is Linux.
///
/// Reads `VmRSS` from `status` rather than the page count from `statm`, because
/// the latter would need the kernel page size and assuming 4 KiB is wrong by 16x
/// on a 64 KiB-page aarch64 host.
fn resident_bytes() -> Option<u64> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    let rest = text.lines().find_map(|l| l.strip_prefix("VmRSS:"))?;
    let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
    Some(kib * 1024)
}

/// Thread counts to sweep on a machine with `cores` cores: 1, 2, 4, ... capped
/// at the core count, always including the core count itself.
pub fn default_thread_counts(cores: usize) -> Vec<usize> {
    let mut counts = vec![1];
    let mut t = 2;
    while t < cores {
        counts.push(t);
        t *= 2;
    }
    if cores > 1 {
        counts.push(cores);
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bench_config() -> Config {
        let mut cfg = Config::default();
        cfg.evolution.population_size = 12;
        cfg.simulation.duration = 0.3;
        cfg.simulation.settle_time = 0.1;
        cfg
    }

    #[test]
    fn default_thread_counts_cover_the_machine() {
        assert_eq!(default_thread_counts(1), vec![1]);
        assert_eq!(default_thread_counts(2), vec![1, 2]);
        assert_eq!(default_thread_counts(8), vec![1, 2, 4, 8]);
        assert_eq!(default_thread_counts(6), vec![1, 2, 4, 6]);
    }

    #[test]
    fn benchmark_reports_plausible_numbers() {
        let cfg = bench_config();
        let report = run(&cfg, &[1, 2], 1).unwrap();

        assert_eq!(report.population, 12);
        assert_eq!(report.steps_per_organism, cfg.total_steps());
        assert!(report.mean_parts >= cfg.body.min_parts as Real);
        assert!(report.genome_bytes > 0);

        for r in &report.results {
            assert!(r.organisms_per_second > 0.0);
            assert!(r.steps_per_second > r.organisms_per_second);
            assert!(r.efficiency > 0.0);
        }
        // The single-threaded case is the baseline by definition.
        assert!((report.results[0].speedup - 1.0).abs() < 1e-9);
        assert!(report.best().is_some());
    }

    /// A sweep that does not start at one thread must still report honest
    /// scaling. Taking the first measurement as the unit baseline regardless of
    /// its thread count made `--threads 4,8` claim 25% efficiency for the
    /// baseline itself.
    #[test]
    fn efficiency_is_relative_to_the_baseline_thread_count() {
        let cfg = bench_config();
        let report = run(&cfg, &[2, 2], 1).unwrap();

        let baseline = &report.results[0];
        assert_eq!(baseline.threads, 2);
        assert!(
            (baseline.speedup - 1.0).abs() < 1e-9,
            "the first measurement is the baseline by definition"
        );
        assert!(
            (baseline.efficiency - 1.0).abs() < 1e-9,
            "a baseline cannot be inefficient relative to itself, got {}",
            baseline.efficiency
        );
        // Same thread count twice: the second measurement carries no core ratio,
        // so its efficiency tracks its speedup exactly.
        let second = &report.results[1];
        assert!((second.efficiency - second.speedup).abs() < 1e-9);
    }

    #[test]
    fn cost_model_is_internally_consistent() {
        let cfg = bench_config();
        let report = run(&cfg, &[1], 1).unwrap();
        let r = &report.results[0];

        // Twice the price is twice the cost.
        let a = report.usd_per_million_evaluations(0.01, r);
        let b = report.usd_per_million_evaluations(0.02, r);
        assert!((b / a - 2.0).abs() < 1e-9);

        // Generations per dollar must agree with cost per evaluation.
        let per_million = report.usd_per_million_evaluations(0.01, r);
        let generations = report.generations_per_usd(0.01, r);
        let expected = 1.0e6 / (report.population as f64 * per_million);
        assert!((generations - expected).abs() / expected < 1e-6);
    }
}
