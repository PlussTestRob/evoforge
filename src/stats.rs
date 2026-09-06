//! Per-generation statistics.
//!
//! The point of these numbers is to answer one question quickly: *is evolution
//! actually happening?* Best fitness alone does not answer it — with elitism the
//! best is monotonically non-decreasing whether or not anything is being learned.
//! Mean and median moving with the best is the signal that the population is
//! improving rather than one lucky individual being preserved. Structural
//! diversity collapsing is the signal that it is about to stop.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::evolution::Population;
use crate::math::Real;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GenerationStats {
    pub generation: u32,
    pub population: usize,
    pub best_fitness: Real,
    pub mean_fitness: Real,
    pub median_fitness: Real,
    pub worst_fitness: Real,
    pub best_id: u64,
    /// Distinct morphologies, ignoring controller weights.
    pub unique_structures: usize,
    pub mean_parts: Real,
    /// Organisms whose simulation diverged and were scored as failures.
    pub diverged: usize,
    /// Seconds spent evaluating this generation.
    pub eval_seconds: f64,
    pub organisms_per_second: f64,
    /// Seconds since the run started.
    pub elapsed_seconds: f64,
}

/// Summarise an evaluated population.
pub fn summarise(pop: &Population, eval_seconds: f64, elapsed_seconds: f64) -> GenerationStats {
    assert!(!pop.is_empty(), "cannot summarise an empty population");

    let mut fitnesses: Vec<Real> = pop.individuals.iter().map(|i| i.fitness).collect();
    fitnesses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = fitnesses.len();
    let sum: f64 = fitnesses.iter().map(|&f| f as f64).sum();
    let median = if n % 2 == 1 {
        fitnesses[n / 2]
    } else {
        0.5 * (fitnesses[n / 2 - 1] + fitnesses[n / 2])
    };

    let structures: HashSet<u64> = pop
        .individuals
        .iter()
        .map(|i| i.genome.structure_hash())
        .collect();
    let total_parts: usize = pop.individuals.iter().map(|i| i.genome.part_count()).sum();
    let best = pop.best();

    GenerationStats {
        generation: pop.generation,
        population: n,
        best_fitness: fitnesses[n - 1],
        mean_fitness: (sum / n as f64) as Real,
        median_fitness: median,
        worst_fitness: fitnesses[0],
        best_id: best.id,
        unique_structures: structures.len(),
        mean_parts: total_parts as Real / n as Real,
        diverged: pop.individuals.iter().filter(|i| i.metrics.diverged).count(),
        eval_seconds,
        organisms_per_second: if eval_seconds > 0.0 {
            n as f64 / eval_seconds
        } else {
            f64::INFINITY
        },
        elapsed_seconds,
    }
}

impl GenerationStats {
    pub const CSV_HEADER: &'static str = "generation,population,best,mean,median,worst,best_id,unique_structures,mean_parts,diverged,eval_seconds,organisms_per_second,elapsed_seconds";

    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{},{},{:.3},{},{:.4},{:.1},{:.2}",
            self.generation,
            self.population,
            self.best_fitness,
            self.mean_fitness,
            self.median_fitness,
            self.worst_fitness,
            self.best_id,
            self.unique_structures,
            self.mean_parts,
            self.diverged,
            self.eval_seconds,
            self.organisms_per_second,
            self.elapsed_seconds
        )
    }

    /// Column headings for the terminal table written by `evo run`.
    pub const TABLE_HEADER: &'static str =
        "  gen |    best     mean   median    worst | uniq  parts  div |   org/s   elapsed";

    pub fn to_table_row(&self) -> String {
        format!(
            "{:5} | {:7.3}  {:7.3}  {:7.3}  {:7.3} | {:4}  {:5.2}  {:3} | {:7.0}  {:7.1}s",
            self.generation,
            self.best_fitness,
            self.mean_fitness,
            self.median_fitness,
            self.worst_fitness,
            self.unique_structures,
            self.mean_parts,
            self.diverged,
            self.organisms_per_second,
            self.elapsed_seconds
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::evolution;

    fn evaluated_population() -> (Config, Population) {
        let mut cfg = Config::default();
        cfg.evolution.population_size = 9;
        cfg.simulation.duration = 0.5;
        cfg.simulation.settle_time = 0.1;
        let mut pop = Population::founding(&cfg);
        evolution::evaluate_population_serial(&mut pop, &cfg);
        (cfg, pop)
    }

    #[test]
    fn summary_orders_the_quartiles_correctly() {
        let (_, pop) = evaluated_population();
        let s = summarise(&pop, 0.5, 1.0);
        assert_eq!(s.population, 9);
        assert!(s.best_fitness >= s.median_fitness);
        assert!(s.median_fitness >= s.worst_fitness);
        assert!(s.mean_fitness <= s.best_fitness && s.mean_fitness >= s.worst_fitness);
        assert_eq!(s.best_fitness, pop.best().fitness);
        assert_eq!(s.best_id, pop.best().id);
    }

    #[test]
    fn median_handles_even_populations() {
        let mut cfg = Config::default();
        cfg.evolution.population_size = 4;
        cfg.simulation.duration = 0.5;
        cfg.simulation.settle_time = 0.1;
        let mut pop = Population::founding(&cfg);
        evolution::evaluate_population_serial(&mut pop, &cfg);
        let mut f: Vec<Real> = pop.individuals.iter().map(|i| i.fitness).collect();
        f.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let s = summarise(&pop, 1.0, 1.0);
        assert!((s.median_fitness - 0.5 * (f[1] + f[2])).abs() < 1e-6);
    }

    #[test]
    fn throughput_is_reported() {
        let (_, pop) = evaluated_population();
        let s = summarise(&pop, 2.0, 2.0);
        assert!((s.organisms_per_second - 4.5).abs() < 1e-9);
    }

    #[test]
    fn structure_count_is_bounded_by_population() {
        let (_, pop) = evaluated_population();
        let s = summarise(&pop, 1.0, 1.0);
        assert!(s.unique_structures >= 1 && s.unique_structures <= pop.len());
    }

    #[test]
    fn csv_row_has_the_same_arity_as_the_header() {
        let (_, pop) = evaluated_population();
        let s = summarise(&pop, 1.0, 1.0);
        assert_eq!(
            s.to_csv_row().split(',').count(),
            GenerationStats::CSV_HEADER.split(',').count()
        );
    }
}
