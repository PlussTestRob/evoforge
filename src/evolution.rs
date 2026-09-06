//! The evolutionary loop: populations, selection, reproduction, lineage.
//!
//! # Identity
//!
//! Every organism ever created in an experiment gets a unique, monotonically
//! increasing `id`, and records the ids of its parents. Together with the
//! experiment seed and the config digest, an id is enough to find, re-simulate
//! and replay an organism — which is the point of "show me organism 1837 from
//! generation 8421".
//!
//! # Determinism
//!
//! Reproduction uses a single sequential stream derived from
//! `(experiment_seed, generation)`, so the next generation is a pure function of
//! the current one. Nothing depends on thread scheduling: evaluation is pure and
//! its results are collected in index order.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::fitness::Metrics;
use crate::genome::{self, Genome};
use crate::math::Real;
use crate::rng::{derive_seed, Rng};
use crate::sim::{self, EvalWorkspace};

/// Domain separator so the reproduction stream can never collide with an
/// evaluation stream derived from the same numbers.
const STREAM_REPRODUCTION: u64 = 0x5245_5052_4f44_0001;
/// Domain separator for the initial population.
const STREAM_FOUNDING: u64 = 0x464f_554e_4400_0001;

/// Fitness of an organism that has not been evaluated yet.
///
/// Finite on purpose: a checkpoint containing an unevaluated generation has to
/// survive a JSON round trip, and JSON has no way to spell negative infinity —
/// `serde_json` silently writes `null`, which then fails to load.
pub const UNEVALUATED_FITNESS: Real = Real::MIN;

/// One organism, with its lineage and its evaluation results.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Individual {
    pub id: u64,
    pub generation: u32,
    /// Parent ids; `0` means "no parent" (founders and immigrants).
    pub parents: [u64; 2],
    pub genome: Genome,
    pub fitness: Real,
    pub metrics: Metrics,
}

impl Individual {
    fn unevaluated(id: u64, generation: u32, parents: [u64; 2], genome: Genome) -> Individual {
        Individual {
            id,
            generation,
            parents,
            genome,
            fitness: UNEVALUATED_FITNESS,
            metrics: Metrics::default(),
        }
    }
}

/// A generation of organisms.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Population {
    pub generation: u32,
    pub individuals: Vec<Individual>,
    /// Next unused organism id. Serialised so a resumed run never reissues an id.
    pub next_id: u64,
}

impl Population {
    /// Create generation 0 from random genomes.
    pub fn founding(cfg: &Config) -> Population {
        let layout = cfg.brain_layout();
        let mut rng = Rng::new(derive_seed(&[cfg.experiment.seed, STREAM_FOUNDING]));
        let mut individuals = Vec::with_capacity(cfg.evolution.population_size);
        for i in 0..cfg.evolution.population_size {
            let genome = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            individuals.push(Individual::unevaluated(i as u64 + 1, 0, [0, 0], genome));
        }
        Population { generation: 0, next_id: cfg.evolution.population_size as u64 + 1, individuals }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    pub fn find(&self, id: u64) -> Option<&Individual> {
        self.individuals.iter().find(|i| i.id == id)
    }

    /// Indices ordered best-first.
    ///
    /// Ties break on id so the ranking is a pure function of the population and
    /// does not depend on sort stability.
    pub fn ranking(&self) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..self.individuals.len()).collect();
        idx.sort_by(|&a, &b| {
            let ia = &self.individuals[a];
            let ib = &self.individuals[b];
            ib.fitness
                .partial_cmp(&ia.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(ia.id.cmp(&ib.id))
        });
        idx
    }

    pub fn best(&self) -> &Individual {
        &self.individuals[self.ranking()[0]]
    }
}

/// Evaluate every organism in the population, in parallel.
///
/// Results are written back by index, so the outcome is identical regardless of
/// how many threads ran or in what order they finished.
pub fn evaluate_population(pop: &mut Population, cfg: &Config, pool: &rayon::ThreadPool) {
    pool.install(|| {
        pop.individuals.par_iter_mut().for_each_init(
            || EvalWorkspace::new(cfg),
            |ws, individual| {
                let result = sim::evaluate_with(&individual.genome, cfg, false, ws);
                individual.fitness = result.fitness;
                individual.metrics = result.metrics;
            },
        );
    });
}

/// Evaluate on the calling thread. Used by tests, benchmarks and replay.
pub fn evaluate_population_serial(pop: &mut Population, cfg: &Config) {
    let mut ws = EvalWorkspace::new(cfg);
    for individual in pop.individuals.iter_mut() {
        let result = sim::evaluate_with(&individual.genome, cfg, false, &mut ws);
        individual.fitness = result.fitness;
        individual.metrics = result.metrics;
    }
}

/// Build the next generation from an evaluated population.
///
/// Composition, in order: elites, then offspring, then a few random immigrants.
/// Elites are carried forward as *new* individuals with the elite as their sole
/// parent, so every organism in a generation has a distinct id and the same
/// provenance record. They are re-evaluated rather than having their fitness
/// copied — a few percent of wasted work in exchange for statistics that always
/// describe organisms actually simulated under the current configuration.
pub fn next_generation(pop: &Population, cfg: &Config) -> Population {
    let layout = cfg.brain_layout();
    let mut rng =
        Rng::new(derive_seed(&[cfg.experiment.seed, pop.generation as u64, STREAM_REPRODUCTION]));

    let ranked = pop.ranking();
    let target = cfg.evolution.population_size;
    let elite_count = cfg.evolution.elite_count.min(ranked.len());
    let immigrant_count = ((target as Real * cfg.evolution.immigrant_rate).round() as usize)
        .min(target.saturating_sub(elite_count));

    let mut next = Vec::with_capacity(target);
    let mut next_id = pop.next_id;
    let generation = pop.generation + 1;
    let mut take_id = || {
        let id = next_id;
        next_id += 1;
        id
    };

    for &i in ranked.iter().take(elite_count) {
        let elite = &pop.individuals[i];
        next.push(Individual::unevaluated(
            take_id(),
            generation,
            [elite.id, 0],
            elite.genome.clone(),
        ));
    }

    while next.len() < target - immigrant_count {
        let a = tournament(pop, &mut rng, cfg.evolution.tournament_size);
        let child = if rng.chance(cfg.evolution.crossover_rate) {
            let b = tournament(pop, &mut rng, cfg.evolution.tournament_size);
            // The fitter parent supplies the topology; a child that inherits a
            // body from one parent and a controller blended from both at least
            // starts from a body that was known to work.
            let (primary, secondary) = if pop.individuals[a].fitness >= pop.individuals[b].fitness {
                (a, b)
            } else {
                (b, a)
            };
            let mut g = genome::crossover(
                &pop.individuals[primary].genome,
                &pop.individuals[secondary].genome,
                &mut rng,
            );
            genome::mutate(&mut g, &mut rng, &cfg.mutation, &cfg.body, &cfg.brain, &layout);
            Individual::unevaluated(
                take_id(),
                generation,
                [pop.individuals[primary].id, pop.individuals[secondary].id],
                g,
            )
        } else {
            let mut g = pop.individuals[a].genome.clone();
            genome::mutate(&mut g, &mut rng, &cfg.mutation, &cfg.body, &cfg.brain, &layout);
            Individual::unevaluated(take_id(), generation, [pop.individuals[a].id, 0], g)
        };
        next.push(child);
    }

    while next.len() < target {
        let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        next.push(Individual::unevaluated(take_id(), generation, [0, 0], g));
    }

    Population { generation, individuals: next, next_id }
}

/// Pick the best of `size` uniformly sampled individuals.
///
/// Tournament selection rather than fitness-proportionate: it is insensitive to
/// the scale and sign of the fitness function, which matters when objectives are
/// configurable and can go negative.
fn tournament(pop: &Population, rng: &mut Rng, size: usize) -> usize {
    let mut best = rng.pick(pop.individuals.len());
    for _ in 1..size.max(1) {
        let challenger = rng.pick(pop.individuals.len());
        if pop.individuals[challenger].fitness > pop.individuals[best].fitness {
            best = challenger;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick_config() -> Config {
        let mut cfg = Config::default();
        cfg.evolution.population_size = 24;
        cfg.simulation.duration = 1.0;
        cfg.simulation.settle_time = 0.2;
        cfg
    }

    fn pool(threads: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap()
    }

    #[test]
    fn founding_population_has_the_configured_size_and_unique_ids() {
        let cfg = quick_config();
        let pop = Population::founding(&cfg);
        assert_eq!(pop.len(), cfg.evolution.population_size);
        let mut ids: Vec<u64> = pop.individuals.iter().map(|i| i.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), pop.len());
        assert!(ids.iter().all(|&id| id > 0), "id 0 is reserved for 'no parent'");
    }

    #[test]
    fn founding_is_reproducible_from_the_seed() {
        let cfg = quick_config();
        assert_eq!(Population::founding(&cfg), Population::founding(&cfg));

        let mut other = quick_config();
        other.experiment.seed += 1;
        assert_ne!(Population::founding(&cfg), Population::founding(&other));
    }

    #[test]
    fn parallel_and_serial_evaluation_agree() {
        let cfg = quick_config();
        let mut a = Population::founding(&cfg);
        let mut b = a.clone();
        evaluate_population(&mut a, &cfg, &pool(4));
        evaluate_population_serial(&mut b, &cfg);
        for (x, y) in a.individuals.iter().zip(&b.individuals) {
            assert_eq!(x.fitness, y.fitness, "organism {}", x.id);
            assert_eq!(x.metrics, y.metrics);
        }
    }

    #[test]
    fn thread_count_does_not_change_results() {
        let cfg = quick_config();
        let mut one = Population::founding(&cfg);
        let mut many = one.clone();
        evaluate_population(&mut one, &cfg, &pool(1));
        evaluate_population(&mut many, &cfg, &pool(8));
        assert_eq!(one, many);
    }

    #[test]
    fn ranking_is_best_first_and_deterministic() {
        let cfg = quick_config();
        let mut pop = Population::founding(&cfg);
        evaluate_population_serial(&mut pop, &cfg);
        let r = pop.ranking();
        for w in r.windows(2) {
            assert!(pop.individuals[w[0]].fitness >= pop.individuals[w[1]].fitness);
        }
        assert_eq!(r, pop.ranking());
        assert_eq!(pop.best().id, pop.individuals[r[0]].id);
    }

    #[test]
    fn next_generation_preserves_size_and_issues_fresh_ids() {
        let cfg = quick_config();
        let mut pop = Population::founding(&cfg);
        evaluate_population_serial(&mut pop, &cfg);
        let next = next_generation(&pop, &cfg);

        assert_eq!(next.len(), cfg.evolution.population_size);
        assert_eq!(next.generation, 1);
        let max_old = pop.individuals.iter().map(|i| i.id).max().unwrap();
        assert!(next.individuals.iter().all(|i| i.id > max_old));
        assert!(next.next_id > next.individuals.iter().map(|i| i.id).max().unwrap());
    }

    #[test]
    fn lineage_is_recorded() {
        let cfg = quick_config();
        let mut pop = Population::founding(&cfg);
        evaluate_population_serial(&mut pop, &cfg);
        let next = next_generation(&pop, &cfg);

        let known: std::collections::HashSet<u64> = pop.individuals.iter().map(|i| i.id).collect();
        let mut with_two_parents = 0;
        for child in &next.individuals {
            for p in child.parents {
                assert!(p == 0 || known.contains(&p), "unknown parent {p}");
            }
            if child.parents[0] != 0 && child.parents[1] != 0 {
                with_two_parents += 1;
            }
        }
        assert!(with_two_parents > 0, "crossover never produced a two-parent child");
    }

    #[test]
    fn elites_are_carried_forward_unchanged() {
        let mut cfg = quick_config();
        cfg.evolution.elite_count = 3;
        let mut pop = Population::founding(&cfg);
        evaluate_population_serial(&mut pop, &cfg);
        let ranked = pop.ranking();
        let next = next_generation(&pop, &cfg);

        for (k, &r) in ranked.iter().take(3).enumerate() {
            let elite = &pop.individuals[r];
            assert_eq!(next.individuals[k].genome, elite.genome);
            assert_eq!(next.individuals[k].parents[0], elite.id);
        }
    }

    #[test]
    fn all_children_are_valid_genomes() {
        let cfg = quick_config();
        let layout = cfg.brain_layout();
        let mut pop = Population::founding(&cfg);
        for _ in 0..8 {
            evaluate_population_serial(&mut pop, &cfg);
            pop = next_generation(&pop, &cfg);
            for i in &pop.individuals {
                assert!(i.genome.is_valid(&cfg.body, &layout), "organism {} invalid", i.id);
            }
        }
    }

    #[test]
    fn reproduction_is_reproducible() {
        let cfg = quick_config();
        let mut pop = Population::founding(&cfg);
        evaluate_population_serial(&mut pop, &cfg);
        assert_eq!(next_generation(&pop, &cfg), next_generation(&pop, &cfg));
    }

    #[test]
    fn stronger_selection_pressure_favours_the_top_of_the_ranking() {
        let cfg = quick_config();
        let mut pop = Population::founding(&cfg);
        evaluate_population_serial(&mut pop, &cfg);

        let mean_rank = |size: usize| -> f64 {
            let mut rng = Rng::new(7);
            let ranked = pop.ranking();
            let mut rank_of = vec![0usize; pop.len()];
            for (r, &i) in ranked.iter().enumerate() {
                rank_of[i] = r;
            }
            let n = 4000;
            (0..n).map(|_| rank_of[tournament(&pop, &mut rng, size)] as f64).sum::<f64>() / n as f64
        };
        assert!(mean_rank(8) < mean_rank(2), "larger tournaments must select better");
    }

    #[test]
    fn immigrants_appear_when_configured() {
        let mut cfg = quick_config();
        cfg.evolution.immigrant_rate = 0.25;
        let mut pop = Population::founding(&cfg);
        evaluate_population_serial(&mut pop, &cfg);
        let next = next_generation(&pop, &cfg);
        let orphans = next.individuals.iter().filter(|i| i.parents == [0, 0]).count();
        assert_eq!(orphans, 6);
    }
}
