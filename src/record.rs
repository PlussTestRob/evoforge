//! Persistence: run directories, statistics, lineage, checkpoints, replays.
//!
//! # What is written, and what deliberately is not
//!
//! Recording every frame of every organism would dominate both the runtime and
//! the storage bill, and almost all of it would never be looked at. So:
//!
//! * **Always**, per organism: id, generation, parents, fitness, metrics. Small,
//!   append-only, and enough to reconstruct lineages and re-score a run later.
//! * **Selectively**, per recorded generation: trajectories for the top N plus a
//!   few random samples, and their genomes.
//! * **Periodically**: a full-population checkpoint, which is what makes a
//!   preempted cloud worker recoverable.
//!
//! Because evaluation is a pure function of `(genome, config)`, anything not
//! recorded can be regenerated exactly from a stored genome — including at a
//! much higher recording rate than the run itself used. That is the trade that
//! keeps the data model small.
//!
//! # Format
//!
//! JSON and JSON Lines. Not the most compact choice, and a binary trajectory
//! format is an obvious later optimisation, but the consumer of this data is
//! going to be a browser-based viewer, and append-only text is very hard to
//! corrupt when a worker is killed mid-write.

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::evolution::{Individual, Population};
use crate::fitness::Metrics;
use crate::genome::Genome;
use crate::math::Real;
use crate::rng::{derive_seed, Rng};
use crate::sim::Trace;
use crate::stats::GenerationStats;

const STREAM_SAMPLING: u64 = 0x5341_4d50_4c45_0001;

/// On-disk schema version for manifest, checkpoint, replay and stored-genome
/// records. Bump when a field is added, removed or reinterpreted.
pub const ARTIFACT_FORMAT: u32 = 1;

fn legacy_format() -> u32 {
    0
}

pub const MANIFEST_FILE: &str = "manifest.json";
pub const CONFIG_FILE: &str = "config.toml";
pub const STATS_FILE: &str = "stats.csv";
pub const ORGANISMS_FILE: &str = "organisms.jsonl";
pub const GENOMES_FILE: &str = "genomes.jsonl";
pub const CHECKPOINT_DIR: &str = "checkpoints";
pub const REPLAY_DIR: &str = "replays";

/// Identifies a run and the exact inputs that produced it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Manifest {
    #[serde(default = "legacy_format")]
    pub format: u32,
    pub experiment_id: String,
    pub experiment_name: String,
    /// Version of the simulator that produced the run. Results are only
    /// comparable across versions if the physics and RNG streams are unchanged.
    pub evoforge_version: String,
    pub seed: u64,
    pub config_digest: u64,
    pub created_unix: u64,
}

/// One line of `organisms.jsonl`: the permanent record of an organism.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrganismRecord {
    pub id: u64,
    pub generation: u32,
    pub parents: [u64; 2],
    pub fitness: Real,
    pub metrics: Metrics,
}

impl From<&Individual> for OrganismRecord {
    fn from(i: &Individual) -> Self {
        OrganismRecord {
            id: i.id,
            generation: i.generation,
            parents: i.parents,
            fitness: i.fitness,
            metrics: i.metrics,
        }
    }
}

/// One line of `genomes.jsonl`: enough to re-simulate an organism exactly.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StoredGenome {
    #[serde(default = "legacy_format")]
    pub format: u32,
    pub id: u64,
    pub generation: u32,
    pub parents: [u64; 2],
    pub fitness: Real,
    pub genome: Genome,
}

/// A full-population snapshot, sufficient to resume a run.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Checkpoint {
    #[serde(default = "legacy_format")]
    pub format: u32,
    pub evoforge_version: String,
    pub config_digest: u64,
    pub population: Population,
}

/// A recorded trajectory plus everything needed to interpret it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Replay {
    #[serde(default = "legacy_format")]
    pub format: u32,
    pub experiment_id: String,
    pub organism_id: u64,
    pub generation: u32,
    pub parents: [u64; 2],
    pub fitness: Real,
    pub metrics: Metrics,
    pub config_digest: u64,
    pub timestep: Real,
    pub genome: Genome,
    pub trace: Trace,
}

/// A run directory on disk.
pub struct Run {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl Run {
    /// Create a fresh run directory named after the experiment and the current
    /// time, and write the manifest and a resolved copy of the configuration.
    ///
    /// The resolved copy matters: it contains every defaulted field explicitly,
    /// so a run remains reproducible even after the defaults change.
    pub fn create(cfg: &Config) -> std::io::Result<Run> {
        let created = unix_time();
        let (experiment_id, dir) = claim_run_directory(
            &cfg.experiment.output_dir,
            &format!("{}-{}", sanitise(&cfg.experiment.name), created),
        )?;
        fs::create_dir_all(dir.join(CHECKPOINT_DIR))?;
        fs::create_dir_all(dir.join(REPLAY_DIR))?;

        let manifest = Manifest {
            format: ARTIFACT_FORMAT,
            experiment_id,
            experiment_name: cfg.experiment.name.clone(),
            evoforge_version: crate::VERSION.to_string(),
            seed: cfg.experiment.seed,
            config_digest: cfg.digest(),
            created_unix: created,
        };

        let run = Run { dir, manifest };
        write_json(&run.dir.join(MANIFEST_FILE), &run.manifest)?;
        fs::write(run.dir.join(CONFIG_FILE), cfg.to_toml_string())?;
        fs::write(
            run.dir.join(STATS_FILE),
            format!("{}\n", GenerationStats::CSV_HEADER),
        )?;
        Ok(run)
    }

    /// Open an existing run directory.
    pub fn open(dir: &Path) -> std::io::Result<Run> {
        let manifest: Manifest = read_json(&dir.join(MANIFEST_FILE))?;
        Ok(Run { dir: dir.to_path_buf(), manifest })
    }

    pub fn config(&self) -> Result<Config, crate::config::ConfigError> {
        Config::load(&self.dir.join(CONFIG_FILE))
    }

    pub fn append_stats(&self, stats: &GenerationStats) -> std::io::Result<()> {
        let mut f = append_file(&self.dir.join(STATS_FILE))?;
        writeln!(f, "{}", stats.to_csv_row())
    }

    /// Append the permanent record for every organism in a generation.
    pub fn append_organisms(&self, pop: &Population) -> std::io::Result<()> {
        let mut f = BufWriter::new(append_file(&self.dir.join(ORGANISMS_FILE))?);
        for individual in &pop.individuals {
            let record = OrganismRecord::from(individual);
            writeln!(f, "{}", serde_json::to_string(&record)?)?;
        }
        f.flush()
    }

    pub fn append_genome(&self, individual: &Individual) -> std::io::Result<()> {
        let stored = StoredGenome {
            format: ARTIFACT_FORMAT,
            id: individual.id,
            generation: individual.generation,
            parents: individual.parents,
            fitness: individual.fitness,
            genome: individual.genome.clone(),
        };
        let mut f = append_file(&self.dir.join(GENOMES_FILE))?;
        writeln!(f, "{}", serde_json::to_string(&stored)?)
    }

    /// Find a stored genome by organism id, looking in `genomes.jsonl` first and
    /// then in checkpoints (which hold whole populations).
    pub fn find_genome(&self, id: u64) -> std::io::Result<Option<StoredGenome>> {
        let path = self.dir.join(GENOMES_FILE);
        if path.exists() {
            let reader = BufReader::new(File::open(&path)?);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let Some(stored) = parse_jsonl_line::<StoredGenome>(&line)? else {
                    continue;
                };
                if stored.id == id {
                    return Ok(Some(stored));
                }
            }
        }

        for path in self.checkpoint_paths()? {
            let checkpoint: Checkpoint = read_json(&path)?;
            if let Some(individual) = checkpoint.population.find(id) {
                return Ok(Some(StoredGenome {
                    format: ARTIFACT_FORMAT,
                    id: individual.id,
                    generation: individual.generation,
                    parents: individual.parents,
                    fitness: individual.fitness,
                    genome: individual.genome.clone(),
                }));
            }
        }
        Ok(None)
    }

    pub fn checkpoint_path(&self, generation: u32) -> PathBuf {
        self.dir
            .join(CHECKPOINT_DIR)
            .join(format!("gen_{generation:06}.json"))
    }

    /// Checkpoint files, newest first — the order a resume wants.
    pub fn checkpoint_paths(&self) -> std::io::Result<Vec<PathBuf>> {
        let dir = self.dir.join(CHECKPOINT_DIR);
        if !dir.exists() {
            return Ok(Vec::new());
        }
        let mut paths: Vec<PathBuf> = fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        paths.sort();
        paths.reverse();
        Ok(paths)
    }

    pub fn write_checkpoint(&self, pop: &Population, cfg: &Config) -> std::io::Result<PathBuf> {
        let checkpoint = Checkpoint {
            format: ARTIFACT_FORMAT,
            evoforge_version: crate::VERSION.to_string(),
            config_digest: cfg.evolution_digest(),
            population: pop.clone(),
        };
        let path = self.checkpoint_path(pop.generation);
        write_json(&path, &checkpoint)?;
        Ok(path)
    }

    pub fn latest_checkpoint(&self) -> std::io::Result<Option<Checkpoint>> {
        match self.checkpoint_paths()?.first() {
            Some(path) => Ok(Some(read_json(path)?)),
            None => Ok(None),
        }
    }

    pub fn replay_path(&self, generation: u32, id: u64) -> PathBuf {
        self.dir
            .join(REPLAY_DIR)
            .join(format!("gen_{generation:06}_org_{id:08}.json"))
    }

    pub fn write_replay(
        &self,
        individual: &Individual,
        trace: Trace,
        cfg: &Config,
    ) -> std::io::Result<PathBuf> {
        let replay = Replay {
            format: ARTIFACT_FORMAT,
            experiment_id: self.manifest.experiment_id.clone(),
            organism_id: individual.id,
            generation: individual.generation,
            parents: individual.parents,
            fitness: individual.fitness,
            metrics: individual.metrics,
            config_digest: cfg.evolution_digest(),
            timestep: cfg.simulation.timestep,
            genome: individual.genome.clone(),
            trace,
        };
        let path = self.replay_path(individual.generation, individual.id);
        write_json(&path, &replay)?;
        Ok(path)
    }
}

/// Choose which organisms of an evaluated generation to record.
///
/// Champions first, then a deterministic random sample so the record is not
/// exclusively of winners — a population of near-identical also-rans is a fact
/// worth being able to see afterwards.
pub fn selection_for_recording(pop: &Population, cfg: &Config) -> Vec<u64> {
    if cfg.recording.every_generations == 0
        || pop.generation % cfg.recording.every_generations != 0
    {
        return Vec::new();
    }

    let ranked = pop.ranking();
    let mut chosen: Vec<u64> = ranked
        .iter()
        .take(cfg.recording.top_n.min(ranked.len()))
        .map(|&i| pop.individuals[i].id)
        .collect();

    if cfg.recording.random_samples > 0 {
        let mut seen: HashSet<u64> = chosen.iter().copied().collect();
        let mut rng = Rng::new(derive_seed(&[
            cfg.experiment.seed,
            pop.generation as u64,
            STREAM_SAMPLING,
        ]));
        // Bounded attempts: with a small population and many requested samples,
        // insisting on distinct picks could otherwise spin.
        for _ in 0..cfg.recording.random_samples * 8 {
            if chosen.len() >= cfg.recording.top_n + cfg.recording.random_samples {
                break;
            }
            let id = pop.individuals[rng.pick(pop.len())].id;
            if seen.insert(id) {
                chosen.push(id);
            }
        }
    }
    chosen
}

pub fn should_checkpoint(generation: u32, cfg: &Config) -> bool {
    cfg.checkpoint.every_generations > 0 && generation % cfg.checkpoint.every_generations == 0
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Atomically claim a fresh directory named `base`, or `base-1`, `base-2`, ...
///
/// Run directories are named after the experiment and the current *second*, so
/// two runs launched together would otherwise land in the same directory and
/// silently interleave their statistics and checkpoints. `create_dir` fails if
/// the directory exists, which makes the claim atomic even between processes.
fn claim_run_directory(parent: &Path, base: &str) -> std::io::Result<(String, PathBuf)> {
    fs::create_dir_all(parent)?;
    for attempt in 0..1000 {
        let id = if attempt == 0 {
            base.to_string()
        } else {
            format!("{base}-{attempt}")
        };
        let dir = parent.join(&id);
        match fs::create_dir(&dir) {
            Ok(()) => return Ok((id, dir)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        format!("could not find an unused run directory under {}", parent.display()),
    ))
}

fn append_file(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = {
        let mut name = path.as_os_str().to_os_string();
        name.push(".tmp");
        PathBuf::from(name)
    };
    {
        let file = File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    replace_file(&tmp, path)
}

/// Replace `to` with `from`. `rename` over an existing file is atomic on
/// Unix and fails on Windows, so Windows removes the destination first.
fn replace_file(from: &Path, to: &Path) -> std::io::Result<()> {
    match fs::rename(from, to) {
        Ok(()) => Ok(()),
        Err(_) if to.exists() => {
            fs::remove_file(to)?;
            fs::rename(from, to)
        }
        Err(e) => Err(e),
    }
}

/// Parse one JSON Lines record. Empty lines and a truncated final line
/// (typical of a kill mid-append) are skipped rather than failing the run.
fn parse_jsonl_line<T: for<'de> Deserialize<'de>>(line: &str) -> std::io::Result<Option<T>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    match serde_json::from_str(line) {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.is_eof() => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> std::io::Result<T> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(BufReader::new(file))?)
}

fn unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Make a name safe to use as a directory component.
fn sanitise(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    if cleaned.is_empty() {
        "experiment".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution;
    use crate::stats;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> TempDir {
            let dir = std::env::temp_dir().join(format!(
                "evoforge-test-{tag}-{}-{:?}",
                unix_time(),
                std::thread::current().id()
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

    fn test_config(dir: &TempDir) -> Config {
        let mut cfg = Config::default();
        cfg.experiment.name = "unit test".into();
        cfg.experiment.output_dir = dir.0.clone();
        cfg.evolution.population_size = 8;
        cfg.simulation.duration = 0.5;
        cfg.simulation.settle_time = 0.1;
        cfg
    }

    fn evaluated(cfg: &Config) -> Population {
        let mut pop = Population::founding(cfg);
        evolution::evaluate_population_serial(&mut pop, cfg);
        pop
    }

    #[test]
    fn create_lays_out_the_run_directory() {
        let tmp = TempDir::new("layout");
        let cfg = test_config(&tmp);
        let run = Run::create(&cfg).unwrap();

        assert!(run.dir.join(MANIFEST_FILE).exists());
        assert!(run.dir.join(CONFIG_FILE).exists());
        assert!(run.dir.join(STATS_FILE).exists());
        assert!(run.dir.join(CHECKPOINT_DIR).is_dir());
        assert!(run.dir.join(REPLAY_DIR).is_dir());
        // Directory names must be filesystem-safe even when the experiment is not.
        assert!(!run.manifest.experiment_id.contains(' '));

        let reopened = Run::open(&run.dir).unwrap();
        assert_eq!(reopened.manifest, run.manifest);
        assert_eq!(reopened.manifest.format, ARTIFACT_FORMAT);
        assert_eq!(reopened.config().unwrap(), cfg);
    }

    #[test]
    fn write_json_replaces_an_existing_file() {
        let tmp = TempDir::new("atomic");
        let path = tmp.0.join("value.json");
        write_json(&path, &1u32).unwrap();
        write_json(&path, &2u32).unwrap();
        let loaded: u32 = read_json(&path).unwrap();
        assert_eq!(loaded, 2);
        assert!(!path.with_extension("json.tmp").exists());
        // The sibling temp name is `value.json.tmp`, not `value.tmp`.
        assert!(!tmp.0.join("value.json.tmp").exists());
    }

    #[test]
    fn truncated_jsonl_lines_are_skipped() {
        let tmp = TempDir::new("truncated");
        let cfg = test_config(&tmp);
        let run = Run::create(&cfg).unwrap();
        let pop = evaluated(&cfg);
        run.append_genome(&pop.individuals[0]).unwrap();
        {
            let mut f = append_file(&run.dir.join(GENOMES_FILE)).unwrap();
            write!(f, "{{\"format\":1,\"id\":").unwrap();
        }
        let found = run.find_genome(pop.individuals[0].id).unwrap().unwrap();
        assert_eq!(found.format, ARTIFACT_FORMAT);
        assert_eq!(found.genome, pop.individuals[0].genome);
    }

    #[test]
    fn concurrent_runs_get_distinct_directories() {
        let tmp = TempDir::new("distinct");
        let cfg = test_config(&tmp);
        let a = Run::create(&cfg).unwrap();
        let b = Run::create(&cfg).unwrap();
        let c = Run::create(&cfg).unwrap();
        assert_ne!(a.dir, b.dir);
        assert_ne!(b.dir, c.dir);
        assert_ne!(a.manifest.experiment_id, b.manifest.experiment_id);
    }

    #[test]
    fn stats_file_accumulates_rows() {
        let tmp = TempDir::new("stats");
        let cfg = test_config(&tmp);
        let run = Run::create(&cfg).unwrap();
        let pop = evaluated(&cfg);
        for g in 0..3 {
            let mut s = stats::summarise(&pop, 1.0, 1.0);
            s.generation = g;
            run.append_stats(&s).unwrap();
        }
        let text = fs::read_to_string(run.dir.join(STATS_FILE)).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "header plus three rows");
        assert_eq!(lines[0], GenerationStats::CSV_HEADER);
    }

    #[test]
    fn organism_records_are_one_line_each() {
        let tmp = TempDir::new("organisms");
        let cfg = test_config(&tmp);
        let run = Run::create(&cfg).unwrap();
        let pop = evaluated(&cfg);
        run.append_organisms(&pop).unwrap();
        run.append_organisms(&pop).unwrap();

        let text = fs::read_to_string(run.dir.join(ORGANISMS_FILE)).unwrap();
        assert_eq!(text.lines().count(), pop.len() * 2);
        let first: OrganismRecord = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(first.id, pop.individuals[0].id);
    }

    #[test]
    fn checkpoints_roundtrip_and_are_ordered_newest_first() {
        let tmp = TempDir::new("checkpoint");
        let cfg = test_config(&tmp);
        let run = Run::create(&cfg).unwrap();
        let mut pop = evaluated(&cfg);
        run.write_checkpoint(&pop, &cfg).unwrap();
        pop.generation = 12;
        run.write_checkpoint(&pop, &cfg).unwrap();

        let paths = run.checkpoint_paths().unwrap();
        assert_eq!(paths.len(), 2);
        let latest = run.latest_checkpoint().unwrap().unwrap();
        assert_eq!(latest.format, ARTIFACT_FORMAT);
        assert_eq!(latest.population.generation, 12);
        assert_eq!(latest.population, pop);
        assert_eq!(latest.config_digest, cfg.evolution_digest());
    }

    /// A checkpoint is written *after* the next generation has been bred but
    /// before it is evaluated, so it necessarily contains unevaluated
    /// individuals. JSON cannot represent infinities, so the sentinel fitness has
    /// to stay finite or resume breaks.
    #[test]
    fn an_unevaluated_population_survives_a_checkpoint_round_trip() {
        let tmp = TempDir::new("unevaluated");
        let cfg = test_config(&tmp);
        let run = Run::create(&cfg).unwrap();

        let mut pop = evaluated(&cfg);
        pop = evolution::next_generation(&pop, &cfg);
        assert!(pop
            .individuals
            .iter()
            .all(|i| i.fitness == evolution::UNEVALUATED_FITNESS));

        let path = run.write_checkpoint(&pop, &cfg).unwrap();
        let loaded: Checkpoint = read_json(&path).unwrap();
        assert_eq!(loaded.population, pop);
    }

    #[test]
    fn genomes_can_be_found_by_id_from_either_source() {
        let tmp = TempDir::new("find");
        let cfg = test_config(&tmp);
        let run = Run::create(&cfg).unwrap();
        let pop = evaluated(&cfg);

        // Only the first organism is written to genomes.jsonl.
        run.append_genome(&pop.individuals[0]).unwrap();
        let found = run.find_genome(pop.individuals[0].id).unwrap().unwrap();
        assert_eq!(found.genome, pop.individuals[0].genome);

        // The rest are only recoverable via a checkpoint.
        let last = pop.individuals.last().unwrap();
        assert!(run.find_genome(last.id).unwrap().is_none());
        run.write_checkpoint(&pop, &cfg).unwrap();
        let found = run.find_genome(last.id).unwrap().unwrap();
        assert_eq!(found.genome, last.genome);

        assert!(run.find_genome(999_999).unwrap().is_none());
    }

    #[test]
    fn replays_roundtrip() {
        let tmp = TempDir::new("replay");
        let cfg = test_config(&tmp);
        let run = Run::create(&cfg).unwrap();
        let pop = evaluated(&cfg);
        let individual = &pop.individuals[0];
        let trace = crate::sim::evaluate(&individual.genome, &cfg, true)
            .trace
            .expect("a settled organism should produce a trace");

        let path = run.write_replay(individual, trace.clone(), &cfg).unwrap();
        let loaded: Replay = read_json(&path).unwrap();
        assert_eq!(loaded.format, ARTIFACT_FORMAT);
        assert_eq!(loaded.organism_id, individual.id);
        assert_eq!(loaded.genome, individual.genome);
        assert_eq!(loaded.trace, trace);
        assert_eq!(loaded.config_digest, cfg.evolution_digest());
    }

    #[test]
    fn recording_selection_respects_the_policy() {
        let tmp = TempDir::new("select");
        let mut cfg = test_config(&tmp);
        cfg.recording.every_generations = 5;
        cfg.recording.top_n = 2;
        cfg.recording.random_samples = 3;

        let mut pop = evaluated(&cfg);

        pop.generation = 5;
        let chosen = selection_for_recording(&pop, &cfg);
        assert_eq!(chosen.len(), 5);
        assert_eq!(chosen[0], pop.best().id);
        let unique: HashSet<u64> = chosen.iter().copied().collect();
        assert_eq!(unique.len(), chosen.len(), "no organism recorded twice");
        // Deterministic.
        assert_eq!(chosen, selection_for_recording(&pop, &cfg));

        // Not a recording generation.
        pop.generation = 6;
        assert!(selection_for_recording(&pop, &cfg).is_empty());
    }

    #[test]
    fn checkpoint_schedule_can_be_disabled() {
        let tmp = TempDir::new("schedule");
        let mut cfg = test_config(&tmp);
        cfg.checkpoint.every_generations = 4;
        assert!(should_checkpoint(8, &cfg));
        assert!(!should_checkpoint(9, &cfg));
        cfg.checkpoint.every_generations = 0;
        assert!(!should_checkpoint(8, &cfg));
    }
}
