//! Tests for the `evo` command line.
//!
//! These drive the real binary rather than the library, because the behaviours
//! that matter here only exist at that level: which file a command decides to
//! write, and whether it survives a run directory that was damaged by a kill.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const EVO: &str = env!("CARGO_BIN_EXE_evo");

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!(
            "evoforge-cli-{tag}-{}",
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

const TINY_EXPERIMENT: &str = r#"
[experiment]
name = "cli-test"
seed = 99

[evolution]
population_size = 6
generations = 2

[simulation]
duration = 0.3
settle_time = 0.1

[recording]
every_generations = 1
top_n = 1

[checkpoint]
every_generations = 0
on_finish = true
"#;

fn evo(args: &[&str]) -> Output {
    let output = Command::new(EVO).args(args).output().expect("failed to launch evo");
    if !output.status.success() {
        panic!(
            "evo {args:?} failed with {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    output
}

fn stdout_of(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Set up a run directory and return its path along with the config that made it.
fn run_experiment(tmp: &TempDir) -> (PathBuf, PathBuf) {
    let config = tmp.0.join("experiment.toml");
    fs::write(&config, TINY_EXPERIMENT).unwrap();
    let out = tmp.0.join("runs");
    evo(&["run", config.to_str().unwrap(), "--out", out.to_str().unwrap(), "--quiet"]);
    let dir = fs::read_dir(&out)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("no run directory was created");
    (config, dir)
}

fn replays(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir.join("replays"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn inspect_summarises_a_run() {
    let tmp = TempDir::new("inspect");
    let (_, dir) = run_experiment(&tmp);
    let out = stdout_of(&evo(&["inspect", dir.to_str().unwrap()]));
    assert!(out.contains("cli-test"), "{out}");
    assert!(out.contains("2 generations recorded"), "{out}");
    assert!(out.contains("best stored genome"), "{out}");
}

/// `evo replay --hz` re-samples the same simulation, which is the command's
/// documented purpose, so refreshing the run's own recording in place is correct.
#[test]
fn replay_at_a_higher_rate_refreshes_the_runs_own_recording() {
    let tmp = TempDir::new("refresh");
    let (_, dir) = run_experiment(&tmp);
    let before = replays(&dir);

    let out = stdout_of(&evo(&["replay", dir.to_str().unwrap(), "--best", "--hz", "120"]));
    assert!(!out.contains("left untouched"), "{out}");
    assert_eq!(replays(&dir), before, "no new file should appear");
}

/// `--duration` changes the simulation, so the result is not the trajectory the
/// run recorded and must not be written over it.
#[test]
fn replay_with_different_dynamics_does_not_overwrite_the_runs_recording() {
    let tmp = TempDir::new("variant");
    let (_, dir) = run_experiment(&tmp);

    let before = replays(&dir);
    let originals: Vec<(String, u64)> = before
        .iter()
        .map(|name| {
            let len = fs::metadata(dir.join("replays").join(name)).unwrap().len();
            (name.clone(), len)
        })
        .collect();

    let out = stdout_of(&evo(&["replay", dir.to_str().unwrap(), "--best", "--duration", "2.0"]));
    assert!(out.contains("left untouched"), "the user must be told: {out}");

    // Every original replay is byte-for-byte the size it was.
    for (name, len) in &originals {
        assert_eq!(
            fs::metadata(dir.join("replays").join(name)).unwrap().len(),
            *len,
            "{name} was modified"
        );
    }

    // And the variant landed beside them, tagged with its own config digest.
    let after = replays(&dir);
    let added: Vec<&String> = after.iter().filter(|n| !before.contains(n)).collect();
    assert_eq!(added.len(), 1, "expected exactly one new file, got {after:?}");
    assert!(added[0].contains("_cfg_"), "{:?}", added[0]);
}

/// A run killed mid-append leaves a partial final line. The commands used to work
/// out what happened must still work — that is the entire reason the JSON Lines
/// reader is tolerant.
#[test]
fn inspect_and_replay_survive_a_truncated_genome_line() {
    let tmp = TempDir::new("truncated");
    let (_, dir) = run_experiment(&tmp);

    let genomes = dir.join("genomes.jsonl");
    let mut text = fs::read_to_string(&genomes).unwrap();
    text.push_str("{\"format\":2,\"id\":123,\"gen");
    fs::write(&genomes, text).unwrap();

    let out = stdout_of(&evo(&["inspect", dir.to_str().unwrap()]));
    assert!(out.contains("best stored genome"), "{out}");
    let out = stdout_of(&evo(&["replay", dir.to_str().unwrap(), "--best"]));
    assert!(out.contains("re-simulated fitness"), "{out}");
}

/// A resumed run must not append a second copy of a generation, whichever
/// checkpoint it landed on. Driven through the CLI because this is how a
/// preempted worker actually comes back.
#[test]
fn resuming_through_the_cli_does_not_duplicate_records() {
    let tmp = TempDir::new("resume");
    let config = tmp.0.join("experiment.toml");
    // Periodic checkpoints, no on-finish snapshot: a killed worker's situation.
    fs::write(
        &config,
        TINY_EXPERIMENT
            .replace("every_generations = 0", "every_generations = 1")
            .replace("on_finish = true", "on_finish = false"),
    )
    .unwrap();
    let out_dir = tmp.0.join("runs");
    let config = config.to_str().unwrap();
    let out_arg = out_dir.to_str().unwrap();

    evo(&["run", config, "--out", out_arg, "--quiet"]);
    let dir = fs::read_dir(&out_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .unwrap();

    evo(&[
        "run",
        config,
        "--out",
        out_arg,
        "--quiet",
        "--resume",
        dir.to_str().unwrap(),
        "--generations",
        "4",
    ]);

    let stats = fs::read_to_string(dir.join("stats.csv")).unwrap();
    let generations: Vec<&str> = stats
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.split(',').next().unwrap())
        .collect();
    assert_eq!(generations, ["0", "1", "2", "3"], "stats.csv: {generations:?}");

    let organisms = fs::read_to_string(dir.join("organisms.jsonl")).unwrap();
    let count = organisms.lines().filter(|l| !l.trim().is_empty()).count();
    assert_eq!(count, 4 * 6, "one record per organism per generation");
}

#[test]
fn verify_proves_determinism_for_a_real_config() {
    let tmp = TempDir::new("verify");
    let config = tmp.0.join("experiment.toml");
    fs::write(&config, TINY_EXPERIMENT).unwrap();
    let out = stdout_of(&evo(&["verify", config.to_str().unwrap(), "--generations", "2"]));
    assert!(out.contains("identical"), "{out}");
}
