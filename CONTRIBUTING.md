# Contributing to EvoForge

EvoForge is an open artificial-life laboratory. The best contributions are new experiments, new diagnostic probes, and findings from running longer or differently-seeded versions of existing experiments.

## Building and testing

Rust **1.82** or later is required (`rustup show` to check).

```bash
# Build the release binary
cargo build --release

# Run the full test suite (mirrors CI: runs on Ubuntu, Windows, and macOS)
cargo test --workspace --all-targets

# Check formatting
cargo fmt --all --check

# Lint
cargo clippy --workspace --all-targets -- -D warnings

# Verify the determinism claim (same result on 1 thread and N threads)
./target/release/evo verify experiments/first-walkers.toml
```

All four of these are run in CI on every push. A contribution should pass all four before review.

## The golden tests and "off is exact"

`tests/golden.rs` contains committed numeric constants for a short deterministic run. These are the strongest correctness guarantee in the codebase — they prove that the result has not moved since the constants were recorded, on any supported platform.

**Do not casually rewrite the golden constants.** A change to them means either:
1. A deliberate, versioned change to the simulator that intentionally breaks prior results (increment `ARTIFACT_FORMAT` and update `CHANGELOG.md`), or
2. A bug.

"Off is exact" is the related invariant: a feature that is disabled (by setting its rate, probability, or weight to zero) must not consume any randomness, must not change the controller's input count, and must leave every earlier result reproducible bit for bit. Adding a new optional feature means adding it in a way that satisfies this invariant *before* it is enabled.

## Adding an experiment

Experiments live in `experiments/`. Copy the closest existing TOML, change what you intend to change, and run it:

```bash
./target/release/evo run experiments/your-experiment.toml
```

Document what you expected, what you observed, and whether the corpse gate passes:

```bash
cargo run --release --example dead_organism_probe -- runs/<your-run>
```

See [RESULTS.md](RESULTS.md) for what "the corpse gate passes" means and why it matters.

## Adding a diagnostic probe

Probes live in `examples/`. Each answers one question: "is what I measured real?" They run against a finished run or a fixed random seed and write their answer to stdout. Add them to `Cargo.toml` as `[[example]]` entries.

Existing probes and what they ask:

| Probe | Question |
|---|---|
| `dead_organism_probe` | With motors off, how far does this champion still travel? |
| `conveyor_probe` | Which property of the ground gives distance away? |
| `drift_probe` | Is a bigger body genuinely better, or is it collecting more free ride per part? |
| `energy_probe` | Does a passive body ever end with more mechanical energy than it started with? |
| `terrain_probe` | Is the ground actually crossable, and how is its difficulty distributed? |
| `leak_probe` | Where does un-earned travel come from? |
| `refine_probe` | Which subsystem loses its travel when the solver is refined? |
| `friction_probe` | Does a resting body get the friction Coulomb says it is owed? |
| `sensor_probe` | Are sensor readings plausible on known geometry? |
| `terrain_samples` | Export height samples for viewer terrain-check verification |
| `golden_probe` | Reproduce the golden constants for a given config |

## Adding a new fitness objective or metric

**Fitness reads `Metrics` and nothing else.** A new objective is a new function over the already-recorded metric set, not a new sensor into the physics world. Write it in `src/fitness.rs`, record the needed quantities in `src/sim.rs` (as fields on `Metrics`), and verify with `evo rescore` that the new scoring is sensible on an existing run before breeding under it.

If the new metric changes the artifact format, increment `ARTIFACT_FORMAT` in `src/record.rs` and add a note to `CHANGELOG.md`.

## Style

The project uses `rustfmt` with the settings in `rustfmt.toml` (`max_width = 100`, `use_small_heuristics = "Max"`). Run `cargo fmt` before committing.

Clippy is run with `-D warnings` — all warnings are errors in CI.

Commit messages follow the style of the existing log: short, descriptive, past-tense or present-tense imperative, no ticket numbers.
