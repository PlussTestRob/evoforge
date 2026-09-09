# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

EvoForge is a headless evolutionary artificial-life simulator in Rust. Organisms built from jointed primitive parts and driven by small feed-forward controllers are evaluated in a purpose-built rigid-body simulator; fitness drives selection, crossover, and mutation. The `evo` CLI is the only user-facing surface. No game engine, no ML framework, no GPU path, no rendering.

Rust 1.82+. Binary is `target/release/evo` (`evo.exe` on Windows).

## Commands

```bash
cargo build --release
cargo test --workspace --all-targets          # full suite; CI runs this on Ubuntu, Windows, macOS
cargo test --test golden                      # one test file
cargo test --test pipeline evolution_improves_the_population   # one test
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# The four CI gates are: test, fmt, clippy, and:
cargo run --release --bin evo -- verify experiments/first-walkers.toml --generations 3
```

CLI subcommands: `run`, `bench`, `inspect`, `replay`, `verify`, `rescore`. `evo run <config.toml>` writes a self-describing directory under `runs/` (gitignored); `--resume runs/<dir>` continues it. `evo rescore` re-weights a finished run without re-simulating — use it to sanity-check a new fitness term before breeding under it.

Diagnostic probes are `examples/*.rs`, auto-discovered (no `[[example]]` entries in `Cargo.toml`):

```bash
cargo run --release --example dead_organism_probe -- runs/<run>   # the corpse gate
cargo run --release --example golden_probe                        # regenerate golden constants
```

The viewer (`viewer/`) is plain HTML + JS with Three.js from a CDN — no build step. Serve it with `python -m http.server 8000 --directory viewer`.

## Architecture

Straight-line pipeline, one module per stage, dependencies pointing downward only:

```
genome → phenotype → physics → sim → fitness → evolution
```

- `math` — `Real`(=f32)/Vec3/Quat/Mat3 plus hand-rolled `sin`, `cos`, `ln`, `tanh`
- `rng` — xoshiro256++, `derive_seed`
- `config` — TOML experiment config; source of truth for types and defaults
- `genome` — heritable description, mutation, slot-aligned crossover
- `brain` — fixed-topology feed-forward controller over a weight slice
- `phenotype` — the only place genes become geometry
- `physics/` — bodies, shapes, contacts, joints, motors, limits, terrain (`Flat`/`Rough`/`Fractal` behind one height function)
- `sim` — one evaluation: inputs → controller → motors → step → `Metrics`
- `fitness` — `Metrics` → scalar
- `evolution` — population, selection, reproduction, lineage
- `record` — run dirs, checkpoints, replays, `ARTIFACT_FORMAT`
- `runner` — the generation loop; the only module that does I/O or reads a clock

Recording is a **second** evaluation of a handful of organisms, not a flag threaded through the hot loop — that works because evaluation is pure. Checkpoints are written *after* breeding, so a snapshot holds a bred-but-unevaluated generation; this ordering is what makes resume idempotent.

`PartGene.slot` is a stable controller index, unique per genome and never reused. Controller inputs/outputs are indexed by slot, not tree position, so deleting a limb does not rewire the rest of the controller.

## Invariants — these are correctness contracts, not style

**1. Evaluation is a pure function of `(genome, config)`.** `sim::evaluate` reads no globals, no clock, no shared state. Never add ambient state or side effects to the evaluation path.

**2. Randomness is derived, never ambient.** Every stream comes from `rng::derive_seed`. Never draw from a shared generator whose position depends on execution order.

**3. Off is exact.** A feature disabled by setting its rate/probability/weight to zero must draw the identical random stream, keep the identical controller input count, and reproduce every earlier result bit for bit. Never "almost disable" something with a tiny nonzero value — that draws randomness and changes controller layout. A new optional feature needs a test asserting it reproduces the pre-feature golden before it merges.

**4. `fitness::score` reads `Metrics` and nothing else.** It must never reach into the physics world. New objectives go through recorded metrics + `evo rescore`, not through the solver.

**No `std` transcendentals in `src/`.** `f32::sin`, `cos`, `ln`, `tanh` are not bitwise portable. Use `crate::math` versions. Arithmetic and `sqrt` are IEEE-exact and used freely.

**A metric must measure what its name says.** When a strategy exploits a metric, repair the instrument, not the behaviour — do not add a penalty term to paper over a dishonest measurement.

## Golden tests

`tests/golden.rs` holds committed numeric constants for a frozen config (`tests/golden.toml`), compared on all three CI platforms. They are the strongest correctness guarantee here. A changed constant means either a deliberate versioned change (bump `ARTIFACT_FORMAT` in `src/record.rs`, update `CHANGELOG.md`, regenerate with `golden_probe`, explain it) or a bug. **Treat it as a bug until proven otherwise.** Never `#[ignore]` a failing test.

Standing gates that must not be deleted or weakened: `a_dead_organism_does_not_travel`, `travel_survives_refining_the_solver`, `self_collision_is_not_a_motor`, `standing_still_does_not_beat_travelling`.

## Adding things

**A `Metrics` field:** add to the struct in `src/sim.rs` → record it in `sim::evaluate` (or trial aggregation) → expose as an optional `FitnessCfg` term in `src/fitness.rs` defaulting to zero → bump `ARTIFACT_FORMAT` in `src/record.rs` with a fallback default for older artifacts → add the off-is-exact gate test.

**A config field:** add to `src/config.rs` with `#[serde(default)]`. Include it in `fingerprint()` only if it affects dynamics — that function is the resume guard, and recording/display-only fields do not belong there. There is deliberately no escape hatch for a changed fingerprint.

**An experiment:** copy the closest `experiments/*.toml`, change only the independent variable, name it after that variable (`brittle-walkers.toml`, not `joints-should-fail.toml`). Verify the corpse gate afterwards and record the result. If dynamics change it is a new experiment — do not resume an old run under new settings.

**A probe:** `examples/`, one question each, answer to stdout.

## Style

`rustfmt.toml`: `max_width = 100`, `use_small_heuristics = "Max"`. Clippy runs with `-D warnings`; fix warnings rather than `#[allow]`ing them without a comment explaining why. Commit messages: short, descriptive, no ticket numbers.

## Documentation

Each doc has one job; put content in the right place rather than duplicating it. `README.md` is a front door only (~250 lines) — no config essays, results tables, or lab notes. `RESULTS.md` = measured findings. `CONFIG.md` = TOML guide in prose. `ARCHITECTURE.md` = why the code is shaped this way, incl. replaceable seams. `ROADMAP.md` = phases and open questions. `CHANGELOG.md` = `ARTIFACT_FORMAT` bumps; do not invent history. `*_PLAN.md` = lab notes, status marked at top, historical ones left as archives.

When fixing a stale claim (future tense for shipped work, "planned"/"not yet built" for things that exist, phase numbers contradicting Phase 0's inventory), edit it in place — do not add a correction note beside the wrong text.
