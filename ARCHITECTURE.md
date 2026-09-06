# EvoForge architecture

This document records *why* the code is shaped the way it is, and where it is
expected to change. For what the project is, see [README.md](README.md).

## The three properties everything else follows from

**1. Evaluation is a pure function of `(genome, config)`.**
`sim::evaluate` reads no globals, no clock and no shared state. Consequences:

- population evaluation is embarrassingly parallel today and distributable later,
  with no synchronisation beyond collecting results by index;
- any organism can be reconstructed from its genome alone, years later, which is
  what makes selective recording viable;
- a champion can be re-simulated at higher recording fidelity without having
  stored anything extra during the run.

**2. Randomness is derived, never ambient.**
Every stream comes from `rng::derive_seed(&[experiment_seed, identity...])`.
Nothing draws from a shared generator whose position depends on execution order.
`evo verify` checks the resulting property directly: identical results on 1 and
N threads, compared bit for bit.

**3. Nothing renders.**
The simulator writes trajectories for a handful of selected organisms and stops.
There is no window, no event loop, no frame budget. Visualisation is a separate
program reading a separate file.

## Module map

```
math        Real/Vec3/Quat/Mat3 + deterministic sin, cos, ln, tanh
rng         xoshiro256++, seed derivation
config      TOML experiment configuration
genome      heritable description; mutation and crossover
brain       fixed-topology feed-forward controller, evaluated over a weight slice
phenotype   genome -> bodies and joints (the only place genes become geometry)
physics     rigid bodies, shapes, contacts, joints, motors, limits
sim         one evaluation: sensors -> controller -> motors -> step -> metrics
fitness     metrics -> scalar
evolution   population, selection, reproduction, lineage
record      run directories, statistics, checkpoints, replays
stats       per-generation aggregates
runner      the generation loop (the only module that does I/O or reads a clock)
bench       throughput and cost measurement
```

Dependencies point downward only. `fitness` cannot see the physics world;
`evolution` cannot see a `RigidBody`.

## The pipeline

```
Genome ──► Phenotype ──► World ──► Behaviour ──► Metrics ──► Fitness
                                                               │
                                    Next generation ◄── Selection, crossover, mutation
```

One generation:

1. Evaluate every organism in parallel (`rayon`, results collected by index).
2. Summarise into `GenerationStats`; append to `stats.csv` and the terminal table.
3. Append one `organisms.jsonl` line per organism: id, generation, parents,
   fitness, metrics.
4. If this is a recording generation, **re-simulate** the top N plus a few random
   samples with trajectory capture, and write replays and genomes.
5. Breed: elites, then tournament-selected offspring with crossover and mutation,
   then a small number of random immigrants.
6. Checkpoint if scheduled — *after* breeding, so the snapshot holds the new,
   unevaluated generation.

Step 4 is worth noting: recording is a *second* evaluation of a handful of
organisms rather than a flag threaded through the hot loop. Because evaluation is
pure this costs a few evaluations per recorded generation and keeps the common
path free of frame buffers it would throw away.

Step 6's ordering is load-bearing, and is the reason resume is idempotent. A
checkpoint holds a generation that has been bred but not yet evaluated, appended
to `stats.csv` or written to `organisms.jsonl`. Snapshotting the generation just
*finished* instead would make a resume redo work that is already on disk and
append a second copy of its records — and that is the common case, not the exotic
one, because a preempted worker has no on-finish checkpoint to land on and must
resume from a periodic one. `ARTIFACT_FORMAT` 2 marks the change, and resume
refuses an older checkpoint rather than silently duplicating a generation.

## The measured window

The controller is held off during `settle_time`. Motors stay at target zero while
the organism drops, so the measured window does not start from a pose the network
already shoved.

Every field on `Metrics` describes the same interval: from the settled pose at
`settle_time` to the end of the evaluation. That includes `actuation`, which
subtracts the impulse spent during the drop. Charging for the settle would price a
fall the organism cannot influence, and would make the charge scale with
`settle_time`, mass and hinge count — turning `energy_penalty` into a morphology
penalty applied before the controller has any say.

`Trace::measure_start_t` names the instant `Metrics::start` was taken from, and
the recorder always emits a frame there even when the boundary does not fall on
the recording grid, so a viewer can draw that pose rather than interpolate
towards it.

## Key data structures

| Type | Notes |
|---|---|
| `Genome { parts: Vec<PartGene>, weights: Vec<Real> }` | Body is a tree stored flat with `parts[i].parent < i`, so cycles are unrepresentable and construction is one forward pass. |
| `PartGene.slot` | Stable controller index, unique per genome, never reused. See below. |
| `weights` | **Fixed length** for an experiment, sized for `max_parts`. Aligned crossover becomes a one-liner. |
| `RigidBody` | Flat `Vec`, referenced by index. No parent pointers, no `Rc`; the world is two contiguous vectors. |
| `Metrics` | The only thing `fitness` may read. |
| `Individual` | Genome plus id, generation and parent ids — the lineage record. |

### Why slots

Controller inputs and outputs are indexed by `slot`, not by position in the part
tree. If a mutation deletes a limb, every remaining limb keeps its slot, and
therefore keeps the meaning of every weight referring to it. Indexing by tree
position would silently rewire the entire controller whenever morphology
changed — the "competing conventions" failure that makes naive morphology
evolution not work.

## Determinism, in detail

Reproducibility here means *bitwise*, not "close enough". Three things threaten
that, and each is closed:

1. **libm.** `sin`, `cos`, `ln`, `tanh` are not bitwise portable across platforms
   and libc versions. All four are implemented in `math` with fixed polynomials
   (tested against `std` to ~1e-5). Arithmetic and `sqrt` are IEEE-exact, so they
   are used freely. Quaternion integration uses only multiplies and one `sqrt`;
   controller clocks are triangle waves rather than sinusoids; joint angles are
   carried as `(cos, sin)` pairs so no `atan2` is ever needed.
2. **RNG drift.** `rand` does not promise a stable stream across releases, so the
   generator is in-repo and pinned by a golden-value test.
3. **Scheduling.** Seeds derive from identity, not from draw order; parallel
   results are collected by index.

Each of those is closed at the level it occurs, and the *result* is pinned
end to end: `tests/golden.rs` asserts committed fitness and metric bit patterns
for a frozen configuration, and CI runs it on Linux, Windows and macOS. Without
that, all three mechanisms could be intact while the pipeline still produced
different numbers than it did last year — which is the failure the mechanisms
exist to prevent, and the only one an in-process comparison cannot see.

`Real` is `f32`. Bodies are few, so throughput and cache behaviour matter more
than precision. It is a type alias; changing it is one line.

Config digests are a versioned, field-by-field byte stream, not pretty-printed
TOML. A serializer upgrade or a comment in the experiment file cannot change
whether a checkpoint is considered the same experiment.

## Physics: what and why

Semi-implicit Euler with sequential-impulse constraint solving in maximal
coordinates, Baumgarte stabilisation, per-shape contacts against a terrain
height function, Coulomb friction.

**Why not Rapier or similar.** They solve a much larger problem: arbitrary
geometry, broad-phase acceleration, continuous collision, sleeping, scene graphs.
We need a few convex primitives on a ground plane connected by hinges, with organisms that do not
collide with themselves. Each of those simplifications deletes a subsystem. What
remains is small enough to read in one sitting, has no version-drift risk to
reproducibility, and has no per-evaluation setup cost worth measuring — which
matters when the workload is millions of very short evaluations rather than one
long one.

**Not implemented, deliberately.** Self-collision: an organism's blocks pass
through each other, as in Sims (1994). This removes the broad phase entirely and
avoids the jitter that overlapping freshly-mutated limbs would cause.

**The intended replacement.** Organisms are trees with no self-collision, which
is exactly the case where a reduced-coordinate articulated-body formulation
(Featherstone's ABA) is both faster and far more stable: joints become exactly
satisfied by construction instead of approximately satisfied by iteration, so the
solver-iteration budget disappears. Sequential impulses came first because they
are much harder to get *wrong*.

## Assumptions that would affect scaling

Stated explicitly, because they are the ones worth revisiting before spending
money on compute:

- **No self-collision.** Removes the broad phase. Adding it later is the single
  biggest per-evaluation cost increase available.
- **Small bodies (<= a few dozen parts).** The solver is O(iterations x
  constraints) with dense 3x3 solves; fine at this size, poor at hundreds.
- **Fixed controller topology.** Weight vectors are uniform across a population,
  which is what makes crossover trivial and memory predictable. Evolving topology
  breaks that and needs a different alignment scheme.
- **Whole population evaluated per generation.** Generational, not steady-state.
  A distributed version will want steady-state or island models so that a slow or
  preempted worker does not stall a barrier.
- **JSON persistence.** Fine for the current volume; trajectories should become
  binary before recording gets aggressive.
- **`organisms.jsonl` grows with population x generations.** At 100 x 100 it is
  trivial. At 1000 x 1,000,000 it is not, and will need a compact binary format
  or aggregation.

## Deliberately replaceable seams

| Seam | Today | Replace with |
|---|---|---|
| `physics::World::step` | sequential impulses | Featherstone ABA |
| `physics::TerrainModel` | `Flat`, behind `height_at`/`normal_at` | heightfield, obstacles |
| `fitness::Objective` | three variants over `Metrics` | anything that reads `Metrics` |
| `genome::crossover` | slot-aligned uniform | morphological crossover |
| `brain` | one hidden layer, fixed size | recurrent, or evolved topology |
| `record` | JSON / JSON Lines | binary traces, object storage |
| `evolution::next_generation` | generational, tournament + elitism | steady-state, islands, novelty search |

None of these are behind traits or plugin registries. Enums and free functions
are enough for one implementation each, and a trait added at the point a second
implementation actually exists will be a better trait.

## Cloud readiness (design only, nothing built)

The pieces that matter are in place, and nothing else has been built:

- headless, no graphics dependencies, Linux-friendly;
- checkpoint and restart, with a config-digest check that refuses to resume a run
  whose dynamics changed (while still allowing `generations` to be raised, so a
  finished run can be extended). A version mismatch is refused unless
  `--force-resume` is given, and so is a checkpoint older than
  `MIN_RESUMABLE_FORMAT`. Resume is *idempotent*: it never re-appends a generation
  that is already on disk, whether it lands on a periodic checkpoint or the
  on-finish one;
- checkpoints and replays are written to a temp file, fsynced, and renamed into
  place, with the containing directory flushed afterwards on Unix so the rename
  itself survives a kill. There is deliberately no delete-then-rename fallback:
  unlinking a good manifest before its replacement is in place would turn a
  transient failure into permanent loss;
- append-only text for the growing files, and every reader of a `.jsonl` tolerates
  the truncated final line a kill mid-append leaves behind;
- pure evaluation, so work can be distributed and re-done freely after a
  preemption;
- `evo bench` reports scaling efficiency and cost per million evaluations,
  because "more cores" and "more evolution per dollar" are not the same thing —
  on the development machine, 12 threads deliver 6.7x throughput but each
  evaluation costs 1.8x more than single-threaded.

No AWS, Kubernetes, Terraform or object-storage integration. Those are premature
until the simulator is fast and measured.

## Test strategy

- **Unit tests** for the deterministic components: RNG stream stability (golden
  values), transcendental accuracy against `std`, genome invariants under
  thousands of mutations, phenotype construction, and physics behaviours stated
  as properties — a dropped box comes to rest on the ground, free fall matches
  the analytic solution, friction stops a slide and its absence does not, a weld
  holds, a hinge motor turns, a hinge limit stops it.
- **Integration tests** for the claims that only exist end to end: fitness
  improves, the same seed gives the same answer on any thread count, a recorded
  champion re-simulates to the identical fitness, a run directory is
  self-describing.
- **Golden values** (`tests/golden.rs`) for the reproducibility contract itself.
  Every other determinism test compares two runs inside the same process on the
  same binary, which cannot catch the simulator quietly producing *different*
  numbers than it used to — after a refactor, on another platform, under another
  compiler. These compare against committed constants instead, over a frozen
  config in `tests/golden.toml`, and are what make a multi-platform CI matrix
  meaningful. They are sensitive enough to catch a change in the seventh decimal
  place of a cosine coefficient, which the `2e-5` tolerance in
  `dsincos_matches_std` sails straight past. A failure here is a bug report, not a
  constant to update.
- **CLI tests** (`tests/cli.rs`) drive the real binary, for the behaviours that
  only exist at that level: which file a command decides to write, and whether it
  survives a run directory damaged by a kill.
- **`evo verify`** does the determinism check as a command, so it can be run
  against a real experiment config rather than a test fixture.

CI runs the suite on Linux, Windows and macOS — the determinism claim is a
cross-platform one, and `math`'s hand-rolled transcendentals exist precisely so
that results do not depend on the platform libm. It also enforces `cargo fmt` and
`clippy -D warnings`, and runs `evo verify` against a real experiment.
