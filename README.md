# EvoForge

**A headless evolutionary artificial-life simulator written in Rust.**

EvoForge explores how complex physical structures and behaviors can emerge through evolution from relatively simple rules.

Creatures are constructed from blocks and joints, controlled by small neural networks, and evaluated in a simulated physical environment. Successful organisms reproduce, unsuccessful organisms are culled, and mutations introduce variation across generations.

The goal is not to build a game or a general-purpose AI framework. EvoForge is an **artificial-life laboratory**: a fast, reproducible environment for experimenting with evolution, morphology, neural control, and emergent behavior.

## Core Idea

The basic evolutionary loop is:

```text
Genome
   ↓
Phenotype
   ↓
Physical Simulation
   ↓
Behavior
   ↓
Fitness
   ↓
Selection
   ↓
Crossover + Mutation
   ↓
Next Generation
```

Each organism has both a physical body and a controller. Its genome determines characteristics such as its body structure, joints, neural-network parameters, and potentially other traits as the project evolves.

The organism is then placed into an environment and allowed to act for a limited simulation period.

Its behavior determines its fitness.

The fittest organisms become the parents of the next generation.

Repeat.

## Goals

EvoForge is being designed around a few principles:

* **Evolution first** — evolutionary algorithms are the primary mechanism for discovering behavior.
* **Physical embodiment** — organisms have bodies, constraints, sensors, and physical consequences.
* **Emergence** — interesting behavior should arise from simple rules rather than being explicitly programmed.
* **Performance** — maximize evolutionary progress per unit of compute.
* **Determinism** — experiments should be reproducible from their configuration and random seed.
* **Headless operation** — the simulator should run efficiently without a graphical environment.
* **Experimentation** — terrain, objectives, population parameters, morphology, and neural architectures should be configurable.
* **Replayability** — interesting organisms can be recorded and visualized after simulation.
* **Cloud-friendly execution** — experiments should eventually be able to run continuously across inexpensive, disposable compute.

## What EvoForge Is Not

EvoForge is intentionally **not**:

* A game engine
* A real-time game
* A general-purpose physics engine
* A machine-learning framework
* A reinforcement-learning framework
* A GPU-first simulation
* A visualization project

Visualization is a separate concern.

The simulator should be able to run for hours or days on a Linux server without ever creating a window.

## Architecture

The project is centered around a clean separation between simulation and visualization.

```text
                    ┌──────────────┐
                    │    Genome    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Phenotype   │
                    │ Body + Brain │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Simulation  │
                    │    Physics   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   Fitness    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Evolution   │
                    └──────┬───────┘
                           │
                           ▼
                     Next Generation
```

Selected simulation data can be recorded independently:

```text
Simulation
    │
    ├── Evolutionary metadata
    │
    └── Selected trajectories
             │
             ▼
       Replay / Renderer
```

The renderer does not participate in the simulation. It consumes recorded data and reconstructs what happened afterward.

## Organisms

The initial concept is based on modular, block-based organisms.

A creature may consist of:

* Rigid parts — a box, a taper, a sphere, a capsule or a cylinder
* Fixed connections
* Hinged joints
* Rotational/orbital joints
* Joint limits
* Motorized joints
* Sensors
* A neural-network controller

The exact morphology system is intentionally expected to evolve alongside the project.

The important distinction is that the **body and controller are part of the organism**, rather than treating the neural network as an abstract agent operating independently of its physical form.

## Neural Networks

Neural networks provide control signals for an organism's joints.

The initial implementation is expected to use small feed-forward networks implemented directly in Rust.

Networks should remain:

* Small
* Fast
* Serializable
* Mutatable
* Crossable
* Deterministic
* Easy to inspect

There is no dependency on a heavyweight machine-learning framework.

The purpose of the neural network is not to train through gradient descent. Its parameters can instead become part of the organism's genome and evolve alongside its physical structure.

## Evolution

The initial evolutionary algorithm will remain intentionally simple.

A typical generation may:

1. Create or load a population.
2. Construct each organism from its genome.
3. Simulate each organism.
4. Calculate its fitness.
5. Rank the population.
6. Preserve or reproduce successful organisms.
7. Cull unsuccessful organisms.
8. Generate offspring through crossover and mutation.
9. Begin the next generation.

Population size, selection pressure, mutation rate, crossover behavior, and other parameters should be experiment configuration rather than hard-coded assumptions.

## Fitness

Fitness is experiment-dependent.

Potential objectives include:

* Distance traveled
* Average velocity
* Elevation gained
* Progress toward a target
* Energy efficiency
* Stability
* Combination objectives

An experiment should be able to define its objective without requiring fundamental changes to the simulation architecture.

## Experiments

Experiments should be reproducible from configuration.

A future experiment might look conceptually like:

```toml
[experiment]
name = "first-walkers"
seed = 12345

[evolution]
population_size = 100
generations = 1000
mutation_rate = 0.05

[simulation]
duration = 10.0
gravity = 9.81

[environment]
terrain = "flat"

[fitness]
objective = "distance"
```

The exact configuration format and schema are subject to change.

## Performance

Performance is a first-class feature rather than an optimization to be considered later.

The primary metric is not simply simulation steps per second.

The more useful question is:

> **How much evolutionary progress can be purchased with a given amount of compute?**

Important measurements will include:

* Organisms evaluated per second
* Simulation steps per second
* Population evaluation time
* Scaling across CPU cores
* Memory usage
* Generations per hour
* Evaluations per dollar
* Generations per dollar

Individual organism evaluations are naturally parallel, making evolutionary evaluation a good candidate for multi-core and eventually distributed execution.

## Determinism

Experiments should be reproducible whenever practical.

A simulation should be defined by its:

* Experiment configuration
* Random seed
* Genome
* Simulation parameters
* Software version

This makes it possible to investigate interesting results, reproduce failures, compare algorithm changes, and replay organisms independently of the original evolutionary run.

## Recording & Replay

Recording every organism at every simulation step would create enormous amounts of unnecessary data.

Instead, EvoForge should selectively record detailed trajectories for organisms such as:

* Generation champions
* Top-N organisms
* Random samples
* Specific lineages
* Organisms explicitly selected for inspection

A champion can also be re-simulated from its genome when higher-fidelity recording is required.

This allows the evolutionary simulation to remain focused on computation while still making interesting results observable.

## Command Line

The simulator is intended to be usable entirely from the command line.

The eventual interface may look something like:

```bash
evo run experiment.toml
evo benchmark
evo replay <organism-id>
```

The exact commands are not yet finalized.

## Technology

Initial technology choices:

* **Rust** — simulation and evolution
* **Cargo** — build and dependency management
* **TOML** — experiment configuration
* **Linux** — primary execution environment

The project intentionally starts with a small dependency footprint.

There is no requirement for:

* Unity
* Godot
* Roblox
* Python
* PyTorch
* TensorFlow
* GPU compute

Those technologies may be useful for future tooling, but the core simulator should not depend on them.

## Development Philosophy

EvoForge should favor:

* Simple data structures
* Explicit behavior
* Strong typing
* Deterministic execution
* Measurable performance
* Small, well-defined components
* Minimal dependencies
* Replaceable subsystems
* Tests around evolutionary correctness
* Benchmarks around simulation performance

The project should resist premature abstraction.

In particular, EvoForge should **not become a generic game engine or generic AI framework**.

Build the smallest, fastest, clearest artificial-life laboratory capable of demonstrating genuine evolution.

## Quick Start

```bash
cargo build --release

# Evolve. ~15 seconds for 100 generations of 100 organisms on a laptop.
# On Windows: target\release\evo.exe
./target/release/evo run experiments/first-walkers.toml

# Stricter locomotion: signed +X progress and an upright bonus.
./target/release/evo run experiments/directed-walkers.toml

# The same experiment with all five part shapes enabled.
./target/release/evo run experiments/shaped-walkers.toml

# Every animal-oriented constraint at once: symmetry, muscle-limited torque,
# tendons, self-collision, rough ground, repeated trials, commanded headings.
./target/release/evo run experiments/animals.toml

# The same, on seeded fractal terrain that differs in every trial.
./target/release/evo run experiments/fractal-animals.toml

# Measure throughput and its scaling across cores.
./target/release/evo bench experiments/first-walkers.toml

# Summarise a run.
./target/release/evo inspect runs/first-walkers-<timestamp>

# Re-simulate the best recorded organism, at higher recording fidelity.
./target/release/evo replay runs/first-walkers-<timestamp> --best --hz 60

# Prove the reproducibility claim: same results on 1 core and on N.
./target/release/evo verify experiments/first-walkers.toml
```

`run` prints a table per generation and writes everything to a self-describing
run directory:

```text
runs/first-walkers-1788654317/
  manifest.json        experiment id, seed, version, config digest
  config.toml          the fully resolved configuration
  stats.csv            per-generation statistics, ready to plot
  organisms.jsonl      every organism: id, generation, parents, fitness, metrics
  genomes.jsonl        genomes of recorded organisms
  checkpoints/         full-population snapshots for resume
  replays/             recorded trajectories
```

Resume an interrupted run, or extend a finished one, with
`--resume runs/<dir>`; raising `generations` is allowed, but changing anything
that affects the dynamics is refused rather than silently accepted.

### Part shapes

A part is not necessarily a cuboid. Each one is *carved* from the box its
`half_extents` describe, as one of five primitives:

| `shape` | what it is | ground contact |
|---|---|---|
| `box` | fills the box | 8 corners |
| `taper` | rectangular frustum along the box's longest axis — a wedge, a foot, a claw | 8 corners |
| `sphere` | inscribed sphere | 1 point |
| `capsule` | inscribed capsule along the longest axis | 2 points |
| `cylinder` | inscribed cylinder along the longest axis | 3 points per rim |

The gene is only the *kind*; dimensions always come from `half_extents`, so one
size gene keeps doing one job and a part never grows when its shape changes.

Shape matters because of that last column, not because of appearance. Parts do
not collide with each other, so geometry reaches the simulation through exactly
two doors: how mass is distributed, and how a part meets the ground. A cuboid
catches on its corners; a capsule rolls and pivots; a sphere or a cylinder can
roll outright, which is a gait that was not previously available. What shape
does *not* change is what can be built — that is the genome's tree structure,
a separate question.

Enable them per experiment:

```toml
[body]
shapes = ["box", "taper", "sphere", "capsule", "cylinder"]
taper_top_scale = 0.45   # a taper's far end, as a fraction of its base

[mutation]
shape_rate = 0.04        # per part, per generation
```

`experiments/shaped-walkers.toml` is `directed-walkers.toml` with exactly that
added, so the two can be run against each other.

The default is `shapes = ["box"]`, and a single-entry roster is not merely a
restriction — it spends *no randomness* choosing, so a box-only experiment draws
the identical random stream it drew before shapes existed and reproduces earlier
results bit for bit. That is what lets `tests/golden.rs` keep the constants it
was born with. Replays and stored genomes gain a `shape` field that defaults to
`box` on read, so v2 artefacts still load and still mean what they meant.

### Toward animals

Animals are not the product of being scored on looking like animals. They are
what falls out of three pressures at once, and each is a lever here — none of
them a fitness term that mentions a leg.

The test applied to every rule below is whether it is justifiable *without
reference to the shape it produces*. Symmetry from developmental axes, torque
from muscle physiology, rough ground from ecology: all pass. "Must have four
legs" would not, and neither would a bonus for having them.

Every one is off by default, and off is exact rather than approximate: a
configuration that does not enable them draws the identical random stream, keeps
the identical controller layout, and reproduces every earlier result bit for bit.

**What a genome can say.** A `PartGene` can be `paired`, in which case it appears
twice as mirror images across the sagittal plane — and both copies share one
controller slot, so two legs are driven by one leg controller and move as a
pair, which is what a gait is. `antiphase` chooses whether the halves alternate
(a walk) or move together (a bound). A part can also `repeat` into a chain of
segments, which is where spines, tails and segmented limbs come from; children
hang off the end rather than sprouting from every segment. Structures on the
midline are forced to be symmetric about it, because a lopsided spine makes the
whole organism lopsided however carefully its limbs are paired.

```toml
[body]
pair_probability = 0.45   # 0 disables bilateral symmetry entirely
max_repeat = 3            # 1 disables segmentation
```

**What a body can be.** `muscle_stress` caps a joint's torque at what its own
girth could physically host — muscle force scales with cross-section, and the
torque it exerts with a moment arm that grows with the limb's width, so the
ceiling goes as `stress * area^1.5`. Without it `motor_torque` is drawn
independently of size and a matchstick can be as strong as a thigh.
`self_collision` stops an organism's parts passing through each other, which is
what makes limbs be outside the torso rather than inside it; parts are
approximated by capsules and jointed pairs are exempt, because they are meant to
touch. `tendon_frequency` gives every hinge a passive spring, expressed as a
natural frequency rather than a stiffness so that it means the same thing on a
thigh and on a toe — tendon elasticity is most of why animal running and hopping
are efficient.

```toml
[body]
muscle_stress = 12000.0   # 0 leaves motor_torque a free gene
tendon_frequency = 6.0    # rad/s; 0 for no tendon
tendon_damping = 0.5      # damping ratio

[environment]
self_collision = true
```

**What the world demands.** Rolling ground makes wheels and sliders stop being
optimal, so legs stop being strictly worse — the elegant version of asking for
legs without ever mentioning them. Repeating each evaluation from varied starts
means a single well-timed lunge no longer scores like a gait; every organism
faces the same set of starts, drawn from the experiment seed and the trial index,
so a score difference is a difference between organisms rather than between the
worlds they drew. And a commanded heading, given to the controller as an input
and scored along that direction, means an organism has to be steerable rather
than committed to one launch.

```toml
[simulation]
trials = 3                # 1 is a single trial, as before
start_jitter = 0.5        # how much the start pose varies
aggregate = "mean"        # or "worst", which asks for no bad day at all
steer = true              # each trial commands a direction
steer_spread = 0.9        # radians either side of +X

[environment]
terrain = "rough"
terrain_amplitude = 0.05
terrain_wavelength = 1.6

[fitness]
objective = "heading"     # distance along the commanded direction
```

`experiments/animals.toml` turns on all of it at once. It is slow — three trials,
self-collision and twelve solver iterations cost roughly an order of magnitude
against `directed-walkers` — and it does not yet produce quadrupeds: sixty
generations reached 22 m of commanded travel with an organism that is still
mostly rolling, upright for under two seconds of eight. The machinery is there;
what it needs is compute and tuning, not more rules.

### Ground that is actually ground

`terrain = "rough"` is two octaves of a sine field. It repeats every wavelength,
it has no seed, and it has features at one scale only, so every organism in
every trial of every experiment meets the same 13 cm ripple — memorisable in
principle, and not much like a landscape.

`terrain = "fractal"` is four octaves of seeded gradient noise over a warped
domain: aperiodic, keyed on the experiment seed, and slid and turned under the
organism between trials so no two trials share a hill. There is no
transcendental anywhere in it — integer hashing and polynomial arithmetic only —
which makes it a *better* reproducibility story than the sine field, not a
worse one, and its gradient is still exact, so contacts get the slope's own
normal rather than a finite-difference guess.

```toml
[environment]
terrain = "fractal"
terrain_seed = 0          # 0 derives the landscape from experiment.seed
terrain_amplitude = 0.25  # relief runs to about 2.5x this
terrain_wavelength = 3.0  # largest feature, metres
terrain_octaves = 4       # finest is wavelength / lacunarity^(octaves-1)
terrain_lacunarity = 2.0
terrain_gain = 0.5
terrain_warp = 0.6        # bends the field into ridges and basins
terrain_per_trial = true  # move the landscape between trials
```

`experiments/fractal-animals.toml` is `animals.toml` with exactly that changed.
Measured with `cargo run --release --example terrain_probe`, those settings give
0.62 m of relief against the old 0.13 m, at about the same median slope (9.9
degrees against 10.3) but with a far longer tail: 46 degrees at the steepest
against 21.

Two things it is honestly not. It is **not heterogeneous** — the domain warp is
usually sold as making some regions flat and others broken, and measurement says
otherwise: relief per 12 m tile varies by 10% of its mean whether the warp is
off or at full strength, because warping a stationary field with a stationary
displacement leaves it stationary. And it is **still a height field**:
single-valued and smooth, with no overhangs, walls or gaps, which is exactly
what a wheel is good at. Raising the amplitude makes the ground steeper, not a
different kind of problem. What defeats a wheel is a discontinuity at or above
its own radius — discrete obstacles — and whether that is needed is a question
for after this has run, not before.

The viewer mirrors the height field in JavaScript rather than being shipped a
sampled patch with every replay. That is cheap and it can drift, so every
non-flat trace records the physics' own height at sixteen points and the viewer
checks itself against them on load, saying so loudly if they disagree. The same
check runs offline:

```bash
cargo run --release --example terrain_samples > samples.json
node viewer/terrain_check.mjs samples.json
```

### Rewarding a jump

Distance objectives have no opinion about the ground, and the cheapest way to
travel is to stay on it. Two optional fitness terms ask for something else:

* `air_bonus` — per second with **every attached part** clear of the ground.
* `height_bonus` — per metre the centre of mass rises above where it started,
  so being tall is worth nothing and only *gaining* height counts.

Neither works alone. Hang time by itself rewards a long low skim; height by
itself rewards rearing up without ever taking off. Together, with distance still
the base objective, they ask for locomotion that leaves the ground.

"Clear of the ground" means a real margin — `AIRBORNE_CLEARANCE`, two
centimetres — not merely "no contact". A contact exists only once a point is
*below* the terrain, so on the naive definition a body hovering a millimetre
above it counts as flying. Asked for hang time that way, evolution needed forty
generations to produce organisms spending half the trial "airborne" while never
rising above the grass.

```toml
[fitness]
objective = "distance_x"
air_bonus = 3.0
height_bonus = 8.0
```

`experiments/jumpers.toml` combines those with durable joints
(`joint_endurance = 150`). A 430-generation run reached fitness 19.0 with a
four-part organism that travels 9.5 m while making six distinct hops, 1.67 s of
it airborne, peaking 17 cm off the ground. Both terms default to zero, so every
experiment that predates them is unaffected.

### Joints that wear out

A joint's `motor_torque` has always been a gene: the most force that joint can
deliver. With `body.joint_endurance` set, exceeding it finally costs something.

While a motor is **saturated** — the controller asking for a speed the joint
cannot reach at its torque — the joint loses health equal to the rotation it
fell short by, in radians. A joint driven within its means is never harmed, so
this is a charge for overreach rather than for being used. At zero health the
joint fails: it stops constraining anything, and everything hanging below it
falls away.

Detached parts keep tumbling in the world but stop counting toward the centre of
mass that fitness measures. Losing a limb costs you its usefulness, not a
phantom position penalty from where the wreckage lands.

The organism gets both halves of the trade-off:

* **Sensing** — each slot gains a controller input carrying that joint's
  remaining health, so a creature can feel a joint going and ease off it.
* **Disposition** — a heritable `caution` gene in `[0, 1]` throttles how hard
  every motor is driven. It is a standing bet, not a reaction: an organism
  cannot know how long its trial will last, so whether to sprint and risk
  tearing a joint or to pace itself has to be inherited rather than deduced.

`Metrics` gains `joints_lost`, recorded for every organism, so breakage can be
measured across a whole run without opening a replay.

```toml
[body]
joint_endurance = 25.0   # radians of undelivered rotation a joint survives; 0 = off
min_drive = 0.2          # however cautious it gets, it can still move this hard

[mutation]
caution_rate = 0.08
caution_sigma = 0.12
```

`experiments/brittle-walkers.toml` is `shaped-walkers.toml` with exactly that
added. Note that `joint_endurance = 25.0` is *aggressive*: a 120-generation run
at that setting averaged 2.5 lost joints per organism, and only 3 of 48 recorded
organisms finished intact. Raise it for a gentler world.

The default is `joint_endurance = 0`, which disables wear entirely — and, as
with shapes, disabling it is exact rather than approximate. No randomness is
spent on the caution gene, and the controller keeps its original input count, so
the weight vector stays the length it always was and every earlier result
reproduces bit for bit.

### Viewer

`viewer/` is a standalone browser player for one recorded replay. It is plain
HTML and JavaScript with Three.js from a CDN — no build step, no package
manager, and no dependency on the simulator. It reads poses off disk and never
simulates; the genome in a replay file is metadata, not something it re-expresses.

Serve the directory and open it (a `file://` URL will not work — browsers block
ES module imports and `fetch` from it):

```bash
# Unix
python3 -m http.server 8000 --directory viewer

# Windows (PowerShell)
python -m http.server 8000 --directory viewer
```

Then open <http://localhost:8000>. Click **Load sample** for the checked-in
two-block generation-0 replay, or use the file picker — or drag and drop — to
open any file from `runs/<run>/replays/`, such as a champion produced by
`evo replay <run> --best --hz 60`.

**Browsing a whole run.** *Open run folder…* takes the run directory itself —
the one holding `manifest.json` and `replays/` — and lists everything it
recorded. Rows are sortable by fitness, distance, speed, path length, wander,
seconds upright, mean height, actuation, distance-per-effort, joints lost,
caution, part count, generation or id; groupable by generation, parent, shape
mix, part count, or whether a joint failed; and there is a free-text filter that
matches ids, generations, shapes, parents and `broke`. The preset chips — Best,
Worst, Fastest, Most upright, Wanderers, Most efficient, Hardest working, Broke
a joint, Most cautious, Latest — are shortcuts onto that same machinery, not a
fixed menu: any combination of sort, grouping and filter is reachable directly.

Only recorded organisms can be played. A run stores trajectories for the top few
plus a random sample per recorded generation, so the worst organism overall
usually has no replay — the list shows what is actually watchable.

Opening a directory reads only each replay's header, stopping at the trajectory,
so a few hundred replays cost a few megabytes of reads rather than tens. The
poses are loaded only for the one being watched. Where a replay recorded a joint
failing, the moment is marked in red on the timeline.

Orbit with the left mouse button, pan with the right, zoom with the wheel.
Play/pause is the button or the space bar; the scrubber seeks by simulation
time, interpolating between recorded frames. The dark part of the timeline is
the settling drop before `measure_start_t`, where the controller is held off;
the lit part is the measured window that fitness is computed over.

## Project Status

**Milestone 1 complete: the pipeline works end to end and evolution demonstrably
occurs.**

A first run of `experiments/first-walkers.toml` — 100 organisms, 100 generations,
seventeen seconds of wall clock — took best fitness from 0.79 m to 4.76 m and the
*median* from 0.08 m to 3.20 m, with no divergent simulations and 77 of 100
morphologies still distinct. The rising median is the part that matters: with
elitism the best score cannot fall, so only the middle of the distribution moving
shows the population as a whole is improving. Omnidirectional `distance` can be
satisfied by tumbling; `experiments/directed-walkers.toml` asks for signed +X
progress and an upright bonus.

Implemented:

* block-based organisms with fixed and hinged joints, limits and motors
* a small hand-written feed-forward controller, evolved rather than trained
* a purpose-built impulse-based rigid-body solver — gravity, ground contact,
  friction, joints, joint limits, joint motors
* flat terrain behind a height-function interface
* genome and phenotype as distinct concepts, with stable controller slots that
  survive morphological mutation
* tournament selection, elitism, slot-aligned crossover, per-gene mutation,
  random immigration
* bitwise determinism, including hand-written transcendentals so results do not
  depend on the platform libm
* multi-core evaluation whose results are independent of thread count
* checkpoint, resume and run extension
* selective recording, and exact re-simulation of any stored genome
* `evo bench`, reporting organisms/second, scaling efficiency, and cost per
  million evaluations

Not built, deliberately: self-collision, a renderer, evolved network topology,
non-flat terrain, and any cloud infrastructure. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the reasoning behind each of those, the
assumptions that would affect scaling, and where the implementation is meant to
be replaced.

## License

[MIT](LICENSE).
