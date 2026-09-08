# EvoForge

**A headless evolutionary artificial-life simulator written in Rust.**

EvoForge explores how complex physical structures and behaviors can emerge through evolution from relatively simple rules.

Organisms are constructed from primitive parts and joints, controlled by small neural networks, and evaluated in a simulated physical environment. Successful organisms reproduce, unsuccessful organisms are culled, and mutations introduce variation across generations.

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

## Where This Is Going

Organisms are currently scored on how far they travel. That is the *first*
fitness criterion, not the defining one — it was chosen because it is the
simplest measurement that separates a body which does something from a body
which does nothing.

The direction of the project is to make the evaluation system capable of asking
harder questions — go uphill, go downhill, reach that beacon, reach as many
beacons as you can — and then to give organisms sensors so that what they are
being asked about is something they can perceive. The progression is:

> evolved locomotion → richer evaluation → varied environments and tasks →
> sensors → bodies, controllers and environments under selection together

[ROADMAP.md](ROADMAP.md) sets that out in phases, each anchored to a seam that
already exists in the code.

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

The initial concept is based on modular organisms built from primitive parts.

An organism may consist of:

* Rigid parts — a box, a taper, a sphere, a capsule or a cylinder
* Fixed connections
* Hinged joints
* Rotational/orbital joints
* Joint limits
* Motorized joints
* Passive tendons
* A neural-network controller

The exact morphology system is intentionally expected to evolve alongside the project.

The important distinction is that the **body and controller are part of the organism**, rather than treating the neural network as an abstract agent operating independently of its physical form.

### Sensing

An organism's controller already receives inputs describing the organism *to
itself*: its orientation, its velocity, its height, and for each joint an angle,
a ground-contact flag and — where joints can wear out — remaining health. These
are proprioceptive, fixed for an experiment, and identical for every organism in
it.

Sensors in the fuller sense — reporting something about the world *outside* the
body, mounted on a particular part, described by the genome, and aimed by the
controller when that part is on a moving limb — are Phase 3 of
[ROADMAP.md](ROADMAP.md), planned in [SENSOR_PLAN.md](SENSOR_PLAN.md).

**The rule that governs them: anything an organism knows about the world outside
its own body must arrive through a sensor with a position and an orientation on
that body.** World information is never handed to the controller as a free input,
however much cheaper that would be to build.

That is not fastidiousness. The question this project asks is what structure and
control an environment selects for *when the organism has to perceive that
environment to succeed in it*, and an organism given the answer is not answering
it — it is executing a policy over a coordinate computed somewhere else. The
shortcut is also invisible after the fact: a fitness curve produced by an
organism that was told where the target is looks exactly like one produced by an
organism that found it.

The line between a sense and an instruction is whether it changes during the
trial. A commanded heading is fixed at the start and never revised, so it is a
cue, like one given to a trained animal. A beacon's bearing, or the height of the
ground ahead, updates continuously as the organism acts — that is perception, and
it needs an organ. [ROADMAP.md](ROADMAP.md) audits the existing inputs against
this rule, including the one that fails it.

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

Fitness is experiment-dependent, and the set of questions it can ask is expected
to grow. Distance traveled is where it starts, not where it stops.

Objectives available today:

* Distance traveled, in any direction
* Signed displacement along +X
* Average velocity
* Progress along a commanded heading

with optional terms for time spent upright, time spent airborne, height gained,
and energy spent.

Objectives the evaluation system is being built to support:

* Elevation gained, and elevation lost under control
* Reaching a particular beacon
* Reaching as many beacons as possible within a run
* Evaluation across several environments and tasks within one experiment

An experiment should be able to define its objective without requiring
fundamental changes to the simulation architecture. Scoring reads a recorded
metric set and never reaches into the physics world, so an objective added later
can be applied to results gathered earlier.

## Unexpected Behavior

Evolution under a simple objective finds simple answers, and they are frequently
not the answers anyone had in mind. An organism that discovers an unanticipated
way to score well, while staying inside the rules of the simulation, has done
exactly what it was asked to do. That is a **result**, not a defect, and this
project does not treat it as something to be suppressed.

The goal is not organisms that look like animals, humans, or conventionally
designed robots. The goal is an environment in which physical structures and
neural controllers can evolve toward success under evaluation criteria that get
progressively more meaningful.

Three things do look alike from a fitness curve and are worth telling apart:

| | | |
|---|---|---|
| **A simulation fault** | the score required the simulator to break its own physics | fix the bug |
| **A measurement fault** | the metric did not measure what its name says | fix the instrument |
| **A strategy** | correct physics, honest measurement, and it scores anyway | keep it, and if you want something else, ask a different question |

Both faults have happened here, and each was fixed at the level it occurred —
never by making a behavior illegal. The tools in [examples/](examples/) exist to
make the distinction decidable rather than arguable; `dead_organism_probe`, which
switches a champion's motors off and measures how far it still travels, is the
sharpest of them. See [ROADMAP.md](ROADMAP.md) for the full treatment.

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

`experiments/animals.toml` turns all of it on at once. It is slow — three
trials, self-collision and twelve solver iterations cost roughly an order of
magnitude against `directed-walkers` — and it does not yet produce quadrupeds.

**A result reported here previously was wrong, and how it was wrong is worth
keeping.** It is the reference example of a *simulation fault* rather than a
strategy: the score was not something the organism earned under the rules, it was
the solver breaking its own physics. A 270-generation run reached 35 m of
commanded travel, and the organism that did it covered 26 of those metres with
its motors switched off.
Self-collision plus Baumgarte stabilisation had made a motor: overlapping parts
were shoved apart, the shove was added straight into velocity and *kept*, and a
body whose joints pulled those parts back together every step collected it
forever. Evolution found that long before it found walking, and no test in the
suite objected, because every determinism test compares a run against itself.

Version 0.3.0 fixes it — the correction now displaces bodies without ever
becoming momentum — and adds the test that would have caught it:
`a_dead_organism_does_not_travel`, which switches an organism's motors off and
insists it goes nowhere. Genomes stored under 0.2.x will not re-simulate to
their recorded fitness, and results from those runs should be read as upper
bounds rather than distances. `examples/dead_organism_probe.rs` will say how
much of any given champion was real.

### Ground that is actually ground

`terrain = "rough"` is two octaves of a sine field. It repeats every wavelength,
it has no seed, and it has features at one scale only, so every organism in
every trial of every experiment meets the same 13 cm ripple.

`terrain = "fractal"` is four bands of seeded gradient noise, and the reason it
is four is that one band cannot do the job. Fractional Brownian motion has a
single steepness, set by amplitude over wavelength, and applies it at every
scale it spans — so scaling it up gives uniformly steep ground, never occasional
cliffs. Ten metres of relief still tops out around 50 degrees with the median
climbing in lockstep.

| band | what it does |
|---|---|
| landscape | hills, tens of metres across and metres deep |
| detail | ground texture at the scale of an organism's own body |
| modulation | a slow field saying where the ground is calm and where it is savage |
| terracing | quantises height to steps, turning hillside into plateau-and-cliff |

Terracing is the one that makes a sheer face: the riser is steeper than the
underlying slope by exactly `1 / terrain_riser`, and because a terrace is flat
for most of its span, the difficulty ends up concentrated in a small fraction of
the area with the rest left crossable. Measured on the shipped settings with
`cargo run --release --example terrain_probe`: 6.3 m of relief, median slope 8
degrees, 99th percentile 79, maximum 87, **93% of the plane walkable**, and only
1 straight 20 m crossing in 400 that avoids meeting a wall.

```toml
[environment]
terrain = "fractal"
terrain_seed = 0                     # 0 derives it from experiment.seed
terrain_amplitude = 3.0              # the landscape band
terrain_wavelength = 25.0
terrain_octaves = 5
terrain_warp = 0.6
terrain_detail_amplitude = 0.35      # organism-scale texture
terrain_detail_wavelength = 3.0
terrain_modulation = 0.9             # calm regions and savage ones
terrain_modulation_wavelength = 35.0
terrain_step = 0.8                   # terrace height: the cliffs
terrain_riser = 0.12                 # fraction of a terrace spent climbing
terrain_terrace_mask = true          # cliffs only where it is savage
terrain_per_trial = true             # move the landscape between trials
```

There is no seam anywhere in this: `TerrainModel` is one analytic function over
an unbounded domain, with no chunks, tiles or stitching, and the physics never
touches a mesh. Its gradient is exact, so contacts get the slope's own normal
rather than a finite-difference guess, and there is no transcendental in it at
all — integer hashing and polynomial arithmetic only, which makes it a *better*
reproducibility story than the sine field rather than a worse one.

Two limits worth stating plainly. `terrain_riser` has a floor that is physics
rather than taste — a wall thinner than a few integration steps is not a cliff
but a tunnelling bug, so `Config::validate` measures the walls a config would
produce and refuses the ones that are too thin. And it is **still a height
field**: single-valued and smooth, with a measured ceiling near 88 degrees and a
contact solver that is unreliable above about 75. Overhangs, gaps and true
vertical need discrete obstacles, which are not built.

The viewer mirrors the height field in JavaScript rather than being shipped a
sampled patch with every replay. That is cheap and it can drift, so every
non-flat trace records the physics' own height at sixteen points and the viewer
checks itself against them on load, saying so loudly if they disagree. The same
check runs offline:

```bash
cargo run --release --example terrain_samples > samples.json
node viewer/terrain_check.mjs samples.json
```

### What each ground selects for

Three arms of `animals.toml`, identical but for `[environment]`, same seed, 30
generations of 100: flat, the sine field, and the terraced fractal landscape.

| arm | best | mean | median | median gen 0 → 29 | mean parts 0 → 29 | distinct structures |
|---|---|---|---|---|---|---|
| flat | 9.18 | 6.65 | 8.94 | 1.07 → 8.94 | 4.52 → 2.12 | 21 / 100 |
| rough | 7.53 | 4.21 | 4.98 | 0.86 → 4.98 | 4.52 → 3.14 | 46 / 100 |
| fractal | 3.97 | 2.44 | 2.53 | 0.80 → 2.53 | 4.52 → 7.82 | 97 / 100 |

Scores fall monotonically with difficulty. That is the result the earlier
comparison could not produce: on the pre-0.3.0 solver the fractal run scored 28.1
against the sine field's 12.0 — harder ground scoring more than twice as high,
which was the tell that the simulator, not the organisms, was doing the
travelling. The median rises in every arm, so all three populations are improving
as populations rather than carrying one lucky champion.

**What the three grounds actually produce**, watched in the viewer rather than
inferred from the numbers:

* **Flat** selects small machines that *vibrate*. Two parts, buzzing, and that is
  enough — nothing in a flat plane plus a distance objective asks for more.
* **Rough** selects slightly larger bodies and visibly less vibration. A 13 cm
  ripple is enough to stop buzzing from working as well as it does on glass.
* **Fractal** selects large machines — nearly the eight-part maximum — that are
  not obviously good at moving themselves. Roughly half the population recorded
  at generation 20 moves *just enough to fall off a nearby drop* and then stops.
  Organism 2032 is the clearest case: 96% of its travel comes in the first half
  of the measured window while it descends 0.34 m, after which it spends six
  seconds thrashing in place — path length still growing, displacement flat, and
  13,337 units of actuation spent on neither. On ground with 5.6 m of relief and
  cliffs to 82 degrees, that is a perfectly sound reading of "travel as far as
  you can". A plausible untested prediction is that more generations would find
  rolling.

  It is not the whole population, and at generation 20 it is not winning: the
  other half move steadily, two of the seven recorded organisms net *climb*, and
  the steady movers outscore the fallers (1.67 m against 1.31 m and below). So
  falling reads as a cheap competing optimum that caps out low rather than as
  the dominant strategy — which is a different problem from an exploit, and a
  more interesting one.

  **How to see this, since the obvious statistic does not.** Net elevation
  change over a run says how far down an organism ended up, not when it earned
  its distance, and on these bodies it is uniformly about −0.22 m whatever the
  organism is doing — which makes it look modest and makes it correlate weakly
  with fitness. The signature that works is *temporal*: what share of the final
  displacement was reached in the first half of the window. A gait splits it
  roughly 50/50, as every flat and rough organism does. A fall-and-stop puts 80%
  or more in the first half.

None of those three is a defect, and none of them is something to legislate
against. They are correct answers to the question actually being asked, which is
"how far did you get". The fractal population in particular is doing something
the objective genuinely rewards; if what we want is controlled descent rather
than a well-aimed fall, the fix is a task that can tell those apart — which is
what [ROADMAP.md](ROADMAP.md) Phase 2 is for — and not a penalty for falling.

**The corpse gate passes**, which is what makes the comparison valid at all.
Champions re-evaluated with their motors fully off:

| arm | champion alive | motors off | share |
|---|---|---|---|
| flat | 6.28 m | −0.14 m | −2% |
| rough | 2.96 m | 0.20 m | 7% |
| fractal | 1.54 m | 0.34 m | 22% |

Against 97% before the split-impulse fix. The gate is the absolute figure — under
2 m in eight seconds — and 0.34 m clears it comfortably; the 22% share is
inflated by a small denominator, because locomotion on that ground is only 1.54 m
to begin with.

**One blind spot in that probe worth writing down.** It measures free distance
from where the organism *starts*. An organism that spends a little actuation
getting itself to a cliff edge and then falls is not doing anything a
motors-off corpse can imitate, because a corpse never reaches the edge. So the
figures above are a lower bound on how much of the fractal score the terrain is
handing over, not a full accounting. That is a limitation of the diagnostic, not
a reason to distrust the comparison — the point of the gate is to catch the
simulator propelling things, and it does.

**Part count moves in opposite directions**, and it is not free drift doing it.
`examples/drift_probe.rs`, motors off, 40 random organisms per part count:

| parts | flat | rough | fractal |
|---|---|---|---|
| 2 | −0.02 m | −0.07 m | 0.09 m |
| 4 | 0.01 m | −0.04 m | 0.10 m |
| 8 | 0.04 m | 0.04 m | 0.17 m |

Drift does rise with part count on the fractal field, but going from four parts
to eight buys 0.07 m against a spread of roughly 1.5 m between the population
mean and the best — about 5% of what selection is working on. And on flat and
rough, where drift is essentially zero, part count *collapses* rather than
growing. So the growth to 7.8 parts reads as a real finding about hard ground
rather than as bodies farming the solver.

**Cost, decomposed.** At equal body size (generation 0, 4.52 parts in every arm)
the terrain alone costs 5.6x from flat to fractal and 2.1x from rough to fractal,
which matches what the terrain plan predicted. The rest is endogenous: hard
ground evolves bigger bodies and bigger bodies cost more to simulate, so the
fractal arm got 2x slower over the run while the flat arm got 1.5x faster. End to
end the arms differed 11x in wall clock, and only about half of that is the
ground itself.

| arm | generation 0 | generation 29 |
|---|---|---|
| flat | 82.2 organisms/s @ 4.52 parts | 122.7 @ 2.12 parts |
| rough | 30.6 organisms/s @ 4.52 parts | 38.9 @ 3.14 parts |
| fractal | 14.7 organisms/s @ 4.52 parts | 7.4 @ 7.82 parts |

**Read this as a direction check, not a settled comparison.** Thirty generations
on one seed, against the eighty of the original A/B, with no repetition — which
is precisely the kind of measurement [TERRAIN_PLAN.md](TERRAIN_PLAN.md) §12 warns
about relying on. The ordering and the corpse gate are solid; the part-count
finding deserves a longer run before it is treated as established.

### What 150 generations on the fractal field actually does

The fractal arm was extended to generation 150 to find out whether it plateaus.
It does not, and the answer changes what the numbers above appear to say.

| | gen 29 | gen 149 |
|---|---|---|
| best fitness | 3.97 | **7.24** |
| median fitness | 2.53 | 4.14 |
| median travel, recorded organisms | 1.31 m | **6.58 m** |
| fall-and-stop share of recorded organisms | 3 of 7 | **0–1 of 7** |
| median temporal split | 51% | 45% |
| distance covered dead (corpse gate) | 22% | 13% |

Two things happened, and only one of them was expected.

**The crude strategy dissolved on its own.** The fall-and-stop organisms of
generation 20 are essentially gone by generation 120, without any intervention:
the median temporal split settles at 45%, which is what a gait looks like, and
median travel among recorded organisms rises eighteenfold to 6.58 m — level with
what *flat* ground produced. Best fitness gains per 30-generation block run +2.21,
+0.27, +0.36, +0.43, so progress is decelerating but had not stopped at 150.

**A subtler bias entrenched instead.** At generation 149 every organism in the
population ends lower than it started — 100 of 100, spanning −0.010 to −0.725 m —
and fitness now correlates with elevation change at **−0.60**. At generation 29
that correlation was −0.15. So while the obvious downhill strategy was
disappearing, selection under a pure distance objective was quietly getting
*better* at travelling downhill: champions now cover about ten metres of ground
per metre of height they give up, a ratio stable since generation 40.

That is the more interesting result, and it is a specification finding rather
than an optimisation one. Distance on sloped ground pays for descent, and 150
generations is long enough for that to become the population's defining
characteristic. It is what [FITNESS_PLAN.md](FITNESS_PLAN.md) exists to address —
not by penalising the behaviour, but by asking a question that elevation is part
of the answer to.

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

That is a *measurement fault*, and the fix was to make the word mean what it
says rather than to penalise hovering. The same reasoning produced the
centre-of-mass correction in the previous section: shedding a limb used to move
the measured average for free, so the discontinuity is cancelled and detaching a
part is now worth exactly zero metres — not forbidden, just not paid for.

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
  remaining health, so an organism can feel a joint going and ease off it.
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

### Positional correction, and why it is small

`simulation.baumgarte` controls how much of a contact's penetration is corrected
per step. It defaults to **0.05**, which is low, and the reason is worth knowing
before raising it.

Correction is applied at the contact point, which is offset from the body's
centre of mass, so it induces rotation as well as separation — and integrating
that rotation moves the body. With few solver iterations contacts stay deeply
penetrated, the correction stays large, and an organism that arranges to
penetrate the ground rhythmically converts the correction into travel. It looks
exactly like vibration-driven locomotion in a replay, and it is not.

Measured on evolved champions at `baumgarte = 0.2`: they covered 2.72 m, and
refining the solver from 12 iterations to 96 removed **94%** of it. At 0.05 they
cover 0.36 m and refinement removes almost nothing. Raising `solver_iterations`
fixes it too — 48 iterations costs 1.7x and 96 costs 2.8x — where lowering
`baumgarte` costs 1.08x.

The general test is `travel_survives_refining_the_solver`: distance that exists
only at a coarse solve is the integrator propelling the organism. Runs made
before this default changed record `baumgarte = 0.2` in their own `config.toml`,
so they still resume and still reproduce — but their distances should be read as
upper bounds. `examples/leak_probe.rs` will say how much of any given champion
was real.

### Sensing the ground

A sensor is **carried by a part**, not bolted to one. The part has mass, hangs
off a joint, and is aimed by whatever drives that joint — so an organism that
grows a stalk can point it, using the same controller outputs that would
otherwise swing a leg, and pays for it in weight and actuation exactly as it pays
for any other limb. Perception is not free, and on a project about embodied
evolution a weightless sensor would have been the anomaly.

The first kind is a lidar-like range sense: a fan of rays cast against the
terrain, each returning `1 - distance / range`, so 1 is a surface at the sensor
and 0 is nothing within reach.

```toml
[body]
sensor_probability = 0.35   # chance a part carries one; 0 disables sensing exactly

[sensor]
range = 4.0     # how far a ray reaches
rays = 3        # readings per sensor, fanned about the aim
spread = 0.35   # angular spread between adjacent rays

[mutation]
sensor_rate = 0.04
sensor_dir_sigma = 0.15
```

`experiments/sensing-climbers.toml` combines this with the elevation terms below.

**What one ray buys, and what two do.** Ground that rises ahead returns a
*shorter* range than level ground; ground that falls away returns a longer one,
or nothing. So a single ray already separates climbing from falling. Telling a
gentle slope from a wall needs at least two, because one distance carries no
gradient.

The sensor deliberately does not classify anything. Whether a slope is climbable
is a property of the body and controller meeting it — a small weak organism and a
large strong one get different answers from the same ground, and neither can know
which it is without trying. The sensor reports geometry; what that geometry means
is for evolution to discover, per lineage.

**Resolution is a stated physical property.** Rays march at a fixed 100 mm
stride, refined by bisection, so terrain that rises above the ray and drops back
within one stride is invisible. The stride is chosen against the terrace risers
the fractal field produces, which measure about 152 mm. Resolution follows the
stride rather than a step count, so a longer-sighted experiment does not quietly
become blind to cliffs. Sensing costs about 23% of evaluation throughput at three
rays.

The default is `sensor_probability = 0`, and off is exact: no sensor gene is
drawn, so the random stream is untouched, the controller keeps the input count it
had, and every earlier result reproduces bit for bit.

### Scoring elevation, not just distance

Distance is the first fitness criterion, not the only one. Four terms score what
an organism did with its height, and all of them default to zero.

```toml
[fitness]
objective = "distance_x"

climb_bonus = 10.0        # per metre ended above the settled start
descent_penalty = 4.0     # per metre ended below it

cumulative_climb_bonus = 0.0      # per metre of total ascent
cumulative_descent_penalty = 0.0  # per metre of total descent
climb_deadband = 0.05             # metres of movement before either registers
```

The first pair is **net**: where the organism finished, relative to where it
settled. Clamped per trial and only then averaged, so climbing on one trial and
falling on another reports both rather than netting to nothing. It cannot be
farmed — the only way to raise it is to end higher.

The second pair is **cumulative**: total ascent and descent over the run, so a
hill climbed and descended still counts. That is richer and it is the pair that
needs watching, because anything paying per unit of vertical movement invites an
organism to bob on the spot. `climb_deadband` is a hysteresis band, not a
per-step threshold: the reference height moves only when a move registers, so a
slow drift still accumulates while a gait's wobble never leaves the band.
Measured on organisms evolved under a pure distance objective, an honest gait
produces at most 0.05 m of incidental ascent over a whole run, which is where the
default came from. `Config::validate` refuses a cumulative term with no band.

Two cautions, both of them measured rather than theoretical.

**An organism that never moves loses no elevation**, and unlike `energy_penalty`
it is not even charged for standing there. If `descent_penalty` outweighs what
distance pays, the best strategy is to do nothing — keep a distance term in the
objective. `standing_still_does_not_beat_travelling` is the gate.

**A climb bonus is worth nothing on ground with no reachable climb.** Across 150
generations of the shipped fractal experiment, not one organism ever ended higher
than it started, so `climb_bonus` changes no score and no ranking there at any
value. Check what climb is actually reachable before tuning the weight.

### Re-scoring a finished run

Because full metrics are stored for every organism and `fitness::score` reads
nothing else, a completed run can be scored under a different question without
re-simulating anything:

```bash
# What would this population have looked like under a descent penalty?
./target/release/evo rescore runs/<run> --descent-penalty 8 --tail 10

# And which organisms would that have promoted?
./target/release/evo rescore runs/<run> --descent-penalty 8 --show-generation 149
```

It reports the best and median under both scorings and how much of the top ten
survives the re-weighting — a scoring that reorders nobody is not asking a new
question. With no weights overridden it is a round trip and must reproduce the
recorded fitness exactly, which is what makes the rest of its output worth
reading.

Runs that finished before elevation was recorded can still be re-scored on the
net terms: `start` and `end` were always stored, so the pair is recovered from
them, approximately for multi-trial runs because the stored endpoints are already
averaged.

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
two-part generation-0 replay, or use the file picker — or drag and drop — to
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

**Phase 0 of [ROADMAP.md](ROADMAP.md) is complete: the pipeline works end to end,
evolution demonstrably occurs, and it reproduces bit for bit.**

A first run of `experiments/first-walkers.toml` — 100 organisms, 100 generations,
seventeen seconds of wall clock — took best fitness from 0.79 m to 4.76 m and the
*median* from 0.08 m to 3.20 m, with no divergent simulations and 77 of 100
morphologies still distinct. The rising median is the part that matters: with
elitism the best score cannot fall, so only the middle of the distribution moving
shows the population as a whole is improving. Omnidirectional `distance` can be
satisfied by tumbling; `experiments/directed-walkers.toml` asks for signed +X
progress and an upright bonus.

Everything since has widened what a body can be and what the ground can do —
five part shapes, bilateral symmetry, segmentation, muscle-limited torque,
tendons, self-collision, joint wear, and seeded fractal terrain with hills and
cliffs. What has *not* widened much is the question being asked, which is what
Phases 1 and 2 of the roadmap are about.

Implemented:

* organisms of jointed primitive parts, with limits, motors and optional tendons
* a small hand-written feed-forward controller, evolved rather than trained
* a purpose-built impulse-based rigid-body solver — gravity, ground contact,
  friction, joints, joint limits, joint motors, optional self-collision
* flat, sine and seeded fractal terrain behind one analytic height function
* repeated trials with per-trial start, terrain and commanded-heading variation
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

Not built: discrete obstacles, sensors that perceive the world outside the body,
tasks other than travelling, a renderer, evolved network topology, and any cloud
infrastructure. The first three are on [ROADMAP.md](ROADMAP.md); see
[ARCHITECTURE.md](ARCHITECTURE.md) for the reasoning behind the rest, the
assumptions that would affect scaling, and where the implementation is meant to
be replaced.

The A/B that the solver bug invalidated has been rerun at 30 generations, across
three grounds rather than two — see [What each ground selects for](#what-each-ground-selects-for).
Scores now fall monotonically with difficulty and the corpse gate passes on every
arm. A longer replicated run is still wanted before the part-count result is
treated as established.

## License

[MIT](LICENSE).
