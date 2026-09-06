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

* Rigid blocks
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
