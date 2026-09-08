# EvoForge — Experiment Configuration Guide

This document covers everything that goes into a TOML experiment file. For measured results from real runs see [RESULTS.md](RESULTS.md). For the complete field list with types and defaults, read [`src/config.rs`](src/config.rs) — it is the canonical source and is always current.

The load-bearing rule throughout: **off is exact.** A feature disabled by setting its rate, probability, or weight to zero must not consume any randomness, must not change the controller's input count, and must not alter any other experiment's output. Enabling a new feature on an existing config is always a new experiment, not an extension of the old one.

---

## Experiment Catalog

| Experiment | What it demonstrates |
|---|---|
| `first-walkers.toml` | Minimal setup: flat terrain, distance objective, 100 organisms |
| `directed-walkers.toml` | Signed +X progress and an upright bonus — harder to satisfy by tumbling |
| `shaped-walkers.toml` | `directed-walkers` with all five part shapes enabled |
| `animals.toml` | Every animal-oriented constraint at once: symmetry, muscle torque, tendons, self-collision, rough terrain, three trials, commanded headings |
| `fractal-animals.toml` | `animals` on seeded fractal terrain with hills, cliffs, and per-trial variation |
| `brittle-walkers.toml` | `shaped-walkers` with joint wear: motors can fail mid-trial |
| `jumpers.toml` | `distance_x` objective with `air_bonus` and `height_bonus` — rewards leaving the ground |
| `sensing-climbers.toml` | Fractal terrain with elevation scoring and a one-ray lidar sensor; best settings from the 300-generation comparison |

Run any of them with:

```bash
./target/release/evo run experiments/<name>.toml
```

---

## Part Shapes

A part is not necessarily a cuboid. Each one is *carved* from the box its `half_extents` describe, as one of five primitives:

| `shape` | what it is | ground contact |
|---|---|---|
| `box` | fills the box | 8 corners |
| `taper` | rectangular frustum along the box's longest axis — a wedge, a foot, a claw | 8 corners |
| `sphere` | inscribed sphere | 1 point |
| `capsule` | inscribed capsule along the longest axis | 2 points |
| `cylinder` | inscribed cylinder along the longest axis | 3 points per rim |

The gene is only the *kind*; dimensions always come from `half_extents`. Shape matters because of the contact column — a capsule rolls and pivots; a sphere or cylinder can roll outright, which is a gait that was not previously available.

Enable them per experiment:

```toml
[body]
shapes = ["box", "taper", "sphere", "capsule", "cylinder"]
taper_top_scale = 0.45   # a taper's far end, as a fraction of its base

[mutation]
shape_rate = 0.04        # per part, per generation
```

The default is `shapes = ["box"]`. A single-entry roster spends *no randomness* choosing, so a box-only experiment draws the identical random stream it drew before shapes existed and reproduces earlier results bit for bit.

---

## Toward Animals

Animals are not the product of being scored on looking like animals. They are what falls out of three pressures at once, and each is a lever here — none of them a fitness term that mentions a leg. Every one is off by default, and off is exact.

### Bilateral symmetry and segmentation

A `PartGene` can be `paired`, in which case it appears twice as mirror images across the sagittal plane — and both copies share one controller slot, so two legs are driven by one leg controller and move as a pair, which is what a gait is. `antiphase` chooses whether the halves alternate (a walk) or move together (a bound). A part can also `repeat` into a chain of segments; children hang off the end rather than sprouting from every segment.

```toml
[body]
pair_probability = 0.45   # 0 disables bilateral symmetry entirely
max_repeat = 3            # 1 disables segmentation
```

### Muscle-limited torque

`muscle_stress` caps a joint's torque at what its own girth could physically host — muscle force scales with cross-section, and the torque it exerts with a moment arm that grows with the limb's width, so the ceiling goes as `stress * area^1.5`. Without it `motor_torque` is drawn independently of size.

```toml
[body]
muscle_stress = 12000.0   # 0 leaves motor_torque a free gene
```

### Tendons

`tendon_frequency` gives every hinge a passive spring, expressed as a natural frequency rather than a stiffness so that it means the same thing on a thigh and on a toe.

```toml
[body]
tendon_frequency = 6.0    # rad/s; 0 for no tendon
tendon_damping = 0.5      # damping ratio
```

### Self-collision

Stops an organism's parts passing through each other, making limbs be outside the torso rather than inside it. Parts are approximated by capsules; jointed pairs are exempt, because they are meant to touch.

```toml
[environment]
self_collision = true
```

### Repeated trials and commanded headings

Repeating each evaluation from varied starts means a single well-timed lunge no longer scores like a gait. A commanded heading, given to the controller as an input and scored along that direction, means an organism has to be steerable.

```toml
[simulation]
trials = 3                # 1 is a single trial
start_jitter = 0.5        # how much the start pose varies
aggregate = "mean"        # or "worst", which asks for no bad day at all
steer = true              # each trial commands a direction
steer_spread = 0.9        # radians either side of +X

[fitness]
objective = "heading"     # distance along the commanded direction
```

`experiments/animals.toml` turns all of it on at once. It is slow — three trials, self-collision and twelve solver iterations cost roughly an order of magnitude against `directed-walkers` — and it does not yet produce quadrupeds.

---

## Ground Types

### Rough terrain

Two octaves of a sine field. It repeats every wavelength, has no seed, and has features at one scale only, so every organism in every trial of every experiment meets the same ripple.

```toml
[environment]
terrain = "rough"
terrain_amplitude = 0.05
terrain_wavelength = 1.6
```

### Fractal terrain

Four bands of seeded gradient noise (fractional Brownian motion with terracing):

| band | what it does |
|---|---|
| landscape | hills, tens of metres across and metres deep |
| detail | ground texture at the scale of an organism's own body |
| modulation | a slow field saying where the ground is calm and where it is savage |
| terracing | quantises height to steps, turning hillside into plateau-and-cliff |

Terracing is the one that makes a sheer face: the riser is steeper than the underlying slope by exactly `1 / terrain_riser`. Measured on shipped settings with `cargo run --release --example terrain_probe`: 6.3 m of relief, median slope 8 degrees, 99th percentile 79, maximum 87, **93% of the plane walkable**, and only 1 straight 20 m crossing in 400 that avoids meeting a wall.

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

`terrain_riser` has a floor that is physics rather than taste — a wall thinner than a few integration steps is not a cliff but a tunnelling bug, so `Config::validate` measures the walls a config would produce and refuses the ones that are too thin.

The viewer mirrors the height field in JavaScript. Every non-flat trace records the physics' own height at sixteen points and the viewer checks itself against them on load. Offline:

```bash
cargo run --release --example terrain_samples > samples.json
node viewer/terrain_check.mjs samples.json
```

---

## Rewarding a Jump

Two optional fitness terms ask for locomotion that leaves the ground:

* `air_bonus` — per second with **every attached part** clear of the ground.
* `height_bonus` — per metre the centre of mass rises above where it started.

Neither works alone. Hang time by itself rewards a long low skim; height by itself rewards rearing up without ever taking off. Together, with distance still the base objective, they ask for locomotion that leaves the ground.

"Clear of the ground" means a real margin — `AIRBORNE_CLEARANCE`, two centimetres — not merely "no contact". A contact exists only once a point is *below* the terrain, so on the naive definition a body hovering a millimetre above it counts as flying. Asked for hang time that way, evolution needed forty generations to produce organisms spending half the trial "airborne" while never rising above the grass. The fix was to make the word mean what it says.

```toml
[fitness]
objective = "distance_x"
air_bonus = 3.0
height_bonus = 8.0
```

`experiments/jumpers.toml` combines those with durable joints (`joint_endurance = 150`). A 430-generation run reached fitness 19.0 with a four-part organism that travels 9.5 m while making six distinct hops, 1.67 s of it airborne, peaking 17 cm off the ground.

Both terms default to zero, so every experiment that predates them is unaffected.

---

## Joints That Wear Out

With `body.joint_endurance` set, exceeding a joint's `motor_torque` finally costs something. While a motor is **saturated** — the controller asking for a speed the joint cannot reach at its torque — the joint loses health equal to the rotation it fell short by, in radians. At zero health the joint fails: it stops constraining anything, and everything hanging below it falls away.

Detached parts keep tumbling in the world but stop counting toward the centre of mass that fitness measures. Losing a limb costs you its usefulness, not a phantom position penalty from where the wreckage lands.

The organism gets both halves of the trade-off:

* **Sensing** — each slot gains a controller input carrying that joint's remaining health, so an organism can feel a joint going and ease off it.
* **Disposition** — a heritable `caution` gene in `[0, 1]` throttles how hard every motor is driven.

`Metrics` gains `joints_lost`, recorded for every organism, so breakage can be measured across a whole run without opening a replay.

```toml
[body]
joint_endurance = 25.0   # radians of undelivered rotation a joint survives; 0 = off
min_drive = 0.2          # however cautious it gets, it can still move this hard

[mutation]
caution_rate = 0.08
caution_sigma = 0.12
```

`experiments/brittle-walkers.toml` is `shaped-walkers.toml` with exactly that added. Note that `joint_endurance = 25.0` is *aggressive*: a 120-generation run at that setting averaged 2.5 lost joints per organism, and only 3 of 48 recorded organisms finished intact. Raise it for a gentler world.

The default is `joint_endurance = 0`, which disables wear entirely — and disabling it is exact rather than approximate: no randomness is spent on the caution gene, and the controller keeps its original input count, so every earlier result reproduces bit for bit.

---

## Sensing the Ground

A sensor is **carried by a part**, not bolted to one. The part has mass, hangs off a joint, and is aimed by whatever drives that joint — so an organism that grows a stalk can point it, using the same controller outputs that would otherwise swing a leg, and pays for it in weight and actuation exactly as it pays for any other limb. Perception is not free.

The first kind is a lidar-like range sense: a fan of rays cast against the terrain, each returning `1 - distance / range`, so 1 is a surface at the sensor and 0 is nothing within reach.

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

**What one ray buys, and what two do.** Ground that rises ahead returns a *shorter* range than level ground; ground that falls away returns a longer one, or nothing. So a single ray already separates climbing from falling. Telling a gentle slope from a wall needs at least two, because one distance carries no gradient.

The sensor deliberately does not classify anything. Whether a slope is climbable is a property of the body and controller meeting it — the sensor reports geometry; what that geometry means is for evolution to discover, per lineage.

**Resolution is a stated physical property.** Rays march at a fixed 100 mm stride, refined by bisection, so terrain that rises above the ray and drops back within one stride is invisible. The stride is chosen against the terrace risers the fractal field produces, which measure about 152 mm. Sensing costs about 23% of evaluation throughput at three rays.

The default is `sensor_probability = 0`, and off is exact: no sensor gene is drawn, the random stream is untouched, the controller keeps the input count it had, and every earlier result reproduces bit for bit.

`experiments/sensing-climbers.toml` combines the sensor with elevation scoring. See [RESULTS.md](RESULTS.md) for the 300-generation comparison.

---

## Scoring Elevation

Four terms score what an organism did with its height. All default to zero.

```toml
[fitness]
objective = "distance_x"

climb_bonus = 10.0        # per metre ended above the settled start
descent_penalty = 4.0     # per metre ended below it

cumulative_climb_bonus = 0.0      # per metre of total ascent
cumulative_descent_penalty = 0.0  # per metre of total descent
climb_deadband = 0.05             # metres of movement before either registers
```

The first pair is **net**: where the organism finished, relative to where it settled. Clamped per trial and only then averaged, so climbing on one trial and falling on another reports both rather than netting to nothing. It cannot be farmed — the only way to raise it is to end higher.

The second pair is **cumulative**: total ascent and descent over the run. A hill climbed and descended still counts. That richness is the pair that needs watching, because anything paying per unit of vertical movement invites an organism to bob on the spot. `climb_deadband` is a hysteresis band, not a per-step threshold: the reference height moves only when a move registers, so a slow drift still accumulates while a gait's wobble never leaves the band. `Config::validate` refuses a cumulative term with no band.

`fall_penalty` is separate and targeted:

```toml
[fitness]
fall_penalty = 8.0        # per metre lost while out of contact with the ground
```

It charges only for height given away *while no attached part is touching the ground* — the part of a descent the organism did not choose. Walking down a slope keeps contact and costs nothing; stepping off a terrace does not. See [RESULTS.md](RESULTS.md) for the standing-still hazard that `descent_penalty` produces and why `fall_penalty` avoids it.

**A climb bonus is worth nothing on ground with no reachable climb.** Across 150 generations of the shipped fractal experiment, not one organism ever ended higher than it started, so `climb_bonus` changes no score and no ranking there at any value. Check what climb is actually reachable before tuning the weight.

---

## Re-Scoring a Finished Run

Because full metrics are stored for every organism and `fitness::score` reads nothing else, a completed run can be scored under a different question without re-simulating anything:

```bash
# What would this population have looked like under a descent penalty?
./target/release/evo rescore runs/<run> --descent-penalty 8 --tail 10

# Which organisms would that have promoted?
./target/release/evo rescore runs/<run> --descent-penalty 8 --show-generation 149
```

It reports the best and median under both scorings and how much of the top ten survives the re-weighting — a scoring that reorders nobody is not asking a new question. With no weights overridden it is a round trip and must reproduce the recorded fitness exactly.

Runs that finished before elevation was recorded can still be re-scored on the net terms: `start` and `end` were always stored, so the pair is recovered from them, approximately for multi-trial runs because the stored endpoints are already averaged.
