# EvoForge roadmap

What this project is building toward, in the order it intends to build it. For
what exists today see [README.md](README.md); for why the code is shaped the way
it is, [ARCHITECTURE.md](ARCHITECTURE.md).

This is a direction, not a schedule. Each phase names a seam that already exists
in the code and says what has to change at it. Nothing here is committed to a
date, and later phases are deliberately sketched rather than specified — the
evidence that decides them has not been gathered yet.

## The through-line

```text
evolved locomotion  ->  richer evaluation  ->  varied environments and tasks
                                                          |
                                                          v
   bodies, controllers and environments together  <-  sensors
```

An organism is a physical body of jointed primitives, driven by a small evolved
neural controller. It is placed in an environment, allowed to act, scored, and
the best are bred. The only thing that changes across this roadmap is **what it
is scored on, and what it can perceive while doing it.**

Today an organism is scored on how far it travels. That is criterion number one,
chosen because it is the simplest thing that can distinguish a body that does
something from a body that does nothing — not because distance is what the
project is about. Everything in Phase 1 and Phase 2 exists to make the evaluation
system capable of asking harder questions: go uphill, go downhill, reach that
beacon, reach as many beacons as you can. Phase 3 gives an organism the means to
perceive what it is being asked about. Phase 4 is what happens when body,
controller and sensing are all under selection at once.

## Unexpected behaviour is the point

Evolution under a simple objective finds simple answers, and they are frequently
not the answers anyone had in mind. An organism that discovers a way to score
well that nobody anticipated has done its job. **This roadmap contains no work
whose purpose is to prevent that.**

What it does contain is the machinery to tell three different things apart,
because from a fitness curve they look identical:

| | What happened | What it demands |
|---|---|---|
| **Simulation fault** | The score required the simulator to violate the physics it claims to implement. A body gained energy nothing supplied. | A bug fix and a standing test. Not a fitness change. |
| **Measurement fault** | The physics was right, but the metric did not measure what its name says. | Repair the instrument. Still not a fitness change, and never a penalty. |
| **Strategy** | Correct physics, honest measurement, and it scores anyway. Sliding down a hill; a single well-timed lunge; a gait nobody would design. | Nothing. It is a result. If we wanted something else, we should have asked a different question. |

Both faults have occurred here and both were fixed at the level they occurred.
The self-collision conveyor — organisms crossing 26 m with their motors switched
off — was a solver bug: positional correction was being added to real velocity
and kept. `self_collision_is_not_a_motor` and `a_dead_organism_does_not_travel`
are the gates that stop it coming back. Shedding a limb used to move the measured
centre of mass for free, worth 3.34 m to one organism; the fix was to cancel the
discontinuity in the measurement (`World::centre_of_mass`), so detaching a part
is now worth exactly zero metres rather than being forbidden. "Airborne" once
meant "not touching", which a body hovering a millimetre up satisfies;
`AIRBORNE_CLEARANCE` made the word mean what it says.

Note what none of those did: none of them made a behaviour illegal. An organism
may still fling a limb, still ride a slope down, still commit everything to one
launch. Those score what they genuinely earn under the objective in force.

**Three strategies currently in the repository, none of them a problem.** The
same experiment run on three grounds produces three different answers to "travel
as far as you can", and all three are correct:

| ground | what evolves |
|---|---|
| flat | small two-part machines that *vibrate* along. Nothing about a plane and a distance objective asks for more. |
| sine | slightly larger bodies, visibly less vibration. A 13 cm ripple is enough to make buzzing stop paying so well. |
| terraced fractal | large bodies, near the eight-part maximum. About half of them move *just enough to fall off a nearby drop* and then stop; the rest move steadily and some net climb. On ground with 5.6 m of relief, falling is a sound reading of the question. |

The third is the interesting one, and the temptation is to call it cheating. It
is not: the physics is right, the measurement is honest, and falling down a hill
really is a way of covering ground. If what we want is *controlled* descent
rather than a well-aimed fall, then the objective has to be able to tell those
apart — which is Phase 2, not a penalty term.

Note also that it is not winning. The organisms that fall and stop score *below*
the ones that move steadily on the same ground, so this is a cheap local optimum
that caps out low rather than a strategy taking over the population. That is a
more interesting problem than an exploit, and a different one: the question is
whether a gait can be established before the cheap answer is discovered.

**Finding it needs the right statistic.** Net elevation change does not reveal
it — it is near-uniform across the population and correlates weakly with
fitness, so it reads as unremarkable. The signature is temporal: the share of
final displacement reached in the first half of the measured window. A gait
splits it about 50/50; a fall-and-stop puts 80% or more in the first half. The
same blind spot affects `dead_organism_probe` above, and for the same reason —
an organism that works its way to a lip and then falls is doing something no
motors-off corpse can imitate.

The diagnostic tools in [examples/](examples/) exist to make the distinction
decidable rather than arguable, and they carry forward into every phase below:

| Tool | Question it answers |
|---|---|
| `dead_organism_probe` | With the motors off, how far does this champion still travel? Whatever it covers dead, it did not earn. |
| `conveyor_probe` | Which property of the ground gives distance away — steepness, feature size, or number of scales? |
| `drift_probe` | Is a bigger body genuinely better here, or is it collecting more free ride per part? |
| `energy_probe` | Does a passive body ever end with more mechanical energy than it started with? |
| `terrain_probe` | Is the ground actually crossable, and how is its difficulty distributed? |
| `leak_probe` | Where does un-earned travel come from — internal forces, positional correction, or friction? |
| `refine_probe` | Which subsystem loses its travel when the solver is refined? |
| `friction_probe` | Does a resting body get the friction Coulomb says it is owed? |

**Refinement is the sharpest test there is**, and it deserves naming separately.
Real locomotion is a property of the equations, so solving them more accurately
perturbs a trajectory without systematically abolishing the distance covered.
Travel that exists only at a coarse solve is the integrator moving the organism.
Measured once: refining 12 solver iterations to 96 removed **94%** of an evolved
champion's travel, which is how the Baumgarte default was found to be propelling
a whole population. `travel_survives_refining_the_solver` is the standing gate.

They are observation instruments. Their output belongs in the description of a
result, not in a penalty term.

**A known blind spot, since it bears on how the corpse figure is read.**
`dead_organism_probe` measures free distance from where an organism *starts*. An
organism that spends a little actuation getting itself to a cliff edge and then
falls is doing something no motors-off corpse can imitate, because a corpse never
reaches the edge. The figure is therefore a lower bound on how much of a score
the terrain handed over, not a full accounting. It remains the right gate for
what it was built for — catching the simulator propelling things — and Phase 1's
task descriptions are what would let a metric distinguish the rest.

## Phase 0 — where we are

Evolved locomotion works end to end and is reproducible bit for bit.

Implemented: tree-structured bodies of five primitive shapes; fixed and hinged
joints with limits, motors, optional passive tendons, and optional wear leading
to failure; optional bilateral symmetry and segmental repetition; optional
self-collision; a fixed-topology feed-forward controller evolved rather than
trained; flat, sine and seeded fractal terrain behind one analytic height
function; four objectives over a recorded metric set; repeated trials with
per-trial start, terrain and heading variation; tournament selection with
elitism, slot-aligned crossover and mutation; checkpoint, resume and extension;
selective recording and exact re-simulation; a browser replay viewer.

Two items carried forward from the terrain work, both stated in
[TERRAIN_PLAN_2.md](TERRAIN_PLAN_2.md):

- **The terrain A/B has been rerun once, briefly.** The 80-generation comparison
  of sine against fractal ground was voided by the solver bug, which the fractal
  run happened to amplify. A 30-generation, three-arm rerun (flat, sine, fractal)
  now scores monotonically with difficulty — 9.18, 7.53, 3.97 — and passes the
  corpse gate on every arm, so the comparison is valid again. What is still
  outstanding is a longer replicated run: thirty generations on one seed is a
  direction check, and the part-count result below deserves more than that. See
  [README.md](README.md) for the full figures.
- **Discrete obstacles were deferred** and remain the only proposed feature that
  produces a discontinuity at or above a wheel's own radius. Picked up in
  Phase 2.

## Phase 1 — evaluation as a first-class concept

**The seam already exists and is called `steer`.** A trial draws a commanded
heading, hands it to the controller as two inputs, and scores displacement along
it. That is the general shape of a task: *a per-trial parameter of the world, an
input the organism can act on, and a metric that says how well it did.* Beacons,
slopes and everything else in Phase 2 have that same shape. What this phase does
is stop it being a special case.

The boundary that must survive: `fitness::score` reads `Metrics` and nothing
else. An objective that can reach into the physics world acquires dependencies on
solver details, and results stop being comparable across solver changes.

Work:

- **Name the task.** Give a trial an explicit description of what was asked of it
  — today a commanded heading, later a target position or a sequence of them.
  Record it in the trace and in `organisms.jsonl` next to the metrics, so a
  result is self-describing: what the organism was asked, and what it did.
- **Widen `Metrics` where evaluation needs it.** It is already the sole input to
  scoring and is already recorded per organism. Elevation gained and lost, and
  distance to a target, are the two additions Phase 2 needs. The elevation pair
  is planned in [FITNESS_PLAN.md](FITNESS_PLAN.md), together with the weighting
  and the re-scoring tool below.
- **Re-scoring without re-simulating.** Because full metrics are stored for every
  organism, an existing run can be scored under a different objective after the
  fact. This is stated as a design intent in `fitness.rs` and is not yet
  reachable from the command line. It is the cheapest possible way to ask "what
  would this population have looked like under a different question", and it is
  how a new objective gets a sanity check before anything is bred under it.
- **Multiple tasks per experiment, sequentially.** An experiment already runs
  several trials and aggregates them. Letting different trials pose different
  tasks — a mean or a worst case across them — is the smallest change that makes
  evaluation multi-environment. This is not multi-objective optimisation: the
  result is still one scalar per organism.

Exit test: a task can be added without touching the physics, the genome, or the
breeding loop.

## Phase 2 — environments and tasks

The environment already varies — three terrain models, per-trial offset and
rotation of the field, spawn search. What it does not yet do is *ask for anything
in particular*.

- **Uphill and downhill.** The cheapest new tasks available: the fractal
  landscape already produces 5–6 m of relief, and net elevation change over the
  measured window is one subtraction. Posing it properly means placing the
  organism deliberately relative to the local gradient rather than wherever the
  spawn search lands, and commanding a direction with respect to the slope.
  Downhill is deliberately included even though it is the easier problem — an
  organism that can only fall down a hill and an organism that can descend under
  control both score well on distance, and the point of the task is to build the
  measurement that separates them. This is no longer hypothetical: on the
  terraced landscape the current objective already selects for bodies that move
  just far enough to fall off something, which is the clearest demonstration
  available that distance alone cannot ask the question we want.
- **Beacons.** A target position in the world; the metric is closest approach, or
  whether it was touched. **A beacon needs a sensor that can detect it**, and it
  should wait for one rather than being handed to the controller as a free input.
  An earlier draft of this roadmap proposed exactly that — the beacon's relative
  position given the way the commanded heading is given — on the grounds that it
  decoupled this phase from Phase 3. It does, and it also hollows the task out:
  an organism told continuously where the target is has not found the target, and
  what evolves is a policy for consuming a coordinate rather than anything that
  could be called seeking. See *Sensing must be sensed*, below.

  **A beacon is made detectable by being tall.** It stands high enough that a
  ray clears the terrain and meets it, so visibility is a physical fact about
  where the organism is standing rather than a fact about what the simulator
  chose to tell it: a beacon behind a ridge is invisible until the ridge is
  crested, and one across a valley is visible from the far rim. Beacon height
  against terrain relief is then the difficulty knob, and the sensor's range is
  the other. Detection is a *signature* the same ray returns alongside distance,
  so no separate beacon sense exists and one organ perceives both ground and
  target. Beacons need no collision — they are markers, not walls — so this does
  not wait on the discrete-obstacle work. See [SENSOR_PLAN.md](SENSOR_PLAN.md).

  **A beacon task and a large descent penalty are in direct conflict**, and the
  weights have to be settled before either is run. On a landscape with 5.6 m of
  relief roughly half of all beacons are below the organism, and at
  `descent_penalty = 8` descending a single metre costs 8 fitness — more than the
  best organism of the 45-generation penalty run scored in total. Every downhill
  beacon would be correctly ignored. The resolution is not to raise the beacon
  reward until it drowns the penalty out, but to notice that the penalty was
  always a proxy for *falling* and that `net_loss` does not measure falling: it
  charges a controlled walk downhill exactly what it charges a tumble.
- **Multiple beacons.** How many can be reached in one run. Needs an ordering or
  consumption rule and a count in `Metrics`, and little else. It is the first
  task where an organism must do something, finish, and then do something
  different — a different demand on a controller than any objective currently in
  the codebase.
- **Discrete obstacles.** Inherited from both terrain plans. A height field is
  single-valued and smooth: no overhangs, no gaps, no true vertical, and a
  measured ceiling near 88 degrees with a contact solver unreliable above about
  75. Seeded static bodies with `inv_mass = 0` are the only proposal that
  produces a step taller than a wheel's radius. Most of the machinery exists —
  capsule-capsule contacts are already there for self-collision — and the care
  needed is keeping obstacles out of everything that assumes a body belongs to
  the organism: `centre_of_mass`, `body_slots`, `detached`, the spawn drop.

**Scheduling difficulty within a run.** The terraced landscape offers a cheap
answer — move a little, fall off something, stop — that is reachable in very few
generations and caps out below real locomotion. Ramping the terrain's sharpness
across a run, so a gait is established before that answer becomes available, is
planned in [CURRICULUM_PLAN.md](CURRICULUM_PLAN.md). It is an enabler rather than
a task: it changes when the existing objective gets hard, not what it asks for.
The plan also settles why the config-digest check that blocks a staged resume
should be left alone rather than given an escape hatch.

Standing gates for this phase, because harder ground raises the price of honest
locomotion and therefore the relative value of anything free: the corpse gate
(motors off, under 2 m in eight seconds), the energy gate (peak mechanical energy
under 1.1x start), and the crossability gate (at least 85% of the plane under 40
degrees).

## Phase 3 — sensors

### Sensing must be sensed

**Anything an organism knows about the world outside its own body must arrive
through a sensor that has a position and an orientation on that body.**
Information about the world is never handed to the controller as a free input.

This is the whole point of the phase rather than a stylistic preference. The
project's question is what physical structure and control an environment selects
for *when the organism has to perceive that environment to succeed in it*. An
organism given the answer directly is not solving that problem — it is executing
a policy over a coordinate that something outside it computed, and whatever
evolves says nothing about perception. Worse, the shortcut is invisible in the
results: the fitness curve of an organism handed the answer looks exactly like
the fitness curve of one that found it.

The rule is cheap to state and easy to violate accidentally, because a free input
is always the quickest way to make a task work. Two rules of thumb:

* **If it changes as the organism moves, it is perception**, and needs an organ.
  A beacon's bearing, the height of the ground ahead, the distance to an
  obstacle: all of these update continuously as the organism acts, and all of
  them need a sensor.
* **If it is fixed for the whole trial, it is an instruction**, and may be given.
  `COMMAND_X`/`COMMAND_Z` is the existing case: the experiment says "go that
  way" once, at the start, and never revises it. That is a cue given to a trained
  animal, not a sense.

**An honest audit of the inputs that already exist**, because the rule is worth
nothing if it is only applied to new work:

| input | what it is | verdict |
|---|---|---|
| `BIAS`, `CLOCK_A`, `CLOCK_B` | an internal pacemaker | not a sense; fine |
| `UP_Y`, `RIGHT_Y` | orientation with respect to gravity | vestibular; fine |
| joint angle, ground contact, joint health | the body's own configuration | proprioception; fine |
| `COMMAND_X`, `COMMAND_Z` | the commanded heading | an instruction, fixed per trial; fine |
| `VEL_X`, `VEL_Y`, `VEL_Z` | world-frame velocity of the root | arguable — an animal senses acceleration and optic flow, not world-frame velocity |
| `HEIGHT` | `root.pos.y`, absolute altitude | **fails the rule.** No organ reports altitude above an arbitrary datum. On flat ground it happens to equal clearance; on the fractal landscape it tells the organism where it is in the terrain, which is exactly the kind of free world knowledge this rule exists to stop. |

`HEIGHT` is left as it is for now, deliberately: changing it alters the dynamics
and moves every golden constant, so it is a versioned decision rather than a
tidy-up. The honest options are to replace it with height above the ground
directly beneath — still exteroception, but at least something an organ could
measure — or to delete it and let a sensor supply it. Whichever is chosen, it
should be chosen rather than inherited.

**A distinction worth making before any of this is built.** The controller
already has inputs, and they are all *proprioceptive*: a bias, two clock waves,
two orientation components, three velocity components, height, and per joint an
angle, a ground-contact flag and optionally remaining health. They describe the
organism to itself. They are fixed for an experiment, identical for every
organism in it, and not described by the genome at all.

Sensors, in the sense this phase means, are *exteroceptive*: they report
something about the world outside the body. The conceptual goal is that sensing
is part of the organism rather than an external channel — a sensor is mounted on
a part, moves with that part, and if that part is on a jointed limb then the
controller is already aiming it, for free, using machinery that exists.

**The constraint that shapes all of it.** Input count sets the controller's
weight-vector length, which sets how much randomness a genome consumes, which
sets whether earlier results still reproduce. Every optional input so far is
conditional for exactly this reason — `INPUTS_PER_SLOT_WITH_HEALTH`,
`GLOBAL_INPUTS_WITH_COMMAND`. So sensors need a **budget** sized per experiment,
the way `max_slots` sizes joints today: an experiment declares how many sensors
of what kind an organism may carry, the controller layout is sized for that, and
an organism carrying fewer feeds zeros into the rest. An experiment that declares
no sensors must draw the identical random stream it drew before sensors existed.

In order:

- **Lidar-like.** A small fan of rays cast from a mounted part, returning
  distance to the ground along each. Planned in
  [SENSOR_PLAN.md](SENSOR_PLAN.md). Cheap, and well matched to what already
  exists: `TerrainModel` is one analytic function over an unbounded domain with
  an exact gradient, so a ray meets it by marching rather than by touching a
  mesh, and there is no transcendental anywhere in it — which keeps the
  determinism story intact. First because it is the least machinery for the most
  information.
- **Camera-like.** A low-resolution directional sample of what lies along a
  bearing. What "what" means is the open question, and should be answered by the
  task that first needs it — a beacon task needs to see beacons, an obstacle
  course needs to see obstacles. Not specified further here.
- **Hearing-like.** Requires sources in the world to hear. A beacon is the
  obvious first one. Later than the above, and deliberately left thin.
- **Other kinds** as tasks demand them, and not before.

Cost discipline: a sensor is per-step work in the innermost loop, on a workload
of millions of very short evaluations. Each kind gets measured against the
throughput it displaces, and each gets the same "off is exact" treatment as every
feature before it.

## Phase 4 — bodies, controllers and environments together

Sketched deliberately thinly. This is the direction the earlier phases imply, not
a specification.

With Phases 1–3 in place, what is under selection is a body, a controller, and
the placement and aim of whatever senses the world — evaluated against a set of
tasks rather than one. That is the point at which the project's actual question
becomes askable: what physical structure and control does a given environment
select for, when the organism has to perceive that environment to succeed in it?

Two seams already named in [ARCHITECTURE.md](ARCHITECTURE.md) become relevant
here and not before. Controller topology is fixed, which is what makes crossover
a one-liner; evolving it needs a different alignment scheme. Selection is
generational, which will want to become steady-state or island-based before it
distributes across machines.

The goal is **not** organisms that resemble animals, humans, or conventionally
designed robots. Resemblance is not a fitness term and never will be. Every
body-shaping rule in the codebase passes the same test — it must be justifiable
without reference to the shape it produces. Symmetry from developmental axes,
torque limits from muscle physiology, rough ground from ecology: all pass. "Must
have four legs" would not, and neither would a bonus for having them.

## Invariants every phase inherits

Each of these has been paid for here at least once.

- **Determinism is bitwise.** No `std` transcendentals, no ambient randomness, no
  iteration-order dependence. `tests/golden.rs` pins the contract against
  committed constants on three platforms. A golden that moves means the change is
  wrong, unless moving it is the deliberate, version-bumped decision.
- **Default off, and off is exact.** A feature that is not enabled must draw the
  identical random stream, keep the identical controller layout, and reproduce
  every earlier result bit for bit. Fold new fields into the config digest only
  when the feature is on.
- **Bump `ARTIFACT_FORMAT`** for anything that changes what is written, and give
  readers a defaulted fallback so older artefacts still load and still mean what
  they meant.
- **Evaluation stays pure.** A function of `(genome, config)` and nothing else.
  That is what makes evaluation parallel now, distributable later, and any
  organism reconstructable from its genome alone.
- **Anything that adds energy to the simulation gets an adversarial test first.**
  The tendon shipped unstable and produced organisms crossing 100 m in eight
  seconds. Sensors do not add energy; obstacles and any new contact path do.
- **Measure more than once.** Two conclusions in this project were drawn from
  single unreplicated measurements and were wrong. Benchmarks on a laptop vary by
  2x with power state.
- **Verify recorded output independently.** Reconstructing metrics from replay
  files with a separate implementation has caught real bugs more than once.

## Not on this roadmap

Named so the boundary is written down rather than assumed. These may be
interesting later; none of them is what the phases above are building toward, and
work should not be justified by appeal to them.

- Ecosystem simulation, predation, or organisms that interact with each other
- Multiple competing or co-evolving populations
- Multi-objective evolutionary algorithms and Pareto fronts
- Energy, metabolism or resource economies
- Gradient-based learning, reinforcement learning, or a machine-learning
  framework dependency
- GPU compute
- A real-time renderer, or a game
- Cloud infrastructure, until the simulator is fast and measured
