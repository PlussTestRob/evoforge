# Plan: the first sensor

Status: **built, stages 1-3.** Phase 3 of [ROADMAP.md](ROADMAP.md), which this
plan covers only the first entry of: a lidar-like range sense against the ground.

What landed, against the staging in §4:

1. `TerrainModel::raycast`, validated against a dense independent march over
   flat, sine and fractal ground — 1200 rays, agreeing to within the stated
   resolution, with sub-stride misses under 2%.
2. Sensors as **carried parts**: `PartGene` gains `sensor` and `sensor_dir`,
   `Phenotype` gains `SensorMount`, and readings arrive as per-slot inputs after
   the proprioceptive ones. Because a sensor rides the part that owns the slot,
   a sensor on a hinged limb is already aimed by the controller driving that
   hinge — stage 4 needed no separate work.
3. `[sensor]` config, `body.sensor_probability`, mutation operators, guarded
   fingerprint, `ARTIFACT_FORMAT` 9, and `experiments/sensing-climbers.toml`.

**One correction to §2 made while building it.** The march was specified as a
fixed *step count*, which gave a 375 mm stride at 6 m range — and stepped clean
over the 152 mm terrace risers that are the single most important thing to see.
Resolution now follows a fixed **stride** (`MARCH_STRIDE`, 100 mm) with the step
count derived from range and capped, so sight distance and sight resolution stop
being the same knob. Cost measured at **23%** on top of a fractal experiment
(44 s to 54 s for three generations), which is what §2 predicted.

Stage 5 (beacons and the signature channel) is not built, and §3b's argument
still stands: it should wait for a term that separates falling from descending.

Not yet done, and worth knowing before a long run: nothing records what a sensor
*read*, so §6's third success criterion cannot be checked yet. See §7's third
risk.

Goal: let an organism perceive the terrain around it, so that avoiding a drop can
be something it *does* rather than something its shape happens to cause.

---

## 0. What this is for

The descent-penalty run (`runs/ab3-descent-*`, 45 generations against a
distance-only control on the same seed) reversed an entrenched bias: from 0 of
100 organisms ever ending higher than they started, to 59 of 100. It worked.

But watching the replays says *how* it worked, and it is not what was wanted.
The population converged on three-part bodies that vibrate along level ground —
mean part count 3.02 against the control's 7.84, and 41 distinct structures
against 94. They score well by never going anywhere interesting. The ones that
avoid falling avoid it because of how they are built, not because they noticed
anything: **an organism with only proprioceptive inputs cannot perceive a drop,
so not falling into one can only ever be luck.**

That is the gap. Not the objective — the objective now asks the right question.
What is missing is any means of answering it deliberately.

**`descent_penalty = 8` is a deliberate instrument, not a mistake to be fixed.**
Its job is to make climbing the only way to score well, and it is doing that job:
before it, no organism in 15,000 had ever ended higher than it started. What it
cannot do is supply the *means*. The population it produced climbs by 0.14 m at
best, which is not climbing, it is failing to fall. The weight stays until the
organisms can act on it, and §3b's argument about beacons applies to beacons —
a later task — rather than to this one.

### What a range sense actually buys on this terrain

Measured with `terrain_probe` on the shipped settings: 6.3 m of relief, **median
slope 8 degrees**, 99th percentile 79, maximum 87, and 93% of the plane walkable.
So the landscape is mostly gentle, with the difficulty concentrated in a small
fraction of near-vertical risers — which is what terracing was built to produce.

That distribution is what makes a sensor worth having, and it says what the
sensor is *for*:

* **Most ground is climbable.** An organism does not need to scale an 80-degree
  wall to gain height; it needs to find the 8-degree slope and go up it. Climbing
  is physically available almost everywhere.
* **A single forward-and-down ray already discriminates rising from falling
  ground.** Ground that rises ahead returns a *shorter* range than level ground;
  ground that falls away returns a longer one, or nothing. That is exactly the
  signal a climb objective needs, from one ray.
* **More than one ray gives the network geometry rather than a number.** One
  range cannot distinguish an 8-degree rise from an 80-degree riser; two at
  different angles carry the slope between them. Note that the sensor must not
  try to *classify* anything: what counts as climbable is a property of the
  organism — its size, its torque, its gait, its controller — and not of the
  ground. A small weak body and a large strong one meet the same slope and get
  different answers, and neither can know which it is before trying. The sensor
  reports geometry; what that geometry means is for evolution to discover, per
  lineage.

**A swept single ray is not a substitute, and the reason is structural.** The
controller is fixed-topology feed-forward with no memory, so it cannot integrate a
sensor reading across time — a stalk sweeping an arc presents the network with one
unrelated number per tick and no way to assemble them into a picture. Spatial
sampling beats temporal sweeping until the controller is recurrent. Start with a
fan of two or three rays, and note this as another thing recurrence would unlock.

## 1. The rule this plan is written under

**Anything an organism knows about the world outside its own body must arrive
through a sensor with a position and an orientation on that body.** World
information is never handed to the controller as a free input.

This is stated in full in [ROADMAP.md](ROADMAP.md) under *Sensing must be
sensed*, and it is repeated here because it is the constraint that makes this
plan more expensive than it would otherwise be, and because the cheap violation
is always available and always tempting.

**A rejected first draft of this plan, recorded so it is not re-proposed.** The
obvious way to start is to skip the sensor and sample the terrain height at three
fixed distances ahead of the organism, handing the results over as global inputs.
It is about thirty lines, it costs nothing measurable, and it would answer
"does terrain information change what evolves" in one run.

It is also exactly the shortcut the rule forbids, and the reason is not
bookkeeping. An organism handed the profile of the ground ahead has not perceived
the ground; something outside it did, and what evolves is a policy over a number
rather than an organism that senses. The result would look identical in
`stats.csv` to one obtained honestly, which is what makes it corrosive rather
than merely inelegant — nothing downstream could tell the two apart.

The cost of obeying the rule is that the first useful experiment is bigger. That
is the correct price.

## 2. What a ray costs here, and why this sensor is the cheap one

Lidar is first among the sensor kinds because the terrain is already the right
shape for it. `TerrainModel` is a single analytic function over an unbounded
domain — `height_at(x, z)`, with `sample()` returning value and exact gradient
together — with no mesh, no chunks and no acceleration structure. A ray meets it
by marching, and marching is a loop over a function the physics already calls
tens of thousands of times per trial.

**The march.** Step along the ray in fixed increments until `ray.y - h(ray.x,
ray.z)` changes sign, then bisect a few times to refine the crossing. Sixteen
coarse steps and four bisections gives about 20 samples per ray.

**Fixed iteration counts, always.** Not "march until you hit": a data-dependent
loop makes cost vary per organism and per terrain, which turns a benchmark into a
function of what evolved. Fixed counts also keep the work identical on every
thread, which is what the determinism contract needs.

**No transcendentals.** Rotating the ray by the mounting part's orientation is
quaternion multiplication, and the height field is integer hashing and polynomial
arithmetic. Nothing here touches `sin`, `cos` or `atan2`, so the reproducibility
story is unchanged rather than merely preserved.

**Cost.** Control runs at 20 Hz against physics at 120 Hz, so a trial has about
160 control ticks. Four rays at 20 samples each is 80 terrain samples per tick,
12,800 per trial — against roughly 92,000 the contact solver already does. Call
it 14% on top, which is affordable and should still be measured rather than
assumed, on AC power, with interleaved repetitions.

**What it returns.** `1 - min(distance, range) / range`, so 1 means the surface is
at the sensor and 0 means nothing within range. Bounded in `[0, 1]`, and zero is
the quiet value — which matters because `clear_inputs` zeroes everything, so a
slot with no sensor and a sensor seeing nothing read alike, and neither perturbs
a freshly initialised network.

## 3. A sensor is a jointed part, not a fitting bolted to one

This is the architectural decision the rest of the phase depends on, and it is
what makes this sensor the first member of a system rather than a one-off.

**A sensor hangs off a joint exactly as a limb hangs off the root.** It is a part
in the body tree: it has a parent, a joint, a motor, mass and inertia, and a
stable slot. Nothing about mounting, aiming or indexing is special-cased for
sensing, because a sensor *is* a limb whose function happens to be perception.

Everything then falls out of machinery that already exists:

* **Aiming is free.** The slot already owns a motor output driven by the
  controller. An organism that evolves a sensor on a hinged stalk can point it,
  using the same weights that would otherwise swing a leg. No sensor-aiming
  subsystem is needed, and none should be built.
* **Slot alignment is inherited.** The reading is indexed by the same stable slot
  as the joint that moves it, so deleting a limb never rewires the controller's
  understanding of any other — the competing-conventions problem stays solved.
* **Crossover stays a one-liner**, because the weight vector keeps its uniform
  length across the population.
* **Perception costs something.** A sensor part has mass, it loads the joint that
  carries it, and swinging it spends actuation. A long stalk that can sweep a
  wide arc is heavy and destabilising. That is not a tax invented to balance the
  feature — it is the same embodiment every other part is subject to, and it
  means evolution has to *pay* for looking. On a project about embodied
  evolution, a weightless sensor would be the anomaly.

**The extensible axis is the sensor's kind, not its plumbing.** A part gene gains
two things: which kind of sensor it carries, if any, and the direction it faces in
its own local frame. Lidar is the first kind; a camera-like and a hearing-like
kind are later values of the same gene, reading a different function and writing
the same input channels. That is what makes this the first child of a sensor
system rather than a special case that a second sensor would have to be worked
around.

**A fixed channel count per slot, interpreted by kind.** Sensor readings take a
fixed number of per-slot inputs — `INPUTS_PER_SLOT` went 3 → 4 for joint health
by exactly this route, via a separate constant, because input count sets the
weight-vector length.

As built, a slot contributes **one channel per ray** (`CHANNELS_PER_RAY`), and
that channel is range. Signature waits for beacons, because with only terrain to
hit it would be a constant zero occupying a weight — and a dead input is not free,
it lengthens the vector every genome carries. When beacons arrive,
`CHANNELS_PER_RAY` rises to 2 for the whole experiment.

What must not happen is a layout that depends on which sensors a *genome* happens
to carry: that would destroy the uniform weight-vector length that makes aligned
crossover a one-liner. A kind needing more channels raises the constant for the
experiment; it never varies per organism.

Range is an experiment constant to begin with. Making it a gene needs it to cost
something, or evolution simply maximises it.

## 3b. Beacons, and what the ray returns

A beacon is **a tall marker standing on the terrain**, and that is what makes it
detectable: it rises far enough above the ground that a ray clears the
intervening surface and meets it. Nothing tells the organism a beacon exists.
Whether one can be seen is a physical fact about where the organism is standing
and which way its sensor points — a beacon behind a ridge is invisible until the
ridge is crested, one across a valley is visible from the far rim, and one in the
next hollow is not visible at all.

That is the property worth protecting. It makes *looking* a thing an organism can
be bad at, which is the whole point of the phase.

**What a ray returns becomes two numbers, not one.** Range, as in §2, plus a
**signature** identifying what was hit: 0 for terrain, 1 for a beacon. This is
what a real lidar does — range and return intensity — so it is physically honest
rather than a convenience, and it generalises: a later object kind takes another
signature value without adding a channel.

Consequences worth stating, because each is a design decision rather than an
implementation detail:

* **Nearest hit wins**, so a beacon *occludes* the terrain behind it along that
  ray. An organism looking at a beacon gives up ground information in that
  direction. That trade is real and correct, and it is a reason to want more than
  one sensor.
* **The geometry is closed form.** A beacon is a vertical capsule or cylinder;
  ray-against-it is a quadratic, roughly ten operations, against the twenty
  terrain samples the march already costs. A handful of beacons needs no broad
  phase, and "a handful" is the scope.
* **Beacons do not collide.** They are markers, not walls. That keeps them out of
  `centre_of_mass`, `body_slots`, `detached` and the spawn drop, and it means
  this work does not wait on the discrete-obstacle layer deferred from both
  terrain plans.
* **Range must be bounded.** It is the difficulty knob — long range makes the
  task "walk toward the blip", short range makes it exploration — and if it were
  a free gene evolution would simply maximise it. Fix it per experiment first;
  making it evolvable needs it to cost something.
* **Beacon height against terrain relief** is the other knob, and the more
  interesting one: it sets how much of the map a beacon is visible from, and
  therefore whether the task is navigation or search.

### The conflict with the descent penalty, which must be settled first

The 45-generation penalty run scored best 5.71 and median 3.51. At
`descent_penalty = 8`, descending one metre costs 8 — more than the best organism
earned in total — and two metres costs 16.

On terrain with 5.6 m of relief, roughly half of all beacons are below the
organism. Under those weights **the correct policy is to ignore every downhill
beacon**, and evolution will find that policy long before it finds navigation. A
beacon task run against the current weights would measure nothing except the
penalty.

Raising the beacon reward until it drowns the penalty out is the wrong fix: it
makes the penalty inert, which is a confusing way to remove it. The real issue is
that `descent_penalty` was always a proxy for *falling*, and `net_loss` does not
measure falling — it charges a controlled walk downhill exactly what it charges a
tumble down a terrace.

**The targeted replacement**, and the machinery for it mostly exists: charge for
descent that happens while the organism is *out of control* rather than for
descent as such. Two candidate measures, both cheap:

* descent accumulated while `ground_clearance()` exceeds `AIRBORNE_CLEARANCE` —
  i.e. height lost while falling rather than while walking, reusing the airborne
  test `Metrics::airborne_seconds` already applies;
* peak downward velocity of the centre of mass, which distinguishes a step down
  from a drop without reference to contact at all.

Either makes "do not fall" expressible without also making "do not go downhill"
expressible, which is what a beacon task needs. This should land before beacons,
and it belongs in [FITNESS_PLAN.md](FITNESS_PLAN.md)'s territory rather than
here.

## 4. Staging

Each stage is separately runnable and separately falsifiable.

1. **The ray, tested against the field.** `TerrainModel::raycast(origin, dir,
   range) -> Option<Real>`, with the fixed-count march and bisection. No genome,
   no controller wiring. Validated against brute-force sampling (§5) — this is
   where a bug will hide, exactly as the terrain gradient was.
2. **A fixed sensor part, fixed direction.** Every organism gets a sensor part on
   the root, carrying two or three rays pointing forward and down, wired into the
   per-slot channels. No new genes, and the joint is welded rather than motorised
   — this stage asks whether perception changes what evolves, not whether aiming
   does. Two rays rather than one, for the reason in §0.
3. **The genome chooses.** Which parts carry sensors and where they point, with
   mutation and the "off is exact" discipline.
4. **The joint gets a motor**, so the controller aims its own sensor. Stage 3
   makes this possible without further work; it is a stage of its own because it
   is the first point at which an organism directs its attention, and because
   that deserves its own run and its own look at the replays. This is also where
   the cost of perception first bites: a stalk long enough to be useful is heavy
   enough to unbalance the body carrying it.
5. **Beacons and the signature channel** (§3b), once the falling-versus-descending
   term has replaced `descent_penalty`. Not before: under the current weights a
   beacon task measures the penalty rather than the navigation.

Stop after stage 2 and measure. If perception does not change behaviour with a
fixed forward-facing sensor, an evolvable one will not rescue it, and the reason
will be somewhere else — most likely §7's first risk.

## 5. Validation

* **The ray against brute force.** March a dense sample of the height field along
  the same ray and compare the crossing. Thousands of rays across several seeds,
  terrains and angles, including rays that start underground, rays that never
  hit, rays that graze a terrace edge, and vertical rays. This is the
  central-differences test of this feature.
* **Determinism.** `evo verify` identical across thread counts on a sensored
  experiment.
* **Off is exact.** An experiment declaring no sensors draws the identical random
  stream, keeps the identical controller layout, and reproduces every golden
  constant bit for bit.
* **A sensor reads its own mounting.** A sensor pointing straight down from a
  body at rest on flat ground must report the body's own clearance, which is a
  quantity `ground_clearance()` already computes independently.
* **The rule holds.** A test that the controller's global input count is unchanged
  by adding sensors — world information arrives per slot, through an organ, or it
  does not arrive.

## 6. Measuring whether it worked

**The experiment already exists.** Re-run the descent-penalty arm — same seed,
same weights, same terrain — with sensors on and off. That is a direct reading of
what perception is worth on a task where perception should matter, and the
control arm is already on disk.

What would count as success, in order of how convincing it is:

* the sensored arm travels further than the blind arm's 2.61 m while keeping its
  height;
* part count does not collapse to three, because a body that can see a drop does
  not need to be shaped so as never to meet one;
* in the replays, an organism visibly alters course *before* reaching an edge.

The third is the one actually being asked for and the only one that distinguishes
intent from build. It is also the one no aggregate will show, so it needs
watching rather than measuring — which is what the viewer is for.

## 7. Risks

* **A sense with nothing to do is worthless.** Seeing a drop only helps an
  organism that can act on it — stop, turn, change gait. If the evolved bodies
  cannot steer, perception buys nothing and the finding will be misread as "the
  sensor did not help". The steering machinery exists (`simulation.steer`), and
  the sensored experiment should have it on.
* **A new input is noise before it is information.** It arrives with random
  weights and takes generations to mean anything; early generations may be
  *worse* than blind. The 45-generation runs used so far are too short to judge
  this. Budget 150+, which on this terrain is around 30 minutes.
* **The sensor may be aimed nowhere useful.** A fixed forward-and-down ray from
  the root part of an organism that travels by vibrating may spend the whole
  trial pointing at the ground 10 cm away. Check what the sensor actually reads
  before concluding anything about what the organism did with it — record the
  readings in the trace and look at them.
* **Cost compounds with everything else.** 14% on a fractal experiment already
  running at 7 organisms/second, and stage 3 multiplies it by the number of
  sensors a genome may carry. The budget needs a ceiling in config, sized per
  experiment, exactly as `max_parts` bounds the body.
