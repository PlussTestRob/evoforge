# Plan: a difficulty ramp for terrain

Status: **proposed, not started.** Successor in spirit to
[TERRAIN_PLAN_2.md](TERRAIN_PLAN_2.md), which built the terraced landscape this
schedules. Sits under Phase 2 of [ROADMAP.md](ROADMAP.md) as an enabler: it does
not add a task, it changes when the existing one gets hard.

Goal: start a population on ground that has the fractal field's *character* but
none of its cliffs, and sharpen the cliffs in over the course of a run, so that a
gait is established before falling off something becomes an available answer.

Every number below was measured. Rerun the wall-width figures with the probe in
§8; do not trust the table if the constants in `src/config.rs` have moved.

---

## 0. What this is for

Three of seven organisms recorded at generation 20 of `ab2-fractal` do not
locomote. They move, descend, and stop:

| organism | share of travel in first half of window | descent in that half | total |
|---|---|---|---|
| 2008 | 81% | −0.76 m | 1.31 m |
| 2027 | 86% | −0.33 m | 0.49 m |
| 2032 | **96%** | −0.34 m | 0.50 m |

Organism 2032 is the clearest: it covers 0.48 m of its 0.50 m in the first two
seconds while dropping 0.34 m, then spends six seconds thrashing in place — path
length still growing, displacement flat, 13,337 units of actuation spent on
neither. Every flat and rough organism recorded at the same generation splits its
travel 46–51% between the two halves, which is what a gait looks like.

Two things make this worth acting on rather than watching.

**It is not winning, which means it is not self-correcting either.** The steady
movers on the same ground score higher (1.67 m against 1.31 m and below). So this
is a cheap local optimum that caps out low — reachable in very few generations,
and occupying population slots that would otherwise be exploring locomotion.

**It is not a fault.** The physics is right and the measurement is honest;
falling down a hill really is a way of covering ground. Nothing here should
penalise it. The intervention is to make the cheap answer *unavailable* early,
not to make it *unprofitable* — which is why this is a schedule and not a
fitness term.

## 1. Why a ramp, and why not the obvious alternatives

**Not flat → rough → fractal.** Measured on the same three arms: flat drives
mean part count from 4.52 to **2.12** and structural diversity to **21/100**,
while the fractal arm goes to **7.82** parts and holds **97/100**. The flat stage
does not build a foundation for the hard stage; it converges the population onto
the opposite body plan and destroys the variation needed to leave it. An easy
task is a stepping stone only when its optimum is in the same basin as the hard
task's, and here it demonstrably is not.

**Not "just run it longer".** The fractal arm was still climbing at generation 30
(median 0.80 → 2.53), so there is no evidence it is stuck. But the replays show a
competing attractor rather than merely slow progress, and time alone does not
remove an attractor.

**A ramp on the fractal field keeps the character and removes the affordance.**
Same relief, same detail bands, same per-trial variation, same body plan pressure
— with the cliffs smoothed out at the start and sharpened in as the run proceeds.

## 2. The knob is `terrain_riser`, not `terrain_step`

This is the load-bearing finding and it contradicts the obvious choice.

`terrain_step` is the terrace height and reads like the drop-height control, so
it looks like the right thing to ramp. It is not, for a reason that is pure
arithmetic:

```
riser_width = step * riser / smooth_gradient
```

Wall width is **linear in step**. `Config::validate` refuses walls thinner than
`MIN_WALL_STEPS * WALL_CROSSING_SPEED * timestep` = 4 × 3 × 1/120 = **100 mm**,
because a wall a body crosses in under four integration steps is met as one
enormous penetration or tunnelled through entirely. Ramping step up from zero
therefore walks straight through the floor:

| difficulty | step | wall width | |
|---|---|---|---|
| 0.0 | 0.00 | ∞ (terracing off) | OK |
| 0.1 | 0.08 | 15 mm | **rejected** |
| 0.3 | 0.24 | 46 mm | **rejected** |
| 0.5 | 0.40 | 76 mm | **rejected** |
| 0.6 | 0.48 | 91 mm | **rejected** |
| 0.7 | 0.56 | 106 mm | OK |
| 1.0 | 0.80 | 152 mm | OK |

Six of eleven rungs are invalid configurations. A step ramp is only legal over
the top third of its range, which is not a curriculum.

**`terrain_riser` has the opposite property.** It is the fraction of a terrace
spent climbing, and the riser is steeper than the underlying slope by exactly
`1 / riser`. At `riser = 1.0` the steepening factor is 1: the terracing does
nothing and the ground is the underlying fBm hills. At `riser = 0.12` the riser
is 8.3× steeper than the base slope and you have walls. Because width is linear
in riser, ramping it *down* means width falls monotonically from its widest to
the shipped value:

| difficulty | riser | wall width | |
|---|---|---|---|
| 0.0 | 1.00 | 1264 mm | OK |
| 0.25 | 0.78 | 986 mm | OK |
| 0.5 | 0.56 | 708 mm | OK |
| 0.75 | 0.34 | 430 mm | OK |
| 1.0 | 0.12 | 152 mm | OK |

**If the endpoint validates, every rung validates.** That is a property of the
schedule, not a coincidence of these numbers, and it is why riser is the right
primary knob.

It is also the better control for what we actually want to control. The
fall-and-stop strategy needs a *sharp* drop, not a tall one: 0.8 m of height
change spread over 1.26 m of run is a 32° slope you walk down; the same 0.8 m
over 152 mm is a 79° wall you fall off. Same drop height, completely different
affordance. Riser is the knob that separates those.

**`terrain_detail_amplitude` ramps second**, 0 → 0.35, for organism-scale
difficulty. Note that it interacts with the width formula — more detail means a
higher `smooth_gradient` and therefore *narrower* walls — so it cannot be
reasoned about independently. Ramped together with riser it stays monotone
(1479 mm → 152 mm), but that is a measured fact about these values, not a
guarantee. Hence §8.

**`terrain_step` stays fixed** at its configured value. Relief and drop height
are held constant across the run; only the sharpness changes.

## 3. Difficulty is a resolved config, never ambient state

The project's first property is that `sim::evaluate` is a pure function of
`(genome, config)`. A curriculum must not break that, and does not need to.

The runner resolves a per-generation configuration and hands *that* to
evaluation:

```rust
let gen_cfg = cfg.at_generation(population.generation);
evolution::evaluate_population(&mut population, &gen_cfg, &pool);
```

`sim` learns nothing new and reads no clock. `Config::at_generation` is a pure
function returning a `Config` with the ramped `[environment]` fields resolved.

Everything else keeps the **base** config, and this matters:

| call | config |
|---|---|
| `evaluate_population` | `gen_cfg` |
| `record_selected` → `sim::evaluate` | `gen_cfg` (or the replay will not match the fitness) |
| `evolution::next_generation` | **base** — breeding parameters do not ramp |
| `Population::founding` | **base** |
| `run.write_checkpoint` | **base** — the digest must be the base config's |
| `record::should_checkpoint` | **base** |

**Only fields that cannot change the genome's shape or the random stream may
ever be ramped.** Terrain values qualify. `trials`, `steer`, `joint_endurance`,
`shapes`, anything under `[brain]`, `[body]` or `[mutation]` do not: they move
the weight-vector length or the number of draws a genome consumes, and a run
whose genome layout changed halfway through is not one experiment. This should be
enforced structurally — `at_generation` touches `[environment]` and nothing else
— rather than by documentation.

## 4. The digest blocker: leave it alone

`runner.rs:178` refuses to resume a checkpoint whose config digest does not
match, with no `--force-resume` escape (force covers only the artefact-format and
version checks). The staged-resume form of a curriculum — run 30 generations,
edit the terrain, resume — is therefore refused outright.

**That check is correct and should not be weakened.** Defeating it would produce
a run directory whose `stats.csv` interleaves generations evaluated under
configurations that nothing on disk records: `evo replay` could not reproduce a
stored fitness, `evo inspect` would report a config that only described the last
leg, and two runs from the same directory would not be comparable. The digest
exists to stop exactly this, and a curriculum is the case it was written for, not
an exception to it.

**The design in §3 dissolves the problem instead of routing around it.** The
schedule lives in the config, so:

- the digest covers the entire curriculum — the run is self-describing;
- resume works untouched, with no new flag and no new semantics;
- resume stays idempotent, because nothing about the checkpoint ordering changes;
- raising `generations` on resume is still allowed, so a finished curriculum run
  can be extended at full difficulty exactly as any other run can;
- `evo verify` still passes, because the resolved config for a given generation
  is deterministic.

The cost is that the schedule must be decided before the run starts rather than
adapted after seeing results. That is a real loss of flexibility and it is worth
it: an adaptive schedule reintroduces every problem above. If adaptive
advancement is wanted later it should be a *deterministic function of recorded
statistics* — still reproducible from the config and seed — not a human editing
the file between legs.

## 5. Why the ramp cannot move the random stream

Worth checking explicitly, because it is the kind of thing that silently voids
every earlier result.

Per-trial randomness comes from `Rng::new(derive_seed(&[seed, TRIAL_STREAM,
trial]))` in `sim::evaluate_with`, which draws in a fixed order: `perturbation`,
then `commanded_heading`, then `terrain_shift`. The number of draws the first two
make does not depend on terrain.

`terrain_shift` contains a spawn search that breaks early when it finds level
ground, so it *does* consume a terrain-dependent number of draws — but it is the
**last** use of that generator, which is then dropped. Nothing downstream
observes the difference. The comment at that site already says it was placed last
deliberately.

So a change in terrain difficulty changes which ground an organism meets and
nothing else about the stream. This must become a test, not a paragraph: two
configs differing only in `terrain_riser` must produce identical start
perturbations and identical commanded headings.

## 6. Measuring whether it worked

A curriculum makes the fitness column non-comparable across generations: scores
fall when the ground gets harder, and "population got worse" is
indistinguishable from "test got harder" in `stats.csv`. Two instruments fix
that, and neither is optional — without them the run cannot be interpreted.

**A reference evaluation at full difficulty.** On each recorded generation, the
top organism is additionally evaluated at difficulty 1.0 and the result written
to a `reference.csv` alongside `stats.csv`. This is the honest progress curve for
the whole run, and it is nearly free: `record_selected` already performs a second
evaluation of a handful of organisms, so this adds one more. Written only when a
curriculum is configured, so no existing run directory changes shape.

**The temporal split, as the acceptance criterion.** The share of final
displacement reached in the first half of the measured window: about 50% for a
gait, 80%+ for a fall-and-stop. This is what says whether the curriculum did its
job, and net elevation change is *not* a substitute — it is near-uniform across
the population at about −0.22 m whatever the organism is doing, and correlates
with fitness at only −0.15.

Build it as a probe over recorded replays first (`examples/gait_probe.rs`).
Promoting it to a `Metrics` field is the natural first widening under Phase 1 of
the roadmap, but it is not needed here and it would touch `tests/golden.rs`.

## 7. Config surface

```toml
[curriculum]
ramp_generations = 150       # 0 disables the whole feature; difficulty 0 -> 1 over this span
hold_generations = 20        # generations at start_difficulty before the ramp begins
start_difficulty = 0.0

# What difficulty 0 means. Difficulty 1 is always the [environment] block above,
# so the endpoint of every ramp is the configuration that already ships.
easy_terrain_riser = 1.0             # 1.0 = terracing does nothing: smooth hills
easy_terrain_detail_amplitude = 0.0  # no organism-scale texture
easy_terrain_modulation = 0.0        # uniform ground rather than calm and savage regions
```

Difficulty is piecewise linear: `start_difficulty` for `hold_generations`, then
linear to 1.0 over `ramp_generations`, then held at 1.0 for the remainder of the
run. Each ramped field is `lerp(easy_value, configured_value, difficulty)`.

`ramp_generations = 0` is the default and must be **exactly** off: no field
resolved, no digest contribution, the identical random stream, and
`tests/golden.rs` passing with its constants untouched. Fold the `[curriculum]`
fields into `fingerprint()` only when `ramp_generations > 0`, exactly as
`uses_shapes`, `joints_can_break` and the terrain bands already do.

## 8. Validation

**Validate every rung, not just the endpoint.** This is the single most important
new rule and it generalises `validate_terrace_walls`, which today checks one
terrain. With a curriculum, `Config::validate` must resolve the difficulty at
every generation the schedule visits and run the existing terrain checks against
each — the wall-width rule in particular, since §2 shows how easily an
intermediate rung is invalid while both endpoints are fine. Sampling the ramp at
a fixed number of points is not enough if the schedule is non-monotone; validate
per generation, which is cheap.

The error message should name the generation and the resolved values, not just
the field, or a rejected schedule is very hard to debug.

Carried over unchanged: analytic gradient against central differences,
determinism across thread counts, the JS mirror check, spawn clearance.

New gates:

* **Off is exact.** A config without `[curriculum]` reproduces every golden
  constant bit for bit.
* **Stream invariance** (§5). Two configs differing only in `terrain_riser`
  produce identical start perturbations and commanded headings.
* **Every rung validates**, asserted directly against a schedule whose midpoint
  would be invalid — a step ramp from zero is the ready-made example.
* **Replay reproduces.** `a_recorded_champion_re_simulates_to_the_same_fitness`
  in `tests/pipeline.rs` must pass under a curriculum, which means `evo replay`
  has to resolve the rung for the organism's own generation rather than using
  the base config. This test will fail first if that is missed, which is what it
  is for.
* **The corpse gate still passes at every rung**, not just at difficulty 1.
  Easier ground is not obviously safer — it has been the source of every free
  ride so far.

## 9. Staging

1. **`Config::at_generation` plus the `[curriculum]` surface**, digest guard,
   and the per-rung validation of §8. No runner changes yet; unit-testable on
   its own, and the validation is where the subtle bugs are.
2. **Runner wiring**: resolve per generation, base config everywhere in the table
   in §3. The stream-invariance and off-is-exact tests land here.
3. **`evo replay` rung resolution**, and the pipeline test that catches it.
4. **`reference.csv`**, piggybacked on `record_selected`.
5. **`examples/gait_probe.rs`** — the temporal split over a run's replays. Needed
   before the first real run, because it is the acceptance criterion.
6. **A run.** `experiments/fractal-curriculum.toml`, seeded to match
   `fractal-animals.toml` so the only difference is the schedule. Compare
   against a fixed-difficulty control of the same length — without that control
   the run says nothing, since any long run improves.
7. Only then consider adaptive advancement, and only as a deterministic function
   of recorded statistics.

Acceptance: `cargo test --workspace` green with golden constants unchanged;
`cargo fmt --check` and `cargo clippy` clean; `evo verify` identical across
thread counts on a curriculum experiment; and the curriculum arm showing a lower
share of fall-and-stop organisms than the control at equal reference fitness.

## 10. Risks

* **The control is the experiment.** A curriculum run that reaches fitness *X*
  proves nothing on its own. The comparison is against a fixed-difficulty run of
  the same generation count, on the same seed, judged by `reference.csv` and the
  gait probe rather than by raw fitness. Budget for both arms from the start.
* **The ramp may simply be too fast or too slow**, and 150 generations is a
  guess. A ramp that outruns the population produces the same fall-and-stop
  organisms three quarters of the way through; one that crawls wastes compute on
  ground nothing is learning from any more. `reference.csv` is what shows which,
  and the schedule is cheap to change between runs.
* **Cost.** The fractal arm ran at 7.4 organisms/s at generation 29 with 7.82
  parts. Early curriculum generations should be *cheaper* — smoother ground,
  fewer contacts, and probably smaller bodies — so the run may be less than
  proportionally expensive. Measure it rather than assuming, on AC power, with
  interleaved repetitions.
* **Easy ground has its own cheap answers.** Smooth hills are what wheels and
  sliders are good at, which is the failure the terrain work was undertaken to
  escape. Starting at `riser = 1.0` with detail and modulation at zero is close
  to plain fBm, and a population that spends 20 generations learning to roll has
  been taught the wrong lesson. Consider starting the ramp at
  `start_difficulty = 0.2` rather than 0, and check the first recorded
  generations with the gait probe before trusting a long run.
* **Nothing here has been run.** The evidence in §0 is seven replays from one
  generation of one seed, two of them near-duplicate elites. It is enough to
  motivate the work and not enough to conclude the curriculum will help.
