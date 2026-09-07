# Plan: fitness beyond distance

Status: **built, stages 1-6.** This is Phase 1 of [ROADMAP.md](ROADMAP.md) —
"widen `Metrics` where evaluation needs it" — done for its first two cases.

What landed, against the staging in §10:

1. `Metrics` gained `net_gain`, `net_loss`, `climb` and `descent`, with
   `ElevationTracker` carrying the hysteresis band. The aggregation contract is
   pinned by `trial_count_does_not_change_the_metrics_when_the_trials_are_identical`.
2. `evo rescore`, with the round trip gate. Measured on the 150-generation
   fractal run: worst `|rescored - recorded|` is 9.5e-7, which is float32 noise.
3. **Weights chosen from data — and the answer was not the one this plan
   predicted.** See the correction below.
4. `FitnessCfg` gained all five knobs, guarded in `fingerprint()`,
   `ARTIFACT_FORMAT` 8.
5. Cumulative terms and the deadband, with `bobbing_on_the_spot_earns_no_climb`
   written first, plus band monotonicity and slow-drift gates.
6. `--climb-bonus` / `--descent-penalty` on `evo run` and `evo rescore`; viewer
   columns for net elevation, total ascent and total descent.

Stage 7 (`Objective::Climb`) and stage 8 (a run) are deliberately not done: the
measurement in stage 3 says neither is worth doing until the terrain question
below is settled.

> ### Correction from stage 3: `climb_bonus` currently does nothing
>
> Re-scoring all 15,000 organisms of the 150-generation fractal run under
> `climb_bonus` values of 0, 10 and 50 changes **not one score and not one
> ranking**. `net_gain` is zero for every organism ever recorded in that run,
> because not one of them has ever ended higher than it started. The §11 risk was
> real and it is now measured rather than predicted.
>
> `descent_penalty` does work, but it has to be pushed hard before it changes
> anything. Fraction of the top ten that survives the re-weighting:
>
> | descent_penalty | top-10 kept | median fitness |
> |---|---|---|
> | 1 | 10/10 | 4.13 → 3.71 |
> | 2 | 10/10 | 4.13 → 3.30 |
> | 4 | 9/10 | 4.13 → 2.53 |
> | 8 | 6/10 | 4.13 → 1.07 |
>
> Below 4 it is very nearly a constant offset — it lowers everyone and reorders
> nobody, exactly as §4 warned. At 8 it genuinely reorders: organism 14970 rises
> from rank 31 to rank 2, and 14978 from 55 to 5, both trading roughly a third of
> their travel for less than half the descent. But the median falls to a quarter
> of what it was, which is the flat-gradient regime.
>
> **So the immediate next step is not a run, it is terrain.** A climb bonus is
> inert until uphill is reachable within a trial from where the spawn search puts
> an organism, and that is a question about spawn placement and trial length, not
> about weights.

Goal: score an organism on elevation gained and elevation lost as well as on
distance travelled, with every term weighted from configuration rather than
compiled in. Elevation gained should be able to outweigh distance.

Every number below was measured against `runs/ab2-*`. Rerun them before trusting
them; the probes are named where they were used.

---

## 0. The ask, and the one thing it must not become

Three things:

* fitness reads more than distance;
* **elevation gained**, weighted above distance covered;
* **elevation lost**, penalised.

A note on framing, because [ROADMAP.md](ROADMAP.md) says the project does not add
penalties to suppress strategies, and a descent penalty will certainly suppress
the fall-and-stop organisms of [CURRICULUM_PLAN.md](CURRICULUM_PLAN.md) §0.

That is a side effect, not the justification. What is being built here is a
*different question* — "get higher", rather than "get further" — and the roadmap
lists exactly that under Phase 2. It should be judged on whether it produces
organisms that climb, not on whether it removed organisms that fell. If it turns
out to suppress falling without producing climbing, it has failed even though it
achieved the side effect.

## 1. What exists today

Scoring is two stages and the boundary is load-bearing: `sim` produces `Metrics`,
`fitness::score` reduces `Metrics` to a scalar, and **`fitness` cannot see the
physics world**. Everything below stays on the `Metrics` side of that line.

```rust
base + upright_bonus * upright_seconds
     - energy_penalty * actuation
     + air_bonus * airborne_seconds
     + height_bonus * (peak_height - start.y).max(0)
```

`base` is picked by `Objective` — `Distance`, `DistanceX`, `Speed`, `Heading`.
So the "several weighted terms over one base" shape already exists, and this work
extends it rather than replacing it.

There is already one height term. `height_bonus` pays for `peak_height -
start.y`: the highest point reached, measured from where the organism began, so
being tall is worth nothing and only rising counts. It is not what is being asked
for here — it is a *jump* term, paired with `air_bonus`, and it ignores whether
the height was kept.

Full `Metrics` are recorded per organism in `organisms.jsonl`, explicitly so a run
can be re-scored later without re-simulating. Nothing uses that yet. §5 does.

## 2. What "elevation gained" means, and the trap in the obvious answer

Two definitions, and they behave completely differently.

**Net** — `end.y - start.y`, split into gain and loss:

```
net_gain = (end.y - start.y).max(0)
net_loss = (start.y - end.y).max(0)
```

Cheap, and **unexploitable**: there is no way to accumulate net gain except by
ending higher. Its weakness is that it cannot tell "climbed a hill and came down
the far side" from "never moved".

**Cumulative** — sum the positive vertical movements over the run, and the
negative ones separately. Richer: it measures how much climbing was actually
done. And it is the one that will be gamed, because an organism that bobs up and
down on the spot accumulates ascent forever without going anywhere.

**Measured baseline for the deadband.** Incidental ascent produced by an honest
gait, from the recorded replays at generation 20:

| arm | horizontal travel | naive cumulative ascent | with 5 cm deadband | with 10 cm |
|---|---|---|---|---|
| flat (6 organisms) | 6.3–6.7 m | 0.03–0.05 m | 0.00 m | 0.00 m |
| rough (7) | 2.2–3.5 m | 0.11–0.18 m | 0.00 m | 0.00 m |
| fractal (7) | 0.5–1.7 m | 0.01–0.17 m | 0.00–0.05 m | 0.00 m |

**This does not prove the naive metric is safe, and it must not be read that
way.** These organisms have never been paid to bob. The repository's own history
says what happens next: the first hang-time metric counted "no contact" as
airborne and evolution produced organisms hovering a millimetre above the ground
within forty generations. A quantity that pays per unit of vertical wobble will
be taken at its word. What the table actually establishes is the *floor* — how
much vertical movement honest locomotion produces incidentally — which is what a
deadband has to clear.

Note also that these are 30 Hz replay frames while the simulator steps at 120 Hz.
A per-step accumulation would be larger, so these are lower bounds.

**Recommendation.** Record all four — `net_gain`, `net_loss`, `climb`, `descent`
— because it is four floats and it is what makes §5 possible. Default the
*fitness weights* so that only the net pair is used, and put the cumulative pair
behind its own weights with a hysteresis deadband that defaults to 0.05 m. Then
the question "is cumulative worth the risk" gets answered by measurement instead
of by argument.

The hysteresis is a band, not a per-step threshold: track a reference height, and
only register movement (and move the reference) once the centre of mass has left
it by more than `climb_deadband`. A per-step threshold would still accumulate
under a slow drift; a band will not.

## 3. Weighting elevation above distance

Per-metre weights are not comparable until you know how many metres of each are
available. Measured on the fractal arm at **generation 149**, after 150
generations of selection under a pure distance objective:

| | achievable in one 8 s trial |
|---|---|
| horizontal travel, champions | 5.4 – 5.6 m |
| net elevation change, whole population | −0.010 to −0.725 m, median −0.389 |
| net elevation change, best ten | mean −0.513 m |
| organisms that net **ascended** | **0 of 100** |

So the two quantities differ by about an order of magnitude: champions cover
roughly ten metres of ground per metre of height they give up, and that ratio has
been stable since generation 40. To make elevation genuinely dominate,
`climb_bonus` wants to be around **10.0** against a distance base weighted 1.0 —
at which point half a metre of climb (5.0 points) is worth about as much as the
entire 5.5 m an evolved champion currently travels.

**Do not take that number on faith — it is an estimate from a population that has
never been asked to climb, and that currently never climbs at all.** §5 exists so
it can be chosen from data instead.

Two ways to express "elevation matters more", and they are not equivalent:

1. **Keep distance as the base, add weighted elevation terms.** Smallest change,
   and distance keeps a floor under the score so an organism that cannot climb
   still has a gradient to follow. Recommended first.
2. **Add `Objective::Climb`**, making elevation the base and demoting distance to
   a weighted bonus. Cleaner statement of intent, and it is what the roadmap
   means by an uphill task. Worth doing, but after (1) has shown what the weights
   should be — an objective with no gradient for a population that cannot yet
   climb is a flat fitness landscape, which is worse than a badly weighted one.

## 4. The descent penalty, and the hazard it carries

A penalty for losing elevation has a failure mode that the codebase has already
written down once, about a different term:

> Zero by default: energy pressure before locomotion exists just selects for
> doing nothing. — `FitnessCfg::energy_penalty`

The same applies here, and more sharply. **An organism that stands still loses no
elevation.** If `descent_penalty` is large relative to what distance pays, the
highest-scoring strategy is to not move at all — and unlike the energy penalty,
this one is not even paid for by the effort of standing. That is a real risk on
terrain where every organism currently descends.

Three things contain it:

* **Keep a distance term in the objective** so immobility scores below movement.
  This is the main reason to prefer §3 option (1) initially.
* **`descent_penalty` defaults to zero**, like every other opt-in term.
* **Gate it.** A test that a motionless organism does not outscore a moving one
  under the shipped weights, and the standing corpse gate.

Whether the penalty is *fair* is a separate question and the answer is yes: every
organism in a generation faces the same set of trials, drawn from the experiment
seed and the trial index, so two organisms differ in descent because they differ,
not because one drew a steeper hill. That property already holds and must not be
broken.

One measured observation, and it is the strongest single argument for this work.
At generation 149 of the fractal arm, **every organism in the population had
negative net elevation change** — 100 of 100, ranging −0.010 to −0.725 m — and
fitness correlates with elevation change at **−0.60**. Descending more predicts
scoring higher, strongly.

That correlation was only −0.15 at generation 29. It grew over 120 generations of
selection under a pure distance objective, while the crude fall-and-stop
behaviour was *disappearing*. So this is not a degenerate strategy that time will
remove; it is the objective quietly paying for downhill travel, and selection
finding that more thoroughly the longer it runs. There is ample gradient for a
descent penalty to act on — the population spans a 0.7 m range — and it is acting
on a real, entrenched bias rather than on noise.

## 5. Runtime configuration, and the tool that makes the weights choosable

The weights belong in `[fitness]` in the experiment TOML, which is already how
every fitness term is configured. Two additions make them genuinely tunable.

**`evo rescore <run> --fitness <file.toml>`** — re-score a completed run from its
stored `organisms.jsonl` under different weights, printing what the per-generation
best and median *would have been*, without simulating anything. `fitness::score`
is a pure function of `(FitnessCfg, &Metrics)` and full metrics are already on
disk, so this is a reader and a loop.

This is the highest-value item in the plan and it should be built first, because
of an accident of what is already recorded: `Metrics::start` and `Metrics::end`
are stored today. **The net gain and loss of §2 are therefore computable against
existing runs** — including the 150-generation fractal run — so their weights can
be chosen from real data before a single new experiment is run. The cumulative
pair cannot be, since nothing recorded it; that asymmetry is another reason to
land the net pair first.

**CLI overrides** — `--climb-bonus`, `--descent-penalty` and friends on `evo run`,
following `--seed`, `--generations` and `--population`. Cheap, and it makes a
sweep a shell loop. Overrides must flow through the config digest exactly as the
existing ones do, so a run directory still records what actually produced it.

## 6. Where the code changes

| file | change |
|---|---|
| `fitness.rs` | four new `Metrics` fields, all `#[serde(default)]`; new terms in `score` |
| `sim.rs` | accumulate the four during the step loop; **add all four to `accumulate` and `scale_metrics`** |
| `config.rs` | `FitnessCfg` weights plus `climb_deadband`; guarded `fingerprint`; range validation |
| `bin/evo.rs` | `rescore` subcommand; weight overrides |
| `record.rs` | `ARTIFACT_FORMAT` bump; older records read the new fields as zero |
| `viewer/library.js` | new sortable columns and preset chips, beside `mean height` and `actuation` |

**The easiest mistake in this list is `accumulate` and `scale_metrics`.** They
enumerate every field by hand. A field added to `Metrics` but forgotten in
`accumulate` silently reads as zero in any multi-trial experiment; forgotten in
`scale_metrics` it reads `n` times too large. Neither fails to compile. Note that
`steps` and `joints_lost` are deliberately summed and not scaled, so "scale
everything" is not the rule either.

The test that catches it: evaluate one genome at `trials = 1` and at `trials = 3`
with `start_jitter = 0` and no per-trial terrain variation, so all three trials
are identical, and assert every scaled field matches while the summed fields are
exactly three times larger. That is a structural test of the aggregation contract
rather than of any particular field, and it will keep catching this.

## 7. Compatibility discipline

The usual rules, none of them optional:

* **Every new weight defaults to zero**, so `score` returns exactly what it
  returns today and `tests/golden.rs` passes with its constants untouched. The
  goldens assert named fields — `fitness`, `displacement`, `path_length` and so
  on — so new `Metrics` fields do not disturb them provided the *fitness value*
  is unchanged.
* **No new randomness.** Metrics are measured, not drawn; nothing here may touch
  the RNG stream or the controller layout, so every earlier result reproduces bit
  for bit.
* **Fold the new weights into `fingerprint()` only when non-zero**, guarded
  exactly as the `air_bonus` / `height_bonus` pair already is.
* **Bump `ARTIFACT_FORMAT`**, with `#[serde(default)]` so a v7 record still loads
  and still means what it meant.

## 8. Validation

Carried over: determinism across thread counts, goldens, the corpse and energy
gates.

New gates, in rough order of how much they matter:

* **Bobbing does not pay.** The adversarial test for the cumulative terms: an
  organism oscillating vertically on the spot, with no horizontal travel, must
  score approximately zero climb under the shipped deadband. Write this *before*
  the cumulative terms are enabled anywhere, not after.
* **Standing still does not win.** Under the shipped weights, a motionless
  organism must score below one that travels — the §4 hazard.
* **Deadband monotonicity.** Cumulative climb must be non-increasing as the
  deadband widens, which catches sign and reference-tracking errors in the
  hysteresis.
* **The aggregation contract**, per §6.
* **Net and cumulative agree in the simple case.** On a monotone climb with no
  descent, `net_gain` and `climb` must be equal to within the deadband.
* **Re-score agrees with simulation.** Scoring a run's stored metrics under its
  *own* config must reproduce the fitness values already in `organisms.jsonl`,
  exactly. This is what makes `rescore` trustworthy and it is a one-line
  comparison over a real run directory.

## 9. Config surface

```toml
[fitness]
objective = "distance_x"

# Net elevation, measured from the settled pose to the end of the run.
# Unexploitable: the only way to gain is to end higher.
climb_bonus = 5.0          # per metre ended above the start
descent_penalty = 2.0      # per metre ended below it

# Cumulative elevation, hysteresis-filtered. Richer, and the pair that needs
# watching — see the bobbing gate. Both default to zero.
cumulative_climb_bonus = 0.0
cumulative_descent_penalty = 0.0
climb_deadband = 0.05      # metres of movement before a change is registered

# Existing terms, unchanged.
upright_bonus = 0.0
energy_penalty = 0.0
air_bonus = 0.0
height_bonus = 0.0
```

## 10. Staging

1. **`Metrics` gains the four fields**, with `accumulate`, `scale_metrics` and the
   aggregation test of §6. No scoring change; goldens untouched. This stage is
   worth landing alone because it is where the silent bugs are.
2. **`evo rescore`**, and the round-trip gate from §8. Buildable against the four
   new fields *and* against existing runs for the net pair, since `start` and
   `end` are already recorded.
3. **Choose the weights** by re-scoring the 150-generation fractal run under a
   sweep of `climb_bonus` and `descent_penalty`, and look at which organisms each
   setting would have promoted. Costs no simulation. Replaces the guess in §3.
4. **`FitnessCfg` net terms plus `score`**, guarded fingerprint, format bump, and
   the standing-still gate.
5. **Cumulative terms plus the deadband**, with the bobbing gate written first.
6. **CLI overrides**, viewer columns.
7. **`Objective::Climb`**, if stages 3–5 show the weights alone do not express it.
8. **A run**, against a distance-objective control on the same seed and terrain.

Acceptance: `cargo test --workspace` green with golden constants unchanged; `evo
verify` identical across thread counts; a re-score of any existing run under its
own config reproducing its recorded fitnesses exactly; and the bobbing and
standing-still gates passing.

## 11. Risks

* **A climb objective on terrain that offers no climb.** This is the biggest
  risk in the plan and there is now evidence for it: at generation 149, **zero of
  100 organisms ended higher than they started**. The spawn search places
  organisms on ground level to within about 16°, the trial ends after 8 s, and
  nothing in 150 generations has ever climbed. If uphill is not reachable within
  a trial, the climb term is a constant zero, and all the weighting achieves is
  diluting the distance gradient while the descent penalty scores everyone
  equally badly. **Measure the achievable climb before running anything**:
  evaluate a few hundred random and evolved organisms and look at the best net
  gain any of them manages. If it is near zero, the fix is the spawn placement or
  the trial length, not the weights.
* **Doing nothing beats trying.** §4. The most likely way this plan produces a
  disappointing run.
* **The cumulative terms get gamed.** Expected, planned for, gated. If it happens
  anyway, that is a measurement fault under the roadmap's taxonomy: fix the
  instrument, do not add a counter-penalty.
* **Weights tuned by re-scoring may not survive being evolved against.**
  Re-scoring says which *existing* organisms a weighting would promote; it cannot
  say what a population would have become under it. It narrows the search a long
  way, and it does not remove the need to run the experiment.
