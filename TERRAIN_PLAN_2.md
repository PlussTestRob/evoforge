# Plan: hills, cliffs, and ground worth crossing

> **Historical record.** This is a completed plan, kept for the reasoning and the
> measurements in it. Two items were never done: the A/B comparison of §8 stage 7,
> and the discrete obstacles of §5. Both are carried forward into
> [ROADMAP.md](ROADMAP.md), which is where current direction lives.

Status: **built, stages 1-6.** Successor to [TERRAIN_PLAN.md](TERRAIN_PLAN.md),
which built `TerrainModel::Fractal`.

What landed, against the staging in §8:

1. **Split impulse.** Positional correction accumulates in a per-body
   pseudo-velocity used only to displace, never becoming momentum — contacts,
   pair contacts and joints alike. The champions that crossed 26 m dead now
   travel nothing. `self_collision_is_not_a_motor` is the regression test, and
   it reproduces the bug in forty lines: a folded three-part chain crawls 2.74 m
   per 7.5 s on the old solver against 0.46 m now. Version bumped to 0.3.0 and
   the goldens re-baselined, which stored genomes under 0.2.x will notice.
2. **Perpendicular contact depth**, `depth * normal.y`, plus
   `a_dead_organism_does_not_travel` as a standing gate.
3. **Bands 1-3** and **4**: `FractalField` now carries landscape, detail,
   modulation and terracing, each defaulting to off so a v6 trace still means
   what it meant. Wall-width validation measures the field rather than
   estimating it.
4. **Spawn search**: a trial tries up to twelve placements and takes the first
   whose footprint is under 16 degrees, so nothing starts inside a cliff.
5. **Viewer**: all four bands mirrored in `viewer/terrain.js`, verified against
   the simulator at 23 cases x 361 points; terraced ground gets a smaller, finer
   sheet.

Two corrections to this document, both from measurements taken while building it
and both marked in place below: **domain warping does raise heterogeneity**
(§2), and the **3.71x energy gain attributed to the sine field never existed**
(§4) — it was an artifact of a datum that followed the body downhill. With the
metric fixed, no terrain creates energy on either the old solver or the new one.
The bug was self-collision and joints, not the ground.

Goal: terrain with substantial hills, near-vertical faces, and difficulty that
varies from place to place — hard enough to make legs pay for themselves,
without becoming a wall an organism cannot cross anywhere.

Every number below was measured. The feasibility model that produced them has
since been deleted, because the bands it modelled are now implemented and two
copies of the same arithmetic is exactly how a mirror drifts: rerun the numbers
with `cargo run --release --example terrain_probe`, which measures the real
field. Figures quoted from the model are marked where the implementation moved
them.

---

## 0. Nothing here can be trusted until the solver is fixed

`examples/dead_organism_probe.rs` takes an evolved champion, switches its motors
fully off, and evaluates it again. On the current `fractal-animals` run the
corpse travels **97% as far as the living organism**. On flat ground it travels
just as far. Turn self-collision off and it travels 0.02 m.

The propulsion is `solve_pair_contacts` adding its Baumgarte `bias` straight
into real velocity through `apply_impulse`, with no split-impulse pass, so an
organism whose parts keep re-penetrating collects up to `max_correction_speed`
per step and keeps it as momentum. It predates all terrain work:
`runs/animals-1788676600`, from before any of it, has a champion covering
26.6 m of its 35.7 m dead.

**Harder terrain makes this worse, not better.** The 80-generation A/B already
showed why: the fractal run scored 28.1 against the sine field's 12.0, not
because the organisms were better but because harder ground raised the price of
locomotion, so the fault's relative advantage grew and it took the whole
population. Note that this is a defect in the simulator rather than an organism
outsmarting its objective — the physics itself was wrong. Every increase in
difficulty proposed below pushes harder in that same direction.

Fix the solver first. Everything after §1 is void until then.

## 1. What the current field cannot do, and why

At the shipped `fractal-animals` settings, and then scaled up:

| field | relief | slope p50 | p90 | p99 | max | walkable | roughness spread |
|---|---|---|---|---|---|---|---|
| shipped, a=0.25 w=3 | 0.60 m | 10.0° | 17.4° | 23.1° | 33.8° | 100% | 2% |
| hills, a=1 w=12 | 2.20 m | 11.1° | 19.3° | 25.5° | 36.0° | 100% | 5% |
| hills, a=3 w=25 | 5.29 m | 16.2° | 27.1° | 34.3° | 43.8° | 100% | 9% |
| hills, a=6 w=40 | 9.92 m | 20.2° | 33.2° | 40.9° | 47.8° | 99% | 13% |

"Walkable" is the fraction of the plane under 40°. "Roughness spread" is the
standard deviation of mean slope across 12 m tiles, over its mean — how much
the character of the ground varies from place to place.

Two things fall straight out:

* **Scaling fBm up does not produce cliffs.** Ten metres of relief still tops
  out at 48°, and the median climbs in lockstep with the maximum. Fractional
  Brownian motion has one steepness, set by amplitude over wavelength, and it
  applies it everywhere. You cannot get *occasional* cliffs out of it by turning
  a knob, only uniform steepness — which is precisely the "so rough it is
  un-navigable" failure.
* **Big features do buy some heterogeneity** (2% → 13%), because a 40 m hill is
  large enough that one tile is a summit and another a flank. That is worth
  having, and it is not enough on its own.

## 2. The lever is the transfer function, not the noise basis

Simplex is a fair suggestion and it is not the thing that will help here.

What simplex gives over Perlin: better isotropy (Perlin's gradient set aligns to
the axes and diagonals, which shows as faint directional streaking in a single
octave), and O(n²) rather than O(2ⁿ) corner cost as dimension rises. The patent
that used to complicate it expired in 2022.

What it costs here: a rewrite of the analytic derivative through the skew and
unskew transforms, re-validation against central differences, a rewrite of the
JavaScript mirror in `viewer/terrain.js`, and an `ARTIFACT_FORMAT` bump. And it
would change no number in the table above by more than a rounding — because
those numbers are set by fBm's statistics, which simplex shares.

If the axis alignment ever becomes visible, the cheap fix is to widen
`GRADIENTS` in `src/physics/noise.rs` from 8 unit vectors to 16 or 24. That is a
table edit, and the three-bit shift in `gradient` becomes a mask. Simplex earns
its keep in 3D and above — see §6, where it becomes the right answer if we go
volumetric.

**What actually produces cliffs is what you do to the noise after you sample
it.** Two operations, both cheap, both analytically differentiable:

### Terracing

Quantise the height to steps, with a smooth riser between them:

```
t = h / step
h' = (floor(t) + smootherstep((frac(t) - 0.5) / riser + 0.5)) * step
```

`smootherstep` has zero derivative at both ends, so the composite stays C¹ and
the exact-normal property survives. The riser is steeper than the underlying
slope by exactly `1 / riser`, which is the knob that turns a 20° hillside into
an 80° wall.

| field (over a=3 w=25 base) | relief | p50 | p90 | p99 | max | walkable | roughness spread | median wall |
|---|---|---|---|---|---|---|---|---|
| step 0.5, riser 0.30 | 5.50 m | 0.0° | 54.0° | 73.4° | 82.4° | 85% | 9% | 190 mm × 0.39 m |
| step 0.5, riser 0.12 | 5.50 m | 0.0° | 19.1° | 81.5° | 86.9° | 91% | 13% | 130 mm × 0.47 m |
| **step 0.8, riser 0.12** | 5.60 m | 0.0° | 16.8° | 81.6° | 86.7° | 91% | 17% | 200 mm × 0.76 m |
| step 0.8, riser 0.06 | 5.60 m | 0.0° | 0.0° | 84.6° | 88.4° | 95% | 20% | 110 mm × 0.78 m |
| step 1.2, riser 0.06 | 4.80 m | 0.0° | 0.0° | 84.6° | 88.3° | 95% | 28% | 190 mm × 1.17 m |

This is the thing being asked for. Median slope **0°** — flat plateaus — with
the 99th percentile at 81° and walls three quarters of a metre tall. 91% of the
plane is walkable and **not one** straight 20 m crossing in 400 avoids meeting a
wall. The difficulty is concentrated instead of spread, which is what real
landscape does and what makes a fitness gradient rather than a flat ceiling.

### Roughness modulation, and masking

Multiply a component's amplitude by a second, much slower field.

*Corrected during implementation:* this plan said domain warping "measurably
does not" do the same thing, on the strength of a measurement that used the
spread of *relief* per tile. Relief is dominated by whatever large smooth hill
the tile sits on, so it cannot see a change in local steepness. Measured as the
spread of mean *slope* — the metric that matters — warp roughly doubles
heterogeneity too. Modulation is still the more direct lever and the mask is
still the strongest, but the contrast drawn here was not real.

The strongest version is to use that slow field to blend between *smooth hills*
and *terraced badlands*:

| field | relief | p50 | p90 | p99 | max | walkable | roughness spread | median wall |
|---|---|---|---|---|---|---|---|---|
| masked, step 0.5 | 5.53 m | 8.9° | 29.8° | 74.7° | 84.3° | 93% | **26%** | 90 mm × 0.18 m |
| masked, step 0.8 | 5.52 m | 8.9° | 29.5° | 74.6° | 84.5° | 93% | **26%** | 150 mm × 0.28 m |
| masked, step 1.2 | 5.47 m | 9.0° | 30.5° | 74.7° | 84.2° | 93% | **27%** | 200 mm × 0.30 m |

Roughness spread of 27% against the shipped field's 2% — thirteen times the
regional variation, and the first thing in either plan that moves this number
properly. The trade is wall height: blending softens the risers, so the median
wall drops from 0.76 m to 0.30 m. Whether concentrated badlands or taller walls
matter more is an empirical question for stage 3, and the config should express
both so it can be answered rather than argued.

## 3. Seams

The historical problem does not arise here, and it is worth saying why so nobody
spends time on it.

There are no seams. `TerrainModel` is one analytic function over an unbounded
domain: no chunks, no tiles, no LOD, no stitching. `height_at(x, z)` is defined
and continuous for every finite coordinate, and the physics never touches a
mesh. Whatever clipping happened in a chunked implementation cannot happen in
this one.

The one place two representations must agree is the viewer, which mirrors the
field in JavaScript — and that is already guarded: every non-flat trace records
`terrain_check`, sixteen heights the physics itself produced, and the viewer
recomputes them on load and says so loudly if they disagree. A one-bit change to
a hash constant is caught at 0.34 m against a 1 mm tolerance.

## 4. The physics work, which is most of the work

Steep ground breaks three assumptions the contact code currently makes. None of
these is optional if slopes are going past about 45°.

**Contact depth is measured vertically.** `build_contacts` uses
`depth = ground - corner.y`, a vertical distance, then resolves it along the
*surface normal*. On a slope of angle θ the vertical distance overstates the
true perpendicular penetration by `1 / cos θ` — 1.4× at 45°, 7× at 82°. The
correction is proportional to depth, so on a cliff face it is seven times too
large. The fix is one multiply — `depth * normal.y` — exact for a locally planar
surface.

*Corrected during implementation:* this section claimed the effect was "already
visible" as a passive box gaining 3.71× its energy on steep sine ground. That
number was wrong. The probe subtracted a datum that tracked the ground beneath
the body, so a box converting a hill's potential energy into speed reported the
speed while the datum followed it downhill. Measured as total mechanical energy
in a fixed frame, every terrain reads 1.00× on both the old solver and the new
one — terrain contacts never leaked. The depth fix is still correct and still
matters at 80°, where it stops the correction over-shoving; it just was not
fixing an energy leak.

**The candidate-point normal is sampled at the body's centre.** `build_contacts`
calls `terrain.normal_at(body.pos.x, body.pos.z)` once per body to choose which
of a curved shape's points can touch, with a comment saying a shape is small
relative to any terrain feature. Terraced ground breaks that: the walls are
200 mm wide and parts are up to 400 mm. Either sample per candidate point, or
sample the normal at the body's lowest point rather than its centre, and
re-measure the cost — this is the innermost loop.

**Spawn placement lifts vertically.** `phenotype::build_with_start` raises the
organism until no corner is under the ground beneath it. On a plateau that is
right; straddling a riser it puts the body half inside a wall, and the solver's
response to that is exactly the free energy §0 is about. Give the per-trial
terrain shift a spawn search: try a handful of deterministic candidate offsets
and take the first whose local slope is under, say, 15° across the organism's
footprint. Cheap, deterministic, and it also makes trials fair to each other.

**Wall width is a hard constraint, not a taste.** At 3 m/s and `dt = 1/120` a
body moves 25 mm per step. The 200 mm walls of the `step 0.8, riser 0.12`
setting take eight steps to cross, which the solver can handle. The 110 mm walls
of `riser 0.06` take four, which is marginal, and anything thinner will be
tunnelled through or met as a single enormous penetration. **Validate
`riser × step / typical_gradient ≥ 8 × max_speed × dt` and reject configs that
violate it in `Config::validate`.** This is the rule that keeps "sheer" on the
right side of "broken".

## 5. Ninety degrees, honestly

A height field cannot do 90°. `h(x, z)` is single-valued, and the analytic
normal `normalize(-dh/dx, 1, -dh/dz)` degenerates as `dh → ∞`: at vertical,
`n.y = 0`, the perpendicular-depth correction of §4 multiplies the depth by
zero, and contacts vanish — bodies pass through the wall. The measured ceiling
is **88.4°**, and the physics is unreliable above roughly 75°.

So: terracing gets genuinely sheer faces, up to about 80° with wall widths the
solver can survive. That is a wall an organism must climb, hook over or go
around. It is not an overhang, not a gap, and not a vertical face.

For true 90° and beyond there are two options.

**Static obstacles** — layer 3 of the original plan. Seeded rocks, ledges and
logs as bodies with `inv_mass = 0`. Most of the machinery exists:
`build_pair_contacts` / `solve_pair_contacts` already do capsule-capsule
two-body contacts. The care needed is keeping obstacles out of everything that
assumes a body belongs to the organism — `centre_of_mass`, `body_slots`,
`detached`, the spawn drop — which argues for a separate `obstacles` list rather
than pushing them into `World::bodies`. This gives exact vertical faces, gaps
and steps of any height, and it is the only thing in either plan that produces a
discontinuity at or above a wheel's radius, which is what actually decides
wheels against legs.

**A volumetric field** — the isosurface of a 3D density field rather than a
height map. This is where the 3D noise idea belongs, and where simplex would
genuinely earn its rewrite. It buys overhangs, arches and caves. It costs the
whole contact path: no closed-form surface, so penetration and normal come from
sphere-marching or a bounded Lipschitz approximation, and the exact-normal
property that the current design rests on has to be rebuilt from scratch. Real
research, and out of proportion to what it would settle.

**Recommendation: terracing for the landscape, obstacles for the walls.**
Together they cover everything asked for except overhangs.

## 6. Config surface

Composed as bands rather than one noise call, because that is what makes each
piece independently measurable:

```toml
[environment]
terrain = "fractal"

# Band 1 — the landscape. Large, smooth, this is what "substantial hills" means.
terrain_amplitude = 3.0        # metres; relief runs to about 1.8x this
terrain_wavelength = 25.0
terrain_octaves = 5

# Band 2 — detail laid over it, at organism scale.
terrain_detail_amplitude = 0.35
terrain_detail_wavelength = 3.0
terrain_detail_octaves = 4

# Band 3 — where the ground is calm and where it is savage.
terrain_modulation = 0.9       # 0 = uniform everywhere, as today
terrain_modulation_wavelength = 35.0

# Band 4 — cliffs.
terrain_step = 0.8             # terrace height, metres; 0 = smooth
terrain_riser = 0.12           # fraction of a terrace spent climbing
terrain_terrace_mask = true    # terrace only the savage regions

terrain_per_trial = true
```

Every one of these defaults to a value that reproduces today's field exactly
(`detail_amplitude = 0`, `modulation = 0`, `step = 0`), and every one is folded
into `fingerprint()` only when non-zero, exactly as `uses_shapes`,
`joints_can_break`, the tendon guard and the existing fractal guard already do.
`terrain = "flat"` and `"rough"` stay bit-identical. `tests/golden.rs` passes
with its constants untouched, or the change is wrong.

## 7. Validation

The existing tests carry over: analytic gradient against central differences
(now through the terrace and the mask, where a dropped `dm × gap` term will
hide), determinism, `evo verify` across thread counts, the JS mirror check, and
the spawn-clearance tests.

Four new gates, all of which have a tool already:

* **A corpse must not travel.** `examples/dead_organism_probe.rs`. Big hills mean
  running downhill is free metres, and `heading_progress` will pay for it. Gate:
  a motors-off organism covers less than 2 m in eight seconds. This is the test
  that would have caught the current bug on day one.
* **Passive bodies must not gain energy.** `examples/energy_probe.rs`, extended
  over the new fields. Gate: peak mechanical energy under 1.1× start, at every
  slope the config can produce. Today `rough` at 42° fails this at 3.71×.
* **The ground must stay crossable.** `examples/terrain_probe.rs`'s walkable
  fraction and crossing statistics, and `terracing_makes_cliffs_and_leaves_the_ground_crossable`: at least 85% of the
  plane under 40°, and a straight 20 m crossing must meet between one and six
  walls — zero means the cliffs are decoration, more than six means an organism
  spends the whole trial climbing.
* **Walls must be wider than a step.** The `Config::validate` rule from §4.

## 8. Staging

1. **Split-impulse the solver.** Positional bias accumulates in a pseudo-velocity
   used only for integrating position, never becoming momentum. Applies to
   contacts, pair contacts and joints. This moves every golden constant, which
   the golden header says is a deliberate, versioned decision — so it needs the
   version bump and a re-baseline, and it should be its own change with nothing
   else in it.
2. **Perpendicular contact depth**, and the corpse and energy gates above. Also
   re-baselines goldens; fold into step 1 if both are landing together.
3. **Bands 1–3**: hills, detail, modulation. Measurable, no new failure modes,
   and enough on its own to answer whether scale alone changes behaviour.
4. **Band 4**: terracing and the mask, with the wall-width validation and the
   per-point normal fix.
5. **Spawn search**, once there is ground worth failing to spawn on.
6. **Viewer**: the 200 mm walls need about 50 mm of mesh resolution to read as
   walls, against the 150 mm the current 90 m span and 600-segment cap give.
   Shrink the drawn span — organisms will not cross 90 m of this terrain — or
   tessellate by local gradient.
7. **Rerun the A/B**, which is void until step 1 lands.
8. Only then, obstacles.

## 9. Risks

* **Downhill is free fitness.** Five metres of relief is worth roughly 10 m/s of
  potential; `heading_progress` does not care how the metres were won. The
  per-trial rotation randomises slope against commanded heading and three trials
  average some of it out, but the corpse gate is what actually bounds it.
* **A flat ceiling is worse than a low one.** If the terrain is impassable
  everywhere, every organism scores the same and evolution has no gradient to
  climb. That is why the walkable fraction is a gate and not a note.
* **Cost.** Bands 2 and 3 add roughly one and a half fBm evaluations per sample
  on top of the current one. The fractal field is already 1.6× the sine field
  end-to-end; expect another 1.5–2× on top, and measure it on AC power with
  interleaved repetitions, because this laptop's throughput drifted from 82 to
  40 organisms/s across three consecutive benchmark runs.
* **Terracing can look like rice paddies.** Uniform step height reads as
  artificial. If it does, modulate `terrain_step` by the same slow field that
  drives the mask before reaching for a different noise basis.
