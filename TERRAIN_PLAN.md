# Plan: seeded fractal terrain

Status: **built, stages 1-4 and the experiment file. Run — and the run found a
pre-existing bug in the simulator that invalidates the comparison.**

## What the run found

80 generations, population 100, both configs identical but for `[environment]`,
same seed (`runs/ab-rough-*`, `runs/ab-fractal-*`). The fractal run scored
*far higher* than the baseline — best 28.1 against 12.0, population mean 14.6
against 3.5 — which is the wrong direction for harder ground and was the tell.

It is not the terrain. Switch a champion's motors fully off (`caution = 1`,
`min_drive = 0`) and it still travels **97% as far**: 26.1 m of the 26.99 m.
`examples/dead_organism_probe.rs`. Isolating it with
`examples/conveyor_probe.rs`:

| condition | corpse travels |
|---|---|
| fractal, as shipped | 26.1 m |
| **flat ground** | 26.1 m |
| self-collision off | 0.02 m |
| `max_correction_speed` 2.0 -> 0.25 | 26.1 m -> 0.15 m |
| `baumgarte` 0.2 -> 0.0 | 26.1 m -> -0.05 m |
| timestep 1/120 -> 1/480 | 26.1 m -> -5.0 m |

The propulsion is **self-collision plus the Baumgarte positional correction**.
`solve_pair_contacts` adds its `bias` term straight into real velocity via
`apply_impulse`, with no split-impulse or pseudo-velocity pass, so an organism
whose own parts keep re-penetrating collects up to `max_correction_speed` = 2
m/s per step and keeps it. Terrain is irrelevant — it works on a flat plane.

**This predates the terrain work entirely.** `runs/animals-1788676600`, a
270-generation run from 6 September on the committed code, has a champion that
covers 26.6 m of its 35.7 m dead — 75% free. That is the run the README
describes as reaching "22 m of commanded travel". The diff for this feature
touches the terrain enum, one call site in `build_contacts`, and tests; no
solver code.

What the fractal terrain did was make the *honest* strategy more expensive, so
the exploit's relative advantage grew and it took over the population instead
of appearing in one champion of five. Harder ground did not select for better
locomotion; it selected harder for the cheat. This is §12's first lesson
arriving on schedule, one layer lower than expected.

The A/B is therefore void as a terrain comparison and has to be rerun after the
solver is fixed. The standard fix is a split-impulse pass: accumulate the
positional bias in a separate pseudo-velocity used only for integrating
position, so it never becomes momentum. That changes the dynamics and will move
every golden, which the golden header says is a deliberate, versioned decision
rather than something to do quietly.


What landed, against the staging in §11:

1. `TerrainModel::Fractal`, `physics::noise::perlin_d`, and the combined
   `sample()`. Gradient validated against central differences at both levels.
   Goldens untouched.
2. Config surface, digest guard, validation, `ARTIFACT_FORMAT` 6.
3. Viewer mirror, extracted to `viewer/terrain.js` so it can be checked without
   a browser: `cargo run --release --example terrain_samples > samples.json &&
   node viewer/terrain_check.mjs samples.json`. 18 cases x 361 points agree to
   8e-6 m; a one-bit change to a hash constant is caught at 0.34 m. Mesh
   resolution now follows the finest octave, capped.
4. Per-trial offset and rotation, drawn after every pre-existing draw so no
   existing stream moved.
5. `experiments/fractal-animals.toml`, seeded to match `animals.toml` so the
   only difference between the two runs is the ground. **The run itself has not
   been done.**
6. Obstacles: not started, as intended.

Two findings that contradict this plan as written, both measured with
`examples/terrain_probe.rs`:

* **Domain warping does not produce heterogeneity** (§4). Relief per 12 m tile
  varies by 10% of its mean whether warp is 0 or 1 — warping a stationary field
  with a stationary displacement leaves it stationary. What warp actually buys
  is the tail of the slope distribution: at a=0.25, w=6, the steepest slope
  anywhere goes from 19 degrees to 29. The docs say this rather than the
  adjective.
* **Peak-to-trough is 2.34x amplitude**, not 2.5x, and the fractal field at a
  given amplitude is *gentler* than `rough` unless the wavelength comes down
  with it — local slope is set by the finest octave, not the nominal one. The
  shipped experiment uses a=0.25, w=3.0, which matches the old median slope
  (9.9 degrees against 10.3) at five times the relief.

Two things §12 asked for that could not be honoured: this machine was on
battery throughout, so the A/B timing below is indicative only, and the
`src/phenotype.rs` allocation fix in §13 still has not been measured.

Indicative cost, three interleaved repetitions at 12 threads: `animals.toml`
28-29 organisms/s, `fractal-animals.toml` 17-18. That is roughly the 2x the
plan predicted, but it confounds the cost of sampling the field with the extra
solver work of ground that is actually rough.

---

Written to be executable without the conversation that produced it.

Goal: replace the current sine ground with a seeded, multi-scale, non-periodic
height field that looks and behaves like real terrain, without giving up
determinism, the exact surface normal, or the reproducibility of every existing
result.

---

## 1. What exists today

`TerrainModel` in [src/physics/world.rs](src/physics/world.rs) has two variants,
`Flat` and `Rough`. `Rough` is two octaves of a separable sine field:

```
h(x,z) = A·sin(kx)·cos(kz) + 0.5A·sin(2kx + 1.7)·cos(2kz + 0.9),   k = 2π/wavelength
```

`normal_at` returns the exact analytic gradient. Both use the project's own
`dsin`/`dcos` rather than `std`, because reproducibility depends on every
transcendental in the pipeline being ours.

It is configured by `environment.terrain = "rough"` with `terrain_amplitude` and
`terrain_wavelength`, and it is recorded into every replay as `Trace::terrain` so
the viewer can draw the same surface the physics used.

### Measured character of the current field

At the `experiments/animals.toml` settings (amplitude 0.05, wavelength 1.6):

| amplitude (λ=1.6) | peak-to-trough | median slope | max slope |
|---|---|---|---|
| 0.05 (current) | 0.126 m | 10.3° | 21.0° |
| 0.10 | 0.253 m | 20.0° | 37.5° |
| 0.15 | 0.379 m | 28.6° | 49.0° |
| 0.25 | 0.632 m | 42.3° | 62.5° |

Peak-to-trough is about **2.5 × amplitude** (the bound is 3×, but the octaves'
phase offsets mean they never quite peak together). Steepness is set by the
*ratio* `amplitude / wavelength`, not by amplitude alone: amplitude 0.10 at
wavelength 1.6 and amplitude 0.05 at wavelength 0.8 both give a 20.0° median
slope.

## 2. Why replace it

* **Periodic and fixed.** It repeats every `wavelength`, and there is no seed —
  every organism in every trial meets the identical surface. `start_jitter`
  shifts the starting point by up to ±0.5 m, which is a real fraction of a
  period, but the field itself never changes and is memorisable in principle.
* **Single-scale.** Two octaves an octave apart is not enough to read as
  landscape. Real ground has features at many scales at once.
* **Homogeneous.** Every region is equally rough. A gait evolved on uniform
  ground need not generalise, and heterogeneity — flat valleys, rough slopes —
  is most of what makes terrain a real selective pressure.

## 3. Hard constraints

These are not negotiable and each has bitten before.

**`TerrainModel` must stay `Copy` and small.** It lives inside `WorldParams`,
which is `Copy`, and is copied into every `World` — once per trial per organism.
No permutation table, no `Vec` of octaves. A handful of scalars.

**`height_at` is in the innermost loop.** Call sites:

* [world.rs:729](src/physics/world.rs) — per ground point per body per step (up
  to 8 points per body)
* [world.rs:726](src/physics/world.rs) and [world.rs:734](src/physics/world.rs) —
  `normal_at` once per body *and* once per contact point
* [world.rs:529](src/physics/world.rs) — `ground_clearance`, per point per step
* [phenotype.rs:525](src/phenotype.rs) — spawn placement, once per build

Note the redundancy already present: each contact point costs one `height_at`
**and** one `normal_at`. A combined `sample(x, z) -> (height, Vec3)` would halve
the terrain work at those sites, since value and gradient share nearly all their
computation. Add it as part of this work.

**Determinism.** No `std` transcendentals, no iteration-order dependence, no
platform-dependent arithmetic. `tests/golden.rs` pins bit patterns and its header
says explicitly not to update the constants to make it pass.

**The viewer must draw the same ground.** See §7.

## 4. Layer 1 — seeded fractal noise (the build)

Add `TerrainModel::Fractal`, summing octaves of **hash-based gradient noise**:

```
h(x,z) = A · Σᵢ gainⁱ · noise(seed, p · lacunarityⁱ / wavelength)
```

**Gradients by hashing, not by table.** Hash the integer lattice coordinates with
`splitmix64` (already in [src/rng.rs](src/rng.rs)) mixed with the seed, and index
a fixed set of 8 or 16 unit vectors. No table means the model stays `Copy`; a
`u64` seed plus six scalars is about 40 bytes.

**No transcendentals at all.** Integer hashing plus polynomial arithmetic is
*more* defensibly deterministic than the current sine field. This is a genuine
improvement to the reproducibility story, not just a feature.

**Quintic fade** `6t⁵ − 15t⁴ + 10t³`, whose derivative `30t⁴ − 60t³ + 30t²` is
what makes the analytic gradient available.

**Analytic derivative.** Perlin noise has a known closed-form gradient; carry it
through the dot products and the lerps with the product rule. This is the fiddly
part and the place a bug will hide — see §6 for how to validate it. Keeping it is
what preserves the exact-normal property the current field has.

**Domain warping.** Offset the sample position by a second, low-frequency noise
field before evaluating, behind a single `warp` scalar. This is the standard
trick that turns uniform fBm into something geological — ridges, valleys, and
crucially *heterogeneity*, so some regions are flat and others rough.

Expected cost: roughly 2× the current field at four octaves. No trig, but more
work per octave. Octave count is a config knob, and the combined `sample()`
recovers much of it.

## 5. Layer 2 — per-trial variation

Offset and rotate the field per trial, derived from the trial seed
(`TRIAL_STREAM` in [src/sim.rs](src/sim.rs) already exists for this kind of
thing). Nearly free, and it closes the memorisation hole in §2.

## 6. Layer 3 — obstacles (deferred, and the honest part)

**Layers 1 and 2 will make terrain look far better without much changing the
wheel-versus-leg outcome.** A height field is single-valued and smooth: no
overhangs, no true vertical walls, no gaps. Smooth undulation is exactly what
wheels are good at. Raising amplitude makes the ground *steeper*, not a different
kind of problem.

What actually defeats a wheel is **discontinuity at or above its own radius**. If
the goal is legs rather than scenery, the decisive layer is discrete obstacles:
seeded static bodies — rocks, ledges, logs.

Most of the machinery exists. Self-collision already does capsule-capsule
two-body contacts (`build_pair_contacts` / `solve_pair_contacts` in
[world.rs](src/physics/world.rs)); a static obstacle is the same constraint with
`inv_mass = 0`. The care needed is keeping obstacles out of everything that
assumes a body belongs to the organism — `centre_of_mass`, `body_slots`,
`detached`, the spawn drop — which argues for a separate `obstacles` list rather
than pushing them into `World::bodies`.

**Do not start this until layer 1 has run and the evidence says smooth terrain
is not enough.**

## 7. Viewer strategy

The viewer currently **mirrors the sine formula in JavaScript**
(`terrainHeight` in [viewer/main.js](viewer/main.js), around line 75). That was
flagged as a drift risk when it was written. For multi-octave hashed noise it
would be a much worse one: get the hash subtly wrong and the viewer draws a
completely different world, with organisms apparently floating.

Two options:

1. **Record a sampled patch** of the traversed region in the trace. Robust, the
   viewer stays dumb, but 25–160 KB per replay depending on resolution.
2. **Mirror the function in JS, and record ~16 verification samples** in the
   trace. The viewer computes its own heights at those points and warns loudly if
   they disagree.

**Take option 2.** Sixteen floats, and drift becomes a visible error instead of a
silent one. Fall back to option 1 only if the JS mirror proves unmaintainable.

Also note `shapeGround` in the viewer chooses mesh resolution from the
wavelength; with multiple octaves it must resolve the *smallest* octave, or the
relief aliases away. This already caught me once with the single-scale field.

## 8. Compatibility discipline

This repository's identity is that old results reproduce exactly. Every feature
added so far follows the same pattern, and this one must too:

* **Default off, and off must be exact.** `terrain = "flat"` and the existing
  `"rough"` must behave bit-identically to today. New parameters must not be
  drawn, hashed or folded in unless the new terrain is selected.
* **Guard the config digest.** Fold the new fields into `fingerprint()` in
  [src/config.rs](src/config.rs) **only when `terrain = "fractal"`**, exactly as
  `uses_shapes`, `joints_can_break` and the tendon and steer guards already do.
  Otherwise every existing run directory stops being resumable.
* **`tests/golden.rs` must pass with its constants untouched.** If a golden
  moves, the change is wrong — not the constant.
* **Bump `ARTIFACT_FORMAT`** in [src/record.rs](src/record.rs) (currently 5) and
  add a history line. `Trace::terrain` is a tagged enum, so a new variant needs a
  viewer fallback for unknown kinds.

## 9. Validation plan

The gradient is the part that will be wrong. Validate it the way the shape mass
properties were validated — against an independent brute-force computation, not
against itself:

* **Analytic gradient vs central differences** at thousands of sampled points,
  across several seeds and parameter sets. This is the single most important test.
* **Determinism**: same seed and coordinates give bit-identical heights, and
  `evo verify` passes on a fractal experiment across thread counts.
* **Range and continuity**: heights stay within the amplitude bound; no NaNs at
  lattice points, at the origin, or at negative coordinates (a common hashing
  bug is asymmetry about zero).
* **Spawn placement**: `a_shaped_organism_also_sits_on_the_ground` and friends in
  [src/phenotype.rs](src/phenotype.rs) should be extended to fractal terrain —
  every organism must still start exactly `SPAWN_CLEARANCE` above the surface.
* **Statistical character**: measure peak-to-trough and the slope distribution as
  in §1, so the config parameters can be documented with real numbers rather than
  adjectives.

## 10. Config surface

```toml
[environment]
terrain = "fractal"
terrain_seed = 0           # 0 derives from experiment.seed
terrain_amplitude = 0.25   # metres; peak-to-trough is roughly 2.5x this
terrain_wavelength = 6.0   # largest feature, metres
terrain_octaves = 4
terrain_lacunarity = 2.0   # frequency step per octave
terrain_gain = 0.5         # amplitude step per octave
terrain_warp = 0.3         # 0 = plain fBm, higher = more geological
terrain_per_trial = true   # offset and rotate the field per trial
```

All default to the current behaviour. Validate ranges in `Config::validate`
alongside the existing terrain checks.

## 11. Staging and acceptance

1. `TerrainModel::Fractal` with value and analytic gradient, plus the combined
   `sample()`. Gradient test against central differences. **Goldens untouched.**
2. Config surface, digest guard, validation, format bump.
3. Viewer: JS mirror plus recorded verification samples; fix the mesh resolution
   to the smallest octave.
4. Per-trial offset and rotation.
5. An experiment file, and a run long enough to see whether behaviour changes.
6. Only then consider layer 3 (obstacles).

Acceptance: `cargo test --workspace` green with golden constants unchanged;
`cargo fmt --check` and `cargo clippy` clean; `evo verify` identical across
thread counts on a fractal experiment; organisms visibly sit on the drawn ground
in the viewer.

## 12. Lessons from this codebase worth carrying in

Each of these was learned the hard way here:

* **Anything that adds energy to the simulation needs an adversarial test
  first.** The tendon shipped unstable: applied as an explicit torque impulse,
  its damping term inverted for light limbs and produced organisms crossing 100 m
  in 8 seconds. The fix was to express the spring as a *natural frequency* rather
  than a stiffness, making it independent of the inertia it acts on. There is now
  `a_tendon_never_adds_energy` guarding it.
* **Anything that scores behaviour gets gamed within tens of generations.** The
  first hang-time metric counted "no contact" as airborne, and evolution promptly
  produced organisms hovering a millimetre above the ground. Contacts only exist
  once a point is *below* the surface, so any clearance test needs a real margin.
* **Measure more than once.** Two conclusions in this project were drawn from
  single unreplicated measurements and were wrong. Benchmarks on a laptop vary by
  2× between identical runs depending on power state — check whether the machine
  is on battery before trusting any A/B timing.
* **Verify recorded output independently.** Reconstructing metrics from the
  replay files with a separate implementation has caught real bugs more than once.

## 13. Repository state when this was written

* Branch `more-like-animals`, last commit `4364149`.
* **Uncommitted: `src/phenotype.rs`** — an allocation fix in `build` replacing a
  `Vec<Vec<Mount>>` and per-part `Vec` allocations with fixed-size arrays, and
  restoring `Vec::with_capacity` hints. Behaviour-preserving (the goldens pass),
  but its performance benefit was **never successfully measured** — the benchmark
  that appeared to show a gain was contaminated by laptop power state. Re-measure
  on AC power with interleaved repetitions before claiming anything for it.
