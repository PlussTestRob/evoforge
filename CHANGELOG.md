# Changelog

## 0.3.0

**Artifact format: 10.** Runs produced by 0.2.x will not re-simulate to their recorded fitness. Load them with `evo inspect` or `evo rescore` — the stored metrics are still valid — but treat their fitness values as upper bounds and do not resume them for new generations.

### What changed

**Solver: split-impulse positional correction.**
The self-collision conveyor bug is fixed. Positional correction now displaces bodies without adding to velocity. Before this fix, organisms with self-collision could cross 26 m with their motors switched off — 97% of a champion's travel was the solver, not the organism. `a_dead_organism_does_not_travel` and `self_collision_is_not_a_motor` are the standing gates. Results from 0.2.x runs that used `self_collision = true` should be read as upper bounds.

**Fractal terrain.**
Seeded fractional Brownian motion with four bands (landscape, detail, modulation, terracing) behind a single analytic height function — no chunks, no tiles, no mesh. The terrain's gradient is exact; contacts get the slope's own normal. Per-trial terrain variation (`terrain_per_trial`), domain warping (`terrain_warp`), and a terracing layer that produces cliffs are all new. `Config::validate` refuses wall configurations too thin for the physics to resolve.

**Elevation metrics and `evo rescore`.**
`Metrics` gained `net_gain`, `net_loss`, `climb`, `descent`, and `fall_distance`. `FitnessCfg` gained five new terms: `climb_bonus`, `descent_penalty`, `cumulative_climb_bonus`, `cumulative_descent_penalty`, and `fall_penalty`, each defaulting to zero. `evo rescore` re-weights a finished run without re-simulating; `evo rescore` with no overrides is a round trip and must reproduce stored fitness exactly.

**Lidar-like range sensor.**
A fan of rays cast from a mounted body part, returning `1 - distance / range` per ray. The sensor is carried by a part (has mass, moves with joints, costs actuation) and requires no mesh — it marches against the same analytic terrain function. `sensor_probability = 0` (the default) is exact: no sensor gene is drawn and no earlier result is affected.

**Viewer.**
A standalone browser replay player in `viewer/`. No build step. Serves from any static file server. Opens individual replays or a full run directory. Sortable and groupable by fitness, speed, actuation, parts, generation, joints lost, and more. Verifies terrain rendering against stored physics samples on load.

**`evo rescore` on the CLI.**
Re-score any finished run from the command line. Overrides: `--climb-bonus`, `--descent-penalty`, `--cumulative-climb-bonus`, `--cumulative-descent-penalty`, `--fall-penalty`. `--show-generation` lists individual organisms for a given generation under both the original and new scoring.

**Other changes.**
- `Metrics` gained `joints_lost` (joint-wear feature).
- `air_bonus`, `height_bonus`, `fall_penalty` fitness terms.
- Viewer columns for net elevation, total ascent, total descent, joints lost, and caution.
- `immigrant_rate` is now a configurable evolution parameter.
- `baumgarte` default lowered from 0.2 to 0.05 (see [RESULTS.md](RESULTS.md)).
- Hand-written transcendentals in `src/math.rs` ensure platform-independent bitwise determinism.
- CI matrix: Ubuntu, Windows, macOS; plus `evo verify` as a separate job.
