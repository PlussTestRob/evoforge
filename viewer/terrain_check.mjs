// Checks viewer/terrain.js against the simulator that produced the replays.
//
//   cargo run --release --example terrain_samples > samples.json
//   node viewer/terrain_check.mjs samples.json
//
// or, with no argument, against every replay JSON under `runs/`, using the
// `terrain_check` samples each one carries.
//
// This is the other half of the bargain struck in the plan for this feature: the
// viewer mirrors the height field rather than being shipped a sampled patch,
// which is cheap and fast, and pays for it by being checkable. Without a check
// the mirror silently drifts and the viewer draws a world nobody ran in.
//
// The only difference allowed is arithmetic width: the simulator works in f32
// and JavaScript has only doubles, which shows up around the seventh
// significant figure. Anything integer — the hash, the gradient choice — must
// agree exactly, and a mistake there moves heights by tens of centimetres.

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { terrainHeight, seedPair } from './terrain.js';

const TOLERANCE = 1e-3;

/** Compare one `{terrain, samples}` pair. Returns the worst absolute error. */
function check(label, terrain, samples) {
  const seed = terrain && terrain.kind === 'fractal' ? seedPair(terrain.seed) : null;
  let worst = 0;
  let at = null;
  for (let i = 0; i + 2 < samples.length; i += 3) {
    const mine = terrainHeight(terrain, samples[i], samples[i + 1], seed);
    const err = Math.abs(mine - samples[i + 2]);
    if (err > worst) {
      worst = err;
      at = { x: samples[i], z: samples[i + 1], rust: samples[i + 2], js: mine };
    }
  }
  const ok = worst <= TOLERANCE;
  const n = samples.length / 3;
  console.log(
    `${ok ? 'ok  ' : 'FAIL'} ${label.padEnd(46)} ${n} points, worst ${worst.toExponential(2)} m`,
  );
  if (!ok) console.log(`      at (${at.x}, ${at.z}): rust ${at.rust}, js ${at.js}`);
  return ok;
}

function* replays(dir) {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) yield* replays(path);
    else if (name.endsWith('.json')) yield path;
  }
}

const arg = process.argv[2];
let cases = [];

if (arg) {
  const doc = JSON.parse(readFileSync(arg, 'utf8'));
  cases = doc.cases.map((c) => [c.label, c.terrain, c.samples]);
} else {
  for (const path of replays('runs')) {
    let doc;
    try {
      doc = JSON.parse(readFileSync(path, 'utf8'));
    } catch {
      continue;
    }
    const tr = doc.trace;
    if (tr && Array.isArray(tr.terrain_check) && tr.terrain_check.length) {
      cases.push([path, tr.terrain, tr.terrain_check]);
    }
  }
  if (cases.length === 0) {
    console.log('no replays under runs/ carry terrain_check samples.');
    console.log('run `cargo run --release --example terrain_samples > samples.json` instead.');
    process.exit(0);
  }
}

const failed = cases.filter(([label, terrain, samples]) => !check(label, terrain, samples));
console.log(`\n${cases.length - failed.length}/${cases.length} agree`);
process.exit(failed.length ? 1 : 0);
