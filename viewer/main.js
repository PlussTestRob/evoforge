// EvoForge replay viewer — phase 1.
//
// Reads one recorded replay JSON and draws it. It never simulates: the genome in
// the file is metadata, and every pose shown here came off disk. See
// ARCHITECTURE.md property 3 ("Nothing renders") — this is the other half of
// that split, the part that renders and does not simulate.
//
// Pose layout, from src/sim.rs: poses.length === bodies.length * 7, and for body
// i the values are position x,y,z then quaternion x,y,z,w. Y is up.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { initLibrary } from './library.js';
import { terrainHeight, finestFeature, terrainCheck, seedPair } from './terrain.js';

const POSE_STRIDE = 7;

// Stable per-slot colours so the same limb keeps its colour across replays.
const PALETTE = [
  0x6fb2ff, 0xffb454, 0x7ddc8b, 0xff7b9c,
  0xc79bff, 0x53d3d1, 0xe8d867, 0xff8f6b,
];

const el = (id) => document.getElementById(id);
const ui = {
  file: el('file'), sample: el('load-sample'), follow: el('follow'), loop: el('loop'),
  host: el('canvas-host'), hud: el('hud'), error: el('error'), empty: el('empty'),
  scrub: el('scrub'), track: el('track-measured'), play: el('play'),
  restart: el('restart'), speed: el('speed'), readout: el('readout'),
  openRun: el('open-run'), folder: el('folder'), toggleLib: el('toggle-library'),
  library: el('library'), trackEl: el('track'),
};

// ---------------------------------------------------------------- scene setup

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
ui.host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);
scene.fog = new THREE.Fog(0x14161a, 30, 95);

const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 400);
camera.position.set(2.6, 1.9, 3.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.4, 0);
controls.maxPolarAngle = Math.PI * 0.495; // stay above the ground plane

scene.add(new THREE.HemisphereLight(0x9fb4d0, 0x2a2d33, 1.5));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(6, 10, 5);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
const sc = sun.shadow.camera;
sc.left = -14; sc.right = 14; sc.top = 14; sc.bottom = -14; sc.near = 0.5; sc.far = 45;
scene.add(sun, sun.target);

// Terrain for these experiments is a plane at y = 0 (environment.terrain = "flat").
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(200, 200), // reshaped per replay by `shapeGround`
  new THREE.MeshStandardMaterial({ color: 0x343a45, roughness: 0.95, metalness: 0.0 }),
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// ------------------------------------------------------------------- terrain
//
// The height field itself lives in `terrain.js`, so that it can be checked
// against the simulator without a browser. See `viewer/terrain_check.mjs`.

/** Reshape the ground to match the terrain a replay actually ran on. */
function shapeGround(terrain) {
  const kind = (terrain && terrain.kind) || 'flat';
  const shaped = kind === 'rough' || kind === 'fractal';
  // Shaped ground needs several segments per feature or the relief aliases
  // away, and it is the *finest* octave that has to be resolved, not the
  // nominal wavelength. This is where the single-scale field caught me out
  // once, and a four-octave one has eight times as far to fall.
  //
  // The cap is a real limit, not a formality. At the shipped fractal settings
  // the finest octave is 0.375 m across, and resolving it properly would ask
  // for 1900 segments: 3.6 million vertices, and — measured in node — some
  // eight seconds of hashing, for a ripple three centimetres deep. The cap
  // spends 780 ms on 361k vertices instead and smooths that ripple away.
  // Everything an organism can actually climb is drawn.
  const span = shaped ? ROUGH_SPAN : GROUND_SPAN;
  const segments = shaped
    ? Math.min(MAX_GROUND_SEGMENTS, Math.ceil((span / finestFeature(terrain)) * 8))
    : 1;
  const next = new THREE.PlaneGeometry(span, span, segments, segments);
  if (shaped) {
    // Resolve the seed once rather than per vertex; it costs a BigInt parse.
    const seed = kind === 'fractal' ? seedPair(terrain.seed) : null;
    // PlaneGeometry lies in XY until it is rotated, so its local y is world -z.
    const pos = next.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = -pos.getY(i);
      pos.setZ(i, terrainHeight(terrain, x, z, seed));
    }
    next.computeVertexNormals();
  }
  ground.geometry.dispose();
  ground.geometry = next;
  grid.visible = !shaped; // a flat grid over shaped ground reads as a mistake
}

const GROUND_SPAN = 200;
const ROUGH_SPAN = 90;
const MAX_GROUND_SEGMENTS = 600;
/** `TerrainModel`'s variants. Anything else came from a newer evoforge. */
const KNOWN_TERRAIN = new Set(['flat', 'rough', 'fractal']);
const grid = new THREE.GridHelper(80, 80, 0x8b96a8, 0x5b6474); // 1 m cells
grid.position.y = 0.002;
grid.material.transparent = true;
grid.material.opacity = 0.75;
scene.add(grid);

// X red, Y green, Z blue. Fitness for the directed experiments is +X progress,
// so which way is red is worth being able to see.
const axes = new THREE.AxesHelper(1);
axes.position.y = 0.004;
scene.add(axes);

const organism = new THREE.Group();
scene.add(organism);

// ---------------------------------------------------------------- viewer state

let replay = null;     // parsed replay JSON
let frames = [];       // replay.trace.frames
let meshes = [];       // one per trace.bodies[i]
let t0 = 0, t1 = 0;    // time bounds of the trace
let simTime = 0;
let playing = false;
let lastTick = 0;
let followFrom = null; // last root position while "follow root" is on

const qA = new THREE.Quaternion();
const qB = new THREE.Quaternion();
const vA = new THREE.Vector3();
const vB = new THREE.Vector3();
const rootNow = new THREE.Vector3();

// ---------------------------------------------------------------- loading

function showError(msg) {
  ui.error.textContent = msg;
  ui.error.hidden = false;
}

function clearError() {
  ui.error.hidden = true;
}

/** Reject anything we cannot draw, with a message that says which field is wrong. */
function validate(r) {
  if (!r || typeof r !== 'object') throw new Error('not a JSON object');
  const tr = r.trace;
  if (!tr || !Array.isArray(tr.bodies) || !Array.isArray(tr.frames)) {
    throw new Error('missing trace.bodies or trace.frames — is this a replay file?');
  }
  if (tr.bodies.length === 0) throw new Error('trace.bodies is empty');
  if (tr.frames.length === 0) throw new Error('trace.frames is empty');
  const arity = tr.bodies.length * POSE_STRIDE;
  for (let i = 0; i < tr.frames.length; i++) {
    const f = tr.frames[i];
    if (!Array.isArray(f.poses) || f.poses.length !== arity) {
      const got = f && f.poses ? f.poses.length : 'undefined';
      throw new Error(
        `frame ${i}: poses.length is ${got}, expected ${arity} ` +
        `(${tr.bodies.length} bodies x ${POSE_STRIDE})`,
      );
    }
    if (typeof f.t !== 'number') throw new Error(`frame ${i}: missing numeric t`);
  }
  for (const b of tr.bodies) {
    const h = b && b.half_extents;
    if (!h || typeof h.x !== 'number' || typeof h.y !== 'number' || typeof h.z !== 'number') {
      throw new Error('a body is missing half_extents {x,y,z}');
    }
  }
}

// The two axes that are not `axis`, in ascending order. Mirrors `cross_axes` in
// src/physics/shape.rs.
const CROSS_AXES = [[1, 2], [0, 2], [0, 1]];

/** Turn a body's local +Y into its shape axis. Shapes are built along Y. */
function alignToAxis(geo, axis) {
  if (axis === 0) geo.rotateZ(-Math.PI / 2);
  else if (axis === 2) geo.rotateX(Math.PI / 2);
  return geo;
}

/**
 * Geometry for one recorded body, in its body frame.
 *
 * A replay written before shapes existed has no `shape`, and every part in it
 * was exactly the box `half_extents` describes — which is the fallback here.
 */
function buildGeometry(body) {
  const bounds = body.half_extents;
  const s = body.shape || { kind: 'box', half_extents: bounds };
  const size = (v) => [v.x, v.y, v.z];

  switch (s.kind) {
    case 'sphere':
      return new THREE.SphereGeometry(s.radius, 28, 18);

    case 'capsule':
      // CapsuleGeometry's `length` is the cylindrical section only, so the total
      // height is length + 2r — the same convention the simulator uses.
      return alignToAxis(
        new THREE.CapsuleGeometry(s.radius, s.half_length * 2, 8, 24),
        s.axis,
      );

    case 'cylinder':
      return alignToAxis(
        new THREE.CylinderGeometry(s.radius, s.radius, s.half_length * 2, 28),
        s.axis,
      );

    case 'taper': {
      // A four-sided cylinder is a square frustum once it is turned an eighth of
      // a turn; at that angle the circumradius sqrt(2) gives unit half-sides.
      const e = size(s.half_extents);
      const [b, c] = CROSS_AXES[s.axis];
      // `flip` puts the wide end at +axis instead of -axis; a reflected limb
      // has to point the other way or a symmetric organism is not symmetric.
      const wideAtTop = !!s.flip;
      const geo = new THREE.CylinderGeometry(
        Math.SQRT2 * (wideAtTop ? 1 : s.top_scale),
        Math.SQRT2 * (wideAtTop ? s.top_scale : 1),
        2,
        4,
      );
      geo.rotateY(Math.PI / 4);
      geo.scale(e[b], e[s.axis], e[c]);
      alignToAxis(geo, s.axis);
      // A frustum's centre of mass is not its geometric centre, and the pose in
      // the trace is the centre of mass. Same expression as `Shape::com_offset`
      // in src/physics/shape.rs; at top_scale 1 it is zero and at 0 it is half
      // the half-height, which is a pyramid's quarter-height from the base.
      const a = s.top_scale - 1;
      let off = (e[s.axis] * (a * (2 + a))) / (2 * (a * a + 3 * a + 3));
      if (wideAtTop) off = -off;
      const shift = [0, 0, 0];
      shift[s.axis] = -off;
      geo.translate(shift[0], shift[1], shift[2]);
      return geo;
    }

    case 'box':
    default: {
      const h = s.half_extents || bounds;
      // BoxGeometry takes full extents; the trace stores half-extents.
      return new THREE.BoxGeometry(h.x * 2, h.y * 2, h.z * 2);
    }
  }
}

function buildMeshes(bodies) {
  for (const m of meshes) {
    organism.remove(m);
    m.geometry.dispose();
    m.material.dispose();
  }
  meshes = bodies.map((b) => {
    const geo = buildGeometry(b);
    const mat = new THREE.MeshStandardMaterial({
      color: PALETTE[(b.slot ?? 0) % PALETTE.length],
      roughness: 0.55,
      metalness: 0.05,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    organism.add(mesh);
    return mesh;
  });
}

function num(x, digits = 3) {
  return typeof x === 'number' && isFinite(x) ? x.toFixed(digits) : '—';
}

function fillHud(r) {
  const m = r.metrics || {};
  el('h-org').textContent = r.organism_id ?? '—';
  el('h-gen').textContent = r.generation ?? '—';
  el('h-fit').textContent = num(r.fitness);
  el('h-dx').textContent = `${num(m.displacement_x)} m`;
  el('h-up').textContent = `${num(m.upright_seconds, 2)} s`;
  el('h-bodies').textContent = r.trace.bodies.length;
  const counts = new Map();
  for (const b of r.trace.bodies) {
    const kind = (b.shape && b.shape.kind) || 'box';
    counts.set(kind, (counts.get(kind) || 0) + 1);
  }
  el('h-shapes').textContent = [...counts]
    .sort((a, b) => b[1] - a[1])
    .map(([kind, n]) => `${n} ${kind}`)
    .join(', ');
  el('h-hz').textContent = num(r.trace.record_hz, 0);
  const breaks = (r.trace.breaks || []).length;
  el('h-breaks').textContent = breaks ? `${breaks}` : 'none';
  el('h-breaks').className = breaks ? 'warn' : '';
  el('h-terrain').textContent = (r.trace.terrain && r.trace.terrain.kind) || 'flat';
  el('h-diverged').hidden = !m.diverged;
  ui.hud.hidden = false;
}

function load(json, sourceName) {
  try {
    validate(json);
  } catch (e) {
    showError(`${sourceName}: ${e.message}`);
    return;
  }
  clearError();

  replay = json;
  frames = json.trace.frames;
  t0 = frames[0].t;
  t1 = frames[frames.length - 1].t;

  shapeGround(json.trace.terrain);
  buildMeshes(json.trace.bodies);
  fillHud(json);

  // The poses are read off disk and are always right. The ground is recomputed
  // here, and might not be — so say so, loudly, rather than drawing a plausible
  // landscape that is not the one the organism ran on.
  const drift = terrainCheck(json.trace.terrain, json.trace.terrain_check);
  const kind = (json.trace.terrain && json.trace.terrain.kind) || 'flat';
  if (drift) {
    showError(`${sourceName}: ${drift}`);
  } else if (!KNOWN_TERRAIN.has(kind)) {
    showError(
      `${sourceName}: terrain "${kind}" was recorded by a newer evoforge than this ` +
      `viewer knows about. The organism is drawn on a flat plane instead, so it ` +
      `will look like it is walking on nothing.`,
    );
  }

  // Shade the measured span of the timeline; everything before measure_start_t
  // is the settling drop, where the controller is held off.
  const span = Math.max(t1 - t0, 1e-6);  // also used to place the break marks
  const ms = typeof json.trace.measure_start_t === 'number' ? json.trace.measure_start_t : t0;
  const measuredPct = 100 * (1 - Math.min(Math.max((ms - t0) / span, 0), 1));
  ui.track.style.width = `${measuredPct}%`;

  // Mark on the timeline the moment each limb came off.
  for (const mark of [...ui.trackEl.querySelectorAll('.brk')]) mark.remove();
  for (const b of json.trace.breaks || []) {
    const mark = document.createElement('div');
    mark.className = 'brk';
    mark.title = `joint failed at ${b.t.toFixed(2)} s`;
    mark.style.cssText =
      `position:absolute;top:0;bottom:0;width:2px;background:var(--bad);` +
      `left:${(100 * (b.t - t0)) / span}%`;
    ui.trackEl.appendChild(mark);
  }

  ui.scrub.min = t0;
  ui.scrub.max = t1;
  ui.scrub.step = Math.min(1 / (json.trace.record_hz || 60) / 4, 0.005);
  for (const c of [ui.scrub, ui.play, ui.restart, ui.speed]) c.disabled = false;
  ui.empty.hidden = true;

  if (selecting && selecting.id === json.organism_id) {
    library.setActive(json.organism_id);
  }
  selecting = null;

  simTime = t0;
  setPlaying(false);
  apply(simTime);
  frameCamera();
  document.title = `EvoForge — gen ${json.generation} org ${json.organism_id}`;
}

/** Point the camera at the organism's starting pose. */
function frameCamera() {
  const box = new THREE.Box3().setFromObject(organism);
  if (box.isEmpty()) return;
  const centre = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length(), 0.5);
  controls.target.copy(centre);
  camera.position.copy(centre).add(new THREE.Vector3(radius * 1.6, radius * 1.3, radius * 2.6));
  controls.update();
  followFrom = ui.follow.checked ? centre.clone() : null;
}

// ---------------------------------------------------------------- playback

/** Index of the last frame at or before `t`. */
function frameIndexAt(t) {
  if (t <= frames[0].t) return 0;
  const last = frames.length - 1;
  if (t >= frames[last].t) return last;
  let lo = 0, hi = last;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/** Place every mesh at time `t`, interpolating between the two bracketing frames. */
function apply(t) {
  if (!replay) return;
  const i = frameIndexAt(t);
  const j = Math.min(i + 1, frames.length - 1);
  const a = frames[i], b = frames[j];
  const dt = b.t - a.t;
  const alpha = dt > 1e-9 ? Math.min(Math.max((t - a.t) / dt, 0), 1) : 0;

  for (let k = 0; k < meshes.length; k++) {
    const o = k * POSE_STRIDE;
    vA.set(a.poses[o], a.poses[o + 1], a.poses[o + 2]);
    qA.set(a.poses[o + 3], a.poses[o + 4], a.poses[o + 5], a.poses[o + 6]);
    if (alpha > 0) {
      vB.set(b.poses[o], b.poses[o + 1], b.poses[o + 2]);
      qB.set(b.poses[o + 3], b.poses[o + 4], b.poses[o + 5], b.poses[o + 6]);
      vA.lerp(vB, alpha);
      qA.slerp(qB, alpha);
    }
    meshes[k].position.copy(vA);
    meshes[k].quaternion.copy(qA);
  }

  // Body 0 is the root. Follow it by translating camera and orbit target
  // together, so orbiting still works while following.
  const root = meshes[0];
  if (root) {
    rootNow.copy(root.position);
    if (ui.follow.checked) {
      if (followFrom) {
        vB.copy(rootNow).sub(followFrom);
        camera.position.add(vB);
        controls.target.add(vB);
      }
      followFrom = (followFrom || new THREE.Vector3()).copy(rootNow);
    } else {
      followFrom = null;
    }
  }

  ui.scrub.value = t;
  const ms = replay.trace.measure_start_t;
  const phase = typeof ms === 'number' && t < ms ? 'settling' : 'measured';
  ui.readout.innerHTML =
    `t = <b>${t.toFixed(3)} s</b> / ${t1.toFixed(2)} s · ` +
    `frame ${i + 1} / ${frames.length} · ${phase}`;
}

function setPlaying(on) {
  playing = on && !!replay;
  ui.play.textContent = playing ? 'Pause' : 'Play';
  lastTick = performance.now();
}

function tick(now) {
  requestAnimationFrame(tick);

  if (playing) {
    const dt = Math.min((now - lastTick) / 1000, 0.1); // clamp after a tab stall
    simTime += dt * parseFloat(ui.speed.value);
    if (simTime >= t1) {
      if (ui.loop.checked) {
        simTime = t0;
      } else {
        simTime = t1;
        setPlaying(false);
      }
    }
    apply(simTime);
  }
  lastTick = now;

  controls.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- wiring

function readFile(file) {
  const reader = new FileReader();
  reader.onerror = () => showError(`could not read ${file.name}`);
  reader.onload = () => {
    let json;
    try {
      json = JSON.parse(reader.result);
    } catch (e) {
      showError(`${file.name}: not valid JSON (${e.message})`);
      return;
    }
    load(json, file.name);
  };
  reader.readAsText(file);
}

// ---------------------------------------------------------------- run library

const library = initLibrary({
  onSelect(entry) {
    selecting = entry;
    readFile(entry.file);
  },
});
let selecting = null;

ui.openRun.addEventListener('click', () => ui.folder.click());

ui.folder.addEventListener('change', async (e) => {
  const files = e.target.files;
  if (!files || !files.length) return;
  ui.library.hidden = false;
  ui.toggleLib.hidden = false;
  ui.toggleLib.textContent = 'Hide list';
  el('lib-title').textContent = 'Reading…';
  el('lib-sub').textContent = '';
  clearError();
  const n = await library.open(files, (done, total) => {
    el('lib-sub').textContent = `reading replay headers ${done} / ${total}`;
  });
  if (!n) {
    showError(
      'No replays in that folder. Choose the run directory itself — the one ' +
      'containing manifest.json and a replays/ subfolder.',
    );
  }
  resize();
  // Let the picker fire again for the same folder.
  e.target.value = '';
});

ui.toggleLib.addEventListener('click', () => {
  ui.library.hidden = !ui.library.hidden;
  ui.toggleLib.textContent = ui.library.hidden ? 'Show list' : 'Hide list';
  resize();
});

ui.file.addEventListener('change', (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) readFile(f);
});

ui.sample.addEventListener('click', async () => {
  const url = './samples/gen0-two-blocks.json';
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    load(await res.json(), url);
  } catch (e) {
    showError(
      `could not fetch ${url}: ${e.message}. ` +
      'Serve the viewer over http (see README) — file:// blocks fetch and module imports.',
    );
  }
});

ui.play.addEventListener('click', () => {
  if (!playing && simTime >= t1) simTime = t0; // play again from the start
  setPlaying(!playing);
});

ui.restart.addEventListener('click', () => {
  simTime = t0;
  apply(simTime);
});

ui.scrub.addEventListener('input', () => {
  simTime = parseFloat(ui.scrub.value);
  apply(simTime);
});

ui.follow.addEventListener('change', () => {
  followFrom = ui.follow.checked && meshes[0] ? meshes[0].position.clone() : null;
});

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && replay && e.target === document.body) {
    e.preventDefault();
    ui.play.click();
  }
});

// Dropping a replay onto the page is the same as picking it.
for (const type of ['dragover', 'drop']) {
  document.addEventListener(type, (e) => {
    e.preventDefault();
    if (type === 'drop') {
      const f = e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) readFile(f);
    }
  });
}

function resize() {
  const w = ui.host.clientWidth, h = ui.host.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();
requestAnimationFrame(tick);
