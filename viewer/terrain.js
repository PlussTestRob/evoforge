// The ground, mirroring TerrainModel in src/physics/world.rs and perlin_d in
// src/physics/noise.rs.
//
// A replay is drawn on recomputed ground rather than being shipped a sampled
// patch of it, which would cost 25-160 KB per replay. The risk a mirror carries
// is drift: get the hash subtly wrong and this draws a completely different
// world, one that still looks like plausible terrain, with organisms apparently
// floating over it. `terrainCheck` below is what turns that into a loud error
// instead of a silent one — every non-flat trace records the physics' own
// height at sixteen points, and the viewer recomputes them on load.
//
// It lives apart from main.js so it can run without a browser or three.js:
// `node viewer/terrain_check.mjs` checks it against heights the simulator
// itself produced. A mirror that cannot be tested is a mirror that drifts.
//
// Only the arithmetic below is allowed to differ from Rust: this runs in
// doubles where the simulator runs in f32, which shows up around the seventh
// significant figure and is invisible at a metre's scale. Everything integer —
// the hash, the gradient choice — must agree exactly, and does.

// 64-bit unsigned arithmetic, as pairs of 32-bit halves held in plain numbers.
//
// BigInt would say what it means and is far too slow: shaping the ground is a
// few hundred thousand vertices, each needing two dozen hashes. So is any
// version of this that returns a `[lo, hi]` array — that is tens of millions of
// allocations. Hence the scratch pair below and the one long function.
let mulLo = 0, mulHi = 0;

/** `[mulLo, mulHi] = a * b`, wrapping, for 64-bit values split into halves. */
function mul64(aLo, aHi, bLo, bHi) {
  // The high half of the low-by-low product, in 16-bit pieces so that nothing
  // exceeds a double's exact integer range.
  const a0 = aLo & 0xffff, a1 = aLo >>> 16, b0 = bLo & 0xffff, b1 = bLo >>> 16;
  const p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const mid = (p00 >>> 16) + (p01 & 0xffff) + (p10 & 0xffff);
  const carry = p11 + (p01 >>> 16) + (p10 >>> 16) + Math.floor(mid / 0x10000);
  mulLo = Math.imul(aLo, bLo) >>> 0;
  mulHi = (carry + Math.imul(aHi, bLo) + Math.imul(aLo, bHi)) >>> 0;
}

// SplitMix64's constants, and the two lattice multipliers, split into halves.
const GOLDEN_LO = 0x7f4a7c15, GOLDEN_HI = 0x9e3779b9;   // 0x9E3779B97F4A7C15
const SPLIT_A_LO = 0x1ce4e5b9, SPLIT_A_HI = 0xbf58476d; // 0xBF58476D1CE4E5B9
const SPLIT_B_LO = 0x133111eb, SPLIT_B_HI = 0x94d049bb; // 0x94D049BB133111EB
const HASH_X_LO = GOLDEN_LO, HASH_X_HI = GOLDEN_HI;
const HASH_Z_LO = 0x27d4eb4f, HASH_Z_HI = 0xc2b2ae3d;   // 0xC2B2AE3D27D4EB4F

/**
 * Which of the eight gradients sits at lattice point `(i, j)`.
 *
 * Mirrors `gradient` in src/physics/noise.rs: the coordinates are sign-extended
 * to 64 bits, multiplied by two odd constants, XOR-ed into the seed, run through
 * SplitMix64, and the top three bits taken. Written out flat rather than built
 * from 64-bit helpers because it is the hottest thing in this file by far.
 */
function gradientIndex(seedLo, seedHi, i, j) {
  mul64(i >>> 0, i < 0 ? 0xffffffff : 0, HASH_X_LO, HASH_X_HI);
  let sLo = (seedLo ^ mulLo) >>> 0, sHi = (seedHi ^ mulHi) >>> 0;
  mul64(j >>> 0, j < 0 ? 0xffffffff : 0, HASH_Z_LO, HASH_Z_HI);
  sLo = (sLo ^ mulLo) >>> 0;
  sHi = (sHi ^ mulHi) >>> 0;

  // SplitMix64, inlined. The state advance first: s += 0x9E3779B97F4A7C15.
  const sum = sLo + GOLDEN_LO;
  const hi = (sHi + GOLDEN_HI + (sum > 0xffffffff ? 1 : 0)) >>> 0;
  const lo = sum >>> 0;

  let zLo = (lo ^ ((lo >>> 30) | (hi << 2))) >>> 0;
  let zHi = (hi ^ (hi >>> 30)) >>> 0;
  mul64(zLo, zHi, SPLIT_A_LO, SPLIT_A_HI);
  zLo = (mulLo ^ ((mulLo >>> 27) | (mulHi << 5))) >>> 0;
  zHi = (mulHi ^ (mulHi >>> 27)) >>> 0;
  mul64(zLo, zHi, SPLIT_B_LO, SPLIT_B_HI);
  // Only the top three bits of `z ^ (z >>> 31)` are wanted, so the low half of
  // that last XOR need not be computed at all.
  return ((mulHi ^ (mulHi >>> 31)) >>> 29) & 7;
}

// The eight unit gradients, as flat arrays so the inner loop indexes numbers
// rather than chasing a pointer to a pair.
const SQRT_HALF = Math.fround(Math.SQRT1_2);
const GRAD_X = [1, -1, 0, 0, SQRT_HALF, -SQRT_HALF, SQRT_HALF, -SQRT_HALF];
const GRAD_Z = [0, 0, 1, -1, SQRT_HALF, SQRT_HALF, -SQRT_HALF, -SQRT_HALF];
const PERLIN_SCALE = Math.fround(Math.SQRT2);
const FIELD_LIMIT = 4194304;

const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

/** Gradient noise, mirroring `perlin_d` in src/physics/noise.rs. Value only:
 *  the viewer lets `computeVertexNormals` find its own normals. */
function perlin(seedLo, seedHi, x, z) {
  if (!(Math.abs(x) < FIELD_LIMIT && Math.abs(z) < FIELD_LIMIT)) return 0;
  const xi = Math.floor(x), zi = Math.floor(z);
  const u = x - xi, v = z - zi;
  const g00 = gradientIndex(seedLo, seedHi, xi, zi);
  const g10 = gradientIndex(seedLo, seedHi, xi + 1, zi);
  const g01 = gradientIndex(seedLo, seedHi, xi, zi + 1);
  const g11 = gradientIndex(seedLo, seedHi, xi + 1, zi + 1);
  const n00 = GRAD_X[g00] * u + GRAD_Z[g00] * v;
  const n10 = GRAD_X[g10] * (u - 1) + GRAD_Z[g10] * v;
  const n01 = GRAD_X[g01] * u + GRAD_Z[g01] * (v - 1);
  const n11 = GRAD_X[g11] * (u - 1) + GRAD_Z[g11] * (v - 1);
  const su = fade(u), sv = fade(v);
  const nx0 = n00 + su * (n10 - n00);
  const nx1 = n01 + su * (n11 - n01);
  return (nx0 + sv * (nx1 - nx0)) * PERLIN_SCALE;
}

// Constants shared with TerrainModel::Fractal. Changing one there means changing
// it here, and `terrainCheck` is what says so out loud when that is forgotten.
const OCTAVE_OFFSET_X = 0.5137;
const OCTAVE_OFFSET_Z = 0.9421;
const WARP_FREQUENCY = 0.5;
const WARP_OFFSET_X = 3.311;
const WARP_OFFSET_Z = -1.749;
const WARP_SEED_X_LO = 0x5f580001, WARP_SEED_X_HI = 0x57415250;
const WARP_SEED_Z_LO = 0x5f5a0001, WARP_SEED_Z_HI = 0x57415250;
const OCTAVE_STRIDE_LO = 0x56450001, OCTAVE_STRIDE_HI = 0x4f435441;

/**
 * A trace's `terrain.seed` as a `[lo, hi]` pair.
 *
 * It arrives as a decimal string precisely so that this can be exact: a double
 * cannot hold a `u64` above 2^53, and a seed off by one is a different world.
 * Called once per replay, so BigInt is fine here.
 */
function seedPair(seed) {
  let v;
  try {
    v = BigInt(seed);
  } catch {
    return [0, 0];
  }
  return [Number(v & 0xffffffffn) >>> 0, Number((v >> 32n) & 0xffffffffn) >>> 0];
}

/** Height of the ground a replay ran on. Mirrors `TerrainModel::sample`. */
function terrainHeight(terrain, x, z, seed) {
  if (!terrain) return 0;
  if (terrain.kind === 'rough') {
    const k = (Math.PI * 2) / Math.max(terrain.wavelength, 1e-3);
    const a = terrain.amplitude;
    return (
      a * Math.sin(k * x) * Math.cos(k * z) +
      0.5 * a * Math.sin(2 * k * x + 1.7) * Math.cos(2 * k * z + 0.9)
    );
  }
  if (terrain.kind !== 'fractal') return terrain.height ?? 0;

  const [seedLo, seedHi] = seed || seedPair(terrain.seed);
  const invW = 1 / Math.max(terrain.wavelength, 1e-3);
  const px = (x * terrain.rot_cos - z * terrain.rot_sin) * invW + terrain.offset_x;
  const pz = (x * terrain.rot_sin + z * terrain.rot_cos) * invW + terrain.offset_z;

  let qx = px, qz = pz;
  if (terrain.warp !== 0) {
    const wx = perlin(
      (seedLo ^ WARP_SEED_X_LO) >>> 0, (seedHi ^ WARP_SEED_X_HI) >>> 0,
      px * WARP_FREQUENCY + WARP_OFFSET_X, pz * WARP_FREQUENCY + WARP_OFFSET_Z);
    const wz = perlin(
      (seedLo ^ WARP_SEED_Z_LO) >>> 0, (seedHi ^ WARP_SEED_Z_HI) >>> 0,
      px * WARP_FREQUENCY + WARP_OFFSET_Z, pz * WARP_FREQUENCY + WARP_OFFSET_X);
    qx = px + terrain.warp * wx;
    qz = pz + terrain.warp * wz;
  }

  let sum = 0, frequency = 1, weight = 1;
  let octLo = seedLo, octHi = seedHi;
  const octaves = Math.min(terrain.octaves, 8);
  for (let o = 0; o < octaves; o++) {
    const step = o + 1;
    sum += weight * perlin(octLo, octHi,
      qx * frequency + step * OCTAVE_OFFSET_X, qz * frequency + step * OCTAVE_OFFSET_Z);
    frequency *= terrain.lacunarity;
    weight *= terrain.gain;
    // The next octave's seed: `seed.wrapping_add(o * OCTAVE_SEED_STRIDE)`.
    const carried = octLo + OCTAVE_STRIDE_LO;
    octHi = (octHi + OCTAVE_STRIDE_HI + (carried > 0xffffffff ? 1 : 0)) >>> 0;
    octLo = carried >>> 0;
  }
  return terrain.amplitude * sum;
}

/** Size of the finest feature the field has, which is what the mesh must resolve. */
function finestFeature(terrain) {
  if (!terrain) return Infinity;
  // Rough is two octaves an octave apart, so its finest is half its wavelength.
  if (terrain.kind === 'rough') return Math.max(terrain.wavelength, 0.1) / 2;
  if (terrain.kind !== 'fractal') return Infinity;
  const steps = Math.max(terrain.lacunarity, 1) ** Math.max(terrain.octaves - 1, 0);
  return Math.max(terrain.wavelength / steps, 0.1);
}

/**
 * How far the mirror may legitimately be out, in metres.
 *
 * Only arithmetic width is allowed to differ: the simulator works in f32 and
 * JavaScript has only doubles. Measured over 6,500 points across every terrain
 * kind and parameter — see `terrain_check.mjs` — the worst disagreement is
 * 8e-6 m, on a field offset by sixty wavelengths, where f32 has the least of
 * its precision left. A wrong hash or a stale constant is not a near miss: it
 * picks a different gradient and lands tenths of a metre out.
 */
const DRIFT_TOLERANCE = 1e-3;

/**
 * Check this file's mirror against the heights the physics actually used.
 *
 * Returns a message if they disagree, or null. A wrong hash or a stale constant
 * moves these by tens of centimetres; what is left when everything is right is
 * f32-versus-f64, some six orders of magnitude below that.
 */
function terrainCheck(terrain, samples) {
  if (!Array.isArray(samples) || samples.length < 3) return null;
  const seed = terrain && terrain.kind === 'fractal' ? seedPair(terrain.seed) : null;
  let worst = 0;
  let at = null;
  for (let i = 0; i + 2 < samples.length; i += 3) {
    const mine = terrainHeight(terrain, samples[i], samples[i + 1], seed);
    const err = Math.abs(mine - samples[i + 2]);
    if (err > worst) { worst = err; at = [samples[i], samples[i + 1]]; }
  }
  if (worst <= DRIFT_TOLERANCE) return null;
  return (
    `the ground drawn here is not the ground this replay ran on: ` +
    `${worst.toFixed(3)} m out at (${at[0]}, ${at[1]}). ` +
    `viewer/main.js has drifted from src/physics/world.rs — the poses are still ` +
    `correct, the terrain under them is not.`
  );
}


export { terrainHeight, finestFeature, terrainCheck, seedPair };
