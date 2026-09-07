//! Emits ground heights as JSON, for `node viewer/terrain_check.mjs` to check
//! the viewer's mirror of the height field against.
//!
//!   cargo run --release --example terrain_samples > samples.json
//!   node viewer/terrain_check.mjs samples.json
//!
//! Wider than the sixteen points a replay carries: every terrain kind, every
//! parameter that changes the field, seeds above 2^53 (which is where a `u64`
//! stops surviving `JSON.parse` as a number), negative coordinates, and a
//! rotated and offset field. If the mirror agrees on all of this it is the same
//! function.

use evoforge::math::Real;
use evoforge::physics::TerrainModel;

/// Deliberately awkward: every octave of gradient noise is exactly zero at its
/// lattice points, so round numbers would agree between two different fields.
fn sample_points() -> Vec<(Real, Real)> {
    let mut pts = Vec::new();
    for a in -9..10 {
        for b in -9..10 {
            pts.push((a as Real * 1.317 + 0.29, b as Real * 0.911 - 0.37));
        }
    }
    pts
}

fn case(label: &str, m: TerrainModel) -> String {
    let mut samples = String::new();
    for (i, (x, z)) in sample_points().into_iter().enumerate() {
        if i > 0 {
            samples.push(',');
        }
        samples.push_str(&format!("{},{},{}", x, z, m.height_at(x, z)));
    }
    format!(
        "{{\"label\":{:?},\"terrain\":{},\"samples\":[{}]}}",
        label,
        serde_json::to_string(&m).unwrap(),
        samples
    )
}

/// The fractal variant's parameters as a struct, so that a case can vary one of
/// them with `..` — which an enum variant does not allow.
#[derive(Clone, Copy)]
struct Frac {
    seed: u64,
    amplitude: Real,
    wavelength: Real,
    octaves: u32,
    lacunarity: Real,
    gain: Real,
    warp: Real,
    offset_x: Real,
    offset_z: Real,
    rot_sin: Real,
    rot_cos: Real,
}

impl Frac {
    fn seeded(seed: u64) -> Frac {
        Frac {
            seed,
            amplitude: 0.25,
            wavelength: 6.0,
            octaves: 4,
            lacunarity: 2.0,
            gain: 0.5,
            warp: 0.3,
            offset_x: 0.0,
            offset_z: 0.0,
            rot_sin: 0.0,
            rot_cos: 1.0,
        }
    }

    fn model(self) -> TerrainModel {
        TerrainModel::Fractal {
            seed: self.seed,
            amplitude: self.amplitude,
            wavelength: self.wavelength,
            octaves: self.octaves,
            lacunarity: self.lacunarity,
            gain: self.gain,
            warp: self.warp,
            offset_x: self.offset_x,
            offset_z: self.offset_z,
            rot_sin: self.rot_sin,
            rot_cos: self.rot_cos,
        }
    }
}

fn main() {
    let d = |seed| Frac::seeded(seed).model();
    let cases = vec![
        case("flat", TerrainModel::Flat { height: 0.0 }),
        case("flat, raised", TerrainModel::Flat { height: 1.25 }),
        case("rough, as in animals.toml", TerrainModel::Rough { amplitude: 0.05, wavelength: 1.6 }),
        case("rough, coarse", TerrainModel::Rough { amplitude: 0.25, wavelength: 6.0 }),
        case("fractal, seed 0", d(0)),
        case("fractal, seed 1", d(1)),
        // Above 2^53, where a `u64` stops surviving `JSON.parse` as a number.
        case("fractal, seed 2^53+1", d(9_007_199_254_740_993)),
        case("fractal, seed u64::MAX", d(u64::MAX)),
        case("fractal, seed 0x8000...", d(0x8000_0000_0000_0000)),
        case("fractal, no warp", Frac { warp: 0.0, ..Frac::seeded(7) }.model()),
        case("fractal, heavy warp", Frac { warp: 1.0, ..Frac::seeded(7) }.model()),
        case("fractal, one octave", Frac { octaves: 1, ..Frac::seeded(7) }.model()),
        case("fractal, six octaves", Frac { octaves: 6, ..Frac::seeded(7) }.model()),
        case(
            "fractal, odd lacunarity",
            Frac { lacunarity: 2.7, gain: 0.65, ..Frac::seeded(7) }.model(),
        ),
        case("fractal, fine", Frac { amplitude: 0.05, wavelength: 1.5, ..Frac::seeded(7) }.model()),
        // A long way out in field space, where an f32 has the least of its
        // precision left and the mirror has the most room to disagree.
        case(
            "fractal, offset",
            Frac { offset_x: 41.5, offset_z: -63.25, ..Frac::seeded(7) }.model(),
        ),
        case("fractal, rotated", Frac { rot_sin: 0.6, rot_cos: 0.8, ..Frac::seeded(7) }.model()),
        case(
            "fractal, moved as a trial would",
            Frac {
                wavelength: 3.0,
                warp: 0.6,
                offset_x: -37.125,
                offset_z: 58.5,
                rot_sin: -0.28,
                rot_cos: 0.96,
                ..Frac::seeded(0xDEAD_BEEF_CAFE_F00D)
            }
            .model(),
        ),
    ];
    println!("{{\"cases\":[{}]}}", cases.join(","));
}
