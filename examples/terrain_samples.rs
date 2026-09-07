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
use evoforge::physics::{FractalField, TerrainModel};

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

/// The shipped defaults with one seed changed. `FractalField::default()` is the
/// landscape band alone — every later band off — so a case that varies one field
/// with `..` is varying exactly that one thing.
fn seeded(seed: u64) -> FractalField {
    FractalField { seed, ..FractalField::default() }
}

fn main() {
    let d = |seed| TerrainModel::Fractal(seeded(seed));
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
        case("fractal, no warp", TerrainModel::Fractal(FractalField { warp: 0.0, ..seeded(7) })),
        case("fractal, heavy warp", TerrainModel::Fractal(FractalField { warp: 1.0, ..seeded(7) })),
        case(
            "fractal, one octave",
            TerrainModel::Fractal(FractalField { octaves: 1, ..seeded(7) }),
        ),
        case(
            "fractal, six octaves",
            TerrainModel::Fractal(FractalField { octaves: 6, ..seeded(7) }),
        ),
        case(
            "fractal, odd lacunarity",
            TerrainModel::Fractal(FractalField { lacunarity: 2.7, gain: 0.65, ..seeded(7) }),
        ),
        case(
            "fractal, fine",
            TerrainModel::Fractal(FractalField { amplitude: 0.05, wavelength: 1.5, ..seeded(7) }),
        ),
        // A long way out in field space, where an f32 has the least of its
        // precision left and the mirror has the most room to disagree.
        case(
            "fractal, offset",
            TerrainModel::Fractal(FractalField { offset_x: 41.5, offset_z: -63.25, ..seeded(7) }),
        ),
        case(
            "fractal, rotated",
            TerrainModel::Fractal(FractalField { rot_sin: 0.6, rot_cos: 0.8, ..seeded(7) }),
        ),
        case(
            "fractal, moved as a trial would",
            TerrainModel::Fractal(FractalField {
                wavelength: 3.0,
                warp: 0.6,
                offset_x: -37.125,
                offset_z: 58.5,
                rot_sin: -0.28,
                rot_cos: 0.96,
                ..seeded(0xDEAD_BEEF_CAFE_F00D)
            }),
        ),
        // The bands added after the first, one at a time and then together.
        case(
            "fractal, detail band",
            TerrainModel::Fractal(FractalField {
                amplitude: 3.0,
                wavelength: 25.0,
                octaves: 5,
                detail_amplitude: 0.35,
                ..seeded(11)
            }),
        ),
        case(
            "fractal, modulated detail",
            TerrainModel::Fractal(FractalField {
                amplitude: 3.0,
                wavelength: 25.0,
                octaves: 5,
                detail_amplitude: 0.35,
                modulation: 0.9,
                ..seeded(11)
            }),
        ),
        case(
            "fractal, terraced",
            TerrainModel::Fractal(FractalField {
                amplitude: 3.0,
                wavelength: 25.0,
                octaves: 5,
                step: 0.8,
                riser: 0.12,
                ..seeded(11)
            }),
        ),
        case(
            "fractal, terraced and masked",
            TerrainModel::Fractal(FractalField {
                amplitude: 3.0,
                wavelength: 25.0,
                octaves: 5,
                detail_amplitude: 0.35,
                modulation: 0.9,
                step: 0.8,
                riser: 0.12,
                terrace_mask: true,
                ..seeded(11)
            }),
        ),
        case(
            "fractal, every band, moved as a trial would",
            TerrainModel::Fractal(FractalField {
                amplitude: 3.0,
                wavelength: 25.0,
                octaves: 5,
                detail_amplitude: 0.35,
                modulation: 0.9,
                step: 1.2,
                riser: 0.2,
                terrace_mask: true,
                offset_x: -13.5,
                offset_z: 21.25,
                rot_sin: -0.28,
                rot_cos: 0.96,
                ..seeded(0xFEED_FACE_DEAD_BEEF)
            }),
        ),
    ];
    println!("{{\"cases\":[{}]}}", cases.join(","));
}
