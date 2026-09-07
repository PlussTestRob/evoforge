//! Measures the character of a terrain model: relief and slope distribution.
//!
//! The terrain parameters are documented in `src/config.rs` with numbers rather
//! than adjectives, and this is where those numbers come from. Run it after
//! changing anything in the height field, and again before writing "roughly" in
//! a comment.
//!
//! Not part of the suite: it reports, it does not assert. What the suite pins is
//! in `physics::world::tests`.

use evoforge::math::Real;
use evoforge::physics::TerrainModel;

/// Sampled on a deliberately awkward grid: every octave of gradient noise is
/// exactly zero at its lattice points, so a grid of round numbers would report a
/// flatter field than the one an organism actually walks on.
const STEP: Real = 0.0731;
const SIDE: i32 = 700;

fn measure(label: &str, m: TerrainModel) {
    let mut peak = Real::NEG_INFINITY;
    let mut trough = Real::INFINITY;
    let mut slopes = Vec::with_capacity((SIDE as usize * 2).pow(2));
    for a in -SIDE..SIDE {
        for b in -SIDE..SIDE {
            let (x, z) = (a as Real * STEP, b as Real * STEP);
            let (h, n) = m.sample(x, z);
            peak = peak.max(h);
            trough = trough.min(h);
            // Angle between the surface normal and vertical.
            slopes.push(n.y.clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| slopes[((slopes.len() - 1) as f64 * q) as usize];
    println!(
        "{label:<44} relief {:.3} m (bound {:.3})  slope p50 {:.1} p90 {:.1} max {:.1} deg",
        peak - trough,
        2.0 * m.height_bound(),
        at(0.5),
        at(0.9),
        at(1.0),
    );
}

fn fractal(amplitude: Real, wavelength: Real, octaves: u32, warp: Real) -> TerrainModel {
    TerrainModel::Fractal {
        seed: 0x5EED_0001,
        amplitude,
        wavelength,
        octaves,
        lacunarity: 2.0,
        gain: 0.5,
        warp,
        offset_x: 0.0,
        offset_z: 0.0,
        rot_sin: 0.0,
        rot_cos: 1.0,
    }
}

fn main() {
    println!("-- rough, for comparison");
    for amplitude in [0.05, 0.10, 0.15, 0.25] {
        measure(
            &format!("rough  a={amplitude:.2} w=1.6"),
            TerrainModel::Rough { amplitude, wavelength: 1.6 },
        );
    }

    println!("\n-- fractal, amplitude sweep at w=6.0, 4 octaves, warp 0.3");
    for amplitude in [0.10, 0.15, 0.25, 0.40] {
        measure(&format!("fractal a={amplitude:.2} w=6.0"), fractal(amplitude, 6.0, 4, 0.3));
    }

    println!("\n-- fractal, wavelength sweep at a=0.25");
    for wavelength in [3.0, 6.0, 12.0] {
        measure(&format!("fractal a=0.25 w={wavelength:.1}"), fractal(0.25, wavelength, 4, 0.3));
    }

    println!("\n-- fractal, octaves at a=0.25 w=6.0");
    for octaves in [1, 2, 4, 6] {
        measure(&format!("fractal a=0.25 w=6.0 oct={octaves}"), fractal(0.25, 6.0, octaves, 0.3));
    }

    println!("\n-- fractal, warp at a=0.25 w=6.0, 4 octaves");
    for warp in [0.0, 0.3, 0.6, 1.0] {
        measure(&format!("fractal a=0.25 w=6.0 warp={warp:.1}"), fractal(0.25, 6.0, 4, warp));
    }

    println!("\n-- candidates for the shipped experiment");
    for (a, w, warp) in [(0.25, 3.0, 0.3), (0.25, 3.0, 0.6), (0.20, 2.5, 0.6), (0.30, 4.0, 0.6)] {
        measure(&format!("fractal a={a:.2} w={w:.1} warp={warp:.1}"), fractal(a, w, 4, warp));
    }

    // Heterogeneity is the claim domain warping is made for: some regions flat,
    // others broken. A uniform field has the same roughness in every tile, so
    // the spread of per-tile relief is what tells the two apart.
    println!("\n-- heterogeneity: spread of relief across 12 m tiles");
    for warp in [0.0, 0.3, 0.6, 1.0] {
        let m = fractal(0.25, 6.0, 4, warp);
        let mut tiles = Vec::new();
        for tx in -8..8 {
            for tz in -8..8 {
                let (mut lo, mut hi) = (Real::INFINITY, Real::NEG_INFINITY);
                for a in 0..40 {
                    for b in 0..40 {
                        let x = tx as Real * 12.0 + a as Real * 0.3;
                        let z = tz as Real * 12.0 + b as Real * 0.3;
                        let h = m.height_at(x, z);
                        lo = lo.min(h);
                        hi = hi.max(h);
                    }
                }
                tiles.push(hi - lo);
            }
        }
        tiles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = tiles.iter().sum::<Real>() / tiles.len() as Real;
        let sd =
            (tiles.iter().map(|t| (t - mean).powi(2)).sum::<Real>() / tiles.len() as Real).sqrt();
        println!(
            "warp {warp:.1}: tile relief mean {mean:.3} sd {sd:.3} ({:.0}%)  \
             calmest {:.3}  roughest {:.3}",
            100.0 * sd / mean,
            tiles[0],
            tiles[tiles.len() - 1],
        );
    }
}
