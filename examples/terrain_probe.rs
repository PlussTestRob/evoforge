//! Measures the character of a terrain model.
//!
//! The terrain parameters are documented in `src/config.rs` and
//! `experiments/fractal-animals.toml` with numbers rather than adjectives, and
//! this is where those numbers come from. Run it after changing anything in the
//! height field, and again before writing "roughly" in a comment.
//!
//! Not part of the suite: it reports, it does not assert. What the suite pins is
//! in `physics::world::tests`.

use evoforge::math::Real;
use evoforge::physics::{FractalField, TerrainModel};

/// Sampled on a deliberately awkward grid: every octave of gradient noise is
/// exactly zero at its lattice points, so a grid of round numbers would report a
/// flatter field than the one an organism actually walks on.
const STEP: Real = 0.0731;
const SIDE: i32 = 600;

/// Slope an organism can plausibly walk up. Everything steeper is an obstacle.
const WALKABLE_DEGREES: Real = 40.0;

fn slope_at(m: &TerrainModel, x: Real, z: Real) -> Real {
    m.normal_at(x, z).y.clamp(-1.0, 1.0).acos().to_degrees()
}

/// Spread of mean slope across 12 m tiles, over its mean.
///
/// Slope rather than relief, deliberately. A tile on the flank of a large hill
/// has enormous relief and can still be billiard-smooth, so relief per tile
/// answers a different question — and answering it was what produced the
/// long-held and wrong claim that domain warping does nothing for
/// heterogeneity.
fn roughness_spread(m: &TerrainModel) -> Real {
    let mut tiles = Vec::new();
    for tx in -6..6 {
        for tz in -6..6 {
            let mut sum = 0.0;
            for a in 0..30 {
                for b in 0..30 {
                    sum += slope_at(
                        m,
                        tx as Real * 12.0 + a as Real * 0.4,
                        tz as Real * 12.0 + b as Real * 0.4,
                    );
                }
            }
            tiles.push(sum / 900.0);
        }
    }
    let mean = tiles.iter().sum::<Real>() / tiles.len() as Real;
    let var = tiles.iter().map(|t| (t - mean).powi(2)).sum::<Real>() / tiles.len() as Real;
    var.sqrt() / mean
}

/// Fraction of straight 20 m crossings that never meet anything impassable, and
/// the median width and height of the walls they do meet.
fn crossings(m: &TerrainModel) -> (Real, Real, Real) {
    const PROBE: Real = 0.01;
    let mut clear = 0;
    let mut widths: Vec<Real> = Vec::new();
    let mut climbs: Vec<Real> = Vec::new();
    let trials = 400;
    for i in 0..trials {
        let ang = i as Real * 0.0157;
        let (s, c) = (ang.sin(), ang.cos());
        let (ox, oz) = ((i % 20) as Real * 1.7 - 17.0, (i / 20) as Real * 1.7 - 17.0);
        let (mut blocked, mut run, mut run_h) = (false, 0, 0.0);
        for k in 0..2000 {
            let d = k as Real * PROBE;
            let (px, pz) = (ox + c * d, oz + s * d);
            let (h, n) = m.sample(px, pz);
            if n.y.clamp(-1.0, 1.0).acos().to_degrees() > 55.0 {
                if run == 0 {
                    run_h = h;
                }
                run += 1;
                blocked = true;
            } else if run > 0 {
                widths.push(run as Real * PROBE);
                climbs.push((h - run_h).abs());
                run = 0;
            }
        }
        if !blocked {
            clear += 1;
        }
    }
    let median = |mut v: Vec<Real>| {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    (clear as Real / trials as Real, median(widths), median(climbs))
}

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
            slopes.push(n.y.clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |q: f64| slopes[((slopes.len() - 1) as f64 * q) as usize];
    let walkable =
        slopes.iter().filter(|s| **s < WALKABLE_DEGREES).count() as Real / slopes.len() as Real;
    let (clear, wall_w, wall_h) = crossings(&m);
    println!(
        "{label:<38} relief {:5.2} m | slope p50 {:5.1} p90 {:5.1} p99 {:5.1} max {:5.1} \
         | walkable {:3.0}% | spread {:4.2} | clear {:3.0}% | wall {:3.0}mm x {:4.2}m",
        peak - trough,
        at(0.5),
        at(0.9),
        at(0.99),
        at(1.0),
        100.0 * walkable,
        roughness_spread(&m),
        100.0 * clear,
        wall_w * 1000.0,
        wall_h,
    );
}

/// The field `experiments/fractal-animals.toml` ships with.
fn shipped() -> FractalField {
    FractalField {
        seed: 0x5EED_0001,
        amplitude: 3.0,
        wavelength: 25.0,
        octaves: 5,
        warp: 0.6,
        detail_amplitude: 0.35,
        detail_wavelength: 3.0,
        detail_octaves: 4,
        modulation: 0.9,
        modulation_wavelength: 35.0,
        step: 0.8,
        riser: 0.12,
        terrace_mask: true,
        ..FractalField::default()
    }
}

fn plain(amplitude: Real, wavelength: Real, octaves: u32, warp: Real) -> TerrainModel {
    TerrainModel::Fractal(FractalField {
        seed: 0x5EED_0001,
        amplitude,
        wavelength,
        octaves,
        warp,
        ..FractalField::default()
    })
}

fn main() {
    println!("-- the sine field, for comparison");
    for amplitude in [0.05, 0.25] {
        measure(
            &format!("rough a={amplitude:.2} w=1.6"),
            TerrainModel::Rough { amplitude, wavelength: 1.6 },
        );
    }

    println!("\n-- one band of noise, scaled up: steepness rises everywhere at once");
    for (a, w) in [(0.25, 3.0), (1.0, 12.0), (3.0, 25.0), (6.0, 40.0)] {
        measure(&format!("hills a={a:.2} w={w:.0}"), plain(a, w, 5, 0.6));
    }

    println!("\n-- warp, on the landscape band alone");
    for warp in [0.0, 0.3, 0.6, 1.0] {
        measure(&format!("a=3 w=25 warp={warp:.1}"), plain(3.0, 25.0, 5, warp));
    }

    println!("\n-- the shipped experiment, band by band");
    let s = shipped();
    measure(
        "band 1 only (hills)",
        TerrainModel::Fractal(FractalField {
            detail_amplitude: 0.0,
            modulation: 0.0,
            step: 0.0,
            ..s
        }),
    );
    measure(
        "+ band 2 (detail)",
        TerrainModel::Fractal(FractalField { modulation: 0.0, step: 0.0, ..s }),
    );
    measure("+ band 3 (modulation)", TerrainModel::Fractal(FractalField { step: 0.0, ..s }));
    measure(
        "+ band 4 (terraces), unmasked",
        TerrainModel::Fractal(FractalField { terrace_mask: false, ..s }),
    );
    measure("+ band 4, masked  [as shipped]", TerrainModel::Fractal(s));

    println!("\n-- terrace knobs. Wall width is the constraint: at 5 m/s and dt=1/120");
    println!("   a body moves 25 mm per step, and `Config::validate` demands four steps.");
    for (step, riser) in [(0.5, 0.12), (0.8, 0.06), (0.8, 0.12), (0.8, 0.25), (1.2, 0.12)] {
        let f = FractalField { step, riser, ..s };
        let width = 1000.0 * f.riser_width();
        measure(
            &format!("step={step:.1} riser={riser:.2} (predicted {width:.0} mm)"),
            TerrainModel::Fractal(f),
        );
    }
}
