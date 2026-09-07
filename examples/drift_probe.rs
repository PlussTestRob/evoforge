//! Does a bigger body get more for free?
//!
//! Evolution on the terraced landscape drives part count to the maximum while
//! the same experiment on the sine field settles at three. That is either a
//! real finding — more limbs are worth having on hard ground — or another
//! exploit, and the two look identical from a fitness curve.
//!
//! This tells them apart. It generates random organisms of each part count,
//! switches their motors fully off, and measures how far the corpses travel.
//! An honest advantage does not survive having the motors removed; a free ride
//! does, and grows with the number of parts collecting it.
//!
//!   cargo run --release --example drift_probe -- experiments/fractal-animals.toml

use std::path::Path;

use evoforge::config::Config;
use evoforge::genome::Genome;
use evoforge::math::Real;
use evoforge::sim;

fn main() {
    let path = std::env::args().nth(1).expect("usage: drift_probe <config.toml>");
    let base = Config::load(Path::new(&path)).expect("config");

    println!("{path}");
    println!(
        "motors off, 40 random organisms per part count, {} s each\n",
        base.simulation.duration
    );
    println!("  parts | mean heading | worst | mean |displacement| | alive, for scale");

    for parts in 2..=base.body.max_parts {
        let mut cfg = base.clone();
        cfg.body.min_parts = parts;
        cfg.body.max_parts = parts;
        // `caution` is the organism's own throttle and `min_drive` its floor, so
        // this turns an existing mechanism to zero rather than special-casing
        // the simulator.
        cfg.body.min_drive = 0.0;
        let layout = cfg.brain_layout();

        let (mut dead_sum, mut dead_worst, mut disp_sum, mut alive_sum) = (0.0, 0.0f32, 0.0, 0.0);
        let n = 40;
        for seed in 0..n {
            let mut rng = evoforge::rng::Rng::new(9_000 + seed as u64);
            let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            alive_sum += sim::evaluate(&g, &cfg, false).metrics.heading_progress;

            let mut corpse = g.clone();
            corpse.caution = 1.0;
            let m = sim::evaluate(&corpse, &cfg, false).metrics;
            dead_sum += m.heading_progress;
            dead_worst = dead_worst.max(m.heading_progress);
            disp_sum += m.displacement;
        }
        let f = n as Real;
        println!(
            "  {parts:5} | {:12.3} | {dead_worst:5.2} | {:20.3} | {:8.3}",
            dead_sum / f,
            disp_sum / f,
            alive_sum / f,
        );
    }
}
