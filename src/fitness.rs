//! Fitness: turning behaviour into a single number.
//!
//! Fitness is computed from [`Metrics`] alone, never from the physics world.
//! That boundary is deliberate: an objective function that can reach into the
//! simulation state tends to acquire dependencies on solver details, and then
//! results stop being comparable across solver changes.

use serde::{Deserialize, Serialize};

use crate::config::{FitnessCfg, Objective};
use crate::math::{Real, Vec3};

/// Everything measured during one evaluation.
///
/// Recorded alongside fitness so that a run can be re-scored under a different
/// objective after the fact, without re-simulating.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Centre of mass at the start and end of the measured window.
    pub start: Vec3,
    pub end: Vec3,
    /// Straight-line horizontal displacement over the window.
    pub displacement: Real,
    /// Signed displacement along +X.
    pub displacement_x: Real,
    /// Total horizontal distance travelled, which exceeds `displacement` for
    /// anything that wanders.
    pub path_length: Real,
    /// Furthest the organism got from its start at any point, which catches
    /// creatures that travel and then fall back.
    pub max_displacement: Real,
    /// Mean height of the centre of mass.
    pub mean_height: Real,
    /// Seconds spent with the root block's local up axis within ~45 degrees of
    /// world up.
    pub upright_seconds: Real,
    /// Accumulated absolute motor angular impulse over the measured window: a
    /// proxy for effort.
    ///
    /// Includes the impulse spent *holding* a joint still, because that is what
    /// a real actuator spends too. An organism that does nothing is not free.
    ///
    /// Excludes the settle drop. The controller is held off there, so charging
    /// for it would price a fall the organism could not influence — and would
    /// make the charge scale with `settle_time`, mass and hinge count, turning
    /// `energy_penalty` into a morphology penalty applied before the controller
    /// has any say.
    pub actuation: Real,
    /// Measured window length, seconds.
    pub duration: Real,
    pub steps: u32,
    /// Highest the centre of mass reached during the measured window, absolute.
    /// Compare against `start.y` to get height gained.
    #[serde(default)]
    pub peak_height: Real,
    /// Seconds with no attached part touching the ground: hang time.
    ///
    /// Measured over attached parts only, so shedding a limb and watching it
    /// bounce away does not read as flight.
    #[serde(default)]
    pub airborne_seconds: Real,
    /// Displacement along the direction the organism was told to travel.
    ///
    /// Identical to `displacement_x` in an experiment that never steers, because
    /// the standing command there is +X. In one that does, this is the only
    /// distance measure that means anything: an organism sent left and rewarded
    /// for going right has learned nothing worth having.
    #[serde(default)]
    pub heading_progress: Real,
    /// Joints that wore out and let a limb detach during the run.
    ///
    /// An outcome, not a rendering detail: this is recorded for *every*
    /// organism, so how often bodies tear themselves apart can be measured
    /// across a whole run without opening a single replay. Always zero for an
    /// experiment whose joints cannot break.
    #[serde(default)]
    pub joints_lost: u32,
    /// The solver gave up on this organism. Its metrics are meaningless.
    pub diverged: bool,
}

impl Metrics {
    #[inline]
    pub fn mean_speed(&self) -> Real {
        if self.duration > 0.0 {
            self.displacement / self.duration
        } else {
            0.0
        }
    }
}

/// Score assigned to an organism whose simulation diverged.
///
/// Large and negative rather than zero, so a broken body can never out-rank a
/// merely useless one, and so divergence is obvious in the statistics.
pub const DIVERGED_FITNESS: Real = -1000.0;

/// Reduce metrics to a single scalar under the configured objective.
pub fn score(cfg: &FitnessCfg, m: &Metrics) -> Real {
    if m.diverged || !m.displacement.is_finite() {
        return DIVERGED_FITNESS;
    }
    let base = match cfg.objective {
        Objective::Distance => m.displacement,
        Objective::DistanceX => m.displacement_x,
        Objective::Speed => m.mean_speed(),
        Objective::Heading => m.heading_progress,
    };
    // Height is measured from where the organism started rather than from the
    // ground, so simply being tall is worth nothing — only *gaining* height is.
    let climbed = (m.peak_height - m.start.y).max(0.0);
    base + cfg.upright_bonus * m.upright_seconds - cfg.energy_penalty * m.actuation
        + cfg.air_bonus * m.airborne_seconds
        + cfg.height_bonus * climbed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::vec3;

    fn metrics() -> Metrics {
        Metrics {
            start: Vec3::ZERO,
            end: vec3(3.0, 0.5, 4.0),
            displacement: 5.0,
            displacement_x: 3.0,
            path_length: 7.0,
            max_displacement: 5.5,
            mean_height: 0.4,
            upright_seconds: 6.0,
            actuation: 100.0,
            duration: 10.0,
            steps: 1200,
            heading_progress: 3.0,
            peak_height: 0.0,
            airborne_seconds: 0.0,
            joints_lost: 0,
            diverged: false,
        }
    }

    /// Height is scored from where the organism started, so a tall organism
    /// that never leaves the ground earns nothing for its stature.
    #[test]
    fn only_height_actually_gained_is_rewarded() {
        let cfg = FitnessCfg { height_bonus: 10.0, ..Default::default() };
        let mut m = metrics();
        m.start = vec3(0.0, 0.9, 0.0); // starts tall
        m.peak_height = 0.9; // and never rises
        let flat = score(&cfg, &m);
        m.peak_height = 1.4; // now it jumps half a metre
        assert!(
            (score(&cfg, &m) - flat - 5.0).abs() < 1e-4,
            "half a metre gained should be worth height_bonus/2"
        );

        // Sinking below the starting height is worth zero, not a penalty: the
        // objective already charges for going nowhere.
        m.peak_height = 0.2;
        assert!((score(&cfg, &m) - flat).abs() < 1e-4);
    }

    #[test]
    fn hang_time_is_rewarded_per_second() {
        let cfg = FitnessCfg { air_bonus: 2.0, ..Default::default() };
        let mut m = metrics();
        m.airborne_seconds = 0.0;
        let grounded = score(&cfg, &m);
        m.airborne_seconds = 1.5;
        assert!((score(&cfg, &m) - grounded - 3.0).abs() < 1e-4);
    }

    #[test]
    fn objectives_read_the_field_they_claim_to() {
        let m = metrics();
        let with = |objective| FitnessCfg { objective, ..Default::default() };

        assert_eq!(score(&with(Objective::Distance), &m), 5.0);
        assert_eq!(score(&with(Objective::DistanceX), &m), 3.0);
        assert_eq!(score(&with(Objective::Speed), &m), 0.5);
    }

    #[test]
    fn penalties_and_bonuses_apply() {
        let m = metrics();
        let cfg = FitnessCfg {
            objective: Objective::Distance,
            energy_penalty: 0.01,
            upright_bonus: 0.5,
            ..Default::default()
        };
        // 5 + 0.5*6 - 0.01*100
        assert!((score(&cfg, &m) - 7.0).abs() < 1e-5);
    }

    #[test]
    fn divergence_dominates_everything() {
        let mut m = metrics();
        m.diverged = true;
        let cfg = FitnessCfg::default();
        assert_eq!(score(&cfg, &m), DIVERGED_FITNESS);

        let mut nan = metrics();
        nan.displacement = Real::NAN;
        assert_eq!(score(&cfg, &nan), DIVERGED_FITNESS);
    }

    #[test]
    fn mean_speed_handles_zero_duration() {
        let mut m = metrics();
        m.duration = 0.0;
        assert_eq!(m.mean_speed(), 0.0);
    }
}
