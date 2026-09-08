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
    /// Net elevation ended *above* the settled start, in metres; zero if the
    /// organism finished level or lower.
    ///
    /// Deliberately not derived from `end.y - start.y` at scoring time. Both are
    /// clamped at zero per trial and only then averaged, and `max(0, mean)` is
    /// not `mean(max(0, ..))` — an organism that climbs on one trial and falls on
    /// another has genuinely gained on one of them.
    ///
    /// Unexploitable by construction: the only way to raise it is to end higher.
    #[serde(default)]
    pub net_gain: Real,
    /// Net elevation ended *below* the settled start, in metres; zero if level
    /// or higher. The mirror of [`Metrics::net_gain`], clamped the same way.
    #[serde(default)]
    pub net_loss: Real,
    /// Total ascent over the run, hysteresis-filtered by
    /// `fitness.climb_deadband`.
    ///
    /// Measures how much climbing was *done* rather than where the organism
    /// ended up, so a hill climbed and descended still counts. That richness is
    /// also the risk: an unfiltered version pays per unit of vertical wobble,
    /// and an organism bobbing on the spot would accumulate ascent forever. The
    /// deadband is what stops that, and `bobbing_on_the_spot_earns_no_climb`
    /// is the test that keeps it stopped.
    #[serde(default)]
    pub climb: Real,
    /// Total descent over the run, filtered by the same band as
    /// [`Metrics::climb`].
    #[serde(default)]
    pub descent: Real,
    /// Height lost while no attached part was touching the ground: falling, as
    /// opposed to walking downhill.
    ///
    /// `descent` charges a controlled descent exactly what it charges a tumble,
    /// and an objective built on it makes standing still the best answer —
    /// an organism that never moves never loses height. This separates the two.
    /// Losing contact is what distinguishes a fall from a step down, and the
    /// same clearance margin `airborne_seconds` uses decides it.
    #[serde(default)]
    pub fall_distance: Real,
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
        // Elevation, as two independent questions: how much height was kept, and
        // how much was given away. Kept separate rather than netted so that an
        // experiment can pay for one without implying the other — asking for
        // climbing is not the same as forbidding descent.
        + cfg.climb_bonus * m.net_gain
        - cfg.descent_penalty * m.net_loss
        + cfg.cumulative_climb_bonus * m.climb
        - cfg.cumulative_descent_penalty * m.descent
        // Charged for height given away while out of contact, which is the part
        // of a descent the organism did not choose. Distinct from
        // `descent_penalty`, which cannot tell a controlled walk downhill from a
        // fall and therefore prices immobility as the safest strategy.
        - cfg.fall_penalty * m.fall_distance
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
            net_gain: 0.0,
            net_loss: 0.0,
            climb: 0.0,
            descent: 0.0,
            fall_distance: 0.0,
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

    /// Elevation ended above the start is paid for; elevation given away is
    /// charged for. The two are independent, so an experiment can ask for
    /// climbing without also forbidding descent.
    #[test]
    fn elevation_is_paid_for_in_both_directions() {
        let cfg = FitnessCfg { climb_bonus: 10.0, descent_penalty: 4.0, ..Default::default() };
        let mut m = metrics();
        let level = score(&cfg, &m);

        m.net_gain = 0.5;
        assert!((score(&cfg, &m) - level - 5.0).abs() < 1e-4, "half a metre up is worth 5");

        m.net_gain = 0.0;
        m.net_loss = 0.5;
        assert!((score(&cfg, &m) - level + 2.0).abs() < 1e-4, "half a metre down costs 2");
    }

    /// The failure mode `energy_penalty` already documents, in its sharper form.
    ///
    /// An organism that never moves loses no elevation, and unlike actuation it
    /// is not even charged for standing there. If the descent penalty outweighs
    /// what distance pays, evolution's best answer is to do nothing — so the
    /// objective has to keep a term that immobility cannot satisfy.
    #[test]
    fn standing_still_does_not_beat_travelling() {
        // Weights in the range the plan proposes: elevation dominant per metre,
        // distance still the base objective.
        let cfg = FitnessCfg {
            objective: Objective::DistanceX,
            climb_bonus: 10.0,
            descent_penalty: 4.0,
            ..Default::default()
        };

        let still = Metrics { diverged: false, ..Default::default() };

        // A champion of the shipped fractal experiment: travels well, and gives
        // away half a metre of height doing it.
        let mut mover = Metrics { diverged: false, ..Default::default() };
        mover.displacement = 5.5;
        mover.displacement_x = 5.5;
        mover.net_loss = 0.5;

        assert!(
            score(&cfg, &mover) > score(&cfg, &still),
            "doing nothing scored {} against {} for an organism that travelled 5.5 m;              lower descent_penalty or keep a larger distance term",
            score(&cfg, &still),
            score(&cfg, &mover)
        );
    }

    /// The distinction the term exists to make.
    ///
    /// Two organisms end the run the same distance below where they started.
    /// One walked down; one fell. `descent_penalty` cannot tell them apart and
    /// charges both, which is why an objective built on it prices standing
    /// still as the safest strategy. `fall_penalty` charges only the one that
    /// lost contact with the ground.
    #[test]
    fn falling_is_charged_where_walking_downhill_is_not() {
        let cfg = FitnessCfg { fall_penalty: 8.0, ..Default::default() };

        let mut walker = metrics();
        walker.net_loss = 0.5;
        walker.fall_distance = 0.0;

        let mut faller = metrics();
        faller.net_loss = 0.5;
        faller.fall_distance = 0.5;

        assert_eq!(score(&cfg, &walker), score(&cfg, &metrics()), "descending on foot is free");
        assert!(
            (score(&cfg, &walker) - score(&cfg, &faller) - 4.0).abs() < 1e-4,
            "half a metre dropped should cost fall_penalty/2"
        );
    }

    /// And the failure mode it removes: under `descent_penalty`, doing nothing
    /// beats going somewhere and losing a little height. Under `fall_penalty`
    /// it does not.
    #[test]
    fn fall_penalty_does_not_reward_standing_still() {
        let still = Metrics { diverged: false, ..Default::default() };
        let mut walker = Metrics { diverged: false, ..Default::default() };
        walker.displacement = 3.0;
        walker.displacement_x = 3.0;
        walker.net_loss = 0.4; // walked down a slope, never left the ground
        walker.fall_distance = 0.0;

        let harsh = FitnessCfg {
            objective: Objective::DistanceX,
            descent_penalty: 8.0,
            ..Default::default()
        };
        assert!(
            score(&harsh, &still) > score(&harsh, &walker),
            "the behaviour being replaced: descent_penalty makes immobility win"
        );

        let targeted =
            FitnessCfg { objective: Objective::DistanceX, fall_penalty: 8.0, ..Default::default() };
        assert!(
            score(&targeted, &walker) > score(&targeted, &still),
            "fall_penalty must leave a controlled descent worth making"
        );
    }

    /// The weights are the only thing that turns these terms on, so a default
    /// configuration must score exactly as it did before they existed.
    #[test]
    fn elevation_terms_are_inert_by_default() {
        let cfg = FitnessCfg::default();
        let mut m = metrics();
        let before = score(&cfg, &m);
        m.net_gain = 3.0;
        m.net_loss = 3.0;
        m.climb = 9.0;
        m.descent = 9.0;
        m.fall_distance = 9.0;
        assert_eq!(score(&cfg, &m), before, "an unweighted elevation term changed the score");
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
