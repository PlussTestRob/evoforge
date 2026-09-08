//! The genome: the heritable description of an organism.
//!
//! A genome is deliberately *not* a simulatable object. It is a compact,
//! serialisable description that [`crate::phenotype`] expands into bodies and
//! joints. Keeping the two apart means the expression rules (how a block is
//! placed, how mass is derived) can change without invalidating stored genomes
//! in a way we cannot detect.
//!
//! # Structure
//!
//! The body is a tree of parts, stored flat with the invariant
//! `parts[i].parent < i`. Part 0 is the root and has no joint. That ordering
//! makes construction a single forward pass and makes cycles unrepresentable.
//!
//! Each part also carries a [`PartGene::slot`], a stable identifier used to index
//! controller inputs and outputs. See [`crate::brain`] for why.

use serde::{Deserialize, Serialize};

use crate::brain::BrainLayout;
use crate::config::{fnv1a, BodyLimits, BrainCfg, MutationParams};
use crate::math::{clamp, vec3, Real, Vec3};
use crate::rng::Rng;

/// How a part is attached to its parent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JointKind {
    /// Rigid weld. Lets evolution build compound shapes out of blocks.
    Fixed,
    /// Single-axis motorised hinge with symmetric limits.
    Hinge,
}

/// The joint connecting a part to its parent.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointGene {
    pub kind: JointKind,
    /// Which of the attachment face's two tangent directions is the hinge axis
    /// (0 or 1). Restricting the axis to the face frame keeps every generated
    /// hinge geometrically sensible without needing to evolve a unit vector.
    pub axis: u8,
    /// Symmetric limit in radians: the joint is constrained to `[-limit, limit]`.
    pub limit: Real,
    /// Maximum angular velocity the motor will drive, rad/s.
    pub motor_speed: Real,
    /// Maximum torque the motor may apply, N m.
    pub motor_torque: Real,
}

impl JointGene {
    fn random(rng: &mut Rng, limits: &BodyLimits) -> JointGene {
        let kind =
            if rng.chance(limits.hinge_probability) { JointKind::Hinge } else { JointKind::Fixed };
        JointGene {
            kind,
            axis: rng.below(2) as u8,
            limit: rng.range(limits.min_joint_limit, limits.max_joint_limit),
            motor_speed: rng.range(0.2 * limits.max_motor_speed, limits.max_motor_speed),
            motor_torque: rng.range(0.2 * limits.max_motor_torque, limits.max_motor_torque),
        }
    }

    fn clamp_to(&mut self, limits: &BodyLimits) {
        self.axis &= 1;
        self.limit = clamp(self.limit, limits.min_joint_limit, limits.max_joint_limit);
        self.motor_speed = clamp(self.motor_speed, 0.0, limits.max_motor_speed);
        self.motor_torque = clamp(self.motor_torque, 0.0, limits.max_motor_torque);
    }
}

/// Which primitive a part's block is carved from.
///
/// The gene is only the *kind*. Dimensions always come from
/// [`PartGene::half_extents`], and every shape is inscribed in that box, so one
/// size gene keeps doing one job and the existing size operator mutates a
/// capsule as sensibly as it mutates a cuboid. [`crate::phenotype::carve`] is
/// the rule that turns the two into geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShapeKind {
    /// Fills the box.
    #[default]
    Box,
    /// Rectangular frustum along the box's longest axis: a wedge, a foot, a claw.
    Taper,
    /// Inscribed sphere. One ground contact, so it rolls.
    Sphere,
    /// Inscribed capsule along the box's longest axis. Rolls sideways, slides
    /// lengthwise, and does not catch on its corners the way a cuboid does.
    Capsule,
    /// Inscribed cylinder along the box's longest axis. A wheel, or a limb that
    /// can stand on its end.
    Cylinder,
}

/// One block of the body.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PartGene {
    /// Stable controller index, unique within the genome and `< max_parts`.
    pub slot: u8,
    /// Index of the parent part; always less than this part's own index.
    /// Meaningless for part 0.
    pub parent: u8,
    /// Half-extents of the box the part is inscribed in, metres.
    pub half_extents: Vec3,
    /// Which primitive is carved from that box. Defaults to a box, so genomes
    /// recorded before shapes existed load unchanged.
    #[serde(default)]
    pub shape: ShapeKind,
    /// Whether this part appears twice, mirrored across the sagittal plane.
    ///
    /// Bilateral symmetry is the most animal-defining property a body has, and a
    /// free tree of parts makes it no cheaper to express than any of the vastly
    /// more numerous lopsided arrangements. One bit here makes a matched pair as
    /// easy to say as a single limb — and, because both copies share a
    /// controller slot, as easy to *control* as one.
    #[serde(default)]
    pub paired: bool,
    /// For a mirrored pair, whether the two halves are driven in opposition.
    ///
    /// A reflected limb given the same command moves as its reflection, which
    /// for a fore-and-aft hinge means the two halves alternate — a walk.
    /// Negating one gives a bound. Both are real gaits, so evolution chooses.
    #[serde(default)]
    pub antiphase: bool,
    /// How many copies of this part are chained end to end. 1 is a single part.
    ///
    /// Repetition is where spines, tails and segmented limbs come from. Children
    /// attach to the last segment, so a chain extends rather than branching.
    #[serde(default = "one")]
    pub repeat: u8,
    /// Whether this part carries a range sensor.
    ///
    /// A sensor is not a fitting bolted to a part — it *is* carried by a part,
    /// which has mass, hangs off a joint and is aimed by whatever drives that
    /// joint. Perception therefore costs what any other limb costs, and a long
    /// stalk that can sweep a wide arc is heavy and destabilising. Evolution
    /// pays for looking.
    #[serde(default)]
    pub sensor: bool,
    /// Direction the sensor faces, in the part's own frame. Normalised on use,
    /// so mutation may perturb the components freely.
    #[serde(default)]
    pub sensor_dir: Vec3,
    /// Which face of the parent this part attaches to (0..6, see
    /// [`crate::phenotype::FACES`]). Meaningless for part 0.
    pub attach_face: u8,
    /// Position on the parent face in normalised face coordinates, `[-1, 1]`.
    pub attach_u: Real,
    pub attach_v: Real,
    /// Joint to the parent. Meaningless for part 0.
    pub joint: JointGene,
}

impl PartGene {
    fn clamp_to(&mut self, limits: &BodyLimits) {
        let lo = limits.min_half_extent;
        let hi = limits.max_half_extent;
        self.half_extents = vec3(
            clamp(self.half_extents.x, lo, hi),
            clamp(self.half_extents.y, lo, hi),
            clamp(self.half_extents.z, lo, hi),
        );
        self.attach_face %= 6;
        self.attach_u = clamp(self.attach_u, -1.0, 1.0);
        self.attach_v = clamp(self.attach_v, -1.0, 1.0);
        // A genome carried into an experiment that does not allow its shape
        // falls back to the first shape that experiment does allow, the same way
        // an out-of-range motor torque is pulled into bounds rather than refused.
        if !limits.shapes.is_empty() && !limits.shapes.contains(&self.shape) {
            self.shape = limits.shapes[0];
        }
        if limits.pair_probability <= 0.0 {
            self.paired = false;
            self.antiphase = false;
        }
        self.repeat = clamp_u8(self.repeat.max(1), 1, limits.max_repeat.max(1));
        self.joint.clamp_to(limits);
    }
}

/// Default segment count for a part in a genome recorded before repetition
/// existed: exactly one, which is what it meant.
fn one() -> u8 {
    1
}

/// The complete heritable description of an organism.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub parts: Vec<PartGene>,
    /// Controller parameters, laid out by [`BrainLayout`]. Fixed length for a
    /// given experiment, which is what makes crossover a simple aligned choice.
    pub weights: Vec<Real>,
    /// How hard this organism is willing to drive its motors, in `[0, 1]`, where
    /// 0 is reckless and 1 is maximally restrained.
    ///
    /// Only consulted when joints can break. It is a standing disposition rather
    /// than a reaction, which is the point: an organism does not know how long
    /// its trial will last, so whether to sprint and risk tearing a joint or to
    /// pace itself has to be a heritable bet rather than something it can work
    /// out from what it senses. The per-joint health input is the reactive half
    /// of the same trade.
    #[serde(default)]
    pub caution: Real,
}

impl Genome {
    /// Generate a random genome.
    pub fn random(
        rng: &mut Rng,
        limits: &BodyLimits,
        brain: &BrainCfg,
        layout: &BrainLayout,
    ) -> Genome {
        let n =
            limits.min_parts + rng.below((limits.max_parts - limits.min_parts + 1) as u32) as usize;

        let mut parts = Vec::with_capacity(n);
        for i in 0..n {
            let parent = if i == 0 { 0 } else { rng.below(i as u32) as u8 };
            parts.push(PartGene {
                slot: i as u8,
                parent,
                half_extents: random_extents(rng, limits),
                shape: random_shape(rng, limits),
                paired: random_paired(rng, limits),
                antiphase: random_antiphase(rng, limits),
                repeat: random_repeat(rng, limits),
                sensor: random_sensor(rng, limits),
                sensor_dir: random_sensor_dir(rng, limits),
                attach_face: rng.below(6) as u8,
                attach_u: rng.range(-0.7, 0.7),
                attach_v: rng.range(-0.7, 0.7),
                joint: JointGene::random(rng, limits),
            });
        }

        let weights = (0..layout.weight_count())
            .map(|_| {
                clamp(rng.normal_scaled(brain.init_sigma), -brain.weight_limit, brain.weight_limit)
            })
            .collect();

        Genome { parts, weights, caution: random_caution(rng, limits) }
    }

    #[inline]
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Number of joints, which is one fewer than the number of parts because the
    /// root is unattached.
    #[inline]
    pub fn joint_count(&self) -> usize {
        self.parts.len().saturating_sub(1)
    }

    pub fn hinge_count(&self) -> usize {
        self.parts[1..].iter().filter(|p| p.joint.kind == JointKind::Hinge).count()
    }

    /// A fingerprint of the *morphology* only, ignoring controller weights and
    /// continuous sizes. Used to report how much structural diversity survives in
    /// a population; a population of one structure is a warning sign even when
    /// fitness is still climbing.
    pub fn structure_hash(&self) -> u64 {
        let mut bytes = Vec::with_capacity(self.parts.len() * 8);
        for (i, p) in self.parts.iter().enumerate() {
            bytes.push(if i == 0 { 255 } else { p.parent });
            bytes.push(if i == 0 { 0 } else { p.attach_face });
            bytes.push(match p.joint.kind {
                JointKind::Fixed => 0,
                JointKind::Hinge => 1,
            });
            bytes.push(if i == 0 { 0 } else { p.joint.axis });
            // Quantise sizes coarsely: two genomes differing by a millimetre are
            // the same structure for this purpose.
            for e in [p.half_extents.x, p.half_extents.y, p.half_extents.z] {
                bytes.push((e * 20.0) as u8);
            }
        }
        // Appended only when some part is not a box, so that a box-only genome
        // hashes exactly as it did before shapes existed. Two organisms that
        // differ only in shape are genuinely different structures, so when
        // shapes are in use they have to be part of the identity.
        if self.parts.iter().any(|p| p.shape != ShapeKind::Box) {
            for p in &self.parts {
                bytes.push(p.shape as u8);
            }
        }
        fnv1a(&bytes)
    }

    /// Check the structural invariants this module promises to maintain.
    pub fn is_valid(&self, limits: &BodyLimits, layout: &BrainLayout) -> bool {
        if self.parts.is_empty()
            || self.parts.len() < limits.min_parts
            || self.parts.len() > limits.max_parts
        {
            return false;
        }
        if self.weights.len() != layout.weight_count() {
            return false;
        }
        let mut seen_slots = vec![false; layout.max_slots];
        for (i, p) in self.parts.iter().enumerate() {
            if i > 0 && (p.parent as usize) >= i {
                return false;
            }
            let s = p.slot as usize;
            if s >= layout.max_slots || seen_slots[s] {
                return false;
            }
            seen_slots[s] = true;
            if p.attach_face > 5 || p.joint.axis > 1 {
                return false;
            }
            if !p.half_extents.is_finite() {
                return false;
            }
        }
        self.weights.iter().all(|w| w.is_finite())
    }

    fn used_slots(&self) -> Vec<bool> {
        let mut used = vec![false; 256];
        for p in &self.parts {
            used[p.slot as usize] = true;
        }
        used
    }

    /// Lowest slot index not currently in use, if any is available.
    fn free_slot(&self, max_slots: usize) -> Option<u8> {
        let used = self.used_slots();
        (0..max_slots).find(|&s| !used[s]).map(|s| s as u8)
    }

    /// Indices of parts with no children. Only leaves can be deleted without
    /// re-parenting, which keeps the removal operator trivial and structure
    /// preserving.
    fn leaf_indices(&self) -> Vec<usize> {
        let mut has_child = vec![false; self.parts.len()];
        for p in self.parts.iter().skip(1) {
            has_child[p.parent as usize] = true;
        }
        (1..self.parts.len()).filter(|&i| !has_child[i]).collect()
    }
}

/// Draw a shape, consuming randomness only when there is a choice to make.
///
/// The guard is not an optimisation. An experiment that has not opted into
/// shapes has to produce exactly the random stream it produced before shapes
/// existed, or every result recorded before this feature becomes unreproducible
/// and the constants in `tests/golden.rs` become lies.
/// Draw a caution disposition, but only for an experiment whose joints can
/// break. Same discipline as [`random_shape`]: a feature that is switched off
/// must not move the random stream.
/// Whether a freshly drawn part carries a sensor.
///
/// Draws *nothing* when the experiment has no sensors, which is what makes the
/// feature exactly off: the random stream is identical to the one an experiment
/// drew before sensors existed, so every earlier result reproduces bit for bit.
fn random_sensor(rng: &mut Rng, limits: &BodyLimits) -> bool {
    if limits.sensor_probability <= 0.0 {
        return false;
    }
    rng.unit() < limits.sensor_probability
}

/// Where a freshly drawn sensor points, in its part's frame.
///
/// Biased forward and downward — the direction terrain is in — so that a new
/// sensor starts somewhere useful rather than facing the sky. Evolution moves it
/// from there.
fn random_sensor_dir(rng: &mut Rng, limits: &BodyLimits) -> Vec3 {
    if limits.sensor_probability <= 0.0 {
        return Vec3::ZERO;
    }
    vec3(1.0 + rng.signed() * 0.3, -0.5 + rng.signed() * 0.4, rng.signed() * 0.5)
        .normalize_or(vec3(1.0, -0.5, 0.0).normalize_or(Vec3::X))
}

fn random_caution(rng: &mut Rng, limits: &BodyLimits) -> Real {
    if limits.joint_endurance > 0.0 {
        rng.unit()
    } else {
        0.0
    }
}

/// Draw whether a part is a mirrored pair. Spends nothing when bilateral
/// symmetry is switched off, on the same discipline as every other opt-in gene.
fn random_paired(rng: &mut Rng, limits: &BodyLimits) -> bool {
    limits.pair_probability > 0.0 && rng.chance(limits.pair_probability)
}

/// Which way the halves of a pair are driven. Only meaningful for a pair, but
/// drawn whenever pairing is on so the gene is available to mutation.
fn random_antiphase(rng: &mut Rng, limits: &BodyLimits) -> bool {
    limits.pair_probability > 0.0 && rng.chance(0.5)
}

/// Draw a segment count. One segment unless segmentation is enabled.
fn random_repeat(rng: &mut Rng, limits: &BodyLimits) -> u8 {
    if limits.max_repeat > 1 {
        1 + rng.below(limits.max_repeat as u32) as u8
    } else {
        1
    }
}

fn random_shape(rng: &mut Rng, limits: &BodyLimits) -> ShapeKind {
    match limits.shapes.len() {
        0 => ShapeKind::Box,
        1 => limits.shapes[0],
        n => limits.shapes[rng.pick(n)],
    }
}

fn clamp_u8(v: u8, lo: u8, hi: u8) -> u8 {
    v.max(lo).min(hi)
}

fn random_extents(rng: &mut Rng, limits: &BodyLimits) -> Vec3 {
    vec3(
        rng.range(limits.min_half_extent, limits.max_half_extent),
        rng.range(limits.min_half_extent, limits.max_half_extent),
        rng.range(limits.min_half_extent, limits.max_half_extent),
    )
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

/// Mutate `genome` in place.
///
/// Operators are independent and each is gated by its own rate, so a single
/// experiment can dial morphological search separately from controller search.
pub fn mutate(
    genome: &mut Genome,
    rng: &mut Rng,
    params: &MutationParams,
    limits: &BodyLimits,
    brain: &BrainCfg,
    layout: &BrainLayout,
) {
    // Controller weights.
    for w in genome.weights.iter_mut() {
        if rng.chance(params.weight_rate) {
            if rng.chance(params.weight_reset_rate) {
                *w = rng.normal_scaled(brain.init_sigma);
            } else {
                *w += rng.normal_scaled(params.weight_sigma);
            }
            *w = clamp(*w, -brain.weight_limit, brain.weight_limit);
        }
    }

    if limits.joint_endurance > 0.0 && rng.chance(params.caution_rate) {
        genome.caution = clamp(genome.caution + rng.normal_scaled(params.caution_sigma), 0.0, 1.0);
    }

    // Morphology.
    for i in 0..genome.parts.len() {
        let is_root = i == 0;
        let p = &mut genome.parts[i];

        if rng.chance(params.size_rate) {
            let axis = rng.below(3);
            let delta = rng.normal_scaled(params.size_sigma);
            match axis {
                0 => p.half_extents.x += delta,
                1 => p.half_extents.y += delta,
                _ => p.half_extents.z += delta,
            }
        }

        // Order matters: the length check has to short-circuit before `chance`,
        // or a box-only experiment would consume a random number here that it
        // did not consume before shapes existed.
        if limits.shapes.len() > 1 && rng.chance(params.shape_rate) {
            p.shape = random_shape(rng, limits);
        }

        // Guarded on the experiment having sensors before `chance` is called, so
        // a sensorless run consumes no randomness here and keeps the stream it
        // had before sensing existed.
        if limits.sensor_probability > 0.0 {
            if rng.chance(params.sensor_rate) {
                p.sensor = !p.sensor;
                if p.sensor && p.sensor_dir.length_sq() < 1e-6 {
                    p.sensor_dir = random_sensor_dir(rng, limits);
                }
            }
            if p.sensor && rng.chance(params.sensor_rate) {
                // Aim drifts by perturbing the raw components; normalisation
                // happens where it is used, so no angle can wrap or degenerate.
                let s = params.sensor_dir_sigma;
                p.sensor_dir = vec3(
                    p.sensor_dir.x + rng.normal_scaled(s),
                    p.sensor_dir.y + rng.normal_scaled(s),
                    p.sensor_dir.z + rng.normal_scaled(s),
                )
                .normalize_or(vec3(1.0, -0.5, 0.0).normalize_or(Vec3::X));
            }
        }

        if limits.pair_probability > 0.0 && rng.chance(params.pair_rate) {
            // One operator flips both halves of the body-plan decision: whether
            // the part is a pair at all, and how a pair is driven.
            if rng.chance(0.5) {
                p.paired = !p.paired;
            } else {
                p.antiphase = !p.antiphase;
            }
        }
        if limits.max_repeat > 1 && rng.chance(params.repeat_rate) {
            p.repeat = random_repeat(rng, limits);
        }

        if !is_root {
            if rng.chance(params.attach_rate) {
                p.attach_u += rng.normal_scaled(params.attach_sigma);
                p.attach_v += rng.normal_scaled(params.attach_sigma);
            }
            if rng.chance(params.attach_rate * 0.5) {
                p.attach_face = rng.below(6) as u8;
            }
            if rng.chance(params.joint_limit_rate) {
                p.joint.limit += rng.normal_scaled(params.joint_limit_sigma);
            }
            if rng.chance(params.joint_axis_rate) {
                p.joint.axis ^= 1;
            }
            if rng.chance(params.joint_kind_rate) {
                p.joint.kind = match p.joint.kind {
                    JointKind::Fixed => JointKind::Hinge,
                    JointKind::Hinge => JointKind::Fixed,
                };
            }
            if rng.chance(params.motor_rate) {
                p.joint.motor_speed +=
                    rng.normal_scaled(params.motor_sigma * limits.max_motor_speed);
                p.joint.motor_torque +=
                    rng.normal_scaled(params.motor_sigma * limits.max_motor_torque);
            }
        }

        p.clamp_to(limits);
    }

    // Topology. At most one structural change per genome per generation:
    // morphology mutations are far more disruptive than weight mutations, and
    // stacking them makes offspring almost never viable.
    if genome.parts.len() > limits.min_parts && rng.chance(params.remove_part_rate) {
        remove_random_leaf(genome, rng);
    } else if genome.parts.len() < limits.max_parts && rng.chance(params.add_part_rate) {
        add_random_part(genome, rng, limits, layout);
    }
}

fn add_random_part(genome: &mut Genome, rng: &mut Rng, limits: &BodyLimits, layout: &BrainLayout) {
    let Some(slot) = genome.free_slot(layout.max_slots) else {
        return;
    };
    let parent = rng.below(genome.parts.len() as u32) as u8;
    let mut part = PartGene {
        slot,
        parent,
        half_extents: random_extents(rng, limits),
        shape: random_shape(rng, limits),
        paired: random_paired(rng, limits),
        antiphase: random_antiphase(rng, limits),
        repeat: random_repeat(rng, limits),
        sensor: random_sensor(rng, limits),
        sensor_dir: random_sensor_dir(rng, limits),
        attach_face: rng.below(6) as u8,
        attach_u: rng.range(-0.7, 0.7),
        attach_v: rng.range(-0.7, 0.7),
        joint: JointGene::random(rng, limits),
    };
    part.clamp_to(limits);
    // Appending preserves `parent < index` by construction.
    genome.parts.push(part);
}

fn remove_random_leaf(genome: &mut Genome, rng: &mut Rng) {
    let leaves = genome.leaf_indices();
    if leaves.is_empty() {
        return;
    }
    let victim = leaves[rng.pick(leaves.len())];
    genome.parts.remove(victim);
    // Nothing referenced the leaf, but later parts shifted down by one.
    for p in genome.parts.iter_mut().skip(1) {
        if p.parent as usize > victim {
            p.parent -= 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Crossover
// ---------------------------------------------------------------------------

/// Produce one offspring from two parents.
///
/// `primary` supplies the topology; `secondary` contributes gene values for any
/// part whose slot it also has. Slots are what make this meaningful: a part in
/// slot 3 refers to the same lineage of limb in both parents even if it sits at a
/// different index in the tree.
///
/// Controller weights use uniform per-weight crossover, which is cheap and works
/// well when the vectors are aligned — which they are, by construction.
///
/// This operator is intentionally simple and is expected to be replaced;
/// morphological crossover is an open research question, not a solved one.
pub fn crossover(primary: &Genome, secondary: &Genome, rng: &mut Rng) -> Genome {
    let mut child = primary.clone();

    let mut secondary_by_slot = [usize::MAX; 256];
    for (i, p) in secondary.parts.iter().enumerate() {
        secondary_by_slot[p.slot as usize] = i;
    }

    for (i, part) in child.parts.iter_mut().enumerate() {
        let j = secondary_by_slot[part.slot as usize];
        if j == usize::MAX {
            continue;
        }
        let other = &secondary.parts[j];
        if rng.chance(0.5) {
            part.half_extents = other.half_extents;
            // Shape travels with the box it is carved from: inheriting a
            // capsule's proportions but a cuboid's geometry would be a
            // recombination of two things that only mean anything together. It
            // also costs no extra draw, so the stream is unchanged.
            part.shape = other.shape;
            part.paired = other.paired;
            part.antiphase = other.antiphase;
            part.repeat = other.repeat;
        }
        // Attachment and joint genes only mean anything for non-root parts, and
        // only if the other genome's part is also non-root.
        if i > 0 && j > 0 {
            if rng.chance(0.5) {
                part.attach_face = other.attach_face;
                part.attach_u = other.attach_u;
                part.attach_v = other.attach_v;
            }
            if rng.chance(0.5) {
                part.joint = other.joint;
            }
        }
    }

    // Same guard as everywhere else: with joint damage off both parents carry
    // exactly 0.0, so no draw is spent and the stream is unchanged. With it on,
    // caution recombines by the same uniform coin flip as every other gene.
    if (primary.caution != 0.0 || secondary.caution != 0.0) && rng.chance(0.5) {
        child.caution = secondary.caution;
    }

    debug_assert_eq!(child.weights.len(), secondary.weights.len());
    for (w, &o) in child.weights.iter_mut().zip(secondary.weights.iter()) {
        if rng.chance(0.5) {
            *w = o;
        }
    }

    child
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn fixture() -> (Config, BrainLayout) {
        let cfg = Config::default();
        let layout = cfg.brain_layout();
        (cfg, layout)
    }

    /// The compatibility invariant, stated directly: when there is only one
    /// shape to choose, choosing it must cost no randomness. If it ever does,
    /// every result recorded before shapes existed stops reproducing — which is
    /// what the constants in `tests/golden.rs` would then be quietly wrong about.
    #[test]
    fn a_single_shape_roster_spends_no_randomness() {
        let (mut cfg, layout) = fixture();
        let mut ends = Vec::new();
        for only in [ShapeKind::Box, ShapeKind::Sphere, ShapeKind::Cylinder] {
            cfg.body.shapes = vec![only];
            let mut rng = Rng::new(99);
            let mut g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            mutate(&mut g, &mut rng, &cfg.mutation, &cfg.body, &cfg.brain, &layout);
            // The state of the stream after the same work, for different shapes.
            ends.push(rng.next_u64());
            assert!(g.parts.iter().all(|p| p.shape == only));
        }
        assert!(
            ends.windows(2).all(|w| w[0] == w[1]),
            "the shape of a one-shape experiment changed the random stream: {ends:?}"
        );
    }

    /// And the other half: offering a choice does draw, so the guard above is
    /// not simply dead code that never picks anything.
    #[test]
    fn a_multi_shape_roster_actually_varies_shapes() {
        let (mut cfg, layout) = fixture();
        cfg.body.shapes =
            vec![ShapeKind::Box, ShapeKind::Sphere, ShapeKind::Capsule, ShapeKind::Cylinder];
        let mut seen = std::collections::HashSet::new();
        for seed in 0..50 {
            let mut rng = Rng::new(seed);
            let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            seen.extend(g.parts.iter().map(|p| p.shape));
        }
        assert_eq!(seen.len(), 4, "not every offered shape was ever drawn: {seen:?}");
    }

    /// The same compatibility invariant as for shapes, for the caution gene and
    /// the health input: with joint damage off, not one random number is spent
    /// on either, and the weight vector is the length it always was.
    #[test]
    fn joint_damage_off_spends_no_randomness_and_keeps_the_brain_layout() {
        let plain = Config::default();
        let mut healthy = Config::default();
        healthy.body.joint_endurance = 12.0;
        assert!(!plain.joints_can_break() && healthy.joints_can_break());

        let a = plain.brain_layout();
        let b = healthy.brain_layout();
        assert!(!a.senses_health() && b.senses_health());
        assert!(b.weight_count() > a.weight_count(), "enabling health should widen the controller");

        // Two genomes drawn from the same seed under the plain config: identical,
        // and the stream is left in the same place.
        let mut r1 = Rng::new(4242);
        let g1 = Genome::random(&mut r1, &plain.body, &plain.brain, &a);
        assert_eq!(g1.caution, 0.0);
        assert_eq!(g1.weights.len(), a.weight_count());

        let mut g2 = g1.clone();
        mutate(&mut g2, &mut r1, &plain.mutation, &plain.body, &plain.brain, &a);
        assert_eq!(g2.caution, 0.0, "caution drifted in an experiment that has no wear");

        // And with it on, caution is a real, varying trait.
        let mut seen = Vec::new();
        for seed in 0..20 {
            let mut r = Rng::new(seed);
            seen.push(Genome::random(&mut r, &healthy.body, &healthy.brain, &b).caution);
        }
        assert!(seen.iter().all(|c| (0.0..=1.0).contains(c)));
        assert!(seen.windows(2).any(|w| w[0] != w[1]), "caution never varied across seeds");
    }

    #[test]
    fn random_genomes_are_valid() {
        let (cfg, layout) = fixture();
        for seed in 0..200 {
            let mut rng = Rng::new(seed);
            let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            assert!(g.is_valid(&cfg.body, &layout), "invalid genome from seed {seed}");
        }
    }

    #[test]
    fn random_generation_is_reproducible() {
        let (cfg, layout) = fixture();
        let a = Genome::random(&mut Rng::new(7), &cfg.body, &cfg.brain, &layout);
        let b = Genome::random(&mut Rng::new(7), &cfg.body, &cfg.brain, &layout);
        assert_eq!(a, b);
        let c = Genome::random(&mut Rng::new(8), &cfg.body, &cfg.brain, &layout);
        assert_ne!(a, c);
    }

    #[test]
    fn mutation_preserves_validity() {
        let (cfg, layout) = fixture();
        let mut rng = Rng::new(3);
        let mut g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        for step in 0..3_000 {
            mutate(&mut g, &mut rng, &cfg.mutation, &cfg.body, &cfg.brain, &layout);
            assert!(g.is_valid(&cfg.body, &layout), "invalid after {step} mutations: {g:?}");
        }
    }

    #[test]
    fn mutation_is_reproducible() {
        let (cfg, layout) = fixture();
        let base = Genome::random(&mut Rng::new(11), &cfg.body, &cfg.brain, &layout);

        let mut a = base.clone();
        let mut ra = Rng::new(21);
        let mut b = base.clone();
        let mut rb = Rng::new(21);
        for _ in 0..50 {
            mutate(&mut a, &mut ra, &cfg.mutation, &cfg.body, &cfg.brain, &layout);
            mutate(&mut b, &mut rb, &cfg.mutation, &cfg.body, &cfg.brain, &layout);
        }
        assert_eq!(a, b);
        assert_ne!(a, base, "mutation had no effect at all");
    }

    #[test]
    fn mutation_eventually_changes_topology_in_both_directions() {
        let (cfg, layout) = fixture();
        let mut rng = Rng::new(5);
        let mut g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        let mut min_seen = g.part_count();
        let mut max_seen = g.part_count();
        for _ in 0..5_000 {
            mutate(&mut g, &mut rng, &cfg.mutation, &cfg.body, &cfg.brain, &layout);
            min_seen = min_seen.min(g.part_count());
            max_seen = max_seen.max(g.part_count());
        }
        assert_eq!(min_seen, cfg.body.min_parts);
        assert_eq!(max_seen, cfg.body.max_parts);
    }

    #[test]
    fn slots_stay_unique_across_add_and_remove() {
        let (cfg, layout) = fixture();
        let mut rng = Rng::new(13);
        let mut g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        for _ in 0..2_000 {
            mutate(&mut g, &mut rng, &cfg.mutation, &cfg.body, &cfg.brain, &layout);
            let mut slots: Vec<u8> = g.parts.iter().map(|p| p.slot).collect();
            slots.sort_unstable();
            let before = slots.len();
            slots.dedup();
            assert_eq!(before, slots.len(), "duplicate slots: {slots:?}");
        }
    }

    #[test]
    fn crossover_produces_valid_children() {
        let (cfg, layout) = fixture();
        let mut rng = Rng::new(31);
        for _ in 0..300 {
            let a = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            let b = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
            let c = crossover(&a, &b, &mut rng);
            assert!(c.is_valid(&cfg.body, &layout));
            assert_eq!(c.part_count(), a.part_count(), "topology comes from primary");
        }
    }

    #[test]
    fn crossover_of_identical_parents_is_the_parent() {
        let (cfg, layout) = fixture();
        let mut rng = Rng::new(41);
        let a = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        let c = crossover(&a, &a, &mut rng);
        assert_eq!(a, c);
    }

    #[test]
    fn crossover_mixes_weights_from_both_parents() {
        let (cfg, layout) = fixture();
        let mut rng = Rng::new(51);
        let a = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        let b = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        let c = crossover(&a, &b, &mut rng);
        let from_a = c.weights.iter().zip(&a.weights).filter(|(x, y)| x == y).count();
        let from_b = c.weights.iter().zip(&b.weights).filter(|(x, y)| x == y).count();
        assert!(from_a > 0 && from_b > 0, "{from_a} from a, {from_b} from b");
    }

    #[test]
    fn structure_hash_ignores_weights_but_not_topology() {
        let (cfg, layout) = fixture();
        let mut rng = Rng::new(61);
        let g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);

        let mut same_structure = g.clone();
        same_structure.weights[0] += 1.0;
        assert_eq!(g.structure_hash(), same_structure.structure_hash());

        let mut different = g.clone();
        different.parts[1].attach_face = (different.parts[1].attach_face + 1) % 6;
        assert_ne!(g.structure_hash(), different.structure_hash());
    }

    #[test]
    fn removal_only_targets_leaves_and_keeps_parent_ordering() {
        let (cfg, layout) = fixture();
        let mut rng = Rng::new(71);
        let mut g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
        for _ in 0..500 {
            if g.part_count() <= cfg.body.min_parts {
                g = Genome::random(&mut rng, &cfg.body, &cfg.brain, &layout);
                continue;
            }
            remove_random_leaf(&mut g, &mut rng);
            for (i, p) in g.parts.iter().enumerate().skip(1) {
                assert!((p.parent as usize) < i);
            }
        }
    }
}
