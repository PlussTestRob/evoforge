//! The organism's controller: a small fixed-topology feed-forward network.
//!
//! The network is *data*, not an object graph. A [`BrainLayout`] describes the
//! shape (shared by every organism in an experiment) and the genome carries a
//! flat `Vec<Real>` of weights. That makes the controller trivially cloneable,
//! serialisable, crossable and mutable, and evaluation is a couple of dense
//! matrix-vector products with no allocation.
//!
//! # Slot indexing
//!
//! Inputs and outputs are indexed by *slot*, not by body-part index. A slot is a
//! stable identifier assigned to a part when it is created and never reused
//! within a genome. If evolution deletes a limb, the remaining limbs keep their
//! slots and therefore keep the meaning of every weight that refers to them.
//! Indexing by part position instead would silently rewire the whole controller
//! whenever the morphology changed — the classic "competing conventions"
//! failure mode in evolved neural networks.

use serde::{Deserialize, Serialize};

use crate::math::{tanh_approx, Real};

/// Number of controller inputs that describe the organism as a whole, rather
/// than an individual slot.
pub const GLOBAL_INPUTS: usize = 9;

/// Global inputs when the experiment tells organisms where to go: the nine above
/// plus the commanded direction, as a horizontal unit vector.
///
/// Conditional for the same reason the health input is: the input count sets the
/// weight-vector length, and an experiment that does not steer has to keep the
/// one it always had.
pub const GLOBAL_INPUTS_WITH_COMMAND: usize = 11;

/// Inputs contributed by each slot when joints cannot be damaged: joint angle as
/// a (cos, sin) pair plus a ground-contact flag.
pub const INPUTS_PER_SLOT: usize = 3;

/// Inputs per slot when joint health is enabled: the three above plus the
/// joint's remaining health.
///
/// This is a separate constant rather than an unconditional fourth input
/// because the input count sets the weight-vector length, which sets how much
/// randomness a genome consumes. An experiment that has not enabled joint health
/// keeps [`INPUTS_PER_SLOT`] and therefore keeps every result it ever produced.
pub const INPUTS_PER_SLOT_WITH_HEALTH: usize = 4;

/// Fixed shape of every controller in an experiment.
///
/// Sizing for `max_slots` rather than the organism's actual part count means all
/// genomes in an experiment carry weight vectors of identical length, which is
/// what makes aligned crossover a one-liner. Unused slots simply feed zeros in
/// and have their outputs ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainLayout {
    pub max_slots: usize,
    pub hidden: usize,
    /// Inputs each slot contributes: [`INPUTS_PER_SLOT`], or
    /// [`INPUTS_PER_SLOT_WITH_HEALTH`] when the experiment lets joints break.
    pub slot_inputs: usize,
    /// Inputs describing the organism as a whole.
    pub global_inputs: usize,
}

impl BrainLayout {
    pub fn new(max_slots: usize, hidden: usize) -> BrainLayout {
        BrainLayout::with_health(max_slots, hidden, false)
    }

    pub fn with_health(max_slots: usize, hidden: usize, health: bool) -> BrainLayout {
        BrainLayout::new_with(max_slots, hidden, health, false)
    }

    pub fn new_with(max_slots: usize, hidden: usize, health: bool, steer: bool) -> BrainLayout {
        assert!(max_slots > 0 && hidden > 0);
        let slot_inputs = if health { INPUTS_PER_SLOT_WITH_HEALTH } else { INPUTS_PER_SLOT };
        let global_inputs = if steer { GLOBAL_INPUTS_WITH_COMMAND } else { GLOBAL_INPUTS };
        BrainLayout { max_slots, hidden, slot_inputs, global_inputs }
    }

    /// Whether this layout carries a commanded direction of travel.
    #[inline]
    pub fn is_steered(&self) -> bool {
        self.global_inputs >= GLOBAL_INPUTS_WITH_COMMAND
    }

    /// Whether this layout carries a health input for each slot.
    #[inline]
    pub fn senses_health(&self) -> bool {
        self.slot_inputs >= INPUTS_PER_SLOT_WITH_HEALTH
    }

    #[inline]
    pub fn inputs(&self) -> usize {
        self.global_inputs + self.slot_inputs * self.max_slots
    }

    #[inline]
    pub fn outputs(&self) -> usize {
        self.max_slots
    }

    /// Total number of evolvable parameters.
    #[inline]
    pub fn weight_count(&self) -> usize {
        let n_in = self.inputs();
        let n_out = self.outputs();
        self.hidden * n_in + self.hidden + n_out * self.hidden + n_out
    }

    /// Index of the first slot-local input for `slot`.
    #[inline]
    pub fn slot_input_base(&self, slot: usize) -> usize {
        self.global_inputs + slot * self.slot_inputs
    }
}

/// Global input indices, in the order the controller sees them.
pub mod input {
    pub const BIAS: usize = 0;
    pub const CLOCK_A: usize = 1;
    pub const CLOCK_B: usize = 2;
    pub const UP_Y: usize = 3;
    pub const RIGHT_Y: usize = 4;
    pub const VEL_X: usize = 5;
    pub const VEL_Y: usize = 6;
    pub const VEL_Z: usize = 7;
    pub const HEIGHT: usize = 8;
    /// Commanded direction of travel, a horizontal unit vector. Present only in
    /// a steered experiment.
    pub const COMMAND_X: usize = 9;
    pub const COMMAND_Z: usize = 10;
}

/// Reusable scratch space for evaluation, so the hot loop never allocates.
#[derive(Clone, Debug)]
pub struct BrainScratch {
    pub inputs: Vec<Real>,
    pub hidden: Vec<Real>,
    pub outputs: Vec<Real>,
}

impl BrainScratch {
    pub fn new(layout: &BrainLayout) -> BrainScratch {
        BrainScratch {
            inputs: vec![0.0; layout.inputs()],
            hidden: vec![0.0; layout.hidden],
            outputs: vec![0.0; layout.outputs()],
        }
    }

    /// Zero the input vector. Slots belonging to parts the organism does not
    /// have stay zeroed.
    pub fn clear_inputs(&mut self) {
        for v in self.inputs.iter_mut() {
            *v = 0.0;
        }
    }
}

/// Evaluate the network described by `layout` with parameters `weights`.
///
/// Reads `scratch.inputs`, writes `scratch.outputs`. Outputs are in `[-1, 1]`.
pub fn evaluate(layout: &BrainLayout, weights: &[Real], scratch: &mut BrainScratch) {
    debug_assert_eq!(weights.len(), layout.weight_count());
    let n_in = layout.inputs();
    let n_hid = layout.hidden;
    let n_out = layout.outputs();

    let (w1, rest) = weights.split_at(n_hid * n_in);
    let (b1, rest) = rest.split_at(n_hid);
    let (w2, b2) = rest.split_at(n_out * n_hid);

    // Destructured so the three buffers are borrowed independently, and iterated
    // with `chunks_exact`/`zip` so the bounds checks fall out of the inner loops.
    let BrainScratch { inputs, hidden, outputs } = scratch;
    let inputs = &inputs[..n_in];

    for ((h, row), &bias) in hidden.iter_mut().zip(w1.chunks_exact(n_in)).zip(b1) {
        let mut acc = bias;
        for (w, x) in row.iter().zip(inputs) {
            acc += w * x;
        }
        *h = tanh_approx(acc);
    }

    for ((o, row), &bias) in outputs.iter_mut().zip(w2.chunks_exact(n_hid)).zip(b2) {
        let mut acc = bias;
        for (w, x) in row.iter().zip(hidden.iter()) {
            acc += w * x;
        }
        *o = tanh_approx(acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn layout() -> BrainLayout {
        BrainLayout::new(6, 8)
    }

    #[test]
    fn weight_count_matches_manual_sum() {
        let l = layout();
        let n_in = GLOBAL_INPUTS + 3 * 6;
        assert_eq!(l.inputs(), n_in);
        assert_eq!(l.outputs(), 6);
        assert_eq!(l.weight_count(), 8 * n_in + 8 + 6 * 8 + 6);
    }

    #[test]
    fn evaluation_is_deterministic_and_bounded() {
        let l = layout();
        let mut rng = Rng::new(5);
        let weights: Vec<Real> = (0..l.weight_count()).map(|_| rng.signed() * 3.0).collect();
        let mut a = BrainScratch::new(&l);
        let mut b = BrainScratch::new(&l);
        for i in 0..l.inputs() {
            let v = rng.signed();
            a.inputs[i] = v;
            b.inputs[i] = v;
        }
        evaluate(&l, &weights, &mut a);
        evaluate(&l, &weights, &mut b);
        assert_eq!(a.outputs, b.outputs);
        for o in &a.outputs {
            assert!((-1.0..=1.0).contains(o));
        }
    }

    #[test]
    fn zero_weights_give_zero_outputs() {
        let l = layout();
        let weights = vec![0.0; l.weight_count()];
        let mut s = BrainScratch::new(&l);
        s.inputs[input::BIAS] = 1.0;
        evaluate(&l, &weights, &mut s);
        assert!(s.outputs.iter().all(|&o| o == 0.0));
    }

    #[test]
    fn inputs_actually_influence_outputs() {
        let l = layout();
        let mut rng = Rng::new(17);
        let weights: Vec<Real> = (0..l.weight_count()).map(|_| rng.signed()).collect();
        let mut s = BrainScratch::new(&l);
        s.inputs[input::BIAS] = 1.0;
        evaluate(&l, &weights, &mut s);
        let base = s.outputs.clone();
        s.inputs[l.slot_input_base(2)] = 1.0;
        evaluate(&l, &weights, &mut s);
        assert!(base != s.outputs);
    }
}
