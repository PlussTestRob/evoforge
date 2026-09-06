//! Deterministic pseudo-random number generation.
//!
//! Implemented in-repo rather than pulling in `rand`, for two reasons:
//!
//! 1. `rand` explicitly does not guarantee that a given seed produces the same
//!    stream across releases. An experiment recorded today must replay
//!    identically years from now.
//! 2. The algorithms are twenty lines each.
//!
//! [`Rng`] is xoshiro256++ seeded through SplitMix64, which is fast, has a long
//! period, and passes the usual statistical batteries.

use serde::{Deserialize, Serialize};

use crate::math::{dcos, dln, Real, TAU};

/// SplitMix64 — used to expand a single seed into well-distributed state, and as
/// the mixing function for [`derive_seed`].
#[inline]
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive a stream seed from a list of identifying values.
///
/// This is how an organism's evaluation seed is produced from
/// `(experiment_seed, generation, organism_id)`. Because it is a pure function of
/// identity, evaluation order and thread scheduling cannot affect results, and a
/// single organism can be re-simulated in isolation years later.
pub fn derive_seed(parts: &[u64]) -> u64 {
    let mut acc = 0x243F_6A88_85A3_08D3;
    for &p in parts {
        acc ^= p;
        let mut s = acc;
        acc = splitmix64(&mut s);
    }
    acc
}

/// xoshiro256++ generator.
///
/// The state is serialisable so that a checkpoint can resume an experiment with
/// the exact stream it would have had.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut sm = seed;
        Rng {
            s: [
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
            ],
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Uniform in `[0, 1)`, using the top 24 bits so every result is exactly
    /// representable.
    #[inline]
    pub fn unit(&mut self) -> Real {
        ((self.next_u64() >> 40) as Real) * (1.0 / 16_777_216.0)
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn range(&mut self, lo: Real, hi: Real) -> Real {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform in `[-1, 1)`.
    #[inline]
    pub fn signed(&mut self) -> Real {
        self.unit() * 2.0 - 1.0
    }

    /// Uniform integer in `[0, n)`. Debiased by rejection so that the
    /// distribution is exact and independent of `n`.
    pub fn below(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        if n.is_power_of_two() {
            return self.next_u32() & (n - 1);
        }
        let zone = u32::MAX - (u32::MAX % n) - 1;
        loop {
            let v = self.next_u32();
            if v <= zone {
                return v % n;
            }
        }
    }

    /// `true` with probability `p`.
    #[inline]
    pub fn chance(&mut self, p: Real) -> bool {
        self.unit() < p
    }

    /// Standard normal deviate via Box-Muller.
    ///
    /// Uses the deterministic `ln`/`cos` from [`crate::math`], so the stream is
    /// bitwise identical across platforms. The paired second deviate is
    /// discarded rather than cached: caching would make the output depend on the
    /// history of calls, which makes reproducibility harder to reason about for
    /// a cost we do not care about here.
    pub fn normal(&mut self) -> Real {
        // Guard against log(0).
        let u1 = self.unit().max(1.0 / 16_777_216.0);
        let u2 = self.unit();
        (-2.0 * dln(u1)).sqrt() * dcos(TAU * u2)
    }

    /// Normal deviate with the given standard deviation.
    #[inline]
    pub fn normal_scaled(&mut self, sigma: Real) -> Real {
        self.normal() * sigma
    }

    /// Pick an element index from a slice.
    #[inline]
    pub fn pick(&mut self, len: usize) -> usize {
        self.below(len as u32) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values, captured once. See `stream_is_stable`.
    const STABLE_STREAM: [u64; 4] = [
        18_245_054_023_354_416_621,
        6_498_265_956_716_577_385,
        8_819_707_495_113_362_485,
        16_465_575_537_134_391_893,
    ];

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seed_different_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(43);
        let differs = (0..64).any(|_| a.next_u64() != b.next_u64());
        assert!(differs);
    }

    /// The stream is part of the reproducibility contract: if this changes,
    /// previously recorded experiments no longer replay. Changing it should be a
    /// deliberate, versioned decision.
    #[test]
    fn stream_is_stable() {
        let mut r = Rng::new(0xE00F_0000_0000_0001u64);
        let got: Vec<u64> = (0..4).map(|_| r.next_u64()).collect();
        assert_eq!(got, STABLE_STREAM);
    }

    #[test]
    fn unit_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..100_000 {
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn below_is_in_range_and_covers() {
        let mut r = Rng::new(11);
        let mut seen = [0u32; 7];
        for _ in 0..70_000 {
            let v = r.below(7);
            assert!(v < 7);
            seen[v as usize] += 1;
        }
        // Every bucket should be hit roughly 10k times; a loose bound is enough
        // to catch a broken implementation.
        for count in seen {
            assert!(count > 8_000 && count < 12_000, "skewed: {seen:?}");
        }
    }

    #[test]
    fn normal_has_expected_moments() {
        let mut r = Rng::new(99);
        let n = 200_000;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for _ in 0..n {
            let v = r.normal() as f64;
            assert!(v.is_finite());
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }

    #[test]
    fn derive_seed_is_order_sensitive_and_stable() {
        assert_eq!(derive_seed(&[1, 2, 3]), derive_seed(&[1, 2, 3]));
        assert_ne!(derive_seed(&[1, 2, 3]), derive_seed(&[3, 2, 1]));
        assert_ne!(derive_seed(&[1, 2, 3]), derive_seed(&[1, 2, 4]));
    }

    #[test]
    fn state_roundtrips_through_serde() {
        let mut r = Rng::new(1234);
        for _ in 0..50 {
            r.next_u64();
        }
        let json = serde_json::to_string(&r).unwrap();
        let mut restored: Rng = serde_json::from_str(&json).unwrap();
        for _ in 0..50 {
            assert_eq!(r.next_u64(), restored.next_u64());
        }
    }
}
