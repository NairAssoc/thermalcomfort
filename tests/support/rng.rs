//! Deterministic PRNG for the differential sweep.
//!
//! SplitMix64: tiny, well-distributed, and — critically — a *fixed, specified*
//! algorithm. A failing sweep must be reproducible from its seed alone, on any machine
//! and any future toolchain, which rules out `rand`'s unspecified defaults. It also
//! keeps a deliberately dependency-light crate that way.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1), using 53 bits to match f64's mantissa.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Uniform in [lo, hi].
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.unit() * (hi - lo)
    }

    pub fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}
