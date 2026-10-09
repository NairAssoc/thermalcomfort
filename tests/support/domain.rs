//! Description of one model's samplable input space.
//!
//! A `Domain` is deliberately dumb — it knows names and ranges, not models. Mapping a
//! drawn `Sample` onto a Rust call and a Python call lives with the model, in
//! `tests/differential_sweep.rs`.

use crate::support::rng::Rng;

#[derive(Debug, Clone)]
pub enum Axis {
    Real {
        name: &'static str,
        lo: f64,
        hi: f64,
    },
    Enum {
        name: &'static str,
        n: usize,
    },
    Flag {
        name: &'static str,
    },
}

#[derive(Debug, Clone, Default)]
pub struct Domain {
    pub axes: Vec<Axis>,
}

impl Domain {
    pub fn new() -> Self {
        Self { axes: Vec::new() }
    }

    pub fn real(mut self, name: &'static str, lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "axis {name}: lo {lo} > hi {hi}");
        self.axes.push(Axis::Real { name, lo, hi });
        self
    }

    pub fn enumerated(mut self, name: &'static str, n: usize) -> Self {
        assert!(n > 0, "axis {name}: needs at least one variant");
        self.axes.push(Axis::Enum { name, n });
        self
    }

    pub fn flag(mut self, name: &'static str) -> Self {
        self.axes.push(Axis::Flag { name });
        self
    }
}

#[derive(Debug, Clone)]
enum Value {
    Real(f64),
    Index(usize),
    Flag(bool),
}

#[derive(Debug, Clone)]
pub struct Sample {
    values: Vec<(&'static str, Value)>,
}

impl Sample {
    fn get(&self, name: &str) -> &Value {
        self.values
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v)
            .unwrap_or_else(|| panic!("unknown axis {name:?}"))
    }

    pub fn real(&self, name: &str) -> f64 {
        match self.get(name) {
            Value::Real(x) => *x,
            other => panic!("axis {name:?} is not real: {other:?}"),
        }
    }

    pub fn index(&self, name: &str) -> usize {
        match self.get(name) {
            Value::Index(i) => *i,
            other => panic!("axis {name:?} is not enumerated: {other:?}"),
        }
    }

    pub fn flag(&self, name: &str) -> bool {
        match self.get(name) {
            Value::Flag(b) => *b,
            other => panic!("axis {name:?} is not a flag: {other:?}"),
        }
    }

    /// One-line rendering used in failure messages, so a failing vector can be pasted
    /// straight into a reproduction.
    pub fn describe(&self) -> String {
        self.values
            .iter()
            .map(|(n, v)| match v {
                Value::Real(x) => format!("{n}={x:.6}"),
                Value::Index(i) => format!("{n}=#{i}"),
                Value::Flag(b) => format!("{n}={b}"),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Replace one real axis; used by the shrink pass.
    pub fn with_real(&self, name: &str, value: f64) -> Sample {
        let mut out = self.clone();
        for (n, v) in out.values.iter_mut() {
            if *n == name {
                *v = Value::Real(value);
            }
        }
        out
    }
}

pub fn draw(domain: &Domain, rng: &mut Rng) -> Sample {
    let values = domain
        .axes
        .iter()
        .map(|axis| match axis {
            Axis::Real { name, lo, hi } => (*name, Value::Real(rng.uniform(*lo, *hi))),
            Axis::Enum { name, n } => (*name, Value::Index((rng.next_u64() % *n as u64) as usize)),
            Axis::Flag { name } => (*name, Value::Flag(rng.bool())),
        })
        .collect();
    Sample { values }
}
