use std::f32::{INFINITY, NEG_INFINITY};

#[derive(Debug, Clone, Copy)]
pub struct Interval {
    pub min: f32,
    pub max: f32,
}

impl Interval {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    pub fn none() -> Self {
        Self {
            min: INFINITY,
            max: NEG_INFINITY,
        }
    }

    pub fn size(&self) -> f32 {
        self.max - self.min
    }

    pub fn contains(&self, x: f32) -> bool {
        self.min <= x && x <= self.max
    }

    pub fn surrounds(&self, x: f32) -> bool {
        self.min < x && x < self.max
    }

    pub fn clamp(&self, x: f32) -> f32 {
        if (x < self.min) {
            return self.min;
        }
        if (x > self.max) {
            return self.max;
        }

        return x;
    }

    // pads an interval so no NaN
    pub fn expand(&self, delta: f32) -> Interval {
        let padding = delta / 2.;
        Interval::new(self.min - padding, self.max + padding)
    }

    pub fn combine(a: &Interval, b: &Interval) -> Self {
        let min = if (a.min <= b.min) { a.min } else { b.min };

        let max = if (a.max >= b.max) { a.max } else { b.max };

        Self { min, max }
    }
}
