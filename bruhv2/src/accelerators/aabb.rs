use glam::Vec3;

use crate::primitives::triangle::Triangle;

#[derive(Clone, Copy, Debug, Default)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_triangle(tri: &Triangle) -> Self {
        let min = tri.p1.min(tri.p2.min(tri.p3));
        let max = tri.p1.max(tri.p2.max(tri.p3));
        Self { min, max }
    }

    pub fn union_with(&mut self, other: &AABB) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    pub fn union_with_point(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }

    pub fn diagonal(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn surface_area(&self) -> f32 {
        let diag = self.diagonal();
        2.0 * (diag.x * diag.y + diag.x * diag.z + diag.y * diag.z)
    }

    pub fn maximum_axis(&self) -> usize {
        let diag = self.diagonal();
        if diag.x > diag.y && diag.x > diag.z {
            0
        } else if diag.y > diag.z {
            1
        } else {
            2
        }
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
}
