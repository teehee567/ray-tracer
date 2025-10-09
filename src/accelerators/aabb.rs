use std::ops::Index;

use glam::Vec3;

#[derive(Clone, Copy, Default, Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    #[inline]
    pub fn new(p0: Vec3, p1: Vec3) -> Self {
        let min = Vec3::new(p0.x.min(p1.x), p0.y.min(p1.y), p0.z.min(p1.z));
        let max = Vec3::new(p0.x.max(p1.x), p0.y.max(p1.y), p0.z.max(p1.z));
        AABB { min, max }
    }

    pub fn grow_bb_mut(&mut self, aabb: &Self) {
        *self = Self::combine(self, aabb)
    }

    pub fn offset_by(&mut self, delta: f32) {
        let delta = Vec3::splat(delta);
        let min = self.min - delta;
        let max = self.max + delta;

        self.min = min;
        self.max = max;
    }

    #[inline(always)]
    pub fn grow(&mut self, vertex: Vec3) -> AABB {
        AABB::combine(
            self,
            &AABB {
                min: vertex,
                max: vertex,
            },
        )
    }

    #[inline(always)]
    pub fn grow_bb(&self, aabb: &Self) -> AABB {
        Self::combine(self, aabb)
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.diagonal();
        let x = d.x;
        let y = d.y;
        let z = d.z;
        2.0f32 * (x * y + y * z + z * x)
    }

    pub fn half_area(&self) -> f32 {
        let d = self.diagonal();
        d.x * d.y + d.y * d.z + d.z * d.x
    }

    pub fn diagonal(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn largest_axis(&self) -> usize {
        iamax(self.diagonal())
    }
}

fn iamax(vec: Vec3) -> usize {
    let abs_values = vec.abs(); // Get absolute values of components
    if abs_values.x >= abs_values.y && abs_values.x >= abs_values.z {
        0
    } else if abs_values.y >= abs_values.z {
        1
    } else {
        2
    }
}

impl AABB {
    #[inline(always)]
    pub fn combine_scalar(box0: &AABB, box1: &AABB) -> AABB {
        let min = Vec3::from([
            box0.min.x.min(box1.min.x),
            box0.min.y.min(box1.min.y),
            box0.min.z.min(box1.min.z),
        ]);
        let max = Vec3::from([
            box0.max.x.max(box1.max.x),
            box0.max.y.max(box1.max.y),
            box0.max.z.max(box1.max.z),
        ]);
        AABB { min, max }
    }

    #[inline(always)]
    pub fn combine(box0: &AABB, box1: &AABB) -> AABB {
        Self::combine_scalar(box0, box1)
    }
}

impl Index<usize> for AABB {
    type Output = Vec3;

    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.min,
            1 => &self.max,
            _ => panic!("Index out of range! AABB only has two elements: min and max"),
        }
    }
}
