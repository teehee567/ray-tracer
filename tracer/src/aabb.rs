use nalgebra::{Point3, Vector3};

use crate::{interval::Interval, ray::Ray};

#[derive(Clone, Copy, Default)]
pub struct AABB {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl AABB {
    #[inline]
    pub fn new(p0: Point3<f32>, p1: Point3<f32>) -> Self {
        let min = Point3::new(
            p0.x.min(p1.x),
            p0.y.min(p1.y),
            p0.z.min(p1.z),
        );
        let max = Point3::new(
            p0.x.max(p1.x),
            p0.y.max(p1.y),
            p0.z.max(p1.z),
        );
        AABB { min, max }
    }

    pub fn combine(box0: &AABB, box1: &AABB) -> AABB {
        let min = Point3::new(
            box0.min.x.min(box1.min.x),
            box0.min.y.min(box1.min.y),
            box0.min.z.min(box1.min.z),
        );
        let max = Point3::new(
            box0.max.x.max(box1.max.x),
            box0.max.y.max(box1.max.y),
            box0.max.z.max(box1.max.z),
        );
        AABB { min, max }
    }

    pub fn hit(&self, ray: &Ray, ray_t: Interval) -> bool {
        let mut t_min = ray_t.min;
        let mut t_max = ray_t.max;
        for a in 0..3 {
            let inv_d = 1.0 / ray.direction()[a];
            let t0 = (self.min[a] - ray.origin()[a]) * inv_d;
            let t1 = (self.max[a] - ray.origin()[a]) * inv_d;
            let (t0, t1) = if inv_d < 0.0 { (t1, t0) } else { (t0, t1) };
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
            if t_max <= t_min {
                return false;
            }
        }
        true
    }

    #[inline]
    pub fn union(&mut self, point: Point3<f32>) -> AABB {
        let min = Point3::new(
            self.min.x.min(point.x),
            self.min.y.min(point.y),
            self.min.z.min(point.z),
        );
        let max = Point3::new(
            self.max.x.max(point.x),
            self.max.y.max(point.y),
            self.max.z.max(point.z),
        );
        AABB { min, max }
    }
}
