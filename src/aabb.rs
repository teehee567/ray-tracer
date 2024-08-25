use crate::{interval::Interval, ray::Ray, vec3::Point3};

#[derive(Clone)]
pub struct AABB {
    pub x: Interval,
    pub y: Interval,
    pub z: Interval,
}

impl AABB {
    pub fn new(x: &Interval, y: &Interval, z: &Interval) -> Self {
        Self {
            x: *x,
            y: *y,
            z: *z,
        }
    }

    pub fn none() -> Self {
        Self {
            x: Interval::none(),
            y: Interval::none(),
            z: Interval::none(),
        }
    }

    pub fn combine(box0: &AABB, box1: &AABB) -> Self {
        Self {
            x: Interval::combine(&box0.x, &box1.x),
            y: Interval::combine(&box0.y, &box1.y),
            z: Interval::combine(&box0.z, &box1.z),
        }
    }

    pub fn new_points(a: &Point3, b: &Point3) -> Self {
        let x = if (a[0] <= b[0]) {
            Interval::new(a[0], b[0])
        } else {
            Interval::new(b[0], a[0])
        };

        let y = if (a[1] <= b[1]) {
            Interval::new(a[1], b[1])
        } else {
            Interval::new(b[1], a[1])
        };

        let z = if (a[2] <= b[2]) {
            Interval::new(a[2], b[2])
        } else {
            Interval::new(b[2], a[2])
        };

        Self { x, y, z }
    }

    pub fn axis_interval(&self, n: i32) -> &Interval {
        if (n == 1) {
            return &self.y;
        }
        if (n == 2) {
            return &self.z;
        }
        return &self.x;
    }

    pub fn hit(&self, ray: &Ray, mut ray_t: Interval) -> bool {
        let ray_orig = ray.origin();
        let ray_dir = ray.direction();

        for axis in 0..3 {
            let ax = self.axis_interval(axis);
            let adinv = 1. / ray_dir[axis as usize];

            let t0 = (ax.min - ray_orig[axis as usize]) * adinv;
            let t1 = (ax.max - ray_orig[axis as usize]) * adinv;

            if (t0 < t1) {
                if (t0 > ray_t.min) {
                    ray_t.min = t0;
                }
                if (t1 < ray_t.max) {
                    ray_t.max = t1;
                }
            } else {
                if (t1 > ray_t.min) {
                    ray_t.min = t1;
                }
                if (t0 < ray_t.max) {
                    ray_t.max = t0;
                }
            }

            if (ray_t.max <= ray_t.min) {
                return false;
            }
        }

        return true;
    }
}
