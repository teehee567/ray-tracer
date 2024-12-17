use nalgebra::Vector3;

use crate::accelerators::aabb::AABB;
use crate::core::hittable::{HitRecord, Hittable};
use crate::core::interval::Interval;
use crate::core::ray::Ray;

pub struct Translate<H: Hittable> {
    hittable: H,
    offset: Vector3<f32>,
    aabb: AABB,
}

impl<H: Hittable> Translate<H> {
    pub fn new(hittable: H, offset: Vector3<f32>) -> Self {
        let mut aabb = *hittable.bounding_box();
        aabb.min += offset;
        aabb.max += offset;

        Self {
            hittable,
            offset,
            aabb,
        }
    }
}

impl<H: Hittable> Hittable for Translate<H> {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let moved_ray = Ray::new_tm(ray.origin() - self.offset, *ray.direction(), ray.time());
        if let Some(mut rec) = self.hittable.hit(&moved_ray, ray_t) {
            rec.p += self.offset;
            return Some(rec)
        }
        None
    }

    fn bounding_box(&self) -> &AABB {
        &self.aabb
    }
}
