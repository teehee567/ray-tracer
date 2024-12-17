use std::sync::Arc;

use crate::accelerators::aabb::AABB;
use crate::core::hittable::{HitRecord, Hittable};
use crate::core::interval::Interval;
use crate::core::ray::Ray;

pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable>>,
    bbox: AABB,
}

impl HittableList {
    pub fn new(object: impl Hittable + 'static) -> Self {
        Self {
            objects: vec![Box::new(object)],
            bbox: AABB::default(),
        }
    }

    pub fn none() -> Self {
        Self {
            objects: vec![],
            bbox: AABB::default(),
        }
    }

    pub fn clear(&mut self) {
        self.objects.clear();
    }

    pub fn add(&mut self, object: impl Hittable + 'static) {
        self.bbox = AABB::combine(&self.bbox, object.bounding_box());
        self.objects.push(Box::new(object));
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn to_raw(self) -> Vec<Box<dyn Hittable>> {
        return self.objects;
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        self.objects
            .iter()
            .filter_map(|object| {
                let interval = Interval::new(ray_t.min, ray_t.max);
                object.hit(ray, interval)
            })
            .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal))
}

    fn bounding_box(&self) -> &AABB {
        return &self.bbox;
    }
}
