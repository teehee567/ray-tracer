
use std::sync::Arc;

use crate::{aabb::AABB, hittable::{HitRecord, Hittable}, interval::Interval, ray::Ray};

pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable>>,
    bbox: AABB 
}

impl HittableList {
    pub fn new(object: impl Hittable + 'static) -> Self {
        Self {
            objects: vec![Box::new(object)],
            bbox: AABB::none(),
        }
    }

    pub fn none() -> Self {
        Self {
            objects: vec![],
            bbox: AABB::none(),
        }
    }
    
    pub fn clear(&mut self) {
        self.objects.clear();
    }

    pub fn add(&mut self, object: impl Hittable + 'static) {
        self.bbox = AABB::combine(&self.bbox, object.bounding_box());
        self.objects.push(Box::new(object));
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let mut hit_anything = false;
        let mut closest_so_far = ray_t.max;

        for object in &self.objects {
            let mut temp_rec = HitRecord::new();  // Local temporary record
            let mut current_interval = Interval::new(ray_t.min, closest_so_far);

            if object.hit(ray, current_interval, &mut temp_rec) {
                hit_anything = true;
                closest_so_far = temp_rec.t;

                // Update only the fields in rec instead of copying the entire struct
                rec.t = temp_rec.t;
                rec.p = temp_rec.p;
                rec.normal = temp_rec.normal;
                rec.front_face = temp_rec.front_face;
                rec.mat = Arc::clone(&temp_rec.mat);
                // FIX: FUCK YOU DOESNT AUTO SET STRUCT INSIDES
                rec.u = temp_rec.u;
                rec.v = temp_rec.v;
            }
        }

        hit_anything
    }

    fn bounding_box(&self) -> &AABB {
        return &self.bbox
    }
}
