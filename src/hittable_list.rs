
use std::sync::Arc;

use crate::{hittable::{HitRecord, Hittable}, interval::Interval, ray::Ray};

pub struct HittableList {
    objects: Vec<Box<dyn Hittable>>
}

impl HittableList {
    pub fn new(object: impl Hittable + 'static) -> Self {
        Self {
            objects: vec![Box::new(object)],
        }
    }

    pub fn none() -> Self {
        Self {
            objects: vec![]
        }
    }
    
    pub fn clear(&mut self) {
        self.objects.clear();
    }

    pub fn add(&mut self, object: impl Hittable + 'static) {
        self.objects.push(Box::new(object));
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let mut hit_anything = false;
        let mut closest_so_far = ray_t.max;

        for object in &self.objects {
            let mut temp_rec = HitRecord::new();  // Local temporary record
            let current_interval = Interval::new(ray_t.min, closest_so_far);

            if object.hit(ray, current_interval, &mut temp_rec) {
                hit_anything = true;
                closest_so_far = temp_rec.t;

                // Update only the fields in rec instead of copying the entire struct
                rec.t = temp_rec.t;
                rec.p = temp_rec.p;
                rec.normal = temp_rec.normal;
                rec.front_face = temp_rec.front_face;
                rec.mat = Arc::clone(&temp_rec.mat);
            }
        }

        hit_anything
    }
}
