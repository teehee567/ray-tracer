use std::sync::Arc;

use crate::{hittable::{HitRecord, Hittable}, interval::Interval, material::Material, ray::Ray, vec3::{Point3, Vec3}};



pub struct Sphere {
    center: Point3,
    radius: f64,
    mat: Arc<dyn Material>
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, mat: Arc<dyn Material>) -> Self {
        Self {
            center,
            radius,
            mat
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let oc = self.center - *ray.origin();
        let a = ray.direction().length_squared();
        let h = Vec3::dot(*ray.direction(), oc);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = h * h - a * c;
        if (discriminant < 0.) {
            return false;
        }

        let sqrtd = discriminant.sqrt();

        let mut root = (h - sqrtd) / a;

        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) {
                return false;
            }
        }

        rec.t = root;
        rec.p = ray.at(rec.t);
        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(ray, &outward_normal);
        rec.mat = self.mat.clone();

        return true;
    }
}
