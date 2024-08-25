use std::{f64::{self, consts::{FRAC_PI_2, PI}}, sync::Arc};

use crate::{
    aabb::{self, AABB}, hittable::{HitRecord, Hittable}, interval::Interval, material::Material, ray::Ray, vec3::{Point3, Vec3}
};

pub struct Sphere {
    center1: Point3,
    radius: f64,
    mat: Arc<dyn Material>,
    is_moving: bool,
    center_vec: Vec3,
    bbox: AABB,
}

impl Sphere {
    pub fn new(center: Point3, radius: f64, mat: Arc<dyn Material>) -> Self {
        let rvec = Vec3::new(radius, radius, radius);
        let bbox = AABB::new_points(&(center - rvec), &(center + rvec));
        Self {
            center1: center,
            radius,
            mat,
            is_moving: false,
            center_vec: Vec3::none(),
            bbox,
        }
    }

    pub fn new_mov(center1: Point3, center2: Point3, radius: f64, mat: Arc<dyn Material>) -> Self {
        let rvec = Vec3::new(radius, radius, radius);
        let box1 = AABB::new_points(&(center1 - rvec), &(center1 + rvec));
        let box2 = AABB::new_points(&(center2 - rvec), &(center2 + rvec));
        let bbox = AABB::combine(&box1, &box2);
        Self {
            center1,
            radius,
            mat,
            is_moving: true,
            center_vec: center2 - center1,
            bbox,
        }
    }

    // Map Point3 to sphere texture
    fn get_sphere_uv(p: Point3) -> (f64, f64) {
        let theta = f64::acos(-p.y());
        let phi = f64::atan2(-p.z(), p.x()) + std::f64::consts::PI;
        let u = phi / (2.0 * std::f64::consts::PI);
        let v = theta / std::f64::consts::PI;
        
        (u, v)
    }

    fn sphere_center(&self, time: f64) -> Point3 {
        self.center1 + time * self.center_vec
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let center = if (self.is_moving) {
            self.sphere_center(ray.time())
        } else {
            self.center1
        };

        let oc = center - *ray.origin();
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
        let outward_normal = (rec.p - self.center1) / self.radius;
        let (u, v) = Self::get_sphere_uv(outward_normal);
        rec.set_face_normal(ray, &outward_normal);
        rec.mat = self.mat.clone();
        rec.u = u;
        rec.v = v;

        return true;
    }

    fn bounding_box(&self) -> &AABB {
        return &self.bbox;
    }
}
