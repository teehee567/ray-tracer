use std::f32::consts::{FRAC_PI_2, PI};
use std::f32::{self};
use std::sync::Arc;

use nalgebra::{center, Normed, Point3, Vector3};

use crate::accelerators::aabb::{self, AABB};
use crate::core::hittable::{HitRecord, Hittable};
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::materials::material::Material;

pub struct Sphere {
    center1: Point3<f32>,
    radius: f32,
    mat: Arc<dyn Material>,
    is_moving: bool,
    center_vec: Vector3<f32>,
    bbox: AABB,
}

impl Sphere {
    pub fn new(center: Point3<f32>, radius: f32, mat: Arc<dyn Material>) -> Self {
        let rvec = Vector3::new(radius, radius, radius);
        let bbox = AABB::new((center - rvec), (center + rvec));
        Self {
            center1: Point3::new(center.x, center.y, center.z),
            radius,
            mat,
            is_moving: false,
            center_vec: Vector3::default(),
            bbox,
        }
    }

    pub fn new_mov(
        center1: Point3<f32>,
        center2: Point3<f32>,
        radius: f32,
        mat: Arc<dyn Material>,
    ) -> Self {
        let rvec = Vector3::new(radius, radius, radius);
        let box1 = AABB::new((center1 - rvec), (center1 + rvec));
        let box2 = AABB::new((center2 - rvec), (center2 + rvec));
        let bbox = AABB::combine(&box1, &box2);

        let center2 = Point3::new(center2.x, center2.y, center2.z);
        let center1 = Point3::new(center1.x, center1.y, center1.z);
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
    fn get_sphere_uv(p: Point3<f32>) -> (f32, f32) {
        let theta = f32::acos(-p.y);
        let phi = f32::atan2(-p.z, p.x) + std::f32::consts::PI;
        let u = phi / (2.0 * std::f32::consts::PI);
        let v = theta / std::f32::consts::PI;

        (u, v)
    }

    fn sphere_center(&self, time: f32) -> Point3<f32> {
        let v = self.center1 + time * self.center_vec;
        Point3::new(v.x, v.y, v.z)
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let center = if (self.is_moving) {
            let v = self.sphere_center(ray.time());
            Point3::new(v.x, v.y, v.z)
        } else {
            self.center1
        };

        let oc = center - ray.origin();
        let a = ray.direction().norm_squared();
        let h = Vector3::dot(ray.direction(), &oc);
        let c = oc.norm_squared() - self.radius * self.radius;

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
        let (u, v) = Self::get_sphere_uv(outward_normal.into());
        rec.set_face_normal(ray, &outward_normal);
        rec.mat = self.mat.clone();
        rec.u = u;
        rec.v = v;

        true
    }

    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
}
