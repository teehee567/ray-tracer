use std::sync::Arc;

use nalgebra::{Point3, Vector3};

use crate::{
    aabb::AABB,
    hittable::{HitRecord, Hittable},
    hittable_list::HittableList,
    interval::Interval,
    material::Material,
    ray::Ray,
};

pub struct Quad {
    q: Point3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    material: Arc<dyn Material>,
    bbox: AABB,
    normal: Vector3<f32>,
    d: f32,
    w: Vector3<f32>,
}

impl Quad {
    pub fn new(q: Point3<f32>, u: Vector3<f32>, v: Vector3<f32>, material: Arc<dyn Material>) -> Self {
        let q = q;
        let bbox_diagonal1 = AABB::new_points(&q, &(q + u + v));
        let bbox_diagonal2 = AABB::new_points(&(q + u), &(q + v));
        let bbox = AABB::combine(&bbox_diagonal1, &bbox_diagonal2);

        let n = Vector3::cross(&u, &v);
        let normal = n.normalize();
        let d = Vector3::dot(&normal, &q.coords);
        let w = n / Vector3::dot(&n, &n);

        Self {
            q,
            u,
            v,
            material,
            bbox,
            normal,
            d,
            w,
        }
    }

    fn is_interior(a: f32, b: f32, rec: &mut HitRecord) -> bool {
        let unit_interval = Interval::new(0., 1.);

        if !unit_interval.contains(a) || !unit_interval.contains(b) {
            return false;
        }

        rec.u = a;
        rec.v = b;

        return true;
    }
}

impl Hittable for Quad {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let denom = Vector3::dot(&self.normal, ray.direction());

        if denom.abs() < 1e-8 {
            return false;
        }

        let t = (self.d - Vector3::dot(&self.normal, &ray.origin().coords)) / denom;
        if !ray_t.contains(t) {
            return false;
        }

        let intersection = ray.at(t);
        let planar_hitpt_vector = intersection - self.q;
        let alpha = Vector3::dot(&self.w, &Vector3::cross(&planar_hitpt_vector, &self.v));
        let beta = Vector3::dot(&self.w, &Vector3::cross(&self.u, &planar_hitpt_vector));

        if !Self::is_interior(alpha, beta, rec) {
            return false;
        }

        rec.t = t;
        rec.p = intersection;
        rec.mat = self.material.clone();
        rec.set_face_normal(ray, &self.normal);

        return true;
    }

    fn bounding_box(&self) -> &AABB {
        return &self.bbox;
    }
}
