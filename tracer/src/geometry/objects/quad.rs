use std::sync::Arc;

use image::RgbImage;
use nalgebra::{Point3, Vector3};

use crate::accelerators::aabb::AABB;
use crate::core::camera::Camera;
use crate::core::hittable::{HitRecord, Hittable, ToScreen};
use crate::core::hittable_list::HittableList;
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::materials::material::Material;
use crate::utils::colour::Colour;

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
    pub fn new(
        q: Point3<f32>,
        u: Vector3<f32>,
        v: Vector3<f32>,
        material: Arc<dyn Material>,
    ) -> Self {
        // Compute the four vertices of the quad
        let p0 = q;
        let p1 = q + u;
        let p2 = q + v;
        let p3 = q + u + v;

        // Find the minimum and maximum coordinates among the vertices
        let min_x = p0.x.min(p1.x).min(p2.x).min(p3.x);
        let min_y = p0.y.min(p1.y).min(p2.y).min(p3.y);
        let min_z = p0.z.min(p1.z).min(p2.z).min(p3.z);

        let max_x = p0.x.max(p1.x).max(p2.x).max(p3.x);
        let max_y = p0.y.max(p1.y).max(p2.y).max(p3.y);
        let max_z = p0.z.max(p1.z).max(p2.z).max(p3.z);

        // Slightly expand the bounding box to prevent zero-size boxes
        let padding = 1e-4;
        let min_point = Point3::new(min_x - padding, min_y - padding, min_z - padding);
        let max_point = Point3::new(max_x + padding, max_y + padding, max_z + padding);

        let bbox = AABB::new(min_point, max_point);

        // Rest of your existing code
        let n = u.cross(&v);
        let normal = n.normalize();
        let d = normal.dot(&q.coords);
        let w = n / n.dot(&n);

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

    // FIX: create proper generic wireframe
    pub fn draw_wireframe(&self, img: &mut RgbImage, colour: Colour, camera: &Camera) {
        let vertices = [
            self.q,
            self.q + self.u,
            self.q + self.u + self.v,
            self.q + self.v,
        ];

        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0), // Edges of the quad
        ];

        for &(start, end) in &edges {
            if let (Some((x0, y0)), Some((x1, y1))) = (
                camera.world_to_screen(vertices[start]),
                camera.world_to_screen(vertices[end]),
            ) {
                println!("here: {}, {} -> {}, {}", x0, y0, x1, y1);
                Camera::draw_line(img, x0, y0, x1, y1, colour);
            }
        }
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
