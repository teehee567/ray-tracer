use std::sync::Arc;

use image::RgbImage;
use nalgebra::{Point3, Vector3};

use crate::accelerators::aabb::AABB;
use crate::core::camera::Camera;
use crate::core::hittable::{self, HitRecord, Hittable};
use crate::core::hittable_list::HittableList;
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::geometry::objects::quad::Quad;
use crate::materials::material::Material;
use crate::utils::colour::Colour;

pub struct Cube {
    box_min: Point3<f32>,
    box_max: Point3<f32>,
    sides: HittableList,
    bbox: AABB,
}

impl Cube {
    pub fn new(
        a: nalgebra::Point3<f32>,
        b: nalgebra::Point3<f32>,
        material: Arc<dyn Material>,
    ) -> Self {
        let min = Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
        let max = Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));

        let dx = Vector3::new(max.x - min.x, 0., 0.);
        let dy = Vector3::new(0., max.y - min.y, 0.);
        let dz = Vector3::new(0., 0., max.z - min.z);

        let mut sides = HittableList::none();
        sides.add(Quad::new(
            Point3::new(min.x, min.y, max.z),
            dx,
            dy,
            material.clone(),
        )); // front
        sides.add(Quad::new(
            Point3::new(max.x, min.y, max.z),
            -dz,
            dy,
            material.clone(),
        )); // right
        sides.add(Quad::new(
            Point3::new(max.x, min.y, min.z),
            -dx,
            dy,
            material.clone(),
        )); // back
        sides.add(Quad::new(
            Point3::new(min.x, min.y, min.z),
            dz,
            dy,
            material.clone(),
        )); // left
        sides.add(Quad::new(
            Point3::new(min.x, max.y, max.z),
            dx,
            -dz,
            material.clone(),
        )); // top
        sides.add(Quad::new(
            Point3::new(min.x, min.y, min.z),
            dx,
            dz,
            material.clone(),
        )); // bottom

        Self {
            box_min: min,
            box_max: max,
            sides,
            bbox: AABB::new(min, max),
        }
    }
}

impl Hittable for Cube {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        self.sides.hit(ray, ray_t, rec)
    }

    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
}
