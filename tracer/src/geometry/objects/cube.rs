use std::sync::Arc;

use image::{RgbImage, RgbaImage};
use nalgebra::{Point3, Vector3};

use crate::accelerators::aabb::AABB;
use crate::core::camera::Camera;
use crate::core::hittable::{self, HitRecord, Hittable};
use crate::core::hittable_list::HittableList;
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::geometry::objects::quad::Quad;
use crate::geometry::wireframe::WireFrame;
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
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        self.sides.hit(ray, ray_t)
    }

    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
}

impl WireFrame for Cube {
    fn draw_wireframe(&self, img: &mut RgbaImage, colour: Colour, camera: &Camera) {
        // Define the 8 vertices of the cube
        let vertices = [
            self.box_min,                                       // min (x, y, z)
            Point3::new(self.box_min.x, self.box_min.y, self.box_max.z), // min (x, y, max z)
            Point3::new(self.box_min.x, self.box_max.y, self.box_min.z), // min (x, max y, z)
            Point3::new(self.box_min.x, self.box_max.y, self.box_max.z), // min (x, max y, max z)
            Point3::new(self.box_max.x, self.box_min.y, self.box_min.z), // max (x, y, z)
            Point3::new(self.box_max.x, self.box_min.y, self.box_max.z), // max (x, y, max z)
            Point3::new(self.box_max.x, self.box_max.y, self.box_min.z), // max (x, max y, z)
            self.box_max,                                       // max (x, y, z)
        ];

        // Define the 12 edges of the cube (pairs of vertex indices)
        let edges = [
            (0, 1), (1, 3), (3, 2), (2, 0), // Bottom face
            (4, 5), (5, 7), (7, 6), (6, 4), // Top face
            (0, 4), (1, 5), (2, 6), (3, 7), // Vertical edges connecting bottom and top
        ];

        // Draw each edge
        for &(start, end) in &edges {
            if let (Some((x0, y0)), Some((x1, y1))) = (
                camera.world_to_screen(vertices[start]),
                camera.world_to_screen(vertices[end]),
            ) {
                println!("Drawing edge: {}, {} -> {}, {}", x0, y0, x1, y1);
                Camera::draw_line(img, x0, y0, x1, y1, colour);
            }
        }
    }
}
