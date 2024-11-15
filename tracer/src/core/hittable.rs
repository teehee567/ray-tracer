use std::sync::Arc;

use image::{RgbImage, RgbaImage};
use nalgebra::{Point3, Vector3};

use crate::accelerators::aabb::AABB;
use crate::core::camera::Camera;
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::geometry::objects::cube::Cube;
use crate::geometry::objects::quad::Quad;
use crate::geometry::wireframe::WireFrame;
use crate::materials::material::{Lambertian, Material};
use crate::utils::colour::Colour;

#[derive(Clone)]
pub struct HitRecord {
    pub p: Point3<f32>,
    pub normal: Vector3<f32>,
    pub mat: Arc<dyn Material>,
    pub t: f32,
    pub u: f32,
    pub v: f32,
    pub front_face: bool,
}

impl HitRecord {
    pub fn new() -> Self {
        Self {
            p: Point3::default(),
            normal: Vector3::default(),
            mat: Arc::new(Lambertian::new(&Colour::new(0.8, 0.8, 0.8, 1.))),
            t: 0.,
            u: 123.,
            v: 153.,
            front_face: false,
        }
    }

    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vector3<f32>) {
        self.front_face = Vector3::dot(ray.direction(), &outward_normal) < 0.;
        if (self.front_face) {
            self.normal = *outward_normal;
        } else {
            self.normal = -*outward_normal;
        }
    }
}

pub trait Hittable: Send + Sync {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool;
    fn bounding_box(&self) -> &AABB;
}

pub trait ToScreen {
    fn wire_frame<'a>(&'a self) -> Option<&'a [(u32, u32, u32, u32)]>;
}

// FIX: create proper generic wireframe
pub unsafe fn call_wireframe_for_quad(
    hittable: &Box<dyn Hittable>,
    img: &mut RgbaImage,
    colour: Colour,
    camera: &Camera,
) {
    let cube_ptr = &**hittable as *const dyn Hittable as *const Quad;
    let cube_ref = &*cube_ptr;

    cube_ref.draw_wireframe(img, colour, camera)
}

pub unsafe fn call_wireframe_for_cube(
    hittable: &Box<dyn Hittable>,
    img: &mut RgbaImage,
    colour: Colour,
    camera: &Camera,
) {
    let cube_ptr = &**hittable as *const dyn Hittable as *const Cube;
    let cube_ref = &*cube_ptr;

    cube_ref.draw_wireframe(img, colour, camera)
}
