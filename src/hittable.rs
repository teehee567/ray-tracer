use std::sync::Arc;

use nalgebra::{Point3, Vector3};

use crate::{
    aabb::AABB,
    colour::Colour,
    interval::Interval,
    material::{Lambertian, Material},
    ray::Ray,
};

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
            mat: Arc::new(Lambertian::new(&Colour::new(0.8, 0.8, 0.8))),
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
