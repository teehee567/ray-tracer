use std::sync::Arc;

use crate::{
    aabb::AABB,
    colour::Colour,
    interval::Interval,
    material::{Lambertian, Material},
    ray::Ray,
    vec3::{Point3, Vec3},
};

#[derive(Clone)]
pub struct HitRecord {
    pub p: Point3,
    pub normal: Vec3,
    pub mat: Arc<dyn Material>,
    pub t: f64,
    pub u: f64,
    pub v: f64,
    pub front_face: bool,
}

impl HitRecord {
    pub fn new() -> Self {
        Self {
            p: Point3::none(),
            normal: Vec3::none(),
            mat: Arc::new(Lambertian::new(&Colour::new(0.8, 0.8, 0.8))),
            t: 0.,
            u: 123.,
            v: 153.,
            front_face: false,
        }
    }

    pub fn set_face_normal(&mut self, ray: &Ray, outward_normal: &Vec3) {
        self.front_face = Vec3::dot(*ray.direction(), *outward_normal) < 0.;
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
