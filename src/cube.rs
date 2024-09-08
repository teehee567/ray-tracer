use std::sync::Arc;

use crate::{
    aabb::AABB,
    hittable::{self, HitRecord, Hittable},
    hittable_list::HittableList,
    interval::Interval,
    material::Material,
    quad::Quad,
    ray::Ray,
    vec3::{Point3, Vec3},
};

pub struct Cube {
    box_min: Point3,
    box_max: Point3,
    sides: HittableList,
    bbox: AABB,
}

impl Cube {
    pub fn new(a: Point3, b: Point3, material: Arc<dyn Material>) -> Self {
        let min = Point3::new(a.x().min(b.x()), a.y().min(b.y()), a.z().min(b.z()));
        let max = Point3::new(a.x().max(b.x()), a.y().max(b.y()), a.z().max(b.z()));

        let dx = Vec3::new(max.x() - min.x(), 0., 0.);
        let dy = Vec3::new(0., max.y() - min.y(), 0.);
        let dz = Vec3::new(0., 0., max.z() - min.z());

        let mut sides = HittableList::none();
        sides.add(Quad::new(
            Point3::new(min.x(), min.y(), max.z()),
            dx,
            dy,
            material.clone(),
        )); // front
        sides.add(Quad::new(
            Point3::new(max.x(), min.y(), max.z()),
            -dz,
            dy,
            material.clone(),
        )); // right
        sides.add(Quad::new(
            Point3::new(max.x(), min.y(), min.z()),
            -dx,
            dy,
            material.clone(),
        )); // back
        sides.add(Quad::new(
            Point3::new(min.x(), min.y(), min.z()),
            dz,
            dy,
            material.clone(),
        )); // left
        sides.add(Quad::new(
            Point3::new(min.x(), max.y(), max.z()),
            dx,
            -dz,
            material.clone(),
        )); // top
        sides.add(Quad::new(
            Point3::new(min.x(), min.y(), min.z()),
            dx,
            dz,
            material.clone(),
        )); // bottom

        let x = Interval::new(min.x(), max.x());
        let y = Interval::new(min.y(), max.y());
        let z = Interval::new(min.z(), max.z());

        Self {
            box_min: min,
            box_max: max,
            sides,
            bbox: AABB::new(&x, &y, &z),
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
