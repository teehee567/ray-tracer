use aabb::AABB;
use glam::Vec3;


pub mod bvh;
pub mod aabb;


pub trait Primitive: Send + Sync {
    fn centroid(&self) -> Vec3;

    fn aabb(&self) -> AABB;
}
