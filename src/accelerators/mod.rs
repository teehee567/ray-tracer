use aabb::AABB;
use glam::Vec3;

pub mod aabb;
pub mod bvh;

pub trait Primitive: Send + Sync {
    fn centroid(&self) -> Vec3;

    fn aabb(&self) -> AABB;
}
