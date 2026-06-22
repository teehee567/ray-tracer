use aabb::AABB;
use glam::Vec3;

use crate::{Material, Triangle};

pub mod aabb;
pub mod bvh_based;
pub mod visualiser;
use bvh_based::bvh::BvhNode;

pub use bvh_based::*;

pub trait Primitive: Send + Sync {
    fn centroid(&self) -> Vec3;

    fn aabb(&self) -> AABB;
}

/// reorders traingels in place and flat vec.
pub trait Accelerator {
    fn build(&self, triangles: &mut Vec<Triangle>, materials: &mut Vec<Material>) -> Vec<BvhNode>;
}
