use aabb::AABB;
use bvh::BvhNode;
use glam::Vec3;

use crate::{Material, Triangle};

pub mod aabb;
pub mod bin_sah;
pub mod bvh;

pub trait Primitive: Send + Sync {
    fn centroid(&self) -> Vec3;

    fn aabb(&self) -> AABB;
}

/// reorders traingels in place and flat vec.
pub trait Accelerator {
    fn build(&self, triangles: &mut Vec<Triangle>, materials: &mut Vec<Material>) -> Vec<BvhNode>;
}
