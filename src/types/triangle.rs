use glam::Vec3;

use crate::accelerators::{self, Primitive};

use super::{AVec2, AVec3, Au32};

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Triangle {
    pub material_index: Au32,
    pub is_sphere: Au32,
    pub vertices: [AVec3; 3],
    pub normals: [AVec3; 3],
    pub uvs: [AVec2; 3],
}

impl Primitive for Triangle {
    fn centroid(&self) -> Vec3 {
        (self.vertices[0].0 + self.vertices[1].0 + self.vertices[2].0) * 0.3333333f32
    }

    fn aabb(&self) -> accelerators::aabb::AABB {
        accelerators::aabb::AABB::new(self.vertices[0].0, self.vertices[1].0)
            .grow(self.vertices[2].0)
    }
}

impl Triangle {
    pub fn min_bound(&self) -> Vec3 {
        if self.is_sphere.0 == 1 {
            let radius = self.vertices[1].0.x;
            self.vertices[0].0 - Vec3::splat(radius)
        } else {
            Vec3 {
                x: self
                    .vertices
                    .iter()
                    .map(|v| v.0.x)
                    .fold(f32::INFINITY, f32::min),
                y: self
                    .vertices
                    .iter()
                    .map(|v| v.0.y)
                    .fold(f32::INFINITY, f32::min),
                z: self
                    .vertices
                    .iter()
                    .map(|v| v.0.z)
                    .fold(f32::INFINITY, f32::min),
            }
        }
    }

    pub fn max_bound(&self) -> Vec3 {
        if self.is_sphere.0 == 1 {
            let radius = self.vertices[1].0.x;
            self.vertices[0].0 + Vec3::splat(radius)
        } else {
            Vec3 {
                x: self
                    .vertices
                    .iter()
                    .map(|v| v.0.x)
                    .fold(f32::NEG_INFINITY, f32::max),
                y: self
                    .vertices
                    .iter()
                    .map(|v| v.0.y)
                    .fold(f32::NEG_INFINITY, f32::max),
                z: self
                    .vertices
                    .iter()
                    .map(|v| v.0.z)
                    .fold(f32::NEG_INFINITY, f32::max),
            }
        }
    }
}
