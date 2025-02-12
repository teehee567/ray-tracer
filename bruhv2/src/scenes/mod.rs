use glam::{Mat4, Vec3, Vec4};
use std::sync::Arc;
mod gltf_import;

use crate::vulkan::api::Buffer;

pub struct Camera {
    pub origin: Vec3,
    pub lower_left_corner: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub lens_radius: f32,
}

pub struct Metadata {
    pub cam: Camera,
    pub max_bounce: u32,
    pub min_bounce: u32,
    pub width: u32,
    pub height: u32,
    pub sample_index: u32,
    pub enable_dof: u32,
    pub debug_bvh: u32,
    pub downscale_factor: i32,
}

impl Metadata {
    pub fn new(cam: Camera, width: u32, height: u32) -> Self {
        Self {
            cam,
            max_bounce: 3,
            min_bounce: 1,
            width,
            height,
            sample_index: 0,
            enable_dof: false as u32,
            debug_bvh: false as u32,
            downscale_factor: 1,
        }
    }
}

pub struct Scene {
    pub meta: Metadata,
    pub scene_buffer: Option<Arc<Buffer>>,
    pub indices_buffer: Option<Arc<Buffer>>,
    pub positions_buffer: Option<Arc<Buffer>>,
    pub normals_buffer: Option<Arc<Buffer>>,
    pub uvs_buffer: Option<Arc<Buffer>>,
    pub bvh_buffer: Option<Arc<Buffer>>,
    pub materials_buffer: Option<Arc<Buffer>>,
}

impl Scene {
    pub fn new(cam: Camera, width: u32, height: u32) -> Self {
        Self {
            meta: Metadata::new(cam, width, height),
            scene_buffer: None,
            indices_buffer: None,
            positions_buffer: None,
            normals_buffer: None,
            uvs_buffer: None,
            bvh_buffer: None,
            materials_buffer: None,
        }
    }
}

