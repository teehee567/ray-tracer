use glam::Vec3;

use super::{AVec3, Au32, Af32};

pub const POINT_LIGHT: u32 = 0;
pub const MESH_LIGHT: u32 = 1;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EmissiveTri {
    pub tri_index: Au32,
    pub mesh_index: Au32,
    pub area: Af32,
    pub power: Af32,
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct MeshLightSampler {
    pub tris: Vec<EmissiveTri>,
    pub cdf: Vec<f32>,
    pub total_power: f32,
}

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Light {
    pub position: AVec3,
    pub emission: AVec3,
    pub light_type: Au32,
}

impl Light {
    pub fn point(position: Vec3, emission: Vec3) -> Self {
        Self {
            position: AVec3(position),
            emission: AVec3(emission),
            light_type: Au32(POINT_LIGHT),
        }
    }
}
