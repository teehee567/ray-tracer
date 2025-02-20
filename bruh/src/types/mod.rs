use glam::Vec3;

use crate::{accelerators::{self, bvh::BvhNode, Primitive}, scene::TextureData};

mod camera_buffer_obj;
pub use camera_buffer_obj::*;

mod aligned;
pub use aligned::*;


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub base_colour: AVec3,
    pub emission: AVec3,
    pub metallic: Af32,
    pub roughness: Af32,
    pub subsurface: Af32,
    pub anisotropic: Af32,
    pub specular_tint: Af32,
    pub sheen: Af32,
    pub sheen_tint: Af32,
    pub clearcoat: Af32,
    pub clearcoat_roughness: Af32,
    pub spec_trans: Af32,
    pub ior: Af32,

    pub shade_smooth: Au32,
    // textures
    pub base_color_tex: Au32,
    pub metallic_roughness_tex: Au32,
    pub normal_tex: Au32,
    pub emission_tex: Au32,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_colour: AVec3(Vec3::new(0.8, 0.8, 0.8)),  // Light gray - typical diffuse surface
            emission: AVec3(Vec3::new(0.0, 0.0, 0.0)),     // Most materials don't emit light
            metallic: Af32(0.0),                     // Non-metallic by default (dielectric)
            roughness: Af32(0.5),                    // Medium roughness
            subsurface: Af32(0.0),                   // No subsurface scattering
            anisotropic: Af32(0.0),                  // Isotropic by default
            specular_tint: Af32(0.0),                // No tint to specular
            sheen: Af32(0.0),                        // No sheen
            sheen_tint: Af32(0.5),                   // Neutral sheen tint
            clearcoat: Af32(0.0),                    // No clearcoat
            clearcoat_roughness: Af32(0.03),         // Smooth clearcoat when enabled
            spec_trans: Af32(0.0),                   // Opaque
            ior: Af32(1.45),                         // Common IOR for plastics

            shade_smooth: Au32(1),                   // Smooth shading by default
            
            base_color_tex: Au32(u32::MAX),
            metallic_roughness_tex: Au32(u32::MAX),
            normal_tex: Au32(u32::MAX),
            emission_tex: Au32(u32::MAX),
        }
    }
}


#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default)]
pub struct Triangle {
    pub material_index: Au32,
    pub is_sphere: Au32,
    pub vertices: [AVec3; 3],
    pub normals: [AVec3; 3],
    pub uvs: [AVec2; 3],
}

#[repr(C)]
#[derive(Clone, Default)]
pub struct SceneComponents {
    pub camera: CameraBufferObject,
    pub bvh: Vec<BvhNode>,
    pub materials: Vec<Material>,
    pub triangles: Vec<Triangle>,
    pub textures: Vec<TextureData>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Sphere {
    center: AVec3,
    radius: Af32,
}


impl Primitive for Triangle {
    fn centroid(&self) -> Vec3 {
        (self.vertices[0].0 + self.vertices[1].0 + self.vertices[2].0) * 0.3333333f32
    }

    fn aabb(&self) -> accelerators::aabb::AABB {
        accelerators::aabb::AABB::new(self.vertices[0].0, self.vertices[1].0).grow(self.vertices[2].0)
    }
}

impl Triangle {
    pub fn min_bound(&self) -> Vec3 {
        if self.is_sphere.0 == 1 {
            let radius = self.vertices[1].0.x;
            self.vertices[0].0 - Vec3::splat(radius)
        } else {
            Vec3 {
                x: self.vertices.iter().map(|v| v.0.x).fold(f32::INFINITY, f32::min),
                y: self.vertices.iter().map(|v| v.0.y).fold(f32::INFINITY, f32::min),
                z: self.vertices.iter().map(|v| v.0.z).fold(f32::INFINITY, f32::min),
            }
        }
    }

    pub fn max_bound(&self) -> Vec3 {
        if self.is_sphere.0 == 1 {
            let radius = self.vertices[1].0.x;
            self.vertices[0].0 + Vec3::splat(radius)
        } else {
            Vec3 {
                x: self.vertices.iter().map(|v| v.0.x).fold(f32::NEG_INFINITY, f32::max),
                y: self.vertices.iter().map(|v| v.0.y).fold(f32::NEG_INFINITY, f32::max),
                z: self.vertices.iter().map(|v| v.0.z).fold(f32::NEG_INFINITY, f32::max),
            }
        }
    }
}

