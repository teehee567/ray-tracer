use glam::Vec3;
use serde::{Deserialize, Serialize};

use super::{AVec3, Af32, Au32};


#[repr(C)]
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
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

