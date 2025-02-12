pub mod texture;
use texture::Texture;

use crate::{colour::Colour};

pub struct Material<'a> {
    pub base_color: Colour,
    pub metallic_roughness_texture: &'a Texture,
    pub base_color_texture: &'a Texture,
    pub metalness: f32,
    pub roughness: f32,
}

#[derive(Default)]
pub struct GpuMaterial {
    pub base_color: Colour,
    pub albedo_texture_id: u32,
    pub albedo_texture_sampler_id: u32,
    pub metallic_roughness_texture_id: u32,
    pub metallic_roughness_texture_sampler_id: u32,
    pub metalness: f32,
    pub roughness: f32,
}
