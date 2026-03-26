use crate::scene::Scene;
use crate::vulkan::texture::Texture;
use vulkanalia::prelude::v1_0::*;

#[derive(Clone, Debug)]
pub struct SceneResources {
    pub scene: Scene,
    pub textures: Vec<Texture>,
    pub texture_sampler: vk::Sampler,
    pub skybox_texture: Texture,
    pub skybox_sampler: vk::Sampler,
}
