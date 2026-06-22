
use vulkanalia::{prelude::v1_0::*, vk::ImageView};

use anyhow::Result;
use crate::{scene::Scene, vulkan::core::{context::VulkanContext, descriptors::{allocate_descriptor_sets, binding, create_descriptor_pool, create_descriptor_set_layout, image_info, image_write, pool_size}, sampler::{SamplerDesc, create_sampler}}};

pub struct Compositer {
    sampler: vk::Sampler,
    composite_set_layout: vk::DescriptorSetLayout,
    composite_pool: vk::DescriptorPool,
    composite_set: vk::DescriptorSet,
}

impl Compositer {
    pub(crate) unsafe fn new(
        ctx: &VulkanContext,
        swapchain_pass: vk::RenderPass,
        extent: vk::Extent2D,
        scene: &Scene,
    ) -> Result<Self> {
        let device = &ctx.device;

        let sampler = create_sampler(device, &SamplerDesc {
            filter: vk::Filter::NEAREST,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            max_anisotropy: None,
        })?;

        let bindings = [binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, vk::ShaderStageFlags::FRAGMENT)];

        let set_layout = create_descriptor_set_layout(device, &bindings)?;

        let pool_sizes = [pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1)];
        let pool = create_descriptor_pool(device, &pool_sizes, 1, vk::DescriptorPoolCreateFlags::empty())?;

        let set = allocate_descriptor_sets(device, pool, set_layout, 1)?[0];


        Ok(Self {
            sampler,
            composite_set_layout: set_layout,
            composite_pool: pool,
            composite_set: set,

        })
    }

    unsafe fn update_composite_set(&self, device: &Device, image_view: vk::ImageView) {
        let infos = [image_info(self.sampler, image_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = image_write(self.composite_set, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, &infos);

        device.update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);
    }

}
