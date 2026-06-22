
use vulkanalia::{prelude::v1_0::*, vk::ImageView};

use anyhow::Result;
use crate::{scene::Scene, vulkan::core::{context::VulkanContext, sampler::{SamplerDesc, create_sampler}}};

pub struct Compositer {
    sampler: vk::Sampler,
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


        Ok(Self {
            sampler,

        })
    }

}
