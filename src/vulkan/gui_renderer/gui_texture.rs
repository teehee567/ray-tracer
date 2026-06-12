use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::core::context::VulkanContext;
use crate::vulkan::core::descriptors::{image_info, image_write};
use crate::vulkan::core::image::Image;

#[derive(Clone, Debug, Default)]
pub struct GuiTexture {
    pub image: Image,
    pub descriptor_set: vk::DescriptorSet,
}

impl GuiTexture {
    pub fn size(&self) -> [u32; 2] {
        [self.image.width, self.image.height]
    }
}

pub unsafe fn create_gui_texture(
    ctx: &VulkanContext,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    size: [u32; 2],
) -> Result<GuiTexture> {
    let layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let descriptor_set = ctx.device.allocate_descriptor_sets(&alloc_info)?[0];

    let image = Image::new_2d(
        ctx,
        size[0],
        size[1],
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        1,
        vk::ImageCreateFlags::empty(),
        vk::ImageViewType::_2D,
    )?;

    let infos = [image_info(
        sampler,
        image.view,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )];
    let write = image_write(
        descriptor_set,
        0,
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        &infos,
    );
    ctx.device
        .update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);

    Ok(GuiTexture {
        image,
        descriptor_set,
    })
}

pub unsafe fn destroy_gui_texture(
    device: &Device,
    pool: vk::DescriptorPool,
    texture: &mut GuiTexture,
) -> Result<()> {
    texture.image.destroy(device);
    if texture.descriptor_set != vk::DescriptorSet::null() {
        device.free_descriptor_sets(pool, &[texture.descriptor_set])?;
        texture.descriptor_set = vk::DescriptorSet::null();
    }
    Ok(())
}
