use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::image::{create_image_2d, create_image_view_2d};

#[derive(Clone, Debug)]
pub struct GuiTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub descriptor_set: vk::DescriptorSet,
    pub size: [u32; 2],
}

impl Default for GuiTexture {
    fn default() -> Self {
        Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            memory: vk::DeviceMemory::null(),
            descriptor_set: vk::DescriptorSet::null(),
            size: [0, 0],
        }
    }
}

pub unsafe fn create_texture_resource(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    size: [u32; 2],
) -> Result<GuiTexture> {
    let layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let descriptor_set = device.allocate_descriptor_sets(&alloc_info)?[0];

    let (image, memory) = create_image_2d(
        instance,
        device,
        physical_device,
        size[0],
        size[1],
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        1,
        vk::ImageCreateFlags::empty(),
    )?;

    let view = create_image_view_2d(
        device,
        image,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageViewType::_2D,
        vk::ImageAspectFlags::COLOR,
        1,
    )?;

    let image_info = vk::DescriptorImageInfo::builder()
        .sampler(sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .build();

    let write = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(std::slice::from_ref(&image_info))
        .build();

    device.update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);

    Ok(GuiTexture {
        image,
        view,
        memory,
        descriptor_set,
        size,
    })
}
