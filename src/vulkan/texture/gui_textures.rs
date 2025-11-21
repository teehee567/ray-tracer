
use vulkanalia::prelude::v1_0::*;
use anyhow::Result;

use crate::{AppData, vulkan::utils::get_memory_type_index};

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
    data: &AppData,
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

    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width: size[0].max(1),
            height: size[1].max(1),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = device.create_image(&image_info, None)?;
    let requirements = device.get_image_memory_requirements(image);
    let memory_type = get_memory_type_index(
        instance,
        data,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        requirements,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(image, memory, 0)?;

    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        );
    let view = device.create_image_view(&view_info, None)?;

    let sampler_info = vk::DescriptorImageInfo::builder()
        .sampler(sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .build();
    let image_info = vk::DescriptorImageInfo::builder()
        .sampler(vk::Sampler::null())
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .build();

    let sampler_write = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::SAMPLER)
        .image_info(std::slice::from_ref(&sampler_info))
        .build();
    let image_write = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
        .image_info(std::slice::from_ref(&image_info))
        .build();

    let writes = [sampler_write, image_write];
    device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);

    Ok(GuiTexture {
        image,
        view,
        memory,
        descriptor_set,
        size,
    })
}

