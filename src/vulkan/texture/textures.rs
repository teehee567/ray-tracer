use std::ptr;
use vulkanalia::prelude::v1_0::*;

use anyhow::Result;

use crate::vulkan::context::VulkanContext;
use crate::vulkan::image::{cmd_transition_image_layout, create_image_2d, create_image_view_2d};
use crate::vulkan::single_time::with_single_time;
use crate::vulkan::utils::create_buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub memory: vk::DeviceMemory,
}

pub unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    ctx: &VulkanContext,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<Texture> {
    let size = (width * height * 4) as u64;

    let (staging_buffer, staging_memory) = create_buffer(
        instance,
        device,
        ctx.physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    let mapped = device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?;
    ptr::copy_nonoverlapping(pixels.as_ptr(), mapped.cast(), pixels.len());
    device.unmap_memory(staging_memory);

    let (image, image_memory) = create_image_2d(
        instance,
        device,
        ctx.physical_device,
        width,
        height,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        1,
        vk::ImageCreateFlags::empty(),
    )?;

    with_single_time(device, ctx.command_pool, ctx.compute_queue, |cb| {
        cmd_transition_image_layout(
            device,
            cb,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        let region = vk::BufferImageCopy::builder()
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        device.cmd_copy_buffer_to_image(
            cb,
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        cmd_transition_image_layout(
            device,
            cb,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;
        Ok(())
    })?;

    let view = create_image_view_2d(
        device,
        image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageViewType::_2D,
        vk::ImageAspectFlags::COLOR,
        1,
    )?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);

    Ok(Texture {
        width,
        height,
        image,
        view,
        memory: image_memory,
    })
}

pub unsafe fn create_texture_sampler(device: &Device) -> Result<vk::Sampler> {
    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);

    Ok(device.create_sampler(&sampler_info, None)?)
}
