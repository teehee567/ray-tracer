use vulkanalia::{Device, Instance};
use vulkanalia::prelude::v1_0::*;

use anyhow::{Result, anyhow};

use crate::AppData;

use super::{begin_single_time_commands, end_single_time_commands, transition_texture_layout};
use super::Texture;
use crate::vulkan::utils::{create_buffer, get_memory_type_index};


pub unsafe fn create_cubemap_texture(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> Result<Texture> {
    let size = (width * height * 4 * 6) as u64; // 6 faces

    // Create staging buffer
    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // Copy pixels to staging buffer
    let memory = device.map_memory(
        staging_buffer_memory,
        0,
        size,
        vk::MemoryMapFlags::empty(),
    )?;
    std::ptr::copy_nonoverlapping(pixels.as_ptr(), memory.cast(), pixels.len());
    device.unmap_memory(staging_buffer_memory);

    // Create cubemap image
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(6)
        .format(vk::Format::R8G8B8A8_SRGB)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::_1)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE); // Important for cubemap

    let image = device.create_image(&image_info, None)?;

    // Allocate memory for image
    let mem_requirements = device.get_image_memory_requirements(image);
    let memory_type = get_memory_type_index(
        instance,
        data,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        mem_requirements,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let image_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(image, image_memory, 0)?;

    // Transition image layout and copy data
    transition_skybox_layout(
        device,
        data,
        image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        true
    )?;

    copy_buffer_to_cubemap(
        device,
        data,
        staging_buffer,
        image,
        width,
        height,
    )?;

    transition_skybox_layout(
        device,
        data,
        image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        true
    )?;

    // Create cubemap image view
    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::CUBE)
        .format(vk::Format::R8G8B8A8_SRGB)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 6,
        });

    let view = device.create_image_view(&view_info, None)?;

    // Cleanup staging buffer
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(Texture {
        width,
        height,
        image,
        view,
        memory: image_memory,
    })
}

unsafe fn copy_buffer_to_cubemap(
    device: &Device,
    data: &AppData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let mut regions = Vec::with_capacity(6);
    let face_size = (width * height * 4) as u64;

    // Create copy regions for all 6 faces
    for face in 0..6 {
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(face_size * face as u64)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: face,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .build();
        regions.push(region);
    }

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &regions,
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

unsafe fn transition_skybox_layout(
    device: &Device,
    data: &AppData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    is_cubemap: bool, // Add this parameter
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, data)?;

    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        _ => return Err(anyhow!("Unsupported image layout transition!")),
    };

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: if is_cubemap { 6 } else { 1 }, // Handle all 6 faces for cubemap
        })
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .build();

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask,
        dst_stage_mask,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, data, command_buffer)?;

    Ok(())
}

pub unsafe fn create_cubemap_sampler(device: &Device) -> Result<vk::Sampler> {
    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
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

