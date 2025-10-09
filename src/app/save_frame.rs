use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use anyhow::{Result, anyhow};
use glam::UVec2;
use image::ImageBuffer;
use log::info;
use vulkanalia::Version;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{
    DeviceV1_0, ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension,
};
use vulkanalia::window as vk_window;
use winit::window::Window;

use super::AppData;
use crate::scene::Scene;
use crate::types::{AUVec2, Au32, CameraBufferObject};
use crate::ui;
use crate::vulkan::accumulate_image::{create_image, transition_image_layout};
use crate::vulkan::buffers::{create_shader_buffers, create_uniform_buffer};
use crate::vulkan::command_buffers::{
    create_command_buffer, record_compute_commands, record_present_commands,
};
use crate::vulkan::command_pool::create_command_pool;
use crate::vulkan::descriptors::{
    create_compute_descriptor_set_layout, create_descriptor_pool, create_descriptor_sets,
};
use crate::vulkan::fps_counter::FPSCounter;
use crate::vulkan::framebuffer::{create_framebuffer_images, transition_framebuffer_images};
use crate::vulkan::instance::create_instance;
use crate::vulkan::logical_device::create_logical_device;
use crate::vulkan::physical_device::{SuitabilityError, pick_physical_device};
use crate::vulkan::pipeline::{create_compute_pipeline, create_render_pass};
use crate::vulkan::sampler::create_sampler;
use crate::vulkan::swapchain::{create_swapchain, create_swapchain_image_views};
use crate::vulkan::sync_objects::create_sync_objects;
use crate::vulkan::texture::{
    Texture, create_cubemap_sampler, create_cubemap_texture, create_texture_image,
    create_texture_sampler,
};
use crate::vulkan::utils::get_memory_type_index;

pub unsafe fn save_frame(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    frame: u32,
) -> Result<()> {
    let size = (data.swapchain_extent.width * data.swapchain_extent.height * 4) as u64;

    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = device.create_buffer(&buffer_info, None)?;

    let mem_requirements = device.get_buffer_memory_requirements(staging_buffer);
    let memory_type = get_memory_type_index(
        instance,
        data,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        mem_requirements,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let staging_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(staging_buffer, staging_memory, 0)?;

    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &begin_info)?;

    let current_layout = data
        .swapchain_image_layouts
        .get(0)
        .copied()
        .unwrap_or(vk::ImageLayout::UNDEFINED);

    let (src_stage, src_access) = match current_layout {
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        ),
        vk::ImageLayout::PRESENT_SRC_KHR => (
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::MEMORY_READ,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::empty(),
        ),
    };

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(current_layout)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(data.swapchain_images[0])
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    let copy = vk::BufferImageCopy::builder()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D {
            width: data.swapchain_extent.width,
            height: data.swapchain_extent.height,
            depth: 1,
        });

    device.cmd_copy_image_to_buffer(
        command_buffer,
        data.swapchain_images[0],
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        staging_buffer,
        &[copy],
    );

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(data.swapchain_images[0])
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    if let Some(layout) = data.swapchain_image_layouts.get_mut(0) {
        *layout = vk::ImageLayout::PRESENT_SRC_KHR;
    }

    device.end_command_buffer(command_buffer)?;

    let submit_info =
        vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));

    device.queue_submit(data.compute_queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(data.compute_queue)?;

    let data_ptr =
        device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())? as *const u8;

    let buffer = std::slice::from_raw_parts(data_ptr, size as usize);

    let width = data.swapchain_extent.width as u32;
    let height = data.swapchain_extent.height as u32;
    let mut img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            let pixel = image::Rgba([buffer[i + 2], buffer[i + 1], buffer[i], buffer[i + 3]]);
            img.put_pixel(x, y, pixel);
        }
    }

    let mut input_img = vec![0.0f32; (3 * width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            for c in 0..3 {
                input_img[3 * ((y * width + x) as usize) + c] = pixel[c] as f32 / 255.0;
            }
        }
    }

    device.unmap_memory(staging_memory);

    device.free_command_buffers(data.command_pool, &[command_buffer]);
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);

    println!("Saved Buffer");

    img.save("images/materials/raw/spec_trans/spec_trans_100.png")?;
    panic!();
    Ok(())
}
