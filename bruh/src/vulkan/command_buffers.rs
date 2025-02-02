use vulkanalia::prelude::v1_0::*;

use crate::{AppData, TILE_SIZE};
use anyhow::Result;

pub unsafe fn create_command_buffer(device: &Device, data: &mut AppData) -> Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    data.compute_command_buffer = device.allocate_command_buffers(&allocate_info)?[0];

    Ok(())
}

pub unsafe fn run_command_buffer(device: &Device, data: &mut AppData, image_index: usize) -> Result<()> {
    let command_buffer = data.compute_command_buffer;
    let info = vk::CommandBufferBeginInfo::builder();

    device.begin_command_buffer(data.compute_command_buffer, &info)?;

    device.cmd_bind_pipeline(data.compute_command_buffer, vk::PipelineBindPoint::COMPUTE, data.compute_pipeline);

    let compute_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .image(data.swapchain_images[image_index])
        .subresource_range(vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE);

    device.cmd_pipeline_barrier(data.compute_command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[] as &[vk::MemoryBarrier], &[] as &[vk::BufferMemoryBarrier], &[compute_barrier]);
    device.cmd_bind_descriptor_sets(data.compute_command_buffer, vk::PipelineBindPoint::COMPUTE, data.compute_pipeline_layout, 0, &[data.compute_descriptor_sets[image_index]], &[]);

    device.cmd_dispatch(data.compute_command_buffer, data.swapchain_extent.width / TILE_SIZE + 1, data.swapchain_extent.height / TILE_SIZE + 1, 1);

    let present_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(data.swapchain_images[image_index])
        .subresource_range(vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::empty());

    device.cmd_pipeline_barrier(data.compute_command_buffer, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::BOTTOM_OF_PIPE, vk::DependencyFlags::empty(), &[] as &[vk::MemoryBarrier], &[] as &[vk::BufferMemoryBarrier], &[present_barrier]);

    device.end_command_buffer(data.compute_command_buffer)?;
    
    Ok(())
}
