use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::{AppData, GuiCopyInfo, OFFSCREEN_FRAME_COUNT, TILE_SIZE};
use anyhow::{Result, anyhow};
use glam::UVec2;

pub unsafe fn create_command_buffer(device: &Device, data: &mut AppData) -> Result<()> {
    let command_buffer_count = (OFFSCREEN_FRAME_COUNT + 1) as u32;
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(command_buffer_count);

    let buffers = device.allocate_command_buffers(&allocate_info)?;
    if buffers.len() < OFFSCREEN_FRAME_COUNT + 1 {
        return Err(anyhow!(
            "failed to allocate compute/present command buffers"
        ));
    }

    data.compute_command_buffers = buffers[..OFFSCREEN_FRAME_COUNT].to_vec();
    data.present_command_buffer = buffers[OFFSCREEN_FRAME_COUNT];
    info!("Created command buffers for: {:?}", data.command_pool);

    Ok(())
}

pub unsafe fn record_compute_commands(
    device: &Device,
    data: &mut AppData,
    command_buffer: vk::CommandBuffer,
    frame_index: usize,
    render_extent: UVec2,
) -> Result<()> {
    let info = vk::CommandBufferBeginInfo::builder();

    device.begin_command_buffer(command_buffer, &info)?;

    if render_extent.x > 0 && render_extent.y > 0 {
        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            data.compute_pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            data.compute_pipeline_layout,
            0,
            &[data.compute_descriptor_sets[frame_index]],
            &[],
        );

        let groups_x = (render_extent.x + TILE_SIZE - 1) / TILE_SIZE;
        let groups_y = (render_extent.y + TILE_SIZE - 1) / TILE_SIZE;
        device.cmd_dispatch(command_buffer, groups_x.max(1), groups_y.max(1), 1);
    }

    device.end_command_buffer(command_buffer)?;

    Ok(())
}

pub unsafe fn record_present_commands(
    device: &Device,
    data: &mut AppData,
    swapchain_index: usize,
    frame_index: usize,
    panel_width: u32,
    render_extent: UVec2,
    gui_copy: Option<GuiCopyInfo>,
) -> Result<()> {
    let command_buffer = data.present_command_buffer;
    let begin_info = vk::CommandBufferBeginInfo::builder();
    device.begin_command_buffer(command_buffer, &begin_info)?;

    let framebuffer_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(data.framebuffer_images[frame_index])
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .build();

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[framebuffer_barrier],
    );

    let current_layout = data
        .swapchain_image_layouts
        .get(swapchain_index)
        .copied()
        .unwrap_or(vk::ImageLayout::UNDEFINED);

    let (swap_src_stage, swap_src_access) = match current_layout {
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

    let swapchain_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(current_layout)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(data.swapchain_images[swapchain_index])
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(swap_src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .build();

    device.cmd_pipeline_barrier(
        command_buffer,
        swap_src_stage,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[swapchain_barrier],
    );

    let clear_value = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0],
    };
    let clear_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();
    device.cmd_clear_color_image(
        command_buffer,
        data.swapchain_images[swapchain_index],
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &clear_value,
        &[clear_range],
    );

    if render_extent.x > 0 && render_extent.y > 0 {
        let copy_region = vk::ImageCopy::builder()
            .src_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .dst_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .src_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .dst_offset(vk::Offset3D {
                x: panel_width as i32,
                y: 0,
                z: 0,
            })
            .extent(vk::Extent3D {
                width: render_extent.x,
                height: render_extent.y,
                depth: 1,
            })
            .build();

        device.cmd_copy_image(
            command_buffer,
            data.framebuffer_images[frame_index],
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            data.swapchain_images[swapchain_index],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );
    }

    if let Some(copy) = gui_copy {
        if copy.width > 0 && copy.height > 0 {
            let region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
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
                    width: copy.width,
                    height: copy.height,
                    depth: 1,
                })
                .build();

            device.cmd_copy_buffer_to_image(
                command_buffer,
                copy.buffer,
                data.swapchain_images[swapchain_index],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
    }

    let framebuffer_to_general = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .image(data.framebuffer_images[frame_index])
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
        .build();

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[framebuffer_to_general],
    );

    let present_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(data.swapchain_images[swapchain_index])
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::empty())
        .build();

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[present_barrier],
    );

    if let Some(layout) = data.swapchain_image_layouts.get_mut(swapchain_index) {
        *layout = vk::ImageLayout::PRESENT_SRC_KHR;
    }

    device.end_command_buffer(command_buffer)?;

    Ok(())
}
