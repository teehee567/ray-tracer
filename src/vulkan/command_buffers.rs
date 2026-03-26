use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::compute::ComputeResources;
use crate::vulkan::gui_renderer::GuiRenderer;
use crate::vulkan::swapchain_data::SwapchainData;
use crate::OFFSCREEN_FRAME_COUNT;
use crate::TILE_SIZE;
use anyhow::{Result, anyhow};
use glam::UVec2;

pub unsafe fn create_command_buffer(
    device: &Device,
    command_pool: vk::CommandPool,
) -> Result<(Vec<vk::CommandBuffer>, vk::CommandBuffer)> {
    let command_buffer_count = (OFFSCREEN_FRAME_COUNT + 1) as u32;
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(command_buffer_count);

    let buffers = device.allocate_command_buffers(&allocate_info)?;
    if buffers.len() < OFFSCREEN_FRAME_COUNT + 1 {
        return Err(anyhow!(
            "failed to allocate compute/present command buffers"
        ));
    }

    let compute_command_buffers = buffers[..OFFSCREEN_FRAME_COUNT].to_vec();
    let present_command_buffer = buffers[OFFSCREEN_FRAME_COUNT];
    info!("Created command buffers for: {:?}", command_pool);

    Ok((compute_command_buffers, present_command_buffer))
}

pub unsafe fn record_compute_commands(
    device: &Device,
    compute: &ComputeResources,
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
            compute.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            compute.pipeline_layout,
            0,
            &[compute.descriptor_sets[frame_index]],
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
    swapchain: &mut SwapchainData,
    compute: &ComputeResources,
    present_command_buffer: vk::CommandBuffer,
    swapchain_index: usize,
    frame_index: usize,
    panel_width: u32,
    render_extent: UVec2,
    gui: &GuiRenderer,
) -> Result<()> {
    let command_buffer = present_command_buffer;
    let begin_info = vk::CommandBufferBeginInfo::builder();
    device.begin_command_buffer(command_buffer, &begin_info)?;

    let framebuffer_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(compute.framebuffer_images[frame_index])
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

    let current_layout = swapchain
        .image_layouts
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
        .image(swapchain.images[swapchain_index])
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
        swapchain.images[swapchain_index],
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
            compute.framebuffer_images[frame_index],
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            swapchain.images[swapchain_index],
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );
    }

    let mut swap_layout_after = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

    if gui.has_draws() {
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(swapchain.images[swapchain_index])
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
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .build();

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        let framebuffer = swapchain.framebuffers[swapchain_index];
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        };
        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(swapchain.render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area);

        device.cmd_begin_render_pass(command_buffer, &begin_info, vk::SubpassContents::INLINE);

        gui.record_draws(device, command_buffer, frame_index, swapchain.extent)?;

        device.cmd_end_render_pass(command_buffer);
        swap_layout_after = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }

    let framebuffer_to_general = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .image(compute.framebuffer_images[frame_index])
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

    let (present_src_stage, present_src_access) =
        if swap_layout_after == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL {
            (
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            )
        } else {
            (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            )
        };

    let present_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(swap_layout_after)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(swapchain.images[swapchain_index])
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
        .src_access_mask(present_src_access)
        .dst_access_mask(vk::AccessFlags::empty())
        .build();

    device.cmd_pipeline_barrier(
        command_buffer,
        present_src_stage,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[present_barrier],
    );

    if let Some(layout) = swapchain.image_layouts.get_mut(swapchain_index) {
        *layout = vk::ImageLayout::PRESENT_SRC_KHR;
    }

    device.end_command_buffer(command_buffer)?;

    Ok(())
}
