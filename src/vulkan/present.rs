use anyhow::Result;
use glam::UVec2;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::utils::save_frame::SaveImage;

use super::core::image::{cmd_image_barrier, image_barrier, subresource_range};
use super::core::swapchain::Swapchain;
use super::gui_renderer::GuiRenderer;

/// Record the present pass: blit the path traced frame into the swapchain
/// image (offset past the GUI panel) and draw the GUI on top.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn record_present_commands(
    device: &Device,
    swapchain: &mut Swapchain,
    framebuffer_image: vk::Image,
    gui: &GuiRenderer,
    command_buffer: vk::CommandBuffer,
    swapchain_index: usize,
    frame_index: usize,
    panel_width: u32,
    render_extent: UVec2,
    save_image: Option<&SaveImage>,
    // for timing present timer
    query_pool: vk::QueryPool,
) -> Result<()> {
    let swapchain_image = swapchain.images[swapchain_index];
    let begin_info = vk::CommandBufferBeginInfo::builder();
    device.begin_command_buffer(command_buffer, &begin_info)?;

    let first_query = frame_index as u32 * 2;
    device.cmd_reset_query_pool(command_buffer, query_pool, first_query, 2);
    device.cmd_write_timestamp(
        command_buffer,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        query_pool,
        first_query,
    );

    // path traced frame: compute output -> transfer source
    cmd_image_barrier(
        device,
        command_buffer,
        image_barrier(
            framebuffer_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            1,
        ),
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
    );

    // swapchain image -> transfer destination
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

    cmd_image_barrier(
        device,
        command_buffer,
        image_barrier(
            swapchain_image,
            current_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            swap_src_access,
            vk::AccessFlags::TRANSFER_WRITE,
            1,
        ),
        swap_src_stage,
        vk::PipelineStageFlags::TRANSFER,
    );

    let clear_value = vk::ClearColorValue {
        float32: [0.0, 0.0, 0.0, 1.0],
    };
    device.cmd_clear_color_image(
        command_buffer,
        swapchain_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &clear_value,
        &[subresource_range(1)],
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
            framebuffer_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy_region],
        );

        if let Some(save_image) = save_image {
            let copy_region = vk::BufferImageCopy::builder()
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
                    width: render_extent.x,
                    height: render_extent.y,
                    depth: 1,
                })
                .build();

            device.cmd_copy_image_to_buffer(
                command_buffer,
                framebuffer_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                save_image.buffer.buffer,
                &[copy_region],
            );
        }
    }

    // optionally draw the GUI on top via the render pass
    let mut swap_layout_after = vk::ImageLayout::TRANSFER_DST_OPTIMAL;

    if gui.has_draws() {
        cmd_image_barrier(
            device,
            command_buffer,
            image_barrier(
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                1,
            ),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );

        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        };
        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(swapchain.render_pass)
            .framebuffer(swapchain.framebuffers[swapchain_index])
            .render_area(render_area);

        device.cmd_begin_render_pass(command_buffer, &begin_info, vk::SubpassContents::INLINE);

        gui.record_draws(device, command_buffer, frame_index, swapchain.extent)?;

        device.cmd_end_render_pass(command_buffer);
        swap_layout_after = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }

    // path traced frame back to compute-writable
    cmd_image_barrier(
        device,
        command_buffer,
        image_barrier(
            framebuffer_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::SHADER_WRITE,
            1,
        ),
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
    );

    // swapchain image -> presentable
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

    cmd_image_barrier(
        device,
        command_buffer,
        image_barrier(
            swapchain_image,
            swap_layout_after,
            vk::ImageLayout::PRESENT_SRC_KHR,
            present_src_access,
            vk::AccessFlags::empty(),
            1,
        ),
        present_src_stage,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    );

    if let Some(layout) = swapchain.image_layouts.get_mut(swapchain_index) {
        *layout = vk::ImageLayout::PRESENT_SRC_KHR;
    }

    // end time
    device.cmd_write_timestamp(
        command_buffer,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        query_pool,
        first_query + 1,
    );

    device.end_command_buffer(command_buffer)?;

    Ok(())
}
