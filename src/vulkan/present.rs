use anyhow::Result;
use glam::{Mat4, UVec2};
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::heatmap_renderer::HeatmapRenderer;
use crate::vulkan::utils::save_frame::SaveImage;

use super::core::image::{cmd_image_barrier, image_barrier, subresource_range};
use super::core::swapchain::Swapchain;
use super::gui_renderer::GuiRenderer;

/// Per-call data for one present recording.
pub(super) struct PresentFrame<'a> {
    pub command_buffer: vk::CommandBuffer,
    pub swapchain_index: usize,
    pub frame_index: usize,
    /// The path tracer's offscreen target for this frame slot.
    pub framebuffer_image: vk::Image,
    pub panel_width: u32,
    pub render_extent: UVec2,
    pub save_image: Option<&'a SaveImage>,
}

/// Heatmap overlay parameters for one present recording.
pub(super) struct HeatmapParams {
    pub active: bool,
    pub view_proj: Mat4,
    pub band: (u32, u32),
}

/// Timer query pools written during the present pass.
pub(super) struct PresentTimers {
    pub present: vk::QueryPool,
    pub heatmap: vk::QueryPool,
    pub compositor: vk::QueryPool,
}

/// Record the present pass: blit the path traced frame into the swapchain
/// image (offset past the GUI panel), optionally accumulate/composite the
/// BVH heatmap, and draw the GUI on top.
pub(super) unsafe fn record_present_commands(
    device: &Device,
    swapchain: &mut Swapchain,
    gui: &GuiRenderer,
    heatmap: &mut HeatmapRenderer,
    frame: &PresentFrame,
    heatmap_params: &HeatmapParams,
    timers: &PresentTimers,
) -> Result<()> {
    let command_buffer = frame.command_buffer;
    let swapchain_image = swapchain.images[frame.swapchain_index];
    let begin_info = vk::CommandBufferBeginInfo::builder();
    device.begin_command_buffer(command_buffer, &begin_info)?;

    let first_query = frame.frame_index as u32 * 2;
    write_begin_timestamps(device, command_buffer, timers, first_query);

    // clamp copy to swapchain
    let copy_w = frame
        .render_extent
        .x
        .min(swapchain.extent.width.saturating_sub(frame.panel_width));
    let copy_h = frame.render_extent.y.min(swapchain.extent.height);

    blit_path_trace_output(device, swapchain, frame, swapchain_image, copy_w, copy_h);

    if heatmap_params.active {
        record_heatmap_prepass(
            device,
            command_buffer,
            heatmap,
            heatmap_params,
            timers,
            first_query,
        );
    }

    let mut swap_layout_after = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
    let draw_in_pass = gui.has_draws() || heatmap_params.active;

    if draw_in_pass {
        record_swapchain_pass(
            device,
            swapchain,
            gui,
            heatmap,
            frame,
            heatmap_params,
            timers,
            swapchain_image,
            first_query,
            copy_w,
            copy_h,
        )?;
        swap_layout_after = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
    }

    finalize_layouts(device, swapchain, frame, swapchain_image, swap_layout_after);

    // end time
    device.cmd_write_timestamp(
        command_buffer,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        timers.present,
        first_query + 1,
    );

    device.end_command_buffer(command_buffer)?;

    Ok(())
}

unsafe fn write_begin_timestamps(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    timers: &PresentTimers,
    first_query: u32,
) {
    device.cmd_reset_query_pool(command_buffer, timers.present, first_query, 2);
    device.cmd_write_timestamp(
        command_buffer,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        timers.present,
        first_query,
    );

    // reset the heatmap/compositor timer slots here (outside any render pass);
    // their start/end timestamps are written around the passes below.
    device.cmd_reset_query_pool(command_buffer, timers.heatmap, first_query, 2);
    device.cmd_reset_query_pool(command_buffer, timers.compositor, first_query, 2);
}

/// Barriers + clear + copy of the path traced frame into the swapchain image,
/// plus the optional copy into the save-frame staging buffer.
unsafe fn blit_path_trace_output(
    device: &Device,
    swapchain: &Swapchain,
    frame: &PresentFrame,
    swapchain_image: vk::Image,
    copy_w: u32,
    copy_h: u32,
) {
    let command_buffer = frame.command_buffer;

    // path traced frame: compute output -> transfer source
    cmd_image_barrier(
        device,
        command_buffer,
        image_barrier(
            frame.framebuffer_image,
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
        .get(frame.swapchain_index)
        .copied()
        .unwrap_or(vk::ImageLayout::UNDEFINED);

    cmd_image_barrier(
        device,
        command_buffer,
        image_barrier(
            swapchain_image,
            current_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            1,
        ),
        // the src stage must be the same as the semaphore
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::TRANSFER,
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

    if frame.render_extent.x > 0 && frame.render_extent.y > 0 {
        // clear writes the whole swapchain image and copy writes the
        // subregion inside it, both transfer so clear before the copy
        cmd_image_barrier(
            device,
            command_buffer,
            image_barrier(
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::TRANSFER_WRITE,
                1,
            ),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
        );

        if copy_w > 0 && copy_h > 0 {
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
                    x: frame.panel_width as i32,
                    y: 0,
                    z: 0,
                })
                .extent(vk::Extent3D {
                    width: copy_w,
                    height: copy_h,
                    depth: 1,
                })
                .build();

            device.cmd_copy_image(
                command_buffer,
                frame.framebuffer_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swapchain_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
        }

        if let Some(save_image) = frame.save_image {
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
                    width: frame.render_extent.x,
                    height: frame.render_extent.y,
                    depth: 1,
                })
                .build();

            device.cmd_copy_image_to_buffer(
                command_buffer,
                frame.framebuffer_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                save_image.buffer.buffer,
                &[copy_region],
            );
        }
    }
}

/// Heatmap accumulation + reduce, both outside the swapchain render pass.
unsafe fn record_heatmap_prepass(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    heatmap: &mut HeatmapRenderer,
    heatmap_params: &HeatmapParams,
    timers: &PresentTimers,
    first_query: u32,
) {
    // time the accumulation pass
    device.cmd_write_timestamp(
        command_buffer,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        timers.heatmap,
        first_query,
    );
    heatmap.record_into(
        device,
        command_buffer,
        heatmap_params.view_proj,
        heatmap_params.band.0,
        heatmap_params.band.1,
    );
    device.cmd_write_timestamp(
        command_buffer,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        timers.heatmap,
        first_query + 1,
    );

    // compositor timer spans the reduce + the composite draw (the end
    // timestamp is written inside the swapchain render pass below).
    device.cmd_write_timestamp(
        command_buffer,
        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        timers.compositor,
        first_query,
    );
    // reduce the accumulation image to its peak overlap (outside any render
    // pass) so the composite can normalize against the true max
    heatmap.record_reduce(device, command_buffer);
}

/// The swapchain render pass: heatmap composite (under the GUI) + GUI draws.
#[allow(clippy::too_many_arguments)]
unsafe fn record_swapchain_pass(
    device: &Device,
    swapchain: &Swapchain,
    gui: &GuiRenderer,
    heatmap: &HeatmapRenderer,
    frame: &PresentFrame,
    heatmap_params: &HeatmapParams,
    timers: &PresentTimers,
    swapchain_image: vk::Image,
    first_query: u32,
    copy_w: u32,
    copy_h: u32,
) -> Result<()> {
    let command_buffer = frame.command_buffer;

    cmd_image_barrier(
        device,
        command_buffer,
        image_barrier(
            swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            // loadop writes and reads for some reason
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
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
        .framebuffer(swapchain.framebuffers[frame.swapchain_index])
        .render_area(render_area);

    device.cmd_begin_render_pass(command_buffer, &begin_info, vk::SubpassContents::INLINE);

    // composite the heatmap into its sub-region first so the GUI panels
    // (drawn next) land on top of it.
    if heatmap_params.active {
        let sub_region = vk::Rect2D {
            offset: vk::Offset2D {
                x: frame.panel_width as i32,
                y: 0,
            },
            extent: vk::Extent2D {
                width: copy_w,
                height: copy_h,
            },
        };
        heatmap.record_composite(device, command_buffer, sub_region);
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            timers.compositor,
            first_query + 1,
        );
    }

    if gui.has_draws() {
        gui.record_draws(device, command_buffer, frame.frame_index, swapchain.extent)?;
    }

    device.cmd_end_render_pass(command_buffer);

    Ok(())
}

/// Return the path traced frame to compute-writable and the swapchain image
/// to presentable, updating the tracked layout.
unsafe fn finalize_layouts(
    device: &Device,
    swapchain: &mut Swapchain,
    frame: &PresentFrame,
    swapchain_image: vk::Image,
    swap_layout_after: vk::ImageLayout,
) {
    let command_buffer = frame.command_buffer;

    // path traced frame back to compute-writable
    cmd_image_barrier(
        device,
        command_buffer,
        image_barrier(
            frame.framebuffer_image,
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

    if let Some(layout) = swapchain.image_layouts.get_mut(frame.swapchain_index) {
        *layout = vk::ImageLayout::PRESENT_SRC_KHR;
    }
}
