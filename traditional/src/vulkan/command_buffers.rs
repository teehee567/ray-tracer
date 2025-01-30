use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;

use crate::AppData;

pub unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        // PRIMARY - Can be submitted to a queue for execution but cannot be called from other buffers
        // SECONDARY - Cannot be submitted directly, but can be called from primary command buffers
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    for (i, command_buffer) in data.command_buffers.iter().enumerate() {
        // let inheritance = vk::CommandBufferInheritanceInfo::builder();
        let info = vk::CommandBufferBeginInfo::builder();
            // ONE_TIME_SUBMIT - The command buffer will be rerecorded right after executing it once
            // RENDER_PASS_CONTINUE - This is a secondary command buffer that will be entirely
            // SIMULTANEOUS_USE - The command buffer can be resubmitted while it is also already
            // .flags(vk::CommandBufferUsageFlags::empty());
            // .inheritance_info(&inheritance);

        device.begin_command_buffer(*command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let clear_values = &[color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(data.render_pass)
            .framebuffer(data.framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);

        // INLINE - The render pass commands will be mbedded in the primary command buffer itself
        // and no secondary command buffers will be exectues
        // SECONDARY_COMMAND_BUFFERS = The render pass commands will be exectued from secondary
        // command buffers.
        device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(
            *command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            data.pipeline,
        );
        device.cmd_draw(*command_buffer, 3, 1, 0, 0);
        device.cmd_end_render_pass(*command_buffer);
        device.end_command_buffer(*command_buffer)?;
    }

    Ok(())
}
