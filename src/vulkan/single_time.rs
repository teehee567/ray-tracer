use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

pub unsafe fn begin_single_time_commands(
    device: &Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(command_buffer, &begin_info)?;

    Ok(command_buffer)
}

pub unsafe fn end_single_time_commands(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) -> Result<()> {
    device.end_command_buffer(command_buffer)?;

    let command_buffers = [command_buffer];
    let submit = vk::SubmitInfo::builder().command_buffers(&command_buffers);
    device.queue_submit(queue, &[submit], vk::Fence::null())?;
    device.queue_wait_idle(queue)?;

    device.free_command_buffers(command_pool, &command_buffers);
    Ok(())
}

pub unsafe fn with_single_time<F>(
    device: &Device,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    f: F,
) -> Result<()>
where
    F: FnOnce(vk::CommandBuffer) -> Result<()>,
{
    let command_buffer = begin_single_time_commands(device, command_pool)?;
    let result = f(command_buffer);
    if result.is_err() {
        let _ = device.end_command_buffer(command_buffer);
        device.free_command_buffers(command_pool, &[command_buffer]);
        return result;
    }
    end_single_time_commands(device, command_pool, queue, command_buffer)
}
