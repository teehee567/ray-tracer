use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{self, DeviceV1_0};

use crate::app::data::AppData;
use crate::vulkan::utils::get_memory_type_index;

/// Transitions an image layout using a pipeline barrier.
pub unsafe fn transition_image_layout(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_stage, src_access) = match old_layout {
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::empty(),
        ),
    };

    let (dst_stage, dst_access) = match new_layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::empty(),
        ),
    };

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        )
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .build();

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );
}

/// Uploads pixel data to an image using a staging buffer.
/// Handles both full uploads and partial updates with offsets.
pub unsafe fn upload_pixels_to_image(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    image: vk::Image,
    pixels: &[u8],
    size: [u32; 2],
    offset: Option<[u32; 2]>,
    is_new: bool,
) -> Result<()> {
    // Create staging buffer
    let staging_info = vk::BufferCreateInfo::builder()
        .size(pixels.len() as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let staging_buffer = device.create_buffer(&staging_info, None)?;

    let requirements = device.get_buffer_memory_requirements(staging_buffer);
    let memory_type = get_memory_type_index(
        instance,
        data,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        requirements,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let staging_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(staging_buffer, staging_memory, 0)?;

    // Copy pixel data to staging buffer
    let ptr = device.map_memory(
        staging_memory,
        0,
        requirements.size,
        vk::MemoryMapFlags::empty(),
    )? as *mut u8;
    ptr.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
    device.unmap_memory(staging_memory);

    // Create command buffer for the transfer
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(command_buffer, &begin_info)?;

    // Transition image to transfer destination
    let old_layout = if is_new {
        vk::ImageLayout::UNDEFINED
    } else {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    };
    transition_image_layout(
        device,
        command_buffer,
        image,
        old_layout,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    // Copy buffer to image
    let offset = offset.unwrap_or([0, 0]);
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
        .image_offset(vk::Offset3D {
            x: offset[0] as i32,
            y: offset[1] as i32,
            z: 0,
        })
        .image_extent(vk::Extent3D {
            width: size[0],
            height: size[1],
            depth: 1,
        })
        .build();

    device.cmd_copy_buffer_to_image(
        command_buffer,
        staging_buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    // Transition image to shader read-only
    transition_image_layout(
        device,
        command_buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    device.end_command_buffer(command_buffer)?;

    // Submit and wait
    let command_buffers = [command_buffer];
    let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
    device.queue_submit(data.compute_queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(data.compute_queue)?;

    // Cleanup
    device.free_command_buffers(data.command_pool, &[command_buffer]);
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);

    Ok(())
}
