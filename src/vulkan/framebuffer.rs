use anyhow::Result;
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::AppData;

use super::utils::get_memory_type_index;

pub unsafe fn create_framebuffer_images(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    count: usize,
) -> Result<()> {
    let mut images = Vec::with_capacity(count);
    let mut views = Vec::with_capacity(count);
    let mut memories = Vec::with_capacity(count);

    for _ in 0..count {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .format(data.swapchain_format)
            .extent(vk::Extent3D {
                width: data.swapchain_extent.width,
                height: data.swapchain_extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = device.create_image(&image_info, None)?;
        info!("Created offscreen framebuffer image: {:?}", image);

        let requirements = device.get_image_memory_requirements(image);
        let allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(get_memory_type_index(
                instance,
                data,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                requirements,
            )?);

        let memory = device.allocate_memory(&allocate_info, None)?;
        device.bind_image_memory(image, memory, 0)?;

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(data.swapchain_format)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );

        let view = device.create_image_view(&view_info, None)?;

        images.push(image);
        views.push(view);
        memories.push(memory);
    }

    data.framebuffer_images = images;
    data.framebuffer_image_views = views;
    data.framebuffer_memories = memories;

    Ok(())
}

pub unsafe fn transition_framebuffer_images(device: &Device, data: &mut AppData) -> Result<()> {
    if data.framebuffer_images.is_empty() {
        return Ok(());
    }

    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &begin_info)?;

    for &image in &data.framebuffer_images {
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
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
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::empty());

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
            &[barrier.build()],
        );
    }

    device.end_command_buffer(command_buffer)?;

    let submit_info =
        vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));

    device.queue_submit(data.compute_queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(data.compute_queue)?;

    device.free_command_buffers(data.command_pool, &[command_buffer]);

    Ok(())
}
