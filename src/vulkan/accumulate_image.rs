use anyhow::Result;
use log::info;
use vulkanalia::{
    prelude::v1_0::*,
    vk::{ImageSubresourceRange, ImageViewCreateInfo},
};

use super::utils::get_memory_type_index;
use crate::AppData;

pub unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    swapchain_extent: vk::Extent2D,
    data: &AppData,
) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory)> {
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent3D {
            width: swapchain_extent.width,
            height: swapchain_extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .usage(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC,
        )
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = device.create_image(&image_info.build(), None)?;
    info!("Created and image: {:?}", image);

    let mem_requirements = device.get_image_memory_requirements(image);

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mem_requirements,
        )?);

    let image_memory = device.allocate_memory(&alloc_info.build(), None)?;

    device.bind_image_memory(image, image_memory, 0)?;
    info!("Binded image memory: {:?}", image_memory);

    let view_info = ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .subresource_range(
            ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        )
        .build();

    let image_view = device.create_image_view(&view_info, None)?;
    info!("Created image view: {:?}", image_view);

    Ok((image, image_view, image_memory))
}

pub unsafe fn transition_image_layout(
    device: &Device,
    command_pool: vk::CommandPool,
    compute_queue: vk::Queue,
    accumulator_image: vk::Image,
) -> Result<()> {
    info!("Creating transition image layout");
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    // Commands

    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &info)?;

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .image(accumulator_image)
        .subresource_range(
            ImageSubresourceRange::builder()
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
        &[barrier],
    );

    device.end_command_buffer(command_buffer)?;

    // Submit

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    device.queue_submit(compute_queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(compute_queue)?;

    // Cleanup

    device.free_command_buffers(command_pool, &[command_buffer]);

    Ok(())
}
