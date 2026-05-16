use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::image::{cmd_transition_image_layout, create_image_2d, create_image_view_2d};
use super::single_time::with_single_time;

pub unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    swapchain_extent: vk::Extent2D,
    physical_device: vk::PhysicalDevice,
) -> Result<(vk::Image, vk::ImageView, vk::DeviceMemory)> {
    let (image, memory) = create_image_2d(
        instance,
        device,
        physical_device,
        swapchain_extent.width,
        swapchain_extent.height,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        1,
        vk::ImageCreateFlags::empty(),
    )?;

    let view = create_image_view_2d(
        device,
        image,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageViewType::_2D,
        vk::ImageAspectFlags::COLOR,
        1,
    )?;

    Ok((image, view, memory))
}

pub unsafe fn transition_image_layout(
    device: &Device,
    command_pool: vk::CommandPool,
    compute_queue: vk::Queue,
    accumulator_image: vk::Image,
) -> Result<()> {
    with_single_time(device, command_pool, compute_queue, |cb| {
        cmd_transition_image_layout(
            device,
            cb,
            accumulator_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            vk::ImageAspectFlags::COLOR,
            1,
        )
    })
}
