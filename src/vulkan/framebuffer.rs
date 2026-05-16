use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::image::{cmd_transition_image_layout, create_image_2d, create_image_view_2d};
use super::single_time::with_single_time;

pub unsafe fn create_framebuffer_images(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    format: vk::Format,
    extent: vk::Extent2D,
    count: usize,
) -> Result<(Vec<vk::Image>, Vec<vk::ImageView>, Vec<vk::DeviceMemory>)> {
    let mut images = Vec::with_capacity(count);
    let mut views = Vec::with_capacity(count);
    let mut memories = Vec::with_capacity(count);

    for _ in 0..count {
        let (image, memory) = create_image_2d(
            instance,
            device,
            physical_device,
            extent.width,
            extent.height,
            format,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            vk::ImageCreateFlags::empty(),
        )?;

        let view = create_image_view_2d(
            device,
            image,
            format,
            vk::ImageViewType::_2D,
            vk::ImageAspectFlags::COLOR,
            1,
        )?;

        images.push(image);
        views.push(view);
        memories.push(memory);
    }

    Ok((images, views, memories))
}

pub unsafe fn transition_framebuffer_images(
    device: &Device,
    command_pool: vk::CommandPool,
    compute_queue: vk::Queue,
    framebuffer_images: &[vk::Image],
) -> Result<()> {
    if framebuffer_images.is_empty() {
        return Ok(());
    }

    with_single_time(device, command_pool, compute_queue, |cb| {
        for &image in framebuffer_images {
            cmd_transition_image_layout(
                device,
                cb,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::ImageAspectFlags::COLOR,
                1,
            )?;
        }
        Ok(())
    })
}

pub unsafe fn create_swapchain_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    swapchain_image_views: &[vk::ImageView],
    swapchain_extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    let mut framebuffers = Vec::with_capacity(swapchain_image_views.len());
    for &view in swapchain_image_views {
        let attachments = [view];
        let info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(swapchain_extent.width)
            .height(swapchain_extent.height)
            .layers(1);

        framebuffers.push(device.create_framebuffer(&info, None)?);
    }

    Ok(framebuffers)
}
