use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::QueueFamilyIndices;
use anyhow::Result;

pub unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<vk::CommandPool> {
    let indices = QueueFamilyIndices::get(instance, surface, physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics);

    let command_pool = device.create_command_pool(&info, None)?;
    info!("Created Command Pool: {:?}", command_pool);

    Ok(command_pool)
}
