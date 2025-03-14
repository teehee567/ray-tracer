
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::{AppData, QueueFamilyIndices};
use anyhow::Result;

pub unsafe fn create_command_pool(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
    .queue_family_index(indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;
    info!("Created Command Pool: {:?}", data.command_pool);

    Ok(())
}
