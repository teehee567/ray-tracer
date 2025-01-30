use vulkanalia::prelude::v1_0::*;
use anyhow::{anyhow, Result};

use crate::AppData;

use super::logical_device::QueueFamilyIndices;

pub unsafe fn create_command_pool(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        // TRANSIENT - Hint that command buffers are rerecorded iwth new commands very often
        // RESET_COMMAND_BUFFER - Allow command buffers to be rerecorded individually
        // PROTECTED - Creates "protected" command buffers which are stored in "protected" memory
        .flags(vk::CommandPoolCreateFlags::empty())
        .queue_family_index(indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;

    Ok(())
}
