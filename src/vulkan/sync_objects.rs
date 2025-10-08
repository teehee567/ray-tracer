
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::AppData;
use anyhow::Result;

pub unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();

    data.image_available_semaphores = device.create_semaphore(&semaphore_info, None)?;
    data.compute_finished_semaphores = device.create_semaphore(&semaphore_info, None)?;

    info!("Creatd Sync Objects");
    // data.images_in_flight = data.swapchain_images.iter().map(|_| vk::Fence::null()).collect();

    Ok(())
}
