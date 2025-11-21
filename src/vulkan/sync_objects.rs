use log::info;
use vulkanalia::prelude::v1_0::*;

use anyhow::Result;

pub unsafe fn create_sync_objects(device: &Device) -> Result<(vk::Semaphore, vk::Semaphore)> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();

    let image_available_semaphores = device.create_semaphore(&semaphore_info, None)?;
    let compute_finished_semaphores = device.create_semaphore(&semaphore_info, None)?;

    info!("Creatd Sync Objects");
    // data.images_in_flight = data.swapchain_images.iter().map(|_| vk::Fence::null()).collect();

    Ok((image_available_semaphores, compute_finished_semaphores))
}
