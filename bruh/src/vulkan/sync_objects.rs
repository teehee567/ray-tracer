
use log::info;
use vulkanalia::prelude::v1_0::*;

use anyhow::Result;

pub unsafe fn create_sync_objects(device: &Device) -> Result<(vk::Semaphore, vk::Semaphore, vk::Fence)> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    let image_available_semaphores = device.create_semaphore(&semaphore_info, None)?;
    let compute_finished_semaphores = device.create_semaphore(&semaphore_info, None)?;

    let compute_in_flight_fences = device.create_fence(&fence_info, None)?;

    info!("Creatd Sync Objects");
    Ok((image_available_semaphores, compute_finished_semaphores, compute_in_flight_fences))
}
