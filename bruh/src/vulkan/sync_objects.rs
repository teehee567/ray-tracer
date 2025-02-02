
use vulkanalia::prelude::v1_0::*;

use crate::AppData;
use anyhow::Result;

pub unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    data.image_available_semaphores = device.create_semaphore(&semaphore_info, None)?;
    data.compute_finished_semaphores = device.create_semaphore(&semaphore_info, None)?;

    data.compute_in_flight_fences = device.create_fence(&fence_info, None)?;

    // data.images_in_flight = data.swapchain_images.iter().map(|_| vk::Fence::null()).collect();

    Ok(())
}