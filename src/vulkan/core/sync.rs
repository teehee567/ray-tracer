use anyhow::Result;
use log::info;
use vulkanalia::prelude::v1_0::*;

use super::commands::allocate_command_buffers;

/// Per-frame command buffers, semaphores and fences for the
/// compute-dispatch / present loop.
#[derive(Clone, Debug)]
pub struct SyncState {
    pub compute_command_buffers: Vec<vk::CommandBuffer>,
    pub present_command_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub frame_fences: Vec<vk::Fence>,
    pub present_fence: vk::Fence,
}

impl SyncState {
    pub unsafe fn new(
        device: &Device,
        command_pool: vk::CommandPool,
        frame_count: usize,
        swapchain_image_count: usize,
    ) -> Result<Self> {
        let mut buffers = allocate_command_buffers(device, command_pool, frame_count as u32 + 1)?;
        let present_command_buffer = buffers.pop().unwrap();
        let compute_command_buffers = buffers;

        let semaphore_info = vk::SemaphoreCreateInfo::builder();
        let image_available_semaphore = device.create_semaphore(&semaphore_info, None)?;
        let mut render_finished_semaphores = Vec::with_capacity(swapchain_image_count);
        for _ in 0..swapchain_image_count {
            render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
        }

        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let mut frame_fences = Vec::with_capacity(frame_count);
        for _ in 0..frame_count {
            frame_fences.push(device.create_fence(&fence_info, None)?);
        }
        let present_fence = device.create_fence(&fence_info, None)?;

        info!("Created sync objects and command buffers");

        Ok(Self {
            compute_command_buffers,
            present_command_buffer,
            image_available_semaphore,
            render_finished_semaphores,
            frame_fences,
            present_fence,
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device, command_pool: vk::CommandPool) {
        let mut all_command_buffers = self.compute_command_buffers.clone();
        all_command_buffers.push(self.present_command_buffer);
        device.free_command_buffers(command_pool, &all_command_buffers);
        self.compute_command_buffers.clear();

        for &fence in &self.frame_fences {
            device.destroy_fence(fence, None);
        }
        self.frame_fences.clear();
        device.destroy_fence(self.present_fence, None);
        device.destroy_semaphore(self.image_available_semaphore, None);
        for &semaphore in &self.render_finished_semaphores {
            device.destroy_semaphore(semaphore, None);
        }
        self.render_finished_semaphores.clear();
    }
}
