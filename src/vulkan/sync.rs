use vulkanalia::prelude::v1_0::*;

#[derive(Clone, Debug)]
pub struct SyncState {
    pub compute_command_buffers: Vec<vk::CommandBuffer>,
    pub present_command_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    pub compute_finished_semaphore: vk::Semaphore,
}
