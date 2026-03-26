use vulkanalia::prelude::v1_0::*;

#[derive(Clone, Debug)]
pub struct VulkanContext {
    pub physical_device: vk::PhysicalDevice,
    pub surface: vk::SurfaceKHR,
    pub command_pool: vk::CommandPool,
    pub compute_queue: vk::Queue,
    pub present_queue: vk::Queue,
}
