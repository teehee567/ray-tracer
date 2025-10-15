use crate::scene::Scene;
use crate::vulkan::texture::Texture;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk;

#[derive(Clone, Debug, Default)]
pub struct AppData {
    pub scene: Scene,

    pub messenger: vk::DebugUtilsMessengerEXT,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_layouts: Vec<vk::ImageLayout>,
    pub render_pass: vk::RenderPass,
    pub present_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub compute_pipeline: vk::Pipeline,
    pub compute_pipeline_layout: vk::PipelineLayout,

    pub compute_descriptor_sets: Vec<vk::DescriptorSet>,
    pub compute_command_buffers: Vec<vk::CommandBuffer>,
    pub present_command_buffer: vk::CommandBuffer,

    pub image_available_semaphores: vk::Semaphore,
    pub compute_finished_semaphores: vk::Semaphore,

    pub uniform_buffer_memory: vk::DeviceMemory,
    pub compute_ssbo_buffer_memory: vk::DeviceMemory,

    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub swapchain_format: vk::Format,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_framebuffers: Vec<vk::Framebuffer>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub command_pool: vk::CommandPool,
    pub uniform_buffer: vk::Buffer,
    pub compute_ssbo_buffer: vk::Buffer,
    pub descriptor_pool: vk::DescriptorPool,
    pub accumulator_view: vk::ImageView,
    pub accumulator_memory: vk::DeviceMemory,
    pub accumulator_image: vk::Image,
    pub sampler: vk::Sampler,

    pub textures: Vec<Texture>,
    pub texture_sampler: vk::Sampler,
    pub skybox_texture: Texture,
    pub skybox_sampler: vk::Sampler,

    pub framebuffer_images: Vec<vk::Image>,
    pub framebuffer_image_views: Vec<vk::ImageView>,
    pub framebuffer_memories: Vec<vk::DeviceMemory>,
}
