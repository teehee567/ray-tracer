use vulkanalia::prelude::v1_0::*;

#[derive(Clone, Debug)]
pub struct ComputeResources {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub uniform_buffer: vk::Buffer,
    pub uniform_buffer_memory: vk::DeviceMemory,
    pub ssbo_buffer: vk::Buffer,
    pub ssbo_buffer_memory: vk::DeviceMemory,
    pub accumulator_image: vk::Image,
    pub accumulator_view: vk::ImageView,
    pub accumulator_memory: vk::DeviceMemory,
    pub sampler: vk::Sampler,
    pub framebuffer_images: Vec<vk::Image>,
    pub framebuffer_image_views: Vec<vk::ImageView>,
    pub framebuffer_memories: Vec<vk::DeviceMemory>,
}
