use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::descriptors::create_descriptor_set_layout;

// 11 bindings for path tracer compute shader
pub unsafe fn create_compute_descriptor_set_layout(
    device: &Device,
    texture_count: usize,
) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        binding(0, vk::DescriptorType::UNIFORM_BUFFER, 1),         // camera ubo
        binding(1, vk::DescriptorType::STORAGE_BUFFER, 1),         // bvh
        binding(2, vk::DescriptorType::STORAGE_BUFFER, 1),         // materials
        binding(3, vk::DescriptorType::STORAGE_BUFFER, 1),         // triangles
        binding(4, vk::DescriptorType::STORAGE_IMAGE, 1),          // accumulator
        binding(5, vk::DescriptorType::STORAGE_IMAGE, 1),          // swapchain target
        binding(6, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, texture_count as u32), // textures
        binding(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1), // skybox
        binding(8, vk::DescriptorType::STORAGE_BUFFER, 1),         // lights
        binding(9, vk::DescriptorType::STORAGE_BUFFER, 1),         // emissive triangles
        binding(10, vk::DescriptorType::STORAGE_BUFFER, 1),        // cdf
    ];
    create_descriptor_set_layout(device, &bindings)
}

fn binding(slot: u32, ty: vk::DescriptorType, count: u32) -> vk::DescriptorSetLayoutBinding {
    vk::DescriptorSetLayoutBinding::builder()
        .binding(slot)
        .descriptor_type(ty)
        .descriptor_count(count)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .build()
}
