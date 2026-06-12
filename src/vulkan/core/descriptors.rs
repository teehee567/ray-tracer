use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

pub unsafe fn create_descriptor_set_layout(
    device: &Device,
    bindings: &[vk::DescriptorSetLayoutBinding],
) -> Result<vk::DescriptorSetLayout> {
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);
    Ok(device.create_descriptor_set_layout(&info, None)?)
}

pub unsafe fn create_descriptor_pool(
    device: &Device,
    pool_sizes: &[vk::DescriptorPoolSize],
    max_sets: u32,
    flags: vk::DescriptorPoolCreateFlags,
) -> Result<vk::DescriptorPool> {
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(max_sets)
        .flags(flags);
    Ok(device.create_descriptor_pool(&info, None)?)
}

pub unsafe fn allocate_descriptor_sets(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: usize,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![layout; count];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    Ok(device.allocate_descriptor_sets(&info)?)
}

pub fn binding(
    slot: u32,
    ty: vk::DescriptorType,
    count: u32,
    stages: vk::ShaderStageFlags,
) -> vk::DescriptorSetLayoutBinding {
    vk::DescriptorSetLayoutBinding::builder()
        .binding(slot)
        .descriptor_type(ty)
        .descriptor_count(count)
        .stage_flags(stages)
        .build()
}

pub fn pool_size(ty: vk::DescriptorType, count: u32) -> vk::DescriptorPoolSize {
    vk::DescriptorPoolSize::builder()
        .type_(ty)
        .descriptor_count(count)
        .build()
}

pub fn buffer_info(buffer: vk::Buffer, offset: u64, range: u64) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::builder()
        .buffer(buffer)
        .offset(offset)
        .range(range)
        .build()
}

pub fn image_info(
    sampler: vk::Sampler,
    view: vk::ImageView,
    layout: vk::ImageLayout,
) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::builder()
        .sampler(sampler)
        .image_view(view)
        .image_layout(layout)
        .build()
}

pub fn buffer_write(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    infos: &[vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet {
    vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(binding)
        .dst_array_element(0)
        .descriptor_type(ty)
        .buffer_info(infos)
        .build()
}

pub fn image_write(
    set: vk::DescriptorSet,
    binding: u32,
    ty: vk::DescriptorType,
    infos: &[vk::DescriptorImageInfo],
) -> vk::WriteDescriptorSet {
    vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(binding)
        .dst_array_element(0)
        .descriptor_type(ty)
        .image_info(infos)
        .build()
}
