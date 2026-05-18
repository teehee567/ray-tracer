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
