
use vulkanalia::prelude::v1_0::*;

use crate::AppData;
use anyhow::Result;

pub unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1);
    let sbo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1);


    let pool_sizes = &[ubo_size, sbo_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

pub unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    // Allocate

    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.compute_descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for (i, &swapchain_image_view) in data.swapchain_image_views.iter().enumerate() {
        let uniform_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE as u64);

        let shader_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.compute_ssbo_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE as u64);

        let accumulator_image_info = vk::DescriptorImageInfo::builder()
            .sampler(data.sampler)
            .image_view(data.accumulator_view)
            .image_layout(vk::ImageLayout::GENERAL);

        let swapchain_image_info = vk::DescriptorImageInfo::builder()
            .sampler(data.sampler)
            .image_view(data.swapchain_image_views[i])
            .image_layout(vk::ImageLayout::GENERAL);

        let write_uniform = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&[uniform_buffer_info])
            .build();

        let write_shader = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&[shader_buffer_info])
            .build();

        let write_accumulator = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&[accumulator_image_info])
            .build();

        let write_swapchain = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&[swapchain_image_info])
            .build();

        let descriptor_writes = [
            write_uniform,
            write_shader,
            write_accumulator,
            write_swapchain,
        ];

        device.update_descriptor_sets(&descriptor_writes, &[] as &[vk::CopyDescriptorSet]);
    }

    // Update

    // for i in 0..data.swapchain_images.len() {
    //     let ubo_info = vk::DescriptorBufferInfo::builder()
    //         .buffer(data.uniform_buffers[i])
    //         .offset(0)
    //         .range(size_of::<UniformBufferObject>() as u64);

    //     let sbo_info = vk::DescriptorBufferInfo::builder()
    //         .buffer(data.shader_buffer)
    //         .offset(0)
    //         .range(vk::WHOLE_SIZE as u64);



    //     let ubo_buffer_info = &[ubo_info];
    //     let ubo_write = vk::WriteDescriptorSet::builder()
    //         .dst_set(data.descriptor_sets[i])
    //         .dst_binding(0)
    //         .dst_array_element(0)
    //         .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
    //         .buffer_info(ubo_buffer_info);

    //     let sbo_buffer_info = &[sbo_info];
    //     let sbo_write = vk::WriteDescriptorSet::builder()
    //         .dst_set(data.descriptor_sets[i])
    //         .dst_binding(1)
    //         .dst_array_element(0)
    //         .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
    //         .buffer_info(sbo_buffer_info);

    //     device.update_descriptor_sets(&[ubo_write, sbo_write], &[] as &[vk::CopyDescriptorSet]);
    // }

    Ok(())
}

pub unsafe fn create_compute_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let sbo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let accumulator_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let swapchain_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(3)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let bindings = &[ubo_binding, sbo_binding, accumulator_binding, swapchain_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}