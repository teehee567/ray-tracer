
use log::{debug, info};
use vulkanalia::prelude::v1_0::*;

use crate::AppData;
use anyhow::Result;

pub unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    let max_sets = data.swapchain_image_views.len() as u32;
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(max_sets);
    let sbo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(2 * max_sets);
    let sbo_size1 = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(2 * max_sets);



    let pool_sizes = &[ubo_size, sbo_size, sbo_size1];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;
    info!("Created descriptor_pool: {:?}", info);

    Ok(())
}

pub unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData, bvh_size: u64, mat_size: u64) -> Result<()> {
    // Allocate

    let layouts = vec![data.descriptor_set_layout; data.swapchain_image_views.len()];
    debug!("Layouts: {:?}", layouts.len());
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.compute_descriptor_sets = device.allocate_descriptor_sets(&info)?;
    info!("Allocated Descriptor sets");

    for (i, &swapchain_image_view) in data.swapchain_image_views.iter().enumerate() {
        debug!("Started Update Descriptor Sets");
        let uniform_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE as u64);

        let bvh_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.compute_ssbo_buffer)
            .offset(0)
            .range(bvh_size);

        let material_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.compute_ssbo_buffer)
            .offset(bvh_size)
            .range(mat_size);

        let triangle_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(data.compute_ssbo_buffer)
            .offset(bvh_size + mat_size)
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

        let bvh_shader = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&[bvh_buffer_info])
            .build();

        let material_shader = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(2)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&[material_buffer_info])
            .build();

        let triangle_shader = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(3)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&[triangle_buffer_info])
            .build();

        let write_accumulator = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(4)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&[accumulator_image_info])
            .build();

        let write_swapchain = vk::WriteDescriptorSet::builder()
            .dst_set(data.compute_descriptor_sets[i])
            .dst_binding(5)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&[swapchain_image_info])
            .build();

        let descriptor_writes = [
            write_uniform,
            bvh_shader,
            material_shader,
            triangle_shader,
            write_accumulator,
            write_swapchain,
        ];

        device.update_descriptor_sets(&descriptor_writes, &[] as &[vk::CopyDescriptorSet]);
        info!("Updated Descriptor Sets");
    }

    Ok(())
}

pub unsafe fn create_compute_descriptor_set_layout(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let bvh_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let material_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let triangle_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(3)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let accumulator_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(4)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let swapchain_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(5)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE);

    let bindings = &[ubo_binding, bvh_binding, material_binding, triangle_binding, accumulator_binding, swapchain_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;
    info!("Created Desciptor set layout: {:?} with bindings: {:?}", data.descriptor_set_layout, bindings);

    Ok(())
}
