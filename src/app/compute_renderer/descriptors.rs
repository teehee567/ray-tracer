use anyhow::Result;
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::types::CameraBufferObject;
use crate::vulkan::descriptors::{allocate_descriptor_sets, create_descriptor_pool};
use crate::vulkan::texture::Texture;

pub unsafe fn create_path_tracer_descriptor_pool(
    device: &Device,
    framebuffer_image_view_count: usize,
    texture_count: usize,
) -> Result<vk::DescriptorPool> {
    let max_sets = framebuffer_image_view_count as u32;
    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(max_sets)
            .build(),
        vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(6 * max_sets)
            .build(),
        vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(2 * max_sets)
            .build(),
        vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(max_sets * (texture_count as u32 + 1))
            .build(),
    ];
    create_descriptor_pool(device, &pool_sizes, max_sets, vk::DescriptorPoolCreateFlags::empty())
}

pub unsafe fn create_path_tracer_descriptor_sets(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    framebuffer_image_views: &[vk::ImageView],
    uniform_buffer: vk::Buffer,
    ssbo_buffer: vk::Buffer,
    accumulator_view: vk::ImageView,
    sampler: vk::Sampler,
    textures: &[Texture],
    texture_sampler: vk::Sampler,
    skybox_texture: &Texture,
    skybox_sampler: vk::Sampler,
    bvh_size: u64,
    mat_size: u64,
    triangle_size: u64,
    light_size: u64,
    emissive_tri_size: u64,
    cdf_size: u64,
) -> Result<Vec<vk::DescriptorSet>> {
    let sets = allocate_descriptor_sets(
        device,
        descriptor_pool,
        descriptor_set_layout,
        framebuffer_image_views.len(),
    )?;
    info!("Allocated Descriptor sets len {}", sets.len());

    for (i, framebuffer_view) in framebuffer_image_views.iter().enumerate() {
        let uniform_info = buffer_info(uniform_buffer, 0, std::mem::size_of::<CameraBufferObject>() as u64);
        let bvh_info = buffer_info(ssbo_buffer, 0, bvh_size);
        let material_info = buffer_info(ssbo_buffer, bvh_size, mat_size);
        let triangle_info = buffer_info(ssbo_buffer, bvh_size + mat_size, triangle_size);
        let light_info = buffer_info(ssbo_buffer, bvh_size + mat_size + triangle_size, light_size);
        let emissive_info = buffer_info(
            ssbo_buffer,
            bvh_size + mat_size + triangle_size + light_size,
            emissive_tri_size,
        );
        let cdf_info = buffer_info(
            ssbo_buffer,
            bvh_size + mat_size + triangle_size + light_size + emissive_tri_size,
            cdf_size,
        );

        let accumulator_info = storage_image_info(sampler, accumulator_view);
        let framebuffer_info = storage_image_info(sampler, *framebuffer_view);

        let texture_infos: Vec<_> = textures
            .iter()
            .map(|tex| sampled_image_info(texture_sampler, tex.view))
            .collect();
        let skybox_info = sampled_image_info(skybox_sampler, skybox_texture.view);

        let uniform_array = [uniform_info];
        let bvh_array = [bvh_info];
        let material_array = [material_info];
        let triangle_array = [triangle_info];
        let light_array = [light_info];
        let emissive_array = [emissive_info];
        let cdf_array = [cdf_info];
        let accumulator_array = [accumulator_info];
        let framebuffer_array = [framebuffer_info];
        let skybox_array = [skybox_info];

        let writes = [
            buffer_write(sets[i], 0, vk::DescriptorType::UNIFORM_BUFFER, &uniform_array),
            buffer_write(sets[i], 1, vk::DescriptorType::STORAGE_BUFFER, &bvh_array),
            buffer_write(sets[i], 2, vk::DescriptorType::STORAGE_BUFFER, &material_array),
            buffer_write(sets[i], 3, vk::DescriptorType::STORAGE_BUFFER, &triangle_array),
            image_write(sets[i], 4, vk::DescriptorType::STORAGE_IMAGE, &accumulator_array),
            image_write(sets[i], 5, vk::DescriptorType::STORAGE_IMAGE, &framebuffer_array),
            image_write(sets[i], 6, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, &texture_infos),
            image_write(sets[i], 7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, &skybox_array),
            buffer_write(sets[i], 8, vk::DescriptorType::STORAGE_BUFFER, &light_array),
            buffer_write(sets[i], 9, vk::DescriptorType::STORAGE_BUFFER, &emissive_array),
            buffer_write(sets[i], 10, vk::DescriptorType::STORAGE_BUFFER, &cdf_array),
        ];

        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }

    Ok(sets)
}

fn buffer_info(buffer: vk::Buffer, offset: u64, range: u64) -> vk::DescriptorBufferInfo {
    vk::DescriptorBufferInfo::builder()
        .buffer(buffer)
        .offset(offset)
        .range(range)
        .build()
}

fn storage_image_info(sampler: vk::Sampler, view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::builder()
        .sampler(sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::GENERAL)
        .build()
}

fn sampled_image_info(sampler: vk::Sampler, view: vk::ImageView) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::builder()
        .sampler(sampler)
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .build()
}

fn buffer_write(
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

fn image_write(
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
