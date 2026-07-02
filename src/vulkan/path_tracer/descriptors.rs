use anyhow::Result;
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::types::CameraBufferObject;
use crate::vulkan::core::buffer::Buffer;
use crate::vulkan::core::descriptors::{
    allocate_descriptor_sets, binding, buffer_info, buffer_write, create_descriptor_pool,
    create_descriptor_set_layout, image_info, image_write, pool_size,
};
use crate::vulkan::core::image::Image;

// 11 bindings for path tracer compute shader
pub(super) unsafe fn create_layout(
    device: &Device,
    texture_count: usize,
) -> Result<vk::DescriptorSetLayout> {
    let stage = vk::ShaderStageFlags::COMPUTE;
    let bindings = [
        binding(0, vk::DescriptorType::UNIFORM_BUFFER, 1, stage), // camera ubo
        binding(1, vk::DescriptorType::STORAGE_BUFFER, 1, stage), // bvh
        binding(2, vk::DescriptorType::STORAGE_BUFFER, 1, stage), // materials
        binding(3, vk::DescriptorType::STORAGE_BUFFER, 1, stage), // triangles
        binding(4, vk::DescriptorType::STORAGE_IMAGE, 1, stage),  // accumulator
        binding(5, vk::DescriptorType::STORAGE_IMAGE, 1, stage),  // framebuffer target
        binding(
            6,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            texture_count as u32,
            stage,
        ), // textures
        binding(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, stage), // skybox
        binding(8, vk::DescriptorType::STORAGE_BUFFER, 1, stage), // lights
        binding(9, vk::DescriptorType::STORAGE_BUFFER, 1, stage), // emissive triangles
        binding(10, vk::DescriptorType::STORAGE_BUFFER, 1, stage), // cdf
    ];
    create_descriptor_set_layout(device, &bindings)
}

pub(super) unsafe fn create_pool(
    device: &Device,
    set_count: usize,
    texture_count: usize,
) -> Result<vk::DescriptorPool> {
    let max_sets = set_count as u32;
    let pool_sizes = [
        pool_size(vk::DescriptorType::UNIFORM_BUFFER, max_sets),
        pool_size(vk::DescriptorType::STORAGE_BUFFER, 6 * max_sets),
        pool_size(vk::DescriptorType::STORAGE_IMAGE, 2 * max_sets),
        pool_size(
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            max_sets * (texture_count as u32 + 1),
        ),
    ];
    create_descriptor_pool(
        device,
        &pool_sizes,
        max_sets,
        vk::DescriptorPoolCreateFlags::empty(),
    )
}

/// Everything the path tracer descriptor sets bind to.
pub(super) struct SceneBindings<'a> {
    pub framebuffer_images: &'a [Image],
    pub uniform_buffer: &'a Buffer,
    pub ssbo: &'a Buffer,
    pub scene_sizes: &'a [u64; 6],
    pub accumulator_view: vk::ImageView,
    pub textures: &'a [Image],
    pub texture_sampler: vk::Sampler,
    pub skybox_view: vk::ImageView,
    pub skybox_sampler: vk::Sampler,
}

pub(super) unsafe fn create_sets(
    device: &Device,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    bindings: &SceneBindings,
) -> Result<Vec<vk::DescriptorSet>> {
    let sets = allocate_descriptor_sets(device, pool, layout, bindings.framebuffer_images.len())?;
    info!("Allocated Descriptor sets len {}", sets.len());

    // ssbo regions for bindings 1, 2, 3, 8, 9, 10 (bvh, materials,
    // triangles, lights, emissive triangles, cdf), packed contiguously
    let mut ssbo_infos = [vk::DescriptorBufferInfo::default(); 6];
    let mut offset = 0;
    for (info, &size) in ssbo_infos.iter_mut().zip(bindings.scene_sizes) {
        *info = buffer_info(bindings.ssbo.buffer, offset, size.max(4));
        offset += size;
    }
    let [bvh, materials, triangles, lights, emissive, cdf] = ssbo_infos.map(|i| [i]);

    let uniform = [buffer_info(
        bindings.uniform_buffer.buffer,
        0,
        std::mem::size_of::<CameraBufferObject>() as u64,
    )];
    let accumulator = [image_info(
        vk::Sampler::null(),
        bindings.accumulator_view,
        vk::ImageLayout::GENERAL,
    )];
    let texture_infos: Vec<_> = bindings
        .textures
        .iter()
        .map(|tex| {
            image_info(
                bindings.texture_sampler,
                tex.view,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
        })
        .collect();
    let skybox = [image_info(
        bindings.skybox_sampler,
        bindings.skybox_view,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )];

    for (set, framebuffer) in sets.iter().zip(bindings.framebuffer_images) {
        let target = [image_info(
            vk::Sampler::null(),
            framebuffer.view,
            vk::ImageLayout::GENERAL,
        )];

        let writes = [
            buffer_write(*set, 0, vk::DescriptorType::UNIFORM_BUFFER, &uniform),
            buffer_write(*set, 1, vk::DescriptorType::STORAGE_BUFFER, &bvh),
            buffer_write(*set, 2, vk::DescriptorType::STORAGE_BUFFER, &materials),
            buffer_write(*set, 3, vk::DescriptorType::STORAGE_BUFFER, &triangles),
            image_write(*set, 4, vk::DescriptorType::STORAGE_IMAGE, &accumulator),
            image_write(*set, 5, vk::DescriptorType::STORAGE_IMAGE, &target),
            image_write(
                *set,
                6,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &texture_infos,
            ),
            image_write(*set, 7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, &skybox),
            buffer_write(*set, 8, vk::DescriptorType::STORAGE_BUFFER, &lights),
            buffer_write(*set, 9, vk::DescriptorType::STORAGE_BUFFER, &emissive),
            buffer_write(*set, 10, vk::DescriptorType::STORAGE_BUFFER, &cdf),
        ];

        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }

    Ok(sets)
}

/// rebind resized target images
pub(super) unsafe fn update_target_bindings(
    device: &Device,
    sets: &[vk::DescriptorSet],
    framebuffer_images: &[Image],
    accumulator_view: vk::ImageView,
) {
    let accumulator = [image_info(
        vk::Sampler::null(),
        accumulator_view,
        vk::ImageLayout::GENERAL,
    )];
    for (set, framebuffer) in sets.iter().zip(framebuffer_images) {
        let target = [image_info(
            vk::Sampler::null(),
            framebuffer.view,
            vk::ImageLayout::GENERAL,
        )];
        let writes = [
            image_write(*set, 4, vk::DescriptorType::STORAGE_IMAGE, &accumulator),
            image_write(*set, 5, vk::DescriptorType::STORAGE_IMAGE, &target),
        ];
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }
}
