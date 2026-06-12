mod descriptors;

use anyhow::Result;
use glam::UVec2;
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::scene::Scene;
use crate::types::CameraBufferObject;
use crate::vulkan::constants::{OFFSCREEN_FRAME_COUNT, TILE_SIZE};
use crate::vulkan::core::buffer::Buffer;
use crate::vulkan::core::commands::with_single_time;
use crate::vulkan::core::context::VulkanContext;
use crate::vulkan::core::image::{Image, cmd_transition_image_layout, create_texture};
use crate::vulkan::core::pipeline::create_compute_pipeline;
use crate::vulkan::core::sampler::{SamplerDesc, create_sampler};

/// The compute-based path tracer: pipeline, descriptors and all GPU
/// resources it reads (scene buffers, textures, skybox) and writes
/// (accumulator, per-frame offscreen targets).
pub struct PathTracer {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub uniform_buffer: Buffer,
    pub ssbo: Buffer,
    pub accumulator: Image,
    pub framebuffer_images: Vec<Image>,
    pub textures: Vec<Image>,
    pub texture_sampler: vk::Sampler,
    pub skybox: Image,
    pub skybox_sampler: vk::Sampler,
}

impl PathTracer {
    pub unsafe fn new(
        ctx: &VulkanContext,
        scene: &Scene,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        // offscreen render targets, one per in-flight frame
        let mut framebuffer_images = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
        for _ in 0..OFFSCREEN_FRAME_COUNT {
            framebuffer_images.push(Image::new_2d(
                ctx,
                extent.width,
                extent.height,
                format,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                1,
                vk::ImageCreateFlags::empty(),
                vk::ImageViewType::_2D,
            )?);
        }

        let accumulator = Image::new_2d(
            ctx,
            extent.width,
            extent.height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            vk::ImageCreateFlags::empty(),
            vk::ImageViewType::_2D,
        )?;

        with_single_time(&ctx.device, ctx.command_pool, ctx.compute_queue, |cb| {
            for image in framebuffer_images.iter().chain(std::iter::once(&accumulator)) {
                cmd_transition_image_layout(
                    &ctx.device,
                    cb,
                    image.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    1,
                )?;
            }
            Ok(())
        })?;

        // scene buffers
        let uniform_buffer = Buffer::new_host(
            ctx,
            size_of::<CameraBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        let scene_sizes = buffer_sizes(scene);
        let ssbo = Buffer::new_host(
            ctx,
            scene_sizes.iter().sum(),
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?;

        // scene textures + skybox
        let texture_sampler = create_sampler(
            &ctx.device,
            &SamplerDesc {
                filter: vk::Filter::LINEAR,
                address_mode: vk::SamplerAddressMode::REPEAT,
                max_anisotropy: Some(16.0),
            },
        )?;
        let skybox_sampler = create_sampler(
            &ctx.device,
            &SamplerDesc {
                filter: vk::Filter::LINEAR,
                address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                max_anisotropy: Some(16.0),
            },
        )?;

        let mut textures = Vec::new();
        for texture_data in &scene.components.textures {
            textures.push(create_texture(
                ctx,
                &texture_data.pixels,
                texture_data.width,
                texture_data.height,
                vk::Format::R8G8B8A8_SRGB,
                1,
                vk::ImageCreateFlags::empty(),
                vk::ImageViewType::_2D,
            )?);
        }

        if textures.is_empty() {
            let default_pixels = [255u8, 255, 255, 255];
            textures.push(create_texture(
                ctx,
                &default_pixels,
                1,
                1,
                vk::Format::R8G8B8A8_SRGB,
                1,
                vk::ImageCreateFlags::empty(),
                vk::ImageViewType::_2D,
            )?);
        }

        let skybox_data = &scene.components.skybox;
        let skybox = create_texture(
            ctx,
            &skybox_data.pixels,
            skybox_data.width,
            skybox_data.height,
            vk::Format::R8G8B8A8_SRGB,
            6,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
            vk::ImageViewType::CUBE,
        )?;

        // pipeline + descriptors
        let descriptor_set_layout = descriptors::create_layout(&ctx.device, textures.len())?;
        let (pipeline_layout, pipeline) = create_compute_pipeline(
            &ctx.device,
            descriptor_set_layout,
            include_bytes!("../../shaders/main.comp.spv"),
        )?;
        let descriptor_pool =
            descriptors::create_pool(&ctx.device, framebuffer_images.len(), textures.len())?;
        let descriptor_sets = descriptors::create_sets(
            &ctx.device,
            descriptor_set_layout,
            descriptor_pool,
            &framebuffer_images,
            &uniform_buffer,
            &ssbo,
            &scene_sizes,
            accumulator.view,
            &textures,
            texture_sampler,
            skybox.view,
            skybox_sampler,
        )?;

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffer,
            ssbo,
            accumulator,
            framebuffer_images,
            textures,
            texture_sampler,
            skybox,
            skybox_sampler,
        })
    }

    pub unsafe fn upload_scene(&self, device: &Device, scene: &Scene) -> Result<()> {
        let sizes = buffer_sizes(scene);
        let total_size: u64 = sizes.iter().sum();

        self.ssbo
            .write_with(device, total_size, |ptr| scene.write_buffers(ptr))?;

        info!(
            "uploaded scene, sizes: bvh({}), mat({}), tri({}), lights({}), emissive_tris({}), cdf({}), total {}",
            sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[5], total_size
        );

        Ok(())
    }

    pub unsafe fn record_dispatch(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        render_extent: UVec2,
    ) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::builder();

        device.begin_command_buffer(command_buffer, &info)?;

        if render_extent.x > 0 && render_extent.y > 0 {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[frame_index]],
                &[],
            );

            let groups_x = render_extent.x.div_ceil(TILE_SIZE);
            let groups_y = render_extent.y.div_ceil(TILE_SIZE);
            device.cmd_dispatch(command_buffer, groups_x.max(1), groups_y.max(1), 1);
        }

        device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);

        self.uniform_buffer.destroy(device);
        self.ssbo.destroy(device);
        self.accumulator.destroy(device);
        for image in &mut self.framebuffer_images {
            image.destroy(device);
        }
        for texture in &mut self.textures {
            texture.destroy(device);
        }
        self.skybox.destroy(device);
        device.destroy_sampler(self.texture_sampler, None);
        device.destroy_sampler(self.skybox_sampler, None);
    }
}

fn buffer_sizes(scene: &Scene) -> [u64; 6] {
    let (bvh, mat, tri, lights, emissive, cdf) = scene.get_buffer_sizes();
    [
        bvh as u64,
        mat as u64,
        tri as u64,
        lights as u64,
        emissive as u64,
        cdf as u64,
    ]
}
