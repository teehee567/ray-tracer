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
use crate::vulkan::core::image::{
    Image, ImageDesc, TextureDesc, cmd_transition_image_layout, create_texture,
};
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
    pub framebuffer_format: vk::Format,
    pub textures: Vec<Image>,
    pub texture_sampler: vk::Sampler,
    pub skybox: Image,
    pub skybox_sampler: vk::Sampler,
}

impl PathTracer {
    /// Swap in a newly compiled compute shader. Descriptor layout/sets are
    /// untouched (the shader's binding interface is fixed), so only the
    /// pipeline and its layout are recreated — create-before-destroy, so a
    /// failed build leaves the old pipeline running. Caller must ensure the
    /// device is idle.
    pub unsafe fn rebuild_pipeline(&mut self, device: &Device, shader_spv: &[u8]) -> Result<()> {
        let (pipeline_layout, pipeline) =
            create_compute_pipeline(device, self.descriptor_set_layout, shader_spv)?;
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        self.pipeline = pipeline;
        self.pipeline_layout = pipeline_layout;
        Ok(())
    }

    pub unsafe fn new(
        ctx: &VulkanContext,
        scene: &Scene,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        // make offscreen render targets
        let (framebuffer_images, accumulator) =
            Self::create_targets(ctx, extent.width, extent.height, format)?;

        // scene buffers
        let uniform_buffer = Buffer::new_host(
            ctx,
            size_of::<CameraBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;

        let scene_sizes = buffer_sizes(scene);
        // pad so the dummy descriptor ranges bound for empty regions
        let ssbo = Buffer::new_host(
            ctx,
            scene_sizes.iter().sum::<u64>() + 4,
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
                &TextureDesc {
                    width: texture_data.width,
                    height: texture_data.height,
                    ..Default::default()
                },
            )?);
        }

        if textures.is_empty() {
            let default_pixels = [255u8, 255, 255, 255];
            textures.push(create_texture(
                ctx,
                &default_pixels,
                &TextureDesc {
                    width: 1,
                    height: 1,
                    ..Default::default()
                },
            )?);
        }

        let skybox_data = &scene.components.skybox;
        let skybox = create_texture(
            ctx,
            &skybox_data.pixels,
            &TextureDesc {
                width: skybox_data.width,
                height: skybox_data.height,
                layer_count: 6,
                flags: vk::ImageCreateFlags::CUBE_COMPATIBLE,
                view_type: vk::ImageViewType::CUBE,
                ..Default::default()
            },
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
            &descriptors::SceneBindings {
                framebuffer_images: &framebuffer_images,
                uniform_buffer: &uniform_buffer,
                ssbo: &ssbo,
                scene_sizes: &scene_sizes,
                accumulator_view: accumulator.view,
                textures: &textures,
                texture_sampler,
                skybox_view: skybox.view,
                skybox_sampler,
            },
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
            framebuffer_format: format,
            textures,
            texture_sampler,
            skybox,
            skybox_sampler,
        })
    }

    /// make render target images
    unsafe fn create_targets(
        ctx: &VulkanContext,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<(Vec<Image>, Image)> {
        let mut framebuffer_images = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
        for _ in 0..OFFSCREEN_FRAME_COUNT {
            framebuffer_images.push(Image::new_2d(
                ctx,
                &ImageDesc {
                    width,
                    height,
                    format,
                    usage: vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                    ..Default::default()
                },
            )?);
        }

        let accumulator = Image::new_2d(
            ctx,
            &ImageDesc {
                width,
                height,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
        )?;

        with_single_time(&ctx.device, ctx.command_pool, ctx.compute_queue, |cb| {
            for image in framebuffer_images
                .iter()
                .chain(std::iter::once(&accumulator))
            {
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

        Ok((framebuffer_images, accumulator))
    }

    /// resize render target images
    pub unsafe fn handle_resize(
        &mut self,
        ctx: &VulkanContext,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let (w, h) = (extent.width.max(1), extent.height.max(1));
        if w == self.accumulator.width && h == self.accumulator.height {
            return Ok(());
        }

        let device = &ctx.device;
        for image in &mut self.framebuffer_images {
            image.destroy(device);
        }
        self.accumulator.destroy(device);

        let (framebuffer_images, accumulator) =
            Self::create_targets(ctx, w, h, self.framebuffer_format)?;
        self.framebuffer_images = framebuffer_images;
        self.accumulator = accumulator;

        descriptors::update_target_bindings(
            device,
            &self.descriptor_sets,
            &self.framebuffer_images,
            self.accumulator.view,
        );
        Ok(())
    }

    /// current render target size
    pub fn render_size(&self) -> UVec2 {
        UVec2::new(self.accumulator.width, self.accumulator.height)
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
        query_pool: vk::QueryPool,
        path_trace: bool,
    ) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::builder();

        device.begin_command_buffer(command_buffer, &info)?;

        // for timing how long frame stays on gpu
        let first_query = frame_index as u32 * 2;
        device.cmd_reset_query_pool(command_buffer, query_pool, first_query, 2);
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            query_pool,
            first_query,
        );

        if path_trace && render_extent.x > 0 && render_extent.y > 0 {
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

        // end time
        device.cmd_write_timestamp(
            command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            query_pool,
            first_query + 1,
        );

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
