
mod compositor;

use std::slice;

use glam::Mat4;
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::{
    accelerators::visualiser::AccelVis, scene::Scene, vulkan::core::{
        buffer::{Buffer, BufferOpts},
        context::VulkanContext,
        image::{cmd_image_barrier, image_barrier, subresource_range, Image},
        descriptors::{
            allocate_descriptor_sets, binding, create_descriptor_pool, create_descriptor_set_layout,
            image_info, image_write, pool_size,
        },
        pipeline::{create_graphics_pipeline, create_shader_module},
    }
};
use anyhow::Result;

use compositor::Compositer;

use super::core::pipeline::GraphicsPipelineConfig;

#[repr(C)]
struct HeatmapVertex {
    pos: [f32; 3],
}

pub struct HeatmapRenderer {
    pub heatmap_pass: vk::RenderPass,
    pub image: Image,
    image_layout: vk::ImageLayout,
    accum_format: vk::Format,
    accum_set_layout: vk::DescriptorSetLayout,
    accum_pool: vk::DescriptorPool,
    accum_set: vk::DescriptorSet,
    pipeline: vk::Pipeline,
    framebuffer: vk::Framebuffer,
    vertex_buffer: Buffer,
    index_buffer: Buffer,

    depth_offsets: Vec<u32>,
    max_depth: u32,
    pipeline_layout: vk::PipelineLayout,
    compositor: Compositer,
}

impl HeatmapRenderer {
    pub(crate) unsafe fn new(
        ctx: &VulkanContext,
        swapchain_pass: vk::RenderPass,
        render_extent: vk::Extent2D,
        scene: &Scene,
    ) -> Result<Self> {
        let device = &ctx.device;

        let accum_format = Self::select_accum_format(ctx)?;
        let heatmap_image = Self::create_image(ctx, render_extent, accum_format)?;

        let render_pass = Self::create_render_pass(device)?;
        let framebuffer = Self::create_framebuffer(device, render_pass, &heatmap_image)?;

        let accum_set_layout = Self::create_accum_set_layout(device)?;
        let accum_pool = create_descriptor_pool(
            device,
            &[pool_size(vk::DescriptorType::STORAGE_IMAGE, 1)],
            1,
            vk::DescriptorPoolCreateFlags::empty(),
        )?;
        let accum_set = allocate_descriptor_sets(device, accum_pool, accum_set_layout, 1)?[0];

        let pipeline_layout = Self::create_pipeline_layout(device, accum_set_layout)?;
        let pipeline = Self::create_heatmap_pipeline(device, pipeline_layout, render_pass)?;

        let accel_vis = AccelVis::from_flat_bvh(&scene.components.bvh);

        let (vertices, indices, depth_offsets, max_depth) = accel_vis.build_geo_layered();
        let vertex_buffer = Buffer::from_slice(
            ctx,
            &vertices,
            BufferOpts {
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                ..Default::default()
            },
        )?;
        let index_buffer = Buffer::from_slice(
            ctx,
            &indices,
            BufferOpts {
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                ..Default::default()
            },
        )?;

        let compositor = Compositer::new(ctx, swapchain_pass, heatmap_image.view)?;
        Self::update_accum_set(device, accum_set, heatmap_image.view);

        Ok(Self {
            heatmap_pass: render_pass,
            image: heatmap_image,
            image_layout: vk::ImageLayout::UNDEFINED,
            accum_format,
            accum_set_layout,
            accum_pool,
            accum_set,
            pipeline,
            vertex_buffer,
            index_buffer,
            depth_offsets,
            max_depth,
            framebuffer,
            pipeline_layout,
            compositor,
        })
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    unsafe fn select_accum_format(ctx: &VulkanContext) -> Result<vk::Format> {
        let format = vk::Format::R32_UINT;
        let required = vk::FormatFeatureFlags::STORAGE_IMAGE
            | vk::FormatFeatureFlags::STORAGE_IMAGE_ATOMIC
            | vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::TRANSFER_DST;
        let features = ctx
            .instance
            .get_physical_device_format_properties(ctx.physical_device, format)
            .optimal_tiling_features;
        if !features.contains(required) {
            anyhow::bail!(
                "heatmap accumulation format {:?} is missing {:?} from {:?}",
                format,
                required - features,
                features
            );
        }
        Ok(format)
    }

    unsafe fn create_image(
        ctx: &VulkanContext,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Image> {
        Image::new_2d(
            ctx,
            extent.width,
            extent.height,
            format,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            vk::ImageCreateFlags::empty(),
            vk::ImageViewType::_2D,
        )
    }

    unsafe fn create_framebuffer(
        device: &Device,
        render_pass: vk::RenderPass,
        image: &Image,
    ) -> Result<vk::Framebuffer> {
        let attachments: [vk::ImageView; 0] = [];
        let info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(image.width)
            .height(image.height)
            .layers(1);
        Ok(device.create_framebuffer(&info, None)?)
    }

    unsafe fn create_accum_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
        let bindings = [binding(
            0,
            vk::DescriptorType::STORAGE_IMAGE,
            1,
            vk::ShaderStageFlags::FRAGMENT,
        )];
        create_descriptor_set_layout(device, &bindings)
    }

    unsafe fn update_accum_set(
        device: &Device,
        accum_set: vk::DescriptorSet,
        image_view: vk::ImageView,
    ) {
        let image = [image_info(
            vk::Sampler::null(),
            image_view,
            vk::ImageLayout::GENERAL,
        )];
        let writes = [image_write(
            accum_set,
            0,
            vk::DescriptorType::STORAGE_IMAGE,
            &image,
        )];
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }

    pub unsafe fn record_reduce(&self, device: &Device, cb: vk::CommandBuffer) {
        self.compositor
            .record_reduce(device, cb, self.image.width, self.image.height);
    }

    pub unsafe fn record_composite(
        &self,
        device: &Device,
        cb: vk::CommandBuffer,
        sub_region: vk::Rect2D,
    ) {
        self.compositor.record_into(device, cb, sub_region);
    }

    pub unsafe fn handle_resize(
        &mut self,
        ctx: &VulkanContext,
        render_extent: vk::Extent2D,
    ) -> Result<()> {
        let device = &ctx.device;
        let (w, h) = (render_extent.width.max(1), render_extent.height.max(1));
        if w == self.image.width && h == self.image.height {
            return Ok(());
        }

        device.device_wait_idle()?;
        device.destroy_framebuffer(self.framebuffer, None);
        self.image.destroy(device);

        self.image = Self::create_image(ctx, render_extent, self.accum_format)?;
        self.image_layout = vk::ImageLayout::UNDEFINED;
        self.framebuffer = Self::create_framebuffer(device, self.heatmap_pass, &self.image)?;
        Self::update_accum_set(device, self.accum_set, self.image.view);
        self.compositor.update_composite_set(device, self.image.view);

        Ok(())
    }

    unsafe fn create_pipeline_layout(
        device: &Device,
        set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout> {
        let push_constant = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<Mat4>() as u32)
            .build();

        let set_layouts = [set_layout];
        let info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(slice::from_ref(&push_constant));
        Ok(device.create_pipeline_layout(&info, None)?)
    }

    pub unsafe fn record_into(
        &mut self,
        device: &Device,
        cb: vk::CommandBuffer,
        view_proj: Mat4,
        low: u32,
        high: u32,
    ) {
        let (w, h) = (self.image.width, self.image.height);
        let (src_access, src_stage) = if self.image_layout == vk::ImageLayout::UNDEFINED {
            (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE)
        } else {
            (
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            )
        };
        cmd_image_barrier(
            device,
            cb,
            image_barrier(
                self.image.image,
                self.image_layout,
                vk::ImageLayout::GENERAL,
                src_access,
                vk::AccessFlags::TRANSFER_WRITE,
                1,
            ),
            src_stage,
            vk::PipelineStageFlags::TRANSFER,
        );

        let clear = vk::ClearColorValue { uint32: [0; 4] };
        device.cmd_clear_color_image(
            cb,
            self.image.image,
            vk::ImageLayout::GENERAL,
            &clear,
            &[subresource_range(1)],
        );
        cmd_image_barrier(
            device,
            cb,
            image_barrier(
                self.image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                1,
            ),
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
        self.image_layout = vk::ImageLayout::GENERAL;

        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: w,
                height: h,
            },
        };
        let begin = vk::RenderPassBeginInfo::builder()
            .render_pass(self.heatmap_pass)
            .framebuffer(self.framebuffer)
            .render_area(area);
        device.cmd_begin_render_pass(cb, &begin, vk::SubpassContents::INLINE);

        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[self.accum_set],
            &[],
        );
        let vp = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: w as f32,
            height: h as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }; // positive height
        device.cmd_set_viewport(cb, 0, &[vp]);
        device.cmd_set_scissor(cb, 0, &[area]);

        let bytes =
            std::slice::from_raw_parts(&view_proj as *const Mat4 as *const u8, size_of::<Mat4>());
        device.cmd_push_constants(
            cb,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            bytes,
        );

        let low = low.min(self.max_depth);
        let high = high.min(self.max_depth).max(low);
        let first = self.depth_offsets[low as usize];
        let count = self.depth_offsets[high as usize + 1] - first;

        if count > 0 {
            device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer.buffer], &[0]);
            device.cmd_bind_index_buffer(cb, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cb, count, 1, first, 0, 0);
        }

        device.cmd_end_render_pass(cb);

        cmd_image_barrier(
            device,
            cb,
            image_barrier(
                self.image.image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::AccessFlags::SHADER_READ,
                1,
            ),
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
    }

    unsafe fn create_heatmap_pipeline(
        device: &Device,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
    ) -> Result<vk::Pipeline> {
        let vert = create_shader_module(device, include_bytes!("../../shaders/heatmap.vert.spv"))?;
        let frag = create_shader_module(device, include_bytes!("../../shaders/heatmap.frag.spv"))?;

        let shaders = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(b"main\0")
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(b"main\0")
                .build(),
        ];

        let vertex_bindings = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<HeatmapVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];
        let vertex_attributes = [vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build()];

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let blend_attachments = [];

        let pipeline = create_graphics_pipeline(
            device,
            GraphicsPipelineConfig {
                shaders: &shaders,
                vertex_bindings: &vertex_bindings,
                vertex_attributes: &vertex_attributes,
                blend_attachments: &blend_attachments,
                dynamic_states: &dynamic_states,
                layout,
                render_pass,
                subpass: 0,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                cull_mode: vk::CullModeFlags::FRONT,
            },
        )?;

        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);

        Ok(pipeline)
    }

    unsafe fn create_render_pass(device: &Device) -> Result<vk::RenderPass> {
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS);

        let attachments: [vk::AttachmentDescription; 0] = [];
        let dependencies: [vk::SubpassDependency; 0] = [];
        let subpasses = &[subpass];
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(subpasses)
            .dependencies(&dependencies);

        let render_pass = device.create_render_pass(&info, None)?;
        info!("Created heatmap render_pass: {:?}", render_pass);

        Ok(render_pass)
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.compositor.destroy(device);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_framebuffer(self.framebuffer, None);
        device.destroy_descriptor_pool(self.accum_pool, None);
        device.destroy_descriptor_set_layout(self.accum_set_layout, None);
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
        self.image.destroy(device);
        device.destroy_render_pass(self.heatmap_pass, None);
    }
}
