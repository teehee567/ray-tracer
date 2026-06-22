
mod compositor;

use std::slice;

use glam::Mat4;
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::{
    accelerators::visualiser::AccelVis, scene::Scene, vulkan::core::{
        buffer::{Buffer, BufferOpts},
        context::VulkanContext,
        image::Image,
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
    pipeline: vk::Pipeline,
    framebuffer: vk::Framebuffer,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    index_count: u32,
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

        let heatmap_image = Self::create_image(ctx, render_extent)?;

        let render_pass = Self::create_render_pass(device, vk::Format::R32_SFLOAT)?;
        let framebuffer = Self::create_framebuffer(device, render_pass, &heatmap_image)?;

        let pipeline_layout = Self::create_pipeline_layout(device)?;
        let pipeline = Self::create_heatmap_pipeline(device, pipeline_layout, render_pass)?;

        let accel_vis = AccelVis::from_flat_bvh(&scene.components.bvh);

        let (vertices, indices) = accel_vis.build_geo();
        let index_count = indices.len() as u32;
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

        Ok(Self {
            heatmap_pass: render_pass,
            image: heatmap_image,
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count,
            framebuffer,
            pipeline_layout,
            compositor,
        })
    }

    unsafe fn create_image(ctx: &VulkanContext, extent: vk::Extent2D) -> Result<Image> {
        Image::new_2d(
            ctx,
            extent.width,
            extent.height,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
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
        let attachments = [image.view];
        let info = vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(image.width)
            .height(image.height)
            .layers(1);
        Ok(device.create_framebuffer(&info, None)?)
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

        self.image = Self::create_image(ctx, render_extent)?;
        self.framebuffer = Self::create_framebuffer(device, self.heatmap_pass, &self.image)?;
        self.compositor.update_composite_set(device, self.image.view);

        Ok(())
    }

    unsafe fn create_pipeline_layout(device: &Device) -> Result<vk::PipelineLayout> {
        let push_constant = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<Mat4>() as u32)
            .build();

        let info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(slice::from_ref(&push_constant));
        Ok(device.create_pipeline_layout(&info, None)?)
    }

    pub unsafe fn record_into(&self, device: &Device, cb: vk::CommandBuffer, view_proj: Mat4) {
        let (w, h) = (self.image.width, self.image.height);
        let clear = vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        };
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
            .render_area(area)
            .clear_values(std::slice::from_ref(&clear));
        device.cmd_begin_render_pass(cb, &begin, vk::SubpassContents::INLINE);

        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
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

        device.cmd_bind_vertex_buffers(cb, 0, &[self.vertex_buffer.buffer], &[0]);
        device.cmd_bind_index_buffer(cb, self.index_buffer.buffer, 0, vk::IndexType::UINT32);
        device.cmd_draw_indexed(cb, self.index_count, 1, 0, 0, 0);

        device.cmd_end_render_pass(cb);
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
        let blend_attachments = [Self::additive_blend_attachment()];

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

    unsafe fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass> {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(format)
            .samples(vk::SampleCountFlags::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let color_attachment_ref = vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachments = &[color_attachment_ref];
        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments);

        let dependency = vk::SubpassDependency::builder()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            // sampled by the reduce compute pass and the composite fragment shader
            .dst_stage_mask(
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        let attachments = &[color_attachment];
        let subpasses = &[subpass];
        let dependencies = &[dependency];
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        let render_pass = device.create_render_pass(&info, None)?;
        info!("Created heatmap render_pass: {:?}", render_pass);

        Ok(render_pass)
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.compositor.destroy(device);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_framebuffer(self.framebuffer, None);
        self.vertex_buffer.destroy(device);
        self.index_buffer.destroy(device);
        self.image.destroy(device);
        device.destroy_render_pass(self.heatmap_pass, None);
    }

    fn additive_blend_attachment() -> vk::PipelineColorBlendAttachmentState {
        //dst=ONE, op=ADD means each fragment does count += src. shader writes 1.0, so every covered fragment adds one. and with this we can buuild a heatmpa
        vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::R)
            .build()
    }
}
