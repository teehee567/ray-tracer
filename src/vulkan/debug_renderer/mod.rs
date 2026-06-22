use glam::UVec2;
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::core::{
    context::VulkanContext, image::Image, pipeline::{create_graphics_pipeline, create_shader_module},
};
use anyhow::Result;


#[repr(C)]
struct DebugVertex {
    pos: [f32; 3],
}

pub struct DebugRenderer {
    pub debug_pass: vk::RenderPass,
    pub swapchain_pass: vk::RenderPass,
    pub image: Image,
}

impl DebugRenderer {
    pub(crate) unsafe fn new(
        ctx: &VulkanContext,
        swapchain_pass: vk::RenderPass,
        extent: vk::Extent2D,
        base_extent: UVec2,
    ) -> Result<Self> {
        let device = &ctx.device;

        let debug_image = Image::new_2d(
            ctx,
            extent.width,
            extent.height,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            1,
            vk::ImageCreateFlags::empty(),
            vk::ImageViewType::_2D,
        )?;

        let render_pass = Self::create_render_pass(device, vk::Format::R32_SFLOAT)?;

        Ok(Self {
            debug_pass: render_pass,
            swapchain_pass,
            image: debug_image,
        })
    }

    unsafe fn create_debug_pipeline(
        &self,
        device: &Device,
        layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline> {
        let vert = create_shader_module(device, include_bytes!("../../shaders/debug.vert.spv"))?;
        let frag = create_shader_module(device, include_bytes!("../../shaders/debug.frag.spv"))?;


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
            .stride(size_of::<DebugVertex>() as u32)
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
            super::core::pipeline::GraphicsPipelineConfig {
                shaders: &shaders,
                vertex_bindings: &vertex_bindings,
                vertex_attributes: &vertex_attributes,
                blend_attachments: &blend_attachments,
                dynamic_states: &dynamic_states,
                layout,
                render_pass: self.debug_pass,
                subpass: 0,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                cull_mode: vk::CullModeFlags::FRONT,
            },
        )?;

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
            .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            // load reads and writes
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        let attachments = &[color_attachment];
        let subpasses = &[subpass];
        let dependencies = &[dependency];
        let info = vk::RenderPassCreateInfo::builder()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        let render_pass = device.create_render_pass(&info, None)?;
        info!("Created debug render_pass: {:?}", render_pass);

        Ok(render_pass)
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.image.destroy(device);
        device.destroy_render_pass(self.debug_pass, None);
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
