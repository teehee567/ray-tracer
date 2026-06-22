use glam::UVec2;
use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::core::{
    context::VulkanContext, image::Image, pipeline::create_graphics_pipeline,
};
use anyhow::Result;

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
            swapchain_pass: render_pass,
            image: debug_image,
        })
    }

    unsafe fn create_debug_pipeline(
        device: &Device,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
    ) -> Result<vk::Pipeline> {
        


        let pipeline = create_graphics_pipeline(
            device,
            super::core::pipeline::GraphicsPipelineConfig {
                shaders: &shaders,
                vertex_bindings: (),
                vertex_attributes: (),
                blend_attachments: (),
                dynamic_states: (),
                layout,
                render_pass,
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
        device.destroy_render_pass(self.render_pass, None);
    }
}
