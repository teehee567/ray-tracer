mod drawing;
mod gui_texture;
mod pipeline;
mod textures;

use std::collections::HashMap;

use anyhow::Result;
use egui::TextureId;
use glam::UVec2;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::constants::OFFSCREEN_FRAME_COUNT;
use crate::vulkan::core::context::VulkanContext;
use crate::vulkan::core::image::upload_pixels;

use drawing::GuiFrameBuffers;
use gui_texture::{GuiTexture, create_gui_texture, destroy_gui_texture};
use pipeline::{
    create_gui_descriptor_pool, create_gui_descriptor_set_layout, create_gui_pipeline,
    create_gui_pipeline_layout, create_gui_sampler,
};

/// Renders egui draw lists on top of the path traced image.
pub struct GuiRenderer {
    base_extent: UVec2,
    render_extent: UVec2,
    panel_width: u32,
    last_generation: Option<u64>,
    textures: HashMap<TextureId, GuiTexture>,
    sampler: vk::Sampler,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    frames: Vec<GuiFrameBuffers>,
    draw_data: Option<drawing::GuiDrawData>,
    fallback: GuiTexture,
}

impl GuiRenderer {
    pub(crate) unsafe fn new(
        ctx: &VulkanContext,
        render_pass: vk::RenderPass,
        swap_extent: vk::Extent2D,
        base_extent: UVec2,
    ) -> Result<Self> {
        let device = &ctx.device;
        let clamped_width = base_extent.x.min(swap_extent.width);
        let clamped_height = base_extent.y.min(swap_extent.height);
        let render_extent = UVec2::new(clamped_width, clamped_height);

        let sampler = create_gui_sampler(device)?;
        let descriptor_set_layout = create_gui_descriptor_set_layout(device)?;
        let pipeline_layout = create_gui_pipeline_layout(device, descriptor_set_layout)?;
        let pipeline = create_gui_pipeline(device, pipeline_layout, render_pass)?;
        let descriptor_pool = create_gui_descriptor_pool(device)?;

        let frames = vec![GuiFrameBuffers::default(); OFFSCREEN_FRAME_COUNT];

        let fallback =
            create_gui_texture(ctx, descriptor_set_layout, descriptor_pool, sampler, [1, 1])?;
        upload_pixels(
            ctx,
            fallback.image.image,
            &[255u8, 255, 255, 255],
            [1, 1],
            [0, 0],
            1,
            vk::ImageLayout::UNDEFINED,
        )?;

        Ok(Self {
            base_extent,
            render_extent,
            panel_width: 0,
            last_generation: None,
            textures: HashMap::new(),
            sampler,
            descriptor_set_layout,
            descriptor_pool,
            pipeline_layout,
            pipeline,
            frames,
            draw_data: None,
            fallback,
        })
    }

    pub(crate) fn render_extent(&self) -> UVec2 {
        self.render_extent
    }

    pub(crate) fn panel_width(&self) -> u32 {
        self.panel_width
    }

    pub(crate) fn has_draws(&self) -> bool {
        self.draw_data
            .as_ref()
            .map(|data| !data.vertices.is_empty() && !data.indices.is_empty())
            .unwrap_or(false)
    }

    pub(super) fn update_render_extent(&mut self, swap_width: u32, swap_height: u32) {
        // fill available render area
        let width = swap_width.saturating_sub(self.panel_width).max(1);
        let height = swap_height.max(1);
        self.render_extent = UVec2::new(width, height);
    }

    pub(crate) fn handle_resize(&mut self, width: u32, height: u32) {
        self.panel_width = self.panel_width.min(width.saturating_sub(1));
        self.update_render_extent(width, height);
        for buffers in &mut self.frames {
            buffers.uploaded_generation = None;
        }
    }

    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        for buffers in &mut self.frames {
            buffers.destroy(device);
        }

        for (_, mut texture) in self.textures.drain() {
            let _ = destroy_gui_texture(device, self.descriptor_pool, &mut texture);
        }

        let _ = destroy_gui_texture(device, self.descriptor_pool, &mut self.fallback);

        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_sampler(self.sampler, None);
    }
}
