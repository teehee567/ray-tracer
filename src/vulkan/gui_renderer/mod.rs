mod drawing;
mod pipeline;
mod textures;

use std::collections::HashMap;

use anyhow::Result;
use egui::TextureId;
use glam::UVec2;
use vulkanalia::prelude::v1_0::*;

use crate::OFFSCREEN_FRAME_COUNT;
use crate::vulkan::context::VulkanContext;
use crate::vulkan::texture::GuiTexture;

use drawing::GuiFrameBuffers;
pub(crate) use drawing::{GuiDrawData, GuiVertex};
use pipeline::{
    create_gui_descriptor_pool, create_gui_descriptor_set_layout, create_gui_pipeline,
    create_gui_pipeline_layout, create_gui_sampler,
};

#[derive(Clone, Debug)]
pub(crate) struct GuiRenderer {
    pub(super) base_extent: UVec2,
    pub(super) render_extent: UVec2,
    pub(super) panel_width: u32,
    pub(super) last_generation: Option<u64>,
    pub(super) textures: HashMap<TextureId, GuiTexture>,
    pub(super) sampler: vk::Sampler,
    pub(super) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(super) descriptor_pool: vk::DescriptorPool,
    pub(super) pipeline_layout: vk::PipelineLayout,
    pub(super) pipeline: vk::Pipeline,
    pub(super) frames: Vec<GuiFrameBuffers>,
    pub(super) draw_data: Option<GuiDrawData>,
    pub(super) fallback: GuiTexture,
}

impl GuiRenderer {
    pub(crate) unsafe fn new(
        instance: &Instance,
        device: &Device,
        ctx: &VulkanContext,
        render_pass: vk::RenderPass,
        swap_extent: vk::Extent2D,
        base_extent: UVec2,
    ) -> Result<Self> {
        let clamped_width = base_extent.x.min(swap_extent.width);
        let clamped_height = base_extent.y.min(swap_extent.height);
        let render_extent = UVec2::new(clamped_width, clamped_height);

        let sampler = create_gui_sampler(device)?;
        let descriptor_set_layout = create_gui_descriptor_set_layout(device)?;
        let pipeline_layout = create_gui_pipeline_layout(device, descriptor_set_layout)?;
        let pipeline = create_gui_pipeline(device, pipeline_layout, render_pass)?;
        let descriptor_pool = create_gui_descriptor_pool(device)?;

        let mut frames = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
        for _ in 0..OFFSCREEN_FRAME_COUNT {
            frames.push(GuiFrameBuffers::default());
        }

        let mut renderer = Self {
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
            fallback: GuiTexture::default(),
        };

        let fallback = crate::vulkan::texture::create_texture_resource(
            instance,
            device,
            ctx.physical_device,
            renderer.descriptor_set_layout,
            renderer.descriptor_pool,
            renderer.sampler,
            [1, 1],
        )?;
        Self::upload_pixels(
            instance,
            device,
            ctx,
            &fallback,
            &[255u8, 255, 255, 255],
            [1, 1],
            None,
            true,
        )?;
        renderer.fallback = fallback;

        Ok(renderer)
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
        let available_width = swap_width.saturating_sub(self.panel_width);
        let width = self.base_extent.x.min(available_width);
        let height = self.base_extent.y.min(swap_height);
        self.render_extent = UVec2::new(width, height);
    }

    pub(crate) fn handle_resize(&mut self, width: u32, height: u32) {
        let max_panel_width = width.saturating_sub(self.base_extent.x);
        self.panel_width = self.panel_width.min(width).min(max_panel_width);
        self.update_render_extent(width, height);
        for buffers in &mut self.frames {
            buffers.uploaded_generation = None;
        }
    }

    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        for buffers in &mut self.frames {
            buffers.destroy(device);
        }

        for texture in self.textures.values() {
            if texture.view != vk::ImageView::null() {
                device.destroy_image_view(texture.view, None);
            }
            if texture.image != vk::Image::null() {
                device.destroy_image(texture.image, None);
            }
            if texture.memory != vk::DeviceMemory::null() {
                device.free_memory(texture.memory, None);
            }
        }
        self.textures.clear();

        if self.fallback.view != vk::ImageView::null() {
            device.destroy_image_view(self.fallback.view, None);
            self.fallback.view = vk::ImageView::null();
        }
        if self.fallback.image != vk::Image::null() {
            device.destroy_image(self.fallback.image, None);
            self.fallback.image = vk::Image::null();
        }
        if self.fallback.memory != vk::DeviceMemory::null() {
            device.free_memory(self.fallback.memory, None);
            self.fallback.memory = vk::DeviceMemory::null();
        }

        if self.descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
        }
        if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.descriptor_set_layout = vk::DescriptorSetLayout::null();
        }
        if self.pipeline != vk::Pipeline::null() {
            device.destroy_pipeline(self.pipeline, None);
            self.pipeline = vk::Pipeline::null();
        }
        if self.pipeline_layout != vk::PipelineLayout::null() {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.pipeline_layout = vk::PipelineLayout::null();
        }
        if self.sampler != vk::Sampler::null() {
            device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }
    }
}
