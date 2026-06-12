use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::core::context::VulkanContext;
use crate::vulkan::core::image::upload_pixels;

use super::GuiRenderer;
use super::gui_texture::{create_gui_texture, destroy_gui_texture};

impl GuiRenderer {
    pub(super) unsafe fn apply_textures(
        &mut self,
        ctx: &VulkanContext,
        delta: &egui::TexturesDelta,
    ) -> Result<()> {
        // pass 1 full replacements
        for (id, delta) in &delta.set {
            if delta.pos.is_some() {
                continue;
            }
            let (pixels, size) = Self::image_to_rgba(&delta.image);

            if let Some(texture) = self.textures.get(id) {
                if texture.size() != size {
                    if let Some(mut texture) = self.textures.remove(id) {
                        destroy_gui_texture(&ctx.device, self.descriptor_pool, &mut texture)?;
                    }
                }
            }

            if let Some(texture) = self.textures.get_mut(id) {
                upload_pixels(
                    ctx,
                    texture.image.image,
                    &pixels,
                    size,
                    [0, 0],
                    1,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?;
            } else {
                let texture = create_gui_texture(
                    ctx,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    self.sampler,
                    size,
                )?;
                upload_pixels(
                    ctx,
                    texture.image.image,
                    &pixels,
                    size,
                    [0, 0],
                    1,
                    vk::ImageLayout::UNDEFINED,
                )?;
                self.textures.insert(*id, texture);
            }
        }

        // pass 2 partial updates
        for (id, delta) in &delta.set {
            let Some([ox, oy]) = delta.pos.map(|[x, y]| [x as u32, y as u32]) else {
                continue;
            };
            let (pixels, sub_size) = Self::image_to_rgba(&delta.image);

            if let Some(texture) = self.textures.get_mut(id) {
                let size = texture.size();
                let fits = ox.saturating_add(sub_size[0]) <= size[0]
                    && oy.saturating_add(sub_size[1]) <= size[1];
                if !fits {
                    continue;
                }
                upload_pixels(
                    ctx,
                    texture.image.image,
                    &pixels,
                    sub_size,
                    [ox, oy],
                    1,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                )?;
            } else {
                let alloc_size = [
                    ox.saturating_add(sub_size[0]).max(1),
                    oy.saturating_add(sub_size[1]).max(1),
                ];
                let texture = create_gui_texture(
                    ctx,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    self.sampler,
                    alloc_size,
                )?;
                upload_pixels(
                    ctx,
                    texture.image.image,
                    &pixels,
                    sub_size,
                    [ox, oy],
                    1,
                    vk::ImageLayout::UNDEFINED,
                )?;
                self.textures.insert(*id, texture);
            }
        }

        for id in &delta.free {
            if let Some(mut texture) = self.textures.remove(id) {
                destroy_gui_texture(&ctx.device, self.descriptor_pool, &mut texture)?;
            }
        }

        Ok(())
    }

    pub(super) fn image_to_rgba(image: &egui::epaint::ImageData) -> (Vec<u8>, [u32; 2]) {
        match image {
            egui::epaint::ImageData::Color(color) => {
                let mut pixels = Vec::with_capacity(color.pixels.len() * 4);
                for px in &color.pixels {
                    let [r, g, b, a] = px.to_array();
                    pixels.extend_from_slice(&[r, g, b, a]);
                }
                (pixels, [color.size[0] as u32, color.size[1] as u32])
            }
        }
    }

    pub(super) fn texture_for(&self, id: egui::TextureId) -> &super::gui_texture::GuiTexture {
        self.textures.get(&id).unwrap_or(&self.fallback)
    }
}
