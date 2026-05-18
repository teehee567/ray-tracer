use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::context::VulkanContext;
use crate::vulkan::image::cmd_transition_image_layout;
use crate::vulkan::single_time::with_single_time;
use crate::vulkan::utils::create_buffer;

use super::GuiRenderer;
use super::gui_texture::{create_gui_texture, destroy_gui_texture};

impl GuiRenderer {
    pub(super) unsafe fn apply_textures(
        &mut self,
        instance: &Instance,
        device: &Device,
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
                if texture.size != size {
                    if let Some(mut texture) = self.textures.remove(id) {
                        destroy_gui_texture(device, self.descriptor_pool, &mut texture)?;
                    }
                }
            }

            if let Some(texture) = self.textures.get_mut(id) {
                Self::upload_pixels(instance, device, ctx, texture.image, &pixels, size, None, false)?;
                texture.size = size;
            } else {
                let mut texture = create_gui_texture(
                    instance,
                    device,
                    ctx.physical_device,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    self.sampler,
                    size,
                )?;
                Self::upload_pixels(instance, device, ctx, texture.image, &pixels, size, None, true)?;
                texture.size = size;
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
                let fits = ox.saturating_add(sub_size[0]) <= texture.size[0]
                    && oy.saturating_add(sub_size[1]) <= texture.size[1];
                if !fits {
                    continue;
                }
                Self::upload_pixels(
                    instance,
                    device,
                    ctx,
                    texture.image,
                    &pixels,
                    sub_size,
                    Some([ox, oy]),
                    false,
                )?;
            } else {
                let alloc_size = [
                    ox.saturating_add(sub_size[0]).max(1),
                    oy.saturating_add(sub_size[1]).max(1),
                ];
                let mut texture = create_gui_texture(
                    instance,
                    device,
                    ctx.physical_device,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    self.sampler,
                    alloc_size,
                )?;
                Self::upload_pixels(
                    instance,
                    device,
                    ctx,
                    texture.image,
                    &pixels,
                    sub_size,
                    Some([ox, oy]),
                    true,
                )?;
                texture.size = alloc_size;
                self.textures.insert(*id, texture);
            }
        }

        for id in &delta.free {
            if let Some(mut texture) = self.textures.remove(id) {
                destroy_gui_texture(device, self.descriptor_pool, &mut texture)?;
            }
        }

        Ok(())
    }

    pub(super) unsafe fn upload_pixels(
        instance: &Instance,
        device: &Device,
        ctx: &VulkanContext,
        image: vk::Image,
        pixels: &[u8],
        size: [u32; 2],
        offset: Option<[u32; 2]>,
        is_new: bool,
    ) -> Result<()> {
        let (staging_buffer, staging_memory) = create_buffer(
            instance,
            device,
            ctx.physical_device,
            pixels.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mapped = device.map_memory(
            staging_memory,
            0,
            pixels.len() as vk::DeviceSize,
            vk::MemoryMapFlags::empty(),
        )? as *mut u8;
        mapped.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
        device.unmap_memory(staging_memory);

        let old_layout = if is_new {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };

        let offset = offset.unwrap_or([0, 0]);

        with_single_time(device, ctx.command_pool, ctx.compute_queue, |cb| {
            cmd_transition_image_layout(
                device,
                cb,
                image,
                old_layout,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
            )?;

            let region = vk::BufferImageCopy::builder()
                .image_subresource(
                    vk::ImageSubresourceLayers::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_offset(vk::Offset3D {
                    x: offset[0] as i32,
                    y: offset[1] as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: size[0],
                    height: size[1],
                    depth: 1,
                });

            device.cmd_copy_buffer_to_image(
                cb,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            cmd_transition_image_layout(
                device,
                cb,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
            )?;
            Ok(())
        })?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_memory, None);

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
