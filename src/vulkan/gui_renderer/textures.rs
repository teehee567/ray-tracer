use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::context::VulkanContext;
use crate::vulkan::texture::{GuiTexture, create_texture_resource};
use crate::vulkan::utils::get_memory_type_index;

use super::GuiRenderer;

impl GuiRenderer {
    pub(super) unsafe fn apply_textures(
        &mut self,
        instance: &Instance,
        device: &Device,
        ctx: &VulkanContext,
        delta: &egui::TexturesDelta,
    ) -> Result<()> {
        // Pass 1: full replacements first (ensure correct size before partial updates)
        for (id, delta) in &delta.set {
            if delta.pos.is_some() {
                continue;
            }
            let (pixels, size) = Self::image_to_rgba(&delta.image);

            if let Some(texture) = self.textures.get(id) {
                if texture.size != size {
                    if let Some(texture) = self.textures.remove(id) {
                        if texture.view != vk::ImageView::null() {
                            device.destroy_image_view(texture.view, None);
                        }
                        if texture.image != vk::Image::null() {
                            device.destroy_image(texture.image, None);
                        }
                        if texture.memory != vk::DeviceMemory::null() {
                            device.free_memory(texture.memory, None);
                        }
                        if texture.descriptor_set != vk::DescriptorSet::null() {
                            device.free_descriptor_sets(
                                self.descriptor_pool,
                                &[texture.descriptor_set],
                            )?;
                        }
                    }
                }
            }

            if let Some(texture) = self.textures.get_mut(id) {
                Self::upload_pixels(instance, device, ctx, texture, &pixels, size, None, false)?;
                texture.size = size;
            } else {
                let mut texture = create_texture_resource(
                    instance,
                    device,
                    ctx.physical_device,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    self.sampler,
                    size,
                )?;
                Self::upload_pixels(instance, device, ctx, &texture, &pixels, size, None, true)?;
                texture.size = size;
                self.textures.insert(*id, texture);
            }
        }

        // Pass 2: partial updates (must fit current texture)
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
                    texture,
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
                let mut texture = create_texture_resource(
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
                    &texture,
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
            if let Some(texture) = self.textures.remove(id) {
                if texture.view != vk::ImageView::null() {
                    device.destroy_image_view(texture.view, None);
                }
                if texture.image != vk::Image::null() {
                    device.destroy_image(texture.image, None);
                }
                if texture.memory != vk::DeviceMemory::null() {
                    device.free_memory(texture.memory, None);
                }
                if texture.descriptor_set != vk::DescriptorSet::null() {
                    device.free_descriptor_sets(self.descriptor_pool, &[texture.descriptor_set])?;
                }
            }
        }

        Ok(())
    }

    pub(super) unsafe fn upload_pixels(
        instance: &Instance,
        device: &Device,
        ctx: &VulkanContext,
        texture: &GuiTexture,
        pixels: &[u8],
        size: [u32; 2],
        offset: Option<[u32; 2]>,
        is_new: bool,
    ) -> Result<()> {
        let staging_info = vk::BufferCreateInfo::builder()
            .size(pixels.len() as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = device.create_buffer(&staging_info, None)?;

        let requirements = device.get_buffer_memory_requirements(staging_buffer);
        let memory_type = get_memory_type_index(
            instance,
            ctx.physical_device,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            requirements,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let staging_memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(staging_buffer, staging_memory, 0)?;

        let ptr = device.map_memory(
            staging_memory,
            0,
            requirements.size,
            vk::MemoryMapFlags::empty(),
        )? as *mut u8;
        ptr.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
        device.unmap_memory(staging_memory);

        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(ctx.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device.begin_command_buffer(command_buffer, &begin_info)?;

        let old_layout = if is_new {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        Self::transition_image(
            device,
            command_buffer,
            texture.image,
            old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        let offset = offset.unwrap_or([0, 0]);
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
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
            })
            .build();

        device.cmd_copy_buffer_to_image(
            command_buffer,
            staging_buffer,
            texture.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );

        Self::transition_image(
            device,
            command_buffer,
            texture.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        device.end_command_buffer(command_buffer)?;
        let submit_info =
            vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));
        device.queue_submit(ctx.compute_queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(ctx.compute_queue)?;
        device.free_command_buffers(ctx.command_pool, &[command_buffer]);

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_memory, None);

        Ok(())
    }

    pub(super) unsafe fn create_gui_buffer(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, vk::DeviceSize)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None)?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let allocation_size = requirements.size.max(size);
        let memory_type = get_memory_type_index(
            instance,
            physical_device,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            requirements,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type);
        let memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(buffer, memory, 0)?;

        Ok((buffer, memory, allocation_size))
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

    pub(super) fn texture_for(&self, id: egui::TextureId) -> &GuiTexture {
        self.textures.get(&id).unwrap_or(&self.fallback)
    }

    pub(super) unsafe fn transition_image(
        device: &Device,
        command_buffer: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let (src_stage, src_access) = match old_layout {
            vk::ImageLayout::UNDEFINED => (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
            ),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::SHADER_READ,
            ),
            _ => (
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::empty(),
            ),
        };

        let (dst_stage, dst_access) = match new_layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            ),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::AccessFlags::SHADER_READ,
            ),
            _ => (
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::empty(),
            ),
        };

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .build();

        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );
    }
}
