use std::mem::size_of;

use anyhow::{Result, anyhow};
use egui::epaint::{ClippedPrimitive, Primitive};
use egui::TextureId;
use glam::UVec2;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::context::VulkanContext;

use super::GuiRenderer;

#[derive(Clone, Debug)]
pub(super) struct GuiFrameBuffers {
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub vertex_capacity: vk::DeviceSize,
    pub index_buffer: vk::Buffer,
    pub index_memory: vk::DeviceMemory,
    pub index_capacity: vk::DeviceSize,
    pub uploaded_generation: Option<u64>,
}

impl Default for GuiFrameBuffers {
    fn default() -> Self {
        Self {
            vertex_buffer: vk::Buffer::null(),
            vertex_memory: vk::DeviceMemory::null(),
            vertex_capacity: 0,
            index_buffer: vk::Buffer::null(),
            index_memory: vk::DeviceMemory::null(),
            index_capacity: 0,
            uploaded_generation: None,
        }
    }
}

impl GuiFrameBuffers {
    pub(super) unsafe fn destroy(&mut self, device: &Device) {
        if self.vertex_buffer != vk::Buffer::null() {
            device.destroy_buffer(self.vertex_buffer, None);
            self.vertex_buffer = vk::Buffer::null();
        }
        if self.vertex_memory != vk::DeviceMemory::null() {
            device.free_memory(self.vertex_memory, None);
            self.vertex_memory = vk::DeviceMemory::null();
        }
        if self.index_buffer != vk::Buffer::null() {
            device.destroy_buffer(self.index_buffer, None);
            self.index_buffer = vk::Buffer::null();
        }
        if self.index_memory != vk::DeviceMemory::null() {
            device.free_memory(self.index_memory, None);
            self.index_memory = vk::DeviceMemory::null();
        }
        self.vertex_capacity = 0;
        self.index_capacity = 0;
        self.uploaded_generation = None;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GuiDrawData {
    pub generation: u64,
    pub vertices: Vec<GuiVertex>,
    pub indices: Vec<u32>,
    pub draws: Vec<GuiDraw>,
    pub panel_width: u32,
    pub panel_height: u32,
    pub pixels_per_point: f32,
}

#[derive(Clone, Debug)]
pub(super) struct GuiDraw {
    pub clip_rect: [f32; 4],
    pub texture: TextureId,
    pub index_count: u32,
    pub index_offset: u32,
    pub vertex_offset: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GuiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl GuiRenderer {
    pub(crate) unsafe fn update(
        &mut self,
        instance: &Instance,
        device: &Device,
        ctx: &VulkanContext,
        swap_extent: vk::Extent2D,
        frame: &crate::gui::GuiFrame,
    ) -> Result<()> {
        if self.last_generation == Some(frame.generation) {
            return Ok(());
        }

        let swap_width = swap_extent.width;
        let swap_height = swap_extent.height;
        let max_panel_width = swap_width.saturating_sub(self.base_extent.x);
        self.panel_width = frame.panel_width.min(swap_width).min(max_panel_width);
        self.update_render_extent(swap_width, swap_height);

        self.apply_textures(instance, device, ctx, &frame.textures_delta)?;
        self.draw_data = self.build_draw_data(
            &frame.clipped_primitives,
            frame.pixels_per_point,
            frame.panel_width,
            frame.panel_height,
            swap_width,
            swap_height,
            frame.generation,
        );

        self.last_generation = Some(frame.generation);
        for buffers in &mut self.frames {
            buffers.uploaded_generation = None;
        }

        Ok(())
    }

    pub(crate) unsafe fn prepare_frame(
        &mut self,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        frame_index: usize,
    ) -> Result<()> {
        let buffers = self
            .frames
            .get_mut(frame_index)
            .ok_or_else(|| anyhow!("invalid frame index"))?;

        let Some(draw_data) = self.draw_data.as_ref() else {
            buffers.uploaded_generation = None;
            return Ok(());
        };

        if buffers.uploaded_generation == Some(draw_data.generation) {
            return Ok(());
        }

        if draw_data.vertices.is_empty() || draw_data.indices.is_empty() {
            buffers.uploaded_generation = Some(draw_data.generation);
            return Ok(());
        }

        let vertex_size = (draw_data.vertices.len() * size_of::<GuiVertex>()) as vk::DeviceSize;
        if buffers.vertex_buffer == vk::Buffer::null() || buffers.vertex_capacity < vertex_size {
            if buffers.vertex_buffer != vk::Buffer::null() {
                device.destroy_buffer(buffers.vertex_buffer, None);
                buffers.vertex_buffer = vk::Buffer::null();
            }
            if buffers.vertex_memory != vk::DeviceMemory::null() {
                device.free_memory(buffers.vertex_memory, None);
                buffers.vertex_memory = vk::DeviceMemory::null();
            }
            let (buffer, memory, capacity) = Self::create_gui_buffer(
                instance,
                device,
                physical_device,
                vertex_size.max(1),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;
            buffers.vertex_buffer = buffer;
            buffers.vertex_memory = memory;
            buffers.vertex_capacity = capacity;
        }

        let ptr = device.map_memory(
            buffers.vertex_memory,
            0,
            vertex_size,
            vk::MemoryMapFlags::empty(),
        )? as *mut GuiVertex;
        ptr.copy_from_nonoverlapping(draw_data.vertices.as_ptr(), draw_data.vertices.len());
        device.unmap_memory(buffers.vertex_memory);

        let index_size = (draw_data.indices.len() * size_of::<u32>()) as vk::DeviceSize;
        if buffers.index_buffer == vk::Buffer::null() || buffers.index_capacity < index_size {
            if buffers.index_buffer != vk::Buffer::null() {
                device.destroy_buffer(buffers.index_buffer, None);
                buffers.index_buffer = vk::Buffer::null();
            }
            if buffers.index_memory != vk::DeviceMemory::null() {
                device.free_memory(buffers.index_memory, None);
                buffers.index_memory = vk::DeviceMemory::null();
            }

            let (buffer, memory, capacity) = Self::create_gui_buffer(
                instance,
                device,
                physical_device,
                index_size.max(1),
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;
            buffers.index_buffer = buffer;
            buffers.index_memory = memory;
            buffers.index_capacity = capacity;
        }

        let ptr = device.map_memory(
            buffers.index_memory,
            0,
            index_size,
            vk::MemoryMapFlags::empty(),
        )? as *mut u32;
        ptr.copy_from_nonoverlapping(draw_data.indices.as_ptr(), draw_data.indices.len());
        device.unmap_memory(buffers.index_memory);

        buffers.uploaded_generation = Some(draw_data.generation);
        Ok(())
    }

    pub(crate) unsafe fn record_draws(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        frame_index: usize,
        swap_extent: vk::Extent2D,
    ) -> Result<()> {
        let Some(draw_data) = self.draw_data.as_ref() else {
            return Ok(());
        };

        if draw_data.vertices.is_empty() || draw_data.indices.is_empty() {
            return Ok(());
        }

        let buffers = self
            .frames
            .get(frame_index)
            .ok_or_else(|| anyhow!("invalid frame index"))?;

        if buffers.uploaded_generation != Some(draw_data.generation) {
            return Ok(());
        }

        device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        let viewport = vk::Viewport {
            x: 0.0,
            y: swap_extent.height as f32,
            width: swap_extent.width as f32,
            height: -(swap_extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        device.cmd_set_viewport(command_buffer, 0, &[viewport]);

        let push = [swap_extent.width as f32, swap_extent.height as f32];
        let push_bytes =
            std::slice::from_raw_parts(push.as_ptr() as *const u8, size_of::<[f32; 2]>());
        device.cmd_push_constants(
            command_buffer,
            self.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            push_bytes,
        );

        let vertex_buffers = [buffers.vertex_buffer];
        let offsets = [0u64];
        device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
        device.cmd_bind_index_buffer(
            command_buffer,
            buffers.index_buffer,
            0,
            vk::IndexType::UINT32,
        );

        for draw in &draw_data.draws {
            if let Some(scissor) = Self::clip_to_scissor(draw.clip_rect, swap_extent) {
                let texture = self.texture_for(draw.texture);
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    0,
                    &[texture.descriptor_set],
                    &[],
                );
                device.cmd_set_scissor(command_buffer, 0, &[scissor]);
                device.cmd_draw_indexed(
                    command_buffer,
                    draw.index_count,
                    1,
                    draw.index_offset,
                    draw.vertex_offset,
                    0,
                );
            }
        }

        Ok(())
    }

    pub(super) fn build_draw_data(
        &self,
        primitives: &[ClippedPrimitive],
        pixels_per_point: f32,
        panel_width: u32,
        panel_height: u32,
        swap_width: u32,
        swap_height: u32,
        generation: u64,
    ) -> Option<GuiDrawData> {
        if panel_width == 0 || panel_height == 0 {
            return None;
        }

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut draws = Vec::new();

        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in primitives
        {
            let mesh = match primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => continue,
            };

            if mesh.indices.is_empty() || mesh.vertices.is_empty() {
                continue;
            }

            let clip_min_x = (clip_rect.min.x * pixels_per_point).floor();
            let clip_min_y = (clip_rect.min.y * pixels_per_point).floor();
            let clip_max_x = (clip_rect.max.x * pixels_per_point).ceil();
            let clip_max_y = (clip_rect.max.y * pixels_per_point).ceil();

            let clip = [
                clip_min_x.max(0.0),
                clip_min_y.max(0.0),
                clip_max_x.min(panel_width as f32),
                clip_max_y.min(panel_height as f32),
            ];

            if clip[0] >= clip[2] || clip[1] >= clip[3] {
                continue;
            }

            let base_vertex = vertices.len() as u32;
            for v in &mesh.vertices {
                let pos = [v.pos.x * pixels_per_point, v.pos.y * pixels_per_point];
                let color = v.color.to_array();
                let color = [
                    color[0] as f32 / 255.0,
                    color[1] as f32 / 255.0,
                    color[2] as f32 / 255.0,
                    color[3] as f32 / 255.0,
                ];
                vertices.push(GuiVertex {
                    pos,
                    uv: [v.uv.x, v.uv.y],
                    color,
                });
            }

            let first_index = indices.len() as u32;
            indices.extend(mesh.indices.iter().map(|i| i + base_vertex));

            draws.push(GuiDraw {
                clip_rect: clip,
                texture: mesh.texture_id,
                index_count: mesh.indices.len() as u32,
                index_offset: first_index,
                vertex_offset: base_vertex as i32,
            });
        }

        if draws.is_empty() {
            None
        } else {
            Some(GuiDrawData {
                generation,
                vertices,
                indices,
                draws,
                panel_width: panel_width.min(swap_width),
                panel_height: panel_height.min(swap_height),
                pixels_per_point,
            })
        }
    }

    pub(super) fn clip_to_scissor(rect: [f32; 4], swap_extent: vk::Extent2D) -> Option<vk::Rect2D> {
        let min_x = rect[0].floor().max(0.0).min(swap_extent.width as f32);
        let min_y = rect[1].floor().max(0.0).min(swap_extent.height as f32);
        let max_x = rect[2].ceil().max(0.0).min(swap_extent.width as f32);
        let max_y = rect[3].ceil().max(0.0).min(swap_extent.height as f32);

        if max_x <= min_x || max_y <= min_y {
            return None;
        }

        Some(vk::Rect2D {
            offset: vk::Offset2D {
                x: min_x as i32,
                y: min_y as i32,
            },
            extent: vk::Extent2D {
                width: (max_x - min_x) as u32,
                height: (max_y - min_y) as u32,
            },
        })
    }
}
