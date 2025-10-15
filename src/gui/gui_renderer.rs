use std::collections::HashMap;
use std::mem::size_of;

use anyhow::{Result, anyhow};
use eframe::egui::epaint::{ClippedPrimitive, Primitive};
use eframe::egui::{self, TextureId};
use glam::UVec2;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{self, DeviceV1_0};

use crate::vulkan::pipeline::create_shader_module;
use crate::vulkan::utils::get_memory_type_index;

use super::GuiFrame;
use crate::app::constants::OFFSCREEN_FRAME_COUNT;
use crate::app::data::AppData;

const MAX_GUI_TEXTURES: u32 = 64;

#[derive(Clone, Debug)]
pub(crate) struct GuiRenderer {
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
    draw_data: Option<GuiDrawData>,
    fallback: GuiTexture,
}

#[derive(Clone, Debug)]
struct GuiTexture {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    descriptor_set: vk::DescriptorSet,
    size: [u32; 2],
}

impl Default for GuiTexture {
    fn default() -> Self {
        Self {
            image: vk::Image::null(),
            view: vk::ImageView::null(),
            memory: vk::DeviceMemory::null(),
            descriptor_set: vk::DescriptorSet::null(),
            size: [0, 0],
        }
    }
}

#[derive(Clone, Debug)]
struct GuiFrameBuffers {
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    vertex_capacity: vk::DeviceSize,
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_capacity: vk::DeviceSize,
    uploaded_generation: Option<u64>,
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
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
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
struct GuiDrawData {
    generation: u64,
    vertices: Vec<GuiVertex>,
    indices: Vec<u32>,
    draws: Vec<GuiDraw>,
    panel_width: u32,
    panel_height: u32,
    pixels_per_point: f32,
}

#[derive(Clone, Debug)]
struct GuiDraw {
    clip_rect: [f32; 4],
    texture: TextureId,
    index_count: u32,
    index_offset: u32,
    vertex_offset: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GuiVertex {
    pos: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

impl GuiRenderer {
    pub(crate) unsafe fn new(
        instance: &Instance,
        device: &Device,
        data: &mut AppData,
        base_extent: UVec2,
    ) -> Result<Self> {
        let clamped_width = base_extent.x.min(data.swapchain_extent.width);
        let clamped_height = base_extent.y.min(data.swapchain_extent.height);
        let render_extent = UVec2::new(clamped_width, clamped_height);

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .min_lod(0.0)
            .max_lod(0.0);
        let sampler = device.create_sampler(&sampler_info, None)?;

        let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let image_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .build();
        let bindings = [sampler_binding, image_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)?;

        let push_constant = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<[f32; 2]>() as u32)
            .build();
        let descriptor_set_layouts = [descriptor_set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&descriptor_set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant));
        let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        let vert_shader =
            create_shader_module(device, include_bytes!("../../src/shaders/gui.vert.spv"))?;
        let frag_shader =
            create_shader_module(device, include_bytes!("../../src/shaders/gui.frag.spv"))?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader)
                .name(b"main\0")
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader)
                .name(b"main\0")
                .build(),
        ];

        let vertex_binding = vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<GuiVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build();
        let vertex_bindings = [vertex_binding];
        let pos_offset = 0u32;
        let uv_offset = size_of::<[f32; 2]>() as u32;
        let color_offset = (size_of::<[f32; 2]>() * 2) as u32;
        let attributes = [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(pos_offset)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(uv_offset)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(color_offset)
                .build(),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_bindings)
            .vertex_attribute_descriptions(&attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::_1);

        let blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build();
        let blend_attachments = [blend_attachment];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(data.render_pass)
            .subpass(0);

        let pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?
            .0[0];

        device.destroy_shader_module(vert_shader, None);
        device.destroy_shader_module(frag_shader, None);

        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::SAMPLER)
                .descriptor_count(MAX_GUI_TEXTURES)
                .build(),
            vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(MAX_GUI_TEXTURES)
                .build(),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_GUI_TEXTURES)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = device.create_descriptor_pool(&pool_info, None)?;

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

        renderer.fallback = Self::create_texture_resource(
            instance,
            device,
            data,
            renderer.descriptor_set_layout,
            renderer.descriptor_pool,
            renderer.sampler,
            [1, 1],
        )?;
        Self::upload_pixels(
            instance,
            device,
            data,
            &renderer.fallback,
            &[255u8, 255, 255, 255],
            [1, 1],
            None,
            true,
        )?;

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

    fn update_render_extent(&mut self, swap_width: u32, swap_height: u32) {
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

    pub(crate) unsafe fn update(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &AppData,
        frame: &GuiFrame,
    ) -> Result<()> {
        if self.last_generation == Some(frame.generation) {
            return Ok(());
        }

        let swap_width = data.swapchain_extent.width;
        let swap_height = data.swapchain_extent.height;
        let max_panel_width = swap_width.saturating_sub(self.base_extent.x);
        self.panel_width = frame.panel_width.min(swap_width).min(max_panel_width);
        self.update_render_extent(swap_width, swap_height);

        self.apply_textures(instance, device, data, &frame.textures_delta)?;
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
        data: &AppData,
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
            let (buffer, memory, capacity) = Self::create_buffer(
                instance,
                device,
                data,
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

            let (buffer, memory, capacity) = Self::create_buffer(
                instance,
                device,
                data,
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

    unsafe fn create_buffer(
        instance: &Instance,
        device: &Device,
        data: &AppData,
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
            data,
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

    unsafe fn create_texture_resource(
        instance: &Instance,
        device: &Device,
        data: &AppData,
        layout: vk::DescriptorSetLayout,
        pool: vk::DescriptorPool,
        sampler: vk::Sampler,
        size: [u32; 2],
    ) -> Result<GuiTexture> {
        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        let descriptor_set = device.allocate_descriptor_sets(&alloc_info)?[0];

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: size[0].max(1),
                height: size[1].max(1),
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = device.create_image(&image_info, None)?;
        let requirements = device.get_image_memory_requirements(image);
        let memory_type = get_memory_type_index(
            instance,
            data,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            requirements,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        let memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_image_memory(image, memory, 0)?;

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        let view = device.create_image_view(&view_info, None)?;

        let sampler_info = vk::DescriptorImageInfo::builder()
            .sampler(sampler)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();
        let image_info = vk::DescriptorImageInfo::builder()
            .sampler(vk::Sampler::null())
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .build();

        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&sampler_info))
            .build();
        let image_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info))
            .build();

        let writes = [sampler_write, image_write];
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);

        Ok(GuiTexture {
            image,
            view,
            memory,
            descriptor_set,
            size,
        })
    }

    unsafe fn apply_textures(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &AppData,
        delta: &egui::TexturesDelta,
    ) -> Result<()> {
        for (id, delta) in &delta.set {
            let size = [delta.image.width() as u32, delta.image.height() as u32];
            let (pixels, size) = Self::image_to_rgba(&delta.image);
            let offset = delta.pos.map(|[x, y]| [x as u32, y as u32]);
            if let Some(texture) = self.textures.get_mut(id) {
                Self::upload_pixels(
                    instance, device, data, texture, &pixels, size, offset, false,
                )?;
                if delta.pos.is_none() {
                    texture.size = size;
                }
            } else {
                let mut texture = Self::create_texture_resource(
                    instance,
                    device,
                    data,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    self.sampler,
                    size,
                )?;
                Self::upload_pixels(instance, device, data, &texture, &pixels, size, None, true)?;
                texture.size = size;
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

    unsafe fn upload_pixels(
        instance: &Instance,
        device: &Device,
        data: &AppData,
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
            data,
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
            .command_pool(data.command_pool)
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
        device.queue_submit(data.compute_queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(data.compute_queue)?;
        device.free_command_buffers(data.command_pool, &[command_buffer]);

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_memory, None);

        Ok(())
    }

    fn image_to_rgba(image: &egui::epaint::ImageData) -> (Vec<u8>, [u32; 2]) {
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

    fn build_draw_data(
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

    fn texture_for(&self, id: TextureId) -> &GuiTexture {
        self.textures.get(&id).unwrap_or(&self.fallback)
    }

    fn clip_to_scissor(rect: [f32; 4], swap_extent: vk::Extent2D) -> Option<vk::Rect2D> {
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

    unsafe fn transition_image(
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
