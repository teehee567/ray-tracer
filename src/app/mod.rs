pub mod render_controller;
pub mod save_frame;

pub use render_controller::{RenderCommand, RenderController};
use save_frame::save_frame;

use std::collections::HashMap;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;

use anyhow::{Result, anyhow};
use glam::UVec2;
use image::ImageBuffer;
use log::info;
use vulkanalia::Version;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{
    DeviceV1_0, ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension,
};
use vulkanalia::window as vk_window;
use winit::window::Window;

use crate::scene::Scene;
use crate::types::{AUVec2, Au32, CameraBufferObject};
use crate::ui;
use crate::vulkan::accumulate_image::{create_image, transition_image_layout};
use crate::vulkan::buffers::{create_shader_buffers, create_uniform_buffer};
use crate::vulkan::command_buffers::{
    create_command_buffer, record_compute_commands, record_present_commands,
};
use crate::vulkan::command_pool::create_command_pool;
use crate::vulkan::descriptors::{
    create_compute_descriptor_set_layout, create_descriptor_pool, create_descriptor_sets,
};
use crate::vulkan::fps_counter::FPSCounter;
use crate::vulkan::framebuffer::{
    create_framebuffer_images, create_swapchain_framebuffers, transition_framebuffer_images,
};
use crate::vulkan::instance::create_instance;
use crate::vulkan::logical_device::create_logical_device;
use crate::vulkan::physical_device::{SuitabilityError, pick_physical_device};
use crate::vulkan::pipeline::create_shader_module;
use crate::vulkan::pipeline::{create_compute_pipeline, create_render_pass};
use crate::vulkan::sampler::create_sampler;
use crate::vulkan::swapchain::{create_swapchain, create_swapchain_image_views};
use crate::vulkan::sync_objects::create_sync_objects;
use crate::vulkan::texture::{
    Texture, create_cubemap_sampler, create_cubemap_texture, create_texture_image,
    create_texture_sampler,
};
use crate::vulkan::utils::get_memory_type_index;

use eframe::egui::epaint::{ClippedPrimitive, Mesh, Primitive, Vertex};
use eframe::egui::{self, TextureId};

pub const VALIDATION_ENABLED: bool = true;
pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION.name,
    vk::KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION.name,
    vk::EXT_DESCRIPTOR_INDEXING_EXTENSION.name,
    vk::KHR_MAINTENANCE3_EXTENSION.name,
];

pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
pub const TILE_SIZE: u32 = 8;
pub const OFFSCREEN_FRAME_COUNT: usize = 3;
const TO_SAVE: usize = 100;

#[derive(Clone, Debug, Default)]
pub struct AppData {
    pub scene: Scene,

    pub messenger: vk::DebugUtilsMessengerEXT,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_layouts: Vec<vk::ImageLayout>,
    pub render_pass: vk::RenderPass,
    pub present_queue: vk::Queue,
    pub compute_queue: vk::Queue,
    pub compute_pipeline: vk::Pipeline,
    pub compute_pipeline_layout: vk::PipelineLayout,

    pub compute_descriptor_sets: Vec<vk::DescriptorSet>,
    pub compute_command_buffers: Vec<vk::CommandBuffer>,
    pub present_command_buffer: vk::CommandBuffer,

    pub image_available_semaphores: vk::Semaphore,
    pub compute_finished_semaphores: vk::Semaphore,

    pub uniform_buffer_memory: vk::DeviceMemory,
    pub compute_ssbo_buffer_memory: vk::DeviceMemory,

    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub swapchain_format: vk::Format,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub swapchain_framebuffers: Vec<vk::Framebuffer>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub command_pool: vk::CommandPool,
    pub uniform_buffer: vk::Buffer,
    pub compute_ssbo_buffer: vk::Buffer,
    pub descriptor_pool: vk::DescriptorPool,
    pub accumulator_view: vk::ImageView,
    pub accumulator_memory: vk::DeviceMemory,
    pub accumulator_image: vk::Image,
    pub sampler: vk::Sampler,

    pub textures: Vec<Texture>,
    pub texture_sampler: vk::Sampler,
    pub skybox_texture: Texture,
    pub skybox_sampler: vk::Sampler,

    pub framebuffer_images: Vec<vk::Image>,
    pub framebuffer_image_views: Vec<vk::ImageView>,
    pub framebuffer_memories: Vec<vk::DeviceMemory>,
}

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
    unsafe fn destroy(&mut self, device: &Device) {
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

#[derive(Clone, Copy, Debug)]
pub struct RenderMetrics {
    pub fps: f64,
}

#[derive(Clone, Debug)]
pub struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    fps_counter: FPSCounter,
    gui: GuiRenderer,
    frame_fences: Vec<vk::Fence>,
    present_fence: vk::Fence,
}

impl GuiRenderer {
    unsafe fn new(
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

    fn render_extent(&self) -> UVec2 {
        self.render_extent
    }

    fn panel_width(&self) -> u32 {
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

    fn handle_resize(&mut self, width: u32, height: u32) {
        let max_panel_width = width.saturating_sub(self.base_extent.x);
        self.panel_width = self.panel_width.min(width).min(max_panel_width);
        self.update_render_extent(width, height);
        for buffers in &mut self.frames {
            buffers.uploaded_generation = None;
        }
    }

    unsafe fn update(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &AppData,
        frame: &ui::GuiFrame,
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

    unsafe fn prepare_frame(
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
            y: 0.0,
            width: swap_extent.width as f32,
            height: swap_extent.height as f32,
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

    unsafe fn destroy(&mut self, device: &Device) {
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

        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_sampler(self.sampler, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
    }

    unsafe fn apply_textures(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &AppData,
        delta: &egui::TexturesDelta,
    ) -> Result<()> {
        for (id, image_delta) in &delta.set {
            self.update_texture(instance, device, data, *id, image_delta)?;
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
            }
        }

        Ok(())
    }

    unsafe fn update_texture(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &AppData,
        id: TextureId,
        delta: &egui::epaint::ImageDelta,
    ) -> Result<()> {
        let (pixels, size) = Self::image_to_rgba(&delta.image);
        if size[0] == 0 || size[1] == 0 {
            return Ok(());
        }

        if let Some(texture) = self.textures.get_mut(&id) {
            if delta.pos.is_none() && texture.size != size {
                if texture.view != vk::ImageView::null() {
                    device.destroy_image_view(texture.view, None);
                }
                if texture.image != vk::Image::null() {
                    device.destroy_image(texture.image, None);
                }
                if texture.memory != vk::DeviceMemory::null() {
                    device.free_memory(texture.memory, None);
                }

                *texture = Self::create_texture_resource(
                    instance,
                    device,
                    data,
                    self.descriptor_set_layout,
                    self.descriptor_pool,
                    self.sampler,
                    size,
                )?;

                Self::upload_pixels(instance, device, data, texture, &pixels, size, None, true)?;
                texture.size = size;
            } else {
                let offset = delta.pos.map(|[x, y]| [x as u32, y as u32]);
                Self::upload_pixels(
                    instance, device, data, texture, &pixels, size, offset, false,
                )?;
                if delta.pos.is_none() {
                    texture.size = size;
                }
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
            self.textures.insert(id, texture);
        }

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
        if pixels.is_empty() {
            return Ok(());
        }

        let buffer_size = pixels.len() as vk::DeviceSize;
        let (staging_buffer, staging_memory, _) = Self::create_buffer(
            instance,
            device,
            data,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
        )?;

        let ptr = device.map_memory(staging_memory, 0, buffer_size, vk::MemoryMapFlags::empty())?
            as *mut u8;
        ptr.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
        device.unmap_memory(staging_memory);

        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
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

impl App {
    pub unsafe fn create(window: &Window, scene: Scene) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        data.scene = scene;
        let scene_sizes = data.scene.get_buffer_sizes();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, window, window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_swapchain_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        create_framebuffer_images(&instance, &device, &mut data, OFFSCREEN_FRAME_COUNT)?;
        transition_framebuffer_images(&device, &mut data)?;
        create_uniform_buffer(&instance, &device, &mut data)?;
        create_shader_buffers(
            &instance,
            &device,
            &mut data,
            (scene_sizes.0 + scene_sizes.1 + scene_sizes.2) as u64,
        )?;
        create_image(&instance, &device, &mut data)?;
        transition_image_layout(&device, &mut data)?;

        data.texture_sampler = create_texture_sampler(&device)?;
        data.skybox_sampler = create_cubemap_sampler(&device)?;

        for texture_data in &data.scene.components.textures {
            let texture = create_texture_image(
                &instance,
                &device,
                &data,
                &texture_data.pixels,
                texture_data.width,
                texture_data.height,
            )?;
            data.textures.push(texture);
        }

        if data.textures.is_empty() {
            let default_pixels = [255u8, 255, 255, 255];
            let texture = create_texture_image(&instance, &device, &data, &default_pixels, 1, 1)?;
            data.textures.push(texture);
        }

        let skybox_data = &data.scene.components.skybox;
        data.skybox_texture = create_cubemap_texture(
            &instance,
            &device,
            &data,
            &skybox_data.pixels,
            skybox_data.width,
            skybox_data.height,
        )?;

        create_compute_descriptor_set_layout(&device, &mut data)?;
        create_compute_pipeline(&device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_sampler(&device, &mut data)?;
        create_descriptor_sets(
            &device,
            &mut data,
            scene_sizes.0 as u64,
            scene_sizes.1 as u64,
            scene_sizes.2 as u64,
        )?;
        create_command_buffer(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        let mut frame_fences = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
        for _ in 0..OFFSCREEN_FRAME_COUNT {
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            frame_fences.push(device.create_fence(&fence_info, None)?);
        }
        let present_fence_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let present_fence = device.create_fence(&present_fence_info, None)?;
        info!("Finished initialisation of Vulkan Resources");
        let render_resolution = data.scene.get_camera_controls().resolution.0;
        let gui = GuiRenderer::new(&instance, &device, &mut data, render_resolution)?;
        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            fps_counter: FPSCounter::new(15),
            gui,
            frame_fences,
            present_fence,
        })
    }

    pub unsafe fn upload_scene(&mut self) -> Result<()> {
        let sizes = self.data.scene.get_buffer_sizes();
        let total_size = sizes.0 + sizes.1 + sizes.2;
        let mapped_ptr = self.device.map_memory(
            self.data.compute_ssbo_buffer_memory,
            0,
            total_size as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        self.data.scene.write_buffers(mapped_ptr);

        println!(
            "sizes: bvh({}), mat({}), tri({})",
            sizes.0, sizes.1, sizes.2
        );
        println!("Total memory: {}", total_size);

        Ok(())
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.data.swapchain_extent.width = width;
        self.data.swapchain_extent.height = height;
        self.gui.handle_resize(width, height);
        self.resized = true;
    }

    unsafe fn dispatch_compute(&mut self, frame_index: usize) -> Result<()> {
        let command_buffer = self.data.compute_command_buffers[frame_index];

        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        self.update_uniform_buffer()?;
        let render_extent = self.gui.render_extent();
        record_compute_commands(
            &self.device,
            &mut self.data,
            command_buffer,
            frame_index,
            render_extent,
        )?;

        self.device
            .reset_fences(&[self.frame_fences[frame_index]])?;

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        self.device.queue_submit(
            self.data.compute_queue,
            &[submit_info],
            self.frame_fences[frame_index],
        )?;

        self.frame += 1;

        if self.frame == TO_SAVE {
            save_frame(
                &self.instance,
                &self.device,
                &mut self.data,
                self.frame as u32,
            )?;
        }

        Ok(())
    }

    unsafe fn present_frame(
        &mut self,
        frame_index: usize,
        gui_frame: Option<ui::GuiFrame>,
    ) -> Result<()> {
        if let Some(frame) = gui_frame {
            self.gui
                .update(&self.instance, &self.device, &self.data, &frame)?;
        }

        self.gui
            .prepare_frame(&self.instance, &self.device, &self.data, frame_index)?;

        let render_extent = self.gui.render_extent();
        let panel_width = self.gui.panel_width();

        self.device
            .wait_for_fences(&[self.frame_fences[frame_index]], true, u64::MAX)?;

        self.device
            .wait_for_fences(&[self.present_fence], true, u64::MAX)?;
        self.device.reset_fences(&[self.present_fence])?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores,
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(e) => return Err(anyhow!(e)),
        };

        self.device.reset_command_buffer(
            self.data.present_command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )?;

        record_present_commands(
            &self.device,
            &mut self.data,
            image_index,
            frame_index,
            panel_width,
            render_extent,
            &self.gui,
        )?;

        let wait_semaphores = &[self.data.image_available_semaphores];
        let command_buffers = &[self.data.present_command_buffer];
        let signal_semaphores = &[self.data.compute_finished_semaphores];
        let wait_stage_masks =
            [vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&wait_stage_masks)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .queue_submit(self.data.present_queue, &[submit_info], self.present_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        Ok(())
    }

    unsafe fn update_uniform_buffer(&self) -> Result<()> {
        let mut ubo = self.data.scene.get_camera_controls();
        let render_extent = self.gui.render_extent();
        let panel_width = self.gui.panel_width();
        ubo.resolution = AUVec2(render_extent);
        ubo.time = Au32(self.frame as u32);

        let memory = self.device.map_memory(
            self.data.uniform_buffer_memory,
            0,
            size_of::<CameraBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.device.unmap_memory(self.data.uniform_buffer_memory);

        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();
        self.gui.destroy(&self.device);

        let mut all_command_buffers = self.data.compute_command_buffers.clone();
        all_command_buffers.push(self.data.present_command_buffer);
        self.device
            .free_command_buffers(self.data.command_pool, &all_command_buffers);
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        for &view in &self.data.framebuffer_image_views {
            self.device.destroy_image_view(view, None);
        }
        for &image in &self.data.framebuffer_images {
            self.device.destroy_image(image, None);
        }
        for &memory in &self.data.framebuffer_memories {
            self.device.free_memory(memory, None);
        }
        for &fence in &self.frame_fences {
            self.device.destroy_fence(fence, None);
        }
        self.device.destroy_fence(self.present_fence, None);
        self.device
            .destroy_semaphore(self.data.image_available_semaphores, None);
        self.device
            .destroy_semaphore(self.data.compute_finished_semaphores, None);
        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    unsafe fn destroy_swapchain(&mut self) {
        for &framebuffer in &self.data.swapchain_framebuffers {
            self.device.destroy_framebuffer(framebuffer, None);
        }
        self.data.swapchain_framebuffers.clear();
        self.device
            .destroy_descriptor_pool(self.data.descriptor_pool, None);
        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.data.swapchain_image_layouts.clear();
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub compute: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);
        if let Some(index) = properties.iter().enumerate().find_map(|(i, p)| {
            if p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && instance
                    .get_physical_device_surface_support_khr(
                        physical_device,
                        i as u32,
                        data.surface,
                    )
                    .unwrap_or(false)
            {
                Some(i as u32)
            } else {
                None
            }
        }) {
            Ok(QueueFamilyIndices {
                graphics: index,
                compute: index,
                present: index,
            })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}
