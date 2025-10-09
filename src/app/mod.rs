pub mod render_controller;
pub mod save_frame;

pub use render_controller::{RenderCommand, RenderController};
use save_frame::save_frame;

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
use crate::vulkan::framebuffer::{create_framebuffer_images, transition_framebuffer_images};
use crate::vulkan::instance::create_instance;
use crate::vulkan::logical_device::create_logical_device;
use crate::vulkan::physical_device::{SuitabilityError, pick_physical_device};
use crate::vulkan::pipeline::{create_compute_pipeline, create_render_pass};
use crate::vulkan::sampler::create_sampler;
use crate::vulkan::swapchain::{create_swapchain, create_swapchain_image_views};
use crate::vulkan::sync_objects::create_sync_objects;
use crate::vulkan::texture::{
    Texture, create_cubemap_sampler, create_cubemap_texture, create_texture_image,
    create_texture_sampler,
};
use crate::vulkan::utils::get_memory_type_index;

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

#[derive(Clone, Debug)]
struct GuiCopyState {
    panel_width: u32,
    render_extent: UVec2,
    base_extent: UVec2,
    last_generation: Option<u64>,
    staging: Option<GuiStaging>,
}

#[derive(Clone, Debug)]
struct GuiStaging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: vk::DeviceSize,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct GuiCopyInfo {
    pub buffer: vk::Buffer,
    pub width: u32,
    pub height: u32,
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
    gui_copy: GuiCopyState,
    frame_fences: Vec<vk::Fence>,
    present_fence: vk::Fence,
}

impl GuiCopyState {
    fn new(initial_extent: UVec2, base_extent: UVec2) -> Self {
        let clamped_width = base_extent.x.min(initial_extent.x);
        let clamped_height = base_extent.y.min(initial_extent.y);
        Self {
            panel_width: 0,
            render_extent: UVec2::new(clamped_width, clamped_height),
            base_extent,
            last_generation: None,
            staging: None,
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
        let copy_width = frame.width.min(swap_width).min(max_panel_width);
        let copy_height = frame.height.min(swap_height);

        self.panel_width = copy_width;
        self.update_render_extent(swap_width, swap_height);

        if copy_width == 0 || copy_height == 0 {
            if let Some(staging) = &mut self.staging {
                staging.width = 0;
                staging.height = 0;
            }
            self.last_generation = Some(frame.generation);
            return Ok(());
        }

        let bytes_per_row = copy_width as usize * 4;
        let required = (bytes_per_row * copy_height as usize) as vk::DeviceSize;

        self.ensure_capacity(instance, device, data, required)?;

        if let Some(staging) = &mut self.staging {
            let ptr = device.map_memory(staging.memory, 0, required, vk::MemoryMapFlags::empty())?
                as *mut u8;

            let src = frame.pixels.as_ref();
            let src_width = frame.width as usize;
            let src_stride = src_width * 4;

            for row in 0..copy_height as usize {
                let src_start = row * src_stride;
                let dst_start = row * bytes_per_row;
                let src_slice = &src[src_start..src_start + bytes_per_row];
                std::ptr::copy_nonoverlapping(
                    src_slice.as_ptr(),
                    ptr.add(dst_start),
                    bytes_per_row,
                );
            }

            device.unmap_memory(staging.memory);
            staging.width = copy_width;
            staging.height = copy_height;
        }

        self.last_generation = Some(frame.generation);
        Ok(())
    }

    fn copy_info(&self) -> Option<GuiCopyInfo> {
        self.staging.as_ref().and_then(|staging| {
            if self.panel_width == 0 || staging.width == 0 || staging.height == 0 {
                None
            } else {
                Some(GuiCopyInfo {
                    buffer: staging.buffer,
                    width: staging.width,
                    height: staging.height,
                })
            }
        })
    }

    fn render_extent(&self) -> UVec2 {
        self.render_extent
    }

    fn panel_width(&self) -> u32 {
        self.panel_width
    }

    fn handle_resize(&mut self, width: u32, height: u32) {
        let max_panel_width = width.saturating_sub(self.base_extent.x);
        let clamped = self.panel_width.min(width).min(max_panel_width);
        self.panel_width = clamped;
        self.update_render_extent(width, height);
        if let Some(staging) = &mut self.staging {
            staging.width = staging.width.min(clamped);
            staging.height = staging.height.min(height);
        }
    }

    unsafe fn destroy(&mut self, device: &Device) {
        if let Some(staging) = self.staging.take() {
            device.destroy_buffer(staging.buffer, None);
            device.free_memory(staging.memory, None);
        }
    }

    unsafe fn ensure_capacity(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &AppData,
        required: vk::DeviceSize,
    ) -> Result<()> {
        if let Some(staging) = &self.staging {
            if staging.capacity >= required {
                return Ok(());
            }
        }

        if let Some(existing) = self.staging.take() {
            device.destroy_buffer(existing.buffer, None);
            device.free_memory(existing.memory, None);
        }

        if required == 0 {
            return Ok(());
        }

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(required)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None)?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let memory_type = get_memory_type_index(
            instance,
            data,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            requirements,
        )?;

        let allocation_size = requirements.size.max(required);
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type);

        let memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(buffer, memory, 0)?;

        self.staging = Some(GuiStaging {
            buffer,
            memory,
            capacity: allocation_size,
            width: 0,
            height: 0,
        });

        Ok(())
    }

    fn update_render_extent(&mut self, swap_width: u32, swap_height: u32) {
        let available_width = swap_width.saturating_sub(self.panel_width);
        let width = self.base_extent.x.min(available_width);
        let height = self.base_extent.y.min(swap_height);
        self.render_extent = UVec2::new(width, height);
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
        let swapchain_width = data.swapchain_extent.width;
        let swapchain_height = data.swapchain_extent.height;
        let render_resolution = data.scene.get_camera_controls().resolution.0;
        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            fps_counter: FPSCounter::new(15),
            gui_copy: GuiCopyState::new(
                UVec2::new(swapchain_width, swapchain_height),
                render_resolution,
            ),
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
        self.gui_copy.handle_resize(width, height);
        self.resized = true;
    }

    unsafe fn dispatch_compute(&mut self, frame_index: usize) -> Result<()> {
        let command_buffer = self.data.compute_command_buffers[frame_index];

        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        self.update_uniform_buffer()?;
        let render_extent = self.gui_copy.render_extent();
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
            self.gui_copy
                .update(&self.instance, &self.device, &self.data, &frame)?;
        }

        let render_extent = self.gui_copy.render_extent();
        let panel_width = self.gui_copy.panel_width();

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

        let gui_copy = self.gui_copy.copy_info();
        record_present_commands(
            &self.device,
            &mut self.data,
            image_index,
            frame_index,
            panel_width,
            render_extent,
            gui_copy,
        )?;

        let wait_semaphores = &[self.data.image_available_semaphores];
        let command_buffers = &[self.data.present_command_buffer];
        let signal_semaphores = &[self.data.compute_finished_semaphores];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
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
        let render_extent = self.gui_copy.render_extent();
        let panel_width = self.gui_copy.panel_width();
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
        self.gui_copy.destroy(&self.device);

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
