use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use glam::UVec2;
use log::info;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{
    DeviceV1_0, ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension,
};
use vulkanalia::window as vk_window;
use winit::window::Window;

use crate::gui;
use crate::scene::Scene;
use crate::types::{AUVec2, Au32, CameraBufferObject};
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
use crate::vulkan::gui_renderer::GuiRenderer;
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

use super::constants::{OFFSCREEN_FRAME_COUNT, TO_SAVE, VALIDATION_ENABLED};
use super::data::AppData;
use super::save_frame::save_frame;

#[derive(Clone, Debug)]
pub struct App {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) data: AppData,
    pub(crate) device: Device,
    pub(crate) frame: usize,
    pub(crate) resized: bool,
    pub(crate) fps_counter: FPSCounter,
    pub(crate) gui: GuiRenderer,
    pub(crate) frame_fences: Vec<vk::Fence>,
    pub(crate) present_fence: vk::Fence,
}

impl App {
    pub unsafe fn create(window: &Window, scene: Scene) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        data.scene = scene;
        let scene_sizes = data.scene.get_buffer_sizes();
        let (instance, messenger) = create_instance(window, &entry)?;
        data.messenger = messenger;
        data.surface = vk_window::create_surface(&instance, window, window)?;
        data.physical_device = pick_physical_device(&instance, &data)?;
        let (device, compute_queue, present_queue) =
            create_logical_device(&entry, &instance, &data)?;
        data.compute_queue = compute_queue;
        data.present_queue = present_queue;
        let (
            swapchain,
            swapchain_format,
            swapchain_extent,
            swapchain_images,
            swapchain_image_layouts,
        ) = create_swapchain(window, &instance, &device, &data)?;
        data.swapchain = swapchain;
        data.swapchain_format = swapchain_format;
        data.swapchain_extent = swapchain_extent;
        data.swapchain_images = swapchain_images;
        data.swapchain_image_layouts = swapchain_image_layouts;
        data.swapchain_image_views =
            create_swapchain_image_views(&device, &data.swapchain_images, data.swapchain_format)?;
        data.render_pass = create_render_pass(&instance, &device, data.swapchain_format)?;
        data.swapchain_framebuffers = create_swapchain_framebuffers(
            &device,
            data.render_pass,
            &data.swapchain_image_views,
            data.swapchain_extent,
        )?;
        data.command_pool = create_command_pool(&instance, &device, &data)?;
        let (framebuffer_images, framebuffer_image_views, framebuffer_memories) =
            create_framebuffer_images(&instance, &device, &data, OFFSCREEN_FRAME_COUNT)?;
        data.framebuffer_images = framebuffer_images;
        data.framebuffer_image_views = framebuffer_image_views;
        data.framebuffer_memories = framebuffer_memories;
        transition_framebuffer_images(
            &device,
            data.command_pool,
            data.compute_queue,
            &data.framebuffer_images,
        )?;
        let (uniform_buffer, uniform_buffer_memory) =
            create_uniform_buffer(&instance, &device, &data)?;
        data.uniform_buffer = uniform_buffer;
        data.uniform_buffer_memory = uniform_buffer_memory;
        let (shader_buffer, shader_buffer_memory) = create_shader_buffers(
            &instance,
            &device,
            &data,
            (scene_sizes.0 + scene_sizes.1 + scene_sizes.2) as u64,
        )?;
        data.compute_ssbo_buffer = shader_buffer;
        data.compute_ssbo_buffer_memory = shader_buffer_memory;
        let (accumulator_image, accumulator_view, accumulator_memory) =
            create_image(&instance, &device, data.swapchain_extent, &data)?;
        data.accumulator_image = accumulator_image;
        data.accumulator_view = accumulator_view;
        data.accumulator_memory = accumulator_memory;
        transition_image_layout(
            &device,
            data.command_pool,
            data.compute_queue,
            data.accumulator_image,
        )?;

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
        let (compute_pipeline_layout, compute_pipeline) =
            create_compute_pipeline(&device, data.descriptor_set_layout)?;
        data.compute_pipeline_layout = compute_pipeline_layout;
        data.compute_pipeline = compute_pipeline;
        data.descriptor_pool = create_descriptor_pool(
            &device,
            data.framebuffer_image_views.len(),
            data.textures.len(),
        )?;
        data.sampler = create_sampler(&device)?;
        data.compute_descriptor_sets = create_descriptor_sets(
            &device,
            &data,
            scene_sizes.0 as u64,
            scene_sizes.1 as u64,
            scene_sizes.2 as u64,
        )?;
        let (compute_command_buffers, present_command_buffer) =
            create_command_buffer(&device, data.command_pool)?;
        data.compute_command_buffers = compute_command_buffers;
        data.present_command_buffer = present_command_buffer;
        let (image_available_semaphores, compute_finished_semaphores) =
            create_sync_objects(&device)?;
        data.image_available_semaphores = image_available_semaphores;
        data.compute_finished_semaphores = compute_finished_semaphores;

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

    pub(crate) unsafe fn dispatch_compute(&mut self, frame_index: usize) -> Result<()> {
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

        Ok(())
    }

    pub(crate) unsafe fn present_frame(
        &mut self,
        frame_index: usize,
        gui_frame: Option<Arc<gui::GuiFrame>>,
    ) -> Result<()> {
        if let Some(frame) = gui_frame.as_deref() {
            self.gui
                .update(&self.instance, &self.device, &self.data, frame)?;
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
