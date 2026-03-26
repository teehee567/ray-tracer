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
use crate::vulkan::compute::ComputeResources;
use crate::vulkan::context::VulkanContext;
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
use crate::vulkan::physical_device::pick_physical_device;
use crate::vulkan::pipeline::{create_compute_pipeline, create_render_pass};
use crate::vulkan::sampler::create_sampler;
use crate::vulkan::scene_resources::SceneResources;
use crate::vulkan::swapchain::{create_swapchain, create_swapchain_image_views};
use crate::vulkan::swapchain_data::SwapchainData;
use crate::vulkan::sync::SyncState;
use crate::vulkan::sync_objects::create_sync_objects;
use crate::vulkan::texture::{
    Texture, create_cubemap_sampler, create_cubemap_texture, create_texture_image,
    create_texture_sampler,
};

use super::constants::{OFFSCREEN_FRAME_COUNT, VALIDATION_ENABLED};

#[derive(Clone, Debug)]
pub struct App {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) device: Device,
    pub(crate) ctx: VulkanContext,
    pub(crate) swapchain: SwapchainData,
    pub(crate) compute: ComputeResources,
    pub(crate) scene_res: SceneResources,
    pub(crate) sync: SyncState,
    pub(crate) messenger: vk::DebugUtilsMessengerEXT,
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
        let (instance, messenger) = create_instance(window, &entry)?;
        let surface = vk_window::create_surface(&instance, window, window)?;
        let physical_device = pick_physical_device(&instance, surface)?;
        let (device, compute_queue, present_queue) =
            create_logical_device(&entry, &instance, surface, physical_device)?;

        let command_pool =
            create_command_pool(&instance, &device, surface, physical_device)?;

        let ctx = VulkanContext {
            physical_device,
            surface,
            command_pool,
            compute_queue,
            present_queue,
        };

        let (
            swapchain_handle,
            swapchain_format,
            swapchain_extent,
            swapchain_images,
            swapchain_image_layouts,
        ) = create_swapchain(window, &instance, &device, surface, physical_device)?;
        let swapchain_image_views =
            create_swapchain_image_views(&device, &swapchain_images, swapchain_format)?;
        let render_pass = create_render_pass(&instance, &device, swapchain_format)?;
        let swapchain_framebuffers = create_swapchain_framebuffers(
            &device,
            render_pass,
            &swapchain_image_views,
            swapchain_extent,
        )?;

        let swapchain = SwapchainData {
            swapchain: swapchain_handle,
            format: swapchain_format,
            extent: swapchain_extent,
            images: swapchain_images,
            image_layouts: swapchain_image_layouts,
            image_views: swapchain_image_views,
            framebuffers: swapchain_framebuffers,
            render_pass,
        };

        let (framebuffer_images, framebuffer_image_views, framebuffer_memories) =
            create_framebuffer_images(
                &instance,
                &device,
                physical_device,
                swapchain_format,
                swapchain_extent,
                OFFSCREEN_FRAME_COUNT,
            )?;
        transition_framebuffer_images(
            &device,
            command_pool,
            compute_queue,
            &framebuffer_images,
        )?;

        let (uniform_buffer, uniform_buffer_memory) =
            create_uniform_buffer(&instance, &device, physical_device)?;

        let scene_sizes = scene.get_buffer_sizes();
        let total_ssbo_size =
            (scene_sizes.0 + scene_sizes.1 + scene_sizes.2 + scene_sizes.3 + scene_sizes.4 + scene_sizes.5) as u64;
        let (ssbo_buffer, ssbo_buffer_memory) =
            create_shader_buffers(&instance, &device, physical_device, total_ssbo_size)?;

        let (accumulator_image, accumulator_view, accumulator_memory) =
            create_image(&instance, &device, swapchain_extent, physical_device)?;
        transition_image_layout(
            &device,
            command_pool,
            compute_queue,
            accumulator_image,
        )?;

        let texture_sampler = create_texture_sampler(&device)?;
        let skybox_sampler = create_cubemap_sampler(&device)?;

        let mut textures = Vec::new();
        for texture_data in &scene.components.textures {
            let texture = create_texture_image(
                &instance,
                &device,
                &ctx,
                &texture_data.pixels,
                texture_data.width,
                texture_data.height,
            )?;
            textures.push(texture);
        }

        if textures.is_empty() {
            let default_pixels = [255u8, 255, 255, 255];
            let texture = create_texture_image(&instance, &device, &ctx, &default_pixels, 1, 1)?;
            textures.push(texture);
        }

        let skybox_data = &scene.components.skybox;
        let skybox_texture = create_cubemap_texture(
            &instance,
            &device,
            &ctx,
            &skybox_data.pixels,
            skybox_data.width,
            skybox_data.height,
        )?;

        let descriptor_set_layout =
            create_compute_descriptor_set_layout(&device, textures.len())?;
        let (compute_pipeline_layout, compute_pipeline) =
            create_compute_pipeline(&device, descriptor_set_layout)?;
        let descriptor_pool = create_descriptor_pool(
            &device,
            framebuffer_image_views.len(),
            textures.len(),
        )?;
        let sampler = create_sampler(&device)?;
        let descriptor_sets = create_descriptor_sets(
            &device,
            descriptor_set_layout,
            descriptor_pool,
            &framebuffer_image_views,
            uniform_buffer,
            ssbo_buffer,
            accumulator_view,
            sampler,
            &textures,
            texture_sampler,
            &skybox_texture,
            skybox_sampler,
            scene_sizes.0 as u64,
            scene_sizes.1 as u64,
            scene_sizes.2 as u64,
            scene_sizes.3 as u64,
            scene_sizes.4 as u64,
            scene_sizes.5 as u64,
        )?;

        let compute = ComputeResources {
            pipeline: compute_pipeline,
            pipeline_layout: compute_pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffer,
            uniform_buffer_memory,
            ssbo_buffer,
            ssbo_buffer_memory: ssbo_buffer_memory,
            accumulator_image,
            accumulator_view,
            accumulator_memory,
            sampler,
            framebuffer_images,
            framebuffer_image_views,
            framebuffer_memories,
        };

        let scene_res = SceneResources {
            scene,
            textures,
            texture_sampler,
            skybox_texture,
            skybox_sampler,
        };

        let (compute_command_buffers, present_command_buffer) =
            create_command_buffer(&device, command_pool)?;
        let (image_available_semaphore, compute_finished_semaphore) =
            create_sync_objects(&device)?;

        let sync = SyncState {
            compute_command_buffers,
            present_command_buffer,
            image_available_semaphore,
            compute_finished_semaphore,
        };

        let mut frame_fences = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
        for _ in 0..OFFSCREEN_FRAME_COUNT {
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            frame_fences.push(device.create_fence(&fence_info, None)?);
        }
        let present_fence_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let present_fence = device.create_fence(&present_fence_info, None)?;

        info!("Finished initialisation of Vulkan Resources");
        let render_resolution = scene_res.scene.get_camera_controls().resolution.0;
        let gui = GuiRenderer::new(
            &instance,
            &device,
            &ctx,
            swapchain.render_pass,
            swapchain.extent,
            render_resolution,
        )?;

        Ok(Self {
            entry,
            instance,
            device,
            ctx,
            swapchain,
            compute,
            scene_res,
            sync,
            messenger,
            frame: 0,
            resized: false,
            fps_counter: FPSCounter::new(15),
            gui,
            frame_fences,
            present_fence,
        })
    }

    pub unsafe fn upload_scene(&mut self) -> Result<()> {
        let sizes = self.scene_res.scene.get_buffer_sizes();
        let total_size = sizes.0 + sizes.1 + sizes.2 + sizes.3 + sizes.4 + sizes.5;
        let mapped_ptr = self.device.map_memory(
            self.compute.ssbo_buffer_memory,
            0,
            total_size as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        self.scene_res.scene.write_buffers(mapped_ptr);

        println!(
            "sizes: bvh({}), mat({}), tri({}), lights({}), emissive_tris({}), cdf({})",
            sizes.0, sizes.1, sizes.2, sizes.3, sizes.4, sizes.5
        );
        println!("Total memory: {}", total_size);

        Ok(())
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.swapchain.extent.width = width;
        self.swapchain.extent.height = height;
        self.gui.handle_resize(width, height);
        self.resized = true;
    }

    pub(crate) unsafe fn dispatch_compute(&mut self, frame_index: usize) -> Result<()> {
        let command_buffer = self.sync.compute_command_buffers[frame_index];

        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        self.update_uniform_buffer()?;
        let render_extent = self.gui.render_extent();
        record_compute_commands(
            &self.device,
            &self.compute,
            command_buffer,
            frame_index,
            render_extent,
        )?;

        self.device
            .reset_fences(&[self.frame_fences[frame_index]])?;

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        self.device.queue_submit(
            self.ctx.compute_queue,
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
            self.gui.update(
                &self.instance,
                &self.device,
                &self.ctx,
                self.swapchain.extent,
                frame,
            )?;
        }

        self.gui.prepare_frame(
            &self.instance,
            &self.device,
            self.ctx.physical_device,
            frame_index,
        )?;

        let render_extent = self.gui.render_extent();
        let panel_width = self.gui.panel_width();

        self.device
            .wait_for_fences(&[self.frame_fences[frame_index]], true, u64::MAX)?;

        self.device
            .wait_for_fences(&[self.present_fence], true, u64::MAX)?;
        self.device.reset_fences(&[self.present_fence])?;

        let result = self.device.acquire_next_image_khr(
            self.swapchain.swapchain,
            u64::MAX,
            self.sync.image_available_semaphore,
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(e) => return Err(anyhow!(e)),
        };

        self.device.reset_command_buffer(
            self.sync.present_command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )?;

        record_present_commands(
            &self.device,
            &mut self.swapchain,
            &self.compute,
            self.sync.present_command_buffer,
            image_index,
            frame_index,
            panel_width,
            render_extent,
            &self.gui,
        )?;

        let wait_semaphores = &[self.sync.image_available_semaphore];
        let command_buffers = &[self.sync.present_command_buffer];
        let signal_semaphores = &[self.sync.compute_finished_semaphore];
        let wait_stage_masks =
            [vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&wait_stage_masks)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .queue_submit(self.ctx.present_queue, &[submit_info], self.present_fence)?;

        let swapchains = &[self.swapchain.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.ctx.present_queue, &present_info);
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
        let mut ubo = self.scene_res.scene.get_camera_controls();
        let render_extent = self.gui.render_extent();
        ubo.resolution = AUVec2(render_extent);
        ubo.time = Au32(self.frame as u32);

        let memory = self.device.map_memory(
            self.compute.uniform_buffer_memory,
            0,
            size_of::<CameraBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.device.unmap_memory(self.compute.uniform_buffer_memory);

        Ok(())
    }

    pub unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();
        self.gui.destroy(&self.device);

        let mut all_command_buffers = self.sync.compute_command_buffers.clone();
        all_command_buffers.push(self.sync.present_command_buffer);
        self.device
            .free_command_buffers(self.ctx.command_pool, &all_command_buffers);
        self.device
            .destroy_command_pool(self.ctx.command_pool, None);
        for &view in &self.compute.framebuffer_image_views {
            self.device.destroy_image_view(view, None);
        }
        for &image in &self.compute.framebuffer_images {
            self.device.destroy_image(image, None);
        }
        for &memory in &self.compute.framebuffer_memories {
            self.device.free_memory(memory, None);
        }
        for &fence in &self.frame_fences {
            self.device.destroy_fence(fence, None);
        }
        self.device.destroy_fence(self.present_fence, None);
        self.device
            .destroy_semaphore(self.sync.image_available_semaphore, None);
        self.device
            .destroy_semaphore(self.sync.compute_finished_semaphore, None);
        self.device
            .destroy_descriptor_set_layout(self.compute.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance
            .destroy_surface_khr(self.ctx.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    unsafe fn destroy_swapchain(&mut self) {
        for &framebuffer in &self.swapchain.framebuffers {
            self.device.destroy_framebuffer(framebuffer, None);
        }
        self.swapchain.framebuffers.clear();
        self.device
            .destroy_descriptor_pool(self.compute.descriptor_pool, None);
        self.swapchain
            .image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.device
            .destroy_swapchain_khr(self.swapchain.swapchain, None);
        self.swapchain.image_layouts.clear();
    }
}
