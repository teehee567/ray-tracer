use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use crossbeam_channel::Sender;
use log::info;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;
use winit::window::Window;

use crate::fps_counter::FPSCounter;
use crate::gui::{self, PushRender, PushGui};
use crate::scene::Scene;
use crate::types::{AUVec2, Au32};
use crate::vulkan::utils::save_frame::SaveImage;

use super::constants::OFFSCREEN_FRAME_COUNT;
use super::core::context::VulkanContext;
use super::core::swapchain::Swapchain;
use super::core::sync::SyncState;
use super::gui_renderer::GuiRenderer;
use super::path_tracer::PathTracer;
use super::present::record_present_commands;
use super::utils::gpu_timer::GpuTimer;

/// The Vulkan backend. Everything Vulkan lives behind this type; the
/// render thread drives it through this API only.
pub struct VulkanRenderer {
    ctx: VulkanContext,
    swapchain: Swapchain,
    path_tracer: PathTracer,
    gui: GuiRenderer,
    sync: SyncState,
    scene: Scene,
    frame: usize,
    resized: bool,
    compute_timer: GpuTimer,
    present_timer: GpuTimer,
    gui_sender: Option<Sender<PushGui>>,
    fps_counter: FPSCounter,
}

impl VulkanRenderer {
    pub unsafe fn create(window: &Window, scene: Scene) -> Result<Self> {
        let ctx = VulkanContext::new(window)?;
        let swapchain = Swapchain::new(&ctx, window)?;
        let path_tracer = PathTracer::new(&ctx, &scene, swapchain.format, swapchain.extent)?;
        let sync = SyncState::new(
            &ctx.device,
            ctx.command_pool,
            OFFSCREEN_FRAME_COUNT,
            swapchain.images.len(),
        )?;

        let compute_timer = GpuTimer::new(&ctx, OFFSCREEN_FRAME_COUNT)?;
        let present_timer = GpuTimer::new(&ctx, OFFSCREEN_FRAME_COUNT)?;

        info!("Finished initialisation of Vulkan Resources");
        let render_resolution = scene.get_camera_controls().resolution.0;
        let gui = GuiRenderer::new(&ctx, swapchain.render_pass, swapchain.extent, render_resolution)?;

        let fps_counter = FPSCounter::new(60);

        Ok(Self {
            ctx,
            swapchain,
            path_tracer,
            gui,
            sync,
            scene,
            frame: 0,
            resized: false,
            compute_timer,
            present_timer,
            gui_sender: None,
            fps_counter,
        })
    }

    pub unsafe fn upload_scene(&mut self) -> Result<()> {
        self.path_tracer.upload_scene(&self.ctx.device, &self.scene)
    }

    pub fn set_gui_sender(&mut self, sender: Sender<PushGui>) {
        self.gui_sender = Some(sender);
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

    pub fn last_timer_ms(&self) -> (f64, f64) {
        (self.compute_timer.last_ms, self.present_timer.last_ms)
    }

    // is frame_index work complete
    pub unsafe fn frame_complete(&self, frame_index: usize) -> Result<bool> {
        let status = self
            .ctx
            .device
            .get_fence_status(self.sync.frame_fences[frame_index])?;
        Ok(status == vk::SuccessCode::SUCCESS)
    }

    pub unsafe fn dispatch_compute(&mut self, frame_index: usize) -> Result<()> {
        let device = &self.ctx.device;
        let command_buffer = self.sync.compute_command_buffers[frame_index];

        if self.frame >= OFFSCREEN_FRAME_COUNT {
            self.compute_timer.read_slot(device, frame_index)?;
        }

        device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        self.update_uniform_buffer()?;
        let render_extent = self.gui.render_extent();
        self.path_tracer
            .record_dispatch(device, command_buffer, frame_index, render_extent, self.compute_timer.query_pool)?;

        device.reset_fences(&[self.sync.frame_fences[frame_index]])?;

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        device.queue_submit(
            self.ctx.compute_queue,
            &[submit_info],
            self.sync.frame_fences[frame_index],
        )?;

        self.frame += 1;

        Ok(())
    }

    pub unsafe fn present_frame(
        &mut self,
        frame_index: usize,
        gui_frame: Option<Arc<gui::GuiFrame>>,
        save_image: Option<PushRender>,
    ) -> Result<()> {
        let device = &self.ctx.device;

        let start = Instant::now();
        device.wait_for_fences(&[self.sync.frame_fences[frame_index]], true, u64::MAX)?;
    

        // the GUI update below may destroy and recreate vertex/index buffers
        // and textures, so the previous present commands must have finished
        device.wait_for_fences(&[self.sync.present_fence], true, u64::MAX)?;
        device.reset_fences(&[self.sync.present_fence])?;
        let end = start.elapsed();
        self.fps_counter.tick();
        if let Some(sender) = &self.gui_sender {
            let _ = sender.try_send(PushGui::Fps(self.fps_counter.get_fps()));
            let _ = sender .try_send(PushGui::PresentWaitTime(end.as_millis() as f64));
        }


        if let Some(frame) = gui_frame.as_deref() {
            self.gui.update(&self.ctx, self.swapchain.extent, frame)?;
        }

        self.gui.prepare_frame(&self.ctx, frame_index)?;

        let render_extent = self.gui.render_extent();
        let panel_width = self.gui.panel_width();

        let result = device.acquire_next_image_khr(
            self.swapchain.swapchain,
            u64::MAX,
            self.sync.image_available_semaphore,
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(e) => return Err(anyhow!(e)),
        };

        if self.frame >= OFFSCREEN_FRAME_COUNT {
            self.present_timer.read_slot(device, frame_index)?;
        }

        device.reset_command_buffer(
            self.sync.present_command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )?;

        let save_image_buffer = if save_image.is_some() {
            Some(SaveImage::new(&self.ctx, render_extent.x, render_extent.y)?)
        } else {
            None
        };

        record_present_commands(
            device,
            &mut self.swapchain,
            self.path_tracer.framebuffer_images[frame_index].image,
            &self.gui,
            self.sync.present_command_buffer,
            image_index,
            frame_index,
            panel_width,
            render_extent,
            save_image_buffer.as_ref(),
            self.present_timer.query_pool
        )?;

        let wait_semaphores = &[self.sync.image_available_semaphore];
        let command_buffers = &[self.sync.present_command_buffer];
        let signal_semaphores = &[self.sync.render_finished_semaphores[image_index]];
        let wait_stage_masks =
            [vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&wait_stage_masks)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        device.queue_submit(
            self.ctx.present_queue,
            &[submit_info],
            self.sync.present_fence,
        )?;

        if let Some(mut staging) = save_image_buffer {
            device.wait_for_fences(&[self.sync.present_fence], true, u64::MAX)?;
            if let Some(PushRender::SaveFrame(path)) = save_image {
                let _ = staging.save_frame(&self.ctx, path);
                staging.destroy(&self.ctx.device);
            }
        }

        let swapchains = &[self.swapchain.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = device.queue_present_khr(self.ctx.present_queue, &present_info);
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
        let mut ubo = self.scene.get_camera_controls();
        let render_extent = self.gui.render_extent();
        ubo.resolution = AUVec2(render_extent);
        ubo.time = Au32(self.frame as u32);

        self.path_tracer
            .uniform_buffer
            .write(&self.ctx.device, std::slice::from_ref(&ubo))
    }

    pub unsafe fn destroy(&mut self) {
        let device = &self.ctx.device;
        device.device_wait_idle().unwrap();

        self.compute_timer.destroy(device);
        self.gui.destroy(device);
        self.path_tracer.destroy(device);
        self.sync.destroy(device, self.ctx.command_pool);
        self.swapchain.destroy(device);
        self.ctx.destroy();
    }
}
