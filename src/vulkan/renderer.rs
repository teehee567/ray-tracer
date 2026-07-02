use std::sync::Arc;

use anyhow::{Result, anyhow};
use crossbeam_channel::Sender;
use glam::{Mat3, Mat4, Vec3};
use log::info;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{KhrSwapchainExtensionDeviceCommands, SuccessCode};
use winit::window::Window;

use crate::fps_counter::FPSCounter;
use crate::gui::{self, PushGui, PushRender, RenderMode};
use crate::scene::Scene;
use crate::types::{AUVec2, AVec2, Au32, viewport_uv};
use crate::vulkan::heatmap_renderer::HeatmapRenderer;
use crate::vulkan::utils::save_frame::SaveImage;

use super::constants::OFFSCREEN_FRAME_COUNT;
use super::core::context::VulkanContext;
use super::core::swapchain::Swapchain;
use super::core::sync::SyncState;
use super::gui_renderer::GuiRenderer;
use super::path_tracer::PathTracer;
use super::present::{HeatmapParams, PresentFrame, PresentTimers, record_present_commands};
use super::utils::gpu_timer::GpuTimer;

/// Snapshot of the renderer's timing counters, handed to the GUI each frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimerPerf {
    pub compute_fps: f64,
    pub compute_ms: f64,
    pub present_fps: f64,
    pub present_ms: f64,
    pub heatmap_ms: f64,
    pub compositor_ms: f64,
}

/// The Vulkan backend. Everything Vulkan lives behind this type; the
/// render thread drives it through this API only.
pub struct VulkanRenderer {
    ctx: VulkanContext,
    swapchain: Swapchain,
    path_tracer: PathTracer,
    gui: GuiRenderer,
    heatmap: HeatmapRenderer,
    sync: SyncState,
    scene: Scene,
    frame: usize,
    resized: bool,
    compute_timer: GpuTimer,
    present_timer: GpuTimer,
    heatmap_timer: GpuTimer,
    compositor_timer: GpuTimer,
    present_rate: FPSCounter,
    gui_sender: Option<Sender<PushGui>>,
    render_mode: RenderMode,
    heatmap_band: (u32, u32),
    last_shader_spv: Option<Arc<Vec<u8>>>,
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
        let heatmap_timer = GpuTimer::new(&ctx, OFFSCREEN_FRAME_COUNT)?;
        let compositor_timer = GpuTimer::new(&ctx, OFFSCREEN_FRAME_COUNT)?;

        info!("Finished initialisation of Vulkan Resources");
        let render_resolution = scene.get_camera_controls().resolution.0;
        let gui = GuiRenderer::new(
            &ctx,
            swapchain.render_pass,
            swapchain.extent,
            render_resolution,
        )?;

        // size to renderer part
        let render_extent = gui.render_extent();
        let heatmap = HeatmapRenderer::new(
            &ctx,
            swapchain.render_pass,
            vk::Extent2D {
                width: render_extent.x,
                height: render_extent.y,
            },
            &scene,
        )?;
        let heatmap_max_depth = heatmap.max_depth();

        Ok(Self {
            ctx,
            swapchain,
            path_tracer,
            gui,
            heatmap,
            sync,
            scene,
            frame: 0,
            resized: false,
            compute_timer,
            present_timer,
            heatmap_timer,
            compositor_timer,
            present_rate: FPSCounter::new(60),
            gui_sender: None,
            render_mode: RenderMode::default(),
            heatmap_band: (0, heatmap_max_depth),
            last_shader_spv: None,
        })
    }

    pub unsafe fn upload_scene(&mut self) -> Result<()> {
        self.path_tracer.upload_scene(&self.ctx.device, &self.scene)
    }

    pub fn set_gui_sender(&mut self, sender: Sender<PushGui>) {
        let _ = sender.try_send(PushGui::HeatmapInfo {
            max_depth: self.heatmap.max_depth(),
        });
        self.gui_sender = Some(sender);
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.gui.handle_resize(width, height);
        let render_extent = self.gui.render_extent();
        let extent = vk::Extent2D {
            width: render_extent.x,
            height: render_extent.y,
        };

        unsafe {
            // rebuild swapchain before targets
            if let Err(e) = self.swapchain.recreate(&self.ctx, width, height) {
                log::error!("swapchain resize failed: {e}");
                return;
            }
            if let Err(e) = self.path_tracer.handle_resize(&self.ctx, extent) {
                log::error!("path tracer resize failed: {e}");
            }
            if let Err(e) = self.heatmap.handle_resize(&self.ctx, extent) {
                log::error!("heatmap resize failed: {e}");
            }
        }

        // restart accumulation after resize
        self.reset_accumulator();
        self.resized = true;
    }

    //reset for new camera
    pub fn set_camera_pose(&mut self, location: Vec3, rotation: Mat4) {
        self.scene.set_camera_pose(location, rotation);
        self.reset_accumulator();
    }

    pub fn reset_accumulator(&mut self) {
        unsafe {
            if let Err(e) = self.ctx.device.device_wait_idle() {
                log::error!("reset_accumulator: device_wait_idle failed: {e}");
            }
        }
        self.frame = 0;
    }

    /// Accumulated path-trace samples (includes any in-flight dispatch).
    pub fn sample_count(&self) -> u32 {
        self.frame as u32
    }

    pub unsafe fn reload_path_tracer_shader(&mut self, spv: &Arc<Vec<u8>>) -> Result<()> {
        self.ctx.device.device_wait_idle()?;
        self.path_tracer.rebuild_pipeline(&self.ctx.device, spv)?;
        self.last_shader_spv = Some(spv.clone());
        // restart accumulation so old-shader samples aren't blended in
        self.frame = 0;
        Ok(())
    }

    pub unsafe fn reload_scene(&mut self, mut new_scene: Scene) -> Result<()> {
        let old_camera = &self.scene.components.camera;
        new_scene.components.camera.location = old_camera.location;
        new_scene.components.camera.rotation = old_camera.rotation;

        self.ctx.device.device_wait_idle()?;

        let render_size = self.path_tracer.render_size();
        let mut new_path_tracer = PathTracer::new(
            &self.ctx,
            &new_scene,
            self.path_tracer.framebuffer_format,
            vk::Extent2D {
                width: render_size.x,
                height: render_size.y,
            },
        )?;
        if let Some(spv) = &self.last_shader_spv {
            if let Err(e) = new_path_tracer.rebuild_pipeline(&self.ctx.device, spv) {
                log::warn!("scene reload kept embedded shader: {e}");
            }
        }
        self.path_tracer.destroy(&self.ctx.device);
        self.path_tracer = new_path_tracer;
        self.scene = new_scene;
        self.path_tracer.upload_scene(&self.ctx.device, &self.scene)?;

        self.heatmap.reload_scene(&self.ctx, &self.scene)?;
        self.heatmap_band = (0, self.heatmap.max_depth());
        if let Some(sender) = &self.gui_sender {
            let _ = sender.try_send(PushGui::HeatmapInfo {
                max_depth: self.heatmap.max_depth(),
            });
        }

        // restart accumulation; device is already idle
        self.frame = 0;
        Ok(())
    }

    /// actual render target size
    pub fn render_resolution(&self) -> (u32, u32) {
        let size = self.path_tracer.render_size();
        (size.x, size.y)
    }

    pub fn last_timer_perf(&self) -> TimerPerf {
        TimerPerf {
            compute_fps: self.compute_timer.fps(),
            compute_ms: self.compute_timer.last_ms(),
            present_fps: self.present_rate.get_fps(),
            present_ms: self.present_rate.last_frame_ms(),
            heatmap_ms: self.heatmap_timer.last_ms(),
            compositor_ms: self.compositor_timer.last_ms(),
        }
    }

    // is frame_index work complete
    pub unsafe fn frame_complete(&self, frame_index: usize) -> Result<bool> {
        let status = self
            .ctx
            .device
            .get_fence_status(self.sync.frame_fences[frame_index])?;
        Ok(status == vk::SuccessCode::SUCCESS)
    }

    pub unsafe fn present_ready(&self) -> Result<bool> {
        let device = &self.ctx.device;

        match device.get_fence_status(self.sync.present_fence)? {
            SuccessCode::SUCCESS => Ok(true),
            SuccessCode::NOT_READY => Ok(false),
            _ => unreachable!(),
        }
    }

    pub unsafe fn dispatch_compute(&mut self, frame_index: usize) -> Result<()> {
        let device = &self.ctx.device;

        self.sync.timeline_counter += 1;
        let timeline_value = self.sync.timeline_counter;
        self.sync.slot_timeline_values[frame_index] = timeline_value;

        let command_buffer = self.sync.compute_command_buffers[frame_index];

        self.compute_timer.read_slot(device, frame_index)?;

        device.reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        let path_trace = self.render_mode == RenderMode::PathTracer;
        // freeze accumulation counter if showing heatmap
        if path_trace {
            self.update_uniform_buffer()?;
        }
        let render_extent = self.path_tracer.render_size();
        self.path_tracer.record_dispatch(
            device,
            command_buffer,
            frame_index,
            render_extent,
            self.compute_timer.query_pool(),
            path_trace,
        )?;

        device.reset_fences(&[self.sync.frame_fences[frame_index]])?;

        let command_buffers = &[command_buffer];

        let signal_semaphores = &[self.sync.compute_timeline];
        let signal_values = &[timeline_value];
        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::builder().signal_semaphore_values(signal_values);
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .push_next(&mut timeline_info);

        device.queue_submit(
            self.ctx.compute_queue,
            &[submit_info],
            self.sync.frame_fences[frame_index],
        )?;

        if path_trace {
            self.frame += 1;
        }

        Ok(())
    }

    pub unsafe fn present_frame(
        &mut self,
        frame_index: usize,
        gui_frame: Option<Arc<gui::GuiFrame>>,
        commands: &[PushRender],
    ) -> Result<()> {
        let device = &self.ctx.device;

        // No fence wait here: the render loop only calls this after present_ready()
        // returned true, so the present fence is already signaled.

        self.present_rate.tick();

        let mut save_request = None;
        for command in commands {
            match command {
                PushRender::SetHeatmapBand { low, high } => self.heatmap_band = (*low, *high),
                PushRender::SetRenderMode(mode) => self.render_mode = *mode,
                PushRender::SaveFrame(path) => save_request = Some(path.clone()),
            }
        }

        if let Some(frame) = gui_frame.as_deref() {
            self.gui.update(&self.ctx, self.swapchain.extent, frame)?;
        }

        self.gui.prepare_frame(&self.ctx, frame_index)?;

        let render_extent = self.path_tracer.render_size();
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

        self.present_timer.read_slot(device, frame_index)?;
        self.heatmap_timer.read_slot(device, frame_index)?;
        self.compositor_timer.read_slot(device, frame_index)?;

        device.reset_command_buffer(
            self.sync.present_command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )?;

        let save_image_buffer = if save_request.is_some() {
            Some(SaveImage::new(&self.ctx, render_extent.x, render_extent.y)?)
        } else {
            None
        };

        let heatmap_view_proj = self.heatmap_view_proj();

        record_present_commands(
            device,
            &mut self.swapchain,
            &self.gui,
            &mut self.heatmap,
            &PresentFrame {
                command_buffer: self.sync.present_command_buffer,
                swapchain_index: image_index,
                frame_index,
                framebuffer_image: self.path_tracer.framebuffer_images[frame_index].image,
                panel_width,
                render_extent,
                save_image: save_image_buffer.as_ref(),
            },
            &HeatmapParams {
                active: self.render_mode == RenderMode::BvhHeatmap,
                view_proj: heatmap_view_proj,
                band: self.heatmap_band,
            },
            &PresentTimers {
                present: self.present_timer.query_pool(),
                heatmap: self.heatmap_timer.query_pool(),
                compositor: self.compositor_timer.query_pool(),
            },
        )?;

        let wait_semaphores = &[
            self.sync.image_available_semaphore,
            self.sync.compute_timeline,
        ];
        let command_buffers = &[self.sync.present_command_buffer];
        let signal_semaphores = &[self.sync.render_finished_semaphores[image_index]];
        let wait_stage_masks = [
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
        ];
        let wait_values = &[0, self.sync.slot_timeline_values[frame_index]];
        let mut timeline_info =
            vk::TimelineSemaphoreSubmitInfo::builder().wait_semaphore_values(wait_values);
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&wait_stage_masks)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores)
            .push_next(&mut timeline_info);

        // Reset immediately before submit so an early return above doesnt kill
        device.reset_fences(&[self.sync.present_fence])?;

        device.queue_submit(
            self.ctx.present_queue,
            &[submit_info],
            self.sync.present_fence,
        )?;

        if let Some(mut staging) = save_image_buffer {
            device.wait_for_fences(&[self.sync.present_fence], true, u64::MAX)?;
            if let Some(path) = save_request {
                let _ = staging.save_frame(&self.ctx, path);
            }
            staging.destroy(&self.ctx.device);
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
        let render_extent = self.path_tracer.render_size();
        ubo.resolution = AUVec2(render_extent);
        ubo.view_port_uv = AVec2(viewport_uv(render_extent));
        ubo.time = Au32(self.frame as u32);

        self.path_tracer
            .uniform_buffer
            .write(&self.ctx.device, std::slice::from_ref(&ubo))
    }

    // view projection
    fn heatmap_view_proj(&self) -> Mat4 {
        let cam = self.scene.get_camera_controls();
        let r = Mat3::from_mat4(cam.rotation.0);
        let view = Mat4::from_mat3(r.transpose()) * Mat4::from_translation(-cam.location.0);
        // match viewport to resize
        let view_port_uv = viewport_uv(self.path_tracer.render_size());
        let tan_half_y = view_port_uv.y / (2.0 * cam.focal_length.0);
        let fov_y = 2.0 * tan_half_y.atan();
        let aspect = view_port_uv.x / view_port_uv.y;
        let mut proj = Mat4::perspective_rh(fov_y, aspect, 0.01, 10_000.0);
        // flip for vulkan
        proj.y_axis.y *= -1.0;
        proj * view
    }

    pub unsafe fn destroy(&mut self) {
        let device = &self.ctx.device;
        device.device_wait_idle().unwrap();

        self.present_timer.destroy(device);
        self.compute_timer.destroy(device);
        self.heatmap_timer.destroy(device);
        self.compositor_timer.destroy(device);
        self.gui.destroy(device);
        self.heatmap.destroy(device);
        self.path_tracer.destroy(device);
        self.sync.destroy(device, self.ctx.command_pool);
        self.swapchain.destroy(device);
        self.ctx.destroy();
    }
}
