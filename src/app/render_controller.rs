use std::collections::VecDeque;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use glam::{Mat4, Vec3};
use log::error;

use crate::app::shader_reload::ShaderBlob;
use crate::gui::PushRender;
use crate::gui::{self, PushGui};
use crate::vulkan::{OFFSCREEN_FRAME_COUNT, VulkanRenderer};

pub struct RenderController {
    command_tx: Sender<RenderCommand>,
    gui_data_rx: Receiver<PushGui>,
    gui_data_tx: Sender<PushGui>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RenderController {
    pub fn spawn(mut renderer: VulkanRenderer, gui_shared: gui::GuiShared) -> Result<Self> {
        let (command_tx, command_rx) = bounded(128);
        let (gui_data_tx, gui_data_rx) = bounded(128);

        let render_gui_shared = gui_shared.clone();
        renderer.set_gui_sender(gui_data_tx.clone());
        let loop_gui_data_tx = gui_data_tx.clone();
        let handle = thread::Builder::new()
            .name("render-thread".into())
            .spawn(move || render_loop(renderer, render_gui_shared, command_rx, loop_gui_data_tx))
            .map_err(|err| anyhow::anyhow!("failed to spawn render thread: {err}"))?;

        Ok(Self {
            command_tx,
            gui_data_rx,
            gui_data_tx,
            handle: Some(handle),
        })
    }

    /// Sender for pushing messages to the GUI from outside the render thread.
    pub fn gui_push_sender(&self) -> Sender<PushGui> {
        self.gui_data_tx.clone()
    }

    pub fn pause(&self) {
        let _ = self.command_tx.try_send(RenderCommand::SetMinimized(true));
    }

    pub fn resume(&self) {
        let _ = self.command_tx.try_send(RenderCommand::SetMinimized(false));
    }

    pub fn resize(&self, width: u32, height: u32) {
        let _ = self
            .command_tx
            .try_send(RenderCommand::Resize { width, height });
    }

    pub fn present(&self) -> bool {
        self.command_tx.try_send(RenderCommand::Present).is_ok()
    }

    pub fn shutdown(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.command_tx.send(RenderCommand::Shutdown);
            let _ = handle.join();
        }
    }

    pub fn gui_channels(&self) -> (Receiver<PushGui>, Sender<RenderCommand>) {
        (self.gui_data_rx.clone(), self.command_tx.clone())
    }
}

impl Drop for RenderController {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Clone, Debug)]
pub enum RenderCommand {
    /// Window minimized/restored; stops dispatching while minimized.
    SetMinimized(bool),
    /// User-requested pause from the GUI; independent of minimize so a
    /// restore doesn't clobber an explicit pause.
    SetUserPaused(bool),
    Resize {
        width: u32,
        height: u32,
    },
    Present,
    Shutdown,
    /// Swap the path tracer pipeline for freshly compiled SPIR-V.
    ReloadShader(ShaderBlob),
    BackendCommand(PushRender),
    SetCamera {
        location: Vec3,
        rotation: Mat4,
    },
}

fn render_loop(
    mut renderer: VulkanRenderer,
    gui_shared: gui::GuiShared,
    command_rx: Receiver<RenderCommand>,
    gui_data_tx: Sender<PushGui>,
) {
    let mut minimized = false;
    let mut user_paused = false;
    let mut running = true;
    let mut available: VecDeque<usize> = (0..OFFSCREEN_FRAME_COUNT).collect();
    let mut in_flight: Vec<usize> = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
    let mut ready: VecDeque<usize> = VecDeque::with_capacity(OFFSCREEN_FRAME_COUNT);
    let mut current_frame: Option<usize> = None;

    let mut pending_backend_commands: Vec<PushRender> = Vec::new();

    let mut pending_camera: Option<(Vec3, Mat4)> = None;

    let mut pending_shader: Option<ShaderBlob> = None;

    // debounce resize before rerender
    const RESIZE_DEBOUNCE: Duration = Duration::from_millis(150);
    let mut pending_resize: Option<(u32, u32)> = None;
    let mut last_resize_at: Option<Instant> = None;

    // push live render size
    let mut last_render_res: Option<(u32, u32)> = None;

    // push sample count / paused state on change
    let mut last_status: Option<(u32, bool)> = None;

    while running {
        let mut present_requested = false;

        loop {
            match command_rx.try_recv() {
                Ok(command) => match command {
                    RenderCommand::SetMinimized(m) => minimized = m,
                    RenderCommand::SetUserPaused(p) => user_paused = p,
                    RenderCommand::Resize { width, height } => {
                        pending_resize = Some((width, height));
                        last_resize_at = Some(Instant::now());
                    }
                    RenderCommand::Present => present_requested = true,
                    RenderCommand::Shutdown => {
                        running = false;
                        break;
                    }
                    RenderCommand::ReloadShader(blob) => {
                        // last one wins if several compiles finished in a burst
                        pending_shader = Some(blob);
                    }
                    RenderCommand::BackendCommand(command) => {
                        pending_backend_commands.push(command);
                    }
                    RenderCommand::SetCamera { location, rotation } => {
                        pending_camera = Some((location, rotation));
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    running = false;
                    break;
                }
            }
        }

        if !running {
            break;
        }

        if let (Some((width, height)), Some(at)) = (pending_resize, last_resize_at) {
            if at.elapsed() >= RESIZE_DEBOUNCE {
                renderer.handle_resize(width, height);
                pending_resize = None;
                last_resize_at = None;
            }
        }

        if let Some((location, rotation)) = pending_camera.take() {
            renderer.set_camera_pose(location, rotation);
        }

        if let Some(blob) = pending_shader.take() {
            // Safe point for the device_wait_idle inside: no command recording
            // is in progress between the command drain and dispatch/present.
            let error = unsafe { renderer.reload_path_tracer_shader(&blob.0) }
                .err()
                .map(|err| err.to_string());
            if let Some(err) = &error {
                error!("shader reload failed: {err}");
            }
            let _ = gui_data_tx.try_send(PushGui::ShaderReload { error });
        }

        let render_res = renderer.render_resolution();
        if last_render_res != Some(render_res) {
            let _ = gui_data_tx.try_send(PushGui::RenderResolution {
                width: render_res.0,
                height: render_res.1,
            });
            last_render_res = Some(render_res);
        }

        let status = (renderer.sample_count(), minimized || user_paused);
        if last_status != Some(status) {
            let _ = gui_data_tx.try_send(PushGui::Status {
                samples: status.0,
                paused: status.1,
            });
            last_status = Some(status);
        }

        let mut completed = Vec::new();
        in_flight.retain(|index| match unsafe { renderer.frame_complete(*index) } {
            Ok(true) => {
                completed.push(*index);
                false
            }
            Ok(false) => true,
            Err(err) => {
                error!("fence status error: {err:?}");
                true
            }
        });

        for index in completed {
            ready.push_back(index);
            let perf = renderer.last_timer_perf();
            let _ = gui_data_tx.try_send(PushGui::PerfUpdate {
                compute_fps: perf.compute_fps,
                compute_ms: perf.compute_ms,
                present_fps: perf.present_fps,
                present_ms: perf.present_ms,
                heatmap_ms: perf.heatmap_ms,
                compositor_ms: perf.compositor_ms,
            });
        }

        let present_ready = match unsafe { renderer.present_ready() } {
            Ok(ready) => ready,
            Err(err) => {
                error!("present-ready error: {err:?}");
                false
            }
        };

        if present_requested && present_ready {
            // Pick the frame to present BEFORE taking the GUI frame: a taken
            // GUI frame that never reaches present_frame loses its texture
            // deltas (the font atlas is only ever uploaded once), so when
            // there is nothing to present yet it must stay pending.
            let target = if let Some(new_frame) = ready.pop_back() {
                while let Some(stale) = ready.pop_front() {
                    available.push_back(stale);
                }
                Some((new_frame, true))
            } else {
                current_frame.map(|frame| (frame, false))
            };

            if let Some((frame_index, is_new)) = target {
                let gui_frame = if let Ok(mut state) = gui_shared.write() {
                    state.take_latest()
                } else {
                    None
                };

                match unsafe {
                    renderer.present_frame(frame_index, gui_frame, &pending_backend_commands)
                } {
                    Err(err) => {
                        error!("present error: {err:?}");
                        if is_new {
                            available.push_back(frame_index);
                        }
                    }
                    Ok(()) => {
                        pending_backend_commands.clear();
                        if is_new {
                            if let Some(previous) = current_frame.replace(frame_index) {
                                available.push_back(previous);
                            }
                        }
                    }
                }
            }
        }

        if !(minimized || user_paused) {
            if let Some(index) = available.pop_front() {
                match unsafe { renderer.dispatch_compute(index) } {
                    Ok(()) => in_flight.push(index),
                    Err(err) => {
                        error!("dispatch error: {err:?}");
                        available.push_front(index);
                        thread::sleep(Duration::from_millis(16));
                    }
                }
            }
        } else {
            thread::sleep(Duration::from_millis(5));
        }

        if available.is_empty() {
            thread::sleep(Duration::from_millis(1));
        }
    }

    unsafe {
        renderer.destroy();
    }
}
