use std::collections::VecDeque;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use log::error;

use crate::fps_counter::FPSCounter;
use crate::gui::{self, PushGui};
use crate::gui::GuiData;
use crate::gui::PushRender;
use crate::vulkan::{OFFSCREEN_FRAME_COUNT, VulkanRenderer};

pub struct RenderController {
    command_tx: Sender<RenderCommand>,
    gui_data_rx: Receiver<PushGui>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RenderController {
    pub fn spawn(mut renderer: VulkanRenderer, gui_shared: gui::GuiShared) -> Result<Self> {
        let (command_tx, command_rx) = bounded(128);
        let (gui_data_tx, gui_data_rx) = bounded(128);

        let render_gui_shared = gui_shared.clone();
        renderer.set_gui_sender(gui_data_tx.clone());
        let handle = thread::Builder::new()
            .name("render-thread".into())
            .spawn(move || render_loop(renderer, render_gui_shared, command_rx, gui_data_tx))
            .map_err(|err| anyhow::anyhow!("failed to spawn render thread: {err}"))?;

        Ok(Self {
            command_tx,
            gui_data_rx,
            handle: Some(handle),
        })
    }

    pub fn pause(&self) {
        let _ = self.command_tx.try_send(RenderCommand::Pause);
    }

    pub fn resume(&self) {
        let _ = self.command_tx.try_send(RenderCommand::Resume);
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
    Pause,
    Resume,
    Resize { width: u32, height: u32 },
    Present,
    Shutdown,
    BackendCommand(PushRender),
}

fn render_loop(
    mut renderer: VulkanRenderer,
    gui_shared: gui::GuiShared,
    command_rx: Receiver<RenderCommand>,
    gui_data_tx: Sender<PushGui>,
) {
    let mut paused = false;
    let mut running = true;
    let mut available: VecDeque<usize> = (0..OFFSCREEN_FRAME_COUNT).collect();
    let mut in_flight: Vec<usize> = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
    let mut ready: VecDeque<usize> = VecDeque::with_capacity(OFFSCREEN_FRAME_COUNT);
    let mut current_frame: Option<usize> = None;

    while running {
        let mut present_requested = false;
        let mut pending_resize: Option<(u32, u32)> = None;
        let mut pending_backend_command: Option<PushRender> = None;

        loop {
            match command_rx.try_recv() {
                Ok(command) => match command {
                    RenderCommand::Pause => paused = true,
                    RenderCommand::Resume => paused = false,
                    RenderCommand::Resize { width, height } => {
                        pending_resize = Some((width, height));
                    }
                    RenderCommand::Present => present_requested = true,
                    RenderCommand::Shutdown => {
                        running = false;
                        break;
                    }
                    RenderCommand::BackendCommand(command) => {
                        pending_backend_command = Some(command);
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

        if let Some((width, height)) = pending_resize {
            renderer.handle_resize(width, height);
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
            // let fps = fps_counter.tick();
            let (compute_ms, present_ms) = renderer.last_timer_ms();
            // let _  = gui_data_tx.try_send(GuiRequest::Fps(fps));
            let _  = gui_data_tx.try_send(PushGui::PerfUpdate{compute_ms, present_ms});
        }

        if present_requested {
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

                if let Err(err) = unsafe { renderer.present_frame(frame_index, gui_frame, pending_backend_command) } {
                    error!("present error: {err:?}");
                    if is_new {
                        available.push_back(frame_index);
                    }
                } else if is_new {
                    if let Some(previous) = current_frame.replace(frame_index) {
                        available.push_back(previous);
                    }
                }
            }
        }

        if !paused {
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
