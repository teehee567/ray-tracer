use std::collections::VecDeque;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use log::error;

use crate::ui;

use super::{App, OFFSCREEN_FRAME_COUNT, RenderMetrics};
use vulkanalia::vk::DeviceV1_0;

pub struct RenderController {
    command_tx: Sender<RenderCommand>,
    metrics_rx: Receiver<RenderMetrics>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RenderController {
    pub fn spawn(app: App, gui_shared: ui::GuiShared) -> Result<Self> {
        let (command_tx, command_rx) = bounded(16);
        let (metrics_tx, metrics_rx) = bounded(32);

        let render_gui_shared = gui_shared.clone();
        let handle = thread::Builder::new()
            .name("render-thread".into())
            .spawn(move || render_loop(app, render_gui_shared, command_rx, metrics_tx))
            .map_err(|err| anyhow::anyhow!("failed to spawn render thread: {err}"))?;

        Ok(Self {
            command_tx,
            metrics_rx,
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

    pub fn present(&self) {
        let _ = self.command_tx.try_send(RenderCommand::Present);
    }

    pub fn shutdown(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.command_tx.send(RenderCommand::Shutdown);
            let _ = handle.join();
        }
    }

    pub fn metrics_receiver(&self) -> Receiver<RenderMetrics> {
        self.metrics_rx.clone()
    }

    pub fn command_sender(&self) -> Sender<RenderCommand> {
        self.command_tx.clone()
    }
}

impl Drop for RenderController {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RenderCommand {
    Pause,
    Resume,
    Resize { width: u32, height: u32 },
    Present,
    Shutdown,
}

fn render_loop(
    mut app: App,
    gui_shared: ui::GuiShared,
    command_rx: Receiver<RenderCommand>,
    metrics_tx: Sender<RenderMetrics>,
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
            app.handle_resize(width, height);
        }

        let mut completed = Vec::new();
        in_flight.retain(|index| {
            match unsafe { app.device.get_fence_status(app.frame_fences[*index]) } {
                Ok(_) => {
                    completed.push(*index);
                    false
                }
                Ok(_) => true,
                Err(err) => {
                    error!("fence status error: {err:?}");
                    true
                }
            }
        });

        for index in completed {
            ready.push_back(index);
            let fps = app.fps_counter.tick();
            let _ = metrics_tx.try_send(RenderMetrics { fps });
        }

        if present_requested {
            let gui_frame = if let Ok(mut state) = gui_shared.write() {
                state.take_latest()
            } else {
                None
            };
            if let Some(new_frame) = ready.pop_back() {
                while let Some(stale) = ready.pop_front() {
                    available.push_back(stale);
                }

                if let Err(err) = unsafe { app.present_frame(new_frame, gui_frame.clone()) } {
                    error!("present error: {err:?}");
                    available.push_back(new_frame);
                } else if let Some(previous) = current_frame.replace(new_frame) {
                    available.push_back(previous);
                }
            } else if let Some(current) = current_frame {
                if let Err(err) = unsafe { app.present_frame(current, gui_frame.clone()) } {
                    error!("present error: {err:?}");
                }
            }
        }

        if !paused {
            if let Some(index) = available.pop_front() {
                match unsafe { app.dispatch_compute(index) } {
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
        app.destroy();
    }
}
