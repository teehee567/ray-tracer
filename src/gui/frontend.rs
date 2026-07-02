use std::mem;
use std::sync::{Arc, RwLock};

use super::GuiData;
use crate::app::camera_controller::{CameraController, CameraInput};
use crate::app::render_controller::RenderCommand;
use crate::app::shader_reload::ReloadRequest;
use crate::gui::PushGui;
use crate::gui::panels::{GuiPanels, GuiTheme};
use crate::types::CameraBufferObject;
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use egui::epaint::ClippedPrimitive;
use egui::{self, Rect, ViewportId, pos2, vec2};
use glam::Vec2;
use winit::event::WindowEvent;
use winit::window::Window;

// egui points
pub const PANEL_WIDTH_POINTS: f32 = 320.0;

pub fn panel_width_pixels(scale_factor: f32) -> u32 {
    (PANEL_WIDTH_POINTS * scale_factor).round().max(0.0) as u32
}

#[derive(Default)]
pub struct GuiState {
    latest: Option<Arc<GuiFrame>>,
}

impl GuiState {
    fn update(&mut self, mut frame: GuiFrame) {
        if let Some(pending) = self.latest.take() {
            if let Ok(pending_frame) = Arc::try_unwrap(pending) {
                let mut merged = pending_frame.textures_delta;
                merged.append(mem::take(&mut frame.textures_delta));
                frame.textures_delta = merged;
            } else {
                // The render thread is already presenting the pending frame, so its
                // texture uploads will be consumed directly from that clone.
            }
        }
        self.latest = Some(Arc::new(frame));
    }

    pub fn take_latest(&mut self) -> Option<Arc<GuiFrame>> {
        self.latest.take()
    }
}

/// Thread-safe handle to the GUI state.
pub type GuiShared = Arc<RwLock<GuiState>>;

/// Create a new shared GUI state used to communicate with the render worker.
pub fn create_shared_state() -> GuiShared {
    Arc::new(RwLock::new(GuiState::default()))
}

/// GPU-ready GUI frame data.
pub struct GuiFrame {
    pub textures_delta: egui::TexturesDelta,
    pub clipped_primitives: Vec<ClippedPrimitive>,
    pub panel_width: u32,
    pub panel_height: u32,
    pub pixels_per_point: f32,
    pub generation: u64,
}

/// Front-end responsible for building egui input and rasterising frames.
pub struct GuiFrontend {
    ctx: egui::Context,
    state: egui_winit::State,
    shared: GuiShared,
    pixels_per_point: f32,
    panel_height: u32,
    generation: u64,
    gui_data_rx: Receiver<PushGui>,

    gui_data: GuiData,
    panels: GuiPanels,

    camera: CameraController,
    render_sender: Sender<RenderCommand>,
}

impl GuiFrontend {
    pub fn new(
        window: &Window,
        shared: GuiShared,
        gui_data_rx: Receiver<PushGui>,
        render_sender: Sender<RenderCommand>,
        reload_sender: Sender<ReloadRequest>,
        initial_camera: CameraBufferObject,
    ) -> Self {
        let ctx = egui::Context::default();
        let scale_factor = window.scale_factor() as f32;
        let size = window.inner_size();
        let state = egui_winit::State::new(
            ctx.clone(),
            ViewportId::ROOT,
            window,
            Some(scale_factor),
            None,
            None,
        );
        Self {
            ctx,
            state,
            shared,
            pixels_per_point: scale_factor,
            panel_height: size.height,
            generation: 0,
            gui_data_rx,

            gui_data: GuiData::new(),

            panels: GuiPanels::new(render_sender.clone(), reload_sender),

            camera: CameraController::from_camera(&initial_camera),
            render_sender,
        }
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) {
        let _ = self.state.on_window_event(window, event);
    }

    pub fn run_frame(&mut self, window: &Window) {
        self.poll_gui_data();
        self.apply_theme();

        let size = window.inner_size();
        self.panel_height = size.height;
        self.pixels_per_point = window.scale_factor() as f32;

        let width_points = PANEL_WIDTH_POINTS;
        let height_points = if self.pixels_per_point <= f32::EPSILON {
            self.panel_height as f32
        } else {
            self.panel_height as f32 / self.pixels_per_point
        };

        let mut raw_input = self.state.take_egui_input(window);
        raw_input.screen_rect = Some(Rect::from_min_size(
            pos2(0.0, 0.0),
            vec2(width_points, height_points.max(0.0)),
        ));

        let theme = self.panels.theme;
        let panel_height = self.panel_height;
        let pixels_per_point = self.pixels_per_point;

        let full_output = self.ctx.run_ui(raw_input, |ui| {
            self.panels
                .draw(ui, &mut self.gui_data, panel_height, pixels_per_point);
        });

        self.state
            .handle_platform_output(window, full_output.platform_output);

        if self.panels.theme != theme {
            self.apply_theme();
        }

        let clipped = self
            .ctx
            .tessellate(full_output.shapes, self.pixels_per_point);

        let panel_width_px = panel_width_pixels(self.pixels_per_point);
        let height_px = self.panel_height.max(1);

        self.generation += 1;
        let frame = GuiFrame {
            textures_delta: full_output.textures_delta,
            clipped_primitives: clipped,
            panel_width: panel_width_px,
            panel_height: height_px,
            pixels_per_point: self.pixels_per_point,
            generation: self.generation,
        };

        if let Ok(mut state) = self.shared.write() {
            state.update(frame);
        }

        self.update_camera();
    }

    // sampe from egui
    fn update_camera(&mut self) {
        let wants_keyboard = self.ctx.egui_wants_keyboard_input();
        let wants_pointer = self.ctx.egui_wants_pointer_input();

        let (input, dt) = self.ctx.input(|i| {
            let kb = !wants_keyboard;
            let looking = i.pointer.secondary_down() && !wants_pointer;
            let look_delta = if looking {
                let d = i.pointer.delta();
                Vec2::new(d.x, d.y)
            } else {
                Vec2::ZERO
            };
            let input = CameraInput {
                forward: kb && i.key_down(egui::Key::W),
                back: kb && i.key_down(egui::Key::S),
                left: kb && i.key_down(egui::Key::A),
                right: kb && i.key_down(egui::Key::D),
                up: kb && i.key_down(egui::Key::E),
                down: kb && i.key_down(egui::Key::Q),
                look_delta,
                scroll: i.smooth_scroll_delta.y,
            };
            (input, i.stable_dt)
        });

        if let Some((location, rotation)) = self.camera.update(input, dt) {
            let _ = self
                .render_sender
                .try_send(RenderCommand::SetCamera { location, rotation });
        }
    }

    fn poll_gui_data(&mut self) {
        loop {
            match self.gui_data_rx.try_recv() {
                Ok(req) => match req {
                    PushGui::Status { samples, paused } => {
                        self.gui_data.sample_count = samples;
                        self.gui_data.effective_paused = paused;
                    }
                    PushGui::ShaderReload { error } => {
                        self.gui_data.shader_error = error;
                    }
                    PushGui::PerfUpdate {
                        compute_fps,
                        compute_ms,
                        present_fps,
                        present_ms,
                        heatmap_ms,
                        compositor_ms,
                    } => {
                        self.gui_data.compute_fps = compute_fps;
                        self.gui_data.compute_ms = compute_ms;
                        self.gui_data.present_fps = present_fps;
                        self.gui_data.present_ms = present_ms;
                        self.gui_data.heatmap_ms = heatmap_ms;
                        self.gui_data.compositor_ms = compositor_ms;
                        self.gui_data.perf_history.push(
                            compute_ms as f32,
                            present_ms as f32,
                            heatmap_ms as f32,
                            compositor_ms as f32,
                        );
                    }
                    PushGui::HeatmapInfo { max_depth } => {
                        self.gui_data.heatmap_max_depth = max_depth;
                        self.gui_data.heatmap_depth_low = 0;
                        self.gui_data.heatmap_depth_high = max_depth;
                    }
                    PushGui::RenderResolution { width, height } => {
                        self.gui_data.render_width = width;
                        self.gui_data.render_height = height;
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    fn apply_theme(&self) {
        match self.panels.theme {
            GuiTheme::Dark => self.ctx.set_visuals(egui::Visuals::dark()),
            GuiTheme::Light => self.ctx.set_visuals(egui::Visuals::light()),
        }
    }
}
