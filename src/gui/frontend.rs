use std::sync::{Arc, RwLock};

use super::GuiData;
use crossbeam_channel::{Receiver, TryRecvError};
use eframe::egui::epaint::ClippedPrimitive;
use eframe::egui::{
    self, ComboBox, Event, Key, Modifiers, MouseWheelUnit, PointerButton, Pos2, Rect, Vec2,
    ViewportId, ViewportInfo, pos2, vec2,
};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::ModifiersState;
use winit::keyboard::{Key as WinitKey, NamedKey};
use winit::window::Window;

// egui points
const PANEL_WIDTH_POINTS: f32 = 320.0;

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
            if let Ok(mut pending_frame) = Arc::try_unwrap(pending) {
                frame.textures_delta.append(pending_frame.textures_delta);
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
    shared: GuiShared,
    raw_events: Vec<Event>,
    modifiers: Modifiers,
    pointer_pos: Option<Pos2>,
    pointer_inside: bool,
    has_focus: bool,
    pixels_per_point: f32,
    panel_height: u32,
    generation: u64,
    gui_data_rx: Receiver<GuiData>,
    latest_gui_data: Option<GuiData>,
    theme: GuiTheme,
}

impl GuiFrontend {
    pub fn new(window: &Window, shared: GuiShared, gui_data_rx: Receiver<GuiData>) -> Self {
        let ctx = egui::Context::default();
        let scale_factor = window.scale_factor() as f32;
        let size = window.inner_size();
        Self {
            ctx,
            shared,
            raw_events: Vec::new(),
            modifiers: Modifiers::default(),
            pointer_pos: None,
            pointer_inside: false,
            has_focus: true,
            pixels_per_point: scale_factor as f32,
            panel_height: size.height,
            generation: 0,
            gui_data_rx,
            latest_gui_data: None,
            theme: GuiTheme::Dark,
        }
    }

    pub fn handle_event(&mut self, _window: &Window, event: &WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                self.panel_height = size.height;
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.pixels_per_point = *scale_factor as f32;
            }
            WindowEvent::Focused(focused) => {
                self.has_focus = *focused;
                if !focused {
                    self.pointer_inside = false;
                    self.pointer_pos = None;
                    self.raw_events.push(Event::PointerGone);
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.pointer_inside = false;
                self.pointer_pos = None;
                self.raw_events.push(Event::PointerGone);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_moved(*position);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if self.pointer_inside {
                    if let Some(event) = self.translate_mouse_button(*button, *state) {
                        self.raw_events.push(event);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.pointer_inside {
                    match delta {
                        MouseScrollDelta::LineDelta(x, y) => {
                            self.raw_events.push(Event::MouseWheel {
                                unit: MouseWheelUnit::Line,
                                delta: vec2(*x, *y),
                                modifiers: self.modifiers,
                            });
                        }
                        MouseScrollDelta::PixelDelta(pos) => {
                            self.raw_events.push(Event::MouseWheel {
                                unit: MouseWheelUnit::Point,
                                delta: vec2(pos.x as f32, pos.y as f32),
                                modifiers: self.modifiers,
                            });
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(key_event) = self.translate_key_event(event) {
                    self.raw_events.push(key_event);
                }
            }
            WindowEvent::Ime(ime) => {
                if let winit::event::Ime::Commit(text) = ime {
                    if !text.is_empty() {
                        self.raw_events.push(Event::Text(text.clone()));
                    }
                }
            }
            _ => {}
        }
    }

    pub fn run_frame(&mut self, _window: &Window) {
        self.poll_gui_data();
        self.apply_theme();

        let width_points = PANEL_WIDTH_POINTS;
        let height_points = if self.pixels_per_point <= f32::EPSILON {
            self.panel_height as f32
        } else {
            self.panel_height as f32 / self.pixels_per_point
        };

        let mut raw_input = egui::RawInput::default();
        raw_input.screen_rect = Some(Rect::from_min_size(
            pos2(0.0, 0.0),
            vec2(width_points, height_points.max(0.0)),
        ));
        raw_input.viewports.insert(
            ViewportId::ROOT,
            ViewportInfo {
                native_pixels_per_point: Some(self.pixels_per_point),
                ..Default::default()
            },
        );
        raw_input.modifiers = self.modifiers;
        raw_input.events.extend(self.raw_events.drain(..));
        raw_input.focused = self.has_focus;

        if let Some(pos) = self.pointer_pos {
            raw_input.events.push(Event::PointerMoved(pos));
        } else {
            raw_input.events.push(Event::PointerGone);
        }

        let mut theme = self.theme;
        let gui_data = self.latest_gui_data;
        let panel_height = self.panel_height;
        let pixels_per_point = self.pixels_per_point;

        let full_output = self.ctx.run(raw_input, |ctx| {
            draw_panels(ctx, &mut theme, gui_data, panel_height, pixels_per_point);
        });

        if self.theme != theme {
            self.theme = theme;
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
    }

    fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        let panel_width = panel_width_pixels(self.pixels_per_point) as f64;
        if position.x <= panel_width {
            let pos = Pos2::new(
                position.x as f32 / self.pixels_per_point,
                position.y as f32 / self.pixels_per_point,
            );
            self.pointer_pos = Some(pos);
            self.pointer_inside = true;
        } else if self.pointer_inside {
            self.pointer_inside = false;
            self.pointer_pos = None;
            self.raw_events.push(Event::PointerGone);
        }
    }

    fn translate_mouse_button(&self, button: MouseButton, state: ElementState) -> Option<Event> {
        let pressed = matches!(state, ElementState::Pressed);
        let pointer_button = match button {
            MouseButton::Left => PointerButton::Primary,
            MouseButton::Right => PointerButton::Secondary,
            MouseButton::Middle => PointerButton::Middle,
            MouseButton::Back => PointerButton::Extra1,
            MouseButton::Forward => PointerButton::Extra2,
            _ => return None,
        };

        let pos = self.pointer_pos?;
        Some(Event::PointerButton {
            pos,
            button: pointer_button,
            pressed,
            modifiers: self.modifiers,
        })
    }

    fn translate_key_event(&self, event: &KeyEvent) -> Option<Event> {
        let logical = &event.logical_key;
        let key = match logical {
            WinitKey::Named(named) => match named {
                NamedKey::ArrowDown => Key::ArrowDown,
                NamedKey::ArrowLeft => Key::ArrowLeft,
                NamedKey::ArrowRight => Key::ArrowRight,
                NamedKey::ArrowUp => Key::ArrowUp,
                NamedKey::Backspace => Key::Backspace,
                NamedKey::Delete => Key::Delete,
                NamedKey::Enter => Key::Enter,
                NamedKey::Escape => Key::Escape,
                NamedKey::Tab => Key::Tab,
                NamedKey::PageDown => Key::PageDown,
                NamedKey::PageUp => Key::PageUp,
                NamedKey::Home => Key::Home,
                NamedKey::End => Key::End,
                NamedKey::Space => Key::Space,
                NamedKey::Insert => Key::Insert,
                _ => return None,
            },
            WinitKey::Character(ch) => {
                let c = ch.chars().next()?;
                match c.to_ascii_lowercase() {
                    'a' => Key::A,
                    'c' => Key::C,
                    'v' => Key::V,
                    'x' => Key::X,
                    'y' => Key::Y,
                    'z' => Key::Z,
                    '0' => Key::Num0,
                    '1' => Key::Num1,
                    '2' => Key::Num2,
                    '3' => Key::Num3,
                    '4' => Key::Num4,
                    '5' => Key::Num5,
                    '6' => Key::Num6,
                    '7' => Key::Num7,
                    '8' => Key::Num8,
                    '9' => Key::Num9,
                    _ => return None,
                }
            }
            _ => return None,
        };

        let pressed = matches!(event.state, ElementState::Pressed);
        Some(Event::Key {
            key,
            physical_key: None,
            pressed,
            repeat: event.repeat,
            modifiers: self.modifiers,
        })
    }

    fn poll_gui_data(&mut self) {
        loop {
            match self.gui_data_rx.try_recv() {
                Ok(gui_data) => self.latest_gui_data = Some(gui_data),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }

    fn apply_theme(&self) {
        match self.theme {
            GuiTheme::Dark => self.ctx.set_visuals(egui::Visuals::dark()),
            GuiTheme::Light => self.ctx.set_visuals(egui::Visuals::light()),
        }
    }
}

fn is_printable(ch: char) -> bool {
    !ch.is_control()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GuiTheme {
    Dark,
    Light,
}

fn draw_panels(
    ctx: &egui::Context,
    theme: &mut GuiTheme,
    gui_data: Option<GuiData>,
    panel_height: u32,
    pixels_per_point: f32,
) {
    egui::SidePanel::left("control_panel")
        .resizable(false)
        .exact_width(PANEL_WIDTH_POINTS)
        .show(ctx, |ui| {
            ui.heading("Ray Tracer");
            ui.label(format!("v{}", env!("CARGO_PKG_VERSION")));
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Theme");
                ComboBox::from_id_source("theme_combo")
                    .selected_text(match theme {
                        GuiTheme::Dark => "Dark",
                        GuiTheme::Light => "Light",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(theme, GuiTheme::Dark, "Dark");
                        ui.selectable_value(theme, GuiTheme::Light, "Light");
                    });
            });
            ui.separator();

            ui.heading("Renderer");
            if let Some(gui_data) = gui_data {
                ui.label(format!("FPS: {:.1}", gui_data.fps));
                let frame_ms = if gui_data.fps > f64::EPSILON {
                    1000.0 / gui_data.fps
                } else {
                    f64::INFINITY
                };
                if frame_ms.is_finite() {
                    ui.label(format!("Frame time: {:.2} ms", frame_ms));
                } else {
                    ui.label("Frame time: ∞");
                }
            } else {
                ui.label("Waiting for renderer…");
            }

            ui.separator();
            ui.heading("Panel");
            let width_px = panel_width_pixels(pixels_per_point);
            ui.label(format!("Resolution: {} × {} px", width_px, panel_height));
        });

    egui::CentralPanel::default().show(ctx, |_| {});
}
