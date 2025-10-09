use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::RenderMetrics;
use crossbeam_channel::{Receiver, TryRecvError};
use eframe::egui::epaint::{ClippedPrimitive, ColorImage, ImageData, Mesh, Primitive, Vertex};
use eframe::egui::{
    self, Color32, ComboBox, Event, Key, Modifiers, MouseWheelUnit, PointerButton, Pos2, Rect,
    TextureId, Vec2, ViewportId, ViewportInfo, pos2, vec2,
};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::ModifiersState;
use winit::keyboard::{Key as WinitKey, NamedKey};
use winit::window::Window;

/// Logical width of the GUI side panel in egui points.
const PANEL_WIDTH_POINTS: f32 = 320.0;

/// Convert the logical panel width into physical pixels for a given scale factor.
pub fn panel_width_pixels(scale_factor: f32) -> u32 {
    (PANEL_WIDTH_POINTS * scale_factor).round().max(0.0) as u32
}

/// Shared GUI state accessible by the render worker.
#[derive(Default)]
pub struct GuiState {
    latest: Option<GuiFrame>,
}

impl GuiState {
    fn update(&mut self, frame: GuiFrame) {
        self.latest = Some(frame);
    }

    pub fn latest(&self) -> Option<GuiFrame> {
        self.latest.clone()
    }
}

/// Thread-safe handle to the GUI state.
pub type GuiShared = Arc<RwLock<GuiState>>;

/// Create a new shared GUI state used to communicate with the render worker.
pub fn create_shared_state() -> GuiShared {
    Arc::new(RwLock::new(GuiState::default()))
}

/// CPU-rasterised GUI frame.
#[derive(Clone)]
pub struct GuiFrame {
    pub pixels: Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    pub generation: u64,
}

#[derive(Clone)]
struct GuiTexture {
    size: [usize; 2],
    pixels: Vec<Color32>,
}

struct SoftwarePainter {
    textures: HashMap<TextureId, GuiTexture>,
}

impl SoftwarePainter {
    fn new() -> Self {
        Self {
            textures: HashMap::new(),
        }
    }

    fn update_textures(&mut self, delta: &egui::TexturesDelta) {
        for (id, image_delta) in &delta.set {
            match &image_delta.image {
                ImageData::Color(color) => {
                    let mut texture = GuiTexture {
                        size: [color.size[0], color.size[1]],
                        pixels: color.pixels.clone(),
                    };
                    if let Some([x, y]) = image_delta.pos {
                        self.sub_update_texture(id, &mut texture, x, y);
                    } else {
                        self.textures.insert(*id, texture);
                    }
                }
            }
        }
        for id in &delta.free {
            self.textures.remove(id);
        }
    }

    fn sub_update_texture(&mut self, id: &TextureId, texture: &mut GuiTexture, x: usize, y: usize) {
        if let Some(existing) = self.textures.get_mut(id) {
            let width = texture.size[0];
            let height = texture.size[1];
            for row in 0..height {
                let dst_index = (row + y) * existing.size[0] + x;
                let src_index = row * width;
                let len = width.min(existing.size[0].saturating_sub(x));
                if len == 0 {
                    continue;
                }
                existing.pixels[dst_index..dst_index + len]
                    .copy_from_slice(&texture.pixels[src_index..src_index + len]);
            }
        } else {
            self.textures.insert(*id, texture.clone());
        }
    }

    fn paint(
        &self,
        width: u32,
        height: u32,
        pixels_per_point: f32,
        background: Color32,
        primitives: &[ClippedPrimitive],
    ) -> Vec<u8> {
        let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];
        fill_background(&mut pixels, width, height, background);

        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in primitives
        {
            let mesh = match primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => continue,
            };

            let clip_min_x = (clip_rect.min.x * pixels_per_point).floor() as i32;
            let clip_min_y = (clip_rect.min.y * pixels_per_point).floor() as i32;
            let clip_max_x = (clip_rect.max.x * pixels_per_point).ceil() as i32;
            let clip_max_y = (clip_rect.max.y * pixels_per_point).ceil() as i32;

            self.paint_mesh(
                mesh,
                width as i32,
                height as i32,
                clip_min_x.clamp(0, width as i32),
                clip_max_x.clamp(0, width as i32),
                clip_min_y.clamp(0, height as i32),
                clip_max_y.clamp(0, height as i32),
                pixels_per_point,
                &mut pixels,
            );
        }

        pixels
    }

    fn paint_mesh(
        &self,
        mesh: &Mesh,
        width: i32,
        height: i32,
        clip_min_x: i32,
        clip_max_x: i32,
        clip_min_y: i32,
        clip_max_y: i32,
        pixels_per_point: f32,
        pixels: &mut [u8],
    ) {
        if mesh.indices.len() < 3 {
            return;
        }

        for tri in mesh.indices.chunks_exact(3) {
            let v0 = &mesh.vertices[tri[0] as usize];
            let v1 = &mesh.vertices[tri[1] as usize];
            let v2 = &mesh.vertices[tri[2] as usize];

            self.rasterize_triangle(
                mesh.texture_id,
                v0,
                v1,
                v2,
                width,
                height,
                clip_min_x,
                clip_max_x,
                clip_min_y,
                clip_max_y,
                pixels_per_point,
                pixels,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rasterize_triangle(
        &self,
        texture_id: TextureId,
        v0: &Vertex,
        v1: &Vertex,
        v2: &Vertex,
        width: i32,
        height: i32,
        clip_min_x: i32,
        clip_max_x: i32,
        clip_min_y: i32,
        clip_max_y: i32,
        pixels_per_point: f32,
        pixels: &mut [u8],
    ) {
        let p0 = v0.pos * pixels_per_point;
        let p1 = v1.pos * pixels_per_point;
        let p2 = v2.pos * pixels_per_point;

        let min_x = clip_min_x.max(f32::floor(p0.x.min(p1.x).min(p2.x)) as i32 - 1);
        let min_y = clip_min_y.max(f32::floor(p0.y.min(p1.y).min(p2.y)) as i32 - 1);
        let max_x = clip_max_x.min(f32::ceil(p0.x.max(p1.x).max(p2.x)) as i32 + 1);
        let max_y = clip_max_y.min(f32::ceil(p0.y.max(p1.y).max(p2.y)) as i32 + 1);

        if min_x >= max_x || min_y >= max_y {
            return;
        }

        let area = edge_function(p0, p1, p2);
        if area.abs() < f32::EPSILON {
            return;
        }
        let inv_area = 1.0 / area;

        let texture = self.textures.get(&texture_id);

        for y in min_y..max_y {
            for x in min_x..max_x {
                if x < 0 || x >= width || y < 0 || y >= height {
                    continue;
                }

                let sample = Pos2::new(x as f32 + 0.5, y as f32 + 0.5);
                let w0 = edge_function(p1, p2, sample) * inv_area;
                let w1 = edge_function(p2, p0, sample) * inv_area;
                let w2 = edge_function(p0, p1, sample) * inv_area;

                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }

                let mut color = blend_vertex_colors(v0, v1, v2, w0, w1, w2);

                if let Some(tex) = texture {
                    let uv =
                        (v0.uv.to_vec2() * w0) + (v1.uv.to_vec2() * w1) + (v2.uv.to_vec2() * w2);
                    let tex_color = sample_texture(tex, uv);
                    color = multiply_colors(color, tex_color);
                }

                let offset = ((y as usize) * (width as usize) + x as usize) * 4;
                let dst = &mut pixels[offset..offset + 4];
                let dst_color = Color32::from_rgba_unmultiplied(dst[0], dst[1], dst[2], dst[3]);
                let blended = alpha_blend(dst_color, color);
                dst.copy_from_slice(&blended.to_array());
            }
        }
    }
}

fn edge_function(a: Pos2, b: Pos2, c: Pos2) -> f32 {
    (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)
}

fn blend_vertex_colors(
    v0: &Vertex,
    v1: &Vertex,
    v2: &Vertex,
    w0: f32,
    w1: f32,
    w2: f32,
) -> Color32 {
    let mut to_vec4 = |color: Color32| -> [f32; 4] {
        let [r, g, b, a] = color.to_array();
        [
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
            a as f32 / 255.0,
        ]
    };

    let c0 = to_vec4(v0.color);
    let c1 = to_vec4(v1.color);
    let c2 = to_vec4(v2.color);

    let r = c0[0] * w0 + c1[0] * w1 + c2[0] * w2;
    let g = c0[1] * w0 + c1[1] * w1 + c2[1] * w2;
    let b = c0[2] * w0 + c1[2] * w1 + c2[2] * w2;
    let a = c0[3] * w0 + c1[3] * w1 + c2[3] * w2;

    Color32::from_rgba_unmultiplied(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
        (a * 255.0) as u8,
    )
}

fn multiply_colors(a: Color32, b: Color32) -> Color32 {
    let [ar, ag, ab, aa] = a.to_array();
    let [br, bg, bb, ba] = b.to_array();
    let mut mul = |ac: u8, bc: u8| -> u8 { (((ac as u16) * (bc as u16)) / 255) as u8 };
    Color32::from_rgba_unmultiplied(mul(ar, br), mul(ag, bg), mul(ab, bb), mul(aa, ba))
}

fn alpha_blend(dst: Color32, src: Color32) -> Color32 {
    let [dr, dg, db, da] = dst.to_array();
    let [sr, sg, sb, sa] = src.to_array();
    let src_a = sa as f32 / 255.0;
    let dst_a = da as f32 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);

    if out_a <= f32::EPSILON {
        return Color32::TRANSPARENT;
    }

    let blend = |dc: u8, sc: u8| -> u8 {
        let dc = dc as f32 / 255.0;
        let sc = sc as f32 / 255.0;
        let out = (sc * src_a + dc * dst_a * (1.0 - src_a)) / out_a;
        (out * 255.0).round().clamp(0.0, 255.0) as u8
    };

    Color32::from_rgba_unmultiplied(
        blend(dr, sr),
        blend(dg, sg),
        blend(db, sb),
        (out_a * 255.0).round().clamp(0.0, 255.0) as u8,
    )
}

fn sample_texture(texture: &GuiTexture, uv: egui::Vec2) -> Color32 {
    if texture.size[0] == 0 || texture.size[1] == 0 {
        return Color32::TRANSPARENT;
    }
    let u = uv.x.clamp(0.0, 1.0) * (texture.size[0] as f32 - 1.0);
    let v = uv.y.clamp(0.0, 1.0) * (texture.size[1] as f32 - 1.0);
    let x = u.round() as usize;
    let y = v.round() as usize;
    texture.pixels[y * texture.size[0] + x]
}

/// Front-end responsible for building egui input and rasterising frames.
pub struct GuiFrontend {
    ctx: egui::Context,
    painter: SoftwarePainter,
    shared: GuiShared,
    raw_events: Vec<Event>,
    modifiers: Modifiers,
    pointer_pos: Option<Pos2>,
    pointer_inside: bool,
    has_focus: bool,
    pixels_per_point: f32,
    panel_height: u32,
    generation: u64,
    metrics_rx: Receiver<RenderMetrics>,
    latest_metrics: Option<RenderMetrics>,
    theme: GuiTheme,
}

impl GuiFrontend {
    pub fn new(window: &Window, shared: GuiShared, metrics_rx: Receiver<RenderMetrics>) -> Self {
        let ctx = egui::Context::default();
        let scale_factor = window.scale_factor() as f32;
        let size = window.inner_size();
        Self {
            ctx,
            painter: SoftwarePainter::new(),
            shared,
            raw_events: Vec::new(),
            modifiers: Modifiers::default(),
            pointer_pos: None,
            pointer_inside: false,
            has_focus: true,
            pixels_per_point: scale_factor as f32,
            panel_height: size.height,
            generation: 0,
            metrics_rx,
            latest_metrics: None,
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
        self.poll_metrics();
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
        let metrics = self.latest_metrics;
        let panel_height = self.panel_height;
        let pixels_per_point = self.pixels_per_point;

        let full_output = self.ctx.run(raw_input, |ctx| {
            draw_panels(ctx, &mut theme, metrics, panel_height, pixels_per_point);
        });

        if self.theme != theme {
            self.theme = theme;
            self.apply_theme();
        }

        self.painter.update_textures(&full_output.textures_delta);
        let clipped = self
            .ctx
            .tessellate(full_output.shapes, self.pixels_per_point);

        let panel_width_px = panel_width_pixels(self.pixels_per_point);
        let height_px = self.panel_height.max(1);

        let background = self.ctx.style().visuals.panel_fill;
        let pixels = self.painter.paint(
            panel_width_px,
            height_px,
            self.pixels_per_point,
            background,
            &clipped,
        );

        self.generation += 1;
        let frame = GuiFrame {
            pixels: Arc::new(pixels),
            width: panel_width_px,
            height: height_px,
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

    fn poll_metrics(&mut self) {
        loop {
            match self.metrics_rx.try_recv() {
                Ok(metrics) => self.latest_metrics = Some(metrics),
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

fn fill_background(pixels: &mut [u8], width: u32, height: u32, color: Color32) {
    let [r, g, b, a] = color.to_array();
    for y in 0..height {
        let row = y as usize * width as usize * 4;
        for x in 0..width as usize {
            let idx = row + x * 4;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = a;
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
    metrics: Option<RenderMetrics>,
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
            if let Some(metrics) = metrics {
                ui.label(format!("FPS: {:.1}", metrics.fps));
                let frame_ms = if metrics.fps > f64::EPSILON {
                    1000.0 / metrics.fps
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
