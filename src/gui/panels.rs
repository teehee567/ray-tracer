use egui::{self, ComboBox};
use super::GuiData;
use super::frontend::{PANEL_WIDTH_POINTS, panel_width_pixels};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuiTheme {
    Dark,
    Light,
}

pub struct GuiPanels {
    pub theme: GuiTheme,
}

impl GuiPanels {
    pub fn new() -> Self {
        Self {
            theme: GuiTheme::Dark,
        }
    }

    pub fn draw(
        &mut self,
        ctx: &egui::Context,
        gui_data: Option<&GuiData>,
        panel_height: u32,
        pixels_per_point: f32,
        ui_fps: f64,
    ) {
        egui::SidePanel::left("control_panel")
            .resizable(false)
            .exact_width(PANEL_WIDTH_POINTS)
            .show(ctx, |ui| {
                ui.heading("Ray Tracer");
                ui.label(format!("v{}", env!("CARGO_PKG_VERSION")));
                ui.separator();
                ui.horizontal(|ui| {
                    ComboBox::from_label("Theme")
                        .selected_text(match self.theme {
                            GuiTheme::Dark => "Dark",
                            GuiTheme::Light => "Light",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.theme, GuiTheme::Dark, "Dark");
                            ui.selectable_value(&mut self.theme, GuiTheme::Light, "Light");
                        });
                });
                ui.separator();

                ui.heading("Renderer");
                if let Some(gui_data) = gui_data {
                    ui.label(format!("FPS: {:.2}", gui_data.fps));
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
                ui.heading("UI");
                ui.label(format!("UI FPS: {:.2}", ui_fps));

                ui.separator();
                ui.heading("Panel");
                let width_px = panel_width_pixels(pixels_per_point);
                ui.label(format!("Resolution: {} × {} px", width_px, panel_height));
            });

        egui::CentralPanel::default().show(ctx, |_| {});
    }
}
