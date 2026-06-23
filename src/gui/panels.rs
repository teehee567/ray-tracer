use std::path::PathBuf;

use crate::app::render_controller::RenderCommand;
use crate::gui::components::perf_graph::draw_perf_graph;
use crate::gui::gui_data::PushRender;
use anyhow::Result;

use super::frontend::{panel_width_pixels, PANEL_WIDTH_POINTS};
use super::GuiData;
use crossbeam_channel::Sender;
use egui::{self, ComboBox};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuiTheme {
    Dark,
    Light,
}

pub struct GuiPanels {
    pub theme: GuiTheme,
    render_sender: Sender<RenderCommand>,
}

impl GuiPanels {
    pub fn new(render_sender: Sender<RenderCommand>) -> Self {
        Self {
            render_sender,
            theme: GuiTheme::Dark,
        }
    }

    pub fn draw(
        &mut self,
        root_ui: &mut egui::Ui,
        gui_data: &mut GuiData,
        panel_height: u32,
        pixels_per_point: f32,
    ) {
        egui::Panel::left("control_panel")
            .resizable(false)
            .exact_size(PANEL_WIDTH_POINTS)
            .show_inside(root_ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Ray Tracer");
                    ui.label(format!("v{}", env!("CARGO_PKG_VERSION")));

                    // ui.separator();
                    // ui.horizontal(|ui| {
                    //     ComboBox::from_label("Theme")
                    //         .selected_text(match self.theme {
                    //             GuiTheme::Dark => "Dark",
                    //             GuiTheme::Light => "Light",
                    //         })
                    //         .show_ui(ui, |ui| {
                    //             ui.selectable_value(&mut self.theme, GuiTheme::Dark, "Dark");
                    //             ui.selectable_value(&mut self.theme, GuiTheme::Light, "Light");
                    //         });
                    // });
                    
                    ui.separator();

                    ui.heading("Renderer");

                    // ui.label(format!("FPS: {:.2}", gui_data.fps));
                    // let frame_ms = if gui_data.fps > f64::EPSILON {
                    //     1000.0 / gui_data.fps
                    // } else {
                    //     f64::INFINITY
                    // };
                    // if frame_ms.is_finite() {
                    //     ui.label(format!("Frame time: {:.2} ms", frame_ms));
                    // } else {
                    //     ui.label("Frame time: ∞");
                    // }

                    ui.label(format!("Compute: {:.2}", gui_data.compute_fps));
                    ui.label(format!("Compute (GPU): {:.2} ms", gui_data.compute_ms));

                    ui.label(format!("Present FPS: {:.2}", gui_data.present_fps));
                    ui.label(format!("Present: {:.2} ms", gui_data.present_ms));

                    ui.label(format!("Heatmap (GPU): {:.2} ms", gui_data.heatmap_ms));
                    ui.label(format!(
                        "Compositor (GPU): {:.2} ms",
                        gui_data.compositor_ms
                    ));

                    ui.separator();
                    self.draw_heatmap_controls(ui, gui_data);

                    ui.separator();
                    ui.heading("Frame timing");
                    draw_perf_graph(ui, &gui_data.perf_history);

                    ui.separator();
                    ui.heading("Panel");
                    let width_px = panel_width_pixels(pixels_per_point);
                    ui.label(format!("Resolution: {} × {} px", width_px, panel_height));

                    ui.text_edit_singleline(&mut gui_data.save_file_path);

                    if ui.button("Save Current Frame").clicked() {
                        let _ = self.send(PushRender::SaveFrame(PathBuf::from(
                            &gui_data.save_file_path,
                        )));
                    }
                });
            });

        egui::CentralPanel::default().show_inside(root_ui, |_| {});
    }

    pub fn send(&self, req: PushRender) -> Result<()> {
        self.render_sender
            .try_send(RenderCommand::BackendCommand(req))?;
        Ok(())
    }

    fn draw_heatmap_controls(&self, ui: &mut egui::Ui, gui_data: &mut GuiData) {
        ui.heading("Heatmap");

        if ui
            .checkbox(&mut gui_data.heatmap_enabled, "Enabled")
            .changed()
        {
            let _ = self.send(PushRender::ToggleHeatmap(gui_data.heatmap_enabled));
        }

        if !gui_data.heatmap_enabled {
            return;
        }

        if Self::draw_heatmap_depth_band(ui, gui_data) {
            let _ = self.send(PushRender::SetHeatmapBand {
                low: gui_data.heatmap_depth_low,
                high: gui_data.heatmap_depth_high,
            });
        }
    }

    fn draw_heatmap_depth_band(ui: &mut egui::Ui, gui_data: &mut GuiData) -> bool {
        let max_depth = gui_data.heatmap_max_depth;
        let mut changed = false;

        if Self::heatmap_depth_slider(ui, "Min depth", &mut gui_data.heatmap_depth_low, max_depth) {
            gui_data.heatmap_depth_high =
                gui_data.heatmap_depth_high.max(gui_data.heatmap_depth_low);
            changed = true;
        }

        if Self::heatmap_depth_slider(ui, "Max depth", &mut gui_data.heatmap_depth_high, max_depth)
        {
            gui_data.heatmap_depth_low =
                gui_data.heatmap_depth_low.min(gui_data.heatmap_depth_high);
            changed = true;
        }

        changed
    }

    fn heatmap_depth_slider(
        ui: &mut egui::Ui,
        label: &str,
        value: &mut u32,
        max_depth: u32,
    ) -> bool {
        ui.add(egui::Slider::new(value, 0..=max_depth).text(label))
            .changed()
    }
}
