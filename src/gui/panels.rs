use std::path::PathBuf;

use crate::app::render_controller::RenderCommand;
use crate::app::shader_reload::ReloadRequest;
use crate::gui::components::perf_graph::draw_perf_graph;
use crate::gui::gui_data::{PushRender, RenderMode};
use anyhow::Result;

use super::GuiData;
use super::frontend::{PANEL_WIDTH_POINTS, panel_width_pixels};
use crossbeam_channel::Sender;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuiTheme {
    Dark,
    #[allow(dead_code)] // theme switcher UI not currently exposed
    Light,
}

pub struct GuiPanels {
    pub theme: GuiTheme,
    render_sender: Sender<RenderCommand>,
    reload_sender: Sender<ReloadRequest>,
}

impl GuiPanels {
    pub fn new(render_sender: Sender<RenderCommand>, reload_sender: Sender<ReloadRequest>) -> Self {
        Self {
            render_sender,
            reload_sender,
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

                    ui.separator();

                    ui.heading("Renderer");

                    if ui
                        .toggle_value(&mut gui_data.user_paused, "Pause rendering")
                        .changed()
                    {
                        let _ =
                            self.send_command(RenderCommand::SetUserPaused(gui_data.user_paused));
                    }
                    let paused_suffix = if gui_data.effective_paused {
                        " (paused)"
                    } else {
                        ""
                    };
                    ui.label(format!(
                        "Samples: {}{}",
                        gui_data.sample_count, paused_suffix
                    ));

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
                    self.draw_render_mode_controls(ui, gui_data);

                    ui.separator();
                    self.draw_shader_controls(ui, gui_data);

                    ui.separator();
                    ui.heading("Frame timing");
                    draw_perf_graph(ui, &gui_data.perf_history);

                    ui.separator();
                    ui.heading("Panel");
                    let width_px = panel_width_pixels(pixels_per_point);
                    ui.label(format!("Resolution: {} × {} px", width_px, panel_height));
                    ui.label(format!(
                        "Render: {} × {} px",
                        gui_data.render_width, gui_data.render_height
                    ));

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

    pub fn send_command(&self, command: RenderCommand) -> Result<()> {
        self.render_sender.try_send(command)?;
        Ok(())
    }

    fn draw_shader_controls(&self, ui: &mut egui::Ui, gui_data: &GuiData) {
        ui.heading("Shaders");

        if ui.button("Reload shaders").clicked() {
            let _ = self.reload_sender.try_send(ReloadRequest::Manual);
        }

        if let Some(error) = &gui_data.shader_error {
            egui::ScrollArea::vertical()
                .id_salt("shader_error")
                .max_height(120.0)
                .show(ui, |ui| {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                });
        }
    }

    fn draw_render_mode_controls(&self, ui: &mut egui::Ui, gui_data: &mut GuiData) {
        ui.heading("Render mode");

        let previous_mode = gui_data.render_mode;
        egui::ComboBox::from_label("Mode")
            .selected_text(gui_data.render_mode.label())
            .show_ui(ui, |ui| {
                for mode in RenderMode::ALL {
                    ui.selectable_value(&mut gui_data.render_mode, mode, mode.label());
                }
            });
        if gui_data.render_mode != previous_mode {
            let _ = self.send(PushRender::SetRenderMode(gui_data.render_mode));
        }

        if gui_data.render_mode != RenderMode::BvhHeatmap {
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
