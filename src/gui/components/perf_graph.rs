use egui::Color32;
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::gui::PerfHistory;

// blue line
const FRAME_COLOR: Color32 = Color32::from_rgb(70, 150, 255);
// pink line
const GPU_COLOR: Color32 = Color32::from_rgb(255, 105, 180);

pub const PERF_HISTORY_LEN: usize = 200;

// perf graph line graph of last render times, cpu and gpu times separate
pub fn draw_perf_graph(ui: &mut egui::Ui, history: &PerfHistory) {
    let frame_points: PlotPoints = history
        .frame_ms
        .iter()
        .enumerate()
        .map(|(i, &v)| [i as f64, v as f64])
        .collect();
    let gpu_points: PlotPoints = history
        .compute_ms
        .iter()
        .enumerate()
        .map(|(i, &v)| [i as f64, v as f64])
        .collect();

    let (y_min, y_max) = perf_y_bounds(history);

    Plot::new("frame_timing_plot")
        .height(140.0)
        .legend(Legend::default())
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .show_x(false)
        .y_axis_label("ms")
        .show(ui, |plot_ui| {
            plot_ui.set_plot_bounds_y(y_min..=y_max);
            plot_ui.line(
                Line::new("Frame (CPU+GPU)", frame_points)
                    .color(FRAME_COLOR)
                    .width(1.5),
            );
            plot_ui.line(
                Line::new("GPU compute", gpu_points)
                    .color(GPU_COLOR)
                    .width(1.5),
            );
        });
}

// bound hte graph so it looks nice
fn perf_y_bounds(history: &PerfHistory) -> (f64, f64) {
    let mut lowest = f32::INFINITY;
    let mut highest = f32::NEG_INFINITY;
    for &v in history.frame_ms.iter().chain(&history.compute_ms) {
        lowest = lowest.min(v);
        highest = highest.max(v);
    }
    if !lowest.is_finite() {
        return (0.0, 10.0);
    }

    let bottom = (lowest as f64 / 10.0).floor() * 10.0;
    let top = (highest as f64 / 10.0).ceil() * 10.0;
    (bottom, top.max(bottom + 10.0))
}