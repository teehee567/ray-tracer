use egui::Color32;
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::gui::PerfHistory;

// blue line
const FRAME_COLOR: Color32 = Color32::from_rgb(70, 150, 255);
// pink line
const GPU_COLOR: Color32 = Color32::from_rgb(255, 105, 180);
// green line
const HEATMAP_COLOR: Color32 = Color32::from_rgb(110, 220, 130);
// amber line
const COMPOSITOR_COLOR: Color32 = Color32::from_rgb(240, 190, 70);

pub const PERF_HISTORY_LEN: usize = 200;

fn series_points(samples: &std::collections::VecDeque<f32>) -> PlotPoints<'static> {
    samples
        .iter()
        .enumerate()
        .map(|(i, &v)| [i as f64, v as f64])
        .collect()
}

pub fn draw_perf_graph(ui: &mut egui::Ui, history: &PerfHistory) {
    let present_points = series_points(&history.present_ms);
    let compute_points = series_points(&history.compute_ms);
    let heatmap_points = series_points(&history.heatmap_ms);
    let compositor_points = series_points(&history.compositor_ms);

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
                Line::new("Frame (CPU+GPU)", present_points)
                    .color(FRAME_COLOR)
                    .width(1.5_f32),
            );
            plot_ui.line(
                Line::new("GPU compute", compute_points)
                    .color(GPU_COLOR)
                    .width(1.5_f32),
            );
            plot_ui.line(
                Line::new("Heatmap (GPU)", heatmap_points)
                    .color(HEATMAP_COLOR)
                    .width(1.5_f32),
            );
            plot_ui.line(
                Line::new("Compositor (GPU)", compositor_points)
                    .color(COMPOSITOR_COLOR)
                    .width(1.5_f32),
            );
        });
}

// bound hte graph so it looks nice
fn perf_y_bounds(history: &PerfHistory) -> (f64, f64) {
    let mut lowest = f32::INFINITY;
    let mut highest = f32::NEG_INFINITY;
    for &v in history
        .present_ms
        .iter()
        .chain(&history.compute_ms)
        .chain(&history.heatmap_ms)
        .chain(&history.compositor_ms)
    {
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
