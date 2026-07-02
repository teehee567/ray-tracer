use std::{collections::VecDeque, path::PathBuf};

use crate::gui::components::perf_graph::PERF_HISTORY_LEN;

/// Which backend output is shown in the viewport.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RenderMode {
    PathTracer,
    #[default]
    BvhHeatmap,
}

impl RenderMode {
    pub const ALL: [RenderMode; 2] = [RenderMode::PathTracer, RenderMode::BvhHeatmap];

    pub fn label(self) -> &'static str {
        match self {
            RenderMode::PathTracer => "Path tracer",
            RenderMode::BvhHeatmap => "BVH heatmap",
        }
    }
}

#[derive(Clone, Debug)]
pub enum PushRender {
    SaveFrame(PathBuf),
    SetHeatmapBand { low: u32, high: u32 },
    SetRenderMode(RenderMode),
}

pub enum PushGui {
    Status {
        samples: u32,
        paused: bool,
    },
    /// Result of a shader reload attempt; `None` means success.
    ShaderReload {
        error: Option<String>,
    },
    PerfUpdate {
        compute_fps: f64,
        compute_ms: f64,
        present_fps: f64,
        present_ms: f64,
        heatmap_ms: f64,
        compositor_ms: f64,
    },
    HeatmapInfo {
        max_depth: u32,
    },
    RenderResolution {
        width: u32,
        height: u32,
    },
}

#[derive(Debug, Default)]
pub struct GuiData {
    pub compute_fps: f64,
    pub compute_ms: f64,

    pub present_fps: f64,
    pub present_ms: f64,

    pub heatmap_ms: f64,
    pub compositor_ms: f64,

    pub save_file_path: String,
    pub perf_history: PerfHistory,

    /// Last shader compile/reload error; cleared by a successful reload.
    pub shader_error: Option<String>,

    pub render_mode: RenderMode,
    pub user_paused: bool,
    pub sample_count: u32,
    /// Actual paused state on the render thread (user pause or minimized).
    pub effective_paused: bool,
    pub heatmap_depth_low: u32,
    pub heatmap_depth_high: u32,
    pub heatmap_max_depth: u32,

    // actual render target size
    pub render_width: u32,
    pub render_height: u32,
}

impl GuiData {
    pub fn new() -> Self {
        Self {
            perf_history: PerfHistory::new(PERF_HISTORY_LEN),
            ..Default::default()
        }
    }
}

#[derive(Debug, Default)]
pub struct PerfHistory {
    pub present_ms: VecDeque<f32>,
    pub compute_ms: VecDeque<f32>,
    pub heatmap_ms: VecDeque<f32>,
    pub compositor_ms: VecDeque<f32>,
    capacity: usize,
}

impl PerfHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            present_ms: VecDeque::with_capacity(capacity),
            compute_ms: VecDeque::with_capacity(capacity),
            heatmap_ms: VecDeque::with_capacity(capacity),
            compositor_ms: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, compute_ms: f32, present_ms: f32, heatmap_ms: f32, compositor_ms: f32) {
        push_capped(&mut self.compute_ms, compute_ms, self.capacity);
        push_capped(&mut self.present_ms, present_ms, self.capacity);
        push_capped(&mut self.heatmap_ms, heatmap_ms, self.capacity);
        push_capped(&mut self.compositor_ms, compositor_ms, self.capacity);
    }
}

fn push_capped(buf: &mut VecDeque<f32>, value: f32, capacity: usize) {
    if buf.len() >= capacity {
        buf.pop_front();
    }
    buf.push_back(value);
}
