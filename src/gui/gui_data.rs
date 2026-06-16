use std::{collections::VecDeque, default, path::{Path, PathBuf}};

use crate::gui::components::perf_graph::PERF_HISTORY_LEN;

#[derive(Clone, Debug)]
pub enum PushRender {
    SaveFrame(PathBuf),
}

pub enum PushGui {
    Fps(f64),
    PerfUpdate{compute_ms: f64, present_ms: f64}
}

#[derive(Debug, Default)]
pub struct GuiData {
    pub fps: f64,
    // frame calculated with intervat from last frame
    pub frame_ms: f64,
    // how long it takes to fully finish on the gpu
    pub compute_ms: f64,
    pub present_ms: f64,

    pub save_file_path: String,
    pub perf_history: PerfHistory,
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
    capacity: usize,
}

impl PerfHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            present_ms: VecDeque::with_capacity(capacity),
            compute_ms: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, compute_ms: f32, present_ms: f32) {
        if self.present_ms.len() >= self.capacity {
            self.present_ms.pop_front();
        }
        if self.compute_ms.len() >= self.capacity {
            self.compute_ms.pop_front();
        }
        self.present_ms.push_back(compute_ms);
        self.compute_ms.push_back(present_ms);
    }
}

