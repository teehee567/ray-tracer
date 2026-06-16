use std::{collections::VecDeque, default, path::{Path, PathBuf}};

use crate::gui::components::perf_graph::PERF_HISTORY_LEN;

#[derive(Clone, Debug)]
pub enum PushRender {
    SaveFrame(PathBuf),
}

pub enum PushGui {
    Fps(f64),
    PerfUpdate{frame_ms: f64, compute_ms: f64}
}

#[derive(Debug, Default)]
pub struct GuiData {
    pub fps: f64,
    // frame calculated with intervat from last frame
    pub frame_ms: f64,
    // how long it takes to fully finish on the gpu
    pub compute_ms: f64,
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
    pub frame_ms: VecDeque<f32>,
    pub compute_ms: VecDeque<f32>,
    capacity: usize,
}

impl PerfHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            frame_ms: VecDeque::with_capacity(capacity),
            compute_ms: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, frame_ms: f32, compute_ms: f32) {
        if self.frame_ms.len() >= self.capacity {
            self.frame_ms.pop_front();
        }
        if self.compute_ms.len() >= self.capacity {
            self.compute_ms.pop_front();
        }
        self.frame_ms.push_back(frame_ms);
        self.compute_ms.push_back(compute_ms);
    }
}

