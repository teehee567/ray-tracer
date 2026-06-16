use std::collections::VecDeque;

#[derive(Clone, Debug)]
pub struct FPSCounter {
    frame_ms: VecDeque<f64>,
    max_samples: usize,
}

impl FPSCounter {
    pub fn new(max_samples: usize) -> Self {
        Self {
            frame_ms: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    pub fn push_ms(&mut self, ms: f64) {
        self.frame_ms.push_back(ms);

        if self.frame_ms.len() > self.max_samples {
            self.frame_ms.pop_front();
        }
    }

    pub fn last_frame_ms(&self) -> f64 {
        self.frame_ms.back().copied().unwrap_or(0.0)
    }

    pub fn get_fps(&self) -> f64 {
        if self.frame_ms.is_empty() {
            return 0.0;
        }

        let total_ms: f64 = self.frame_ms.iter().sum();
        if total_ms > 0.0 {
            self.frame_ms.len() as f64 / (total_ms / 1000.0)
        } else {
            0.0
        }
    }
}
