use std::collections::VecDeque;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct FPSCounter {
    frame_times: VecDeque<Instant>,
    max_samples: usize,
}

impl FPSCounter {
    pub fn new(max_samples: usize) -> Self {
        Self {
            frame_times: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    pub fn update(&mut self) {
        let now = Instant::now();
        self.frame_times.push_back(now);

        if self.frame_times.len() > self.max_samples {
            self.frame_times.pop_front();
        }
    }

    pub fn get_fps(&self) -> f64 {
        if self.frame_times.len() < 2 {
            return 0.0;
        }

        let duration = self.frame_times.back().unwrap().duration_since(*self.frame_times.front().unwrap());
        let frame_count = self.frame_times.len() as f64;

        if duration.as_secs_f64() > 0.0 {
            frame_count / duration.as_secs_f64()
        } else {
            0.0
        }
    }

    pub fn print(&self) {
        println!("FPS: {:.2}", self.get_fps());
    }

}
