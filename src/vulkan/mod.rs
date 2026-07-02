//! The Vulkan backend. The rest of the crate interacts with it solely
//! through [`VulkanRenderer`] and [`OFFSCREEN_FRAME_COUNT`]; no vulkanalia
//! types escape this module.

pub mod constants;
pub mod core;
mod gui_renderer;
mod heatmap_renderer;
mod path_tracer;
mod present;
mod renderer;
mod utils;

pub use constants::OFFSCREEN_FRAME_COUNT;
pub use renderer::VulkanRenderer;
