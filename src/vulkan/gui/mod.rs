/// Vulkan abstraction utilities for GUI rendering.
/// 
/// This module provides reusable Vulkan boilerplate functions for GUI rendering,
/// abstracting away common patterns for buffer management, pipeline creation,
/// and texture uploads.

pub mod buffers;
pub mod pipeline;
pub mod upload;

pub use buffers::{create_dynamic_buffer, destroy_buffer, upload_to_buffer};
pub use pipeline::{
    GuiVertex, create_descriptor_pool, create_descriptor_set_layout, create_pipeline,
    create_pipeline_layout, create_sampler,
};
pub use upload::{transition_image_layout, upload_pixels_to_image};
