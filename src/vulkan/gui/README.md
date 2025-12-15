# Vulkan GUI Utilities

This module provides reusable Vulkan boilerplate abstractions for GUI rendering with egui.

## Purpose

The GUI rendering utilities abstract common Vulkan patterns to:
- **Reduce code duplication** between different renderers
- **Improve maintainability** by centralizing Vulkan boilerplate
- **Make GUI rendering logic clearer** by separating it from low-level Vulkan details

## Modules

### `buffers.rs`
Buffer management utilities for dynamic GUI data:
- `create_dynamic_buffer()` - Creates host-visible buffers with coherent memory
- `upload_to_buffer()` - Efficiently uploads data to buffers via memory mapping
- `destroy_buffer()` - Safely destroys buffers and frees memory

### `pipeline.rs`
Graphics pipeline creation utilities:
- `create_sampler()` - Creates a linear sampler for GUI textures
- `create_descriptor_set_layout()` - Creates descriptor layout for texture sampling
- `create_pipeline_layout()` - Creates pipeline layout with push constants
- `create_pipeline()` - Creates complete graphics pipeline with blending
- `create_descriptor_pool()` - Creates descriptor pool for GUI textures
- `GuiVertex` - Vertex format for GUI rendering (position, UV, color)

### `upload.rs`
Texture upload utilities:
- `transition_image_layout()` - Transitions image layouts with pipeline barriers
- `upload_pixels_to_image()` - Uploads pixel data using staging buffers and command buffers

## Usage Example

```rust
use crate::vulkan::gui;

// Create pipeline resources
let sampler = gui::create_sampler(device)?;
let descriptor_set_layout = gui::create_descriptor_set_layout(device)?;
let pipeline_layout = gui::create_pipeline_layout(device, descriptor_set_layout)?;
let pipeline = gui::create_pipeline(device, pipeline_layout, render_pass)?;

// Create and upload to buffers
let (buffer, memory, capacity) = gui::create_dynamic_buffer(
    instance, device, data, size, vk::BufferUsageFlags::VERTEX_BUFFER
)?;
gui::upload_to_buffer(device, memory, &vertex_data)?;

// Upload texture data
gui::upload_pixels_to_image(
    instance, device, data, image, pixels, size, None, true
)?;
```

## Benefits

1. **Centralized maintenance** - Update Vulkan patterns in one place
2. **Reusability** - Use these utilities in any GUI or rendering code
3. **Clarity** - High-level renderer code focuses on logic, not Vulkan details
4. **Consistency** - Standardized patterns across the codebase
