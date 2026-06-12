// Vulkan code is wall-to-wall unsafe calls inside unsafe fns; per-call
// unsafe blocks (edition 2024 style) would add noise without clarity.
#![allow(unsafe_op_in_unsafe_fn)]

use anyhow::Result;
use scene::Scene;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

// BVH/AABB helpers not all wired up yet; kept for accelerator development
#[allow(dead_code)]
mod accelerators;
mod app;
mod fps_counter;
mod gui;
mod scene;
mod types;
mod vulkan;

pub use types::*;

use app::RenderController;
use vulkan::VulkanRenderer;

fn main() -> Result<()> {
    pretty_env_logger::init();

    let scene = Scene::from_new("./scenes/nice/test_scene.yaml")?;
    let render_resolution = scene.get_camera_controls().resolution.0;

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("ray-tracer")
        .with_inner_size(PhysicalSize::new(render_resolution.x, render_resolution.y))
        .build(&event_loop)?;

    let scale_factor = window.scale_factor() as f32;
    let panel_width_px = gui::panel_width_pixels(scale_factor);
    if panel_width_px > 0 {
        let total_width = render_resolution.x + panel_width_px;
        let _ = window.request_inner_size(PhysicalSize::new(total_width, render_resolution.y));
    }

    let mut renderer = unsafe { VulkanRenderer::create(&window, scene)? };
    unsafe {
        renderer.upload_scene()?;
    }

    let gui_shared = gui::create_shared_state();
    let mut render_controller = RenderController::spawn(renderer, gui_shared.clone())?;
    let gui_data_rx = render_controller.gui_data_receiver();
    let mut gui = gui::GuiFrontend::new(&window, gui_shared.clone(), gui_data_rx);

    let mut minimized = false;
    event_loop.run(move |event, elwt| {
        match event {
            Event::NewEvents(_) => {
                elwt.set_control_flow(ControlFlow::Poll);
            }
            Event::AboutToWait => {
                if !minimized {
                    window.request_redraw();
                }
            }
            Event::WindowEvent { event, .. } => {
                gui.handle_event(&window, &event);

                match event {
                    WindowEvent::RedrawRequested => {
                        gui.run_frame(&window);
                        if render_controller.present() {
                            gui.tick_ui_fps();
                        }
                    }
                    WindowEvent::Resized(size) => {
                        if size.width == 0 || size.height == 0 {
                            if !minimized {
                                minimized = true;
                                render_controller.pause();
                            }
                        } else {
                            minimized = false;
                            render_controller.resize(size.width, size.height);
                            render_controller.resume();
                        }
                    }
                    WindowEvent::CloseRequested => {
                        render_controller.shutdown();
                        elwt.exit();
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    })?;

    Ok(())
}
