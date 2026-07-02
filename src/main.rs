// Vulkan code is wall-to-wall unsafe calls inside unsafe fns; per-call
// unsafe blocks (edition 2024 style) would add noise without clarity.
#![allow(unsafe_op_in_unsafe_fn)]

use anyhow::Result;
use scene::Scene;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

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

struct AppState {
    window: Window,
    render_controller: RenderController,
    gui: gui::GuiFrontend,
    minimized: bool,
}

struct App {
    // Consumed when the window is created on the first `resumed` call.
    scene: Option<Scene>,
    state: Option<AppState>,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<AppState> {
        let scene = self
            .scene
            .take()
            .expect("scene already consumed by a previous resume");
        let initial_camera = scene.get_camera_controls();
        let render_resolution = initial_camera.resolution.0;

        let attributes = Window::default_attributes()
            .with_title("ray-tracer")
            .with_inner_size(PhysicalSize::new(render_resolution.x, render_resolution.y));
        let window = event_loop.create_window(attributes)?;

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
        let render_controller = RenderController::spawn(renderer, gui_shared.clone())?;
        let (gui_data_rx, render_sender) = render_controller.gui_channels();
        let gui = gui::GuiFrontend::new(
            &window,
            gui_shared,
            gui_data_rx,
            render_sender,
            initial_camera,
        );

        Ok(AppState {
            window,
            render_controller,
            gui,
            minimized: false,
        })
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let state = self
                .init(event_loop)
                .expect("failed to initialize renderer");
            self.state = Some(state);
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            if !state.minimized {
                state.window.request_redraw();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        state.gui.handle_event(&state.window, &event);

        match event {
            WindowEvent::RedrawRequested => {
                state.gui.run_frame(&state.window);
                state.render_controller.present();
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    if !state.minimized {
                        state.minimized = true;
                        state.render_controller.pause();
                    }
                } else {
                    state.minimized = false;
                    state.render_controller.resize(size.width, size.height);
                    state.render_controller.resume();
                }
            }
            WindowEvent::CloseRequested => {
                state.render_controller.shutdown();
                event_loop.exit();
            }
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let scene = Scene::from_new("./scenes/nice/lego_bulldozer.yaml")?;

    let event_loop = EventLoop::new()?;
    let mut app = App {
        scene: Some(scene),
        state: None,
    };
    event_loop.run_app(&mut app)?;

    Ok(())
}
