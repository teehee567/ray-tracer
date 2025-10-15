#![allow(warnings)]
#![allow(
    dead_code,
    unused_variables,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use anyhow::Result;
use scene::Scene;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

mod accelerators;
mod app;
mod gui;
mod scene;
mod types;
mod vulkan;

pub use app::{
    App, AppData, DEVICE_EXTENSIONS, OFFSCREEN_FRAME_COUNT, PORTABILITY_MACOS_VERSION,
    QueueFamilyIndices, RenderCommand, RenderController, SwapchainSupport,
    TILE_SIZE, VALIDATION_ENABLED, VALIDATION_LAYER,
};

use gui::GuiData;
pub use types::*;

macro_rules! print_size {
    ($t:ty) => {
        println!(
            "Size of {}: {} bytes",
            stringify!($t),
            std::mem::size_of::<$t>()
        );
    };
}

fn assert_vecs_equal<T: std::fmt::Debug + PartialEq>(v1: &[T], v2: &[T], context: usize) {
    if v1.len() != v2.len() {
        panic!(
            "Vectors have different lengths: {} vs {}",
            v1.len(),
            v2.len()
        );
    }

    for (i, (a, b)) in v1.iter().zip(v2.iter()).enumerate() {
        if a != b {
            let start = i.saturating_sub(context);
            let end = (i + context + 1).min(v1.len());

            println!("Vectors differ at index {}:", i);
            println!("Vector 1 context: {:?}", &v1[start..end]);
            println!("Vector 2 context: {:?}", &v2[start..end]);
            println!("Specific difference: {:?} != {:?}", a, b);
            dbg!(a);
            dbg!(b);

            panic!("Vector mismatch at index {}", i);
        }
    }
}

#[rustfmt::skip]
fn main() -> Result<()> {
    pretty_env_logger::init();

    print_size!(CameraBufferObject);
    print_size!(Triangle);
    print_size!(Material);
    print_size!(SceneComponents);
    print_size!(Sphere);

    let scene = Scene::from_new("./scenes/nice/test_scene.yaml")?;
    let render_resolution = scene.get_camera_controls().resolution.0;

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(PhysicalSize::new(
            render_resolution.x,
            render_resolution.y,
        ))
        .build(&event_loop)?;

    let scale_factor = window.scale_factor() as f32;
    let panel_width_px = gui::panel_width_pixels(scale_factor);
    if panel_width_px > 0 {
        let total_width = render_resolution.x + panel_width_px;
        let _ = window.request_inner_size(PhysicalSize::new(total_width, render_resolution.y));
    }

    let mut app = unsafe { App::create(&window, scene)? };
    unsafe {
        app.upload_scene()?;
    }

    let gui_shared = gui::create_shared_state();
    let mut render_controller = RenderController::spawn(app, gui_shared.clone())?;
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
                        render_controller.present();
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
