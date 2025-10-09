#![allow(warnings)]
#![allow(
    dead_code,
    unused_variables,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use std::collections::VecDeque;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded};
use glam::UVec2;
use image::ImageBuffer;
use log::{error, info};
use scene::Scene;
use vulkan::accumulate_image::{create_image, transition_image_layout};
use vulkan::buffers::{create_shader_buffers, create_uniform_buffer};
use vulkan::command_buffers::{
    create_command_buffer, record_compute_commands, record_present_commands,
};
use vulkan::command_pool::create_command_pool;
use vulkan::descriptors::{
    create_compute_descriptor_set_layout, create_descriptor_pool, create_descriptor_sets,
};
use vulkan::fps_counter::FPSCounter;
use vulkan::framebuffer::{create_framebuffer_images, transition_framebuffer_images};
use vulkan::instance::create_instance;
use vulkan::logical_device::create_logical_device;
use vulkan::physical_device::{SuitabilityError, pick_physical_device};
use vulkan::pipeline::{create_compute_pipeline, create_render_pass};
use vulkan::sampler::create_sampler;
use vulkan::swapchain::{create_swapchain, create_swapchain_image_views};
use vulkan::sync_objects::create_sync_objects;
use vulkan::texture::{
    Texture, create_cubemap_sampler, create_cubemap_texture, create_texture_image,
    create_texture_sampler,
};
use vulkan::utils::get_memory_type_index;
use vulkanalia::Version;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;
use winit::dpi::PhysicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;

mod scene;
mod types;
mod ui;
mod vulkan;
pub use types::*;
mod accelerators;

/// Whether the validation layers should be enabled.
// const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_ENABLED: bool = true;
/// The name of the validation layers.z
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// The required device extensions.
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION.name,
    vk::KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION.name,
    vk::EXT_DESCRIPTOR_INDEXING_EXTENSION.name,
    vk::KHR_MAINTENANCE3_EXTENSION.name,
];
/// The Vulkan SDK version that started requiring the portability subset extension for macOS.
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
const TILE_SIZE: u32 = 8;
const OFFSCREEN_FRAME_COUNT: usize = 3;
const TO_SAVE: usize = 100;
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

    // let scene = Scene::from_yaml("./fancy.yaml")?;
    // let scene = Scene::from_gltf("./weekly_voxel_-_furniture/scene.gltf")?;
    // let scene = Scene::from_gltf("./sponza/Sponza.gltf")?;
    // let scene = Scene::from_gltf("./low_poly_lake_scene/scene.gltf")?;
    // let scene = Scene::from_gltf("./bmw_m4_csl_2023/scene.gltf")?;
    // let scene = Scene::from_gltf("./2017-mclaren-720s-lb/source/untitled.gltf")?;
    // let scene = Scene::from_gltf("./gltf/DragonAttenuation.gltf")?;
    // let scene = Scene::from_weird("./benedikt/lego_bulldozer.json")?;
    // let scene = Scene::from_weird("./benedikt/coffe_maker.json")?;
    // let scene = Scene::from_gltf("./Interior/room.gltf")?;
    // let scene = Scene::from_gltf("./glTF/DamagedHelmet.gltf")?;
    // let scene = Scene::from_new("./scenes/lego_bulldozer.yaml")?;
    let scene = Scene::from_new("./scenes/nice/test_scene.yaml")?;
    let render_resolution = scene.get_camera_controls().resolution.0;
    // let scene = Scene::from_new("./scenes/coffee_machine.yaml")?;

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(PhysicalSize::new(
            render_resolution.x,
            render_resolution.y,
        ))
        .build(&event_loop)?;

    let scale_factor = window.scale_factor() as f32;
    let panel_width_px = ui::panel_width_pixels(scale_factor);
    if panel_width_px > 0 {
        let total_width = render_resolution.x + panel_width_px;
        let _ = window.request_inner_size(PhysicalSize::new(total_width, render_resolution.y));
    }

    // App


    let mut app = unsafe { App::create(&window, scene)? };
    unsafe {

        let sizes = app.data.scene.get_buffer_sizes();
        let total_size = sizes.0 + sizes.1 + sizes.2;
        let mapped_ptr = app.device.map_memory(
            app.data.compute_ssbo_buffer_memory,
            0,
            total_size as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        app.data.scene.write_buffers(mapped_ptr);

        println!("sizes: bvh({}), mat({}), tri({})", sizes.0, sizes.1, sizes.2);
        println!("Total memory: {}", total_size);

    }

    let gui_shared = ui::create_shared_state();
    let mut render_controller = RenderController::spawn(app, gui_shared.clone())?;
    let metrics_rx = render_controller.metrics_receiver();
    let mut gui = ui::GuiFrontend::new(&window, gui_shared.clone(), metrics_rx);

    let mut minimized = false;
    event_loop.run(move |event, elwt| {
        match event {
            // Request a redraw when all events were processed.
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

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    pub scene: Scene,

    messenger: vk::DebugUtilsMessengerEXT,
    swapchain: vk::SwapchainKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_layouts: Vec<vk::ImageLayout>,
    render_pass: vk::RenderPass,
    present_queue: vk::Queue,
    compute_queue: vk::Queue,
    compute_pipeline: vk::Pipeline,
    compute_pipeline_layout: vk::PipelineLayout,

    // there is one of these per concurrently rendered image
    compute_descriptor_sets: Vec<vk::DescriptorSet>,
    compute_command_buffers: Vec<vk::CommandBuffer>,
    present_command_buffer: vk::CommandBuffer,

    // framebuffers: Vec<vk::Framebuffer>,
    image_available_semaphores: vk::Semaphore,
    compute_finished_semaphores: vk::Semaphore,

    uniform_buffer_memory: vk::DeviceMemory,
    compute_ssbo_buffer_memory: vk::DeviceMemory,

    // Surface
    surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    physical_device: vk::PhysicalDevice,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    descriptor_set_layout: vk::DescriptorSetLayout,
    // Command Pool
    command_pool: vk::CommandPool,
    // Buffers
    uniform_buffer: vk::Buffer,
    compute_ssbo_buffer: vk::Buffer,
    // Descriptors
    descriptor_pool: vk::DescriptorPool,
    // Accumulator image
    accumulator_view: vk::ImageView,
    accumulator_memory: vk::DeviceMemory,
    accumulator_image: vk::Image,
    // sampler
    sampler: vk::Sampler,

    //textures
    textures: Vec<Texture>,
    texture_sampler: vk::Sampler,
    skybox_texture: Texture,
    skybox_sampler: vk::Sampler,

    framebuffer_images: Vec<vk::Image>,
    framebuffer_image_views: Vec<vk::ImageView>,
    framebuffer_memories: Vec<vk::DeviceMemory>,
}

#[derive(Clone, Debug)]
struct GuiCopyState {
    panel_width: u32,
    render_extent: UVec2,
    base_extent: UVec2,
    last_generation: Option<u64>,
    staging: Option<GuiStaging>,
}

#[derive(Clone, Debug)]
struct GuiStaging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: vk::DeviceSize,
    width: u32,
    height: u32,
}

impl GuiCopyState {
    fn new(initial_extent: UVec2, base_extent: UVec2) -> Self {
        let clamped_width = base_extent.x.min(initial_extent.x);
        let clamped_height = base_extent.y.min(initial_extent.y);
        Self {
            panel_width: 0,
            render_extent: UVec2::new(clamped_width, clamped_height),
            base_extent,
            last_generation: None,
            staging: None,
        }
    }

    unsafe fn update(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &AppData,
        frame: &ui::GuiFrame,
    ) -> Result<()> {
        if self.last_generation == Some(frame.generation) {
            return Ok(());
        }

        let swap_width = data.swapchain_extent.width;
        let swap_height = data.swapchain_extent.height;

        let max_panel_width = swap_width.saturating_sub(self.base_extent.x);
        let copy_width = frame.width.min(swap_width).min(max_panel_width);
        let copy_height = frame.height.min(swap_height);

        self.panel_width = copy_width;
        self.update_render_extent(swap_width, swap_height);

        if copy_width == 0 || copy_height == 0 {
            if let Some(staging) = &mut self.staging {
                staging.width = 0;
                staging.height = 0;
            }
            self.last_generation = Some(frame.generation);
            return Ok(());
        }

        let bytes_per_row = copy_width as usize * 4;
        let required = (bytes_per_row * copy_height as usize) as vk::DeviceSize;

        self.ensure_capacity(instance, device, data, required)?;

        if let Some(staging) = &mut self.staging {
            let ptr = device.map_memory(staging.memory, 0, required, vk::MemoryMapFlags::empty())?
                as *mut u8;

            let src = frame.pixels.as_ref();
            let src_width = frame.width as usize;
            let src_stride = src_width * 4;

            for row in 0..copy_height as usize {
                let src_start = row * src_stride;
                let dst_start = row * bytes_per_row;
                let src_slice = &src[src_start..src_start + bytes_per_row];
                std::ptr::copy_nonoverlapping(
                    src_slice.as_ptr(),
                    ptr.add(dst_start),
                    bytes_per_row,
                );
            }

            device.unmap_memory(staging.memory);
            staging.width = copy_width;
            staging.height = copy_height;
        }

        self.last_generation = Some(frame.generation);
        Ok(())
    }

    fn copy_info(&self) -> Option<GuiCopyInfo> {
        self.staging.as_ref().and_then(|staging| {
            if self.panel_width == 0 || staging.width == 0 || staging.height == 0 {
                None
            } else {
                Some(GuiCopyInfo {
                    buffer: staging.buffer,
                    width: staging.width,
                    height: staging.height,
                })
            }
        })
    }

    fn render_extent(&self) -> UVec2 {
        self.render_extent
    }

    fn panel_width(&self) -> u32 {
        self.panel_width
    }

    fn handle_resize(&mut self, width: u32, height: u32) {
        let max_panel_width = width.saturating_sub(self.base_extent.x);
        let clamped = self.panel_width.min(width).min(max_panel_width);
        self.panel_width = clamped;
        self.update_render_extent(width, height);
        if let Some(staging) = &mut self.staging {
            staging.width = staging.width.min(clamped);
            staging.height = staging.height.min(height);
        }
    }

    unsafe fn destroy(&mut self, device: &Device) {
        if let Some(staging) = self.staging.take() {
            device.destroy_buffer(staging.buffer, None);
            device.free_memory(staging.memory, None);
        }
    }

    unsafe fn ensure_capacity(
        &mut self,
        instance: &Instance,
        device: &Device,
        data: &AppData,
        required: vk::DeviceSize,
    ) -> Result<()> {
        if let Some(staging) = &self.staging {
            if staging.capacity >= required {
                return Ok(());
            }
        }

        if let Some(existing) = self.staging.take() {
            device.destroy_buffer(existing.buffer, None);
            device.free_memory(existing.memory, None);
        }

        if required == 0 {
            return Ok(());
        }

        let buffer_info = vk::BufferCreateInfo::builder()
            .size(required)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = device.create_buffer(&buffer_info, None)?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let memory_type = get_memory_type_index(
            instance,
            data,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            requirements,
        )?;

        let allocation_size = requirements.size.max(required);
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(allocation_size)
            .memory_type_index(memory_type);

        let memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(buffer, memory, 0)?;

        self.staging = Some(GuiStaging {
            buffer,
            memory,
            capacity: allocation_size,
            width: 0,
            height: 0,
        });

        Ok(())
    }

    fn update_render_extent(&mut self, swap_width: u32, swap_height: u32) {
        let available_width = swap_width.saturating_sub(self.panel_width);
        let width = self.base_extent.x.min(available_width);
        let height = self.base_extent.y.min(swap_height);
        self.render_extent = UVec2::new(width, height);
    }
}

/// Our Vulkan app.
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
    pub data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    fps_counter: FPSCounter,
    gui_copy: GuiCopyState,
    frame_fences: Vec<vk::Fence>,
    present_fence: vk::Fence,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window, scene: Scene) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        data.scene = scene;
        let scene_sizes = data.scene.get_buffer_sizes();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        // create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        create_framebuffer_images(&instance, &device, &mut data, OFFSCREEN_FRAME_COUNT)?;
        transition_framebuffer_images(&device, &mut data)?;
        // create_vertex_buffer(&instance, &device, &mut data)?;
        create_uniform_buffer(&instance, &device, &mut data)?;
        create_shader_buffers(
            &instance,
            &device,
            &mut data,
            (scene_sizes.0 + scene_sizes.1 + scene_sizes.2) as u64,
        )?;
        create_image(&instance, &device, &mut data)?;
        transition_image_layout(&device, &mut data)?;

        data.texture_sampler = create_texture_sampler(&device)?;
        data.skybox_sampler = create_cubemap_sampler(&device)?;

        for texture_data in &data.scene.components.textures {
            let texture = create_texture_image(
                &instance,
                &device,
                &data,
                &texture_data.pixels,
                texture_data.width,
                texture_data.height,
            )?;
            data.textures.push(texture);
        }

        if data.textures.is_empty() {
            let default_pixels = [255u8, 255, 255, 255];
            let texture = create_texture_image(&instance, &device, &data, &default_pixels, 1, 1)?;
            data.textures.push(texture);
        }

        let skybox_data = &data.scene.components.skybox;
        data.skybox_texture = create_cubemap_texture(
            &instance,
            &device,
            &data,
            &skybox_data.pixels,
            skybox_data.width,
            skybox_data.height,
        )?;

        create_compute_descriptor_set_layout(&device, &mut data)?;
        create_compute_pipeline(&device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_sampler(&device, &mut data)?;
        // create_descriptor_sets(&device, &mut data)?;
        create_descriptor_sets(
            &device,
            &mut data,
            scene_sizes.0 as u64,
            scene_sizes.1 as u64,
            scene_sizes.2 as u64,
        )?;
        create_command_buffer(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;
        let mut frame_fences = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
        for _ in 0..OFFSCREEN_FRAME_COUNT {
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            frame_fences.push(device.create_fence(&fence_info, None)?);
        }
        let present_fence_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let present_fence = device.create_fence(&present_fence_info, None)?;
        info!("Finished initialisation of Vulkan Resources");
        let swapchain_width = data.swapchain_extent.width;
        let swapchain_height = data.swapchain_extent.height;
        let render_resolution = data.scene.get_camera_controls().resolution.0;
        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            fps_counter: FPSCounter::new(15),
            gui_copy: GuiCopyState::new(
                UVec2::new(swapchain_width, swapchain_height),
                render_resolution,
            ),
            frame_fences,
            present_fence,
        })
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn dispatch_compute(&mut self, frame_index: usize) -> Result<()> {
        let command_buffer = self.data.compute_command_buffers[frame_index];

        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        self.update_uniform_buffer()?;
        let render_extent = self.gui_copy.render_extent();
        record_compute_commands(
            &self.device,
            &mut self.data,
            command_buffer,
            frame_index,
            render_extent,
        )?;

        self.device
            .reset_fences(&[self.frame_fences[frame_index]])?;

        let command_buffers = &[command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(command_buffers);

        self.device.queue_submit(
            self.data.compute_queue,
            &[submit_info],
            self.frame_fences[frame_index],
        )?;

        self.frame += 1;

        if self.frame == TO_SAVE {
            save_frame(
                &self.instance,
                &self.device,
                &mut self.data,
                self.frame as u32,
            )?;
        }

        Ok(())
    }

    unsafe fn present_frame(
        &mut self,
        frame_index: usize,
        gui_frame: Option<ui::GuiFrame>,
    ) -> Result<()> {
        if let Some(frame) = gui_frame {
            self.gui_copy
                .update(&self.instance, &self.device, &self.data, &frame)?;
        }

        let render_extent = self.gui_copy.render_extent();
        let panel_width = self.gui_copy.panel_width();

        self.device
            .wait_for_fences(&[self.frame_fences[frame_index]], true, u64::MAX)?;

        self.device
            .wait_for_fences(&[self.present_fence], true, u64::MAX)?;
        self.device.reset_fences(&[self.present_fence])?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores,
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            // Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        // let image_in_flight = self.data.images_in_flight[image_index];
        // if !image_in_flight.is_null() {
        //     self.device.wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        // }

        // self.data.images_in_flight[image_index] = in_flight_fence;

        self.device.reset_command_buffer(
            self.data.present_command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )?;

        let gui_copy = self.gui_copy.copy_info();
        record_present_commands(
            &self.device,
            &mut self.data,
            image_index,
            frame_index,
            panel_width,
            render_extent,
            gui_copy,
        )?;

        let wait_semaphores = &[self.data.image_available_semaphores];
        let command_buffers = &[self.data.present_command_buffer];
        let signal_semaphores = &[self.data.compute_finished_semaphores];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .queue_submit(self.data.present_queue, &[submit_info], self.present_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            // self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        Ok(())
    }

    /// Updates the uniform buffer object for our Vulkan app.
    unsafe fn update_uniform_buffer(&self) -> Result<()> {
        // MVP

        let mut ubo = self.data.scene.get_camera_controls();
        let render_extent = self.gui_copy.render_extent();
        let panel_width = self.gui_copy.panel_width();
        ubo.resolution = AUVec2(render_extent);
        ubo.time = Au32(self.frame as u32);

        let memory = self.device.map_memory(
            self.data.uniform_buffer_memory,
            0,
            size_of::<CameraBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.device.unmap_memory(self.data.uniform_buffer_memory);

        Ok(())
    }

    /// Recreates the swapchain for our Vulkan app.
    #[rustfmt::skip]
    // unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
    //     self.device.device_wait_idle()?;
    //     self.destroy_swapchain();
    //     create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
    //     create_swapchain_image_views(&self.device, &mut self.data)?;
    //     create_render_pass(&self.instance, &self.device, &mut self.data)?;
    //     create_pipeline(&self.device, &mut self.data)?;
    //     create_framebuffers(&self.device, &mut self.data)?;
    //     create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
    //     create_descriptor_pool(&self.device, &mut self.data)?;
    //     create_descriptor_sets(&self.device, &mut self.data)?;
    //     create_command_buffers(&self.device, &mut self.data)?;
    //     Ok(())
    // }

    fn handle_resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.data.swapchain_extent.width = width;
        self.data.swapchain_extent.height = height;
        self.gui_copy.handle_resize(width, height);
        self.resized = true;
    }

    /// Destroys our Vulkan app.
    #[rustfmt::skip]
    unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();
        self.gui_copy.destroy(&self.device);

        // self.data.in_flight_fences.iter().for_each(|f| self.device.destroy_fence(*f, None));
        // self.data.render_finished_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        // self.data.image_available_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        // self.device.free_memory(self.data.shader_buffer_memory, None);
        // self.device.destroy_buffer(self.data.shader_buffer, None);
        let mut all_command_buffers = self.data.compute_command_buffers.clone();
        all_command_buffers.push(self.data.present_command_buffer);
        self.device
            .free_command_buffers(self.data.command_pool, &all_command_buffers);
        self.device.destroy_command_pool(self.data.command_pool, None);
        for &view in &self.data.framebuffer_image_views {
            self.device.destroy_image_view(view, None);
        }
        for &image in &self.data.framebuffer_images {
            self.device.destroy_image(image, None);
        }
        for &memory in &self.data.framebuffer_memories {
            self.device.free_memory(memory, None);
        }
        for &fence in &self.frame_fences {
            self.device.destroy_fence(fence, None);
        }
        self.device.destroy_fence(self.present_fence, None);
        self.device
            .destroy_semaphore(self.data.image_available_semaphores, None);
        self.device
            .destroy_semaphore(self.data.compute_finished_semaphores, None);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    /// Destroys the parts of our Vulkan app related to the swapchain.
    #[rustfmt::skip]
    unsafe fn destroy_swapchain(&mut self) {
        // self.device.free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        // self.data.uniform_buffers_memory.iter().for_each(|m| self.device.free_memory(*m, None));
        // self.data.uniform_buffers.iter().for_each(|b| self.device.destroy_buffer(*b, None));
        // self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        // self.device.destroy_pipeline(self.data.pipeline, None);
        // self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.data.swapchain_image_layouts.clear();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RenderMetrics {
    pub fps: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct GuiCopyInfo {
    pub buffer: vk::Buffer,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub enum RenderCommand {
    Pause,
    Resume,
    Resize { width: u32, height: u32 },
    Present,
    Shutdown,
}

pub struct RenderController {
    command_tx: Sender<RenderCommand>,
    metrics_rx: Receiver<RenderMetrics>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RenderController {
    fn spawn(app: App, gui_shared: ui::GuiShared) -> Result<Self> {
        let (command_tx, command_rx) = bounded(16);
        let (metrics_tx, metrics_rx) = bounded(32);

        let render_gui_shared = gui_shared.clone();
        let handle = thread::Builder::new()
            .name("render-thread".into())
            .spawn(move || render_loop(app, render_gui_shared, command_rx, metrics_tx))
            .map_err(|err| anyhow!("failed to spawn render thread: {err}"))?;

        Ok(Self {
            command_tx,
            metrics_rx,
            handle: Some(handle),
        })
    }

    pub fn pause(&self) {
        let _ = self.command_tx.try_send(RenderCommand::Pause);
    }

    pub fn resume(&self) {
        let _ = self.command_tx.try_send(RenderCommand::Resume);
    }

    pub fn resize(&self, width: u32, height: u32) {
        let _ = self
            .command_tx
            .try_send(RenderCommand::Resize { width, height });
    }

    pub fn present(&self) {
        let _ = self.command_tx.try_send(RenderCommand::Present);
    }

    pub fn shutdown(&mut self) {
        if let Some(handle) = self.handle.take() {
            let _ = self.command_tx.send(RenderCommand::Shutdown);
            let _ = handle.join();
        }
    }

    pub fn metrics_receiver(&self) -> Receiver<RenderMetrics> {
        self.metrics_rx.clone()
    }

    pub fn command_sender(&self) -> Sender<RenderCommand> {
        self.command_tx.clone()
    }
}

impl Drop for RenderController {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn render_loop(
    mut app: App,
    gui_shared: ui::GuiShared,
    command_rx: Receiver<RenderCommand>,
    metrics_tx: Sender<RenderMetrics>,
) {
    let mut paused = false;
    let mut running = true;
    let mut available: VecDeque<usize> = (0..OFFSCREEN_FRAME_COUNT).collect();
    let mut in_flight: Vec<usize> = Vec::with_capacity(OFFSCREEN_FRAME_COUNT);
    let mut ready: VecDeque<usize> = VecDeque::with_capacity(OFFSCREEN_FRAME_COUNT);
    let mut current_frame: Option<usize> = None;

    while running {
        let mut present_requested = false;
        let mut pending_resize: Option<(u32, u32)> = None;

        loop {
            match command_rx.try_recv() {
                Ok(command) => match command {
                    RenderCommand::Pause => paused = true,
                    RenderCommand::Resume => paused = false,
                    RenderCommand::Resize { width, height } => {
                        pending_resize = Some((width, height));
                    }
                    RenderCommand::Present => present_requested = true,
                    RenderCommand::Shutdown => {
                        running = false;
                        break;
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    running = false;
                    break;
                }
            }
        }

        if !running {
            break;
        }

        if let Some((width, height)) = pending_resize {
            app.handle_resize(width, height);
        }

        // Promote completed frames to ready queue
        let mut completed = Vec::new();
        in_flight.retain(|index| {
            match unsafe { app.device.get_fence_status(app.frame_fences[*index]) } {
                Ok(_) => {
                    completed.push(*index);
                    false
                }
                Ok(_) => true,
                Err(err) => {
                    error!("fence status error: {err:?}");
                    true
                }
            }
        });

        for index in completed {
            ready.push_back(index);
            let fps = app.fps_counter.tick();
            let _ = metrics_tx.try_send(RenderMetrics { fps });
        }

        if present_requested {
            let gui_frame = gui_shared.read().ok().and_then(|state| state.latest());
            if let Some(new_frame) = ready.pop_back() {
                // Release older ready frames back to available; keep most recent only.
                while let Some(stale) = ready.pop_front() {
                    available.push_back(stale);
                }

                if let Err(err) = unsafe { app.present_frame(new_frame, gui_frame.clone()) } {
                    error!("present error: {err:?}");
                    available.push_back(new_frame);
                } else {
                    if let Some(previous) = current_frame.replace(new_frame) {
                        available.push_back(previous);
                    }
                }
            } else if let Some(current) = current_frame {
                if let Err(err) = unsafe { app.present_frame(current, gui_frame) } {
                    error!("present error: {err:?}");
                }
            }
        }

        if !paused {
            if let Some(index) = available.pop_front() {
                match unsafe { app.dispatch_compute(index) } {
                    Ok(()) => in_flight.push(index),
                    Err(err) => {
                        error!("dispatch error: {err:?}");
                        available.push_front(index);
                        thread::sleep(Duration::from_millis(16));
                    }
                }
            }
        } else {
            thread::sleep(Duration::from_millis(5));
        }

        if available.is_empty() {
            thread::sleep(Duration::from_millis(1));
        }
    }

    unsafe {
        app.destroy();
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    graphics: u32,
    compute: u32,
    present: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);
        if let Some(index) = properties.iter().enumerate().find_map(|(i, p)| {
            if p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && instance
                    .get_physical_device_surface_support_khr(
                        physical_device,
                        i as u32,
                        data.surface,
                    )
                    .unwrap_or(false)
            {
                Some(i as u32)
            } else {
                None
            }
        }) {
            Ok(QueueFamilyIndices {
                graphics: index,
                compute: index,
                present: index,
            })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}

unsafe fn save_frame(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
    frame: u32,
) -> Result<()> {
    let size = (data.swapchain_extent.width * data.swapchain_extent.height * 4) as u64;

    // Create staging buffer
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let staging_buffer = device.create_buffer(&buffer_info, None)?;

    // Allocate memory for staging buffer
    let mem_requirements = device.get_buffer_memory_requirements(staging_buffer);
    let memory_type = get_memory_type_index(
        instance,
        data,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        mem_requirements,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type);

    let staging_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(staging_buffer, staging_memory, 0)?;

    // Create command buffer for copy operation
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(data.command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&alloc_info)?[0];

    // Record copy command
    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &begin_info)?;

    let current_layout = data
        .swapchain_image_layouts
        .get(0)
        .copied()
        .unwrap_or(vk::ImageLayout::UNDEFINED);

    let (src_stage, src_access) = match current_layout {
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        ),
        vk::ImageLayout::PRESENT_SRC_KHR => (
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::MEMORY_READ,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::empty(),
        ),
    };

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(current_layout)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(data.swapchain_images[0]) // Use the first swapchain image, or the current one
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(src_access)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    let copy = vk::BufferImageCopy::builder()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(vk::Extent3D {
            width: data.swapchain_extent.width,
            height: data.swapchain_extent.height,
            depth: 1,
        });

    device.cmd_copy_image_to_buffer(
        command_buffer,
        data.swapchain_images[0], // Use the first swapchain image, or the current one
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        staging_buffer,
        &[copy],
    );

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .image(data.swapchain_images[0]) // Use the first swapchain image, or the current one
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
        .src_access_mask(vk::AccessFlags::TRANSFER_READ)
        .dst_access_mask(vk::AccessFlags::MEMORY_READ);

    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    if let Some(layout) = data.swapchain_image_layouts.get_mut(0) {
        *layout = vk::ImageLayout::PRESENT_SRC_KHR;
    }

    device.end_command_buffer(command_buffer)?;

    // Submit and wait
    let submit_info =
        vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));

    device.queue_submit(data.compute_queue, &[submit_info], vk::Fence::null())?;
    device.queue_wait_idle(data.compute_queue)?;

    // Map memory and save to file
    let data_ptr =
        device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())? as *const u8;

    let buffer = std::slice::from_raw_parts(data_ptr, size as usize);

    // Convert RGBA to proper format and create image
    let width = data.swapchain_extent.width as u32;
    let height = data.swapchain_extent.height as u32;
    let mut img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            let pixel = image::Rgba([
                buffer[i + 2], // Blue channel becomes Red
                buffer[i + 1], // Green stays the same
                buffer[i],     // Red channel becomes Blue
                buffer[i + 3], // Alpha stays the same
            ]);
            img.put_pixel(x, y, pixel);
        }
    }

    let mut input_img = vec![0.0f32; (3 * width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            for c in 0..3 {
                input_img[3 * ((y * width + x) as usize) + c] = pixel[c] as f32 / 255.0;
            }
        }
    }

    // // Apply denoising
    // let mut filter_output = vec![0.0f32; input_img.len()];
    // let odin_device = oidn::Device::new();
    // let mut filter = oidn::RayTracing::new(&odin_device);
    //
    // filter
    //     .srgb(true)
    //     .image_dimensions(width as usize, height as usize);
    //
    // filter
    //     .filter(&input_img[..], &mut filter_output[..])
    //     .map_err(|e| anyhow!("Denoising error: {:?}", e))?;
    //
    // if let Err(e) = odin_device.get_error() {
    //     println!("Warning: Denoising error: {}", e.1);
    // }
    //
    // // Convert back to u8 RGB image
    // let mut denoised_img = ImageBuffer::new(width, height);
    // for y in 0..height {
    //     for x in 0..width {
    //         let idx = 3 * ((y * width + x) as usize);
    //         let r = (filter_output[idx].clamp(0.0, 1.0) * 255.0) as u8;
    //         let g = (filter_output[idx + 1].clamp(0.0, 1.0) * 255.0) as u8;
    //         let b = (filter_output[idx + 2].clamp(0.0, 1.0) * 255.0) as u8;
    //         denoised_img.put_pixel(x, y, image::Rgb([r, g, b]));
    //     }
    // }

    device.unmap_memory(staging_memory);

    // Cleanup
    device.free_command_buffers(data.command_pool, &[command_buffer]);
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_memory, None);

    println!("Saved Buffer");

    img.save("images/materials/raw/spec_trans/spec_trans_100.png")?;
    panic!();
    Ok(())
}
