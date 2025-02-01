// SPDX-License-Identifier: Apache-2.0

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
use std::time::Instant;

use anyhow::{anyhow, Result};
use glam::{Mat4, Vec2, Vec3, Vec4};
use vulkan::accumulate_image::{create_image, transition_image_layout};
use vulkan::bufferbuilder::BufferBuilder;
use vulkan::buffers::{create_shader_buffers, create_uniform_buffer};
use vulkan::command_buffers::{create_command_buffer, run_command_buffer};
use vulkan::command_pool::create_command_pool;
use vulkan::descriptors::{
    create_compute_descriptor_set_layout, create_descriptor_pool, create_descriptor_sets,
};
use vulkan::fps_counter::FPSCounter;
use vulkan::instance::create_instance;
use vulkan::logical_device::create_logical_device;
use vulkan::physical_device::{pick_physical_device, SuitabilityError};
use vulkan::pipeline::{create_compute_pipeline, create_render_pass};
use vulkan::sampler::create_sampler;
use vulkan::swapchain::{create_swapchain, create_swapchain_image_views};
use vulkan::sync_objects::create_sync_objects;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;
use vulkanalia::Version;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use vulkanalia::vk::ExtDebugUtilsExtension;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;

mod vulkan;

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
/// The name of the validation layers.
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// The required device extensions.
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];
/// The Vulkan SDK version that started requiring the portability subset extension for macOS.
const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

const TILE_SIZE: u32 = 16;

#[rustfmt::skip]
fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App

    let mut app = unsafe { App::create(&window)? };

    unsafe {
        let mut info_buffer = BufferBuilder::new();
        let mut triangle_buffer = BufferBuilder::new();



        info_buffer.append(4);
        info_buffer.pad(12);

        // Ground Sphere
        info_buffer.append(Mesh {
            // is_sphere: 0,
            triangle_count: 0,
            offset: triangle_buffer.get_relative_offset::<Triangle>()? as u32,
            material: Material {
                base_colour: AlignedVec4::new(0.7, 0.7, 0.7, 1.),
                emissive_strength: AlignedVec4::default(),
                reflectivity: Alignedf32(0.),
                roughness: Alignedf32(0.),
                is_glass: Alignedu32(0),
                ior: Alignedf32(0.),
                shade_smooth: Alignedu32(0),
            },
        });

        triangle_buffer.append_with_size(Sphere {
            center: AlignedVec3::new(0., -100., 5.),
            radius: Alignedf32(99.),
        }, size_of::<Triangle>());

        // Sun Sphere
        info_buffer.append(Mesh {
            triangle_count: 0,
            offset: triangle_buffer.get_relative_offset::<Triangle>()? as u32,
            material: Material {
                base_colour: AlignedVec4::default(),
                emissive_strength: AlignedVec4::new(15., 15., 15., 1.),
                reflectivity: Alignedf32(0.),
                roughness: Alignedf32(0.),
                is_glass: Alignedu32(0),
                ior: Alignedf32(0.),
                shade_smooth: Alignedu32(0),
            },
        });
        triangle_buffer.append_with_size(Sphere {
            center: AlignedVec3::new(-500., 200., 700.),
            radius: Alignedf32(200.),
        }, size_of::<Triangle>());

        let cube_tris: [i32; 108] = [
            0,0,0, 1,1,0, 1,0,0,
            0,0,0, 0,1,0, 1,1,0,
            1,0,0, 1,1,1, 1,0,1,
            1,0,0, 1,1,0, 1,1,1,
            1,0,1, 0,1,1, 1,1,1,
            1,0,1, 0,0,1, 0,1,1,
            0,0,1, 0,1,0, 0,0,0,
            0,0,1, 0,1,1, 0,1,0,
            0,1,0, 1,1,1, 1,1,0,
            0,1,0, 0,1,1, 1,1,1,
            0,0,0, 1,0,1, 1,0,0,
            0,0,0, 0,0,1, 1,0,1,
        ];


        // Test Triangle
        info_buffer.append(Mesh {
            triangle_count: 12,
            offset: triangle_buffer.get_relative_offset::<Triangle>()? as u32,
            material: Material {
                base_colour: AlignedVec4::new(0.7, 0.1, 0.1, 1.),
                emissive_strength: AlignedVec4::default(),
                reflectivity: Alignedf32(0.),
                roughness: Alignedf32(0.),
                is_glass: Alignedu32(1),
                ior: Alignedf32(1.4),
                shade_smooth: Alignedu32(0),
            },
        });

        for triangle in 0..12 {
            triangle_buffer.append(Triangle {
                vertices: [
                    AlignedVec4::new(-1.0 + (cube_tris[triangle * 9 + 0] as f32) * 2.0, -1.0 + (cube_tris[triangle * 9 + 1] as f32) * 2.0, 3.0 + (cube_tris[triangle * 9 + 2] as f32) * 2.0, 0.0),
                    AlignedVec4::new(-1.0 + (cube_tris[triangle * 9 + 3] as f32) * 2.0, -1.0 + (cube_tris[triangle * 9 + 4] as f32) * 2.0, 3.0 + (cube_tris[triangle * 9 + 5] as f32) * 2.0, 0.0),
                    AlignedVec4::new(-1.0 + (cube_tris[triangle * 9 + 6] as f32) * 2.0, -1.0 + (cube_tris[triangle * 9 + 7] as f32) * 2.0, 3.0 + (cube_tris[triangle * 9 + 8] as f32) * 2.0, 0.0),
                ],
                normals: [AlignedVec4::default(); 3],
            });
        }

        info_buffer.append(Mesh {
            triangle_count: 12,
            offset: triangle_buffer.get_relative_offset::<Triangle>()? as u32,
            material: Material {
                base_colour: AlignedVec4::new(0.2, 0.3, 0.8, 1.),
                emissive_strength: AlignedVec4::default(),
                reflectivity: Alignedf32(0.),
                roughness: Alignedf32(0.),
                is_glass: Alignedu32(0),
                ior: Alignedf32(1.4),
                shade_smooth: Alignedu32(0),
            },
        });

        for triangle in 0..12 {
            triangle_buffer.append(Triangle {
                vertices: [
                    AlignedVec4::new(1.0 + (cube_tris[triangle * 9 + 0] as f32) / 2.0, -1.0 + (cube_tris[triangle * 9 + 1] as f32) / 2.0, 2.0 + (cube_tris[triangle * 9 + 2] as f32) / 2.0, 0.0),
                    AlignedVec4::new(1.0 + (cube_tris[triangle * 9 + 3] as f32) / 2.0, -1.0 + (cube_tris[triangle * 9 + 4] as f32) / 2.0, 2.0 + (cube_tris[triangle * 9 + 5] as f32) / 2.0, 0.0),
                    AlignedVec4::new(1.0 + (cube_tris[triangle * 9 + 6] as f32) / 2.0, -1.0 + (cube_tris[triangle * 9 + 7] as f32) / 2.0, 2.0 + (cube_tris[triangle * 9 + 8] as f32) / 2.0, 0.0),
                ],
                normals: [AlignedVec4::default(); 3],
            });
        }

        create_descriptor_sets(&app.device, &mut app.data, info_buffer.get_offset() as u32)?;

        let mapped_ptr = app.device.map_memory(
            app.data.compute_ssbo_buffer_memory,
            0,
            4096,
            vk::MemoryMapFlags::empty(),
        )?;

        info_buffer.write(mapped_ptr);
        triangle_buffer.write(mapped_ptr.byte_add( info_buffer.get_offset()));

        println!("Total memory: {}", info_buffer.get_offset() + triangle_buffer.get_offset());

    }


    let mut minimized = false;
    event_loop.run(move |event, elwt| {
        match event {
            // Request a redraw when all events were processed.
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                // Render a frame if our Vulkan app is not being destroyed.
                WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                    unsafe { app.render(&window) }.unwrap();
                },
                // Mark the window as having been resized.
                WindowEvent::Resized(size) => {
                    if size.width == 0 || size.height == 0 {
                        minimized = true;
                    } else {
                        minimized = false;
                        app.resized = true;
                    }
                }
                // Destroy our Vulkan app.
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe { app.destroy(); }
                }
                _ => {}
            }
            _ => {}
        }
    })?;

    Ok(())
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    messenger: vk::DebugUtilsMessengerEXT,
    swapchain: vk::SwapchainKHR,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    render_pass: vk::RenderPass,
    present_queue: vk::Queue,
    compute_queue: vk::Queue,
    compute_pipeline: vk::Pipeline,
    compute_pipeline_layout: vk::PipelineLayout,

    // there is one of these per concurrenlty rendered image
    compute_descriptor_sets: Vec<vk::DescriptorSet>,
    compute_command_buffer: vk::CommandBuffer,

    // framebuffers: Vec<vk::Framebuffer>,
    compute_in_flight_fences: vk::Fence,
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
    start: Instant,
    fps_counter: FPSCounter,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry, &instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_compute_descriptor_set_layout(&device, &mut data)?;
        create_compute_pipeline(&device, &mut data)?;
        // create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        // create_vertex_buffer(&instance, &device, &mut data)?;
        create_uniform_buffer(&instance, &device, &mut data)?;
        create_shader_buffers(&instance, &device, &mut data)?;
        create_image(&instance, &device, &mut data)?;
        transition_image_layout(&device, &mut data)?;
        create_descriptor_pool(&device, &mut data)?;
        create_sampler(&device, &mut data)?;
        // create_descriptor_sets(&device, &mut data)?;
        create_command_buffer(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;
        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            fps_counter: FPSCounter::new(15)
        })
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        self.fps_counter.update();
        // self.fps_counter.print();

        self.device
            .wait_for_fences(&[self.data.compute_in_flight_fences], true, u64::MAX)?;

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
            self.data.compute_command_buffer,
            vk::CommandBufferResetFlags::empty(),
        )?;

        self.update_uniform_buffer(image_index)?;
        run_command_buffer(&self.device, &mut self.data, image_index)?;

        self.device
            .reset_fences(&[self.data.compute_in_flight_fences])?;

        let image_semaphores = &[self.data.image_available_semaphores];
        let compute_command_buffer = &[self.data.compute_command_buffer];
        let finish_semaphores = &[self.data.compute_finished_semaphores];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(image_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COMPUTE_SHADER])
            .command_buffers(compute_command_buffer)
            .signal_semaphores(finish_semaphores);

        self.device.queue_submit(
            self.data.compute_queue,
            &[submit_info],
            self.data.compute_in_flight_fences,
        )?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let compute_finished = &[self.data.compute_finished_semaphores];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(compute_finished)
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

        self.frame += 1;

        Ok(())
    }

    /// Updates the uniform buffer object for our Vulkan app.
    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        // MVP
        let time = (self.start.elapsed()).as_secs_f32();
        let origin = Vec4::new(0., 0., 0., 0.);

        let ratio =
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32;
        let (u, v) = if ratio > 1. {
            (ratio, 1.0f32)
        } else {
            (1., 1. / ratio)
        };
        let size = 2.0f32;

        let ubo = UniformBufferObject {
            resolution: Vec2::new(
                self.data.swapchain_extent.width as f32,
                self.data.swapchain_extent.height as f32,
            ),
            view_port_uv: Vec2::new(u, v),
            focal_length: Alignedf32(0.6),
            focus_distance: Alignedf32(4.8),
            aperture_radius: Alignedf32(0.0),
            time: Alignedu32(self.frame as u32),
            origin: AlignedVec4(origin),
            rotation: 
                AlignedMat4(Mat4::look_at_rh(
                    origin.truncate(),
                    Vec3::new(0.0, 0.0, -3.5),
                    Vec3::Y,
                )),
            };

        // Copy

        let memory = self.device.map_memory(
            self.data.uniform_buffer_memory,
            0,
            size_of::<UniformBufferObject>() as u64,
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

    /// Destroys our Vulkan app.
    #[rustfmt::skip]
    unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        // self.data.in_flight_fences.iter().for_each(|f| self.device.destroy_fence(*f, None));
        // self.data.render_finished_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        // self.data.image_available_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        // self.device.free_memory(self.data.shader_buffer_memory, None);
        // self.device.destroy_buffer(self.data.shader_buffer, None);
        self.device.destroy_command_pool(self.data.command_pool, None);
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
    }
}

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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    resolution: Vec2,
    view_port_uv: Vec2,
    focal_length: Alignedf32,
    focus_distance: Alignedf32,
    aperture_radius: Alignedf32,
    time: Alignedu32,
    origin: AlignedVec4,
    rotation: AlignedMat4,
}

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug)]
pub struct AlignedVec3(pub Vec3);
impl AlignedVec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }
}

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug)]
pub struct AlignedMat4(pub Mat4);

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug)]
pub struct AlignedVec4(pub Vec4);
impl AlignedVec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(Vec4::new(x, y, z, w))
    }

    pub fn default() -> Self {
        Self(Vec4::new(0., 0., 0., 0.))
    }
}

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug)]
pub struct Alignedf32(pub f32);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug)]
pub struct Alignedu32(pub u32);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug)]
pub struct AlignedBool(pub bool);

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Material {
    base_colour: AlignedVec4,
    emissive_strength: AlignedVec4,
    reflectivity: Alignedf32,
    roughness: Alignedf32,
    is_glass: Alignedu32,
    ior: Alignedf32,
    shade_smooth: Alignedu32,
}



#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Triangle {
    vertices: [AlignedVec4; 3],
    normals: [AlignedVec4; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Sphere {
    center: AlignedVec3,
    radius: Alignedf32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Mesh {
    triangle_count: u32,
    offset: u32,
    material: Material
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    const fn new(pos: Vec2, color: Vec3) -> Self {
        Self { pos, color }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec2>() as u32)
            .build();
        [pos, color]
    }
}
