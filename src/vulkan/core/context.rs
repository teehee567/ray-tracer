use std::collections::HashSet;
use std::ffi::{CStr, c_void};

use anyhow::{Result, anyhow};
use log::{debug, error, info, trace, warn};
use thiserror::Error;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::window as vk_window;
use vulkanalia::{
    prelude::v1_0::*,
    vk::{ExtDebugUtilsExtensionInstanceCommands, KhrSurfaceExtensionInstanceCommands},
};
use winit::window::Window;

use crate::vulkan::constants::{
    DEVICE_EXTENSIONS, PORTABILITY_MACOS_VERSION, VALIDATION_BEST_PRACTICES, VALIDATION_DEBUG_PRINTF,
    VALIDATION_ENABLED, VALIDATION_GPU_ASSISTED, VALIDATION_LAYER, VALIDATION_SYNC,
};

#[derive(Debug, Error)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);

/// Owns the instance- and device-level Vulkan state shared by every renderer.
pub struct VulkanContext {
    /// Never read, but must stay alive: it owns the loaded Vulkan library.
    #[allow(dead_code)]
    pub entry: Entry,
    pub instance: Instance,
    pub messenger: vk::DebugUtilsMessengerEXT,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub queue_indices: QueueFamilyIndices,
    pub compute_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub command_pool: vk::CommandPool,
}

impl VulkanContext {
    pub unsafe fn new(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let (instance, messenger) = create_instance(window, &entry)?;
        let surface = vk_window::create_surface(&instance, window, window)?;
        let physical_device = pick_physical_device(&instance, surface)?;
        let queue_indices = QueueFamilyIndices::get(&instance, surface, physical_device)?;
        let (device, compute_queue, present_queue) =
            create_logical_device(&entry, &instance, physical_device, queue_indices)?;
        let command_pool = create_command_pool(&device, queue_indices)?;

        Ok(Self {
            entry,
            instance,
            messenger,
            surface,
            physical_device,
            device,
            queue_indices,
            compute_queue,
            present_queue,
            command_pool,
        })
    }

    pub unsafe fn destroy(&mut self) {
        self.device.destroy_command_pool(self.command_pool, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.surface, None);

        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.messenger, None);
        }

        self.instance.destroy_instance(None);
    }
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub compute: u32,
    pub present: u32,
    pub queue_count: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);
        if let Some((index, queue_count)) = properties.iter().enumerate().find_map(|(i, p)| {
            println!("{:#?}", p);
            if p.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                && instance
                    .get_physical_device_surface_support_khr(physical_device, i as u32, surface)
                    .unwrap_or(false)
            {
                Some((i as u32, p.queue_count))
            } else {
                None
            }
        }) {
            Ok(QueueFamilyIndices {
                graphics: index,
                compute: index,
                present: index,
                queue_count,
            })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, surface)?,
            formats: instance.get_physical_device_surface_formats_khr(physical_device, surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, surface)?,
        })
    }
}

unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
) -> Result<(Instance, vk::DebugUtilsMessengerEXT)> {
    // app info

    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"ray-tracer\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 2, 0));

    // layers

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut validation_enables: Vec<*const u8> = Vec::new();
    if VALIDATION_ENABLED {
        if VALIDATION_SYNC {
            validation_enables
                .push(b"VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT\0".as_ptr());
        }
        if VALIDATION_BEST_PRACTICES {
            validation_enables.push(b"VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT\0".as_ptr());
        }
        if VALIDATION_GPU_ASSISTED {
            validation_enables.push(b"VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT\0".as_ptr());
            validation_enables.push(
                b"VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT\0".as_ptr(),
            );
        }
        if VALIDATION_DEBUG_PRINTF {
            validation_enables.push(b"VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT\0".as_ptr());
        }
    }

    // extensions

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // macos portability since 1.3.216
    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
        if !validation_enables.is_empty() {
            extensions.push(vk::EXT_LAYER_SETTINGS_EXTENSION.name.as_ptr());
        }
    }

    // create

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .user_callback(Some(debug_callback));

    let validation_settings = [vk::LayerSettingEXT::builder()
        .layer_name(b"VK_LAYER_KHRONOS_validation\0")
        .setting_name(b"enables\0")
        .values_string(&validation_enables)
        .build()];
    let mut layer_settings_info =
        vk::LayerSettingsCreateInfoEXT::builder().settings(&validation_settings);

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
        if !validation_enables.is_empty() {
            info = info.push_next(&mut layer_settings_info);
        }
    }

    let instance = entry.create_instance(&info, None)?;

    // messenger

    let messenger = if VALIDATION_ENABLED {
        instance.create_debug_utils_messenger_ext(&debug_info, None)?
    } else {
        vk::DebugUtilsMessengerEXT::null()
    };

    Ok((instance, messenger))
}

extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

unsafe fn pick_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
) -> Result<vk::PhysicalDevice> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, surface, physical_device) {
            warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name, error
            );
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            return Ok(physical_device);
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn check_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, surface, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, surface, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}

unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    indices: QueueFamilyIndices,
) -> Result<(Device, vk::Queue, vk::Queue)> {
    // Queue Create Infos
    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0, 0.5];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Extensions
    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    // Features
    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .shader_storage_image_write_without_format(true)
        .shader_sampled_image_array_dynamic_indexing(true)
        .shader_storage_image_array_dynamic_indexing(true);

    // Vulkan 1.2 features (including descriptor indexing)
    let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
        .runtime_descriptor_array(true)
        .descriptor_indexing(true)
        .descriptor_binding_uniform_buffer_update_after_bind(true)
        .descriptor_binding_storage_image_update_after_bind(true)
        .descriptor_binding_storage_buffer_update_after_bind(true)
        .descriptor_binding_sampled_image_update_after_bind(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_variable_descriptor_count(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .timeline_semaphore(true);

    // Create. Note: device layers have been deprecated since Vulkan 1.0 — only instance
    // layers are used (the validation layer is already enabled at instance creation), so we
    // must not set enabled_layer_names here (enabledLayerCount must be 0).
    let info = vk::DeviceCreateInfo::builder()
        .push_next(&mut features12)
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(physical_device, &info, None)?;
    info!("Created Logical Device, {:?}", device);

    // Queues
    let present_queue = device.get_device_queue(indices.present, 0);
    info!("Created Present Queue: {:?}", present_queue);
    let compute_queue = device.get_device_queue(indices.compute, if indices.queue_count >= 2 { 1 } else { 0 });
    info!("Created Compute Queue: {:?}", compute_queue);

    Ok((device, compute_queue, present_queue))
}

unsafe fn create_command_pool(
    device: &Device,
    indices: QueueFamilyIndices,
) -> Result<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics);

    let command_pool = device.create_command_pool(&info, None)?;
    info!("Created Command Pool: {:?}", command_pool);

    Ok(command_pool)
}
