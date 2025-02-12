use ash::{
    ext::debug_utils, khr::{surface, swapchain, }, vk, Entry, Instance
};
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
};
use vk_mem;

pub struct Queue {
    pub(crate) handle: vk::Queue,
    pub(crate) index: u32,
}

pub struct VkContext {
    pub(crate) entry: Entry,
    pub(crate) instance: Instance,
    pub(crate) surface_loader: surface::Instance,
    pub(crate) debug_utils: debug_utils::Instance,
    pub(crate) debug_messenger: vk::DebugUtilsMessengerEXT,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) graphics_queue: Queue,
    pub(crate) allocator: vk_mem::Allocator,
    pub(crate) swapchain_loader: swapchain::Device,
}

impl VkContext {
    pub fn new() -> Self {
        let entry = Entry::linked();
        let instance = create_instance(&entry);
        let surface_loader = surface::Instance::new(&entry, &instance);
        let debug_utils = debug_utils::Instance::new(&entry, &instance);
        let debug_messenger = create_debug_messenger(&debug_utils);
        let physical_device = select_physical_device(&instance);
        let (device, graphics_queue) = create_device(&instance, physical_device);
        let allocator = create_allocator(&instance, physical_device, &device);
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

        Self {
            entry,
            instance,
            surface_loader,
            debug_utils,
            debug_messenger,
            physical_device,
            device,
            graphics_queue,
            allocator,
            swapchain_loader,
        }
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.device.destroy_device(None);
        }
    }
}

fn create_instance(entry: &Entry) -> Instance {
    let app_info = vk::ApplicationInfo::default()
        .application_name(CStr::from_bytes_with_nul(b"path-tracer\0").unwrap())
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(CStr::from_bytes_with_nul(b"gpu-path-tracer\0").unwrap())
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::API_VERSION_1_2);

    let layer_names: Vec<CString> = if cfg!(debug_assertions) {
        vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()]
    } else {
        Vec::new()
    };
    let layer_ptrs: Vec<*const c_char> = layer_names.iter().map(|s| s.as_ptr()).collect();

    let mut extensions = vec![
        debug_utils::NAME.as_ptr(),
        surface::NAME.as_ptr(),
    ];

    #[cfg(target_os = "windows")]
    extensions.push(ash::khr::win32_surface::NAME.as_ptr());
    #[cfg(target_os = "macos")]
    extensions.push(ash::ext::metal_surface::NAME.as_ptr());

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs)
        .enabled_extension_names(&extensions);

    unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Failed to create instance")
    }
}

fn create_debug_messenger(debug_utils: &debug_utils::Instance) -> vk::DebugUtilsMessengerEXT {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    unsafe {
        debug_utils
            .create_debug_utils_messenger(&create_info, None)
            .expect("Failed to create debug messenger")
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let severity_str = match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "VERBOSE",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "INFO",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "WARNING",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "ERROR",
        _ => "UNKNOWN",
    };

    let type_str = match type_ {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "GENERAL",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VALIDATION",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERFORMANCE",
        _ => "UNKNOWN",
    };

    let message = CStr::from_ptr((*data).p_message);
    eprintln!("[{} {}]: {:?}", severity_str, type_str, message);

    vk::FALSE
}

fn select_physical_device(instance: &Instance) -> vk::PhysicalDevice {
    let devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
    };

    for &device in &devices {
        if is_device_suitable(instance, device) {
            return device;
        }
    }

    panic!("No suitable physical device found");
}

fn is_device_suitable(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .runtime_descriptor_array(true)
        .shader_storage_image_array_non_uniform_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_update_unused_while_pending(true)
        .imageless_framebuffer(true);

    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features12);

    unsafe {
        instance.get_physical_device_features2(device, &mut features2);
    }

    features12.buffer_device_address == vk::TRUE
        && features12.runtime_descriptor_array == vk::TRUE
        && features12.shader_storage_image_array_non_uniform_indexing == vk::TRUE
        && features12.shader_sampled_image_array_non_uniform_indexing == vk::TRUE
        && features12.descriptor_binding_partially_bound == vk::TRUE
        && features12.descriptor_binding_update_unused_while_pending == vk::TRUE
        && features12.imageless_framebuffer == vk::TRUE
}

fn create_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> (ash::Device, Queue) {
    let queue_family_index = find_queue_family(instance, physical_device);

    let queue_priorities = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    let extensions = [swapchain::NAME.as_ptr()];

    let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .runtime_descriptor_array(true)
        .shader_storage_image_array_non_uniform_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true)
        .descriptor_binding_partially_bound(true)
        .descriptor_binding_update_unused_while_pending(true)
        .imageless_framebuffer(true);

    let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features12);

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_info))
        .enabled_extension_names(&extensions)
        .push_next(&mut features2);

    let device = unsafe {
        instance
            .create_device(physical_device, &create_info, None)
            .expect("Failed to create device")
    };

    let graphics_queue = Queue {
        handle: unsafe { device.get_device_queue(queue_family_index, 0) },
        index: queue_family_index,
    };

    (device, graphics_queue)
}

fn find_queue_family(instance: &Instance, physical_device: vk::PhysicalDevice) -> u32 {
    let queue_properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    for (i, props) in queue_properties.iter().enumerate() {
        if props.queue_flags.contains(
            vk::QueueFlags::GRAPHICS
                | vk::QueueFlags::COMPUTE
                | vk::QueueFlags::TRANSFER,
        ) {
            return i as u32;
        }
    }

    panic!("No suitable queue family found");
}

fn create_allocator(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
) -> vk_mem::Allocator {
    let mut allocator_info = vk_mem::AllocatorCreateInfo::new(instance, device, physical_device);
    allocator_info.flags = vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

    unsafe {
        vk_mem::Allocator::new(allocator_info).expect("Failed to create allocator")
    }
}
