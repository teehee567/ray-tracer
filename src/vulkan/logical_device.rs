use std::collections::HashSet;

use log::info;
use vulkanalia::{prelude::v1_0::*, vk::InstanceV1_1};

use crate::{
    AppData, DEVICE_EXTENSIONS, PORTABILITY_MACOS_VERSION, QueueFamilyIndices, VALIDATION_ENABLED,
    VALIDATION_LAYER,
};
use anyhow::Result;

pub unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    // Queue Create Infos
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Layers
    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

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
        .shader_sampled_image_array_non_uniform_indexing(true);

    // Create
    let info = vk::DeviceCreateInfo::builder()
        .push_next(&mut features12)
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    // Check device properties
    let mut device_properties = vk::PhysicalDeviceProperties2::default();
    instance.get_physical_device_properties2(data.physical_device, &mut device_properties);

    let device = instance.create_device(data.physical_device, &info, None)?;
    info!("Created Logical Device, {:?}", device);

    // Queues
    data.compute_queue = device.get_device_queue(indices.compute, 0);
    info!("Created Compute Queue: {:?}", data.compute_queue);
    data.present_queue = device.get_device_queue(indices.present, 0);
    info!("Created Present Queue: {:?}", data.present_queue);

    Ok(device)
}
