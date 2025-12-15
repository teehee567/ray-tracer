use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{self, DeviceV1_0};

use crate::app::data::AppData;
use crate::vulkan::utils::get_memory_type_index;

/// Creates a host-visible buffer with coherent memory for dynamic GUI data.
/// Returns the buffer, device memory, and actual allocated size.
pub unsafe fn create_dynamic_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory, vk::DeviceSize)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size.max(1))
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(buffer);
    let memory_type = get_memory_type_index(
        instance,
        data,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        requirements,
    )?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);

    let memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(buffer, memory, 0)?;

    Ok((buffer, memory, requirements.size))
}

/// Uploads data to a host-visible buffer by mapping memory.
pub unsafe fn upload_to_buffer<T>(
    device: &Device,
    memory: vk::DeviceMemory,
    data: &[T],
) -> Result<()> {
    let size = (data.len() * std::mem::size_of::<T>()) as vk::DeviceSize;
    let ptr = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())? as *mut T;
    ptr.copy_from_nonoverlapping(data.as_ptr(), data.len());
    device.unmap_memory(memory);
    Ok(())
}

/// Destroys a buffer and frees its associated memory.
pub unsafe fn destroy_buffer(device: &Device, buffer: vk::Buffer, memory: vk::DeviceMemory) {
    if buffer != vk::Buffer::null() {
        device.destroy_buffer(buffer, None);
    }
    if memory != vk::DeviceMemory::null() {
        device.free_memory(memory, None);
    }
}
