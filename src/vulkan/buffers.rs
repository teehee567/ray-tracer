use log::info;
use vulkanalia::prelude::v1_0::*;

use super::utils::create_buffer;
use crate::{AppData, CameraBufferObject};
use anyhow::Result;

pub unsafe fn create_uniform_buffer(
    instance: &Instance,
    device: &Device,
    data: &AppData,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let (uniform_buffer, uniform_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size_of::<CameraBufferObject>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    info!(
        "Created a uniform buffer with size {:?}",
        size_of::<CameraBufferObject>()
    );

    Ok((uniform_buffer, uniform_buffer_memory))
}

pub unsafe fn create_shader_buffers(
    instance: &Instance,
    device: &Device,
    data: &AppData,
    size: u64,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let (shader_buffer, shader_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    info!("Created a shader buffer with size: {:?}", size);

    Ok((shader_buffer, shader_buffer_memory))
}
