use log::info;
use vulkanalia::prelude::v1_0::*;

use crate::{AppData, CameraBufferObject};
use anyhow::Result;
use super::utils::create_buffer;


pub unsafe fn create_uniform_buffer(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    let (uniform_buffer, uniform_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size_of::<CameraBufferObject>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    data.uniform_buffer = uniform_buffer;
    data.uniform_buffer_memory = uniform_buffer_memory;
    info!("Created a uniform buffer with size {:?}", size_of::<CameraBufferObject>());

    Ok(())
}

pub unsafe fn create_shader_buffers(instance: &Instance, device: &Device, data: &mut AppData, size: u64) -> Result<()> {
    let (shader_buffer, shader_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;
    info!("Created a shader buffer with size: {:?}", size);

    data.compute_ssbo_buffer = shader_buffer;
    data.compute_ssbo_buffer_memory = shader_buffer_memory;
    Ok(())
}
