use std::{ffi::c_void, mem, ptr};

use anyhow::{Result, anyhow};
use vulkanalia::prelude::v1_0::*;

use super::context::VulkanContext;

#[derive(Clone, Debug, Default)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

pub struct BufferOpts {
    pub usage: vk::BufferUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
}

impl Default for BufferOpts {
    fn default() -> Self {
        Self {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            properties: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        }
    }
}

impl Buffer {
    pub unsafe fn new(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Self> {
        let info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = ctx.device.create_buffer(&info, None)?;

        let requirements = ctx.device.get_buffer_memory_requirements(buffer);
        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(get_memory_type_index(ctx, properties, requirements)?);

        let memory = ctx.device.allocate_memory(&memory_info, None)?;
        ctx.device.bind_buffer_memory(buffer, memory, 0)?;

        Ok(Self {
            buffer,
            memory,
            size,
        })
    }

    pub unsafe fn from_slice<T: Copy>(ctx: &VulkanContext, slice: &[T], opts: BufferOpts) -> Result<Self> {
        let size = mem::size_of_val(slice) as u64;
        let buffer = Self::new(ctx, size as u64, opts.usage, opts.properties)?;
        buffer.write(&ctx.device, slice)?;

        Ok(buffer)
    }

    pub unsafe fn new_host(
        ctx: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        Self::new(
            ctx,
            size,
            usage,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
    }


    pub unsafe fn write_with<F>(&self, device: &Device, size: vk::DeviceSize, f: F) -> Result<()>
    where
        F: FnOnce(*mut c_void),
    {
        let mapped = device.map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())?;
        f(mapped);
        device.unmap_memory(self.memory);
        Ok(())
    }

    /// host visible
    pub unsafe fn write<T: Copy>(&self, device: &Device, data: &[T]) -> Result<()> {
        let size = mem::size_of_val(data) as vk::DeviceSize;
        self.write_with(device, size, |mapped| {
            ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast(), data.len());
        })
    }

    /// recreats buffer if lower than needed
    pub unsafe fn ensure_capacity(
        &mut self,
        ctx: &VulkanContext,
        needed: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Result<()> {
        if self.buffer != vk::Buffer::null() && self.size >= needed {
            return Ok(());
        }
        self.destroy(&ctx.device);
        *self = Self::new_host(ctx, needed.max(1), usage)?;
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        if self.buffer != vk::Buffer::null() {
            device.destroy_buffer(self.buffer, None);
            self.buffer = vk::Buffer::null();
        }
        if self.memory != vk::DeviceMemory::null() {
            device.free_memory(self.memory, None);
            self.memory = vk::DeviceMemory::null();
        }
        self.size = 0;
    }
}

pub unsafe fn get_memory_type_index(
    ctx: &VulkanContext,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = ctx
        .instance
        .get_physical_device_memory_properties(ctx.physical_device);
    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}
