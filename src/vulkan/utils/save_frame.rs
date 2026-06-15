use std::path::Path;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::BufferUsageFlags;

use crate::vulkan::core::{buffer::Buffer, context::VulkanContext};

pub struct SaveImage {
    pub buffer: Buffer,
    width: u32,
    height: u32,
}

impl SaveImage {
    pub unsafe fn new(ctx: &VulkanContext, width: u32, height: u32) -> Result<Self> {
        let size = 4_u64 * u64::from(width) * u64::from(height);
        let buffer = Buffer::new_host(ctx, size, BufferUsageFlags::TRANSFER_DST)?;

        Ok(Self {
            buffer,
            width,
            height,
        })
    }

    pub unsafe fn save_frame(&self, ctx: &VulkanContext, path: &Path) -> Result<()> {
        let size = self.buffer.size;
        let mapped = ctx
            .device
            .map_memory(self.buffer.memory, 0, size, vk::MemoryMapFlags::empty())?;
        let bgra = std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize);

        let mut rgba = vec![0_u8; bgra.len()];
        for (dst, src) in rgba.chunks_exact_mut(4).zip(bgra.chunks_exact(4)) {
            dst[0] = src[2]; // R
            dst[1] = src[1]; // G
            dst[2] = src[0]; // B
            dst[3] = src[3]; // A
        }

        ctx.device.unmap_memory(self.buffer.memory);

        image::save_buffer(
            path,
            &rgba,
            self.width,
            self.height,
            image::ExtendedColorType::Rgba8,
        )?;

        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.buffer.destroy(device);
    }
}
