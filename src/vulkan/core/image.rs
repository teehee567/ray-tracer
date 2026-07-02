use anyhow::{Result, anyhow};
use vulkanalia::prelude::v1_0::*;

use super::buffer::{Buffer, get_memory_type_index};
use super::commands::with_single_time;
use super::context::VulkanContext;

/// A 2D image with its backing memory and a default view.
#[derive(Clone, Debug, Default)]
pub struct Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub width: u32,
    pub height: u32,
}

/// Parameters for [`Image::new_2d`]. `..Default::default()` covers the
/// common single-layer 2D DEVICE_LOCAL case.
#[derive(Clone, Copy, Debug)]
pub struct ImageDesc {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
    pub layer_count: u32,
    pub flags: vk::ImageCreateFlags,
    pub view_type: vk::ImageViewType,
}

impl Default for ImageDesc {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            format: vk::Format::UNDEFINED,
            usage: vk::ImageUsageFlags::empty(),
            properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            layer_count: 1,
            flags: vk::ImageCreateFlags::empty(),
            view_type: vk::ImageViewType::_2D,
        }
    }
}

impl Image {
    pub unsafe fn new_2d(ctx: &VulkanContext, desc: &ImageDesc) -> Result<Self> {
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::_2D)
            .format(desc.format)
            .extent(vk::Extent3D {
                width: desc.width.max(1),
                height: desc.height.max(1),
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(desc.layer_count)
            .samples(vk::SampleCountFlags::_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .flags(desc.flags);

        let image = ctx.device.create_image(&info, None)?;

        let requirements = ctx.device.get_image_memory_requirements(image);
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(get_memory_type_index(ctx, desc.properties, requirements)?);

        let memory = ctx.device.allocate_memory(&alloc_info, None)?;
        ctx.device.bind_image_memory(image, memory, 0)?;

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(desc.view_type)
            .format(desc.format)
            .subresource_range(subresource_range(desc.layer_count));

        let view = ctx.device.create_image_view(&view_info, None)?;

        Ok(Self {
            image,
            memory,
            view,
            width: desc.width,
            height: desc.height,
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        if self.view != vk::ImageView::null() {
            device.destroy_image_view(self.view, None);
            self.view = vk::ImageView::null();
        }
        if self.image != vk::Image::null() {
            device.destroy_image(self.image, None);
            self.image = vk::Image::null();
        }
        if self.memory != vk::DeviceMemory::null() {
            device.free_memory(self.memory, None);
            self.memory = vk::DeviceMemory::null();
        }
    }
}

pub fn subresource_range(layer_count: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(layer_count)
        .build()
}

pub fn image_barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    layer_count: u32,
) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range(layer_count))
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .build()
}

/// Record a single image layout transition barrier.
pub unsafe fn cmd_image_barrier(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    barrier: vk::ImageMemoryBarrier,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );
}

/// Record a layout transition for one of the known transition pairs.
pub unsafe fn cmd_transition_image_layout(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    layer_count: u32,
) -> Result<()> {
    let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
        ),
        _ => {
            return Err(anyhow!(
                "Unsupported image layout transition: {:?} -> {:?}",
                old_layout,
                new_layout
            ));
        }
    };

    let barrier = image_barrier(
        image,
        old_layout,
        new_layout,
        src_access,
        dst_access,
        layer_count,
    );
    cmd_image_barrier(device, command_buffer, barrier, src_stage, dst_stage);

    Ok(())
}

/// Upload RGBA8 pixels into `image` via a staging buffer, leaving it
/// SHADER_READ_ONLY_OPTIMAL. For layered images (e.g. cubemaps) `pixels`
/// holds all layers contiguously. `old_layout` is UNDEFINED for freshly
/// created images, or the image's current layout for updates.
pub unsafe fn upload_pixels(
    ctx: &VulkanContext,
    image: vk::Image,
    pixels: &[u8],
    size: [u32; 2],
    offset: [u32; 2],
    layer_count: u32,
    old_layout: vk::ImageLayout,
) -> Result<()> {
    let mut staging = Buffer::new_host(
        ctx,
        pixels.len() as vk::DeviceSize,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    staging.write(&ctx.device, pixels)?;

    with_single_time(&ctx.device, ctx.command_pool, ctx.compute_queue, |cb| {
        cmd_transition_image_layout(
            &ctx.device,
            cb,
            image,
            old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            layer_count,
        )?;

        let layer_size = (size[0] * size[1] * 4) as u64;
        let regions: Vec<_> = (0..layer_count)
            .map(|layer| {
                vk::BufferImageCopy::builder()
                    .buffer_offset(layer_size * layer as u64)
                    .image_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(layer)
                            .layer_count(1)
                            .build(),
                    )
                    .image_offset(vk::Offset3D {
                        x: offset[0] as i32,
                        y: offset[1] as i32,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: size[0],
                        height: size[1],
                        depth: 1,
                    })
                    .build()
            })
            .collect();

        ctx.device.cmd_copy_buffer_to_image(
            cb,
            staging.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
        );

        cmd_transition_image_layout(
            &ctx.device,
            cb,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            layer_count,
        )?;
        Ok(())
    })?;

    staging.destroy(&ctx.device);

    Ok(())
}

/// Parameters for [`create_texture`]. `..Default::default()` covers a plain
/// single-layer sRGB texture.
#[derive(Clone, Copy, Debug)]
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub layer_count: u32,
    pub flags: vk::ImageCreateFlags,
    pub view_type: vk::ImageViewType,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            format: vk::Format::R8G8B8A8_SRGB,
            layer_count: 1,
            flags: vk::ImageCreateFlags::empty(),
            view_type: vk::ImageViewType::_2D,
        }
    }
}

/// Create a sampled texture (optionally layered, e.g. a cubemap) and fill it
/// with `pixels`.
pub unsafe fn create_texture(
    ctx: &VulkanContext,
    pixels: &[u8],
    desc: &TextureDesc,
) -> Result<Image> {
    let image = Image::new_2d(
        ctx,
        &ImageDesc {
            width: desc.width,
            height: desc.height,
            format: desc.format,
            usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            layer_count: desc.layer_count,
            flags: desc.flags,
            view_type: desc.view_type,
            ..Default::default()
        },
    )?;
    upload_pixels(
        ctx,
        image.image,
        pixels,
        [desc.width, desc.height],
        [0, 0],
        desc.layer_count,
        vk::ImageLayout::UNDEFINED,
    )?;
    Ok(image)
}
