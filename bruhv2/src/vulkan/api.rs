use ash::vk::{self, Extent2D, Offset3D};
use ash::{Device, Entry, Instance};
use egui_ash::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use egui_ash::winit;
use std::collections::VecDeque;
use std::ffi::CStr;
use std::path::Path;
use std::{fs, slice};
use vk_mem::Alloc;
use vk_mem::{Allocation, AllocationCreateInfo, Allocator, MemoryUsage};

use anyhow::{anyhow, Result};

use super::context::VkContext;

const GLOBAL_DESCRIPTOR_POOL_SIZE: u32 = 1024;

pub fn include_spirv<P: AsRef<Path>>(path: P) -> Vec<u32> {
    let bytes = fs::read(&path).expect("Failed to read SPIR-V file");
    if bytes.len() % 4 != 0 {
        panic!("SPIR-V file length must be a multiple of 4 bytes");
    }
    let converter: fn([u8; 4]) -> u32 = match bytes[0] {
        0x03 => u32::from_le_bytes,
        0x07 => u32::from_be_bytes,
        _ => panic!("Unknown endianness in SPIR-V file"),
    };
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            // This unwrap will always succeed because we've checked that the length is a multiple of 4.
            let arr: [u8; 4] = chunk.try_into().unwrap();
            converter(arr)
        })
        .collect()
}

struct GlobalDescriptor {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
    index_pools: [VecDeque<u32>; 3],
}

impl GlobalDescriptor {
    fn new(device: &Device, descriptor_pool: vk::DescriptorPool) -> Self {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let binding_flags = vec![
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
            vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        ];

        let mut flags_create_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .push_next(&mut flags_create_info)
            .bindings(&bindings);

        let set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
            .expect("Failed to create descriptor set layout");

        let push_constants = [
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(64),
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)
                .offset(64)
                .size(64),
        ];
        let set_layout_array = [set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layout_array)
            .push_constant_ranges(&push_constants);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .expect("Failed to create pipeline layout");

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layout_array);

        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .expect("Failed to allocate descriptor set")[0];

        let mut index_pools = [
            VecDeque::from_iter((0..GLOBAL_DESCRIPTOR_POOL_SIZE).rev().map(|i| i + 1)),
            VecDeque::from_iter((0..GLOBAL_DESCRIPTOR_POOL_SIZE).rev().map(|i| i + 1)),
            VecDeque::from_iter((0..GLOBAL_DESCRIPTOR_POOL_SIZE).rev().map(|i| i + 1)),
        ];

        GlobalDescriptor {
            set_layout,
            pipeline_layout,
            descriptor_set,
            index_pools,
        }
    }

    fn allocate(&mut self, ty: vk::DescriptorType) -> u32 {
        let pool_idx = GlobalDescriptor::type_to_index(ty);
        self.index_pools[pool_idx].pop_front().unwrap_or(0)
    }

    fn free(&mut self, index: u32, ty: vk::DescriptorType) {
        let pool_idx = GlobalDescriptor::type_to_index(ty);
        self.index_pools[pool_idx].push_back(index);
    }

    fn type_to_index(ty: vk::DescriptorType) -> usize {
        match ty {
            vk::DescriptorType::SAMPLER => 0,
            vk::DescriptorType::SAMPLED_IMAGE => 1,
            vk::DescriptorType::STORAGE_IMAGE => 2,
            _ => panic!("Unsupported descriptor type"),
        }
    }
}

pub(crate) struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: Allocation,
    pub(crate) size: vk::DeviceSize,
    pub(crate) device_address: vk::DeviceAddress,
}

pub(crate) struct Image {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) allocation: Option<Allocation>,
    pub(crate) subresource_range: vk::ImageSubresourceRange,
    pub(crate) format: vk::Format,
    pub(crate) extent: vk::Extent3D,
    pub(crate) bindless_storage_index: Option<u32>,
    pub(crate) bindless_sampled_index: Option<u32>,
    pub(crate) previous_layout: vk::ImageLayout,
    pub(crate) previous_access: vk::AccessFlags,
    pub(crate) previous_stage: vk::PipelineStageFlags,
}

pub struct Sampler {
    pub(crate) sampler: vk::Sampler,
    pub(crate) bindless_index: u32,
}

pub struct VulkanApi {
    device: Device,
    allocator: Allocator,
    descriptor_pool: vk::DescriptorPool,
    bindless_descriptor: GlobalDescriptor,
    buffers: Vec<Option<Buffer>>,
    images: Vec<Option<Image>>,
    samplers: Vec<Option<Sampler>>,
    context: VkContext,
}

pub struct FrameBuffer {
    pub(crate) handle: vk::Framebuffer,
    pub(crate) size: vk::Extent2D,
}

pub(crate) struct Pipeline {
    pub(crate) shader_modules: Vec<vk::ShaderModule>,
    pub(crate) handle: vk::Pipeline,
    pub(crate) bind_point: vk::PipelineBindPoint,
}

pub(crate) struct Swapchain {
    pub(crate) image_count: u32,
    pub(crate) images: Vec<u32>,
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) surface_format: vk::SurfaceFormatKHR,
    pub(crate) extent: vk::Extent2D,
}

impl VulkanApi {
    pub fn new(context: VkContext, device: Device, allocator: Allocator) -> Self {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE),
        ];

        let create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(0xFF)
            .pool_sizes(&pool_sizes);

        let descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None) }
            .expect("Failed to create descriptor pool");

        let bindless_descriptor = GlobalDescriptor::new(&device, descriptor_pool);

        VulkanApi {
            device,
            allocator,
            descriptor_pool,
            bindless_descriptor,
            buffers: Vec::new(),
            images: Vec::new(),
            samplers: Vec::new(),
            context,
        }
    }

    pub fn create_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryUsage,
    ) -> usize {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_create_info = AllocationCreateInfo {
            usage: memory_usage,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let (buffer, allocation) = unsafe {
            self.allocator
                .create_buffer(&buffer_info, &allocation_create_info)
                .expect("Failed to create buffer")
        };
        let _allocation_info = self.allocator.get_allocation_info(&allocation);

        let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let device_address = unsafe { self.device.get_buffer_device_address(&address_info) };

        let buf = Buffer {
            buffer,
            allocation,
            size,
            device_address,
        };

        self.buffers.push(Some(buf));
        self.buffers.len() - 1
    }

    pub fn copy_buffer(
        &mut self,
        cmd_buf: vk::CommandBuffer,
        src: usize,
        dst: usize,
        _size: usize,
        buffer_offset: usize,
    ) {
        let dst_image = self.images[dst]
            .take()
            .expect("Out of range for self.images");

        let image_subresource_layers = vk::ImageSubresourceLayers::default()
            .aspect_mask(dst_image.subresource_range.aspect_mask)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);

        let buffer_to_image_copy = vk::BufferImageCopy::default()
            .buffer_offset(buffer_offset as u64)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(image_subresource_layers)
            .image_offset(Offset3D::default())
            .image_extent(dst_image.extent);
        let buffer_to_image_copy_array = [buffer_to_image_copy];

        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cmd_buf,
                self.buffers[src].take().expect("no buffer").buffer,
                dst_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &buffer_to_image_copy_array,
            )
        };
    }

    pub fn destroy_buffer(&mut self, handle: usize) {
        if let Some(mut buffer) = self.buffers[handle].take() {
            unsafe {
                self.allocator
                    .destroy_buffer(buffer.buffer, &mut buffer.allocation)
            };
        }
    }

    pub fn create_image(
        &mut self,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> usize {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let allocation_info = AllocationCreateInfo {
            usage: MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation) = unsafe {
            self.allocator
                .create_image(&image_info, &allocation_info)
                .expect("Failed to create image")
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .components(vk::ComponentMapping::default())
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe { self.device.create_image_view(&view_info, None) }
            .expect("Failed to create image view");

        let mut img = Image {
            image,
            view,
            allocation: Some(allocation),
            format,
            extent,
            bindless_storage_index: None,
            bindless_sampled_index: None,
            previous_layout: vk::ImageLayout::UNDEFINED,
            previous_access: vk::AccessFlags::empty(),
            previous_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };
        self.images.push(Some(img));

        let image_handle = self.images.len() - 1;
        if usage.contains(vk::ImageUsageFlags::STORAGE) {
            let index = self
                .bindless_descriptor
                .allocate(vk::DescriptorType::STORAGE_IMAGE);
            self.images[index as usize]
                .as_mut()
                .unwrap()
                .bindless_storage_index = Some(index);
            self.update_descriptor_image(image_handle, vk::DescriptorType::STORAGE_IMAGE);
        }

        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            let index = self
                .bindless_descriptor
                .allocate(vk::DescriptorType::SAMPLED_IMAGE);
            self.images[index as usize]
                .as_mut()
                .unwrap()
                .bindless_sampled_index = Some(index);
            self.update_descriptor_image(image_handle, vk::DescriptorType::SAMPLED_IMAGE);
        }

        image_handle
    }

    pub fn destroy_image(&mut self, image: usize) {
        let mut image = self.images[image].take().expect("no image");

        if let Some(index) = image.bindless_storage_index {
            self.bindless_descriptor
                .free(index, vk::DescriptorType::STORAGE_IMAGE);
        }

        if let Some(index) = image.bindless_sampled_index {
            self.bindless_descriptor
                .free(index, vk::DescriptorType::SAMPLED_IMAGE);
        }

        unsafe {
            self.allocator
                .destroy_image(image.image, &mut image.allocation.unwrap())
        };
    }

    pub fn create_fence(&self) -> vk::Fence {
        let create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        return unsafe {
            self.device
                .create_fence(&create_info, None)
                .expect("failed to create fence")
        };
    }

    pub fn destroy_fence(&self, fence: vk::Fence) {
        unsafe { self.device.destroy_fence(fence, None) };
    }

    pub fn create_fences(&self, fences_amt: usize) -> Vec<vk::Fence> {
        (0..fences_amt).map(|_| self.create_fence()).collect()
    }

    pub fn destroy_fences(&self, fences: Vec<vk::Fence>) {
        fences.iter().for_each(|f| self.destroy_fence(*f));
    }

    pub fn create_semaphore(&self) -> vk::Semaphore {
        let create_info = vk::SemaphoreCreateInfo::default();
        unsafe { self.device.create_semaphore(&create_info, None) }
            .expect("Failed to create semaphore")
    }

    pub fn destroy_semaphore(&self, semaphore: vk::Semaphore) {
        unsafe { self.device.destroy_semaphore(semaphore, None) };
    }

    pub fn create_semaphores(&self, semaphore_amt: usize) -> Vec<vk::Semaphore> {
        (0..semaphore_amt)
            .map(|_| self.create_semaphore())
            .collect()
    }

    pub fn destroy_semaphores(&self, semaphores: Vec<vk::Semaphore>) {
        semaphores.iter().for_each(|s| self.destroy_semaphore(*s));
    }

    pub fn create_sampler(
        &mut self,
        filter: vk::Filter,
        address_mode: vk::SamplerAddressMode,
    ) -> usize {
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter)
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .anisotropy_enable(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false);

        let sampler = unsafe { self.device.create_sampler(&create_info, None) }
            .expect("Failed to create sampler");

        let bindless_index = self
            .bindless_descriptor
            .allocate(vk::DescriptorType::SAMPLER);

        let sampler = Sampler {
            sampler,
            bindless_index,
        };

        self.update_descriptor_sampler(self.samplers.len());
        self.samplers.push(Some(sampler));
        self.samplers.len() - 1
    }

    pub fn destroy_sampler(&mut self, sampler: usize) {
        let sampler = self.samplers[sampler].take().expect("no sampler");

        self.bindless_descriptor
            .free(sampler.bindless_index, vk::DescriptorType::SAMPLER);

        unsafe { self.device.destroy_sampler(sampler.sampler, None) };
    }

    pub fn create_command_pool(&self) -> vk::CommandPool {
        let create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.context.graphics_queue.index);

        unsafe { self.device.create_command_pool(&create_info, None) }
            .expect("Failed to create command pool")
    }

    pub fn allocate_command_buffers(
        &self,
        pool: vk::CommandPool,
        count: usize,
    ) -> Vec<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(count as u32);

        unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .expect("Failed to allocate command buffers")
    }

    pub fn create_render_pass(
        &self,
        formats: &[vk::Format],
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> vk::RenderPass {
        let mut attachments = Vec::with_capacity(formats.len());
        for &format in formats {
            attachments.push(
                vk::AttachmentDescription::default()
                    .format(format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(initial_layout)
                    .final_layout(final_layout),
            );
        }

        let color_refs = (0..formats.len())
            .map(|i| vk::AttachmentReference {
                attachment: i as u32,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            })
            .collect::<Vec<_>>();

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs);
        let subpass_slice = [subpass];

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(1)
            .src_stage_mask(vk::PipelineStageFlags::NONE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags::ALL_GRAPHICS)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::INPUT_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_READ,
            )
            .dependency_flags(vk::DependencyFlags::empty());

        let dependency2 = vk::SubpassDependency::default()
            .src_subpass(1)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::ALL_GRAPHICS)
            .dst_stage_mask(vk::PipelineStageFlags::NONE_KHR)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .dependency_flags(vk::DependencyFlags::empty());

        let dependency_slice = [dependency, dependency2];

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpass_slice);
        // .dependencies(&dependency_slice);

        unsafe { self.device.create_render_pass(&create_info, None) }
            .expect("Failed to create render pass")
    }

    pub fn destroy_render_pass(&mut self, render_pass: vk::RenderPass) {
        unsafe { self.device.destroy_render_pass(render_pass, None) };
    }

    pub fn create_framebuffer(
        &self,
        render_pass: vk::RenderPass,
        formats: &[vk::Format],
        extent: vk::Extent2D,
    ) -> FrameBuffer {
        let attachment_infos: Vec<vk::FramebufferAttachmentImageInfo> = formats
            .iter()
            .map(|format| {
                let format_slice = std::slice::from_ref(format);
                vk::FramebufferAttachmentImageInfo::default()
                    .flags(vk::ImageCreateFlags::empty())
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                    .width(extent.width)
                    .height(extent.height)
                    .layer_count(1)
                    .view_formats(format_slice)
            })
            .collect();

        let mut attachments_create_info = vk::FramebufferAttachmentsCreateInfo::default()
            .attachment_image_infos(&attachment_infos);

        let create_info = vk::FramebufferCreateInfo::default()
            .push_next(&mut attachments_create_info)
            .flags(vk::FramebufferCreateFlags::IMAGELESS)
            .render_pass(render_pass)
            .width(extent.width)
            .height(extent.height)
            .layers(1);

        let handle = unsafe { self.device.create_framebuffer(&create_info, None) }
            .expect("Failed to create framebuffer");

        FrameBuffer {
            handle,
            size: extent,
        }
    }

    pub fn create_framebuffers(
        &self,
        render_pass: vk::RenderPass,
        formats: &[vk::Format],
        extent: vk::Extent2D,
        framebuffer_amt: usize,
    ) -> Vec<FrameBuffer> {
        (0..framebuffer_amt)
            .map(|_| self.create_framebuffer(render_pass, formats, extent))
            .collect()
    }

    pub fn destroy_framebuffer(&mut self, framebuffer: FrameBuffer) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("couldnt wait for device idle")
        };

        unsafe { self.device.destroy_framebuffer(framebuffer.handle, None) };
    }

    pub fn destroy_framebuffers(&mut self, framebuffers: Vec<FrameBuffer>) {
        framebuffers
            .into_iter()
            .for_each(|framebuffer| self.destroy_framebuffer(framebuffer));
    }

    pub fn create_compute_pipeline(&self, shader_path: &str) -> Pipeline {
        let shader_code = include_spirv(shader_path);
        let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let shader_module = unsafe { self.device.create_shader_module(&create_info, None) }
            .expect("Failed to create shader module");

        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(CStr::from_bytes_with_nul(b"main\0").unwrap());

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.bindless_descriptor.pipeline_layout);

        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
        }
        .expect("Failed to create compute pipeline")[0];

        Pipeline {
            handle: pipeline,
            shader_modules: vec![shader_module],
            bind_point: vk::PipelineBindPoint::COMPUTE,
        }
    }

    pub fn create_graphics_pipeline(
        &self,
        vertex_shader_path: &str,
        fragment_shader_path: &str,
        render_pass: vk::RenderPass,
        dynamic_states: &[vk::DynamicState],
    ) -> Pipeline {
        let shader_code = include_spirv(vertex_shader_path);
        let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let vert_module = unsafe { self.device.create_shader_module(&create_info, None) }
            .expect("Failed to create shader module");

        let shader_code = include_spirv(fragment_shader_path);
        let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let frag_module = unsafe { self.device.create_shader_module(&create_info, None) }
            .expect("Failed to create shader module");

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(CStr::from_bytes_with_nul(b"main\0").expect("cstr wrong")),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(CStr::from_bytes_with_nul(b"main\0").expect("cstr wrong")),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .min_sample_shading(1.0);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_attachment_state_slice = std::slice::from_ref(&color_blend_attachment);
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(color_attachment_state_slice);

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(self.bindless_descriptor.pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline_info_slice = std::slice::from_ref(&pipeline_info);
        let pipeline = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                pipeline_info_slice,
                None,
            )
        }
        .expect("Failed to create graphics pipeline")[0];

        Pipeline {
            shader_modules: vec![vert_module, frag_module],
            handle: pipeline,
            bind_point: vk::PipelineBindPoint::GRAPHICS,
        }
    }

    pub fn destroy_pipeline(&mut self, pipeline: &Pipeline) {
        pipeline
            .shader_modules
            .iter()
            .for_each(|s| unsafe { self.device.destroy_shader_module(*s, None) });

        unsafe { self.device.destroy_pipeline(pipeline.handle, None) };
    }

    pub fn create_surface(&self, window: &winit::window::Window) -> vk::SurfaceKHR {
        unsafe {
            ash_window::create_surface(
                &self.context.entry,
                &self.context.instance,
                window
                    .display_handle()
                    .expect("Failed to get display handle")
                    .as_raw(),
                window
                    .window_handle()
                    .expect("Failed to get window handle")
                    .as_raw(),
                None,
            )
            .expect("Failed to create surface")
        }
    }

    pub fn create_swapchain(
        &mut self,
        surface: vk::SurfaceKHR,
        min_image_count: u32,
        usages: vk::ImageUsageFlags,
        old_swapchain: vk::SwapchainKHR,
    ) -> Swapchain {
        // Check surface support
        let queue_support = unsafe {
            self.context
                .surface_loader
                .get_physical_device_surface_support(
                    self.context.physical_device,
                    self.context.graphics_queue.index as u32,
                    surface,
                )
        }
        .expect("Failed to get surface support");
        assert!(queue_support);

        // Get surface capabilities
        let caps = unsafe {
            self.context
                .surface_loader
                .get_physical_device_surface_capabilities(self.context.physical_device, surface)
        }
        .expect("Failed to get surface capabilities");

        // Determine swapchain extent
        let mut actual_extent = caps.current_extent;
        actual_extent.width = actual_extent
            .width
            .clamp(caps.min_image_extent.width, caps.max_image_extent.width);
        actual_extent.height = actual_extent
            .height
            .clamp(caps.min_image_extent.height, caps.max_image_extent.height);

        // Get surface formats
        let surface_formats = unsafe {
            self.context
                .surface_loader
                .get_physical_device_surface_formats(self.context.physical_device, surface)
        }
        .expect("Failed to get surface formats");
        let surface_format = surface_formats[0]; // Simplified format selection

        // Get present modes
        let present_modes = unsafe {
            self.context
                .surface_loader
                .get_physical_device_surface_present_modes(self.context.physical_device, surface)
        }
        .expect("Failed to get present modes");
        let present_mode = present_modes[0]; // Simplified present mode selection

        // Create swapchain
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(actual_extent)
            .image_array_layers(1)
            .image_usage(usages)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain_handle = unsafe {
            self.context
                .swapchain_loader
                .create_swapchain(&create_info, None)
        }
        .expect("Failed to create swapchain");

        // Get swapchain images
        let vk_images = unsafe {
            self.context
                .swapchain_loader
                .get_swapchain_images(swapchain_handle)
        }
        .expect("Failed to get swapchain images");

        let mut image_handles = Vec::with_capacity(vk_images.len());
        for image in vk_images {
            // Create image view
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let view = unsafe { self.device.create_image_view(&view_info, None) }
                .expect("Failed to create image view");

            // Allocate bindless descriptors
            let bindless_storage_index = if usages.contains(vk::ImageUsageFlags::STORAGE) {
                let index = self
                    .bindless_descriptor
                    .allocate(vk::DescriptorType::STORAGE_IMAGE);
                self.update_descriptor_image(
                    image_handles.len(),
                    vk::DescriptorType::STORAGE_IMAGE,
                );
                Some(index)
            } else {
                None
            };

            let bindless_sampled_index = if usages.contains(vk::ImageUsageFlags::SAMPLED) {
                let index = self
                    .bindless_descriptor
                    .allocate(vk::DescriptorType::SAMPLED_IMAGE);
                self.update_descriptor_image(
                    image_handles.len(),
                    vk::DescriptorType::SAMPLED_IMAGE,
                );
                Some(index)
            } else {
                None
            };

            // Create image handle
            let image_handle = self.images.len();
            self.images.push(Some(Image {
                image,
                view,
                format: surface_format.format,
                allocation: None,
                extent: vk::Extent3D {
                    width: actual_extent.width,
                    height: actual_extent.height,
                    depth: 1,
                },
                bindless_storage_index,
                bindless_sampled_index,
                previous_layout: vk::ImageLayout::UNDEFINED,
                previous_access: vk::AccessFlags::empty(),
                previous_stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            }));

            image_handles.push(image_handle as u32);
        }

        Swapchain {
            image_count: image_handles.len() as u32,
            images: image_handles,
            handle: swapchain_handle,
            surface_format,
            extent: actual_extent,
        }
    }

    pub fn destroy_swapchain(&mut self, swapchain: Swapchain) {
        unsafe { self.device.device_wait_idle().expect("failed idle") }
        for image in swapchain.images {
            if let Some(image) = &self.images[image as usize] {
                if let Some(bindless_storage_index) = image.bindless_storage_index {
                    self.bindless_descriptor
                        .free(bindless_storage_index, vk::DescriptorType::STORAGE_IMAGE);
                }
                if let Some(bindless_sampled_index) = image.bindless_sampled_index {
                    self.bindless_descriptor
                        .free(bindless_sampled_index, vk::DescriptorType::SAMPLED_IMAGE);
                }
                unsafe { self.device.destroy_image_view(image.view, None) };
            }
        }

        unsafe {
            self.context
                .swapchain_loader
                .destroy_swapchain(swapchain.handle, None)
        };
    }

    fn update_descriptor_image(&self, img: usize, ty: vk::DescriptorType) {
        let image = self.images[img].as_ref().unwrap();
        let binding = if ty == vk::DescriptorType::STORAGE_IMAGE {
            2
        } else {
            1
        };

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let image_info_slice = slice::from_ref(&image_info);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_descriptor.descriptor_set)
            .dst_binding(binding)
            .dst_array_element(if ty == vk::DescriptorType::STORAGE_IMAGE {
                image.bindless_storage_index.unwrap_or(0)
            } else {
                image.bindless_sampled_index.unwrap_or(0)
            })
            .descriptor_type(ty)
            .image_info(image_info_slice);
        let write_slice = slice::from_ref(&write);

        unsafe {
            self.device.update_descriptor_sets(write_slice, &[]);
        }
    }
    fn update_descriptor_images(&self, handles: &[usize], ty: vk::DescriptorType) {
        // Determine the binding based on the descriptor type.
        let binding = if ty == vk::DescriptorType::STORAGE_IMAGE {
            2
        } else {
            1
        };

        // Preallocate vectors for the descriptor image infos and write descriptor sets.
        let mut image_infos = Vec::with_capacity(handles.len());
        let mut writes = Vec::with_capacity(handles.len());

        // Create a descriptor image info for each image.
        for &handle in handles {
            let image = self.images[handle].as_ref().unwrap();
            let image_info = vk::DescriptorImageInfo::default()
                // For storage images, the sampler is typically VK_NULL_HANDLE.
                .sampler(vk::Sampler::null())
                .image_view(image.view)
                .image_layout(vk::ImageLayout::GENERAL);
            image_infos.push(image_info);
        }

        // Create the write descriptor sets for each image.
        for (i, &handle) in handles.iter().enumerate() {
            let image = self.images[handle].as_ref().unwrap();
            let dst_array_element = if ty == vk::DescriptorType::STORAGE_IMAGE {
                image.bindless_storage_index.unwrap_or(0)
            } else {
                image.bindless_sampled_index.unwrap_or(0)
            };

            // Here we pass a slice containing just one image info.
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.bindless_descriptor.descriptor_set)
                .dst_binding(binding)
                .dst_array_element(dst_array_element)
                .descriptor_type(ty)
                .image_info(&image_infos[i..i + 1]);
            writes.push(write);
        }

        // Finally, update the descriptor sets.
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn update_descriptor_sampler(&self, handle: usize) {
        let sampler = self.samplers[handle].as_ref().unwrap();
        let image_info = vk::DescriptorImageInfo::default()
            .sampler(sampler.sampler)
            .image_view(vk::ImageView::null())
            .image_layout(vk::ImageLayout::GENERAL);
        let image_info_slice = slice::from_ref(&image_info);

        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_descriptor.descriptor_set)
            .dst_binding(0)
            .dst_array_element(sampler.bindless_index)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(image_info_slice);
        let write_slice = slice::from_ref(&write);

        unsafe {
            self.device.update_descriptor_sets(write_slice, &[]);
        }
    }

    pub fn update_descriptor_samplers(&self, handles: &[usize]) {
        // Preallocate vectors for the descriptor image infos and write descriptor sets.
        let mut image_infos = Vec::with_capacity(handles.len());
        let mut writes = Vec::with_capacity(handles.len());

        // For each handle, create a descriptor image info for the sampler.
        for &handle in handles {
            let sampler = self.samplers[handle].as_ref().unwrap();
            let image_info = vk::DescriptorImageInfo::default()
                .sampler(sampler.sampler)
                .image_view(vk::ImageView::null()) // As in the single-sampler version.
                .image_layout(vk::ImageLayout::GENERAL);
            image_infos.push(image_info);
        }

        // Now, create a write descriptor set for each sampler.
        for (i, &handle) in handles.iter().enumerate() {
            let sampler = self.samplers[handle].as_ref().unwrap();
            // Each write descriptor set references a slice containing one image info.
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.bindless_descriptor.descriptor_set)
                .dst_binding(0) // Samplers are bound at binding 0.
                .dst_array_element(sampler.bindless_index)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&image_infos[i..i + 1]);
            writes.push(write);
        }

        // Finally, update the descriptor sets with all the writes at once.
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn update_constants(
        &self,
        cmd_buffer: vk::CommandBuffer,
        stage: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.device.cmd_push_constants(
                cmd_buffer,
                self.bindless_descriptor.pipeline_layout,
                stage,
                offset,
                data,
            )
        }
    }

    pub fn start_record(&self, command_buffer: vk::CommandBuffer) {
        let begin_info =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::empty());

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
        }
        .expect("Failed to begin command buffer");
    }

    pub fn image_barrier(
        &mut self,
        command_buffer: vk::CommandBuffer,
        image_handle: usize,
        new_layout: vk::ImageLayout,
        dst_stage: vk::PipelineStageFlags,
        dst_access: vk::AccessFlags,
    ) {
        let image = self.images[image_handle].as_ref().unwrap();

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(image.previous_layout)
            .new_layout(new_layout)
            .src_access_mask(image.previous_access)
            .dst_access_mask(dst_access)
            .image(image.image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(image.subresource_range);
        let barrier_slice = slice::from_ref(&barrier);

        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                image.previous_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                barrier_slice,
            );
        }

        // Update image state
        let image_mut = self.images[image_handle].as_mut().unwrap();
        image_mut.previous_layout = new_layout;
        image_mut.previous_access = dst_access;
        image_mut.previous_stage = dst_stage;
    }

    pub fn begin_render_pass(
        &self,
        cmd_buffer: vk::CommandBuffer,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
    ) {
        let clear_values = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_values_slice = slice::from_ref(&clear_values);

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values_slice);

        unsafe {
            self.device.cmd_begin_render_pass(
                cmd_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );
        }
    }

    pub fn end_render_pass(&mut self, command_buffer: vk::CommandBuffer) {
        unsafe { self.device.cmd_end_render_pass(command_buffer) };
    }

    pub fn run_compute_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline: &Pipeline,
        group_count_x: usize,
        group_count_y: usize,
        group_count_z: usize,
    ) {
        let descriptor_set_slice = slice::from_ref(&self.bindless_descriptor.descriptor_set);
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.bindless_descriptor.pipeline_layout,
                0,
                descriptor_set_slice,
                &[]
            );
            
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.handle);
            self.device.cmd_dispatch(command_buffer, group_count_x as u32, group_count_y as u32, group_count_z as u32);
        }
    }

    pub fn draw(&self, cmd_buffer: vk::CommandBuffer, pipeline: Pipeline, vertex_count: usize, vertex_offset: usize) {
        let descriptor_set_slice = slice::from_ref(&self.bindless_descriptor.descriptor_set);
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.bindless_descriptor.pipeline_layout,
                0,
                descriptor_set_slice,
                &[],
            );
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
            self.device.cmd_draw(cmd_buffer, vertex_count as u32, 1, vertex_offset as u32, 0);
        }
    }

    pub fn draw_indexed(
        &self,
        cmd_buffer: vk::CommandBuffer,
        pipeline: Pipeline,
        index_buffer: usize,
        index_count: usize,
        index_offset: usize,
        vertex_offset: usize,
    ) {
        let buffer = self.buffers[index_buffer].as_ref().unwrap();
        let descriptor_set_slice = slice::from_ref(&self.bindless_descriptor.descriptor_set);

        unsafe {
            self.device
                .cmd_bind_index_buffer(cmd_buffer, buffer.buffer, 0, vk::IndexType::UINT16);
            self.device.cmd_bind_descriptor_sets(
                cmd_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.bindless_descriptor.pipeline_layout,
                0,
                descriptor_set_slice,
                &[],
            );
            self.device
                .cmd_bind_pipeline(cmd_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.handle);
            self.device
                .cmd_draw_indexed(cmd_buffer, index_count as u32, 1, index_offset as u32, vertex_offset as i32, 0);
        }
    }

    pub fn blit_full(&self, cmd_buffer: vk::CommandBuffer, src_image: usize, dst_image: usize) {
        let src = self.images[src_image].as_ref().unwrap();
        let dst = self.images[dst_image].as_ref().unwrap();

        let blit = vk::ImageBlit::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: src.extent.width as i32,
                    y: src.extent.height as i32,
                    z: src.extent.depth as i32,
                },
            ])
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: dst.extent.width as i32,
                    y: dst.extent.height as i32,
                    z: 1,
                },
            ]);

        let blit_slice = slice::from_ref(&blit);
        unsafe {
            self.device.cmd_blit_image(
                cmd_buffer,
                src.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                blit_slice,
                vk::Filter::NEAREST,
            );
        }
    }

    pub fn end_record(&self, command_buffer: vk::CommandBuffer) {
        unsafe { self.device.end_command_buffer(command_buffer).unwrap(); }
    }

    pub fn submit(
        &self,
        command_buffers: &[vk::CommandBuffer],
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        submission_fence: vk::Fence,
    ) {
        let wait_stages = vk::PipelineStageFlags::TOP_OF_PIPE;
        let wait_stages_slice = slice::from_ref(&wait_stages);
        let mut submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        if wait_semaphores.len() > 0 {
            submit_info = submit_info.wait_dst_stage_mask(wait_stages_slice);
        }

        let submit_info_slice = slice::from_ref(&submit_info);
        unsafe { self.device.queue_submit(self.context.graphics_queue.handle, submit_info_slice, submission_fence) }
            .expect("Failed to submit command buffer");
    }

    pub fn present(
        &self,
        swapchain: Swapchain,
        image_index: u32,
        wait_semaphores: vk::Semaphore,
    ) {
        let wait_semaphores_slice = slice::from_ref(&wait_semaphores);
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores_slice)
            .swapchains(slice::from_ref(&swapchain.handle))
            .image_indices(slice::from_ref(&image_index));

        unsafe { self.context.swapchain_loader.queue_present(self.context.graphics_queue.handle, &present_info).unwrap(); }
    }

    pub fn get_image(&self, image_handle: usize) -> &Image {
        self.images[image_handle].as_ref().unwrap()
    }

    pub fn get_buffer(&self, buffer_handle: usize) -> &Buffer {
        self.buffers[buffer_handle].as_ref().unwrap()
    }

    pub fn get_sampler(&self, sampler_handle: usize) -> &Sampler {
        self.samplers[sampler_handle].as_ref().unwrap()
    }

}

impl Drop for VulkanApi {
    fn drop(&mut self) {
        unsafe {
            // Cleanup all Vulkan resources
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.bindless_descriptor.set_layout, None);
            self.device
                .destroy_pipeline_layout(self.bindless_descriptor.pipeline_layout, None);

        }
    }
}
