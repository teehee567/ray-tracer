use ash::{
    vk,
    version::{DeviceV1_0, InstanceV1_0},
    Device, Instance
};
use vk_mem::{Allocator, AllocatorCreateInfo, MemoryUsage};
use std::{
    path::Path,
    collections::{HashMap, VecDeque},
    ffi::CString,
    sync::Arc,
    mem,
    ptr
};

const GLOBAL_DESCRIPTOR_POOL_SIZE: u32 = 1024;

struct VulkanContext {
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    allocator: vk_mem::Allocator,
}

struct Buffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    size: vk::DeviceSize,
    device_address: vk::DeviceAddress,
}

struct Image {
    image: vk::Image,
    allocation: vk_mem::Allocation,
    view: vk::ImageView,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    bindless_storage_index: u32,
    bindless_sampled_index: u32,
}

struct Sampler {
    sampler: vk::Sampler,
    bindless_index: u32,
}

struct DescriptorAllocator {
    free_indices: HashMap<vk::DescriptorType, VecDeque<u32>>,
    max_indices: HashMap<vk::DescriptorType, u32>,
}

impl DescriptorAllocator {
    fn new() -> Self {
        let mut allocator = DescriptorAllocator {
            free_indices: HashMap::new(),
            max_indices: HashMap::new(),
        };
        
        allocator.free_indices.insert(vk::DescriptorType::SAMPLER, VecDeque::new());
        allocator.free_indices.insert(vk::DescriptorType::SAMPLED_IMAGE, VecDeque::new());
        allocator.free_indices.insert(vk::DescriptorType::STORAGE_IMAGE, VecDeque::new());
        
        allocator.max_indices.insert(vk::DescriptorType::SAMPLER, 0);
        allocator.max_indices.insert(vk::DescriptorType::SAMPLED_IMAGE, 0);
        allocator.max_indices.insert(vk::DescriptorType::STORAGE_IMAGE, 0);
        
        allocator
    }

    fn allocate(&mut self, ty: vk::DescriptorType) -> u32 {
        if let Some(free) = self.free_indices.get_mut(&ty) {
            if let Some(index) = free.pop_front() {
                return index;
            }
        }
        
        let max = self.max_indices.entry(ty).or_insert(0);
        *max += 1;
        *max - 1
    }

    fn free(&mut self, index: u32, ty: vk::DescriptorType) {
        self.free_indices.entry(ty).or_default().push_back(index);
    }
}

struct VulkanAPI {
    context: Arc<VulkanContext>,
    descriptor_pool: vk::DescriptorPool,
    bindless_descriptor_layout: vk::DescriptorSetLayout,
    bindless_pipeline_layout: vk::PipelineLayout,
    bindless_descriptor_set: vk::DescriptorSet,
    
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    samplers: Vec<Sampler>,
    
    descriptor_allocator: DescriptorAllocator,
}

impl VulkanAPI {
    pub fn new(context: Arc<VulkanContext>) -> Self {
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE)
                .build(),
            // Add other descriptor types...
        ];

        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1024)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
            
        let descriptor_pool = unsafe {
            context.device.create_descriptor_pool(&descriptor_pool_info, None)
                .expect("Failed to create descriptor pool")
        };

        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .build(),
            // Add other bindings...
        ];

        let binding_flags = vec![
            vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
            // Add other flags...
        ];

        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
            .binding_flags(&binding_flags);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .push_next(&mut flags_info);

        let bindless_descriptor_layout = unsafe {
            context.device.create_descriptor_set_layout(&layout_info, None)
                .expect("Failed to create descriptor set layout")
        };

        let push_constant_ranges = [
            vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(64)
                .build(),
            // Add other ranges...
        ];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&[bindless_descriptor_layout])
            .push_constant_ranges(&push_constant_ranges);

        let bindless_pipeline_layout = unsafe {
            context.device.create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        let allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&[bindless_descriptor_layout]);

        let bindless_descriptor_set = unsafe {
            context.device.allocate_descriptor_sets(&allocate_info)
                .expect("Failed to allocate descriptor sets")[0]
        };

        VulkanAPI {
            context,
            descriptor_pool,
            bindless_descriptor_layout,
            bindless_pipeline_layout,
            bindless_descriptor_set,
            buffers: Vec::new(),
            images: Vec::new(),
            samplers: Vec::new(),
            descriptor_allocator: DescriptorAllocator::new(),
        }
    }

    pub fn create_buffer(&mut self, size: vk::DeviceSize, usage: vk::BufferUsageFlags, memory_usage: MemoryUsage) -> usize {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: memory_usage,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };

        let (buffer, allocation, allocation_info) = self.context.allocator.create_buffer(&buffer_info, &allocation_info)
            .expect("Failed to create buffer");

        let device_address = unsafe {
            self.context.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder().buffer(buffer)
            )
        };

        let buffer = Buffer {
            buffer,
            allocation,
            size,
            device_address,
        };

        self.buffers.push(buffer);
        self.buffers.len() - 1
    }

    pub fn create_image(&mut self, extent: vk::Extent3D, format: vk::Format, usage: vk::ImageUsageFlags) -> usize {
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let allocation_info = vk_mem::AllocationCreateInfo {
            usage: MemoryUsage::GpuOnly,
            ..Default::default()
        };

        let (image, allocation, _) = self.context.allocator.create_image(&image_info, &allocation_info)
            .expect("Failed to create image");

        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe {
            self.context.device.create_image_view(&view_info, None)
                .expect("Failed to create image view")
        };

        let mut image = Image {
            image,
            allocation,
            view,
            format,
            usage,
            bindless_storage_index: 0,
            bindless_sampled_index: 0,
        };

        if usage.contains(vk::ImageUsageFlags::STORAGE) {
            image.bindless_storage_index = self.descriptor_allocator.allocate(vk::DescriptorType::STORAGE_IMAGE);
            self.update_descriptor_image(self.images.len(), vk::DescriptorType::STORAGE_IMAGE);
        }

        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            image.bindless_sampled_index = self.descriptor_allocator.allocate(vk::DescriptorType::SAMPLED_IMAGE);
            self.update_descriptor_image(self.images.len(), vk::DescriptorType::SAMPLED_IMAGE);
        }

        self.images.push(image);
        self.images.len() - 1
    }

    fn update_descriptor_image(&self, image_idx: usize, ty: vk::DescriptorType) {
        let image = &self.images[image_idx];
        let binding = match ty {
            vk::DescriptorType::STORAGE_IMAGE => 2,
            vk::DescriptorType::SAMPLED_IMAGE => 1,
            _ => panic!("Invalid descriptor type"),
        };

        let image_info = vk::DescriptorImageInfo::builder()
            .image_view(image.view)
            .image_layout(vk::ImageLayout::GENERAL);

        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.bindless_descriptor_set)
            .dst_binding(binding)
            .dst_array_element(match ty {
                vk::DescriptorType::STORAGE_IMAGE => image.bindless_storage_index,
                _ => image.bindless_sampled_index,
            })
            .descriptor_type(ty)
            .image_info(&[image_info.build()]);

        unsafe {
            self.context.device.update_descriptor_sets(&[write.build()], &[]);
        }
    }
}

impl Drop for VulkanAPI {
    fn drop(&mut self) {
        unsafe {
            // Cleanup all resources
            self.context.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.context.device.destroy_descriptor_set_layout(self.bindless_descriptor_layout, None);
            self.context.device.destroy_pipeline_layout(self.bindless_pipeline_layout, None);
            
            for buffer in &self.buffers {
                self.context.allocator.destroy_buffer(buffer.buffer, &buffer.allocation);
            }
            
            for image in &self.images {
                self.context.allocator.destroy_image(image.image, &image.allocation);
                self.context.device.destroy_image_view(image.view, None);
            }
            
            for sampler in &self.samplers {
                self.context.device.destroy_sampler(sampler.sampler, None);
            }
        }
    }
}
