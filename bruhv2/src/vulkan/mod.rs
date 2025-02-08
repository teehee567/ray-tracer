use ash::{vk, Device, Instance};
use std::{collections::HashMap, ffi::CString, fs, os::raw::c_void, path::Path, ptr, sync::Arc};
use vk_mem::{AllocationCreateInfo, Allocator, MemoryUsage};

const GLOBAL_DESCRIPTOR_POOL_SIZE: u32 = 1024;
const MAX_PUSH_CONSTANT_SIZE: usize = 128;

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
    extent: vk::Extent3D,
    usage: vk::ImageUsageFlags,
    bindless_indices: [u32; 2], // [sampled, storage]
}

struct Sampler {
    sampler: vk::Sampler,
    bindless_index: u32,
}

struct BindlessDescriptor {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
    counters: [u32; 3], // samplers, sampled_images, storage_images
}

pub struct VkApi {
    device: Arc<Device>,
    allocator: Allocator,
    descriptor_pool: vk::DescriptorPool,
    bindless: BindlessDescriptor,
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    samplers: Vec<Sampler>,
    push_constants: [u8; MAX_PUSH_CONSTANT_SIZE],
}

impl VkApi {
    pub fn new(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        device: Arc<Device>,
        queue_family: u32,
    ) -> Result<Self, vk::Result> {
        let allocator = unsafe {
            Allocator::new(vk_mem::AllocatorCreateInfo::new(
                instance,
                &device,
                physical_device,
            ))?
        };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: GLOBAL_DESCRIPTOR_POOL_SIZE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: GLOBAL_DESCRIPTOR_POOL_SIZE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: GLOBAL_DESCRIPTOR_POOL_SIZE,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: GLOBAL_DESCRIPTOR_POOL_SIZE,
            },
        ];

        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(256)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };

        let bindless_layout = create_bindless_descriptor_layout(&device)?;
        let pipeline_layout = create_pipeline_layout(&device, &bindless_layout)?;
        let descriptor_set =
            allocate_bindless_descriptor_set(&device, descriptor_pool, bindless_layout)?;

        Ok(Self {
            device,
            allocator,
            descriptor_pool,
            bindless: BindlessDescriptor {
                set_layout: bindless_layout,
                pipeline_layout,
                descriptor_set,
                counters: [0; 3],
            },
            buffers: Vec::new(),
            images: Vec::new(),
            samplers: Vec::new(),
            push_constants: [0; MAX_PUSH_CONSTANT_SIZE],
        })
    }

    pub fn create_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryUsage,
    ) -> Result<usize, vk::Result> {
        let (buffer, allocation) = self.allocator.create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(size)
                .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                .build(),
            &AllocationCreateInfo {
                usage: memory_usage,
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                ..Default::default()
            },
        )?;

        let device_address = unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder()
                    .buffer(buffer)
                    .build(),
            )
        };

        self.buffers.push(Buffer {
            buffer,
            allocation,
            size,
            device_address,
        });

        Ok(self.buffers.len() - 1)
    }

    pub fn create_image(
        &mut self,
        extent: vk::Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<usize, vk::Result> {
        let (image, allocation) = self.allocator.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .build(),
            &Default::default(),
        )?;

        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .build(),
                None,
            )?
        };

        let mut bindless_indices = [0; 2];
        if usage.contains(vk::ImageUsageFlags::STORAGE) {
            bindless_indices[1] = self.allocate_bindless_index(vk::DescriptorType::STORAGE_IMAGE);
            self.update_image_descriptor(
                bindless_indices[1],
                view,
                vk::DescriptorType::STORAGE_IMAGE,
            )?;
        }
        if usage.contains(vk::ImageUsageFlags::SAMPLED) {
            bindless_indices[0] = self.allocate_bindless_index(vk::DescriptorType::SAMPLED_IMAGE);
            self.update_image_descriptor(
                bindless_indices[0],
                view,
                vk::DescriptorType::SAMPLED_IMAGE,
            )?;
        }

        self.images.push(Image {
            image,
            allocation,
            view,
            format,
            extent,
            usage,
            bindless_indices,
        });

        Ok(self.images.len() - 1)
    }

    pub fn create_sampler(
        &mut self,
        filter: vk::Filter,
        address_mode: vk::SamplerAddressMode,
    ) -> Result<usize, vk::Result> {
        let sampler = unsafe {
            self.device.create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .mag_filter(filter)
                    .min_filter(filter)
                    .address_mode_u(address_mode)
                    .address_mode_v(address_mode)
                    .address_mode_w(address_mode)
                    .build(),
                None,
            )?
        };

        let index = self.allocate_bindless_index(vk::DescriptorType::SAMPLER);
        self.update_sampler_descriptor(index, sampler)?;

        self.samplers.push(Sampler {
            sampler,
            bindless_index: index,
        });

        Ok(self.samplers.len() - 1)
    }

    fn allocate_bindless_index(&mut self, ty: vk::DescriptorType) -> u32 {
        let index = match ty {
            vk::DescriptorType::SAMPLER => {
                let index = self.bindless.counters[0];
                self.bindless.counters[0] += 1;
                index
            }
            vk::DescriptorType::SAMPLED_IMAGE => {
                let index = self.bindless.counters[1];
                self.bindless.counters[1] += 1;
                index
            }
            vk::DescriptorType::STORAGE_IMAGE => {
                let index = self.bindless.counters[2];
                self.bindless.counters[2] += 1;
                index
            }
            _ => panic!("Unsupported bindless descriptor type"),
        };
        index
    }

    fn update_image_descriptor(
        &self,
        index: u32,
        view: vk::ImageView,
        ty: vk::DescriptorType,
    ) -> Result<(), vk::Result> {
        let binding = match ty {
            vk::DescriptorType::SAMPLED_IMAGE => 1,
            vk::DescriptorType::STORAGE_IMAGE => 2,
            _ => panic!("Invalid image descriptor type"),
        };

        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.bindless.descriptor_set)
            .dst_binding(binding)
            .dst_array_element(index)
            .descriptor_type(ty)
            .image_info(&[vk::DescriptorImageInfo::builder()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)
                .build()])
            .build();

        unsafe {
            self.device.update_descriptor_sets(&[write], &[]);
        }
        Ok(())
    }

    fn update_sampler_descriptor(
        &self,
        index: u32,
        sampler: vk::Sampler,
    ) -> Result<(), vk::Result> {
        let write = vk::WriteDescriptorSet::builder()
            .dst_set(self.bindless.descriptor_set)
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(&[vk::DescriptorImageInfo::builder()
                .sampler(sampler)
                .image_layout(vk::ImageLayout::UNDEFINED)
                .build()])
            .build();

        unsafe {
            self.device.update_descriptor_sets(&[write], &[]);
        }
        Ok(())
    }

    // Additional methods (command buffers, pipelines, etc.) would follow similar patterns
}

impl Drop for VkApi {
    fn drop(&mut self) {
        unsafe {
            // Cleanup all Vulkan resources
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.bindless.set_layout, None);
            self.device
                .destroy_pipeline_layout(self.bindless.pipeline_layout, None);

            for buffer in &self.buffers {
                self.allocator
                    .destroy_buffer(buffer.buffer, &buffer.allocation);
            }

            for image in &self.images {
                self.allocator.destroy_image(image.image, &image.allocation);
                self.device.destroy_image_view(image.view, None);
            }

            for sampler in &self.samplers {
                self.device.destroy_sampler(sampler.sampler, None);
            }
        }
    }
}

// Helper functions
fn create_bindless_descriptor_layout(
    device: &Device,
) -> Result<vk::DescriptorSetLayout, vk::Result> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE)
            .stage_flags(vk::ShaderStageFlags::ALL)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE)
            .stage_flags(vk::ShaderStageFlags::ALL)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(GLOBAL_DESCRIPTOR_POOL_SIZE)
            .stage_flags(vk::ShaderStageFlags::ALL)
            .build(),
    ];

    let flags = vec![
        vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
        vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING,
    ];

    let binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
        .binding_flags(&flags)
        .build();

    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&bindings)
        .push_next(&binding_flags)
        .build();

    unsafe { device.create_descriptor_set_layout(&layout_info, None) }
}

fn create_pipeline_layout(
    device: &Device,
    descriptor_layout: &vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, vk::Result> {
    let push_constants = [
        vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(64)
            .build(),
        vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS)
            .offset(64)
            .size(64)
            .build(),
    ];

    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[*descriptor_layout])
        .push_constant_ranges(&push_constants)
        .build();

    unsafe { device.create_pipeline_layout(&layout_info, None) }
}

fn allocate_bindless_descriptor_set(
    device: &Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, vk::Result> {
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&[layout])
        .build();

    unsafe { device.allocate_descriptor_sets(&allocate_info) }.map(|sets| sets[0])
}
