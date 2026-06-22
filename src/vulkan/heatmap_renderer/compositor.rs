use vulkanalia::prelude::v1_0::*;

use crate::vulkan::core::{
    buffer::Buffer,
    context::VulkanContext,
    descriptors::{
        allocate_descriptor_sets, binding, buffer_info, buffer_write, create_descriptor_pool,
        create_descriptor_set_layout, image_info, image_write, pool_size,
    },
    pipeline::{
        GraphicsPipelineConfig, create_compute_pipeline, create_graphics_pipeline,
        create_shader_module,
    },
    sampler::{SamplerDesc, create_sampler},
};
use anyhow::Result;

const REDUCE_GROUP: u32 = 16;

// finds per frame peak overlap and samples and tone maps it
pub(super) struct Compositer {
    sampler: vk::Sampler,
    max_buffer: Buffer,

    pool: vk::DescriptorPool,

    composite_set_layout: vk::DescriptorSetLayout,
    composite_layout: vk::PipelineLayout,
    composite_pipeline: vk::Pipeline,
    composite_set: vk::DescriptorSet,

    // maybe swithc to anon atomic very slow
    reduce_set_layout: vk::DescriptorSetLayout,
    reduce_layout: vk::PipelineLayout,
    reduce_pipeline: vk::Pipeline,
    reduce_set: vk::DescriptorSet,
}

impl Compositer {
    pub(super) unsafe fn new(
        ctx: &VulkanContext,
        swapchain_pass: vk::RenderPass,
        image_view: vk::ImageView,
    ) -> Result<Self> {
        let device = &ctx.device;

        let sampler = create_sampler(
            device,
            &SamplerDesc {
                filter: vk::Filter::NEAREST,
                address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                max_anisotropy: None,
            },
        )?;

        let max_buffer = Buffer::new(
            ctx,
            size_of::<u32>() as vk::DeviceSize,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let composite_bindings = [
            binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, vk::ShaderStageFlags::FRAGMENT),
            binding(1, vk::DescriptorType::STORAGE_BUFFER, 1, vk::ShaderStageFlags::FRAGMENT),
        ];
        let composite_set_layout = create_descriptor_set_layout(device, &composite_bindings)?;

        let reduce_bindings = [
            binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, vk::ShaderStageFlags::COMPUTE),
            binding(1, vk::DescriptorType::STORAGE_BUFFER, 1, vk::ShaderStageFlags::COMPUTE),
        ];
        let reduce_set_layout = create_descriptor_set_layout(device, &reduce_bindings)?;

        let pool_sizes = [
            pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 2),
            pool_size(vk::DescriptorType::STORAGE_BUFFER, 2),
        ];
        let pool = create_descriptor_pool(
            device,
            &pool_sizes,
            2,
            vk::DescriptorPoolCreateFlags::empty(),
        )?;

        let composite_set = allocate_descriptor_sets(device, pool, composite_set_layout, 1)?[0];
        let reduce_set = allocate_descriptor_sets(device, pool, reduce_set_layout, 1)?[0];

        let composite_layout = Self::create_pipeline_layout(device, composite_set_layout)?;
        let composite_pipeline =
            Self::create_compositor_pipeline(device, composite_layout, swapchain_pass)?;

        let (reduce_layout, reduce_pipeline) = create_compute_pipeline(
            device,
            reduce_set_layout,
            include_bytes!("../../shaders/compositor_reduce.comp.spv"),
        )?;

        let compositer = Self {
            sampler,
            max_buffer,
            pool,
            composite_set_layout,
            composite_layout,
            composite_pipeline,
            composite_set,
            reduce_set_layout,
            reduce_layout,
            reduce_pipeline,
            reduce_set,
        };

        compositer.update_composite_set(device, image_view);

        Ok(compositer)
    }

    pub(super) unsafe fn record_reduce(
        &self,
        device: &Device,
        cb: vk::CommandBuffer,
        width: u32,
        height: u32,
    ) {
        device.cmd_fill_buffer(cb, self.max_buffer.buffer, 0, vk::WHOLE_SIZE, 0);

        let to_compute = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .build();
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[to_compute],
            &[] as &[vk::BufferMemoryBarrier],
            &[] as &[vk::ImageMemoryBarrier],
        );

        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.reduce_pipeline);
        device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            self.reduce_layout,
            0,
            &[self.reduce_set],
            &[],
        );
        device.cmd_dispatch(
            cb,
            width.div_ceil(REDUCE_GROUP),
            height.div_ceil(REDUCE_GROUP),
            1,
        );

        let to_fragment = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[to_fragment],
            &[] as &[vk::BufferMemoryBarrier],
            &[] as &[vk::ImageMemoryBarrier],
        );
    }

    pub(super) unsafe fn record_into(
        &self,
        device: &Device,
        cb: vk::CommandBuffer,
        sub_region: vk::Rect2D,
    ) {
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.composite_pipeline);

        let viewport = vk::Viewport {
            x: sub_region.offset.x as f32,
            y: sub_region.offset.y as f32,
            width: sub_region.extent.width as f32,
            height: sub_region.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        device.cmd_set_viewport(cb, 0, &[viewport]);
        device.cmd_set_scissor(cb, 0, &[sub_region]);

        device.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::GRAPHICS,
            self.composite_layout,
            0,
            &[self.composite_set],
            &[],
        );

        device.cmd_draw(cb, 3, 1, 0, 0);
    }

    pub(super) unsafe fn update_composite_set(&self, device: &Device, image_view: vk::ImageView) {
        let image = [image_info(
            self.sampler,
            image_view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )];
        let max_buf = [buffer_info(self.max_buffer.buffer, 0, vk::WHOLE_SIZE)];

        let writes = [
            image_write(
                self.composite_set,
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &image,
            ),
            buffer_write(self.composite_set, 1, vk::DescriptorType::STORAGE_BUFFER, &max_buf),
            image_write(
                self.reduce_set,
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                &image,
            ),
            buffer_write(self.reduce_set, 1, vk::DescriptorType::STORAGE_BUFFER, &max_buf),
        ];

        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }

    unsafe fn create_pipeline_layout(
        device: &Device,
        set_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout> {
        let set_layouts = [set_layout];
        let info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
        Ok(device.create_pipeline_layout(&info, None)?)
    }

    unsafe fn create_compositor_pipeline(
        device: &Device,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
    ) -> Result<vk::Pipeline> {
        let vert =
            create_shader_module(device, include_bytes!("../../shaders/compositor.vert.spv"))?;
        let frag =
            create_shader_module(device, include_bytes!("../../shaders/compositor.frag.spv"))?;

        let shaders = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(b"main\0")
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(b"main\0")
                .build(),
        ];

        let vertex_bindings = [];
        let vertex_attributes = [];

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let blend_attachments = [Self::composite_blend_attachment()];

        let pipeline = create_graphics_pipeline(
            device,
            GraphicsPipelineConfig {
                shaders: &shaders,
                vertex_bindings: &vertex_bindings,
                vertex_attributes: &vertex_attributes,
                blend_attachments: &blend_attachments,
                dynamic_states: &dynamic_states,
                layout,
                render_pass,
                subpass: 0,
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                cull_mode: vk::CullModeFlags::NONE,
            },
        )?;

        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);

        Ok(pipeline)
    }

    fn composite_blend_attachment() -> vk::PipelineColorBlendAttachmentState {
        vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build()
    }

    pub(super) unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_pipeline(self.composite_pipeline, None);
        device.destroy_pipeline_layout(self.composite_layout, None);
        device.destroy_pipeline(self.reduce_pipeline, None);
        device.destroy_pipeline_layout(self.reduce_layout, None);
        device.destroy_descriptor_pool(self.pool, None);
        device.destroy_descriptor_set_layout(self.composite_set_layout, None);
        device.destroy_descriptor_set_layout(self.reduce_set_layout, None);
        self.max_buffer.destroy(device);
        device.destroy_sampler(self.sampler, None);
    }
}
