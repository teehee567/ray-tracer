
use core::slice;

use vulkanalia::{prelude::v1_0::*, vk::ImageView};

use anyhow::Result;
use crate::{scene::Scene, vulkan::core::{context::VulkanContext, descriptors::{allocate_descriptor_sets, binding, create_descriptor_pool, create_descriptor_set_layout, image_info, image_write, pool_size}, pipeline::{GraphicsPipelineConfig, create_graphics_pipeline, create_shader_module}, sampler::{SamplerDesc, create_sampler}}};

pub struct Compositer {
    sampler: vk::Sampler,
    composite_set_layout: vk::DescriptorSetLayout,
    composite_pool: vk::DescriptorPool,
    composite_set: vk::DescriptorSet,
    composite_layout: vk::PipelineLayout,
    composite_pipeline: vk::Pipeline,
}

impl Compositer {
    pub(crate) unsafe fn new(
        ctx: &VulkanContext,
        swapchain_pass: vk::RenderPass,
        extent: vk::Extent2D,
        scene: &Scene,
    ) -> Result<Self> {
        let device = &ctx.device;

        let sampler = create_sampler(device, &SamplerDesc {
            filter: vk::Filter::NEAREST,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            max_anisotropy: None,
        })?;

        let bindings = [binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, vk::ShaderStageFlags::FRAGMENT)];

        let set_layout = create_descriptor_set_layout(device, &bindings)?;

        let pool_sizes = [pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1)];
        let pool = create_descriptor_pool(device, &pool_sizes, 1, vk::DescriptorPoolCreateFlags::empty())?;

        let set = allocate_descriptor_sets(device, pool, set_layout, 1)?[0];

        let pipeline_layout = Self::create_pipeline_layout(device, set_layout)?;
        let pipeline = Self::


        Ok(Self {
            sampler,
            composite_set_layout: set_layout,
            composite_pool: pool,
            composite_set: set,
            composite_layout: pipeline_layout,
            composite_pipeline: 

        })
    }

    unsafe fn update_composite_set(&self, device: &Device, image_view: vk::ImageView) {
        let infos = [image_info(self.sampler, image_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = image_write(self.composite_set, 0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, &infos);

        device.update_descriptor_sets(&[write], &[] as &[vk::CopyDescriptorSet]);
    }

    unsafe fn create_pipeline_layout(device: &Device, set_layout: vk::DescriptorSetLayout) -> Result<vk::PipelineLayout> {
        let push_constant = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(size_of::<f32>() as u32)
            .build();

        let set_layouts = [set_layout];
        let info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(slice::from_ref(&push_constant));

        Ok(device.create_pipeline_layout(&info, None)?)

    }


    unsafe fn create_heatmap_pipeline(
        device: &Device,
        layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
    ) -> Result<vk::Pipeline> {
        let vert = create_shader_module(device, include_bytes!("../../shaders/compositor.vert.spv"))?;
        let frag = create_shader_module(device, include_bytes!("../../shaders/compositor.frag.spv"))?;

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
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::builder().blend_enable(false).build()];

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

        Ok(pipeline)
    }

}
