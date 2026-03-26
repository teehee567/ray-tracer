use std::mem::size_of;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::pipeline::create_shader_module;
use super::drawing::GuiVertex;

const MAX_GUI_TEXTURES: u32 = 64;

pub(super) unsafe fn create_gui_sampler(device: &Device) -> Result<vk::Sampler> {
    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .unnormalized_coordinates(false)
        .min_lod(0.0)
        .max_lod(0.0);

    Ok(device.create_sampler(&sampler_info, None)?)
}

pub(super) unsafe fn create_gui_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build();
    let bindings = [binding];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&layout_info, None)?)
}

pub(super) unsafe fn create_gui_pipeline_layout(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_constant = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(size_of::<[f32; 2]>() as u32)
        .build();
    let descriptor_set_layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&descriptor_set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_constant));

    Ok(device.create_pipeline_layout(&layout_info, None)?)
}

pub(super) unsafe fn create_gui_pipeline(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<vk::Pipeline> {
    let vert_shader = create_shader_module(device, include_bytes!("../../shaders/gui.vert.spv"))?;
    let frag_shader = create_shader_module(device, include_bytes!("../../shaders/gui.frag.spv"))?;

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader)
            .name(b"main\0")
            .build(),
        vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader)
            .name(b"main\0")
            .build(),
    ];

    let vertex_binding = vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(size_of::<GuiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build();
    let vertex_bindings = [vertex_binding];
    let pos_offset = 0u32;
    let uv_offset = size_of::<[f32; 2]>() as u32;
    let color_offset = (size_of::<[f32; 2]>() * 2) as u32;
    let attributes = [
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(pos_offset)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(uv_offset)
            .build(),
        vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(color_offset)
            .build(),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::_1);

    let blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .build();
    let blend_attachments = [blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachments);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?
        .0[0];

    device.destroy_shader_module(vert_shader, None);
    device.destroy_shader_module(frag_shader, None);

    Ok(pipeline)
}

pub(super) unsafe fn create_gui_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = [vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(MAX_GUI_TEXTURES)
        .build()];
    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(MAX_GUI_TEXTURES)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

    Ok(device.create_descriptor_pool(&pool_info, None)?)
}
