use std::mem::size_of;

use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::vulkan::core::descriptors::{
    binding, create_descriptor_pool, create_descriptor_set_layout, pool_size,
};
use crate::vulkan::core::pipeline::{
    GraphicsPipelineConfig, create_graphics_pipeline, create_shader_module,
};
use crate::vulkan::core::sampler::{SamplerDesc, create_sampler};

use super::drawing::GuiVertex;

pub(super) const MAX_GUI_TEXTURES: u32 = 64;

pub(super) unsafe fn create_gui_sampler(device: &Device) -> Result<vk::Sampler> {
    create_sampler(
        device,
        &SamplerDesc {
            filter: vk::Filter::LINEAR,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            max_anisotropy: None,
        },
    )
}

pub(super) unsafe fn create_gui_descriptor_set_layout(
    device: &Device,
) -> Result<vk::DescriptorSetLayout> {
    let bindings = [binding(
        0,
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        1,
        vk::ShaderStageFlags::FRAGMENT,
    )];
    create_descriptor_set_layout(device, &bindings)
}

pub(super) unsafe fn create_gui_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = [pool_size(
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        MAX_GUI_TEXTURES,
    )];
    create_descriptor_pool(
        device,
        &pool_sizes,
        MAX_GUI_TEXTURES,
        vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
    )
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
    let set_layouts = [descriptor_set_layout];
    let info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&set_layouts)
        .push_constant_ranges(std::slice::from_ref(&push_constant));
    Ok(device.create_pipeline_layout(&info, None)?)
}

pub(super) unsafe fn create_gui_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
) -> Result<vk::Pipeline> {
    let vert = create_shader_module(device, include_bytes!("../../shaders/gui.vert.spv"))?;
    let frag = create_shader_module(device, include_bytes!("../../shaders/gui.frag.spv"))?;

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

    let vertex_bindings = gui_vertex_bindings();
    let vertex_attributes = gui_vertex_attributes();
    let blend_attachments = [alpha_blend_attachment()];
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

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

fn gui_vertex_bindings() -> [vk::VertexInputBindingDescription; 1] {
    [vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(size_of::<GuiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build()]
}

fn gui_vertex_attributes() -> [vk::VertexInputAttributeDescription; 3] {
    let pos_offset = 0u32;
    let uv_offset = size_of::<[f32; 2]>() as u32;
    let color_offset = (size_of::<[f32; 2]>() * 2) as u32;
    [
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
    ]
}

fn alpha_blend_attachment() -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState::builder()
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
        .build()
}
