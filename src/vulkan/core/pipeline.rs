use log::info;
use vulkanalia::{bytecode::Bytecode, prelude::v1_0::*};

use anyhow::Result;

pub unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}

pub unsafe fn create_compute_pipeline(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    shader_spv: &[u8],
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let set_layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
    let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    let shader = create_shader_module(device, shader_spv)?;

    let stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader)
        .name(b"main\0");

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .layout(pipeline_layout)
        .stage(stage_info);

    let pipeline = device
        .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?
        .0[0];
    info!("Created a Compute Pipeline: {:?}", pipeline);

    device.destroy_shader_module(shader, None);

    Ok((pipeline_layout, pipeline))
}

pub struct GraphicsPipelineConfig<'a> {
    pub shaders: &'a [vk::PipelineShaderStageCreateInfo],
    pub vertex_bindings: &'a [vk::VertexInputBindingDescription],
    pub vertex_attributes: &'a [vk::VertexInputAttributeDescription],
    pub blend_attachments: &'a [vk::PipelineColorBlendAttachmentState],
    pub dynamic_states: &'a [vk::DynamicState],
    pub layout: vk::PipelineLayout,
    pub render_pass: vk::RenderPass,
    pub subpass: u32,
    pub topology: vk::PrimitiveTopology,
    pub cull_mode: vk::CullModeFlags,
}

pub unsafe fn create_graphics_pipeline(
    device: &Device,
    config: GraphicsPipelineConfig,
) -> Result<vk::Pipeline> {
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(config.vertex_bindings)
        .vertex_attribute_descriptions(config.vertex_attributes);

    let input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo::builder().topology(config.topology);

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(config.cull_mode)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::builder()
        .rasterization_samples(vk::SampleCountFlags::_1);

    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::builder().attachments(config.blend_attachments);

    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(config.dynamic_states);

    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(config.shaders)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(config.layout)
        .render_pass(config.render_pass)
        .subpass(config.subpass);

    let pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    Ok(pipeline)
}
