use log::info;
use vulkanalia::{bytecode::Bytecode, prelude::v1_0::*};

use anyhow::Result;

pub unsafe fn create_compute_pipeline(
    device: &Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let binding = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&binding);
    let compute_pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    let compute_shader_src = include_bytes!("../../src/shaders/main.comp.spv");
    let compute_shader = create_shader_module(device, compute_shader_src)?;

    let compute_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_shader)
        .name(b"main\0");
    info!("Loaded Compute shader: {:?}", compute_shader);

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .layout(compute_pipeline_layout)
        .stage(compute_shader_stage_info);

    let compute_pipeline = device
        .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?
        .0[0];
    info!("Created a Compute Pipeline: {:?}", compute_pipeline);

    device.destroy_shader_module(compute_shader, None);

    Ok((compute_pipeline_layout, compute_pipeline))
}

pub unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    swapchain_format: vk::Format,
) -> Result<vk::RenderPass> {
    // attachments

    let color_attachment = vk::AttachmentDescription::builder()
        .format(swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // subpasses

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments);

    // dependencies

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    // create

    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    let render_pass = device.create_render_pass(&info, None)?;
    info!("Created a render_pass: {:?}", render_pass);

    Ok(render_pass)
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

pub unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}
