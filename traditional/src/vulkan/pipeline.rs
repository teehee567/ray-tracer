//================================================
// Graphics Pipeline
//================================================

use anyhow::{anyhow, Result};
use vulkanalia::{bytecode::Bytecode, prelude::v1_0::*};

use crate::AppData;

/// Graphics Pipeline entry
pub unsafe fn create_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    let vert = include_bytes!("../../shaders/vert.spv");
    let frag = include_bytes!("../../shaders/frag.spv");

    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        // Could have values like
        // POINT_LIST - points from vertices.
        // LINE_LIST - line from every 2 vertices without reuse.
        // LINE_STRIP - the end vertex of eery line is used as start vertex for next line.
        // TRIANLGE_LIST - triangle from every 3 vertices without reuse.
        // TRIANGLE_STRIP - the second and third vertex of every triangle are.
        //                  used as first two vertices of the next triangle
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // Viewport, area where vertexes can be mapped
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    // Scissor rectangle, area where fragment shading is allowed.
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    // Rasterizing
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        // Could have values like
        // FILL - fill the area of hte polygon with fragments.
        // LINE - polygon edges are drawn as lines.
        // POINT - polygon vertices are drawn as points.
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
        .depth_bias_enable(false);

    // Mulitsampling.
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        // Amount of samples MSAA xx
        .rasterization_samples(vk::SampleCountFlags::_1);

    // Color Blending.
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false);
    // .src_color_blend_factor(vk::BlendFactor::ONE)
    // .dst_color_blend_factor(vk::BlendFactor::ZERO)
    // .color_blend_op(vk::BlendOp::ADD)
    // .src_alpha_blend_factor(vk::BlendFactor::ONE)
    // .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
    // .alpha_blend_op(vk::BlendOp::ADD);

    //     // final_color.rgb = new_alpha * new_color + (1 - new_alpha) * old_color;
    //     // final_color.a = new_alpha.a;
    //     // Requires color_blend_state logic_op to be enabled
    // let attachment = vk::PipelineColorBlendAttachmentState::builder()
    //     .color_write_mask(vk::ColorComponentFlags::all())
    //     .blend_enable(false)
    //     .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
    //     .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
    //     .color_blend_op(vk::BlendOp::ADD)
    //     .src_alpha_blend_factor(vk::BlendFactor::ONE)
    //     .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
    //     .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    //WARN: huh
    //
    // let dynamic_states = &[
    //     vk::DynamicState::VIEWPORT,
    //     vk::DynamicState::LINE_WIDTH,
    // ];
    //
    // let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
    //     .dynamic_states(dynamic_states);

    // Layout

    let layout_info = vk::PipelineLayoutCreateInfo::builder();

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // Create

    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        // provide array of vk::PipelineShaderStageCreateInfo
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    data.pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    // Cleanup

    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

pub unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    // Attachments

    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        // Could be values
        // LOAD - Prserve the existing contents of the attachment
        // CLEAR - Clear the values to a constant at the start
        // DONT_CARE - Existing contents are undefined
        .load_op(vk::AttachmentLoadOp::CLEAR)
        // Could be values
        // STORE - Rendering contents will be stored in memory and can be read later
        // DONT_CARE - Contents of the framebuffer will be undefined after the rendering operation
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        // Could be values
        // COLOR_ATTACHMENT_OPTIMAL - Images used as color attachment
        // PRESENT_SRC_KHR - Images to be presented in the swapchain
        // TRANSFER_DST_OPTIMAL - Images to be used as destination for a memory copy operation
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // Subpasses

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments);

    // Dependencies

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    //Create

    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.render_pass = device.create_render_pass(&info, None)?;

    Ok(())
}

pub unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();
    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}
