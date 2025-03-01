
use log::info;
use vulkanalia::{bytecode::Bytecode, prelude::v1_0::*};

use crate::AppData;
use anyhow::Result;

pub unsafe fn create_compute_pipeline(device: &Device, data: &mut AppData) -> Result<()> {
    let binding = [data.descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&binding);
    data.compute_pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    let compute_shader_src = include_bytes!("../../src/shaders/main.comp.spv");
    let compute_shader = create_shader_module(device, compute_shader_src)?;

    let compute_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(compute_shader)
        .name(b"main\0");
    info!("Loaded Compute shader: {:?}", compute_shader);

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .layout(data.compute_pipeline_layout)
        .stage(compute_shader_stage_info);

    data.compute_pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?.0[0];
    info!("Created a Compute Pipeline: {:?}", data.compute_pipeline);

    device.destroy_shader_module(compute_shader, None);

    Ok(())
}

pub unsafe fn create_render_pass(instance: &Instance, device: &Device, data: &mut AppData) -> Result<()> {
    // Attachments

    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
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

    // Create

    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.render_pass = device.create_render_pass(&info, None)?;
    info!("Created a render_pass: {:?}", data.render_pass);

    Ok(())
}


pub unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}
