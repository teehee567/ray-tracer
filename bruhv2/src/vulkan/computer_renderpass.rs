
use std::ffi::c_void;
use ash::vk::{self, };

use crate::{materials::texture::Texture};

use super::api::{Pipeline, VulkanApi};

pub struct RenderBuffer {
    device_buffer: usize,
    buffer_size: usize,
    usage: vk::BufferUsageFlags,
}

pub struct ComputeRenderpass<'a> {
    api: &'a mut VulkanApi,
    output_texture: Option<Texture>,
    input_textures: Vec<Texture>,
    constants: [u8; 64],
    group_count_x: usize,
    group_count_y: usize,
    group_count_z: usize,
    pipeline: Option<Pipeline>,
}

impl<'a> ComputeRenderpass<'a> {
    pub fn new(api: &'a mut VulkanApi) -> Self {
        Self {
            api,
            output_texture: None,
            input_textures: Vec::new(),
            constants: [0; 64],
            group_count_x: 0,
            group_count_y: 0,
            group_count_z: 0,
            pipeline: None,
        }
    }

    pub fn set_pipeline(&mut self, shader_name: &str) {
        if let Some(pipeline) = self.pipeline.take() {
            self.api.destroy_pipeline(&pipeline);
        }
        self.pipeline = Some(self.api.create_compute_pipeline(shader_name));
    }

    pub fn set_output_texture(&mut self, out_texture: Texture) {
        self.output_texture = Some(out_texture);
        let device_image = self.output_texture.as_ref().unwrap().device_image;
        let image_info = self.api.get_image(device_image);
        
        // Assuming bindless_storage_index is u32 (4 bytes)
        let offset = 56;
        self.constants[offset..offset + 4].copy_from_slice(&image_info.bindless_storage_index.unwrap().to_ne_bytes());
    }

    pub fn set_dispatch_size(&mut self, count_x: usize, count_y: usize, count_z: usize) {
        self.group_count_x = count_x;
        self.group_count_y = count_y;
        self.group_count_z = count_z;
    }

    pub fn set_constant_u64(&mut self, offset: usize, value: u64) {
        self.constants[offset..offset + 8].copy_from_slice(&value.to_ne_bytes());
    }

    pub fn set_constant_texture(&mut self, offset: usize, texture: Texture) {
        let image_info = self.api.get_image(texture.device_image);
        self.constants[offset..offset + 4].copy_from_slice(&image_info.bindless_storage_index.unwrap().to_ne_bytes());
        self.input_textures.push(texture);
    }

    pub fn set_constant_buffer(&mut self, offset: usize, buffer: &RenderBuffer) {
        let buffer_info = self.api.get_buffer(buffer.device_buffer);
        let address_bytes = buffer_info.device_address.to_ne_bytes();
        self.constants[offset..offset + 8].copy_from_slice(&address_bytes);
    }

    pub fn execute(&mut self, command_buffer: vk::CommandBuffer) {
        // Update push constants
        self.api.update_constants(
            command_buffer,
            vk::ShaderStageFlags::COMPUTE,
            0,
            &self.constants,
        );

        // Input texture barriers
        for image in &self.input_textures {
            self.api.image_barrier(
                command_buffer,
                image.device_image,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ,
            );
        }

        // Output texture barrier
        if let Some(texture) = &self.output_texture {
            self.api.image_barrier(
                command_buffer,
                texture.device_image,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            );
        }

        // Dispatch compute
        if let Some(pipeline) = &self.pipeline {
            self.api.run_compute_pipeline(
                command_buffer,
                pipeline,
                self.group_count_x,
                self.group_count_y,
                self.group_count_z,
            );
        }

        // Clear input textures for next frame
        self.input_textures.clear();
    }
}

impl<'a> Drop for ComputeRenderpass<'a> {
    fn drop(&mut self) {
        if let Some(pipeline) = &self.pipeline.take() {
            self.api.destroy_pipeline(pipeline);
        }
    }
}
