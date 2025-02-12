use std::ptr::null_mut;
use std::ffi::c_void;

use ash::vk;

use crate::vulkan::api::{self, Sampler, VulkanApi};

/// A Rust version of your Texture class.
#[derive(Debug)]
pub struct Texture {
    pub sampler: Option<*mut Sampler>,
    pub data: *mut c_void,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub device_image: usize,
}

impl Texture {
    pub fn from_image_handle(api: &VulkanApi, image_handle: usize) -> Self {
        let image = api.get_image(image_handle);
        Self {
            width: image.extent.width as usize,
            height: image.extent.height as usize,
            depth: image.extent.depth as usize,
            sampler: None,
            data: null_mut(),
            device_image: image_handle,
        }
    }

    /// Constructs a new Texture, creating an image with the given dimensions and format.
    pub fn new(
        api: &mut VulkanApi,
        width: usize,
        height: usize,
        depth: usize,
        format: vk::Format,
        texture_sampler: Option<*mut Sampler>,
    ) -> Self {
        let device_image = api.create_image(
            vk::Extent3D {
                width: width as u32,
                height: height as u32,
                depth: 1,
            },
            format,
                vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED
        );
        Self {
            width,
            height,
            depth,
            sampler: texture_sampler,
            data: null_mut(),
            device_image,
        }
    }

    /// Returns the size (in bytes) of the texture.
    pub fn size(&self, api: &VulkanApi) -> usize {
        let image = api.get_image(self.device_image);
        self.width * self.height * self.depth * Texture::pixel_size(image.format)
    }

    /// Returns the size (in bytes) of a single pixel for the given format.
    pub fn pixel_size(format: vk::Format) -> usize {
        match format {
            vk::Format::R8G8B8A8_UNORM => 4,
            _ => 0,
        }
    }
}
