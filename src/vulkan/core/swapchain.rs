use log::info;
use vulkanalia::{prelude::v1_0::*, vk::KhrSwapchainExtensionDeviceCommands};
use winit::window::Window;

use anyhow::{Result, anyhow};

use super::context::{SuitabilityError, SwapchainSupport, VulkanContext};

#[derive(Clone, Debug)]
pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_layouts: Vec<vk::ImageLayout>,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub render_pass: vk::RenderPass,
}

impl Swapchain {
    pub unsafe fn new(ctx: &VulkanContext, window: &Window) -> Result<Self> {
        let device = &ctx.device;
        let support = SwapchainSupport::get(&ctx.instance, ctx.surface, ctx.physical_device)?;

        let required_usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if !support
            .capabilities
            .supported_usage_flags
            .contains(required_usage)
        {
            return Err(anyhow!(SuitabilityError(
                "Current GPU's swapchain images do not support required storage/transfer usage",
            )));
        }

        let surface_format = get_surface_format(&support.formats);
        let present_mode = get_present_mode(&support.present_modes);
        let extent = get_extent(window, support.capabilities);
        let format = surface_format.format;

        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count != 0
            && image_count > support.capabilities.max_image_count
        {
            image_count = support.capabilities.max_image_count;
        }

        let indices = ctx.queue_indices;
        let mut queue_family_indices = vec![];
        let image_sharing_mode = if indices.graphics != indices.present
            && indices.graphics != indices.compute
            && indices.present != indices.compute
        {
            queue_family_indices.push(indices.graphics);
            queue_family_indices.push(indices.present);
            queue_family_indices.push(indices.compute);
            info!("Queues are split");
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(ctx.surface)
            .min_image_count(image_count)
            .image_format(format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(required_usage)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain = device.create_swapchain_khr(&swapchain_info, None)?;
        info!("Created Swapchain: {:?}", swapchain);

        let images = device.get_swapchain_images_khr(swapchain)?;
        info!("Created {} Swapchain Images: {:?}", images.len(), images);
        let image_layouts = vec![vk::ImageLayout::UNDEFINED; images.len()];

        let image_views = images
            .iter()
            .map(|i| {
                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let info = vk::ImageViewCreateInfo::builder()
                    .image(*i)
                    .view_type(vk::ImageViewType::_2D)
                    .format(format)
                    .subresource_range(subresource_range);

                device.create_image_view(&info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let render_pass = create_render_pass(device, format)?;

        let mut framebuffers = Vec::with_capacity(image_views.len());
        for &view in &image_views {
            let attachments = [view];
            let info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);

            framebuffers.push(device.create_framebuffer(&info, None)?);
        }

        Ok(Self {
            swapchain,
            format,
            extent,
            images,
            image_layouts,
            image_views,
            framebuffers,
            render_pass,
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        for &framebuffer in &self.framebuffers {
            device.destroy_framebuffer(framebuffer, None);
        }
        self.framebuffers.clear();
        device.destroy_render_pass(self.render_pass, None);
        for &view in &self.image_views {
            device.destroy_image_view(view, None);
        }
        self.image_views.clear();
        device.destroy_swapchain_khr(self.swapchain, None);
        self.image_layouts.clear();
    }
}

fn get_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_UNORM
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

#[rustfmt::skip]
fn get_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

unsafe fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass> {
    // attachments

    let color_attachment = vk::AttachmentDescription::builder()
        .format(format)
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
        // load reads and writes
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ,
        );

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
