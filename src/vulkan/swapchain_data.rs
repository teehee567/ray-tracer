use vulkanalia::prelude::v1_0::*;

#[derive(Clone, Debug)]
pub struct SwapchainData {
    pub swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_layouts: Vec<vk::ImageLayout>,
    pub image_views: Vec<vk::ImageView>,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub render_pass: vk::RenderPass,
}
