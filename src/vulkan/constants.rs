use vulkanalia::{Version, vk};

pub const VALIDATION_ENABLED: bool = true;
pub const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub const VALIDATION_SYNC: bool = true;
pub const VALIDATION_BEST_PRACTICES: bool = true;

// mutually exclusive to printf
// has gpu overhead, off for now
pub const VALIDATION_GPU_ASSISTED: bool = false;

pub const VALIDATION_DEBUG_PRINTF: bool = false;

pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION.name,
];

pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
pub const TILE_SIZE: u32 = 8;
pub const OFFSCREEN_FRAME_COUNT: usize = 3;
