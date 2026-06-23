use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

pub struct SamplerDesc {
    pub filter: vk::Filter,
    pub address_mode: vk::SamplerAddressMode,
    pub max_anisotropy: Option<f32>,
}

pub unsafe fn create_sampler(device: &Device, desc: &SamplerDesc) -> Result<vk::Sampler> {
    let mipmap_mode = match desc.filter {
        vk::Filter::NEAREST => vk::SamplerMipmapMode::NEAREST,
        vk::Filter::LINEAR => vk::SamplerMipmapMode::LINEAR,
        _ => vk::SamplerMipmapMode::NEAREST,
    };

    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(desc.filter)
        .min_filter(desc.filter)
        .mipmap_mode(mipmap_mode)
        .address_mode_u(desc.address_mode)
        .address_mode_v(desc.address_mode)
        .address_mode_w(desc.address_mode)
        .anisotropy_enable(desc.max_anisotropy.is_some())
        .max_anisotropy(desc.max_anisotropy.unwrap_or(1.0))
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);

    Ok(device.create_sampler(&info, None)?)
}
