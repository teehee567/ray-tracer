
use log::info;
use vulkanalia::prelude::v1_0::*;

use anyhow::Result;

use crate::AppData;


pub unsafe fn create_sampler(device: &Device, data: &mut AppData) -> Result<()> {
    info!("Creating sampler");
    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .mip_lod_bias(0.)
        .min_lod(0.)
        .max_lod(0.)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
        .anisotropy_enable(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .unnormalized_coordinates(true)
        .build();

    let sampler = device.create_sampler(&sampler_info, None)?;

    data.sampler = sampler;
    Ok(())
}
