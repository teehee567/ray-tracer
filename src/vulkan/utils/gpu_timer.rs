use vulkanalia::prelude::v1_0::*;

use crate::fps_counter::FPSCounter;
use crate::vulkan::core::commands::with_single_time;
use crate::vulkan::core::context::VulkanContext;

use anyhow::Result;

pub struct GpuTimer {
    // how long for timestamp tick, this comes from vulkan
    period: f32,

    query_pool: vk::QueryPool,
    // most recent time
    fps_counter: FPSCounter,
}

impl GpuTimer {
    pub unsafe fn new(ctx: &VulkanContext, frame_count: usize) -> Result<Self> {
        let period = ctx
            .instance
            .get_physical_device_properties(ctx.physical_device)
            .limits
            .timestamp_period;
        let info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            // neets * 2 because frame_count is amount of frames and 1 point for start and another for back
            .query_count(frame_count as u32 * 2);
        let query_pool = ctx.device.create_query_pool(&info, None)?;

        // gets reset error even when i just make it???
        with_single_time(&ctx.device, ctx.command_pool, ctx.compute_queue, |cb| {
            ctx.device
                .cmd_reset_query_pool(cb, query_pool, 0, frame_count as u32 * 2);
            Ok(())
        })?;

        Ok(Self {
            query_pool,
            period,
            fps_counter: FPSCounter::new(frame_count.max(1) * 2),
        })
    }

    pub fn fps(&self) -> f64 {
        self.fps_counter.get_fps()
    }

    pub fn last_ms(&self) -> f64 {
        self.fps_counter.last_frame_ms()
    }

    pub fn query_pool(&self) -> vk::QueryPool {
        self.query_pool
    }

    // reads a frame time
    pub unsafe fn read_slot(&mut self, device: &Device, frame_index: usize) -> Result<()> {
        let first_query = frame_index as u32 * 2;
        let mut data = [0u8; 16];
        let status = device.get_query_pool_results(
            self.query_pool,
            first_query,
            2,
            &mut data,
            8,
            vk::QueryResultFlags::_64,
        )?;
        // handle slot hasnt been written to yet
        if status == vk::SuccessCode::NOT_READY {
            return Ok(());
        }
        let start = u64::from_ne_bytes(data[0..8].try_into().unwrap());
        let end = u64::from_ne_bytes(data[8..16].try_into().unwrap());
        let ms = end.wrapping_sub(start) as f64 * self.period as f64 / 1_000_000.0;
        self.fps_counter.push_ms(ms);
        Ok(())
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_query_pool(self.query_pool, None);
    }
}