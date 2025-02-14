use bincode::{deserialize_from, serialize_into};
use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use serde_yaml::Value;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::os::raw::c_void;
use std::time::Instant;

use crate::accelerators::bvh::{BvhBuilder, BvhNode};
use crate::{
    AlignedMat4, AlignedUVec2, AlignedVec2, AlignedVec3, AlignedVec4, Alignedf32, Alignedu32, CameraBufferObject, Material, SceneComponents, Triangle
};

const CONFIG_VERSION: &str = "0.2";

use anyhow::{anyhow, bail, Result};

use crate::vulkan::bufferbuilder::BufferBuilder;
pub mod yaml;
pub mod gltf;
pub mod mitsuba;
pub mod tungsten;

/// A scene loaded from a YAML file.
#[derive(Default, Clone)]
pub struct Scene {
    root: Value,
    components: SceneComponents,
}

impl std::fmt::Debug for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scene")
            .field("root", &self.root)
            // Skip components since it does not implement Debug.
            .finish()
    }
}

impl Scene {
    pub fn build_bvh(&mut self) {
        println!("building BVH...");
        let start = Instant::now();


        // let file = File::create("bvh.bin").unwrap();
        // let writer = BufWriter::new(file);

        let mut builder = BvhBuilder::new(&mut self.components.triangles, &mut self.components.materials);
        self.components.bvh = builder.build_bvh();

        // serialize_into(writer, &self.components.bvh).unwrap();
        //
        // let file = File::open("bvh.bin").unwrap();
        // let reader = BufReader::new(file);
        // let decoded: Vec<BvhNode> = deserialize_from(reader).unwrap();

        // let json = serde_json::to_string_pretty(&self.components.bvh).unwrap();
        // let mut file = File::create("output.json").unwrap();
        // file.write_all(json.as_bytes()).unwrap();

        // let sah = SAH::new(2., 1.);
        // let max_leaf_size = 4;
        // let bvh = BVH::build(self.components.triangles.as_slice(), sah, max_leaf_size);
        // println!("{}", bvh.compute_debug_info());
        // self.components.bvh = bvh.nodes;

        println!("BVH built in :{:.4}s", start.elapsed().as_secs_f32());
        println!("done!");
    }

    pub fn get_buffer_sizes(&self) -> (usize, usize, usize) {
        let mut bvh_buf = BufferBuilder::new();
        let mut mat_buf = BufferBuilder::new();
        let mut tri_buf = BufferBuilder::new();

        for node in &self.components.bvh {
            bvh_buf.append(*node);
        }
        bvh_buf.align_to_16();
        for material in &self.components.materials {
            mat_buf.append(*material);
        }
        mat_buf.align_to_16();
        for triangle in &self.components.triangles {
            tri_buf.append(*triangle);
        }
        tri_buf.align_to_16();
        (
            bvh_buf.get_offset(),
            mat_buf.get_offset(),
            tri_buf.get_offset(),
        )
    }

    // pub fn get_total_size(&self) -> usize {
    //     let a = self.get_buffer_sizes();
    //     a.
    // }

    /// Writes all buffers into the given memory (dummy implementation).
    pub fn write_buffers(&self, memory: *mut c_void) {
        let mut memory_buf = BufferBuilder::new();
        for node in &self.components.bvh {
            memory_buf.append(*node);
        }
        memory_buf.align_to_16();
        for material in &self.components.materials {
            memory_buf.append(*material);
        }
        memory_buf.align_to_16();
        for triangle in &self.components.triangles {
            memory_buf.append(*triangle);
        }
        memory_buf.align_to_16();
        unsafe {
            memory_buf.write(memory);
        }
    }

    pub fn get_camera_controls(&self) -> CameraBufferObject {
        self.components.camera
    }


}

