use serde_yaml::Value;
use std::os::raw::c_void;
use std::path::Path;
use std::time::Instant;

use crate::accelerators::bvh::BvhBuilder;
use crate::{CameraBufferObject, SceneComponents, EmissiveTri, MeshLightSampler, Light, Material, Triangle, Au32, Af32, AVec3};

use anyhow::Result;
use glam::Vec3;

const CONFIG_VERSION: &str = "0.2";

use crate::vulkan::bufferbuilder::BufferBuilder;
pub mod gltf;
pub mod loader;
pub mod weird;
pub mod yaml;

#[derive(Clone, Copy, Debug)]
pub enum TextureFormat {
    R8,
    R8G8,
    R8G8B8,
    R8G8B8A8,
    B8G8R8,
    B8G8R8A8,
}

#[derive(Clone, Debug)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub pixels: Vec<u8>,
}

impl Default for TextureData {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            format: TextureFormat::R8G8B8A8,
            pixels: vec![
                255, 255, 255, 255, // Right face (+X)
                255, 255, 255, 255, // Left face (-X)
                255, 255, 255, 255, // Top face (+Y)
                255, 255, 255, 255, // Bottom face (-Y)
                255, 255, 255, 255, // Front face (+Z)
                255, 255, 255, 255, // Back face (-Z)
            ],
        }
    }
}

/// A scene loaded from a YAML file.
#[derive(Default, Clone)]
pub struct Scene {
    root: Value,
    pub components: SceneComponents,
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

        let builder = BvhBuilder::new(
            &mut self.components.triangles,
            &mut self.components.materials,
        );
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

    pub fn get_buffer_sizes(&self) -> (usize, usize, usize, usize, usize, usize) {
        let mut bvh_buf = BufferBuilder::new();
        let mut mat_buf = BufferBuilder::new();
        let mut tri_buf = BufferBuilder::new();
        let mut light_buf = BufferBuilder::new();
        let mut emissive_tri_buf = BufferBuilder::new();
        let mut cdf_buf = BufferBuilder::new();

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
        for light in &self.components.lights {
            light_buf.append(*light);
        }
        light_buf.align_to_16();
        for emissive_tri in &self.components.mesh_light_sampler.tris {
            emissive_tri_buf.append(*emissive_tri);
        }
        emissive_tri_buf.align_to_16();
        for cdf_val in &self.components.mesh_light_sampler.cdf {
            cdf_buf.append(*cdf_val);
        }
        cdf_buf.align_to_16();
        (
            bvh_buf.get_offset(),
            mat_buf.get_offset(),
            tri_buf.get_offset(),
            light_buf.get_offset(),
            emissive_tri_buf.get_offset(),
            cdf_buf.get_offset(),
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
        for light in &self.components.lights {
            memory_buf.append(*light);
        }
        memory_buf.align_to_16();
        for emissive_tri in &self.components.mesh_light_sampler.tris {
            memory_buf.append(*emissive_tri);
        }
        memory_buf.align_to_16();
        for cdf_val in &self.components.mesh_light_sampler.cdf {
            memory_buf.append(*cdf_val);
        }
        memory_buf.align_to_16();
        unsafe {
            memory_buf.write(memory);
        }
    }

    pub fn get_camera_controls(&self) -> CameraBufferObject {
        self.components.camera
    }

    pub fn build_lights(&mut self) {
        println!("building lights...");
        let start = Instant::now();

        let mut emissive_tris = Vec::new();
        let mut total_power = 0.0;

        // Find all triangles with emissive materials
        for (tri_idx, triangle) in self.components.triangles.iter().enumerate() {
            let mat_idx = triangle.material_index.0 as usize;
            if mat_idx < self.components.materials.len() {
                let material = &self.components.materials[mat_idx];
                let emission = material.emission.0;
                
                // Check if material is emissive (has non-zero emission)
                if emission.length_squared() > 0.0 {
                    // Calculate triangle area: |(v1-v0) x (v2-v0)| * 0.5
                    let v0 = triangle.vertices[0].0;
                    let v1 = triangle.vertices[1].0;
                    let v2 = triangle.vertices[2].0;
                    let edge1 = v1 - v0;
                    let edge2 = v2 - v0;
                    let area = edge1.cross(edge2).length() * 0.5;
                    
                    // Calculate power: luminance(emission) * area
                    // Using luminance = 0.2126*R + 0.7152*G + 0.0722*B
                    let luminance = emission.x * 0.2126 + emission.y * 0.7152 + emission.z * 0.0722;
                    let power = luminance * area;
                    
                    if power > 0.0 {
                        emissive_tris.push(EmissiveTri {
                            tri_index: Au32(tri_idx as u32),
                            mesh_index: triangle.material_index, // Using material index as mesh index
                            area: Af32(area),
                            power: Af32(power),
                        });
                        total_power += power;
                    }
                }
            }
        }

        // Build CDF (cumulative distribution function)
        let mut cdf = Vec::new();
        let mut cumulative = 0.0;
        for emissive_tri in &emissive_tris {
            cumulative += emissive_tri.power.0;
            cdf.push(cumulative / total_power);
        }

        self.components.mesh_light_sampler = MeshLightSampler {
            tris: emissive_tris,
            cdf,
            total_power,
        };

        // For now, we don't have explicit point lights in the scene format
        // This can be extended later if needed
        self.components.lights.clear();

        println!("Lights built in :{:.4}s", start.elapsed().as_secs_f32());
        println!("Found {} emissive triangles", self.components.mesh_light_sampler.tris.len());
        println!("Total power: {}", total_power);
    }

    fn load_cubemap_textures<P: AsRef<Path>>(paths: [P; 6]) -> Result<TextureData> {
        let mut combined_pixels = Vec::new();
        let mut width = 0;
        let mut height = 0;

        for path in paths.iter() {
            let img = image::open(path)?;
            let img_rgba = img.into_rgba8();

            if width == 0 {
                width = img_rgba.width();
                height = img_rgba.height();
            } else if width != img_rgba.width() || height != img_rgba.height() {
                return Err(anyhow::anyhow!(
                    "All cubemap faces must have the same dimensions"
                ));
            }

            combined_pixels.extend_from_slice(&img_rgba.into_raw());
        }

        Ok(TextureData {
            width,
            height,
            format: TextureFormat::R8G8B8A8,
            pixels: combined_pixels,
        })
    }
}
