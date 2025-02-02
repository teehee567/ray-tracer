use std::fs::File;
use std::io::Read;
use serde_yaml::Value;
use anyhow::{anyhow, Result};

use crate::{AlignedVec3, AlignedVec4, Alignedf32, Alignedu32, Material, Mesh, Sphere, Triangle};

use super::bufferbuilder::BufferBuilder;

const CONFIG_VERSION: &'static str = "0.1";

#[derive(Clone, Debug, Default)]
pub struct Scene {
    scene_file: Value,
}

impl Scene {
    pub fn new(path: &'static str) -> Result<Self> {
        let mut file= File::open(path)?;
        let mut file_content = String::new();
        file.read_to_string(&mut file_content)?;
        let scene_file: Value = serde_yaml::from_str(&file_content)?;
        let scene = Scene { scene_file };
        scene.validate_file()?;
        Ok(scene)
    }

    pub fn get_buffer_sizes(&self) -> Result<(usize, usize)> {
        let mut meshes = BufferBuilder::new();
        let mut triangles = BufferBuilder::new();
        self.populate_buffers(&mut meshes, &mut triangles)?;
        Ok((meshes.get_offset(), triangles.get_offset()))
    }

    /// Reads the scene “meshes” and appends data to the given buffers.
    pub fn populate_buffers(
        &self,
        meshes: &mut BufferBuilder,
        triangles: &mut BufferBuilder,
    ) -> Result<()> {
        // dbg!(&self.scene_file);
        // The YAML file is expected to have a "scene" key with a sequence of mesh definitions.
        let scene_array = self
            .scene_file
            .get("scene")
            .and_then(Value::as_sequence)
            .ok_or(anyhow!("Missing or invalid 'scene' sequence"))?;
        let mesh_count = scene_array.len() as u32;
        meshes.append(mesh_count);
        meshes.pad(12); // pad according to alignment rules

        for mesh in scene_array {
            let mesh_type = mesh
                .get("type")
                .and_then(Value::as_str)
                .ok_or(anyhow!("Missing 'type' field in mesh"))?;
            match mesh_type {
                "TriMesh" => self.populate_trimesh(mesh, meshes, triangles)?,
                "Sphere" => self.populate_sphere(mesh, meshes, triangles)?,
                other => return Err(anyhow!("Unknown mesh type: {}", other)),
            }
        }
        Ok(())
    }

    /// Validates the YAML file:
    ///
    /// - The `"version"` key must match `CONFIG_VERSION`.
    /// - Each mesh must have the expected data (for example, a TriMesh must have matching
    ///   `vertices` and `normals` sequences, and the number of vertex values must be a multiple of 9).
    pub fn validate_file(&self) -> Result<()> {
        let version = self
            .scene_file
            .get("version")
            .and_then(Value::as_str)
            .ok_or(anyhow!("Missing 'version' field"))?;
        if version != CONFIG_VERSION {
            return Err(anyhow!("Scene file is of incompatible version!"));
        }

        let scene_array = self
            .scene_file
            .get("scene")
            .and_then(Value::as_sequence)
            .ok_or(anyhow!("Missing or invalid 'scene' sequence"))?;

        for mesh in scene_array {
            let mesh_type = mesh
                .get("type")
                .and_then(Value::as_str)
                .ok_or(anyhow!("Missing 'type' field in mesh"))?;
            if mesh_type == "TriMesh" {
                let data = mesh.get("data").ok_or(anyhow!("Missing 'data' in TriMesh"))?;
                let vertices = data.get("vertices")
                    .and_then(Value::as_sequence)
                    .ok_or(anyhow!("vertices is not a sequence"))?;
                let normals = data.get("normals")
                    .and_then(Value::as_sequence)
                    .ok_or(anyhow!("normals is not a sequence"))?;
                if vertices.len() != normals.len() {
                    return Err(anyhow!("vertices and normals count mismatch"));
                }
                if vertices.len() % 9 != 0 {
                    return Err(anyhow!("vertices size is not a multiple of 9"));
                }
            } else if mesh_type == "Sphere" {
                let data = mesh.get("data").ok_or(anyhow!("Missing 'data' in Sphere"))?;
                if !data.get("center").map_or(false, |v| v.is_sequence()) {
                    return Err(anyhow!("center is not a sequence"));
                }
                if !data.get("radius").map_or(false, |v| v.is_number()) {
                    return Err(anyhow!("radius is not a scalar"));
                }
            }
        }
        Ok(())
    }

    /// Populates the buffers with triangle data for a TriMesh.
    fn populate_trimesh(
        &self,
        mesh: &Value,
        meshes: &mut BufferBuilder,
        triangles: &mut BufferBuilder,
    ) -> Result<()> {
        // Read the material information.
        let material = get_material(
            mesh.get("material")
                .ok_or(anyhow!("Missing 'material' in TriMesh"))?,
        )?;
        // Get the vertex data.
        let data = mesh.get("data").ok_or(anyhow!("Missing 'data' in TriMesh"))?;
        let verts_seq = data
            .get("vertices")
            .and_then(Value::as_sequence)
            .ok_or(anyhow!("Missing vertices sequence"))?;
        let norms_seq = data
            .get("normals")
            .and_then(Value::as_sequence)
            .ok_or(anyhow!("Missing normals sequence"))?;

        // Convert YAML sequences into Vec<f32>.
        let verts: Vec<f32> = verts_seq
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let norms: Vec<f32> = norms_seq
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();

        let triangle_amt = (verts.len() / 9) as u32;

        // Create and append the mesh info.
        let mesh_info = Mesh {
            triangle_count: triangle_amt,
            offset: triangles.get_relative_offset::<Triangle>()? as u32,
            material,
        };
        meshes.append(mesh_info);

        // Create each triangle from three vertices and normals.
        for i in 0..(triangle_amt as usize) {
            let mut tri = Triangle {
                vertices: [AlignedVec4::default(); 3],
                normals: [AlignedVec4::default(); 3],
            };
            for j in 0..3 {
                let off = i * 9 + j * 3;
                tri.vertices[j] = AlignedVec4::new(verts[off], verts[off + 1], verts[off + 2], 0.0);
                tri.normals[j] = AlignedVec4::new(norms[off], norms[off + 1], norms[off + 2], 0.0);
            }
            triangles.append(tri);
        }
        Ok(())
    }

    /// Populates the buffers with a Sphere.
    fn populate_sphere(
        &self,
        mesh: &Value,
        meshes: &mut BufferBuilder,
        triangles: &mut BufferBuilder,
    ) -> Result<()> {
        let material = get_material(
            mesh.get("material")
                .ok_or(anyhow!("Missing 'material' in Sphere"))?,
        )?;
        let mesh_info = Mesh {
            triangle_count: 0,
            offset: triangles.get_relative_offset::<Triangle>()? as u32,
            material,
        };
        meshes.append(mesh_info);

        let data = mesh.get("data").ok_or(anyhow!("Missing 'data' in Sphere"))?;
        let center_seq = data
            .get("center")
            .and_then(Value::as_sequence)
            .ok_or(anyhow!("Missing center sequence"))?;
        let center_vals: Vec<f32> = center_seq
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let center = AlignedVec3::new(
            *center_vals.get(0).unwrap_or(&0.0),
            *center_vals.get(1).unwrap_or(&0.0),
            *center_vals.get(2).unwrap_or(&0.0),
        );
        let radius = data
            .get("radius")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32;
        let sphere = Sphere {
            center,
            radius: Alignedf32(radius),
        };
        // Append the sphere into the triangle buffer. (We pass the alignment equal to the size
        // of a Triangle so that subsequent data remains aligned as in the original code.)
        triangles.append_with_size(sphere, std::mem::size_of::<Triangle>());
        Ok(())
    }
}

/// Constructs a Material from a YAML node containing material parameters.
pub fn get_material(node: &Value) -> Result<Material> {
    // A helper to convert a YAML sequence (of 3 numbers) into an AlignedVec4.
    fn seq_to_aligned_vec4(seq: Option<&Value>) -> AlignedVec4 {
        if let Some(val) = seq.and_then(|v| v.as_sequence()) {
            let mut nums: Vec<f32> =
                val.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            while nums.len() < 3 {
                nums.push(0.0);
            }
            AlignedVec4::new(nums[0], nums[1], nums[2], 0.0)
        } else {
            AlignedVec4::default()
        }
    }

    let base_colour = seq_to_aligned_vec4(node.get("base_color"));
    let emission = seq_to_aligned_vec4(node.get("emission"));
    let reflectivity =
        Alignedf32(node.get("reflectiveness").and_then(Value::as_f64).unwrap_or(0.0) as f32);
    let roughness =
        Alignedf32(node.get("roughness").and_then(Value::as_f64).unwrap_or(0.0) as f32);
    let ior = Alignedf32(node.get("ior").and_then(Value::as_f64).unwrap_or(0.0) as f32);
    let is_glass = if node.get("is_glass").and_then(Value::as_bool).unwrap_or(false) {
        1
    } else {
        0
    };
    let shade_smooth = if node
        .get("smooth_shading")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        1
    } else {
        0
    };

    Ok(Material {
        base_colour,
        emission,
        reflectivity,
        roughness,
        is_glass: Alignedu32(is_glass),
        ior,
        shade_smooth: Alignedu32(shade_smooth),
    })
}
