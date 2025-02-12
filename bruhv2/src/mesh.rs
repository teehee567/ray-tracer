use std::rc::Rc;
use std::sync::{Arc, LazyLock, Mutex};
use bytemuck;

use crate::materials::texture::Texture;

pub type Color = [f32; 3];

#[derive(Debug, Clone)]
pub struct Material {
    pub base_color: Color,
    pub base_color_texture: Option<Arc<Texture>>,
    pub metallic_roughness_texture: Option<Arc<Texture>>,
    pub metalness: f32,
    pub roughness: f32,
}

#[derive(Debug, Clone)]
pub struct GpuMaterial {
    pub base_color: Color,
    pub albedo_texture_id: u32,
    pub albedo_texture_sampler_id: u32,
    pub metallic_roughness_texture_id: u32,
    pub metallic_roughness_texture_sampler_id: u32,
    pub metalness: f32,
    pub roughness: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeType {
    Position,
    Normal,
    Uv0,
    Max,
}

#[derive(Debug)]
pub struct SubmeshInfo {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub element_count: usize,
    pub material: Arc<Material>,
}

#[derive(Debug)]
pub struct Mesh {
    vertex_count: u32,
    index_count: u32,
    attributes: [Vec<f32>; AttributeType::Max as usize],
    indices: Vec<u16>,
    submeshes: Vec<SubmeshInfo>,
}

#[derive(Debug)]
pub struct Submesh {
    pub indices: Vec<u8>,
    pub attributes: [Vec<u8>; AttributeType::Max as usize],
    pub vertex_count: u32,
    pub index_count: u32,
}

impl Mesh {
    pub fn new() -> Self {
        Mesh {
            vertex_count: 0,
            index_count: 0,
            attributes: [
                Vec::new(), // Position
                Vec::new(), // Normal
                Vec::new(), // Uv0
            ],
            indices: Vec::new(),
            submeshes: Vec::new(),
        }
    }

    pub fn add_submesh(&mut self, submesh: Submesh, material: Arc<Material>) {
        let element_count = (submesh.index_count / 3) as usize;
        self.submeshes.push(SubmeshInfo {
            vertex_offset: self.vertex_count,
            index_offset: self.index_count,
            element_count,
            material,
        });

        // Process indices
        let indices_u16 = bytemuck::cast_slice(&submesh.indices);
        self.indices.extend_from_slice(indices_u16);

        // Process attributes
        for attr_type in 0..AttributeType::Max as usize {
            let submesh_attr = &submesh.attributes[attr_type];
            if submesh_attr.is_empty() {
                continue;
            }

            let components = Self::attribute_components(AttributeType::from_usize(attr_type));
            let data_f32 = bytemuck::cast_slice(submesh_attr);
            
            assert_eq!(
                data_f32.len(),
                submesh.vertex_count as usize * components as usize,
                "Attribute data length mismatch"
            );

            self.attributes[attr_type].extend_from_slice(data_f32);
        }

        self.vertex_count += submesh.vertex_count;
        self.index_count += submesh.index_count;
    }

    pub fn attribute_components(attribute_type: AttributeType) -> u32 {
        match attribute_type {
            AttributeType::Position | AttributeType::Normal => 3,
            AttributeType::Uv0 => 2,
            _ => 0,
        }
    }

    pub fn attribute_size(attribute_type: AttributeType) -> usize {
        (std::mem::size_of::<f32>() * Self::attribute_components(attribute_type) as usize)
    }

    pub fn triangle_count(&self) -> u32 {
        self.index_count / 3
    }

    pub fn vertices_count(&self) -> u32 {
        self.vertex_count
    }

    pub fn get_submeshes(&self) -> &[SubmeshInfo] {
        &self.submeshes
    }

    pub fn get_attribute(&self, attribute_type: AttributeType) -> &[f32] {
        &self.attributes[attribute_type as usize]
    }

    pub fn get_indices(&self) -> &[u16] {
        &self.indices
    }

    pub fn instantiate(&self) -> MeshInstance {
        MeshInstance { mesh: self }
    }

    pub fn register_mesh(mesh: &Mesh) {
        MESHES.lock().unwrap().push(mesh.clone());
    }
}

pub struct MeshInstance<'a> {
    pub mesh: &'a Mesh,
}

trait FromUsize {
    fn from_usize(value: usize) -> Self;
}

impl FromUsize for AttributeType {
    fn from_usize(value: usize) -> Self {
        match value {
            0 => AttributeType::Position,
            1 => AttributeType::Normal,
            2 => AttributeType::Uv0,
            _ => panic!("Invalid attribute type"),
        }
    }
}
