use gltf::{Gltf, Scene};
use std::path::Path;

#[derive(Default)]
pub struct Node {
    children: Vec<Node>,
    meshes: Vec<Mesh>, // Now a vector to hold multiple meshes (primitives)
}

pub struct GltfLoader {
    root_node: Node,
    meshes: Vec<Mesh>,
    materials: Vec<Material>,
    textures: Vec<Texture>,
    mesh_primitives: Vec<Vec<usize>>,
}

impl GltfLoader {
    pub fn new(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let (document, buffers, images) = load_gltf(path)?;
        
        let mut loader = Self {
            root_node: Node::default(),
            meshes: Vec::new(),
            materials: Vec::new(),
            textures: Vec::new(),
            mesh_primitives: Vec::new(),
        };

        // Load textures first since materials depend on them
        loader.load_textures(&document, &images)?;
        loader.load_materials(&document)?;
        loader.load_meshes(&document, &buffers)?;
        loader.load_scenes(&document)?;

        Ok(loader)
    }

    fn load_textures(
        &mut self, 
        document: &gltf::Document,
        images: &[gltf::image::Data],
    ) -> Result<(), Box<dyn std::error::Error>> {
        for texture in document.textures() {
            let source = texture.source();
            let image_data = &images[source.index()];
            let texture = Texture::new(
                image_data.width,
                image_data.height,
                &image_data.pixels,
                image_data.format,
            )?;
            self.textures.push(texture);
        }
        Ok(())
    }

    fn load_materials(
        &mut self,
        document: &gltf::Document,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for material in document.materials() {
            let pbr = material.pbr_metallic_roughness();
            
            let base_color_factor = pbr.base_color_factor();
            let metallic_factor = pbr.metallic_factor();
            let roughness_factor = pbr.roughness_factor();

            // Get texture indices if they exist
            let base_color_texture = pbr
                .base_color_texture()
                .map(|tex| self.textures[tex.texture().index()].clone());

            let metallic_roughness_texture = pbr
                .metallic_roughness_texture()
                .map(|tex| self.textures[tex.texture().index()].clone());

            let material = Material {
                base_color: base_color_factor.into(),
                base_color_texture,
                metalness: metallic_factor,
                roughness: roughness_factor,
                metallic_roughness_texture,
            };

            self.materials.push(material);
        }
        Ok(())
    }

    fn load_meshes(
        &mut self,
        document: &gltf::Document,
        buffers: &[gltf::buffer::Data],
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.mesh_primitives.clear();
        for gltf_mesh in document.meshes() {
            let mut primitives = Vec::new();
            for primitive in gltf_mesh.primitives() {
                let mut loaded_mesh = Mesh::new();

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                
                // Read indices
                if let Some(indices) = reader.read_indices() {
                    loaded_mesh.indices = indices.into_u32().collect();
                }

                // Read positions
                if let Some(positions) = reader.read_positions() {
                    loaded_mesh.positions = positions.collect();
                }

                // Read normals
                if let Some(normals) = reader.read_normals() {
                    loaded_mesh.normals = normals.collect();
                }

                // Read texture coordinates
                if let Some(tex_coords) = reader.read_tex_coords(0) {
                    loaded_mesh.uvs = tex_coords.into_f32().collect();
                }

                // Add material reference
                if let Some(material_index) = primitive.material().index() {
                    loaded_mesh.material = Some(self.materials[material_index].clone());
                }

                self.meshes.push(loaded_mesh);
                primitives.push(self.meshes.len() - 1);
            }
            self.mesh_primitives.push(primitives);
        }
        Ok(())
    }

    fn load_scenes(
        &mut self,
        document: &gltf::Document,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(scene) = document.default_scene() {
            self.root_node = self.load_node_hierarchy(&scene)?;
        }
        Ok(())
    }

    fn load_node_hierarchy(&self, scene: &Scene) -> Result<Node, Box<dyn std::error::Error>> {
        let mut root = Node::default();
        
        for scene_node in scene.nodes() {
            let node = self.process_node(&scene_node)?;
            root.children.push(node);
        }

        Ok(root)
    }

    fn process_node(&self, node: &gltf::Node) -> Result<Node, Box<dyn std::error::Error>> {
        let mut new_node = Node::default();

        // Load meshes for all primitives if present
        if let Some(gltf_mesh) = node.mesh() {
            let mesh_index = gltf_mesh.index();
            if let Some(primitives) = self.mesh_primitives.get(mesh_index) {
                for &primitive_index in primitives {
                    new_node.meshes.push(self.meshes[primitive_index].clone());
                }
            }
        }

        // Process children recursively
        for child in node.children() {
            new_node.children.push(self.process_node(&child)?);
        }

        Ok(new_node)
    }
}

fn load_gltf(
    path: &Path,
) -> Result<(gltf::Document, Vec<gltf::buffer::Data>, Vec<gltf::image::Data>), Box<dyn std::error::Error>> {
    let (document, buffers, images) = gltf::import(path)?;
    Ok((document, buffers, images))
}

// Define these types according to your renderer's needs
#[derive(Clone)]
struct Mesh {
    indices: Vec<u32>,
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    material: Option<Material>,
}

#[derive(Clone)]
struct Material {
    base_color: [f32; 4],
    base_color_texture: Option<Texture>,
    metalness: f32,
    roughness: f32,
    metallic_roughness_texture: Option<Texture>,
}

#[derive(Clone)]
struct Texture {
    width: u32,
    height: u32,
    data: Vec<u8>,
    format: gltf::image::Format,
}

impl Texture {
    fn new(
        width: u32,
        height: u32,
        data: &[u8],
        format: gltf::image::Format,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            width,
            height,
            data: data.to_vec(),
            format,
        })
    }
}

impl Mesh {
    fn new() -> Self {
        Self {
            indices: Vec::new(),
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            material: None,
        }
    }
}
