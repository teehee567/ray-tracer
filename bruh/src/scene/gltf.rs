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

use super::Scene;


impl Scene {
    pub fn from_gltf(path: &str) -> Result<Self> {
        let (gltf, buffers, images) = gltf::import(path)?;
        
        let mut scene = Scene {
            components: SceneComponents::default(),
            root: Value::default(),
        };

        // Set default camera if none provided
        let mut ubo = CameraBufferObject {
            focal_length: Alignedf32(1.),
            focus_distance: Alignedf32(55.),
            aperture_radius: Alignedf32(0.),
            location: AlignedVec3::new(35.0, 12.0, 35.0),
            ..Default::default()
        };
        let resolution = UVec2::new(1920, 1080);
        let rotation = Vec3::new(-10., -135., -7.);

        ubo.rotation = AlignedMat4(Mat4::from_rotation_x(rotation[0].to_radians())
            * Mat4::from_rotation_y(rotation[1].to_radians())
            * Mat4::from_rotation_z(rotation[2].to_radians()));


        let ratio = resolution[0] as f32 / resolution[1] as f32;
        let (u, v) = if ratio > 1.0 {
            (ratio, 1.0)
        } else {
            (1.0, 1.0 / ratio)
        };
        ubo.view_port_uv = AlignedVec2(Vec2::new(u, v));
        ubo.resolution = AlignedUVec2(resolution);
        scene.components.camera = ubo;

        // Process the default scene or first available scene
        let gltf_scene = gltf.default_scene()
            .or_else(|| gltf.scenes().next())
            .ok_or_else(|| anyhow!("No scenes found in GLTF file"))?;

        // Process all scenes
        scene.process_gltf_scene(&gltf_scene, &buffers, &images)?;
        println!("Triangles: {}", scene.components.triangles.len());

        scene.build_bvh();

        Ok(scene)
    }
    fn process_gltf_scene(
        &mut self,
        scene: &gltf::Scene,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
    ) -> Result<()> {
        // Process cameras first if any
        for node in scene.nodes() {
            if let Some(camera) = node.camera() {
                self.process_camera(&camera, &node)?;
            }
        }

        // Process meshes and materials
        for node in scene.nodes() {
            self.process_node(&node, &Mat4::IDENTITY, buffers, images)?;
        }

        Ok(())
    }

    fn process_node(
        &mut self,
        node: &gltf::Node,
        parent_transform: &Mat4,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
    ) -> Result<()> {
        let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        let transform = *parent_transform * local_transform;

        if let Some(mesh) = node.mesh() {
            self.process_mesh(&mesh, &transform, buffers, images)?;
        }

        for child in node.children() {
            self.process_node(&child, &transform, buffers, images)?;
        }

        Ok(())
    }

    fn process_mesh(
        &mut self,
        mesh: &gltf::Mesh,
        transform: &Mat4,
        buffers: &[gltf::buffer::Data],
        images: &[gltf::image::Data],
    ) -> Result<()> {
        for primitive in mesh.primitives() {
            let material = self.process_material(&primitive.material(), images)?;
            let material_index = self.components.materials.len() as u32;
            self.components.materials.push(material);

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            
            if let Some(positions) = reader.read_positions() {
                let positions: Vec<[f32; 3]> = positions.collect();
                let normals: Vec<[f32; 3]> = reader.read_normals()
                    .map(|n| n.collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);

                if let Some(indices) = reader.read_indices() {
                    let indices: Vec<u32> = indices.into_u32().collect();
                    
                    for chunk in indices.chunks(3) {
                        if chunk.len() == 3 {
                            let mut triangle = Triangle {
                                material_index: Alignedu32(material_index),
                                is_sphere: Alignedu32(0),
                                vertices: [AlignedVec3(Vec3::ZERO); 3],
                                normals: [AlignedVec3(Vec3::ZERO); 3],
                            };

                            for (i, &index) in chunk.iter().enumerate() {
                                let pos = positions[index as usize];
                                let normal = normals[index as usize];
                                
                                let transformed_pos = transform.transform_point3(Vec3::from(pos));
                                let transformed_normal = transform.transform_vector3(Vec3::from(normal)).normalize();

                                triangle.vertices[i] = AlignedVec3(transformed_pos);
                                triangle.normals[i] = AlignedVec3(transformed_normal);
                            }

                            self.components.triangles.push(triangle);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn process_material(
        &self,
        material: &gltf::Material,
        images: &[gltf::image::Data],
    ) -> Result<Material> {
        let pbr = material.pbr_metallic_roughness();
        
        let base_color = pbr.base_color_factor();
        let metallic = Alignedf32(pbr.metallic_factor());
        let roughness = Alignedf32(pbr.roughness_factor());
        let emissive = material.emissive_factor();
        let ior = Alignedf32(material.ior()
            .unwrap_or(1.5));
        let transmission = Alignedf32(material.transmission()
            .and_then(|ext| Some(ext.transmission_factor() * 0.1))
            .unwrap_or(1.0));
        
        // Get emissive strength from extension
        let emissive_strength = material
            .emissive_strength()
            .unwrap_or(1.0);

        // Combine emissive color with strength
        let emission = AlignedVec3(Vec3::new(
            emissive[0] * emissive_strength,
            emissive[1] * emissive_strength,
            emissive[2] * emissive_strength,
        ));

        let materiala = Material {
            base_colour: AlignedVec3::new(base_color[0], base_color[1], base_color[2]),
            emission,
            metallic,
            roughness,
            ior,
            transmission,
            motion_blur: AlignedVec3(Vec3::ZERO),
            shade_smooth: Alignedu32(0),
        };
        if material.name().unwrap() == "Dock" {
            dbg!(&material);
            dbg!(&materiala);
            panic!();
        }

        Ok(materiala)
    }

    fn process_camera(&mut self, camera: &gltf::Camera, node: &gltf::Node) -> Result<()> {
        match camera.projection() {
            gltf::camera::Projection::Perspective(persp) => {
                let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
                let position = transform.col(3).truncate();
                
                self.components.camera.focal_length = Alignedf32(persp.yfov().to_degrees());
                self.components.camera.focus_distance = Alignedf32(persp.znear());
                self.components.camera.location = AlignedVec3::new(position.x, position.y, position.z);
                
                // Extract rotation from transform
                self.components.camera.rotation = AlignedMat4(Mat4::from_cols_array_2d(&node.transform().matrix()));
            }
            gltf::camera::Projection::Orthographic(_) => {
                unimplemented!("ORTHOGAPHIC CAMERA NOT IMPLEMENTED")
            }
        }
        Ok(())
    }


}


