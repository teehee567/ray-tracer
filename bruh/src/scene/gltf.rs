use glam::{Mat3, Mat4, UVec2, Vec2, Vec3, Vec4};
use serde_yaml::Value;
use anyhow::{anyhow, Result};
use gltf;

use crate::{
    AlignedMat4, AlignedUVec2, AlignedVec2, AlignedVec3, Alignedf32, Alignedu32, CameraBufferObject, Material, Scene, SceneComponents, Triangle
};

impl Scene {
    pub fn from_gltf(path: &str) -> Result<Self> {
        let (gltf, buffers, _images) = gltf::import(path)?;
        
        let mut scene = Scene {
            components: SceneComponents::default(),
            root: Value::default(),
        };


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

        let gltf_scene = gltf.default_scene()
            .or_else(|| gltf.scenes().next())
            .ok_or_else(|| anyhow!("No scenes found"))?;

        scene.process_gltf_scene(&gltf_scene, &buffers)?;
        scene.build_bvh();

        Ok(scene)
    }

    fn process_gltf_scene(
        &mut self,
        scene: &gltf::Scene,
        buffers: &[gltf::buffer::Data],
    ) -> Result<()> {
        // Coordinate system conversion matrix (Y-up to Z-up)
        let y_up_to_z_up = Mat4::from_cols(
            Vec4::new(1.0, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(0.0, -1.0, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );

        for node in scene.nodes() {
            self.process_node(&node, &y_up_to_z_up, buffers)?;
        }

        Ok(())
    }

    fn process_node(
        &mut self,
        node: &gltf::Node,
        parent_transform: &Mat4,
        buffers: &[gltf::buffer::Data],
    ) -> Result<()> {
        let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
        let transform = *parent_transform * local_transform;

        // Process camera first
        if let Some(camera) = node.camera() {
            self.process_camera(&camera, transform)?;
        }

        // Process mesh
        if let Some(mesh) = node.mesh() {
            self.process_mesh(&mesh, &transform, buffers)?;
        }

        // Process children
        for child in node.children() {
            self.process_node(&child, &transform, buffers)?;
        }

        Ok(())
    }

    fn process_camera(&mut self, camera: &gltf::Camera, transform: Mat4) -> Result<()> {
        if let gltf::camera::Projection::Perspective(persp) = camera.projection() {
            let transform = Mat4::from_cols_array_2d(&transform.to_cols_array_2d());
            let position = transform.transform_point3(Vec3::ZERO);

            // Convert Y-up to Z-up by swapping coordinates
            let position = Vec3::new(position.x, position.z, position.y);

            self.components.camera = CameraBufferObject {
                focal_length: Alignedf32(35.0),
                focus_distance: Alignedf32(55.0),
                aperture_radius: Alignedf32(0.0),
                location: AlignedVec3(position),
                rotation: AlignedMat4(Mat4::from_rotation_x(-10.0_f32.to_radians())
                    * Mat4::from_rotation_y(-135.0_f32.to_radians())),
                resolution: AlignedUVec2(UVec2::new(1920, 1080)),
                view_port_uv: AlignedVec2(Vec2::ONE),
                ..Default::default()
            };

            let ratio = self.components.camera.resolution.0[0] as f32 / self.components.camera.resolution.0[1] as f32;
            let (u, v) = if ratio > 1.0 {
                (ratio, 1.0)
            } else {
                (1.0, 1.0 / ratio)
            };
            self.components.camera.view_port_uv = AlignedVec2(Vec2::new(u, v));
            }

        Ok(())
    }

    fn process_mesh(
        &mut self,
        mesh: &gltf::Mesh,
        transform: &Mat4,
        buffers: &[gltf::buffer::Data],
    ) -> Result<()> {
        let normal_matrix = Mat3::from_mat4(transform.inverse().transpose());

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let material = self.process_material(&primitive.material())?;
            let material_index = self.components.materials.len() as u32;
            self.components.materials.push(material);

            let positions = reader.read_positions()
                .ok_or_else(|| anyhow!("Missing positions"))?
                .collect::<Vec<[f32; 3]>>();
            
            // Read or compute normals
            let normals = match reader.read_normals() {
                Some(normal_iter) => normal_iter.collect::<Vec<[f32; 3]>>(),
                None => {
                    let mut computed_normals = vec![[0.0; 3]; positions.len()];
                    let indices = reader.read_indices()
                        .map(|i| i.into_u32().collect::<Vec<u32>>())
                        .unwrap_or_else(|| (0..positions.len() as u32).collect());

                    for chunk in indices.chunks(3) {
                        if chunk.len() != 3 { continue; }
                        let i0 = chunk[0] as usize;
                        let i1 = chunk[1] as usize;
                        let i2 = chunk[2] as usize;

                        let p0 = Vec3::from(positions[i0]);
                        let p1 = Vec3::from(positions[i1]);
                        let p2 = Vec3::from(positions[i2]);

                        let edge1 = p1 - p0;
                        let edge2 = p2 - p0;
                        let normal = edge1.cross(edge2).normalize();

                        computed_normals[i0] = (Vec3::from(computed_normals[i0]) + normal).to_array();
                        computed_normals[i1] = (Vec3::from(computed_normals[i1]) + normal).to_array();
                        computed_normals[i2] = (Vec3::from(computed_normals[i2]) + normal).to_array();
                    }

                    computed_normals.iter()
                        .map(|&n| {
                            let normalized = Vec3::from(n).normalize();
                            [normalized.x, normalized.y, normalized.z]
                        })
                        .collect()
                }
            };

            let indices = reader.read_indices()
                .map(|i| i.into_u32().collect::<Vec<u32>>())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());

            for chunk in indices.chunks(3) {
                if chunk.len() != 3 { continue; }

                let mut triangle = Triangle {
                    material_index: Alignedu32(material_index),
                    is_sphere: Alignedu32(0),
                    vertices: [AlignedVec3(Vec3::ZERO); 3],
                    normals: [AlignedVec3(Vec3::ZERO); 3],
                };

                for (i, &index) in chunk.iter().enumerate() {
                    let pos = positions[index as usize];
                    let normal = normals[index as usize];
                    
                    // Transform vertex and normal using the accumulated transform
                    let world_pos = transform.transform_point3(Vec3::from(pos));
                    let world_normal = (normal_matrix * Vec3::from(normal)).normalize();

                    triangle.vertices[i] = AlignedVec3(world_pos);
                    triangle.normals[i] = AlignedVec3(world_normal);
                }

                self.components.triangles.push(triangle);
            }
        }

        Ok(())
    }
    fn process_material(&mut self, material: &gltf::Material) -> Result<Material> {
        let pbr = material.pbr_metallic_roughness();
        
        Ok(Material {
            base_colour: AlignedVec3(Vec4::from(pbr.base_color_factor()).truncate()),
            emission: AlignedVec3(Vec3::from(material.emissive_factor())),
            metallic: Alignedf32(pbr.metallic_factor()),
            roughness: Alignedf32(pbr.roughness_factor()),
            transmission: Alignedf32(0.0),
            ior: Alignedf32(1.5),
            motion_blur: Default::default(),
            shade_smooth: Alignedu32(1),
        })
    }
}
