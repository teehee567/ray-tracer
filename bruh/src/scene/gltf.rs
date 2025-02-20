use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use serde_yaml::Value;
use vulkanalia::vk;

use crate::{
    AMat4, AUVec2, AVec2, AVec3, Af32, Au32,
    CameraBufferObject, Material, SceneComponents, Triangle,
};

const CONFIG_VERSION: &str = "0.2";

use anyhow::{anyhow, Result};

use super::{Scene, TextureData, TextureFormat};

impl TextureFormat {
    fn from_gltf(format: gltf::image::Format) -> Self {
        match format {
            gltf::image::Format::R8 => TextureFormat::R8,
            gltf::image::Format::R8G8 => TextureFormat::R8G8,
            gltf::image::Format::R8G8B8 => TextureFormat::R8G8B8,
            gltf::image::Format::R8G8B8A8 => TextureFormat::R8G8B8A8,
            _ => unimplemented!(),
        }
    }

    fn to_vulkan(&self) -> vk::Format {
        match self {
            TextureFormat::R8 => vk::Format::R8_UNORM,
            TextureFormat::R8G8 => vk::Format::R8G8_UNORM,
            TextureFormat::R8G8B8 => vk::Format::R8G8B8_SRGB,
            TextureFormat::R8G8B8A8 => vk::Format::R8G8B8A8_SRGB,
            _ => unimplemented!(),
        }
    }
}

fn camera_lake() -> CameraBufferObject {
    let mut ubo = CameraBufferObject {
        focal_length: Af32(1.),
        focus_distance: Af32(55.),
        aperture_radius: Af32(0.),
        location: AVec3(Vec3::new(35.0, 12.0, 35.0)),
        ..Default::default()
    };
    let resolution = UVec2::new(1920, 1080);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    ubo.rotation = AMat4(Mat4::look_at_rh(ubo.location.0, look_at, Vec3::Y).transpose());

    let ratio = resolution[0] as f32 / resolution[1] as f32;
    let (u, v) = if ratio > 1.0 {
        (ratio, 1.0)
    } else {
        (1.0, 1.0 / ratio)
    };
    ubo.view_port_uv = AVec2(Vec2::new(u, v));
    ubo.resolution = AUVec2(resolution);
    ubo
}

fn camera_car() -> CameraBufferObject {
    let mut ubo = CameraBufferObject {
        focal_length: Af32(2.),
        focus_distance: Af32(55.),
        aperture_radius: Af32(0.),
        location: AVec3(Vec3::new(3.0, 1.0, 5.)),
        ..Default::default()
    };
    let resolution = UVec2::new(1920, 1080);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    ubo.rotation = AMat4(Mat4::look_at_rh(ubo.location.0, look_at, Vec3::Y).transpose());

    let ratio = resolution[0] as f32 / resolution[1] as f32;
    let (u, v) = if ratio > 1.0 {
        (ratio, 1.0)
    } else {
        (1.0, 1.0 / ratio)
    };
    ubo.view_port_uv = AVec2(Vec2::new(u, v));
    ubo.resolution = AUVec2(resolution);
    ubo
}

fn camera_sponza() -> CameraBufferObject {
    let mut ubo = CameraBufferObject {
        focal_length: Af32(1.),
        focus_distance: Af32(55.),
        aperture_radius: Af32(0.),
        location: AVec3(Vec3::new(3.0, 2.0, 0.0)),
        ..Default::default()
    };
    let resolution = UVec2::new(1920, 1080);
    let look_at = Vec3::new(-4.0, 2.0, 0.0);
    ubo.rotation = AMat4(Mat4::look_at_rh(ubo.location.0, look_at, Vec3::Y).transpose());

    let ratio = resolution[0] as f32 / resolution[1] as f32;
    let (u, v) = if ratio > 1.0 {
        (ratio, 1.0)
    } else {
        (1.0, 1.0 / ratio)
    };
    ubo.view_port_uv = AVec2(Vec2::new(u, v));
    ubo.resolution = AUVec2(resolution);
    ubo
}

fn camera_mclaren() -> CameraBufferObject {
    let mut ubo = CameraBufferObject {
        focal_length: Af32(70.),
        focus_distance: Af32(55.),
        aperture_radius: Af32(0.),
        location: AVec3(Vec3::new(1.15, 0.1, 2.)),
        ..Default::default()
    };
    let resolution = UVec2::new(1920, 1080);
    let rotation = Vec3::new(-2.5, 210., -0.);
    
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    ubo.rotation = AMat4(Mat4::look_at_rh(ubo.location.0, look_at, Vec3::Y).transpose());

    let ratio = resolution[0] as f32 / resolution[1] as f32;
    let (u, v) = if ratio > 1.0 {
        (ratio, 1.0)
    } else {
        (1.0, 1.0 / ratio)
    };
    ubo.view_port_uv = AVec2(Vec2::new(u, v));
    ubo.resolution = AUVec2(resolution);
    ubo
}

fn camera_interior() -> CameraBufferObject {
    let mut ubo = CameraBufferObject {
        focal_length: Af32(1.),
        focus_distance: Af32(55.),
        aperture_radius: Af32(0.),
        location: AVec3(Vec3::new(0.0, 0.0, 5.0)),
        ..Default::default()
    };
    let resolution = UVec2::new(1920, 1080);
    let look_at = Vec3::new(0.0, 0.0, 0.0);
    ubo.rotation = AMat4(Mat4::look_at_rh(ubo.location.0, look_at, Vec3::Y).transpose());

    let ratio = resolution[0] as f32 / resolution[1] as f32;
    let (u, v) = if ratio > 1.0 {
        (ratio, 1.0)
    } else {
        (1.0, 1.0 / ratio)
    };
    ubo.view_port_uv = AVec2(Vec2::new(u, v));
    ubo.resolution = AUVec2(resolution);
    ubo
}

impl Scene {
    pub fn from_gltf(path: &str) -> Result<Self> {
        let (gltf, buffers, images) = gltf::import(path)?;

        let mut scene = Scene {
            components: SceneComponents::default(),
            root: Value::default(),
        };

        scene.components.camera = camera_sponza();

        scene.load_textures(&images)?;

        // Process the default scene or first available scene
        let gltf_scene = gltf
            .default_scene()
            .or_else(|| gltf.scenes().next())
            .ok_or_else(|| anyhow!("No scenes found in GLTF file"))?;

        // Process all scenes
        scene.process_gltf_scene(&gltf_scene, &buffers, &images)?;
        println!("Triangles: {}", scene.components.triangles.len());

        scene.build_bvh();

        Ok(scene)
    }

    fn load_textures(&mut self, images: &[gltf::image::Data]) -> Result<()> {
        for image in images {
            // Convert RGBA if necessary
            let pixels = match image.format {
                gltf::image::Format::R8G8B8 => {
                    // Convert RGB to RGBA
                    let mut rgba =
                        Vec::with_capacity(image.width as usize * image.height as usize * 4);
                    for chunk in image.pixels.chunks(3) {
                        rgba.extend_from_slice(chunk);
                        rgba.push(255); // Alpha channel
                    }
                    rgba
                }
                _ => image.pixels.clone(),
            };

            let texture = TextureData {
                width: image.width,
                height: image.height,
                format: TextureFormat::R8G8B8A8,
                pixels,
            };
            self.components.textures.push(texture);
        }
        Ok(())
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
                let normals: Vec<[f32; 3]> = reader
                    .read_normals()
                    .map(|n| n.collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0, 1.0]; positions.len()]);

                // Read UV coordinates, default to [0,0] if not present
                let uvs: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .map(|tc| tc.into_f32().collect())
                    .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);

                if let Some(indices) = reader.read_indices() {
                    let indices: Vec<u32> = indices.into_u32().collect();

                    for chunk in indices.chunks(3) {
                        if chunk.len() == 3 {
                            let mut triangle = Triangle {
                                material_index: Au32(material_index),
                                is_sphere: Au32(0),
                                vertices: [AVec3(Vec3::ZERO); 3],
                                normals: [AVec3(Vec3::ZERO); 3],
                                uvs: [AVec2(Vec2::ZERO); 3], // Initialize UVs
                            };

                            for (i, &index) in chunk.iter().enumerate() {
                                let pos = positions[index as usize];
                                let normal = normals[index as usize];
                                let uv = uvs[index as usize];

                                let transformed_pos = transform.transform_point3(Vec3::from(pos));
                                let transformed_normal =
                                    transform.transform_vector3(Vec3::from(normal)).normalize();

                                triangle.vertices[i] = AVec3(transformed_pos);
                                triangle.normals[i] = AVec3(transformed_normal);
                                triangle.uvs[i] = AVec2(Vec2::new(uv[0], uv[1]));
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

        let base_color_tex = pbr
            .base_color_texture()
            .map(|tex| tex.texture().source().index() as u32)
            .unwrap_or(u32::MAX);

        let metallic_roughness_tex = pbr
            .metallic_roughness_texture()
            .map(|tex| tex.texture().source().index() as u32)
            .unwrap_or(u32::MAX);

        let normal_tex = material
            .normal_texture()
            .map(|tex| tex.texture().source().index() as u32)
            .unwrap_or(u32::MAX);

        let emission_tex = material
            .emissive_texture()
            .map(|tex| tex.texture().source().index() as u32)
            .unwrap_or(u32::MAX);

        let base_color = pbr.base_color_factor();
        let metallic = Af32(pbr.metallic_factor());
        let roughness = Af32(pbr.roughness_factor());
        let emissive = material.emissive_factor();
        let ior = Af32(material.ior().unwrap_or(1.5));
        let transmission = Af32(
            material
                .transmission()
                .and_then(|ext| Some(ext.transmission_factor()))
                .unwrap_or(1.0 - base_color[3]),
        );

        // Get emissive strength from extension
        let emissive_strength = material.emissive_strength().unwrap_or(0.0);

        let specular = material
            .specular()
            .and_then(|ext| Some(ext.specular_factor()))
            .unwrap_or(0.);
        let specular_colour = Vec3::from(
            material
                .specular()
                .and_then(|ext| Some(ext.specular_color_factor()))
                .unwrap_or([0.; 3]),
        );
        let specular_tex = material
            .specular()
            .and_then(|spec| spec.specular_texture())
            .map(|tex| tex.texture().source().index() as u32)
            .unwrap_or(u32::MAX);

        let clearcoat = pbr
            .extensions()
            .and_then(|ext| {
                ext.get("KHR_materials_clearcoat")
                    .and_then(|json| json.as_object())
            });

        let clearcoat_factor = clearcoat
            .and_then(|c| c.get("clearcoatFactor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let clearcoat_roughness = clearcoat
            .and_then(|c| c.get("clearcoatRoughnessFactor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;

        let clearcoat_texture = clearcoat
            .and_then(|c| c.get("clearcoatTexture"))
            .and_then(|t| t.as_object())
            .and_then(|t| t.get("index"))
            .and_then(|i| i.as_u64())
            .map(|i| i as u32)
            .unwrap_or(u32::MAX);

        let clearcoat_roughness_texture = clearcoat
            .and_then(|c| c.get("clearcoatRoughnessTexture"))
            .and_then(|t| t.as_object())
            .and_then(|t| t.get("index"))
            .and_then(|i| i.as_u64())
            .map(|i| i as u32)
            .unwrap_or(u32::MAX);

        let emission = AVec3(if emissive_strength != 0.0 {
            Vec3::new(
                emissive[0] * emissive_strength,
                emissive[1] * emissive_strength,
                emissive[2] * emissive_strength,
            )
        } else {
            emissive.into()
        });

        let materiala = Material {
            base_colour: AVec3(Vec4::from(base_color).truncate()),
            emission,
            metallic,
            roughness,
            ior,
            spec_trans: transmission,

            clearcoat: Af32(clearcoat_factor),
            clearcoat_roughness: Af32(clearcoat_roughness),

            shade_smooth: Au32(0),

            base_color_tex: Au32(base_color_tex),
            metallic_roughness_tex: Au32(metallic_roughness_tex),
            normal_tex: Au32(normal_tex),
            emission_tex: Au32(emission_tex),
            ..Default::default()
        };
        // dbg!(&materiala);

        Ok(materiala)
    }

    fn process_camera(&mut self, camera: &gltf::Camera, node: &gltf::Node) -> Result<()> {
        match camera.projection() {
            gltf::camera::Projection::Perspective(persp) => {
                let transform = Mat4::from_cols_array_2d(&node.transform().matrix());
                let position = transform.col(3).truncate();

                self.components.camera.focal_length = Af32(persp.yfov().to_degrees());
                self.components.camera.focus_distance = Af32(persp.znear());
                self.components.camera.location =
                    AVec3(Vec3::new(position.x, position.y, position.z));

                // Extract rotation from transform
                self.components.camera.rotation =
                    AMat4(Mat4::from_cols_array_2d(&node.transform().matrix()));
            }
            gltf::camera::Projection::Orthographic(_) => {
                unimplemented!("ORTHOGAPHIC CAMERA NOT IMPLEMENTED")
            }
        }
        Ok(())
    }
}
