use std::{fs::File, io::Read};

use crate::{
    AMat4, AUVec2, AVec2, AVec3, Af32, Au32, CameraBufferObject, Material, SceneComponents,
    Triangle,
};

use super::Scene;

use anyhow::Result;
use glam::{Mat4, UVec2, Vec2, Vec3};
use serde_json::Value;

fn parse_rgb(hex: &str) -> Vec3 {
    let hex = hex.trim_start_matches('#');

    if hex.len() != 6 {
        panic!();
    }

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap();
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap();
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap();

    Vec3::new(r as f32 / 255., g as f32 / 255., b as f32 / 255.)
}
impl Scene {
    pub fn from_weird(path: &'static str) -> Result<Self> {
        let file = File::open(path)?;
        let mut scene = Scene {
            root: serde_yaml::Value::default(),
            components: SceneComponents::default(),
        };
        scene.components = Self::load_scene(path);

        scene.build_bvh();
        println!("Triangles: {}", scene.components.triangles.len());
        Ok(scene)
    }
    fn load_scene(path: &str) -> SceneComponents {
        let mut file = File::open(path).expect("Failed to open scene file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Failed to read scene file");

        let json: Value = serde_json::from_str(&contents).expect("Failed to parse JSON");

        let mut scene = SceneComponents::default();

        // Create material name to index mapping
        let mut material_name_to_index = std::collections::HashMap::new();

        // Load materials
        if let Some(materials) = json["materials"].as_object() {
            for (name, mat_data) in materials {
                println!("{}", name);
                let mut material = Material::default();

                // Handle reflectance (base_colour)
                if let Some(refl) = mat_data["reflectance"].as_str() {
                    material.base_colour = AVec3(parse_rgb(refl));
                }
                if let Some(refl) = mat_data["reflectance"].as_array() {
                    material.base_colour = AVec3(Vec3::new(
                        refl[0].as_f64().unwrap_or(0.0) as f32,
                        refl[1].as_f64().unwrap_or(0.0) as f32,
                        refl[2].as_f64().unwrap_or(0.0) as f32,
                    ));
                }
                if let Some(refl) = mat_data["reflectance"].as_f64() {
                    material.metallic = Af32(refl as f32);
                }

                if let Some(refl) = mat_data["specular"].as_f64() {
                    material.specular_tint = Af32(refl as f32);
                }

                // Handle emittance
                if let Some(emit) = mat_data["emittance"].as_str() {
                    material.emission = AVec3(parse_rgb(emit));
                    if let Some(emit) = mat_data["emissive_strength"].as_f64() {
                        material.emission = AVec3(material.emission.0 * emit as f32);
                    }
                }

                // Handle IOR
                if let Some(ior) = mat_data["ior"].as_f64() {
                    material.ior = Af32(ior as f32);
                }

                // Handle transparency
                if let Some(trans) = mat_data["transparency"].as_f64() {
                    material.spec_trans = Af32(trans as f32);
                }
                if let Some(trans) = mat_data["transmittance"].as_str() {
                    material.base_colour = AVec3(parse_rgb(trans));
                }

                // Handle perfect mirror
                if let Some(true) = mat_data["perfect_mirror"].as_bool() {
                    material.metallic = Af32(1.0);
                    material.roughness = Af32(0.0);
                }

                // Handle specular roughness
                if let Some(rough) = mat_data["specular_roughness"].as_f64() {
                    material.roughness = Af32(rough as f32);
                }
                material.metallic = Af32(0.3);
                // material.emission = material.base_colour;

                dbg!(&material);
                let material_index = scene.materials.len();
                material_name_to_index.insert(name.clone(), material_index);
                scene.materials.push(material);
            }
        }

        // Load vertices for explicit geometry (lights)
        let mut vertex_sets: std::collections::HashMap<String, Vec<Vec3>> =
            std::collections::HashMap::new();
        if let Some(vertices) = json["vertices"].as_object() {
            for (name, vert_array) in vertices {
                let mut verts = Vec::new();
                if let Some(array) = vert_array.as_array() {
                    for vert in array {
                        if let Some(coords) = vert.as_array() {
                            verts.push(Vec3::new(
                                coords[0].as_f64().unwrap() as f32,
                                coords[1].as_f64().unwrap() as f32,
                                coords[2].as_f64().unwrap() as f32,
                            ));
                        }
                    }
                }
                vertex_sets.insert(name.clone(), verts);
            }
        }

        // Load surfaces/triangles
        if let Some(surfaces) = json["surfaces"].as_array() {
            for surface in surfaces {
                // Handle explicitly defined triangles (like lights)
                if let (Some(vertex_set), Some(triangles)) = (
                    surface["vertex_set"].as_str(),
                    surface["triangles"].as_array(),
                ) {
                    let verts = vertex_sets.get(vertex_set).expect("Vertex set not found");
                    let material_name = surface["material"]
                        .as_str()
                        .expect("Material not specified");
                    let material_index = material_name_to_index[material_name];

                    for tri_indices in triangles {
                        if let Some(indices) = tri_indices.as_array() {
                            let triangle = Triangle {
                                material_index: Au32(material_index as u32),
                                is_sphere: Au32(0),
                                vertices: [
                                    AVec3(verts[indices[0].as_u64().unwrap() as usize]),
                                    AVec3(verts[indices[1].as_u64().unwrap() as usize]),
                                    AVec3(verts[indices[2].as_u64().unwrap() as usize]),
                                ],
                                normals: [AVec3(Vec3::ZERO), AVec3(Vec3::ZERO), AVec3(Vec3::ZERO)],
                                uvs: Default::default(),
                            };
                            scene.triangles.push(triangle);
                        }
                    }
                }

                // Load OBJ files
                if let Some(mut file_path) = surface["file"].as_str() {
                    let base_dir = std::path::Path::new(path).parent().unwrap();
                    let base = base_dir.join(file_path);
                    file_path = base.to_str().unwrap();

                    let material_name = surface["material"]
                        .as_str()
                        .expect("Material not specified");
                    let material_index = material_name_to_index[material_name];
                    let smooth = surface["smooth"].as_bool().unwrap_or(false);

                    // Load the OBJ file
                    let obj_result = tobj::load_obj(
                        file_path,
                        &tobj::LoadOptions {
                            single_index: true,
                            triangulate: true,
                            ..Default::default()
                        },
                    );

                    match obj_result {
                        Ok((models, _)) => {
                            for model in models {
                                let mesh = &model.mesh;

                                // Process each face/triangle
                                for face in 0..mesh.indices.len() / 3 {
                                    let i0 = mesh.indices[face * 3] as usize;
                                    let i1 = mesh.indices[face * 3 + 1] as usize;
                                    let i2 = mesh.indices[face * 3 + 2] as usize;

                                    // Get vertices
                                    let v0 = Vec3::new(
                                        mesh.positions[i0 * 3],
                                        mesh.positions[i0 * 3 + 1],
                                        mesh.positions[i0 * 3 + 2],
                                    );
                                    let v1 = Vec3::new(
                                        mesh.positions[i1 * 3],
                                        mesh.positions[i1 * 3 + 1],
                                        mesh.positions[i1 * 3 + 2],
                                    );
                                    let v2 = Vec3::new(
                                        mesh.positions[i2 * 3],
                                        mesh.positions[i2 * 3 + 1],
                                        mesh.positions[i2 * 3 + 2],
                                    );

                                    // Get normals if they exist
                                    let mut n0 = Vec3::ZERO;
                                    let mut n1 = Vec3::ZERO;
                                    let mut n2 = Vec3::ZERO;

                                    if !mesh.normals.is_empty() {
                                        n0 = Vec3::new(
                                            mesh.normals[i0 * 3],
                                            mesh.normals[i0 * 3 + 1],
                                            mesh.normals[i0 * 3 + 2],
                                        );
                                        n1 = Vec3::new(
                                            mesh.normals[i1 * 3],
                                            mesh.normals[i1 * 3 + 1],
                                            mesh.normals[i1 * 3 + 2],
                                        );
                                        n2 = Vec3::new(
                                            mesh.normals[i2 * 3],
                                            mesh.normals[i2 * 3 + 1],
                                            mesh.normals[i2 * 3 + 2],
                                        );
                                    }

                                    // Get UVs if they exist
                                    let mut uv0 = [0.0, 0.0];
                                    let mut uv1 = [0.0, 0.0];
                                    let mut uv2 = [0.0, 0.0];

                                    if !mesh.texcoords.is_empty() {
                                        uv0 = [mesh.texcoords[i0 * 2], mesh.texcoords[i0 * 2 + 1]];
                                        uv1 = [mesh.texcoords[i1 * 2], mesh.texcoords[i1 * 2 + 1]];
                                        uv2 = [mesh.texcoords[i2 * 2], mesh.texcoords[i2 * 2 + 1]];
                                    }

                                    let triangle = Triangle {
                                        material_index: Au32(material_index as u32),
                                        is_sphere: Au32(0),
                                        vertices: [AVec3(v0), AVec3(v1), AVec3(v2)],
                                        normals: [AVec3(n0), AVec3(n1), AVec3(n2)],
                                        uvs: [
                                            AVec2(uv0.into()),
                                            AVec2(uv1.into()),
                                            AVec2(uv2.into()),
                                        ],
                                    };

                                    scene.triangles.push(triangle);
                                }
                            }
                        }
                        Err(e) => eprintln!("Failed to load OBJ file {}: {:?}", file_path, e),
                    }
                }
            }
        }

        let cameras = json["cameras"].as_array().unwrap();
        let camera = &cameras[0];
        let location = camera["eye"].as_array().unwrap();
        let location = Vec3::new(
            location[0].as_f64().unwrap() as f32,
            location[1].as_f64().unwrap() as f32,
            location[2].as_f64().unwrap() as f32,
        );

        let look_at = camera["look_at"].as_array().unwrap();
        let look_at = Vec3::new(
            look_at[0].as_f64().unwrap() as f32,
            look_at[1].as_f64().unwrap() as f32,
            look_at[2].as_f64().unwrap() as f32,
        );
        let mut ubo = CameraBufferObject {
            focal_length: Af32(3.),
            focus_distance: Af32(1.),
            aperture_radius: Af32(0.),
            location: AVec3(location),
            ..Default::default()
        };
        let resolution = UVec2::new(1920, 1080);

        ubo.rotation = AMat4(Mat4::look_at_rh(ubo.location.0, look_at, Vec3::Y).transpose());

        let ratio = resolution[0] as f32 / resolution[1] as f32;
        let (u, v) = if ratio > 1.0 {
            (ratio, 1.0)
        } else {
            (1.0, 1.0 / ratio)
        };
        ubo.view_port_uv = AVec2(Vec2::new(u, v));
        ubo.resolution = AUVec2(resolution);
        scene.camera = ubo;

        scene
    }
}
