use std::{fs::File, io::Read, path::PathBuf};

use crate::{AVec2, AVec3, Au32, CameraBufferObject, Material, SceneComponents, Triangle};

use super::Scene;

use anyhow::Result;
use glam::Vec3;

impl Scene {
    pub fn from_new(path: &str) -> Result<Self> {
        let mut scene = Scene {
            root: serde_yaml::Value::default(),
            components: SceneComponents::default(),
        };
        scene.components = Self::load_scene_yaml(path);
        scene.build_bvh();
        println!("Triangles: {}", scene.components.triangles.len());
        Ok(scene)
    }

    fn load_scene_yaml(path: &str) -> SceneComponents {
        let mut file = File::open(path).expect("Failed to open scene file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("Failed to read scene file");

        let yaml: serde_yaml::Value = serde_yaml::from_str(&contents).expect("Failed to parse YAML");

        let mut scene = SceneComponents::default();
        let mut material_name_to_index = std::collections::HashMap::new();


        // Load skybox
        if let Some(skybox) = yaml["skybox"].as_sequence() {
            let skyboxes: Vec<PathBuf> = skybox.iter().map(|skybox_path| {
                let base_dir = std::path::Path::new(path).parent().unwrap();
                base_dir.join(skybox_path.as_str().unwrap())
            }).collect();
            let skybox_array: [PathBuf; 6] = skyboxes.try_into().unwrap();
            scene.skybox = Self::load_cubemap_textures(skybox_array).unwrap();
        }

        // Load materials
        if let Some(materials) = yaml["materials"].as_mapping() {
            for (name, mat_data) in materials {
                let name = name.as_str().unwrap();
                
                let material: Material = serde_yaml::from_value(mat_data.clone())
                    .expect(&format!("Failed to deserialize material '{}'", name));

                let material_index = scene.materials.len();
                material_name_to_index.insert(name.to_string(), material_index);
                scene.materials.push(material);
            }
        }

        // Load surfaces/meshes
        if let Some(surfaces) = yaml["surfaces"].as_sequence() {
            for surface in surfaces {
                if let Some(file_path) = surface["file"].as_str() {
                    let base_dir = std::path::Path::new(path).parent().unwrap();
                    let full_path = base_dir.join(file_path);
                    let material_name = surface["material"].as_str().expect("Material not specified");
                    let material_index = material_name_to_index[material_name];
                    let smooth = surface["smooth"].as_bool().unwrap_or(false);

                    // Load the OBJ file
                    let obj_result = tobj::load_obj(
                        full_path,
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
                                for face in 0..mesh.indices.len() / 3 {
                                    let i0 = mesh.indices[face * 3] as usize;
                                    let i1 = mesh.indices[face * 3 + 1] as usize;
                                    let i2 = mesh.indices[face * 3 + 2] as usize;

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
                } else if surface["type"].as_str() == Some("mesh") {
                    let material_name = surface["material"].as_str().expect("Material not specified");
                    let material_index = material_name_to_index[material_name];

                    // Load vertices
                    let vertices: Vec<Vec3> = surface["vertices"]
                        .as_sequence()
                        .expect("Vertices not found or invalid")
                        .iter()
                        .map(|v| {
                            let coords = v.as_sequence().expect("Invalid vertex format");
                            Vec3::new(
                                coords[0].as_f64().unwrap() as f32,
                                coords[1].as_f64().unwrap() as f32,
                                coords[2].as_f64().unwrap() as f32,
                            )
                        })
                        .collect();

                    // Load triangle indices
                    let triangles = surface["triangles"]
                        .as_sequence()
                        .expect("Triangles not found or invalid");

                    // Create triangles from indices
                    for triangle_indices in triangles {
                        let indices = triangle_indices.as_sequence().expect("Invalid triangle format");
                        let i0 = indices[0].as_u64().unwrap() as usize;
                        let i1 = indices[1].as_u64().unwrap() as usize;
                        let i2 = indices[2].as_u64().unwrap() as usize;

                        // Calculate normal (flat shading)
                        let v0 = vertices[i0];
                        let v1 = vertices[i1];
                        let v2 = vertices[i2];
                        
                        let normal = (v1 - v0).cross(v2 - v0).normalize();

                        let triangle = Triangle {
                            material_index: Au32(material_index as u32),
                            is_sphere: Au32(0),
                            vertices: [
                                AVec3(vertices[i0]),
                                AVec3(vertices[i1]), 
                                AVec3(vertices[i2])
                            ],
                            normals: [
                                AVec3(normal),
                                AVec3(normal),
                                AVec3(normal)
                            ],
                            uvs: [
                                AVec2([0.0, 0.0].into()),
                                AVec2([0.0, 0.0].into()),
                                AVec2([0.0, 0.0].into())
                            ],
                        };

                        scene.triangles.push(triangle);
                    }

                }
            }
        }

        // Load camera
        if let Some(camera) = yaml["camera"].as_mapping() {
            let camera_data: CameraBufferObject = serde_yaml::from_value(serde_yaml::Value::Mapping(camera.clone()))
                .expect("Failed to deserialize camera");
            scene.camera = camera_data;
        }

        scene
    }

}
