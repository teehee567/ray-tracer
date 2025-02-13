use crate::{AlignedMat4, AlignedUVec2, AlignedVec2, AlignedVec3, Alignedf32, Alignedu32, CameraBufferObject, Material, SceneComponents, Triangle};

use super::Scene;
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::collections::HashMap;
use std::io::BufReader;
use std::fs::File;
use glam::{Mat4, UVec2, Vec2, Vec3};

use anyhow::{anyhow, Result};

impl Scene {
    /// Loads a Mitsuba XML scene file
    pub fn from_mitsuba(path: &str) -> Result<Self> {
        use quick_xml::Reader;
        use quick_xml::events::Event;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut reader = Reader::from_reader(buf_reader);
        let mut buf = Vec::new();

        let mut scene = Scene {
            root: serde_yaml::Value::Null, // Default empty root
            components: SceneComponents::default(),
        };

        // Temporary storage for scene data
        let mut camera_data = None;
        let mut meshes = Vec::new();
        let mut materials = HashMap::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"sensor" => {
                            // Parse camera/sensor
                            camera_data = Some(parse_mitsuba_camera(&mut reader)?);
                        },
                        b"shape" => {
                            // Parse geometry 
                            let (mesh, material) = parse_mitsuba_shape(&mut reader)?;
                            meshes.push((mesh, material));
                        },
                        b"bsdf" => {
                            // Parse material definition
                            let (id, mat) = parse_mitsuba_material(&mut reader)?;
                            materials.insert(id, mat);
                        },
                        _ => ()
                    }
                },
                Ok(Event::Eof) => break,
                Err(e) => return Err(anyhow::anyhow!("Error parsing XML: {}", e)),
                _ => (),
            }
        }

        // Convert parsed data into scene components
        if let Some(camera) = camera_data {
            scene.components.camera = camera;
        }

        // Process meshes and materials
        for (mesh, material_ref) in meshes {
            // Look up material if referenced by ID
            let material = if let Some(mat_id) = material_ref {
                materials.get(&mat_id).cloned().unwrap_or_default()
            } else {
                Material::default() 
            };

            let material_index = scene.components.materials.len() as u32;
            scene.components.materials.push(material);

            // Convert mesh data to triangles
            scene.components.triangles.extend(mesh.into_triangles(material_index));
        }

        // Build BVH
        scene.build_bvh();

        Ok(scene)
    }
}

// Helper functions to parse specific Mitsuba XML elements

fn parse_mitsuba_camera(reader: &mut Reader<BufReader<File>>) -> Result<CameraBufferObject> {
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
    Ok(ubo)
}

fn parse_mitsuba_shape(reader: &mut Reader<BufReader<File>>) 
    -> Result<(MitsubaMesh, Option<String>)> 
{
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    let mut material_id = None;
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"float" => {
                        // Parse array data
                        if let Some(name) = get_attribute(e, "name")? {
                            match name.as_str() {
                                "positions" => {
                                    vertices = parse_float_array(reader, 3)?;
                                },
                                "normals" => {
                                    normals = parse_float_array(reader, 3)?;
                                },
                                _ => ()
                            }
                        }
                    },
                    b"integer" => {
                        // Parse indices
                        if let Some(name) = get_attribute(e, "name")? {
                            if name == "indices" {
                                let raw_indices = parse_int_array(reader)?;
                                indices = raw_indices
                                    .chunks(3)
                                    .map(|chunk| [chunk[0] as u32, chunk[1] as u32, chunk[2] as u32])
                                    .collect();
                            }
                        }
                    },
                    b"ref" => {
                        // Get material reference
                        if let Some(id) = get_attribute(e, "id")? {
                            material_id = Some(id);
                        }
                    }
                    _ => ()
                }
            },
            Ok(Event::End(ref e)) if e.name() == quick_xml::name::QName(b"shape") => {
                break;
            },
            Ok(Event::Eof) => {
                break;
            },
            Err(e) => return Err(anyhow::anyhow!("Error parsing shape: {}", e)),
            _ => (),
        }
    }

    // If normals weren't provided, calculate them
    if normals.is_empty() {
        normals = calculate_normals(&vertices, &indices)?;
    }

    Ok((MitsubaMesh {
        vertices,
        normals, 
        indices
    }, material_id))
}

fn parse_mitsuba_material(reader: &mut Reader<BufReader<File>>) 
    -> Result<(String, Material)> 
{
    let mut material = Material::default();
    let mut material_id = String::new();
    let mut buf = Vec::new();

    // Get material ID from attributes
    if let Event::Start(ref e) = reader.read_event_into(&mut buf)? {
        if let Some(id) = get_attribute(e, "id")? {
            material_id = id;
        }
    }

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"rgb" => {
                        // Parse color values
                        if let Some(name) = get_attribute(e, "name")? {
                            if let Some(value) = get_attribute(e, "value")? {
                                let color = parse_rgb_string(&value)?;
                                match name.as_str() {
                                    "baseColor" | "diffuseReflectance" => {
                                        material.base_colour = color.into();
                                    },
                                    "emittance" => {
                                        material.emission = color.into();
                                    },
                                    _ => ()
                                }
                            }
                        }
                    },
                    b"float" => {
                        // Parse float parameters
                        if let Some(name) = get_attribute(e, "name")? {
                            if let Some(value) = get_attribute(e, "value")? {
                                let float_val = value.parse::<f32>()?;
                                match name.as_str() {
                                    "roughness" => {
                                        material.roughness = Alignedf32(float_val);
                                    },
                                    "ior" => {
                                        material.ior = Alignedf32(float_val);
                                    },
                                    "metallic" => {
                                        material.metallic = Alignedf32(float_val);
                                    },
                                    "specularTransmission" => {
                                        material.transmission = Alignedf32(float_val);
                                    },
                                    _ => ()
                                }
                            }
                        }
                    },
                    _ => ()
                }
            },
            Ok(Event::End(ref e)) if e.name() == quick_xml::name::QName(b"bsdf") => {
                break;
            },
            Ok(Event::Eof) => break,
            Err(e) => return Err(anyhow::anyhow!("Error parsing material: {}", e)),
            _ => (),
        }
    }

    Ok((material_id, material))
}

// Helper struct for intermediate mesh representation
struct MitsubaMesh {
    vertices: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<[u32; 3]>,
}

impl MitsubaMesh {
    fn into_triangles(self, material_index: u32) -> Vec<Triangle> {
        let mut triangles = Vec::new();
        
        for idx in self.indices {
            let tri = Triangle {
                material_index: Alignedu32(material_index),
                is_sphere: Alignedu32(0),
                vertices: [
                    AlignedVec3::from(self.vertices[idx[0] as usize]),
                    AlignedVec3::from(self.vertices[idx[1] as usize]), 
                    AlignedVec3::from(self.vertices[idx[2] as usize])
                ],
                normals: [
                    AlignedVec3::from(self.normals[idx[0] as usize]),
                    AlignedVec3::from(self.normals[idx[1] as usize]),
                    AlignedVec3::from(self.normals[idx[2] as usize])  
                ]
            };
            triangles.push(tri);
        }
        
        triangles
    }
}

fn parse_float_array(reader: &mut Reader<BufReader<File>>, stride: usize) -> Result<Vec<[f32; 3]>> {
    let mut buf = Vec::new();
    let mut result = Vec::new();

    if let Ok(Event::Text(e)) = reader.read_event_into(&mut buf) {
        // Convert bytes to string
        let text = String::from_utf8(e.into_owned().to_vec())?;
        let values: Vec<f32> = text
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();

        result = values
            .chunks(stride)
            .map(|chunk| [
                chunk[0],
                chunk[1],
                chunk[2],
            ])
            .collect();
    }

    Ok(result)
}

fn parse_int_array(reader: &mut Reader<BufReader<File>>) -> Result<Vec<i32>> {
    let mut buf = Vec::new();
    
    if let Ok(Event::Text(e)) = reader.read_event_into(&mut buf) {
        // Convert bytes to string
        let text = String::from_utf8(e.into_owned().to_vec())?;
        return Ok(text
            .split_whitespace()
            .filter_map(|s| s.parse::<i32>().ok())
            .collect());
    }
    
    Ok(Vec::new())
}

fn get_attribute(e: &BytesStart, name: &str) -> Result<Option<String>> {
    Ok(e.attributes()
        .find(|a| {
            if let Ok(attr) = a {
                attr.key == quick_xml::name::QName(name.as_bytes())
            } else {
                false
            }
        })
        .and_then(|a| a.ok())
        .and_then(|a| String::from_utf8(a.value.into_owned()).ok()))
}

fn parse_rgb_string(s: &str) -> Result<[f32; 3]> {
    let values: Vec<f32> = s
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();

    if values.len() != 3 {
        return Err(anyhow::anyhow!("Invalid RGB string format"));
    }

    Ok([values[0], values[1], values[2]])
}

fn calculate_normals(vertices: &[[f32; 3]], indices: &[[u32; 3]]) -> Result<Vec<[f32; 3]>> {
    let mut normals = vec![[0.0, 0.0, 0.0]; vertices.len()];
    
    for triangle in indices {
        let v0 = Vec3::from(vertices[triangle[0] as usize]);
        let v1 = Vec3::from(vertices[triangle[1] as usize]);
        let v2 = Vec3::from(vertices[triangle[2] as usize]);
        
        let normal = (v1 - v0).cross(v2 - v0).normalize();
        
        // Add the face normal to each vertex normal
        for &index in triangle {
            let n = &mut normals[index as usize];
            *n = [
                n[0] + normal.x,
                n[1] + normal.y, 
                n[2] + normal.z
            ];
        }
    }
    
    // Normalize all normals
    for normal in &mut normals {
        let n = Vec3::from(*normal).normalize();
        *normal = [n.x, n.y, n.z];
    }
    
    Ok(normals)
}
