use std::fs::File;

use serde::{Deserialize, Serialize};
use tobj::{Material, Model};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct SceneFile<'a> {
    version: &'a str,
    camera: Camera,
    scene: Box<Vec<Mesh<'a>>>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Camera {
    resolution: [u32; 2],
    focal_length: f32,
    focus_distance: f32,
    aperture_radius: f32,
    location: [f32; 3],
    rotation: [f32; 3],
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Mesh<'a> {
    #[serde(rename = "type")]
    type_field: &'a str,
    material: SceneMaterial,
    data: Data,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct SceneMaterial {
    base_color: Option<[f32; 3]>,
    emission: Option<[f32; 3]>,
    reflectiveness: Option<f32>,
    roughness: Option<f32>,
    ior: Option<f32>,
    is_glass: Option<bool>,
    smooth_shading: Option<bool>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Data {
    vertices: Box<Vec<f32>>,
    normals: Box<Vec<f32>>,
}

/**
 * Entrypoint of script for converting OBJ and MTL files into the yaml document
 * required by the ray tracer.
 * 
 * Epects the name of the .obj file (with extension) to be passed in as first
 * CLI argument/
 */
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut meshes: Vec<Mesh> = Vec::new();

    let (models, materials) = load_obj();

    // loop over each model, adding it to the list of meshes
    for model in models.iter() {
        let (vertices, normals) = get_mesh_vertices(model);
        let material = get_material(model, &materials);

        meshes.push(Mesh {
            type_field: "TriMesh", // all OBJs are loaded as triangle meshes
            material,
            data: Data {
                vertices: Box::new(vertices),
                normals: Box::new(normals),
            },
        });
    }

    let scene = SceneFile {
        version: "0.2",
        camera: Camera {
            resolution: [800, 600],
            focal_length: 1.0,
            focus_distance: 5.0,
            aperture_radius: 0.0,
            location: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
        },
        scene: Box::new(meshes),
    };

    let file: File = File::create("scene.yaml")?;
    serde_yaml::to_writer(file, &scene)?;

    Ok(())
}

/**
 * Handles loading the OBJ and MTL files. Both must be present in the same dir,
 * and must have the same name to be properly recognized.
 */
fn load_obj() -> (Vec<Model>, Vec<Material>) {
    let obj_file = std::env::args()
        .skip(1)
        .next()
        .expect("Please input the obj filepath");

    let (models, materials) =
        tobj::load_obj(&obj_file, &tobj::LoadOptions::default()).expect("OBJ file not found");

    let materials = materials.expect("MTL file not found");

    return (models, materials);
}

/**
 * Handles converting the material from MTL format to YAML. 
 * 
 * Not a perfect conversion, but glass, base color, ior, and emissiveness are
 * converted.
 */
fn get_material(model: &Model, materials: &Vec<Material>) -> SceneMaterial {
    let material = model.mesh.material_id.and_then(|m| materials.iter().nth(m));

    // the base color is inherited from the "diffuse" property
    let base_color = material.and_then(|m| m.diffuse);

    // whether the material is glass is inherited from the presence of the "Tf" property
    let is_glass = material.and_then(|m| m.unknown_param.get("Tf").and_then(|_| Some(true)));

    // the IOR comes from the "optical density"
    let ior = material.and_then(|m| m.optical_density);

    // the emission comes from the "Ke" property
    let emission = material
        .and_then(|m| m.unknown_param.get("Ke"))
        .and_then(|ke| {
            let a: Vec<f32> = ke.split(" ").filter_map(|s| s.parse().ok()).collect();
            Some([a[0], a[1], a[2]])
        });

    SceneMaterial {
        base_color,
        emission,
        ior,
        is_glass,
        reflectiveness: None,
        roughness: None,
        smooth_shading: None,
    }
}

/**
 * Handles the conversion of vertices from the OBJ file to the YAML.
 * 
 * This requires some care as OBJ supports quads and they need to be converted
 * to two triangles each. OBJ is also indexed whereas the YAML format we use
 * requires raw coordinates for each triangle.
 */
fn get_mesh_vertices(model: &Model) -> (Vec<f32>, Vec<f32>) {
    let mesh = &model.mesh;

    // Create empty lists
    let mut vertices: Vec<f32> = Vec::new();
    let mut normals: Vec<f32> = Vec::new();

    // Check if the file has different n-gon types.
    if mesh.face_arities.len() == 0 {
        for idx in &mesh.indices {
            vertices.push(mesh.positions[(idx * 3 + 0) as usize]);
            vertices.push(mesh.positions[(idx * 3 + 1) as usize]);
            vertices.push(mesh.positions[(idx * 3 + 2) as usize]);
        }

        for idx in &mesh.normal_indices {
            normals.push(mesh.normals[(idx * 3 + 0) as usize]);
            normals.push(mesh.normals[(idx * 3 + 1) as usize]);
            normals.push(mesh.normals[(idx * 3 + 2) as usize]);
        }
    } else {
        let mut next_face = 0;
        for face in 0..mesh.face_arities.len() {
            let end = next_face + mesh.face_arities[face] as usize;
            let face_indices = &mesh.indices[next_face..end];
            let normal_face_indices = &mesh.normal_indices[next_face..end];

            let indices = if mesh.face_arities[face] == 4 {
                vec![0, 1, 2, 0, 2, 3]
            } else if mesh.face_arities[face] == 3 {
                vec![0, 1, 2]
            } else {
                panic!("only 3 or 4-gons are currently supported.")
            };

            for idx in indices {
                vertices.push(mesh.positions[(face_indices[idx] * 3 + 0) as usize]);
                vertices.push(mesh.positions[(face_indices[idx] * 3 + 1) as usize]);
                vertices.push(mesh.positions[(face_indices[idx] * 3 + 2) as usize]);

                normals.push(mesh.normals[(normal_face_indices[idx] * 3 + 0) as usize]);
                normals.push(mesh.normals[(normal_face_indices[idx] * 3 + 1) as usize]);
                normals.push(mesh.normals[(normal_face_indices[idx] * 3 + 2) as usize]);
            }

            next_face = end;
        }
    }

    // return the lists, now populated
    (vertices, normals)
}
