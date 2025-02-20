use std::fs::File;

use crate::{AMat4, AUVec2, AVec2, AVec3, Af32, Au32, CameraBufferObject, Material, SceneComponents, Triangle};

use super::{Scene, CONFIG_VERSION};

use anyhow::{anyhow, bail, Result};
use glam::{Mat4, UVec2, Vec2, Vec3};
use serde_yaml::Value;

impl Scene {
    /// Creates a new scene from a file at `path`.
    pub fn from_yaml(path: &'static str) -> Result<Self> {
        let file = File::open(path)?;
        let root: Value = serde_yaml::from_reader(file)?;
        let mut scene = Scene {
            root,
            components: SceneComponents::default(),
        };

        // Validate the file and load its components.
        scene.load_camera_controls_yaml()?;
        scene.load_meshes()?;
        scene.build_bvh();
        println!("Triangles: {}", scene.components.triangles.len());
        Ok(scene)
    }

    /// Loads the camera controls.
    fn load_camera_controls_yaml(&mut self) -> Result<()> {
        let camera = self.root.get("camera").unwrap();

        let resolution: [u32; 2] =
            serde_yaml::from_value(camera.get("resolution").unwrap().clone())?;
        let focal_length: f32 =
            serde_yaml::from_value(camera.get("focal_length").unwrap().clone())?;
        let focus_distance: f32 =
            serde_yaml::from_value(camera.get("focus_distance").unwrap().clone())?;
        let aperture_radius: f32 =
            serde_yaml::from_value(camera.get("aperture_radius").unwrap().clone())?;
        let location: [f32; 3] = serde_yaml::from_value(camera.get("location").unwrap().clone())?;
        let look_at: [f32; 3] = serde_yaml::from_value(camera.get("look_at").unwrap().clone())?;

        let mut ubo = CameraBufferObject {
            resolution: AUVec2(UVec2::from(resolution)),
            focal_length: Af32(focal_length),
            focus_distance: Af32(focus_distance),
            aperture_radius: Af32(aperture_radius),
            location: AVec3(Vec3::from(location)),
            ..Default::default()
        };

        ubo.rotation = AMat4(Mat4::look_at_rh(ubo.location.0, Vec3::from(look_at), Vec3::Y).transpose());

        let ratio = resolution[0] as f32 / resolution[1] as f32;
        let (u, v) = if ratio > 1.0 {
            (ratio, 1.0)
        } else {
            (1.0, 1.0 / ratio)
        };
        ubo.view_port_uv = AVec2(Vec2::new(u, v));

        self.components.camera = ubo;
        Ok(())
    }

    /// Loads meshes from the YAML file.
    fn load_meshes(&mut self) -> Result<()> {
        let scene_nodes = self
            .root
            .get("scene")
            .and_then(|v| v.as_sequence())
            .ok_or_else(|| anyhow!("Scene field missing or not a sequence"))?
            .clone();
        for mesh in scene_nodes {
            let mesh_type = mesh
                .get("type")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("Mesh missing type field"))?;
            if mesh_type == "TriMesh" {
                self.load_tri_mesh(&mesh)?;
            } else if mesh_type == "Sphere" {
                self.load_sphere_yaml(&mesh)?;
            }
        }
        Ok(())
    }

    /// Loads a triangle mesh.
    fn load_tri_mesh(&mut self, mesh: &Value) -> Result<()> {
        let data = mesh
            .get("data")
            .ok_or_else(|| anyhow!("Mesh missing data field"))?;
        let verts: Vec<f32> = serde_yaml::from_value(
            data.get("vertices")
                .ok_or_else(|| anyhow!("Missing vertices"))?
                .clone(),
        )?;
        let norms: Vec<f32> = serde_yaml::from_value(
            data.get("normals")
                .ok_or_else(|| anyhow!("Missing normals"))?
                .clone(),
        )?;
        if verts.len() % 9 != 0 {
            bail!("Vertices length not a multiple of 9");
        }
        let triangle_amt = verts.len() / 9;

        // Append the material.
        let material_node = mesh
            .get("material")
            .ok_or_else(|| anyhow!("Missing material field"))?;
        let material = self.get_material(material_node)?;
        let material_index = self.components.materials.len() as u32;
        self.components.materials.push(material);

        // Append triangles.
        for i in 0..triangle_amt {
            let mut tri = Triangle {
                material_index: Au32(material_index),
                is_sphere: Au32(0),
                vertices: [
                    AVec3::default(),
                    AVec3::default(),
                    AVec3::default(),
                ],
                normals: [
                    AVec3::default(),
                    AVec3::default(),
                    AVec3::default(),
                ],
                ..Default::default()
            };
            for j in 0..3 {
                let off = i * 9 + j * 3;
                tri.vertices[j] = AVec3(Vec3::new(verts[off], verts[off + 1], verts[off + 2]));
                tri.normals[j] = AVec3(Vec3::new(norms[off], norms[off + 1], norms[off + 2]));
            }
            self.components.triangles.push(tri);
        }
        Ok(())
    }

    /// Loads a sphere.
    fn load_sphere_yaml(&mut self, sphere: &Value) -> Result<()> {
        // Append the material.
        let material_node = sphere
            .get("material")
            .ok_or_else(|| anyhow!("Missing material field"))?;
        let material = self.get_material(material_node)?;
        let material_index = self.components.materials.len() as u32;
        self.components.materials.push(material);

        let data = sphere
            .get("data")
            .ok_or_else(|| anyhow!("Missing data field"))?;
        let center: Vec<f32> = serde_yaml::from_value(
            data.get("center")
                .ok_or_else(|| anyhow!("Missing center field"))?
                .clone(),
        )?;
        if center.len() < 3 {
            bail!("Center does not have 3 components");
        }
        let radius = data
            .get("radius")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("Missing or invalid radius field"))? as f32;
        let mut tri = Triangle {
            material_index: Au32(material_index),
            is_sphere: Au32(1),
            vertices: [
                AVec3::default(),
                AVec3::default(),
                AVec3::default(),
            ],
            normals: [
                AVec3::default(),
                AVec3::default(),
                AVec3::default(),
            ],
            ..Default::default()
        };
        tri.vertices[0] = AVec3(Vec3::new(center[0], center[1], center[2]));
        // The sphere is represented as a triangle that “stores” the radius.
        tri.vertices[1] = AVec3(Vec3::new(radius, 0.0, 0.0));
        self.components.triangles.push(tri);
        Ok(())
    }

    /// Builds the BVH from the triangles.
    /// Constructs a Material from a YAML node.
    fn get_material(&self, node: &Value) -> Result<Material> {
        // Default color value.
        let def: [f32; 3] = [0.0, 0.0, 0.0];

        let base_colour: [f32; 3] = node
            .get("base_color")
            .and_then(|v| serde_yaml::from_value(v.clone()).ok())
            .unwrap_or(def);
        let emission: [f32; 3] = node
            .get("emission")
            .and_then(|v| serde_yaml::from_value(v.clone()).ok())
            .unwrap_or(def);
        let metallic = node
            .get("metallic")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let roughness = node
            .get("roughness")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let ior = node.get("ior").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        let transmission = node
            .get("transmission")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
        let motion_blur: [f32; 3] = node
            .get("motion_blur")
            .and_then(|v| serde_yaml::from_value(v.clone()).ok())
            .unwrap_or(def);
        let shade_smooth = node
            .get("smooth_shading")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(Material {
            base_colour: AVec3(base_colour.into()),
            emission: AVec3(emission.into()),
            metallic: Af32(metallic),
            roughness: Af32(roughness),
            ior: Af32(ior),
            spec_trans: Af32(transmission),
            shade_smooth: Au32(if shade_smooth { 1 } else { 0 }),
            ..Default::default()
        })
    }
}
