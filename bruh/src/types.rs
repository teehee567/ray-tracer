use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use serde::Serialize;
use vulkanalia::vk::{self, HasBuilder};

use crate::{accelerators::{self, bvh::BvhNode, Primitive}, scene::TextureData};

#[repr(C)]
#[repr(align(64))]
#[derive(Copy, Clone, Debug)]
pub struct CameraBufferObject {
    pub resolution: AlignedUVec2,
    pub view_port_uv: AlignedVec2,
    pub focal_length: Alignedf32,
    pub focus_distance: Alignedf32,
    pub aperture_radius: Alignedf32,
    pub time: Alignedu32,
    pub location: AlignedVec3,
    pub rotation: AlignedMat4,
}

impl Default for CameraBufferObject {
    fn default() -> Self {
        let mut ubo = Self {
            focal_length: Alignedf32(1.),
            focus_distance: Alignedf32(1.),
            aperture_radius: Alignedf32(0.),
            location: AlignedVec3::new(1., 0.6, 2.),
            resolution: AlignedUVec2(UVec2::new(1920, 1080)),
            view_port_uv: AlignedVec2::default(),
            time: Alignedu32::default(),
            rotation: AlignedMat4::default(),
        };
        let resolution = UVec2::new(1920, 1080);
        let rotation = Vec3::new(0., 210., 0.);

        ubo.rotation = AlignedMat4(
            Mat4::from_rotation_x(rotation[0].to_radians())
                * Mat4::from_rotation_y(rotation[1].to_radians())
                * Mat4::from_rotation_z(rotation[2].to_radians()),
        );

        let ratio = resolution[0] as f32 / resolution[1] as f32;
        let (u, v) = if ratio > 1.0 { (ratio, 1.0) } else { (1.0, 1.0 / ratio) };
        ubo.view_port_uv = AlignedVec2(Vec2::new(u, v));
        ubo.resolution = AlignedUVec2(resolution);
        ubo
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Material {
    pub base_colour: AlignedVec3,
    pub emission: AlignedVec3,
    pub metallic: Alignedf32,
    pub roughness: Alignedf32,
    pub ior: Alignedf32,
    pub transmission: Alignedf32,
    // Specular extension
    pub specular: Alignedf32,
    pub specular_color: AlignedVec3,
    // Clearcoat extension
    pub clearcoat: Alignedf32,
    pub clearcoat_roughness: Alignedf32,
    pub shade_smooth: Alignedu32,
    // textures
    pub base_color_tex: Alignedu32,
    pub metallic_roughness_tex: Alignedu32,
    pub normal_tex: Alignedu32,
    pub emission_tex: Alignedu32,
    pub specular_tex: Alignedu32,
    pub clearcoat_tex: Alignedu32,
}
impl Default for Material {
    fn default() -> Self {
        Self {
            base_colour: AlignedVec3(Vec3::ONE),
            emission:  AlignedVec3(Vec3::ZERO),
            metallic: Alignedf32(0.),
            roughness: Alignedf32(0.),
            ior: Alignedf32(1.5),
            transmission: Alignedf32(0.),

            specular: Alignedf32(0.),
            specular_color: AlignedVec3(Vec3::ONE),

            clearcoat: Alignedf32(0.),
            clearcoat_roughness: Alignedf32(0.),

            shade_smooth: Alignedu32(0),

            base_color_tex: Alignedu32(u32::MAX),
            metallic_roughness_tex: Alignedu32(u32::MAX),
            normal_tex: Alignedu32(u32::MAX),
            emission_tex: Alignedu32(u32::MAX),
            specular_tex: Alignedu32(u32::MAX),
            clearcoat_tex: Alignedu32(u32::MAX),
        }

    }
}


#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default)]
pub struct Triangle {
    pub material_index: Alignedu32,
    pub is_sphere: Alignedu32,
    pub vertices: [AlignedVec3; 3],
    pub normals: [AlignedVec3; 3],
    pub uvs: [AlignedVec2; 3],
}

#[repr(C)]
#[derive(Clone, Default)]
pub struct SceneComponents {
    pub camera: CameraBufferObject,
    pub bvh: Vec<BvhNode>,
    pub materials: Vec<Material>,
    pub triangles: Vec<Triangle>,
    pub textures: Vec<TextureData>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Sphere {
    center: AlignedVec3,
    radius: Alignedf32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Mesh {
    pub triangle_count: u32,
    pub offset: u32,
    pub material: Material
}


impl Primitive for Triangle {
    fn centroid(&self) -> Vec3 {
        (self.vertices[0].0 + self.vertices[1].0 + self.vertices[2].0) * 0.3333333f32
    }

    fn aabb(&self) -> accelerators::aabb::AABB {
        accelerators::aabb::AABB::new(self.vertices[0].0, self.vertices[1].0).grow(self.vertices[2].0)
    }
}

impl Triangle {
    pub fn min_bound(&self) -> Vec3 {
        if self.is_sphere.0 == 1 {
            let radius = self.vertices[1].0.x;
            self.vertices[0].0 - Vec3::splat(radius)
        } else {
            Vec3 {
                x: self.vertices.iter().map(|v| v.0.x).fold(f32::INFINITY, f32::min),
                y: self.vertices.iter().map(|v| v.0.y).fold(f32::INFINITY, f32::min),
                z: self.vertices.iter().map(|v| v.0.z).fold(f32::INFINITY, f32::min),
            }
        }
    }

    pub fn max_bound(&self) -> Vec3 {
        if self.is_sphere.0 == 1 {
            let radius = self.vertices[1].0.x;
            self.vertices[0].0 + Vec3::splat(radius)
        } else {
            Vec3 {
                x: self.vertices.iter().map(|v| v.0.x).fold(f32::NEG_INFINITY, f32::max),
                y: self.vertices.iter().map(|v| v.0.y).fold(f32::NEG_INFINITY, f32::max),
                z: self.vertices.iter().map(|v| v.0.z).fold(f32::NEG_INFINITY, f32::max),
            }
        }
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec2,
    pub color: Vec3,
}

impl Vertex {
    const fn new(pos: Vec2, color: Vec3) -> Self {
        Self { pos, color }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec2>() as u32)
            .build();
        [pos, color]
    }
}

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default)]
pub struct AlignedVec3(pub Vec3);
impl AlignedVec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }
}

// impl Serialize for AlignedVec3 {
//     fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer {
//         let mut state = serializer.serialize_struct("Vec3", 3)?;
//         state.serialize_field("x", &self.0.x)?;
//         state.serialize_field("y", &self.0.y)?;
//         state.serialize_field("z", &self.0.z)?;
//         state.end()
//     }
// }

impl From<[f32; 3]> for AlignedVec3 {
    fn from(value: [f32; 3]) -> Self {
        AlignedVec3::new(value[0], value[1], value[2])
    }
}

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default)]
pub struct AlignedMat4(pub Mat4);

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default)]
pub struct AlignedVec4(pub Vec4);
impl AlignedVec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(Vec4::new(x, y, z, w))
    }

    pub fn default() -> Self {
        Self(Vec4::new(0., 0., 0., 0.))
    }
}

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default, Serialize)]
pub struct Alignedf32(pub f32);

#[repr(C)]
#[repr(align(8))]
#[derive(Copy, Clone, Debug, Default)]
pub struct AlignedVec2(pub Vec2);

#[repr(C)]
#[repr(align(8))]
#[derive(Copy, Clone, Debug, Default)]
pub struct AlignedUVec2(pub UVec2);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default, Serialize)]
pub struct Alignedu32(pub u32);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default)]
pub struct AlignedBool(pub bool);

