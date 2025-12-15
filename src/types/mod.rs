use crate::{accelerators::bvh::BvhNode, scene::TextureData};

mod camera_buffer_obj;
pub use camera_buffer_obj::*;

mod aligned;
pub use aligned::*;

mod material;
pub use material::*;

mod triangle;
pub use triangle::*;

mod light;
pub use light::*;

#[derive(Clone, Default)]
pub struct SceneComponents {
    pub camera: CameraBufferObject,
    pub bvh: Vec<BvhNode>,
    pub materials: Vec<Material>,
    pub triangles: Vec<Triangle>,
    pub textures: Vec<TextureData>,
    pub skybox: TextureData,
    pub lights: Vec<Light>,
    pub mesh_light_sampler: MeshLightSampler,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Sphere {
    center: AVec3,
    radius: Af32,
}
