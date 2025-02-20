use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};


#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct AVec3(pub Vec3);

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct AMat4(pub Mat4);

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct AVec4(pub Vec4);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Af32(pub f32);

#[repr(C)]
#[repr(align(8))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct AVec2(pub Vec2);

#[repr(C)]
#[repr(align(8))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct AUVec2(pub UVec2);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Au32(pub u32);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Abool(pub bool);

