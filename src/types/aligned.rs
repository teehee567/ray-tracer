use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};


#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default, Serialize, PartialEq)]
pub struct AVec3(pub Vec3);

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AMat4(pub Mat4);

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct AVec4(pub Vec4);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default, Serialize, PartialEq)]
pub struct Af32(pub f32);

#[repr(C)]
#[repr(align(8))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AVec2(pub Vec2);

#[repr(C)]
#[repr(align(8))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct AUVec2(pub UVec2);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Au32(pub u32);

#[repr(C)]
#[repr(align(4))]
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Abool(pub bool);


impl<'de> Deserialize<'de> for AVec3 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let vec: Vec<u64> = Vec::deserialize(deserializer)?;
        Ok(AVec3(Vec3::new(
            vec[0] as f32 / 255.0,
            vec[1] as f32 / 255.0,
            vec[2] as f32 / 255.0,
        )))
    }
}

impl<'de> Deserialize<'de> for Af32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let val: f64 = f64::deserialize(deserializer)?;
        Ok(Af32(val as f32))
    }
}
