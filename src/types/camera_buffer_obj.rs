use std::fmt;

use glam::{Mat4, UVec2, Vec2, Vec3};
use serde::{
    Deserialize, Deserializer,
    de::{self, MapAccess, Visitor},
};

use super::{AMat4, AUVec2, AVec2, AVec3, Af32, Au32};

#[repr(C)]
#[repr(align(64))]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CameraBufferObject {
    pub resolution: AUVec2,
    pub view_port_uv: AVec2,
    pub focal_length: Af32,
    pub focus_distance: Af32,
    pub aperture_radius: Af32,
    pub time: Au32,
    pub location: AVec3,
    pub rotation: AMat4,
}

impl Default for CameraBufferObject {
    fn default() -> Self {
        let mut ubo = Self {
            focal_length: Af32(1.),
            focus_distance: Af32(1.),
            aperture_radius: Af32(0.),
            location: AVec3(Vec3::new(1., 0.6, 2.)),
            resolution: AUVec2(UVec2::new(1920, 1080)),
            view_port_uv: AVec2::default(),
            time: Au32::default(),
            rotation: AMat4::default(),
        };
        let resolution = UVec2::new(1920, 1080);
        let rotation = Vec3::new(0., 210., 0.);

        ubo.rotation = AMat4(
            Mat4::from_rotation_x(rotation[0].to_radians())
                * Mat4::from_rotation_y(rotation[1].to_radians())
                * Mat4::from_rotation_z(rotation[2].to_radians()),
        );

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
}

impl<'de> Deserialize<'de> for CameraBufferObject {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Resolution,
            FocalLength,
            FocusDistance,
            ApertureRadius,
            Location,
            LookAt,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "resolution" => Ok(Field::Resolution),
                            "focal_length" => Ok(Field::FocalLength),
                            "focus_distance" => Ok(Field::FocusDistance),
                            "aperture_radius" => Ok(Field::ApertureRadius),
                            "location" => Ok(Field::Location),
                            "look_at" => Ok(Field::LookAt),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct CameraBufferObjectVisitor;

        const FIELDS: &'static [&'static str] = &[
            "resolution",
            "focal_length",
            "focus_distance",
            "aperture_radius",
            "location",
            "look_at",
        ];

        impl<'de> Visitor<'de> for CameraBufferObjectVisitor {
            type Value = CameraBufferObject;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct CameraBufferObject")
            }

            fn visit_map<V>(self, mut map: V) -> Result<CameraBufferObject, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut resolution: Option<UVec2> = None;
                let mut focal_length: Option<f32> = None;
                let mut focus_distance: Option<f32> = None;
                let mut aperture_radius: Option<f32> = None;
                let mut location: Option<Vec3> = None;
                let mut look_at: Option<Vec3> = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Resolution => resolution = Some(map.next_value()?),
                        Field::FocalLength => focal_length = Some(map.next_value()?),
                        Field::FocusDistance => focus_distance = Some(map.next_value()?),
                        Field::ApertureRadius => aperture_radius = Some(map.next_value()?),
                        Field::Location => location = Some(map.next_value()?),
                        Field::LookAt => look_at = Some(map.next_value()?),
                    }
                }

                let resolution =
                    resolution.ok_or_else(|| de::Error::missing_field("resolution"))?;
                let location = location.ok_or_else(|| de::Error::missing_field("location"))?;
                let look_at = look_at.ok_or_else(|| de::Error::missing_field("look_at"))?;
                let focal_length = focal_length.unwrap_or(1.0);
                let focus_distance = focus_distance.unwrap_or(1.0);
                let aperture_radius = aperture_radius.unwrap_or(0.0);

                let rotation = Mat4::look_at_rh(location, look_at, Vec3::Y).transpose();

                // Calculate view_port_uv
                let ratio = resolution.x as f32 / resolution.y as f32;
                let (u, v) = if ratio > 1.0 {
                    (ratio, 1.0)
                } else {
                    (1.0, 1.0 / ratio)
                };

                Ok(CameraBufferObject {
                    resolution: AUVec2(resolution),
                    view_port_uv: AVec2(Vec2::new(u, v)),
                    focal_length: Af32(focal_length),
                    focus_distance: Af32(focus_distance),
                    aperture_radius: Af32(aperture_radius),
                    time: Au32(0),
                    location: AVec3(location),
                    rotation: AMat4(rotation),
                })
            }
        }

        deserializer.deserialize_struct("CameraBufferObject", FIELDS, CameraBufferObjectVisitor)
    }
}
