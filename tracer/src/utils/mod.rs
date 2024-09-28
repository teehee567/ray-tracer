pub mod colour;

use std::f32::consts::PI;

use nalgebra::Vector3;

#[inline]
pub fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * PI / 180.
}

#[inline]
pub fn random_f32() -> f32 {
    fastrand::f32()
}

#[inline]
pub fn random_f32_in(min: f32, max: f32) -> f32 {
    min + (max - min) * random_f32()
}

#[inline]
pub fn rand_vec() -> Vector3<f32> {
    Vector3::new(random_f32(), random_f32(), random_f32())
}

#[inline]
pub fn rand_vec_in(min: f32, max: f32) -> Vector3<f32> {
    Vector3::new(
        random_f32_in(min, max),
        random_f32_in(min, max),
        random_f32_in(min, max),
    )
}

#[inline]
pub fn random_in_unit_disk() -> Vector3<f32> {
    loop {
        let p = Vector3::new(random_f32() - 0.5, random_f32() - 0.5, 0.0) * 2.0;
        if p.dot(&p) < 1.0 {
            return p;
        }
    }
}

#[inline]
pub fn random_in_unit_sphere() -> Vector3<f32> {
    loop {
        let p = rand_vec_in(-1., 1.);
        if (p.norm_squared() < 1.) {
            return p;
        }
    }
}

#[inline]
pub fn random_in_unit_vector() -> Vector3<f32> {
    random_in_unit_sphere().normalize()
}
