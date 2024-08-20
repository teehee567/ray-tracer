use std::f64::consts::PI;

#[inline]
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * PI / 180.
}

#[inline]
pub fn random_f64() -> f64 {
    fastrand::f64()    
}

#[inline]
pub fn random_f64_in(min: f64, max: f64) -> f64 {
    min + (max - min) * random_f64()
}
