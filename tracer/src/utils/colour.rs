use image::{Rgb, Rgba};
use nalgebra::{Vector3, Vector4};

use crate::core::interval::Interval;

pub type Colour = Vector4<f32>;

#[inline]
pub fn linear_to_gamma(linear_component: f32) -> f32 {
    if (linear_component > 0.) {
        return linear_component.sqrt();
    }

    0.
}

const INTENSITY_MIN: f32 = 0.;
const INTENSITY_MAX: f32 = 0.999;

#[inline]
pub fn convert_colour(colour: Colour) -> Rgba<u8> {
    let mut r = colour.x;
    let mut g = colour.y;
    let mut b = colour.z;

    // gamma create for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    let ir = (255.999 * r.clamp(INTENSITY_MIN, INTENSITY_MAX)) as u8;
    let ig = (255.999 * g.clamp(INTENSITY_MIN, INTENSITY_MAX)) as u8;
    let ib = (255.999 * b.clamp(INTENSITY_MIN, INTENSITY_MAX)) as u8;
    let ia = (255.999 * colour.w.clamp(0.0, 1.0)) as u8;

    Rgba([ir, ig, ib, ia])
}
