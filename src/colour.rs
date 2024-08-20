use image::Rgb;

use crate::{interval::Interval, vec3::Vec3};


pub type Colour = Vec3; 

impl Colour {
    #[inline]
    pub fn linear_to_gamma(linear_component: f64) -> f64 {
        if (linear_component > 0.) {
            return linear_component.sqrt();
        }

        return 0.;
    }

    pub fn convert_colour(&self) -> Rgb<u8> {
        let mut r = self.x();
        let mut g = self.y();
        let mut b = self.z();

        // gamma create for gamma 2
        r = Self::linear_to_gamma(r);
        g = Self::linear_to_gamma(g);
        b = Self::linear_to_gamma(b);

        let intensity = Interval::new(0., 0.999);
        let ir = (255.999 * intensity.clamp(r)) as u8;
        let ig = (255.999 * intensity.clamp(g)) as u8;
        let ib = (255.999 * intensity.clamp(b)) as u8;

        Rgb([ir, ig, ib])
    }
}
