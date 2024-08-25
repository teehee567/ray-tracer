use std::{pin::pin, sync::Arc};

use image::{GenericImageView, ImageReader, RgbaImage};

use crate::{
    colour::Colour, perlin::Perlin, vec3::{Point3, Vec3}
};

pub trait Texture: Send + Sync {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Colour;
}

pub struct SolidColour {
    albedo: Colour,
}

impl SolidColour {
    pub fn new(albedo: Colour) -> Self {
        Self { albedo }
    }

    pub fn none() -> Self {
        Self {
            albedo: Colour::none(),
        }
    }

    pub fn new_colour(red: f64, green: f64, blue: f64) -> Self {
        Self {
            albedo: Colour::new(red, green, blue),
        }
    }
}

impl Texture for SolidColour {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Colour {
        return self.albedo;
    }
}

pub struct CheckerTexture {
    inv_scale: f64,
    even: Arc<dyn Texture>,
    odd: Arc<dyn Texture>,
}

impl CheckerTexture {
    pub fn new(scale: f64, even: Arc<dyn Texture>, odd: Arc<dyn Texture>) -> Self {
        Self {
            inv_scale: 1. / scale,
            even,
            odd,
        }
    }

    pub fn new_colour(scale: f64, c1: &Colour, c2: &Colour) -> Self {
        Self::new(
            scale,
            Arc::new(SolidColour::new(*c1)),
            Arc::new(SolidColour::new(*c2)),
        )
    }
}

impl Texture for CheckerTexture {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Colour {
        let x = (self.inv_scale * p.x()).floor() as i32;
        let y = (self.inv_scale * p.y()).floor() as i32;
        let z = (self.inv_scale * p.z()).floor() as i32;

        if (x + y + z) % 2 == 0 {
            self.even.value(u, v, p)
        } else {
            self.odd.value(u, v, p)
        }
    }
}

pub struct ImageTexture {
    data: RgbaImage,
}

impl ImageTexture {
    pub fn new(data: RgbaImage) -> Self {
        Self { data }
    }

    pub fn new_from(path: impl Into<String>) -> Self {
        let data = ImageReader::open(path.into())
            .unwrap()
            .decode()
            .map(|x| x.to_rgba8())
            .unwrap();

        Self { data }
    }
}

impl Texture for ImageTexture {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Colour {
        let u = u.clamp(0., 1.);
        let v = 1. - v.clamp(0., 1.);

        let (i, j) = {
            let mut i = (u * self.data.width() as f64) as u32;
            let mut j = (v * self.data.height() as f64) as u32;

            // Clamp integer mapping. The actual coordinates should be < 1.0
            if i >= self.data.width() {
                i = self.data.width() - 1;
            }
            if j >= self.data.height() {
                j = self.data.height() - 1;
            }

            (i, j)
        };

        let color_scale = 1.0 / 255.0;
        let pixel = self.data.get_pixel(i, j).0;
        // println!("{}, {}: {}, {}", i, j, u, v);

        Colour::new(
            color_scale * pixel[0] as f64,
            color_scale * pixel[1] as f64,
            color_scale * pixel[2] as f64,
        )
    }
}

pub struct NoiseTexture {
    noise: Perlin,
    scale: f64,
}

impl NoiseTexture {
    pub fn new(scale: f64) -> Self {
        Self {
            noise: Perlin::new(),
            scale,
        }
    }
}

impl Texture for NoiseTexture {
    fn value(&self, u: f64, v: f64, p: &Point3) -> Colour {
        Colour::new(0.5, 0.5, 0.5) * (1. + (self.scale * p.z() + 10. * self.noise.turbulence(*p, 7)).sin())
    }
}
