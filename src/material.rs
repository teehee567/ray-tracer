use std::sync::Arc;

use crate::{colour::Colour, hittable::HitRecord, ray::Ray, texture::{SolidColour, Texture}, vec3::{Point3, Vec3}};

pub trait Material: Send + Sync {
    fn scatter(&self, ray_in: &Ray, rec: &HitRecord, attenuation: &mut Colour, scattered: &mut Ray) -> bool {
        return false;
    }

    fn emitted(&self, u: f64, v: f64, p: &Point3) -> Colour {
        return Colour::new(0., 0., 0.);
    }
}

pub struct Lambertian {
    texture: Arc<dyn Texture>
}

impl Lambertian {
    pub fn new(albedo: &Colour) -> Self {
        Self { texture: Arc::new(SolidColour::new(*albedo)) }
    }

    pub fn new_tex(texture: Arc<dyn Texture>) -> Self {
        Self { texture }
    }
}

impl Material for Lambertian {
    fn scatter(&self, ray_in: &Ray, rec: &HitRecord, attenuation: &mut Colour, scattered: &mut Ray) -> bool {
        let mut scatter_direction = rec.normal + Vec3::random_in_unit_vector();

        if (scatter_direction.near_zero()) {
            scatter_direction = rec.normal;
        }

        // println!("{}, {}", rec.u, rec.v);
        *scattered = Ray::new_tm(rec.p, scatter_direction, ray_in.time());
        *attenuation = self.texture.value(rec.u, rec.v, &rec.p);

        return true;
    }
}

pub struct Metal {
    albedo: Colour,
    fuzz: f64
}

impl Metal {
    pub fn new(albedo: &Colour, fuzz: f64) -> Self {
        Self { albedo: *albedo, fuzz }
    }
}

impl Material for Metal {
    fn scatter(&self, ray_in: &Ray, rec: &HitRecord, attenuation: &mut Colour, scattered: &mut Ray) -> bool {
        let mut reflected = Vec3::reflect(ray_in.direction(), &rec.normal);
        reflected = Vec3::unit_vector(reflected) + (self.fuzz * Vec3::random_in_unit_vector());
        *scattered = Ray::new_tm(rec.p, reflected, ray_in.time());
        *attenuation = self.albedo;

        return (Vec3::dot(*scattered.direction(), rec.normal) > 0.)
    }
}

pub struct Dielectric {
    refraction_index: f64,
}

impl Dielectric {
    pub fn new(refraction_index: f64) -> Self {
        Self {
            refraction_index
        }
    }

    fn reflectance(cosine: f64, refraction_index: f64) -> f64 {
        let mut r0 = (1. - refraction_index) / (1. + refraction_index);
        r0 = r0 * r0;
        r0 + (1. - r0) * (1. - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(&self, ray_in: &Ray, rec: &HitRecord, attenuation: &mut Colour, scattered: &mut Ray) -> bool {
        *attenuation = Colour::new(1., 1., 1.);
        let ri: f64 = if (rec.front_face) {
            1./self.refraction_index
        } else {
            self.refraction_index
        };

        let unit_direction = Vec3::unit_vector(*ray_in.direction());
        let cos_theta = Vec3::dot(-unit_direction, rec.normal).min(1.);
        let sin_theta = (1. - cos_theta * cos_theta).sqrt();

        let cannot_refract = ri * sin_theta > 1.;
        let direction: Vec3 = if (cannot_refract) {
            Vec3::reflect(&unit_direction, &rec.normal)
        } else {
            Vec3::refract(&unit_direction, &rec.normal, ri)
        };

        *scattered = Ray::new_tm(rec.p, direction, ray_in.time());

        return true;
    }
}

pub struct DiffuseLight {
    texture: Arc<dyn Texture>
}

impl DiffuseLight {
    pub fn new(texture: Arc<dyn Texture>) -> Self {
        Self {
            texture
        }
    }

    pub fn new_colour(colour: Colour) -> Self {
        let texture = Arc::new(SolidColour::new(colour));
        Self {
            texture
        }
    }
}

impl Material for DiffuseLight {
    fn emitted(&self, u: f64, v: f64, p: &Point3) -> Colour {
        return self.texture.value(u, v, p);
    }
}
