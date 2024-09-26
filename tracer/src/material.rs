use std::sync::Arc;

use nalgebra::{Point3, Vector3};

use crate::{
    colour::Colour,
    hittable::HitRecord,
    ray::Ray,
    texture::{SolidColour, Texture},
    utils,
};

pub trait Material: Send + Sync {
    fn scatter(
        &self,
        ray_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Colour,
        scattered: &mut Ray,
    ) -> bool {
        return false;
    }

    fn emitted(&self, u: f32, v: f32, p: &Point3<f32>) -> Colour {
        return Colour::new(0., 0., 0.);
    }
}

fn reflect(v: &Vector3<f32>, n: &Vector3<f32>) -> Vector3<f32> {
    v - 2.0 * v.dot(&n) * n
}

// fn refract(v: &Vector3<f32>, n: &Vector3<f32>, ni_over_nt: f32) -> Vector3<f32> {
//     let uv = v.normalize();
//     let dt = uv.dot(&n);
//     let discriminant = 1.0 - ni_over_nt.powi(2) * (1.0 - dt.powi(2));
//     ni_over_nt * (uv - n * dt) - n * discriminant.sqrt()
// }

pub fn refract(uv: &Vector3<f32>, n: &Vector3<f32>, etai_over_etat: f32) -> Vector3<f32> {
    let cos_theta = Vector3::dot(&-uv, n).min(1.);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
    let r_out_parrallel = -(1. - r_out_perp.norm_squared()).abs().sqrt() * *n;

    return r_out_perp + r_out_parrallel;
}

pub struct Lambertian {
    texture: Arc<dyn Texture>,
}

impl Lambertian {
    pub fn new(albedo: &Colour) -> Self {
        Self {
            texture: Arc::new(SolidColour::new(*albedo)),
        }
    }

    pub fn new_tex(texture: Arc<dyn Texture>) -> Self {
        Self { texture }
    }
}

impl Material for Lambertian {
    fn scatter(
        &self,
        ray_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Colour,
        scattered: &mut Ray,
    ) -> bool {
        let mut scatter_direction = rec.normal + utils::random_in_unit_vector();

        // if (scatter_direction.near_zero()) {
        //     scatter_direction = rec.normal;
        // }

        // println!("{}, {}", rec.u, rec.v);
        *scattered = Ray::new_tm(rec.p, scatter_direction, ray_in.time());
        *attenuation = self.texture.value(rec.u, rec.v, &rec.p);

        return true;
    }
}

pub struct Metal {
    albedo: Colour,
    fuzz: f32,
}

impl Metal {
    pub fn new(albedo: &Colour, fuzz: f32) -> Self {
        Self {
            albedo: *albedo,
            fuzz,
        }
    }
}

impl Material for Metal {
    fn scatter(
        &self,
        ray_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Colour,
        scattered: &mut Ray,
    ) -> bool {
        let mut reflected = reflect(
            &ray_in.direction(),
            &rec.normal,
        );
        reflected = reflected.normalize() + (self.fuzz * utils::random_in_unit_vector());
        *scattered = Ray::new_tm(rec.p, reflected, ray_in.time());
        *attenuation = self.albedo;

        return (Vector3::dot(
            scattered.direction(),
            &rec.normal,
        ) > 0.);
    }
}

pub struct Dielectric {
    refraction_index: f32,
}

impl Dielectric {
    pub fn new(refraction_index: f32) -> Self {
        Self { refraction_index }
    }

    fn reflectance(cosine: f32, refraction_index: f32) -> f32 {
        let mut r0 = (1. - refraction_index) / (1. + refraction_index);
        r0 = r0 * r0;
        r0 + (1. - r0) * (1. - cosine).powi(5)
    }
}

impl Material for Dielectric {
    fn scatter(
        &self,
        ray_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Colour,
        scattered: &mut Ray,
    ) -> bool {
        *attenuation = Colour::new(1., 1., 1.);
        let ri: f32 = if (rec.front_face) {
            1. / self.refraction_index
        } else {
            self.refraction_index
        };

        let unit_direction = ray_in.direction().normalize();
        let cos_theta = Vector3::dot(&-unit_direction, &rec.normal).min(1.);
        let sin_theta = (1. - cos_theta * cos_theta).sqrt();

        let cannot_refract = ri * sin_theta > 1.;
        let direction = if (cannot_refract) {
            reflect(&unit_direction, &rec.normal)
        } else {
            refract(&unit_direction, &rec.normal, ri)
        };

        *scattered = Ray::new_tm(rec.p, direction, ray_in.time());

        return true;
    }
}

pub struct DiffuseLight {
    texture: Arc<dyn Texture>,
    intensity: f32,
}

impl DiffuseLight {
    pub fn new(texture: Arc<dyn Texture>, intensity: f32) -> Self {
        Self { texture , intensity}
    }

    pub fn new_colour(colour: Colour, intensity: f32) -> Self {
        let texture = Arc::new(SolidColour::new(colour));
        Self { texture , intensity }
    }
}

impl Material for DiffuseLight {
    fn emitted(&self, u: f32, v: f32, p: &Point3<f32>) -> Colour {
        return self.texture.value(u, v, &p) * self.intensity;
    }
}
