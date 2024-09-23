#![allow(unused)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(portable_simd)]
use std::f32::INFINITY;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fs::File, sync::Arc};

use bvh::BVH;
use camera::Camera;
use cube::Cube;
use hittable::{HitRecord, Hittable};
use hittable_list::HittableList;
use image::{Rgb, RgbImage};
use indicatif::ProgressBar;
use interval::Interval;
use material::{Dielectric, DiffuseLight, Lambertian, Material, Metal};
use nalgebra::{Point3, Vector3};
use quad::Quad;
use rayon::prelude::*;

use colour::Colour;
use ray::Ray;
use sphere::Sphere;
use texture::{CheckerTexture, ImageTexture, NoiseTexture, SolidColour};
use utils::{random_f32, random_f32_in};

pub mod aabb;
pub mod bvh;
pub mod camera;
pub mod colour;
pub mod cube;
pub mod hittable;
pub mod hittable_list;
pub mod interval;
pub mod material;
pub mod perlin;
pub mod quad;
pub mod ray;
pub mod sphere;
pub mod texture;
pub mod utils;
pub mod translate;
pub mod mesh;

fn balls() -> HittableList {
    // let material_ground = Arc::new(Lambertian::new(&Colour::new(0.8, 0.8, 0.)));
    // let material_center = Arc::new(Lambertian::new(&Colour::new(0.1, 0.2, 0.5)));
    // let material_left = Arc::new(Dielectric::new(1.5));
    // let material_bubble = Arc::new(Dielectric::new(1./1.5));
    // let material_right = Arc::new(Metal::new(&Colour::new(0.8, 0.6, 0.2), 1.));
    //
    // // World
    // world.add(Sphere::new(Vec3::new(0., -100.5, -1.), 100., material_ground.clone()));
    // world.add(Sphere::new(Vec3::new(0., 0., -1.2), 0.5, material_center.clone()));
    // world.add(Sphere::new(Vec3::new(-1., 0., -1.), 0.5, material_left.clone()));
    // world.add(Sphere::new(Vec3::new(-1., 0., -1.), 0.4, material_bubble.clone()));
    // world.add(Sphere::new(Vec3::new(1., 0., -1.), 0.5, material_right.clone()));
    //
    // let mut camera = Camera::new();
    // camera.aspect_ratio = 16. / 9.;
    // camera.image_width = 3840;
    // camera.samples_per_pixel = 1000;
    // camera.max_depth = 20;
    //
    // camera.vfov = 90;
    // camera.lookfrom = Point3::new(-2., 2., 1.);
    // camera.lookat = Point3::new(0., 0., -1.);
    // camera.vup = Vec3::new(0., 1., 0.);
    //
    // camera.defocus_angle = 10.;
    // camera.focus_dist = 3.4;
    todo!()
}

fn bouncing_spheres() -> (Camera, HittableList) {
    let checker = Arc::new(CheckerTexture::new_colour(
        0.32,
        &Colour::new(0.2, 0.3, 0.1),
        &Colour::new(0.9, 0.9, 0.9),
    ));

    let mut world = HittableList::none();
    let material_ground = Lambertian::new_tex(checker.clone());
    world.add(Sphere::new(
        nalgebra::Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Arc::new(material_ground),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_f32();
            let center = Point3::new(
                a as f32 + 0.9 * random_f32(),
                0.2,
                b as f32 + 0.9 * random_f32(),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).norm() > 0.9 {
                if choose_mat < 0.8 {
                    // Lambertian
                    let albedo = utils::rand_vec().component_mul(&utils::rand_vec());
                    let material = Arc::new(Lambertian::new(&albedo));
                    let center2 = center + Vector3::new(0., random_f32_in(0., 0.5), 0.);
                    world.add(Sphere::new_mov(center, center2, 0.2, material));
                } else if choose_mat < 0.95 {
                    // Metal
                    let albedo = utils::rand_vec_in(0.5, 1.0);
                    let fuzz = random_f32() * 0.5;
                    let material = Arc::new(Metal::new(&albedo, fuzz));
                    world.add(Sphere::new(center, 0.2, material));
                } else {
                    // Glass
                    let material = Arc::new(Dielectric::new(1.5));
                    world.add(Sphere::new(center, 0.2, material));
                };
            }
        }
    }

    let material1 = Arc::new(Dielectric::new(1.5));
    world.add(Sphere::new(Point3::new(0., 1., 0.), 1., material1));

    let material2 = Arc::new(Lambertian::new(&Colour::new(0.4, 0.2, 0.1)));
    world.add(Sphere::new(Point3::new(-4., 1., 0.), 1., material2));

    let material3 = Arc::new(Metal::new(&Colour::new(0.7, 0.6, 0.5), 0.));
    world.add(Sphere::new(Point3::new(4., 1., 0.), 1., material3));

    world = HittableList::new(BVH::new(world.objects, 0., 1.));

    let mut camera = Camera::new();

    camera.aspect_ratio = 16. / 9.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 30;
    camera.max_depth = 50;
    camera.background = Colour::new(0.7, 0.8, 1.);

    camera.vfov = 20;
    camera.lookfrom = Point3::new(13., 2., 3.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vector3::new(0., 1., 0.);

    camera.defocus_angle = 0.6;
    camera.focus_dist = 10.;

    (camera, world)
}

fn earht() -> (Camera, HittableList) {
    let mut world = HittableList::none();
    let earth_texture = Arc::new(ImageTexture::new_from("earthmap.png"));
    let earth_surface = Arc::new(Lambertian::new_tex(earth_texture.clone()));
    let globe = Sphere::new(Point3::new(0., 0., 0.), 2., earth_surface.clone());
    world.add(globe);

    let mut camera = Camera::new();

    camera.aspect_ratio = 16. / 9.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.background = Colour::new(0.7, 0.8, 1.);

    camera.vfov = 20;
    camera.lookfrom = Point3::new(0., 0., -12.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vector3::new(0., 1., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 12.;

    (camera, world)
}

fn checkerd_sphered() -> (Camera, HittableList) {
    let mut world = HittableList::none();
    let checker = Arc::new(CheckerTexture::new_colour(
        0.32,
        &Colour::new(0.2, 0.3, 0.1),
        &Colour::new(0.9, 0.9, 0.9),
    ));

    world.add(Sphere::new(
        Point3::new(0., -10., 0.),
        10.,
        Arc::new(Lambertian::new_tex(checker.clone())),
    ));
    world.add(Sphere::new(
        Point3::new(0., 10., 0.),
        10.,
        Arc::new(Lambertian::new_tex(checker.clone())),
    ));

    let mut camera = Camera::new();

    camera.aspect_ratio = 16. / 9.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 10;
    camera.max_depth = 50;
    camera.background = Colour::new(0.7, 0.8, 1.);

    camera.vfov = 90;
    camera.lookfrom = Point3::new(13., 2., 3.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vector3::new(0., 1., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 10.;

    (camera, world)
}

fn perlin_spheres() -> (Camera, HittableList) {
    let mut world = HittableList::none();

    let pertext = Arc::new(NoiseTexture::new(4.));
    world.add(Sphere::new(
        Point3::new(0., -1000., 0.),
        1000.,
        Arc::new(Lambertian::new_tex(pertext.clone())),
    ));
    world.add(Sphere::new(
        Point3::new(0., 2., 0.),
        2.,
        Arc::new(Lambertian::new_tex(pertext.clone())),
    ));

    let mut camera = Camera::new();

    camera.aspect_ratio = 16. / 9.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 10;
    camera.max_depth = 50;
    camera.background = Colour::new(0.7, 0.8, 1.);

    camera.vfov = 20;
    camera.lookfrom = Point3::new(13., 2., 3.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vector3::new(0., 1., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 10.;

    (camera, world)
}

fn quads() -> (Camera, HittableList) {
    let mut world = HittableList::none();
    let left_red = Arc::new(Lambertian::new(&Colour::new(1., 0.2, 0.2)));
    let back_green = Arc::new(Lambertian::new(&Colour::new(0.2, 1., 0.2)));
    let right_blue = Arc::new(Lambertian::new(&Colour::new(0.2, 0.2, 1.)));
    let upper_orange = Arc::new(Lambertian::new(&Colour::new(1., 0.5, 0.)));
    let lower_teal = Arc::new(Lambertian::new(&Colour::new(0.2, 0.8, 0.8)));

    world.add(Quad::new(
        Point3::new(-3., -2., 5.),
        Vector3::new(0., 0., -4.),
        Vector3::new(0., 4., 0.),
        left_red,
    ));
    world.add(Quad::new(
        Point3::new(-2., -2., 0.),
        Vector3::new(4., 0., 0.),
        Vector3::new(0., 4., 0.),
        back_green,
    ));
    world.add(Quad::new(
        Point3::new(3., -2., 1.),
        Vector3::new(0., 0., 4.),
        Vector3::new(0., 4., 0.),
        right_blue,
    ));
    world.add(Quad::new(
        Point3::new(-2., 3., 1.),
        Vector3::new(4., 0., 0.),
        Vector3::new(0., 0., 4.),
        upper_orange,
    ));
    world.add(Quad::new(
        Point3::new(-2., -3., 5.),
        Vector3::new(4., 0., 0.),
        Vector3::new(0., 0., -4.),
        lower_teal,
    ));

    let mut camera = Camera::new();

    camera.aspect_ratio = 16. / 9.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;
    camera.background = Colour::new(0.7, 0.8, 1.);

    camera.vfov = 50;
    camera.lookfrom = Point3::new(0., 0., 9.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vector3::new(0., 1., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 10.;

    (camera, world)
}

fn simple_light() -> (Camera, HittableList) {
    let mut world = HittableList::none();

    let pertext = Arc::new(NoiseTexture::new(4.));
    world.add(Sphere::new(
        Point3::new(0., -1000., 0.),
        1000.,
        Arc::new(Lambertian::new_tex(pertext.clone())),
    ));
    world.add(Sphere::new(
        Point3::new(0., 2., 0.),
        2.,
        Arc::new(Lambertian::new_tex(pertext.clone())),
    ));

    let difflight = Arc::new(DiffuseLight::new_colour(Colour::new(4., 4., 4.)));
    world.add(Quad::new(
        Point3::new(3., 1., -2.),
        Vector3::new(2., 0., 0.),
        Vector3::new(0., 2., 0.),
        difflight.clone(),
    ));

    let mut camera = Camera::new();

    camera.aspect_ratio = 16. / 9.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;

    camera.vfov = 20;
    camera.lookfrom = Point3::new(26., 3., 6.);
    camera.lookat = Point3::new(0., 2., 0.);
    camera.vup = Vector3::new(0., 1., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 20.;

    (camera, world)
}

fn cornell_box() -> (Camera, HittableList) {
    let mut world = HittableList::none();

    let red = Arc::new(Lambertian::new(&Colour::new(0.65, 0.05, 0.05)));
    let white = Arc::new(Lambertian::new(&Colour::new(0.73, 0.73, 0.73)));
    let green = Arc::new(Lambertian::new(&Colour::new(0.12, 0.45, 0.15)));
    let light = Arc::new(DiffuseLight::new_colour(Colour::new(15., 15., 15.)));

    world.add(Quad::new(
        Point3::new(555., 0., 0.),
        Vector3::new(0., 555., 0.),
        Vector3::new(0., 0., 555.),
        green,
    ));
    world.add(Quad::new(
        Point3::new(0., 0., 0.),
        Vector3::new(0., 555., 0.),
        Vector3::new(0., 0., 555.),
        red,
    ));
    world.add(Quad::new(
        Point3::new(343., 554., 332.),
        Vector3::new(-130., 0., 0.),
        Vector3::new(0., 0., -105.),
        light,
    ));
    world.add(Quad::new(
        Point3::new(0., 0., 0.),
        Vector3::new(555., 0., 0.),
        Vector3::new(0., 0., 555.),
        white.clone(),
    ));
    world.add(Quad::new(
        Point3::new(555., 555., 555.),
        Vector3::new(-555., 0., 0.),
        Vector3::new(0., 0., -555.),
        white.clone(),
    ));
    world.add(Quad::new(
        Point3::new(0., 0., 555.),
        Vector3::new(555., 0., 0.),
        Vector3::new(0., 555., 0.),
        white.clone(),
    ));

    world.add(Cube::new(
        Point3::new(130., 0., 65.),
        Point3::new(295., 165., 230.),
        white.clone(),
    ));
    world.add(Cube::new(
        Point3::new(265., 0., 295.),
        Point3::new(430., 330., 460.),
        white.clone(),
    ));

    world = HittableList::new(BVH::new(world.objects, 0., 1.));

    let mut camera = Camera::new();

    camera.aspect_ratio = 1.;
    camera.image_width = 840;
    camera.samples_per_pixel = 4;
    camera.max_depth = 10;
    camera.background = Vector3::new(1., 1., 1.);

    camera.vfov = 40;
    camera.lookfrom = Point3::new(278., 278., -400.);
    camera.lookat = Point3::new(278., 278., 0.);
    camera.vup = Vector3::new(0., 1., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 200.;

    (camera, world)
}

fn huh() -> (Camera, HittableList) {
    let mut world = HittableList::none();

    world.add(Sphere::new(
        nalgebra::Point3::new(0., 0., -5.),
        2.,
        Arc::new(DiffuseLight::new(Arc::new(SolidColour::new(Colour::new(0.7, 0.4, 0.2))))),
    ));

    let mut camera = Camera::new();

    camera.aspect_ratio = 1.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 20;
    camera.max_depth = 50;

    camera.vfov = 90;
    camera.lookfrom = Point3::new(0., 0., 1.);
    camera.lookat = Point3::new(0., 0., -5.);
    camera.vup = Vector3::new(0., 1., 0.);
    camera.background = Colour::new(0.5, 0., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 5.;

    (camera, world)
}

fn main() {
    let (mut camera, world) = match 7 {
        1 => bouncing_spheres(),
        2 => checkerd_sphered(),
        3 => earht(),
        4 => perlin_spheres(),
        5 => quads(),
        6 => simple_light(),
        7 => cornell_box(),
        8 => huh(),
        _ => panic!("bro...."),
    };
    camera.render(&world);
}
