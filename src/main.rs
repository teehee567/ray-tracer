#![allow(unused)]
use std::f64::INFINITY;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fs::File, sync::Arc};

use bvh::BVH;
use camera::Camera;
use hittable::{HitRecord, Hittable};
use hittable_list::HittableList;
use image::{Rgb, RgbImage};
use indicatif::ProgressBar;
use interval::Interval;
use material::{Dielectric, Lambertian, Material, Metal};
use rayon::prelude::*;

use colour::Colour;
use ray::Ray;
use sphere::Sphere;
use texture::{CheckerTexture, ImageTexture, NoiseTexture};
use utils::{random_f64, random_f64_in};
use vec3::{Point3, Vec3};

mod aabb;
mod bvh;
mod camera;
mod colour;
mod hittable;
mod hittable_list;
mod interval;
mod material;
mod perlin;
mod ray;
mod sphere;
mod texture;
mod utils;
mod vec3;

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
        Point3::new(0.0, -1000.0, 0.0),
        1000.0,
        Arc::new(material_ground),
    ));

    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_f64();
            let center = Point3::new(
                a as f64 + 0.9 * random_f64(),
                0.2,
                b as f64 + 0.9 * random_f64(),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // Lambertian
                    let albedo = Colour::random() * Colour::random();
                    let material = Arc::new(Lambertian::new(&albedo));
                    let center2 = center + Vec3::new(0., random_f64_in(0., 0.5), 0.);
                    world.add(Sphere::new_mov(center, center2, 0.2, material));
                } else if choose_mat < 0.95 {
                    // Metal
                    let albedo = Colour::random_in(0.5, 1.0);
                    let fuzz = random_f64() * 0.5;
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

    camera.vfov = 20;
    camera.lookfrom = Point3::new(13., 2., 3.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vec3::new(0., 1., 0.);

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

    camera.vfov = 20;
    camera.lookfrom = Point3::new(0., 0., -12.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vec3::new(0., 1., 0.);

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

    camera.vfov = 90;
    camera.lookfrom = Point3::new(13., 2., 3.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vec3::new(0., 1., 0.);

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

    camera.vfov = 20;
    camera.lookfrom = Point3::new(13., 2., 3.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vec3::new(0., 1., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 10.;

    (camera, world)
}

fn main() {
    let (mut camera, world) = match 4 {
        1 => bouncing_spheres(),
        2 => checkerd_sphered(),
        3 => earht(),
        4 => perlin_spheres(),
        _ => panic!("bro...."),
    };
    camera.render(&world);
}
