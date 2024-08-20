#![allow(unused)]
use std::f64::INFINITY;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{fs::File, sync::Arc};
use std::io::Write;

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
use utils::random_f64;
use vec3::{Point3, Vec3};


mod vec3;
mod colour;
mod ray;
mod hittable;
mod sphere;
mod hittable_list;
mod utils;
mod interval;
mod camera;
mod material;

fn main() {
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

    let mut world = HittableList::none();
    let material_ground = Lambertian::new(&Colour::new(0.5, 0.5, 0.5));
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
                let material: Arc<dyn Material> = if choose_mat < 0.8 {
                    // Lambertian
                    let albedo = Colour::random() * Colour::random();
                    Arc::new(Lambertian::new(&albedo))
                } else if choose_mat < 0.95 {
                    // Metal
                    let albedo = Colour::random_in(0.5, 1.0);
                    let fuzz = random_f64() * 0.5;
                    Arc::new(Metal::new(&albedo, fuzz))
                } else {
                    // Glass
                    Arc::new(Dielectric::new(1.5))
                };

                world.add(Sphere::new(center, 0.2, material));
            }
        }
    }

    let material1 = Arc::new(Dielectric::new(1.5));
    world.add(Sphere::new(Point3::new(0., 1., 0.), 1., material1));

    let material2 = Arc::new(Lambertian::new(&Colour::new(0.4, 0.2, 0.1)));
    world.add(Sphere::new(Point3::new(-4., 1., 0.), 1., material2));

    let material3 = Arc::new(Metal::new(&Colour::new(0.7, 0.6, 0.5), 0.));
    world.add(Sphere::new(Point3::new(4., 1., 0.), 1., material3));

    let mut camera = Camera::new();

    camera.aspect_ratio = 16. / 9.;
    camera.image_width = 1200;
    camera.samples_per_pixel = 100;
    camera.max_depth = 50;

    camera.vfov = 20;
    camera.lookfrom = Point3::new(13., 2., 3.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vec3::new(0., 1., 0.);

    camera.defocus_angle = 0.;
    camera.focus_dist = 10.;

    camera.render(&world);
}
