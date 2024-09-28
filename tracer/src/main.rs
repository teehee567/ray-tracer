#![allow(unused)]
#![allow(internal_features)]
#![feature(core_intrinsics)]
#![feature(portable_simd)]
use std::f32::INFINITY;
use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use image::{Rgb, RgbImage};
use indicatif::ProgressBar;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;


use ray_tracer::accelerators::bvh::{self, BVH};
use ray_tracer::core::camera::{self, Camera};
use ray_tracer::core::hittable::{self, call_wireframe_for_quad};
use ray_tracer::core::hittable_list::{self, HittableList};
use ray_tracer::core::{interval, ray};
use ray_tracer::geometry::objects::cube::{self, Cube};
use ray_tracer::geometry::objects::mesh::{self, Mesh};
use ray_tracer::geometry::objects::quad::{self, Quad};
use ray_tracer::geometry::objects::sphere::{self, Sphere};
use ray_tracer::materials::material::{self, Dielectric, DiffuseLight, Lambertian, Metal};
use ray_tracer::materials::{self};
use ray_tracer::textures::texture::{
    self, CheckerTexture, ImageTexture, NoiseTexture, SolidColour,
};
use ray_tracer::textures::{self};
use ray_tracer::utils::colour::Colour;
use ray_tracer::utils::{self, random_f32, random_f32_in};

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

fn test_line_draw() -> (Camera, HittableList) {
    let mut world = HittableList::none();

    let green = Arc::new(Lambertian::new(&Colour::new(0.12, 0.45, 0.15)));

    world.add(Quad::new(
        Point3::new(-1., -1., 0.),
        Vector3::new(2., 0., 0.),
        Vector3::new(0., 2., 0.),
        green.clone(),
    ));

    let mut camera = Camera::new();

    camera.aspect_ratio = 1.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 30;
    camera.max_depth = 50;
    camera.background = Colour::new(0.7, 0.8, 1.);

    camera.vfov = 80;
    camera.lookfrom = Point3::new(-5., -5., -5.);
    camera.lookat = Point3::new(0., 0., 0.);
    camera.vup = Vector3::new(0., 1., 0.);

    camera.defocus_angle = 0.6;
    camera.focus_dist = 10.;

    camera.build();

    // random white dot

    let mut image = RgbImage::new(camera.image_width as u32, camera.image_height as u32);

    image.put_pixel(50, 50, Rgb([255, 255, 255]));

    let colour = Colour::new(1., 0., 0.);
    for cube in &world.objects {
        unsafe { call_wireframe_for_quad(cube, &mut image, colour, &camera) }
    }

    image.save("wireframe.png").expect("failed");

    (camera, world)
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

    let mut huh = 0;
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = random_f32();
            let center = Point3::new(
                a as f32 + 0.9 * random_f32(),
                0.2,
                b as f32 + 0.9 * random_f32(),
            );

            if (center - Point3::new(4.0, 0.2, 0.0)).norm() > 0.9 {
                huh += 1;
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

    println!("huh: {}, {}", huh, world.len());

    let bvh = BVH::new(world.objects, 0., 1.);
    println!("{}", bvh.to_string());

    world = HittableList::new(bvh);

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

    let difflight = Arc::new(DiffuseLight::new_colour(Colour::new(4., 4., 4.), 1.));
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
    let light = Arc::new(DiffuseLight::new_colour(Colour::new(15., 15., 15.), 1.));

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

    let bvh = BVH::new(world.objects, 0., 1.);
    println!("{}", bvh.to_string());

    world = HittableList::new(bvh);

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
        Arc::new(DiffuseLight::new(
            Arc::new(SolidColour::new(Colour::new(0.7, 0.4, 0.2))),
            1.,
        )),
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

fn mesh_please_work() -> (Camera, HittableList) {
    let mut world = HittableList::none();

    world.add(Mesh::from_file(
        "house.obj",
        Arc::new(Lambertian::new(&Vector3::new(0.5, 0.2, 0.2))),
    ));

    let mirror = Metal::new(&Vector3::new(0.2, 0.2, 0.2), 0.2);
    let test = Lambertian::new(&Vector3::new(0.1, 0.1, 0.1));

    world.add(Quad::new(
        Point3::new(-20., 0., -20.),
        Vector3::new(40., 0., 0.),
        Vector3::new(0., 0., 40.),
        Arc::new(mirror),
    ));

    let sun = Arc::new(DiffuseLight::new_colour(Vector3::new(1., 1.4, 0.), 10.));
    // world.add(Sphere::new(Point3::new(-5., 3.2, -3.), 0.5, sun));

    let mut camera = Camera::new();

    camera.aspect_ratio = 16. / 9.;
    camera.image_width = 3840;
    camera.samples_per_pixel = 400;
    camera.max_depth = 20;

    camera.vfov = 60;
    camera.lookfrom = Point3::new(-3., 3., 4.);
    camera.lookat = Point3::new(0., 2., -1.);
    camera.vup = Vector3::new(0., 1., 0.);
    camera.background = Colour::new(0.7, 0.7, 0.7);

    camera.defocus_angle = 0.;
    camera.focus_dist = 5.;

    (camera, world)
}

fn main() {
    let (mut camera, world) = match 1 {
        1 => bouncing_spheres(),
        2 => checkerd_sphered(),
        3 => earht(),
        4 => perlin_spheres(),
        5 => quads(),
        6 => simple_light(),
        7 => cornell_box(),
        8 => huh(),
        9 => mesh_please_work(),
        10 => test_line_draw(),
        _ => panic!("bro...."),
    };
    camera.render(&world);
}
