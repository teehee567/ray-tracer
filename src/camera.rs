use core::panic;
use std::{f32::INFINITY, fmt::Write, time::Instant};

use image::{Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nalgebra::{Point3, Vector, Vector3};

use crate::{
    colour::{self, Colour},
    hittable::{HitRecord, Hittable},
    interval::Interval,
    ray::Ray,
    utils::{self, degrees_to_radians, random_f32},
};

use rayon::prelude::*;

pub struct Camera {
    pub aspect_ratio: f32,
    pub image_width: i32,
    pub samples_per_pixel: i32,
    pub max_depth: i32,
    pub background: Colour,

    pub vfov: i32,
    pub lookfrom: Point3<f32>,
    pub lookat: Point3<f32>,
    pub vup: Vector3<f32>,

    pub defocus_angle: f32,
    pub focus_dist: f32,

    image_height: i32,
    pixel_samples_scale: f32,
    center: Point3<f32>,
    pixel100_loc: Point3<f32>,

    pixel_delta_u: Vector3<f32>,
    pixel_delta_v: Vector3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    w: Vector3<f32>,

    defocus_disk_u: Vector3<f32>,
    defocus_disk_v: Vector3<f32>,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            aspect_ratio: 1.,
            image_width: 3840,
            samples_per_pixel: 10,
            max_depth: 10,
            background: Colour::default(),

            vfov: 90,
            lookfrom: Point3::default(),
            lookat: Point3::default(),
            vup: Vector3::default(),

            defocus_angle: 0.,
            focus_dist: 10.,

            image_height: 0,
            pixel_samples_scale: 0.,
            center: Point3::default(),
            pixel100_loc: Point3::default(),

            pixel_delta_u: Vector3::default(),
            pixel_delta_v: Vector3::default(),
            u: Vector3::default(),
            v: Vector3::default(),
            w: Vector3::default(),

            defocus_disk_u: Vector3::default(),
            defocus_disk_v: Vector3::default(),
        }
    }

    pub fn render(&mut self, world: &impl Hittable) {
        self.initialise();

        let progress_bar = ProgressBar::new(self.image_height as u64)
            .with_finish(indicatif::ProgressFinish::AndClear)
            .with_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f32()).unwrap()));

        // Create an empty image buffer
        let mut img = RgbImage::new(self.image_width as u32, self.image_height as u32);

        let start = Instant::now();

        // Parallel processing of each row
        img.enumerate_rows_mut()
            .par_bridge() // This allows parallel processing of rows
            .for_each(|(j, row)| {
                // Update and print the progress
                progress_bar.inc(1);

                for (i, _, pixel) in row {
                    let mut pixel_color = Colour::new(0., 0., 0.);
                    for _ in 0..self.samples_per_pixel {
                        let ray = self.get_ray(i as i32, j as i32);
                        pixel_color += self.ray_color(&ray, self.max_depth, world);
                    }

                    // Average sample colors.
                    pixel_color = pixel_color * self.pixel_samples_scale;

                    *pixel = colour::convert_colour(pixel_color);
                }
            });

        let duration = start.elapsed();

        progress_bar.finish_and_clear();

        // Finish the progress bar
        println!("Rendering Complete in: {:.3}", duration.as_secs_f32());

        // Save the image as PNG
        img.save("image.png").unwrap();
    }
}

impl Camera {
    fn initialise(&mut self) {
        self.image_height = (self.image_width as f32 / self.aspect_ratio) as i32;

        self.pixel_samples_scale = 1. / self.samples_per_pixel as f32;

        self.center = self.lookfrom;

        // Determine viewport dimensions.
        let focal_length = (self.lookfrom - self.lookat).norm();
        let theta = degrees_to_radians(self.vfov as f32);
        let h = (theta/2.).tan();
        let viewport_height = 2. * h * focal_length;
        let viewport_width = viewport_height * (self.image_width as f32 / self.image_height as f32);

        self.w = (self.lookfrom - self.lookat).normalize();
        self.u = Vector3::cross(&self.vup, &self.w).normalize();
        self.v = Vector3::cross(&self.w, &self.u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u = viewport_width * self.u;
        let viewport_v = viewport_height * - self.v;

        // Calculate the horizontal and vertical delta vectors from the pixel to pixel.
        self.pixel_delta_u = viewport_u / self.image_width as f32;
        self.pixel_delta_v = viewport_v / self.image_height as f32;

        // Calculate the location of the upper left pixel.
        let viewport_upper_left =
            self.center - (focal_length * self.w) - viewport_u / 2. - viewport_v / 2.;

        let viewport_upper_left = self.center - (self.focus_dist * self.w) - viewport_u/2. - viewport_v/2.;
        self.pixel100_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);

        let defocus_radius = self.focus_dist * degrees_to_radians(self.defocus_angle / 2.);
        self.defocus_disk_u = self.u * defocus_radius;
        self.defocus_disk_v = self.v * defocus_radius;
    }

    // Actual ray tracing function
    fn ray_color(&self, ray: &Ray, depth: i32, world: &impl Hittable) -> Colour {
        if (depth <= 0) {
            return Colour::new(0., 0., 0.);
        }

        let mut rec: HitRecord = HitRecord::new();

        // Recursive diffusion bounces
        if (!world.hit(&ray, Interval::new(0.001, INFINITY), &mut rec)) {
            return self.background;
        }

        let mut scattered: Ray = Ray::none();
        let mut attenuation: Colour = Colour::default();
        let mut colour_from_emission = rec.mat.emitted(rec.u, rec.v, &rec.p);

        if (!rec.mat.scatter(ray, &rec, &mut attenuation, &mut scattered)) {
            return colour_from_emission;
        }

        let colour_from_scatter = attenuation.component_mul(&self.ray_color(&scattered, depth - 1, world));

        return colour_from_emission + colour_from_scatter;
    }

    // Gets ray from a pixel position
    fn get_ray(&self, i: i32, j: i32) -> Ray {
        let offset = Self::sample_square();
        let pixel_sample = self.pixel100_loc
            + ((i as f32 + offset.x) * self.pixel_delta_u)
            + ((j as f32 + offset.y) * self.pixel_delta_v);

        let ray_origin = if (self.defocus_angle <= 0.) {
            self.center
        } else {
            self.defocus_disk_sample()
        };

        let ray_direction = pixel_sample - ray_origin;
        let ray_time = random_f32();

        Ray::new_tm(ray_origin, ray_direction, ray_time)
    }

    #[inline]
    fn sample_square() -> Vector3<f32> {
        Vector3::new(random_f32() - 0.5, random_f32() - 0.5, 0.)
    }

    fn defocus_disk_sample(&self) -> Point3<f32> {
        let p = utils::random_in_unit_disk();
        self.center + (p[0] * self.defocus_disk_u) + (p[1] * self.defocus_disk_v)
    }

}
