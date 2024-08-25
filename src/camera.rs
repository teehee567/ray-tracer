use core::panic;
use std::{f64::INFINITY, fmt::Write, time::Instant};

use image::{Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};

use crate::{
    colour::Colour,
    hittable::{HitRecord, Hittable},
    interval::Interval,
    ray::Ray,
    utils::{degrees_to_radians, random_f64},
    vec3::{Point3, Vec3},
};

use rayon::prelude::*;

pub struct Camera {
    pub aspect_ratio: f64,
    pub image_width: i32,
    pub samples_per_pixel: i32,
    pub max_depth: i32,

    pub vfov: i32,
    pub lookfrom: Point3,
    pub lookat: Point3,
    pub vup: Vec3,

    pub defocus_angle: f64,
    pub focus_dist: f64,

    image_height: i32,
    pixel_samples_scale: f64,
    center: Point3,
    pixel100_loc: Point3,

    pixel_delta_u: Vec3,
    pixel_delta_v: Vec3,
    u: Vec3,
    v: Vec3,
    w: Vec3,

    defocus_disk_u: Vec3,
    defocus_disk_v: Vec3,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            aspect_ratio: 1.,
            image_width: 3840,
            samples_per_pixel: 10,
            max_depth: 10,

            vfov: 90,
            lookfrom: Point3::none(),
            lookat: Point3::none(),
            vup: Vec3::none(),

            defocus_angle: 0.,
            focus_dist: 10.,

            image_height: 0,
            pixel_samples_scale: 0.,
            center: Point3::none(),
            pixel100_loc: Point3::none(),

            pixel_delta_u: Vec3::none(),
            pixel_delta_v: Vec3::none(),
            u: Vec3::none(),
            v: Vec3::none(),
            w: Vec3::none(),

            defocus_disk_u: Vec3::none(),
            defocus_disk_v: Vec3::none(),
        }
    }

    pub fn render(&mut self, world: &impl Hittable) {
        self.initialise();

        let progress_bar = ProgressBar::new(self.image_height as u64)
            .with_finish(indicatif::ProgressFinish::AndClear)
            .with_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()));

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
                        pixel_color += Self::ray_color(&ray, self.max_depth, world);
                    }

                    // Average sample colors.
                    pixel_color = pixel_color * self.pixel_samples_scale;

                    *pixel = pixel_color.convert_colour();
                }
            });

        let duration = start.elapsed();

        progress_bar.finish_and_clear();

        // Finish the progress bar
        println!("Rendering Complete in: {:.3}", duration.as_secs_f64());

        // Save the image as PNG
        img.save("image.png").unwrap();
    }
}

impl Camera {
    fn initialise(&mut self) {
        self.image_height = (self.image_width as f64 / self.aspect_ratio) as i32;

        self.pixel_samples_scale = 1. / self.samples_per_pixel as f64;

        self.center = self.lookfrom;

        // Determine viewport dimensions.
        let focal_length = (self.lookfrom - self.lookat).length();
        let theta = degrees_to_radians(self.vfov as f64);
        let h = (theta/2.).tan();
        let viewport_height = 2. * h * focal_length;
        let viewport_width = viewport_height * (self.image_width as f64 / self.image_height as f64);

        self.w = Vec3::unit_vector(self.lookfrom - self.lookat);
        self.u = Vec3::unit_vector(Vec3::cross(self.vup, self.w));
        self.v = Vec3::cross(self.w, self.u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u = viewport_width * self.u;
        let viewport_v = viewport_height * - self.v;

        // Calculate the horizontal and vertical delta vectors from the pixel to pixel.
        self.pixel_delta_u = viewport_u / self.image_width as f64;
        self.pixel_delta_v = viewport_v / self.image_height as f64;

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
    fn ray_color(ray: &Ray, depth: i32, world: &impl Hittable) -> Colour {
        if (depth <= 0) {
            return Colour::new(0., 0., 0.);
        }

        let mut rec: HitRecord = HitRecord::new();
        
        // Recursive diffusion bounces
        if (world.hit(&ray, Interval::new(0.001, INFINITY), &mut rec)) {
            let mut scattered: Ray = Ray::none();
            let mut attenuation: Colour = Colour::none();
            if (rec.mat.scatter(ray, &rec, &mut attenuation, &mut scattered)) {
                return attenuation * Self::ray_color(&scattered, depth - 1, world);
            }
            return Colour::none();
        }

        let unit_direction = Vec3::unit_vector(*ray.direction());
        let a = 0.5 * (unit_direction.y() + 1.);
        return (1. - a) * Colour::new(1., 1., 1.) + a * Colour::new(0.5, 0.7, 1.);
    }

    // Gets ray from a pixel position
    fn get_ray(&self, i: i32, j: i32) -> Ray {
        let offset = Self::sample_square();
        let pixel_sample = self.pixel100_loc
            + ((i as f64 + offset.x()) * self.pixel_delta_u)
            + ((j as f64 + offset.y()) * self.pixel_delta_v);

        let ray_origin = if (self.defocus_angle <= 0.) {
            self.center
        } else {
            self.defocus_disk_sample()
        };

        let ray_direction = pixel_sample - ray_origin;
        let ray_time = random_f64();

        Ray::new_tm(ray_origin, ray_direction, ray_time)
    }

    #[inline]
    fn sample_square() -> Vec3 {
        Vec3::new(random_f64() - 0.5, random_f64() - 0.5, 0.)
    }

    fn defocus_disk_sample(&self) -> Point3 {
        let p = Vec3::random_in_unit_disk();
        self.center + (p[0] * self.defocus_disk_u) + (p[1] * self.defocus_disk_v)
    }

}
