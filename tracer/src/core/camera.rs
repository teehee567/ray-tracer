use core::panic;
use std::f32::INFINITY;
use std::fmt::Write;
use std::time::Instant;

use image::{GenericImage, GenericImageView, Pixel, Rgb, RgbImage, RgbaImage};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nalgebra::{Point3, Vector, Vector3};
use rayon::prelude::*;

use crate::core::hittable::{HitRecord, Hittable};
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::utils::colour::{self, convert_colour, Colour};
use crate::utils::{self, degrees_to_radians, random_f32};

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

    pub image_height: i32,
    pixel_samples_scale: f32,
    center: Point3<f32>,
    pixel100_loc: Point3<f32>,

    pixel_delta_u: Vector3<f32>,
    pixel_delta_v: Vector3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    w: Vector3<f32>,

    viewport_height: f32,
    viewport_width: f32,

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

            viewport_height: 0.0,
            viewport_width: 0.0,

            defocus_disk_u: Vector3::default(),
            defocus_disk_v: Vector3::default(),
        }
    }

    pub fn render(&mut self, world: &impl Hittable) {
        self.build();

        let progress_bar = ProgressBar::new(self.image_height as u64)
            .with_finish(indicatif::ProgressFinish::AndClear)
            .with_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f32()).unwrap()));

        // Create an empty image buffer
        let mut img = RgbaImage::new(self.image_width as u32, self.image_height as u32);

        let start = Instant::now();

        // Parallel processing of each row
        img.enumerate_rows_mut()
            .par_bridge() // This allows parallel processing of rows
            .for_each(|(j, row)| {
                // Update and print the progress
                progress_bar.inc(1);

                for (i, _, pixel) in row {
                    let mut pixel_color = Colour::new(0., 0., 0., 1.);
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
    pub fn build(&mut self) {
        self.image_height = (self.image_width as f32 / self.aspect_ratio) as i32;

        self.pixel_samples_scale = 1. / self.samples_per_pixel as f32;

        self.center = self.lookfrom;

        // Determine viewport dimensions.
        let focal_length = (self.lookfrom - self.lookat).norm();
        let theta = degrees_to_radians(self.vfov as f32);
        let h = (theta / 2.).tan();

        // Calculate viewport height and width at focus distance
        self.viewport_height = 2. * h * focal_length;
        self.viewport_width =
            self.viewport_height * (self.image_width as f32 / self.image_height as f32);

        self.w = (self.lookfrom - self.lookat).normalize();
        self.u = Vector3::cross(&self.vup, &self.w).normalize();
        self.v = Vector3::cross(&self.w, &self.u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u = self.viewport_width * self.u;
        let viewport_v = self.viewport_height * -self.v;

        // Calculate the horizontal and vertical delta vectors from the pixel to pixel.
        self.pixel_delta_u = viewport_u / self.image_width as f32;
        self.pixel_delta_v = viewport_v / self.image_height as f32;

        // Calculate the location of the upper left pixel.
        let viewport_upper_left =
            self.center - (focal_length * self.w) - viewport_u / 2. - viewport_v / 2.;

        let viewport_upper_left =
            self.center - (self.focus_dist * self.w) - viewport_u / 2. - viewport_v / 2.;
        self.pixel100_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);

        let defocus_radius = self.focus_dist * degrees_to_radians(self.defocus_angle / 2.);
        self.defocus_disk_u = self.u * defocus_radius;
        self.defocus_disk_v = self.v * defocus_radius;
    }

    // Actual ray tracing function
    fn ray_color(&self, ray: &Ray, depth: i32, world: &impl Hittable) -> Colour {
        if (depth <= 0) {
            return Colour::new(0., 0., 0., 1.);
        }

        // Recursive diffusion bounces
        if let Some(mut rec) = world.hit(&ray, Interval::new(0.001, INFINITY)) {
            let mut scattered: Ray = Ray::default();
            let mut attenuation: Colour = Colour::default();
            let mut colour_from_emission = rec.mat.emitted(rec.u, rec.v, &rec.p);

            if (!rec.mat.scatter(ray, &rec, &mut attenuation, &mut scattered)) {
                return colour_from_emission;
            }

            let colour_from_scatter =
                attenuation.component_mul(&self.ray_color(&scattered, depth - 1, world));

            return colour_from_emission + colour_from_scatter;
        } else {
            return self.background;
        }
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

/// Debug
impl Camera {
    pub fn world_to_screen(&self, point: Point3<f32>) -> Option<(i32, i32)> {
        let vec_to_point = point - self.lookfrom;

        // Transform to Camera (view) space
        let x = vec_to_point.dot(&self.u);
        let y = vec_to_point.dot(&self.v);
        let z = vec_to_point.dot(&(-self.w));

        // discard points behind camera
        if z <= 0.0 {
            return None;
        }

        // Perspective projection ont hte image plane
        let xp = (self.focus_dist * x) / z;
        let yp = (self.focus_dist * y) / z;

        // Map to normalized device coordinates (NDC)
        let u = (xp + self.viewport_width / 2.) / self.viewport_width;
        let v = (yp + self.viewport_height / 2.) / self.viewport_height;

        // Check if the point is within the viewport
        if u < 0. || u > 1. || v < 0. || v > 1. {
            return None;
        }

        let i = (u * self.image_width as f32) as i32;
        let j = ((1. - v) * self.image_height as f32) as i32;
        // let i = (u * (self.image_width as f32 - 1.0)) as i32;
        // let j = ((1. - v) * (self.image_height as f32 - 1.0)) as i32;

        Some((i, j))
    }

    pub fn draw_line(
        img: &mut RgbaImage,
        mut x0: i32,
        mut y0: i32,
        mut x1: i32,
        mut y1: i32,
        colour: Colour,
    ) {
        let colour = convert_colour(colour);

        let mut steep = false;
        if (x0 - x1).abs() < (y0 - y1).abs() {
            std::mem::swap(&mut x0, &mut y0);
            std::mem::swap(&mut x1, &mut y1);
            steep = true;
        }

        if x0 > x1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
        }

        let dx = x1 - x0;
        let dy = y1 - y0;

        let derror2 = dy.abs() * 2;
        let mut error2 = 0;
        let mut y = y0;

        for x in x0..=x1 {
            let (target_x, target_y) = if steep {
                (y as u32, x as u32)
            } else {
                (x as u32, y as u32)
            };

            // if let Some(existing_pixel) = img.get_pixel(target_x, target_y).cloned() {
            //     // Blend the existing pixel with the new colour
            //     let blended_pixel = existing_pixel.bl colour);
            //     img.put_pixel(target_x, target_y, blended_pixel);
            // } else {
            //     img.put_pixel(target_x, target_y, colour);
            // }
            img.get_pixel_mut(target_x, target_y).blend(&colour);

            error2 += derror2;
            if (error2 > dx) {
                y += if y1 > y0 { 1 } else { -1 };

                error2 -= dx * 2;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use image::{Rgb, RgbaImage};
    use nalgebra::Vector4;

    use super::*;

    const RED: Vector4<f32> = Colour::new(1., 0., 0., 1.);
    const GREEN: Vector4<f32> = Colour::new(0., 1., 0., 1.);
    const BLUE: Vector4<f32> = Colour::new(0., 0., 1., 1.);

    #[test]
    fn test_draw_line_horizontal() {
        let mut img = RgbaImage::new(10, 10);
        Camera::draw_line(&mut img, 2, 5, 7, 5, RED);

        for x in 2..=7 {
            assert_eq!(img.get_pixel(x as u32, 5).0, [255, 0, 0, 255]);
        }
    }

    #[test]
    fn test_draw_line_vertical() {
        let mut img = RgbaImage::new(10, 10);
        Camera::draw_line(&mut img, 5, 2, 5, 7, GREEN);

        for y in 2..=7 {
            assert_eq!(img.get_pixel(5, y as u32).0, [0, 255, 0, 255]);
        }
    }

    #[test]
    fn test_draw_line_diagonal_positive_slope() {
        let mut img = RgbaImage::new(10, 10);
        Camera::draw_line(&mut img, 0, 0, 9, 9, BLUE);

        for i in 0..=9 {
            assert_eq!(img.get_pixel(i, i).0, [0, 0, 255, 255]);
        }
    }

    #[test]
    fn test_draw_line_diagonal_negative_slope() {
        let mut img = RgbaImage::new(10, 10);
        Camera::draw_line(&mut img, 0, 9, 9, 0, RED);

        for i in 0..=9 {
            assert_eq!(img.get_pixel(i, 9 - i).0, [255, 0, 0, 255]);
        }
    }

    #[test]
    fn test_draw_line_steep_slope() {
        let mut img = RgbaImage::new(10, 10);
        Camera::draw_line(&mut img, 4, 0, 5, 9, GREEN);

        // Line will cover multiple y values for x = 4 and x = 5
        for y in 0..=9 {
            let x = if y <= 4 { 4 } else { 5 };
            assert_eq!(img.get_pixel(x, y).0, [0, 255, 0, 255]);
        }
    }

    #[test]
    fn test_draw_line_single_point() {
        let mut img = RgbaImage::new(10, 10);
        Camera::draw_line(&mut img, 5, 5, 5, 5, RED);

        assert_eq!(img.get_pixel(5, 5).0, [255, 0, 0, 255]);
    }

    #[test]
    fn test_world_to_screen_point_in_front_of_camera() {
        let mut camera = Camera::new();
        camera.lookfrom = Point3::new(0., 0., 0.);
        camera.lookat = Point3::new(0., 0., -1.);
        camera.vup = Vector3::new(0., 1., 0.);
        camera.aspect_ratio = 1.0;
        camera.image_width = 100;
        camera.build();

        let point = Point3::new(0., 0., -1.);
        let screen_coords = camera.world_to_screen(point);

        assert!(screen_coords.is_some());
        let (i, j) = screen_coords.unwrap();
        assert_eq!(i, camera.image_width / 2);
        assert_eq!(j, camera.image_height / 2);
    }

    #[test]
    fn test_world_to_screen_point_behind_camera() {
        let mut camera = Camera::new();
        camera.lookfrom = Point3::new(0., 0., 0.);
        camera.lookat = Point3::new(0., 0., -1.);
        camera.vup = Vector3::new(0., 1., 0.);
        camera.aspect_ratio = 1.0;
        camera.image_width = 100;
        camera.build();

        let point = Point3::new(0., 0., 1.);
        let screen_coords = camera.world_to_screen(point);

        assert!(screen_coords.is_none());
    }

    #[test]
    fn test_world_to_screen_point_outside_viewport() {
        let mut camera = Camera::new();
        camera.lookfrom = Point3::new(0., 0., 0.);
        camera.lookat = Point3::new(0., 0., -1.);
        camera.vup = Vector3::new(0., 1., 0.);
        camera.aspect_ratio = 1.0;
        camera.image_width = 100;
        camera.build();

        let point = Point3::new(10., 0., -1.); // Far right, outside the viewport
        let screen_coords = camera.world_to_screen(point);

        assert!(screen_coords.is_none());
    }

    #[test]
    fn test_world_to_screen_point_at_known_position() {
        let mut camera = Camera::new();
        camera.lookfrom = Point3::new(0., 0., 0.);
        camera.lookat = Point3::new(0., 0., -1.);
        camera.vup = Vector3::new(0., 1., 0.);
        camera.aspect_ratio = 1.0;
        camera.image_width = 100;
        camera.build();

        // Point at the top-left corner of the viewport
        let half_width = camera.viewport_width / 2.0;
        let half_height = camera.viewport_height / 2.0;
        let point = Point3::new(-half_width, half_height, -camera.focus_dist);

        let screen_coords = camera.world_to_screen(point);

        assert!(screen_coords.is_some());
        let (i, j) = screen_coords.unwrap();
        assert_eq!(i, 0);
        assert_eq!(j, 0);
    }

    #[test]
    fn test_world_to_screen_point_exact_center() {
        let mut camera = Camera::new();
        camera.lookfrom = Point3::new(0., 0., 0.);
        camera.lookat = Point3::new(0., 0., -1.);
        camera.vup = Vector3::new(0., 1., 0.);
        camera.aspect_ratio = 1.0;
        camera.image_width = 100;
        camera.build();

        // Point directly in front of the camera at focus distance
        let point = Point3::new(0., 0., -camera.focus_dist);

        let screen_coords = camera.world_to_screen(point);

        assert!(screen_coords.is_some());
        let (i, j) = screen_coords.unwrap();
        assert_eq!(i, camera.image_width / 2);
        assert_eq!(j, camera.image_height / 2);
    }

    #[test]
    fn test_world_to_screen_point_with_different_aspect_ratio() {
        let mut camera = Camera::new();
        camera.lookfrom = Point3::new(0., 0., 0.);
        camera.lookat = Point3::new(0., 0., -1.);
        camera.vup = Vector3::new(0., 1., 0.);
        camera.aspect_ratio = 16.0 / 9.0;
        camera.image_width = 1600;
        camera.build();

        // Point directly in front of the camera at focus distance
        let point = Point3::new(0., 0., -camera.focus_dist);

        let screen_coords = camera.world_to_screen(point);

        assert!(screen_coords.is_some());
        let (i, j) = screen_coords.unwrap();
        assert_eq!(i, camera.image_width / 2);
        assert_eq!(j, camera.image_height / 2);
    }

    #[test]
    fn test_world_to_screen_point_far_away() {
        let mut camera = Camera::new();
        camera.lookfrom = Point3::new(0., 0., 0.);
        camera.lookat = Point3::new(0., 0., -1.);
        camera.vup = Vector3::new(0., 1., 0.);
        camera.aspect_ratio = 1.0;
        camera.image_width = 100;
        camera.build();

        // Point very far away in front of the camera
        let point = Point3::new(0., 0., -1000.);

        let screen_coords = camera.world_to_screen(point);

        assert!(screen_coords.is_some());
        let (i, j) = screen_coords.unwrap();
        assert_eq!(i, camera.image_width / 2);
        assert_eq!(j, camera.image_height / 2);
    }

    #[test]
    fn test_world_to_screen_point_far_to_the_side() {
        let mut camera = Camera::new();
        camera.lookfrom = Point3::new(0., 0., 0.);
        camera.lookat = Point3::new(0., 0., -1.);
        camera.vup = Vector3::new(0., 1., 0.);
        camera.aspect_ratio = 1.0;
        camera.image_width = 100;
        camera.build();

        // Point far to the right but still in front of the camera
        let point = Point3::new(1000., 0., -1000.);

        let screen_coords = camera.world_to_screen(point);

        assert!(screen_coords.is_none());
    }
}
