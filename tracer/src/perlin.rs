use nalgebra::{Point3, Vector3};

use crate::utils;

pub struct Perlin {
    random_points: Vec<Vector3<f32>>,
    x: Vec<usize>,
    y: Vec<usize>,
    z: Vec<usize>,
}

impl Perlin {
    const POINT_COUNT: usize = 256;

    fn perlin_generate() -> Vec<Vector3<f32>> {
        (0..Self::POINT_COUNT)
            .map(|_| utils::rand_vec_in(-1., 1.))
            .collect()
    }

    fn generate_permutation() -> Vec<usize> {
        let mut points: Vec<usize> = (0..Self::POINT_COUNT).collect();

        Self::permute(&mut points);

        points
    }

    fn permute(points: &mut [usize]) {
        for i in (1..points.len()).rev() {
            let target = fastrand::usize(0..i);
            points.swap(i, target);
        }
    }

    pub fn new() -> Self {
        Self {
            random_points: Self::perlin_generate(),
            x: Self::generate_permutation(),
            y: Self::generate_permutation(),
            z: Self::generate_permutation(),
        }
    }

    pub fn noise(&self, point: Point3<f32>) -> f32 {
        let u = point.x - point.x.floor();
        let v = point.y - point.y.floor();
        let w = point.z - point.z.floor();

        let i = point.x.floor() as i32;
        let j = point.y.floor() as i32;
        let k = point.z.floor() as i32;
        let mut c = [[[Vector3::default(); 2]; 2]; 2]; // Vec3f c[2][2][2];
        for (di, item) in c.iter_mut().enumerate() {
            for (dj, item) in item.iter_mut().enumerate() {
                for (dk, item) in item.iter_mut().enumerate() {
                    let x = self.x[((i + di as i32) & 255) as usize];
                    let y = self.y[((j + dj as i32) & 255) as usize];
                    let z = self.z[((k + dk as i32) & 255) as usize];
                    *item = self.random_points[x ^ y ^ z];
                }
            }
        }

        Self::perlin_interp(c, u, v, w)
    }

    fn perlin_interp(c: [[[Vector3<f32>; 2]; 2]; 2], u: f32, v: f32, w: f32) -> f32 {
        let uu = u * u * (3. - 2. * u);
        let vv = v * v * (3. - 2. * v);
        let ww = w * w * (3. - 2. * w);
        let mut accumulator = 0.0;

        for (i, item) in c.iter().enumerate() {
            for (j, item) in item.iter().enumerate() {
                for (k, item) in item.iter().enumerate() {
                    let weight = Vector3::new(u - i as f32, v - j as f32, w - k as f32);
                    accumulator += (i as f32 * uu + (1 - i) as f32 * (1. - uu))
                        * (j as f32 * vv + (1 - j) as f32 * (1. - vv))
                        * (k as f32 * ww + (1 - k) as f32 * (1. - ww))
                        * Vector3::dot(item, &weight);
                }
            }
        }
        accumulator
    }

    pub fn turbulence(&self, point: Point3<f32>, depth: u32) -> f32 {
        if depth == 0 {
            return 0.;
        }
        let (accumulator, _temp, _weight) = (0..depth).fold((0.0, point, 1.0), |accumulator, _| {
            let accum = accumulator.0 + accumulator.2 * self.noise(accumulator.1);
            let weight = accumulator.2 * 0.5;
            let temp = accumulator.1 * 2.0;

            (accum, temp, weight)
        });
        accumulator
    }
}
