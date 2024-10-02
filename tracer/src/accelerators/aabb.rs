use std::arch::x86_64::_mm_min_ps;
use std::ops::Index;

use image::{RgbImage, RgbaImage};
use nalgebra::coordinates::XYZ;
use nalgebra::{Point3, SimdValue, Vector3};

use crate::core::camera::Camera;
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::geometry::objects::triangle::Vertex;
use crate::geometry::wireframe::WireFrame;
use crate::utils::colour::Colour;

#[derive(Clone, Copy, Default, Debug)]
pub struct AABB {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl AABB {
    #[inline]
    pub fn new(p0: Point3<f32>, p1: Point3<f32>) -> Self {
        let min = Point3::new(p0.x.min(p1.x), p0.y.min(p1.y), p0.z.min(p1.z));
        let max = Point3::new(p0.x.max(p1.x), p0.y.max(p1.y), p0.z.max(p1.z));
        AABB { min, max }
    }

    pub fn intersect(&self, ray: &Ray) -> bool {
        let inv_d = ray.direction().map(|x| 1.0f32 / x);
        let lbr = (self[0].coords - ray.origin().coords).component_mul(&inv_d);
        let rtr = (self[1].coords - ray.origin().coords).component_mul(&inv_d);

        let (inf, sup) = lbr.inf_sup(&rtr);

        let tmin = inf.max();
        let tmax = sup.min();

        tmax > tmin.max(0.0f32)
    }

    pub fn hit(&self, ray: &Ray, ray_t: Interval) -> bool {
        let mut t_min = ray_t.min;
        let mut t_max = ray_t.max;
        for a in 0..3 {
            let inv_d = 1.0 / ray.direction()[a];
            let t0 = (self.min[a] - ray.origin()[a]) * inv_d;
            let t1 = (self.max[a] - ray.origin()[a]) * inv_d;
            let (t0, t1) = if inv_d < 0.0 { (t1, t0) } else { (t0, t1) };
            t_min = t_min.max(t0);
            t_max = t_max.min(t1);
            if t_max <= t_min {
                return false;
            }
        }
        true
    }
    //
    // pub fn from_list(aabbs: &[Self]) -> Self {
    //     let mut aabb = Self::default();
    //     for box in aabbs.iter() {
    //         aabb = Self::combine(&aabb, &box);
    //     }
    //     aabb
    // }

    pub fn grow_bb_mut(&mut self, aabb: &Self) {
        *self = Self::combine(self, aabb)
    }

    pub fn offset_by(&mut self, delta: f32) {
        let delta = Vector3::repeat(delta);
        let min = self.min - delta;
        let max = self.max + delta;

        self.min = min;
        self.max = max;
    }

    #[inline(always)]
    pub fn grow(&mut self, vertex: Point3<f32>) -> AABB {
        AABB::combine(self, &AABB {min: vertex, max: vertex})
    }

    #[inline(always)]
    pub fn grow_bb(&self, aabb: &Self) -> AABB {
        Self::combine(self, aabb)
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.diagonal();
        let x = d.x;
        let y = d.y;
        let z = d.z;
        2.0f32 * (x * y + y * z + z * x)
    }

    pub fn half_area(&self) -> f32 {
        let d = self.diagonal();
        d.x * d.y + d.y * d.z + d.z * d.x
    }

    pub fn diagonal(&self) -> Vector3<f32> {
        self.max - self.min
    }

    pub fn largest_axis(&self) -> usize {
        self.diagonal().iamax()
    }
}

impl AABB {
    #[inline(always)]
    pub fn combine_scalar(box0: &AABB, box1: &AABB) -> AABB {
        let min = Point3::from([
            box0.min.x.min(box1.min.x),
            box0.min.y.min(box1.min.y),
            box0.min.z.min(box1.min.z),
        ]);
        let max = Point3::from([
            box0.max.x.max(box1.max.x),
            box0.max.y.max(box1.max.y),
            box0.max.z.max(box1.max.z),
        ]);
        AABB { min, max }
    }

    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    // sse
    pub fn combine_sse(box0: &AABB, box1: &AABB) -> AABB {
        use std::arch::x86_64::{_mm_extract_ps, _mm_max_ps, _mm_min_ps, _mm_set_ps, _mm_store_ps};
        unsafe {
            // Create min vectors
            let min0 = _mm_set_ps(0.0, box0.min.z, box0.min.y, box0.min.x);
            let min1 = _mm_set_ps(0.0, box1.min.z, box1.min.y, box1.min.x);
            let min_vec = _mm_min_ps(min0, min1);

            // Create max vectors
            let max0 = _mm_set_ps(0.0, box0.max.z, box0.max.y, box0.max.x);
            let max1 = _mm_set_ps(0.0, box1.max.z, box1.max.y, box1.max.x);
            let max_vec = _mm_max_ps(max0, max1);

            // Store the min_vec and max_vec into arrays
            let mut min_array = [0.0f32; 4];
            let mut max_array = [0.0f32; 4];
            _mm_store_ps(min_array.as_mut_ptr(), min_vec);
            _mm_store_ps(max_array.as_mut_ptr(), max_vec);

            // Create Point3 from the arrays
            let min = Point3::from([min_array[0], min_array[1], min_array[2]]);
            let max = Point3::from([max_array[0], max_array[1], max_array[2]]);

            AABB { min, max }
        }
    }

    #[inline(always)]
    #[cfg(target_arch = "x86_64")]
    // avx2
    // FIX: ASSS
    pub fn combine_avx2(box0: &AABB, box1: &AABB) -> AABB {
        use std::arch::x86_64::{
            _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_min_ps, _mm256_set_ps, _mm256_store_ps, _mm_extract_ps
        };
        unsafe {
            let box0_array = [
                box0.min.x, box0.min.y, box0.min.z, box0.max.x, box0.max.y, box0.max.z, 0.0, 0.0,
            ];
            let box1_array = [
                box1.min.x, box1.min.y, box1.min.z, box1.max.x, box1.max.y, box1.max.z, 0.0, 0.0,
            ];

            let box0_vec = _mm256_loadu_ps(box0_array.as_ptr());
            let box1_vec = _mm256_loadu_ps(box1_array.as_ptr());

            let min_vec = _mm256_min_ps(box0_vec, box1_vec);
            let max_vec = _mm256_max_ps(box0_vec, box1_vec);

            let mut min_array = [0.0f32; 8];
            let mut max_array = [0.0f32; 8];

            _mm256_store_ps(min_array.as_mut_ptr(), min_vec);
            _mm256_store_ps(max_array.as_mut_ptr(), max_vec);

            let min = Point3::from([min_array[0], min_array[1], min_array[2]]);
            let max = Point3::from([max_array[3], max_array[4], max_array[5]]);

            AABB { min, max }
        }
    }

    #[inline(always)]
    pub fn combine(box0: &AABB, box1: &AABB) -> AABB {
        // #[cfg(not(target_arch = "x86_64"))]
        {
            Self::combine_scalar(box0, box1)
        }
        //
        // #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        // {
        //     Self::combine_avx2(box0, box1)
        // }
        //
        // #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
        // {
        //     Self::combine_sse(box0, box1)
        // }
    }
}

impl Index<usize> for AABB {
    type Output = Point3<f32>;

    fn index(&self, idx: usize) -> &Self::Output {
        match idx {
            0 => &self.min,
            1 => &self.max,
            _ => panic!("Index out of range! AABB only has two elements: min and max"),
        }
    }
}

impl WireFrame for AABB {
    fn draw_wireframe(&self, img: &mut RgbaImage, colour: Colour, camera: &Camera) {
        // Define the 8 vertices of the cube
        let vertices = [
            self.min,
            Point3::new(self.min.x, self.min.y, self.max.z),
            Point3::new(self.min.x, self.max.y, self.min.z),
            Point3::new(self.min.x, self.max.y, self.max.z),
            Point3::new(self.max.x, self.min.y, self.min.z),
            Point3::new(self.max.x, self.min.y, self.max.z),
            Point3::new(self.max.x, self.max.y, self.min.z),
            self.max,
        ];

        let edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];

        for &(start, end) in &edges {
            if let (Some((x0, y0)), Some((x1, y1))) = (
                camera.world_to_screen(vertices[start]),
                camera.world_to_screen(vertices[end]),
            ) {
                // println!("Drawing edge: {}, {} -> {}, {}", x0, y0, x1, y1);
                Camera::draw_line(img, x0, y0, x1, y1, colour);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_overlapping_boxes() {
        let box0 = AABB {
            min: Point3::from([0.0, 0.0, 0.0]),
            max: Point3::from([1.0, 1.0, 1.0]),
        };
        let box1 = AABB {
            min: Point3::from([2.0, 2.0, 2.0]),
            max: Point3::from([3.0, 3.0, 3.0]),
        };
        let combined = AABB::combine_sse(&box0, &box1);

        let expected = AABB {
            min: Point3::from([0.0, 0.0, 0.0]),
            max: Point3::from([3.0, 3.0, 3.0]),
        };

        assert_eq!(combined.min, expected.min);
        assert_eq!(combined.max, expected.max);
    }

    #[test]
    fn test_overlapping_boxes() {
        let box0 = AABB {
            min: Point3::from([0.0, 0.0, 0.0]),
            max: Point3::from([2.0, 2.0, 2.0]),
        };
        let box1 = AABB {
            min: Point3::from([1.0, 1.0, 1.0]),
            max: Point3::from([3.0, 3.0, 3.0]),
        };
        let combined = AABB::combine_sse(&box0, &box1);

        let expected = AABB {
            min: Point3::from([0.0, 0.0, 0.0]),
            max: Point3::from([3.0, 3.0, 3.0]),
        };

        assert_eq!(combined.min, expected.min);
        assert_eq!(combined.max, expected.max);
    }

    #[test]
    fn test_one_inside_another() {
        let box0 = AABB {
            min: Point3::from([0.0, 0.0, 0.0]),
            max: Point3::from([4.0, 4.0, 4.0]),
        };
        let box1 = AABB {
            min: Point3::from([1.0, 1.0, 1.0]),
            max: Point3::from([2.0, 2.0, 2.0]),
        };
        let combined = AABB::combine_sse(&box0, &box1);

        let expected = box0;

        assert_eq!(combined.min, expected.min);
        assert_eq!(combined.max, expected.max);
    }

    #[test]
    fn test_identical_boxes() {
        let box0 = AABB {
            min: Point3::from([1.0, 1.0, 1.0]),
            max: Point3::from([2.0, 2.0, 2.0]),
        };
        let box1 = box0.clone();
        let combined = AABB::combine_sse(&box0, &box1);

        let expected = box0;

        assert_eq!(combined.min, expected.min);
        assert_eq!(combined.max, expected.max);
    }

    #[test]
    fn test_negative_coordinates() {
        let box0 = AABB {
            min: Point3::from([-3.0, -3.0, -3.0]),
            max: Point3::from([-1.0, -1.0, -1.0]),
        };
        let box1 = AABB {
            min: Point3::from([-2.0, -2.0, -2.0]),
            max: Point3::from([0.0, 0.0, 0.0]),
        };
        let combined = AABB::combine_sse(&box0, &box1);

        let expected = AABB {
            min: Point3::from([-3.0, -3.0, -3.0]),
            max: Point3::from([0.0, 0.0, 0.0]),
        };

        assert_eq!(combined.min, expected.min);
        assert_eq!(combined.max, expected.max);
    }

    #[test]
    fn test_degenerate_boxes() {
        let box0 = AABB {
            min: Point3::from([1.0, 1.0, 1.0]),
            max: Point3::from([1.0, 1.0, 1.0]),
        };
        let box1 = AABB {
            min: Point3::from([2.0, 2.0, 2.0]),
            max: Point3::from([2.0, 2.0, 2.0]),
        };
        let combined = AABB::combine_scalar(&box0, &box1);

        let expected = AABB {
            min: Point3::from([1.0, 1.0, 1.0]),
            max: Point3::from([2.0, 2.0, 2.0]),
        };

        assert_eq!(combined.min, expected.min);
        assert_eq!(combined.max, expected.max);
    }
}
