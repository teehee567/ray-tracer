#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    e: [f64; 3],
}

impl Vec3 {
    pub fn none() -> Self {
        Vec3 { e: [0.; 3]}
    }

    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { e: [x, y, z] }
    }

    pub fn x(&self) -> f64 {
        self.e[0]
    }

    pub fn y(&self) -> f64 {
        self.e[1]
    }

    pub fn z(&self) -> f64 {
        self.e[2]
    }

    pub fn length(&self) -> f64 {
        self.length_squared().sqrt()
    }

    pub fn length_squared(&self) -> f64 {
        self.e[0] * self.e[0] + self.e[1] * self.e[1] + self.e[2] * self.e[2]
    }

    pub fn near_zero(&self) -> bool {
        let s = 1e-8_f64;
        (f64::abs(self.e[0]) < s) && (f64::abs(self.e[1]) < s) && (f64::abs(self.e[2]) < s)
    }

    pub fn random() -> Self {
        Vec3::new(random_f64(), random_f64(), random_f64())
    }

    pub fn random_in(min: f64, max: f64) -> Self {
        Vec3::new(random_f64_in(min, max), random_f64_in(min, max), random_f64_in(min, max))
    }

    pub fn dot(u: Vec3, v: Vec3) -> f64 {
        u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2]
    }

    pub fn cross(u: Vec3, v: Vec3) -> Vec3 {
        Vec3::new(
            u.e[1] * v.e[2] - u.e[2] * v.e[1],
            u.e[2] * v.e[0] - u.e[0] * v.e[2],
            u.e[0] * v.e[1] - u.e[1] * v.e[0],
        )
    }

    pub fn unit_vector(v: Vec3) -> Vec3 {
        v / v.length()
    }

    pub fn random_in_unit_sphere() -> Vec3 {
        loop {
            let p = Self::random_in(-1., 1.);
            if (p.length_squared() < 1.) {
                return p;
            }
        }
    }

    pub fn random_in_unit_vector() -> Vec3 {
        Self::unit_vector(Self::random_in_unit_sphere())
    }

    pub fn random_on_hemisphere(normal: Vec3) -> Vec3 {
        let on_unit_sphere = Self::random_in_unit_vector();
        if (Self::dot(on_unit_sphere, normal) > 0.0) {
            return on_unit_sphere;
        } else {
            return -on_unit_sphere;
        }
    }

    pub fn reflect(v: &Vec3, n: &Vec3) -> Vec3{
        *v - 2. * Self::dot(*v, *n) * *n
    }

    pub fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3 {
        let cos_theta = Self::dot(-*uv, *n).min(1.);
        let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
        let r_out_parrallel = -(1. - r_out_perp.length_squared()).abs().sqrt() * *n;

        return r_out_perp + r_out_parrallel;
    }

    pub fn random_in_unit_disk() -> Vec3 {
        loop {
            let p = Vec3::new(random_f64_in(-1., 1.), random_f64_in(-1., 1.), 0.);
            if (p.length_squared() < 1.) {
                return p;
            }
        }
    }
}

use std::cmp::min;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, Neg, Sub};

impl Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        Vec3::new(-self.e[0], -self.e[1], -self.e[2])
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;

    fn index(&self, i: usize) -> &f64 {
        &self.e[i]
    }
}

impl IndexMut<usize> for Vec3 {
    fn index_mut(&mut self, i: usize) -> &mut f64 {
        &mut self.e[i]
    }
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.e[0] + other.e[0],
            self.e[1] + other.e[1],
            self.e[2] + other.e[2],
        )
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        self.e[0] += other.e[0];
        self.e[1] += other.e[1];
        self.e[2] += other.e[2];
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.e[0] - other.e[0],
            self.e[1] - other.e[1],
            self.e[2] - other.e[2],
        )
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, t: f64) -> Vec3 {
        Vec3::new(self.e[0] * t, self.e[1] * t, self.e[2] * t)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, t: Vec3) -> Vec3 {
        Vec3::new(t.e[0] * self, t.e[1] * self, t.e[2] * self)
    }
}

impl Mul for Vec3 {
    type Output = Vec3;

    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.e[0] * other.e[0],
            self.e[1] * other.e[1],
            self.e[2] * other.e[2],
        )
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, t: f64) -> Vec3 {
        self * (1.0 / t)
    }
}

impl DivAssign<f64> for Vec3 {
    fn div_assign(&mut self, t: f64) {
        let k = 1.0 / t;
        self.e[0] *= k;
        self.e[1] *= k;
        self.e[2] *= k;
    }
}

// Alias for geometric clarity
pub type Point3 = Vec3;

// Utility function for printing
use std::fmt;

use crate::utils::{random_f64, random_f64_in};

impl fmt::Display for Vec3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.e[0], self.e[1], self.e[2])
    }
}
