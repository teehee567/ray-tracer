use crate::vec3::{Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    orig: Point3,
    dir: Vec3,
    tm: f64,
}

impl Ray {
    pub fn none() -> Self {
        Ray::new(Point3::none(), Point3::none())
    }

    pub fn new_tm(origin: Point3, direction: Vec3, tm: f64) -> Self {
        Self {
            orig: origin,
            dir: direction,
            tm,
        }
    }

    pub fn new(origin: Point3, direction: Vec3) -> Self {
        Self {
            orig: origin,
            dir: direction,
            tm: 0.,
        }
    }

    pub fn time(&self) -> f64 {
        self.tm
    }

    pub fn origin(&self) -> &Point3 {
        &self.orig
    }

    pub fn direction(&self) -> &Vec3 {
        &self.dir
    }

    pub fn at(&self, t: f64) -> Point3 {
        self.orig + self.dir * t
    }
}
