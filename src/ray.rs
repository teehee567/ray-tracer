use nalgebra::{Point3, Vector3};

#[derive(Debug, Clone, Copy, Default)]
pub struct Ray {
    orig: Point3<f32>,
    dir: Vector3<f32>,
    tm: f32,
}

impl Ray {
    pub fn new_tm(origin: Point3<f32>, direction: Vector3<f32>, tm: f32) -> Self {
        Self {
            orig: origin,
            dir: direction,
            tm,
        }
    }

    pub fn new(origin: Point3<f32>, direction: Vector3<f32>) -> Self {
        Self {
            orig: origin,
            dir: direction,
            tm: 0.,
        }
    }

    pub fn time(&self) -> f32 {
        self.tm
    }

    pub fn origin(&self) -> &Point3<f32> {
        &self.orig
    }

    pub fn direction(&self) -> &Vector3<f32> {
        &self.dir
    }

    pub fn at(&self, t: f32) -> Point3<f32> {
        self.orig + self.dir * t
    }
}
