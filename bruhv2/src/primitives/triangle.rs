use glam::Vec3;

#[derive(Clone, Copy, Debug, Default)]
pub struct Triangle {
    pub p1: Vec3,
    pub p2: Vec3,
    pub p3: Vec3,
    pub center: Vec3,
}

impl Triangle {
    pub fn new(p1: Vec3, p2: Vec3, p3: Vec3) -> Self {
        let center = (p1 + p2 + p3) / 3.0;
        Self { p1, p2, p3, center }
    }
}
