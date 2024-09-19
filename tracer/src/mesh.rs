use nalgebra::{Point2, Point3, Vector3};

use crate::{
    aabb::AABB,
    hittable::{HitRecord, Hittable},
    interval::Interval,
    ray::Ray,
};

pub struct TriangleMesh {
    pub vertex_i: Vec<i32>,
    pub p: Vec<Point3<f32>>,
    pub s: Vec<Vector3<f32>>,
    pub n: Vec<Vector3<f32>>,
    pub uv: Vec<Point2<f32>>,
}

impl TriangleMesh {}

pub struct Triangle {
    pub mesh_i: i32,
    pub tri_i: i32,
    pub aabb: AABB, //FIX: get rid of aabb here, make it calculated??? test differences
}

impl Triangle {
    pub fn new(mesh: &TriangleMesh, mesh_i: i32, tri_i: i32) -> Self {
        let v = &mesh.vertex_i[(tri_i * 3)..(tri_i * 3 + 3)];
        let p0 = mesh.p[v[0]];
        let p1 = mesh.p[v[1]];
        let p2 = mesh.p[v[2]];
        let aabb = AABB::new(p0, p1).union(p2);

        Self {
            mesh_i,
            tri_i,
            aabb,
        }
    }
}

impl Hittable for Triangle {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        todo!()
    }

    fn bounding_box(&self) -> &AABB {}
}
