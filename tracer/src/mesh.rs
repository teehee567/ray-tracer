use nalgebra::{Normed, Point2, Point3, Vector3};

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

// PBRT "Unlike the other shapes so far, pbrt provides a stand-alone triangle intersection
// function that takes a ray and the three triangle vertices directly. Having this
// functionality available without needing to instantiate both a Triangle and a
// TriangleMesh in order to do a ray–triangle intersection test is helpful in a few
// other parts of the system"
pub struct TriangleIntersection {

}

pub fn intersect_triange(
    ray: &Ray,
    ray_t: f32,
    p0: Point3<f32>,
    p1: Point3<f32>,
    p2: Point3<f32>,
) -> Option<TriangleIntersection> {
    if Vector3::cross(&(p2 - p0), &(p1 - p0)).norm_squared() == 0. {
        return None;
    }


    todo!()
}

impl Triangle {
    pub fn new(mesh: &TriangleMesh, mesh_i: i32, tri_i: i32) -> Self {
        let v = &mesh.vertex_i[(tri_i as usize * 3)..(tri_i as usize * 3 + 3)];
        let p0 = mesh.p[v[0] as usize];
        let p1 = mesh.p[v[1] as usize];
        let p2 = mesh.p[v[2] as usize];
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

    fn bounding_box(&self) -> &AABB {
        &self.aabb
    }
}
