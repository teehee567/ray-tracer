use nalgebra::{Point3, Vector3};

use crate::{accelerators::aabb::AABB, core::{interval::Interval, ray::Ray}};

use super::mesh::{intersect_triangle, Primitive, TriangleIntersection};

pub struct Triangle {
    pub vertices: [usize; 3],
    pub normal: Vector3<f32>,
}

pub struct Vertex {
    pub position: Point3<f32>,
    pub normal: Vector3<f32>,
}

impl Primitive<f32> for Triangle {
    fn centroid(&self, vertices: &[Vertex]) -> Vector3<f32> {
        (vertices[self.vertices[0]].position.coords
        + vertices[self.vertices[1]].position.coords
        + vertices[self.vertices[2]].position.coords
        ) * 0.3333333f32
    }

    fn aabb(&self, vertices: &[Vertex]) -> AABB {
        let v0 = vertices[self.vertices[0]].position;
        let v1 = vertices[self.vertices[1]].position;
        let v2 = vertices[self.vertices[2]].position;
        AABB::new(v0, v1).grow(v2)
    }

    fn intersect_prim(&self, ray: &Ray, vertex_list: &[Vertex], ray_t: Interval) -> Option<TriangleIntersection> {
        let tri_a = vertex_list[self.vertices[0]].position.coords;
        let tri_b = vertex_list[self.vertices[1]].position.coords;
        let tri_c = vertex_list[self.vertices[2]].position.coords;
        intersect_triangle(ray, ray_t, tri_a, tri_b, tri_c)
    }
}
