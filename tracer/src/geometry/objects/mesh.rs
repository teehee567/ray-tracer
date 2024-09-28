use core::arch::x86_64::*;
use std::f32::EPSILON;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use nalgebra::{ComplexField, Normed, Point2, Point3, Vector3};
use obj::{load_obj, Obj};
use serde::Serialize;

use crate::accelerators::aabb::AABB;
use crate::accelerators::bvh::BVH;
use crate::core::hittable::{HitRecord, Hittable};
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::materials::material::Material;

// NOTE: pbrt container, fix post gpu
pub struct TriangleMesh {
    pub vertex_i: Vec<i32>,
    pub p: Vec<Point3<f32>>,
    pub s: Vec<Vector3<f32>>,
    pub n: Vec<Vector3<f32>>,
    pub uv: Vec<Point2<f32>>,
    pub bbox: AABB,
}

pub struct Triangle {
    vertices: [u32; 3],
    normal: Vector3<f32>,
}

pub struct Vertex {
    position: Point3<f32>,
    normal: Vector3<f32>,
}

pub struct Mesh {
    /// Vertex buffer
    vertices: Vec<Vertex>,
    triangles: Vec<Triangle>,
    bvh_root: Option<BVH>,
    mat: Arc<dyn Material>,
    bbox: AABB,
}

impl Serialize for Mesh {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        loop {
            panic!();
        }
    }
}

impl Mesh {
    pub fn from_file(path: &str, mat: Arc<dyn Material>) -> Self {
        let obj: Obj<obj::Vertex, u32> =
            load_obj(BufReader::new(File::open(path).unwrap())).unwrap();
        let mut vertices: Vec<Vertex> = Vec::new();

        for vertex in obj.vertices {
            vertices.push(Vertex {
                position: Point3::new(vertex.position[0], vertex.position[1], vertex.position[2]),
                normal: Vector3::new(vertex.normal[0], vertex.normal[1], vertex.normal[2]),
            });
        }

        Self::from_buffers(vertices, obj.indices, mat)
    }

    pub fn from_buffers(vertices: Vec<Vertex>, indices: Vec<u32>, mat: Arc<dyn Material>) -> Self {
        let mut triangles = Vec::<Triangle>::new();

        for triangle in indices.chunks_exact(3) {
            let e1 =
                vertices[triangle[0] as usize].position - vertices[triangle[1] as usize].position;
            let e2 =
                vertices[triangle[2] as usize].position - vertices[triangle[1] as usize].position;
            let normal = e2.cross(&e1).normalize();

            triangles.push(Triangle {
                vertices: [triangle[0], triangle[1], triangle[2]],
                normal,
            })
        }

        // Calculate bounding box
        let mut bounds = AABB::default();
        for vertex in &vertices[..] {
            bounds = AABB::combine(
                &bounds,
                &AABB {
                    min: vertex.position,
                    max: vertex.position,
                },
            );
        }

        Self {
            vertices,
            triangles,
            mat,
            bvh_root: None,
            bbox: bounds,
        }
    }
}

impl Hittable for Mesh {
    fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let mut result = None;
        let mut closest_so_far = ray_t.max;

        for triangle in &self.triangles {
            if let Some(hit) = intersect_triangle(
                ray,
                Interval::new(ray_t.min, closest_so_far),
                self.vertices[triangle.vertices[0] as usize].position.coords,
                self.vertices[triangle.vertices[1] as usize].position.coords,
                self.vertices[triangle.vertices[2] as usize].position.coords,
            ) {
                if closest_so_far > hit.t {
                    closest_so_far = hit.t;
                    result = Some(hit);
                }
            }
        }

        if let Some(result) = result {
            rec.t = result.t;
            rec.u = result.u;
            rec.v = result.v;
            rec.mat = self.mat.clone();
            rec.set_face_normal(ray, &result.outward_normal);
            rec.p = result.intersection_point;

            true
        } else {
            false
        }
    }

    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
}

pub struct TriangleIntersection {
    u: f32,
    v: f32,
    t: f32,
    // WARN: Remove
    intersection_point: Point3<f32>,
    outward_normal: Vector3<f32>,
}

#[inline(always)]
pub fn ray_triangle_rcp(x: f32) -> f32 {
    unsafe {
        // avx2

        let a = _mm_set_ss(x);
        return _mm_cvtss_f32(_mm_rcp_ss(a));
        // let r = _mm_rcp_ss(a);
        //
        // return _mm_cvtss_f32(_mm_mul_ss(r, _mm_fnmadd_ss(r, a, _mm_set_ss(2.0f32))))
    }
}

// NOTE: Blender cycles intersect
//
// https://github.com/blender/cycles/blob/main/src/util/math_intersect.h#L160
pub fn intersect_triangle(
    ray: &Ray,
    ray_t: Interval,
    tri_a: Vector3<f32>,
    tri_b: Vector3<f32>,
    tri_c: Vector3<f32>,
) -> Option<TriangleIntersection> {
    let ray_p = &ray.origin().coords;
    let ray_d = ray.direction();

    // Vertices relative to ray origin.
    let v0 = tri_a - ray_p;
    let v1 = tri_b - ray_p;
    let v2 = tri_c - ray_p;

    // Triangle edges.
    let e0 = v2 - v0;
    let e1 = v0 - v1;
    let e2 = v1 - v2;

    // Perfrom edge tests.
    let u = e0.cross(&(v2 + v0)).dot(ray_d);
    let v = e1.cross(&(v0 + v1)).dot(ray_d);
    let w = e2.cross(&(v1 + v2)).dot(ray_d);

    let uvw = u + v + w;
    let eps = EPSILON * uvw.abs();
    let min_uvw = u.min(v.min(w));
    let max_uvw = u.max(v.max(w));

    if !(min_uvw >= -eps || max_uvw <= eps) {
        return None;
    }

    // NOTE: Convert to precomputed normals
    // Calculate geometry normal and denominator.
    let ng1 = e1.cross(&e0);
    let ng = ng1 + ng1;
    let den = ng.dot(ray_d);
    // avoid division by 0.
    if std::intrinsics::unlikely(den == 0.0f32) {
        return None;
    }

    // if den == 0.0f32 {
    //     return None;
    // }

    // Perform depth test.
    let t = v0.dot(&ng) / den;
    if !(t >= ray_t.min && t <= ray_t.max) {
        return None;
    }

    let rcp_uvw = if uvw.abs() < 1e-18f32 {
        0.0f32
    } else {
        1f32 / uvw
        // ray_triangle_rcp(uvw)
    };

    // NOTE: Remove
    let intersection_point = ray.origin() + ray.direction() * t;
    let outward_normal = ng.normalize();

    Some(TriangleIntersection {
        u: 1.0f32.min(u * rcp_uvw),
        v: 1.0f32.min(v * rcp_uvw),
        t,
        intersection_point,
        outward_normal,
    })
}

//
// impl Triangle {
//     pub fn new(mesh: &TriangleMesh, mesh_i: i32, tri_i: i32) -> Self {
//         let v = &mesh.vertex_i[(tri_i as usize * 3)..(tri_i as usize * 3 + 3)];
//         let p0 = mesh.p[v[0] as usize];
//         let p1 = mesh.p[v[1] as usize];
//         let p2 = mesh.p[v[2] as usize];
//         let aabb = AABB::new(p0, p1).union(p2);
//
//         Self {
//             mesh_i,
//             tri_i,
//             aabb,
//         }
//     }
// }
//
// impl Hittable for Triangle {
//     fn hit(&self, ray: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
//         todo!()
//     }
//
//     fn bounding_box(&self) -> &AABB {
//         &self.aabb
//     }
// }

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, Vector3};

    use super::*;

    #[test]
    fn test_triangle_intersect_hit() {
        let tri_a = Vector3::new(0.0, 0.0, 0.0);
        let tri_b = Vector3::new(1.0, 0.0, 0.0);
        let tri_c = Vector3::new(0.0, 1.0, 0.0);

        // Define a ray that intersects the triangle
        let ray_origin = Point3::new(0.25, 0.25, -1.0);
        let ray_direction = Vector3::new(0.0, 0.0, 1.0);
        let ray = Ray::new(ray_origin, ray_direction.normalize());

        let ray_t = Interval {
            min: 0.0,
            max: f32::INFINITY,
        };

        let intersection = intersect_triangle(&ray, ray_t, tri_a, tri_b, tri_c);

        assert!(intersection.is_some());

        let intersection = intersection.unwrap();

        // Check that t is correct (the ray should hit the triangle at t = 1.0)
        assert!((intersection.t - 1.0).abs() < 1e-6);

        // Check that u and v are correct
        // For the point (0.25, 0.25, 0.0) in the triangle, the barycentric coordinates should be u = v = 0.25
        assert!((intersection.u - 0.25).abs() < 1e-6);
        assert!((intersection.v - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_triangle_intersect_miss() {
        let tri_a = Vector3::new(0.0, 0.0, 0.0);
        let tri_b = Vector3::new(1.0, 0.0, 0.0);
        let tri_c = Vector3::new(0.0, 1.0, 0.0);

        let ray_origin = Point3::new(1.5, 1.5, -1.0);
        let ray_direction = Vector3::new(0.0, 0.0, 1.0);
        let ray = Ray::new(ray_origin, ray_direction.normalize());

        let ray_t = Interval {
            min: 0.0,
            max: f32::INFINITY,
        };

        let intersection = intersect_triangle(&ray, ray_t, tri_a, tri_b, tri_c);

        assert!(intersection.is_none());
    }

    #[test]
    fn test_triangle_intersect_edge() {
        let tri_a = Vector3::new(0.0, 0.0, 0.0);
        let tri_b = Vector3::new(1.0, 0.0, 0.0);
        let tri_c = Vector3::new(0.0, 1.0, 0.0);

        // Define a ray that intersects exactly at the edge of the triangle
        let ray_origin = Point3::new(0.0, 0.5, -1.0);
        let ray_direction = Vector3::new(0.0, 0.0, 1.0);
        let ray = Ray::new(ray_origin, ray_direction.normalize());

        let ray_t = Interval {
            min: 0.0,
            max: f32::INFINITY,
        };

        let intersection = intersect_triangle(&ray, ray_t, tri_a, tri_b, tri_c);

        assert!(intersection.is_some());

        let intersection = intersection.unwrap();

        // Check that t is correct
        assert!((intersection.t - 1.0).abs() < 1e-6);

        // The point lies on the edge between tri_a and tri_c
        assert!((intersection.u - 0.0).abs() < 1e-6);
        assert!((intersection.v - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_triangle_intersect_behind_ray() {
        let tri_a = Vector3::new(0.0, 0.0, 0.0);
        let tri_b = Vector3::new(1.0, 0.0, 0.0);
        let tri_c = Vector3::new(0.0, 1.0, 0.0);

        // Define a ray that points away from the triangle
        let ray_origin = Point3::new(0.25, 0.25, 1.0);
        let ray_direction = Vector3::new(0.0, 0.0, 1.0); // Pointing in +Z direction
        let ray = Ray::new(ray_origin, ray_direction.normalize());

        let ray_t = Interval {
            min: 0.0,
            max: f32::INFINITY,
        };

        let intersection = intersect_triangle(&ray, ray_t, tri_a, tri_b, tri_c);

        // The triangle is behind the ray origin; there should be no intersection in the positive t direction
        assert!(intersection.is_none());
    }

    #[test]
    fn test_triangle_intersect_parallel_ray() {
        let tri_a = Vector3::new(0.0, 0.0, 0.0);
        let tri_b = Vector3::new(1.0, 0.0, 0.0);
        let tri_c = Vector3::new(0.0, 1.0, 0.0);

        // Define a ray that is parallel to the triangle plane
        let ray_origin = Point3::new(0.25, 0.25, 1.0);
        let ray_direction = Vector3::new(1.0, 0.0, 0.0); // Parallel to XY plane
        let ray = Ray::new(ray_origin, ray_direction.normalize());

        let ray_t = Interval {
            min: 0.0,
            max: f32::INFINITY,
        };

        let intersection = intersect_triangle(&ray, ray_t, tri_a, tri_b, tri_c);

        // The ray is parallel to the triangle plane; there should be no intersection
        assert!(intersection.is_none());
    }
}

// NOTE: Intersect from pbrt
//
// https://pbr-book.org/4ed/Shapes/Triangle_Meshes#RayndashTriangleIntersection
// PBRT "Unlike the other shapes so far, pbrt provides a stand-alone triangle intersection
// function that takes a ray and the three triangle vertices directly. Having this
// functionality available without needing to instantiate both a Triangle and a
// TriangleMesh in order to do a ray–triangle intersection test is helpful in a few
// other parts of the system"
// pub fn intersect_triangle(
//     ray: &Ray,
//     ray_t: f32,
//     p0: Point3<f32>,
//     p1: Point3<f32>,
//     p2: Point3<f32>,
// ) -> Option<TriangleIntersection> {
//     // Return no intersection if triangle is degenerate
//     if (p2 - p0).cross(&(p1 - p0)).norm_squared() == 0. {
//         return None;
//     }
//
//     // Transform triangle vertices to ray coordinate space
//     // Translate vertices based on ray origin
//     let mut p0t = p0 - ray.origin();
//     let mut p1t = p1 - ray.origin();
//     let mut p2t = p2 - ray.origin();
//
//     // Permute components of triagnle vertices and ray direction
//     let mut kz = max_component_index(ray.direction().abs());
//     let mut kx = kz + 1;
//     if kx == 3 { kx = 0 }
//     let mut ky = kx + 1;
//     if ky == 3 { ky = 0 }
//     let d = permute(*ray.direction(), (kx, ky, kz));
//     p0t = permute(p0t, (kx, ky, kz));
//     p1t = permute(p1t, (kx, ky, kz));
//     p2t = permute(p2t, (kx, ky, kz));
//
//     // Apply shear tranformation to translated vertex positions
//     let sx = -d.x / d.z;
//     let sy = -d.y / d.z;
//     let sz = d.z.recip();
//     p0t.x += sx * p0t.z;
//     p0t.y += sy * p0t.z;
//     p1t.x += sx * p1t.z;
//     p1t.y += sy * p1t.z;
//     p2t.x += sx * p2t.z;
//     p2t.y += sy * p2t.z;
//
//     // Compute edge funciton coefficients _e0_, _e1_, and _e2_
//     let mut e0 = difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
//     let mut e1 = difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
//     let mut e2 = difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);
//
//     // fall back to double precision test at triangle edges
//     // Test if it is worth
//     if e0 == 0. || e1 == 0. || e2 == 0. {
//         let p2txp1ty: f64 = p2t.x as f64 * p1t.y as f64;
//         let p2typ1tx: f64 = p2t.y as f64 * p1t.x as f64;
//         e0 = (p2typ1tx - p2txp1ty) as f32;
//         let p0txp2ty: f64 = p0t.x as f64 * p2t.y as f64;
//         let p0typ2tx: f64 = p0t.y as f64 * p2t.x as f64;
//         e0 = (p0typ2tx - p0txp2ty) as f32;
//         let p1txp0ty: f64 = p1t.x as f64 * p0t.y as f64;
//         let p1typ0tx: f64 = p1t.y as f64 * p0t.x as f64;
//         e0 = (p1typ0tx - p1txp0ty) as f32;
//     }
//
//     // Perform triagnle edge and determinant tests
//     if (e0 < 0. || e1 < 0. || e2 < 0.) && (e0 > 0. || e1 > 0. || e2> 0.) {
//         return None
//     }
//     let det = e0 + e1 + e2;
//     if (det == 0.) {
//         return None
//     }
//
//     // Compute scaled hit distance to triagnle adn test against ray t range
//     p0t.z *= sz;
//     p1t.z *= sz;
//     p2t.z *= sz;
//     let tscaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
//     if (det < 0. && (tscaled >= 0. || tscaled < ray_t * det)) {
//         return None;
//     } else if (det > 0. && (tscaled <= 0. || tscaled > ray_t * det)) {
//         return None;
//     }
//
//     // Compute barycentric coordinates and t value for triangle intersection
//     let invdet = det.recip();
//     let b0 = e0 * invdet;
//     let b1 = e1 * invdet;
//     let b2 = e2 * invdet;
//     let t = tscaled * invdet;
//     assert!(!t.is_nan());
//
//     // Ensure that computed triangle t is conservatively greater than zero
//     // Compute delta_z term for triangle t error bounds
//     let maxzt = max_component_index(Vector3::new(p0t.z, p1t.z, p2t.z).abs()) as f32;
//     let deltaz = 3.0f32.gamma() * maxzt;
//
//     // Compute delta_x an delta_y terms for triangle t error bounrs
//     let maxxt = max_component_index(Vector3::new(p0t.x, p1t.x, p2t.x).abs()) as f32;
//     let maxyt = max_component_index(Vector3::new(p0t.y, p1t.y, p2t.y).abs()) as f32;
//     let deltax = 5.0f32.gamma() * (maxxt + maxzt);
//     let deltay = 5.0f32.gamma() * (maxyt + maxzt);
//
//     // Compute delta_e term for triangels t error bounds
//     let deltae = 2. * (2.0f32.gamma() * maxxt * maxyt + deltay * maxxt + deltax * maxyt);
//
//     // COmpute delta_t term for triagnle t error bounds and check _t_
//     let maxe = max_component_index(Vector3::new(e0, e1, e2).abs()) as f32;
//     let deltat = 3. * (3.0f32.gamma() * maxe * maxzt + deltae * maxzt + deltaz * maxe) * invdet.abs();
//     if (t <= deltat) {
//         return None;
//     }
//
//     // barycentric coordinates and t
//     return TriangleIntersection {b0, b1, b2, t};
// }
//
// #[inline]
// fn max_component_index(v: Vector3<f32>) -> usize {
//     if v.x >= v.y && v.x >= v.z {
//         0
//     } else if v.y >= v.x && v.y >= v.z {
//         1
//     } else {
//         2
//     }
// }
//
// #[inline]
// fn permute(t: Vector3<f32>, p: (usize, usize, usize)) -> Vector3<f32> {
//     return Vector3::new(t[p.0], t[p.1], t[p.2])
// }
//
// #[inline]
// fn difference_of_products(a: f32, b: f32, c: f32, d:f32) -> f32 {
//     let cd = c * d;
//     let difference = a.mul_add(b, -cd);
//     let error = (-c).mul_add(d, cd);
//     difference + error
// }
