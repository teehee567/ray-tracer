use image::{RgbImage, RgbaImage};
use nalgebra::Vector3;

use super::super::aabb::AABB;
use crate::core::camera::Camera;
use crate::core::hittable::{HitRecord, Hittable};
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::geometry::objects::mesh::{Primitive, TriangleIntersection};
use crate::geometry::objects::triangle::{Triangle, Vertex};
use crate::geometry::wireframe::WireFrame;
use crate::utils::colour::Colour;

// pub struct BVH2Triangle {
//     v0: Vector3<f32>,
//     v1: Vector3<f32>,
//     v2: Vector3<f32>,
//     centroid: Vector3<f32>,
// }
//
// pub enum BVH2Node {
//     Leaf {
//         parent_i: usize,
//         shape_i: usize,
//     },
//     Node {
//         parent_i: usize,
//
//         child_l_i: usize,
//         child_l_aabb: AABB,
//
//         child_r_i: usize,
//         child_r_aabb: AABB,
//     },
// }
//
// pub struct BVH2<'a, Prim: Primitive<f32>> {
//     nodes: Vec<BVH2Node>,
//     primitives: &'a [Prim],
//     bboxs: &'a [AABB],
// }

// WARN: Currently Binned SAH
// impl<'a, Prim: Primitive<f32>> BVH2<'a, Prim> {
//     pub fn build(shapes: &'a [Prim], vertices: &'a [Vertex]) -> Self {
//         let triangle_centroids: Vec<Vector3<f32>> = shapes
//             .iter()
//             .map(|shape| shape.centroid(vertices))
//             .collect();
//
//         // let nodes = Vec::with_capacity(2 * shapes.len() - 1);
//
//         todo!()
//     }
// }

use core::fmt;
use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Range, Sub};

// --- SAH Configuration ---
#[derive(Clone, Copy)]
pub struct SAH {
    traversal_cost: f32,
    intersection_cost: f32,
}

impl SAH {
    pub fn new(traversal_cost: f32, intersection_cost: f32) -> Self {
        SAH {
            traversal_cost,
            intersection_cost,
        }
    }

    pub fn leaf_cost(&self, prim_count: usize, bbox: &AABB) -> f32 {
        self.intersection_cost * prim_count as f32 * bbox.surface_area()
    }

    pub fn node_cost(&self, bbox: &AABB) -> f32 {
        self.traversal_cost * bbox.surface_area()
    }
}

// --- BVH Node Definition ---
#[derive(Default)]
pub struct BVHNode {
    pub bbox: AABB,
    pub left: Option<usize>,
    pub right: Option<usize>,
    pub begin: usize,
    pub end: usize,
}

// --- Bin and Split Definitions ---
#[derive(Copy, Clone)]
struct Bin {
    bbox: AABB,
    prim_count: usize,
}

impl Bin {
    fn new() -> Self {
        Bin {
            bbox: AABB::default(),
            prim_count: 0,
        }
    }

    fn add(&mut self, bbox: &AABB, prim_count: usize) {
        self.bbox = AABB::combine(&self.bbox, bbox);
        self.prim_count += prim_count;
    }

    fn add_bin(&mut self, bin: &Bin) {
        self.add(&bin.bbox, bin.prim_count);
    }
}

#[derive(Copy, Clone)]
struct Split {
    bin_id: usize,
    cost: f32,
    axis: usize,
}

// --- Binned SAH Builder ---
pub struct BinnedSahBuilder<'a, const BIN_COUNT: usize> {
    bboxes: &'a [AABB],
    centers: &'a [Vector3<f32>],
    prim_ids: Vec<usize>,
    sah: SAH,
    max_leaf_size: usize,
    phantom: PhantomData<f32>,
}

impl<'a, const BIN_COUNT: usize> BinnedSahBuilder<'a, BIN_COUNT> {
    pub fn new(
        bboxes: &'a [AABB],
        centers: &'a [Vector3<f32>],
        sah: SAH,
        max_leaf_size: usize,
    ) -> Self {
        let prim_ids = (0..bboxes.len()).collect();
        BinnedSahBuilder {
            bboxes,
            centers,
            prim_ids,
            sah,
            max_leaf_size,
            phantom: PhantomData,
        }
    }

    pub fn build(&mut self) -> Vec<BVHNode> {
        let mut nodes = Vec::new();
        self.build_node(0, self.bboxes.len(), &mut nodes);
        nodes
    }

    fn build_node(&mut self, begin: usize, end: usize, nodes: &mut Vec<BVHNode>) -> usize {
        let node_bbox = self.compute_bbox(begin, end);

        let node_index = nodes.len();
        nodes.push(BVHNode::default());

        if let Some(split_index) = self.try_split(&node_bbox, begin, end) {
            let left_child = self.build_node(begin, split_index, nodes);
            let right_child = self.build_node(split_index, end, nodes);

            nodes[node_index] = BVHNode {
                bbox: node_bbox,
                left: Some(left_child),
                right: Some(right_child),
                begin,
                end,
            };
        } else {
            nodes[node_index] = BVHNode {
                bbox: node_bbox,
                left: None,
                right: None,
                begin,
                end,
            };
        }
        node_index
    }

    fn compute_bbox(&self, begin: usize, end: usize) -> AABB {
        let mut bbox = AABB::default();
        for &prim_id in &self.prim_ids[begin..end] {
            bbox = AABB::combine(&bbox, &self.bboxes[prim_id]);
        }
        bbox
    }

    fn try_split(&mut self, bbox: &AABB, begin: usize, end: usize) -> Option<usize> {
        let mut per_axis_bins = [[Bin::new(); BIN_COUNT]; 3];

        self.fill_bins(&mut per_axis_bins, bbox, begin, end);

        let largest_axis = bbox.largest_axis();
        let mut best_split = Split {
            bin_id: BIN_COUNT / 2,
            cost: f32::MAX,
            axis: largest_axis,
        };

        for axis in 0..3 {
            self.find_best_split(axis, &per_axis_bins[axis], &mut best_split);
        }

        let leaf_cost = self.sah.leaf_cost(end - begin, bbox);
        if best_split.cost >= leaf_cost {
            if end - begin <= self.max_leaf_size {
                return None;
            }
            return Some(self.fallback_split(largest_axis, begin, end));
        }

        let split_pos = bbox.min[best_split.axis]
            + (bbox.max[best_split.axis] - bbox.min[best_split.axis])
                * f32::from(best_split.bin_id as f32)
                / f32::from(BIN_COUNT as f32);

        let mid = self.partition_primitives(best_split.axis, split_pos, begin, end);
        if mid == begin || mid == end {
            return Some(self.fallback_split(largest_axis, begin, end));
        }

        Some(mid)
    }

    fn fill_bins(
        &self,
        per_axis_bins: &mut [[Bin; BIN_COUNT]; 3],
        bbox: &AABB,
        begin: usize,
        end: usize,
    ) {
        let bin_scale = Vector3::new(
            f32::from(BIN_COUNT as f32),
            f32::from(BIN_COUNT as f32),
            f32::from(BIN_COUNT as f32),
        )
        .component_div(&bbox.diagonal());
        let bin_offset = bin_scale.component_mul(&bbox.min.coords) * f32::from(-1.0);

        for &prim_id in &self.prim_ids[begin..end] {
            let center = self.centers[prim_id];
            let pos = center.component_mul(&bin_scale) + bin_offset;

            for axis in 0..3 {
                let index = pos[axis]
                    .max(0.0f32)
                    .min(f32::from(BIN_COUNT as f32) - 1.0f32) as usize;
                per_axis_bins[axis][index].add(&self.bboxes[prim_id], 1);
            }
        }
    }

    fn find_best_split(&self, axis: usize, bins: &[Bin; BIN_COUNT], best_split: &mut Split) {
        let mut right_accum = Bin::new();
        let mut right_costs = [0.0f32; BIN_COUNT];

        for i in (1..BIN_COUNT).rev() {
            right_accum.add_bin(&bins[i]);
            right_costs[i] = self
                .sah
                .leaf_cost(right_accum.prim_count, &right_accum.bbox);
        }

        let mut left_accum = Bin::new();
        for i in 0..(BIN_COUNT - 1) {
            left_accum.add_bin(&bins[i]);
            let cost =
                self.sah.leaf_cost(left_accum.prim_count, &left_accum.bbox) + right_costs[i + 1];
            if cost < best_split.cost {
                best_split.bin_id = i + 1;
                best_split.cost = cost;
                best_split.axis = axis;
            }
        }
    }

    fn fallback_split(&mut self, axis: usize, begin: usize, end: usize) -> usize {
        let mid = (begin + end) / 2;
        self.prim_ids[begin..end].sort_by(|&a, &b| {
            self.centers[a][axis]
                .partial_cmp(&self.centers[b][axis])
                .unwrap_or(Ordering::Equal)
        });
        mid
    }

    fn partition_primitives(
        &mut self,
        axis: usize,
        split_pos: f32,
        begin: usize,
        end: usize,
    ) -> usize {
        let prim_ids_slice = &mut self.prim_ids[begin..end];
        let mut left = 0;
        let mut right = prim_ids_slice.len();

        while left < right {
            if self.centers[prim_ids_slice[left]][axis] < split_pos {
                left += 1;
            } else {
                right -= 1;
                prim_ids_slice.swap(left, right);
            }
        }
        begin + left
    }
}

// --- BVH Definition ---
pub struct BVH<Prim: Primitive<f32>> {
    pub nodes: Vec<BVHNode>,
    sah: SAH,
    phantom: PhantomData<Prim>
}

impl<Prim: Primitive<f32>> BVH<Prim> {
    pub fn build(
        primitives: &[Prim],
        vertices: &[Vertex],
        sah: SAH,
        max_leaf_size: usize,
    ) -> Self {
        let bboxes: Vec<AABB> = primitives.iter().map(|prim| prim.aabb(vertices)).collect();
        let centers: Vec<Vector3<f32>> = primitives
            .iter()
            .map(|prim| prim.centroid(vertices))
            .collect();

        let mut builder = BinnedSahBuilder::<8>::new(&bboxes, &centers, sah, max_leaf_size);
        let nodes = builder.build();

        BVH {
            nodes,
            sah,
            phantom: PhantomData,
        }
    }

    pub fn intersect(&self, ray: &Ray, ray_t: Interval, primitives: &[Prim], vertices: &[Vertex]) -> Option<TriangleIntersection> {
        if self.nodes.is_empty() {
            None
        } else {
            self.intersect_recursive(ray, ray_t, 0, primitives, vertices)
        }
    }

    pub fn intersect_recursive(
        &self,
        ray: &Ray,
        ray_t: Interval,
        node_index: usize,
        primitives: &[Prim],
        vertices: &[Vertex]
    ) -> Option<TriangleIntersection> {
        let node = &self.nodes[node_index];

        // Step 1: Test if the ray intersects the current node's bounding box
        if !node.bbox.intersect(ray) {
            return None;
        }

        // Step 2: Check if it is a leaf node or an internal node
        let mut result: Option<TriangleIntersection> = None;
        let mut closest_so_far = ray_t.max;

        if node.left.is_none() && node.right.is_none() {
            // Leaf node: Check intersection with the contained primitives
            for prim_id in node.begin..node.end {
                let primitive = &primitives[prim_id];
                if let Some(intersect) = primitive.intersect_prim(
                    ray,
                    vertices,
                    Interval::new(ray_t.min, closest_so_far),
                ) {
                    closest_so_far = intersect.t;
                    result = Some(intersect);
                }
            }
        } else {
            // Internal node: Traverse children
            if let Some(left_index) = node.left {
                if let Some(intersect) = self.intersect_recursive(
                    ray,
                    Interval::new(ray_t.min, closest_so_far),
                    left_index,
                    primitives,
                    vertices
                ) {
                    if intersect.t < closest_so_far {
                        closest_so_far = intersect.t;
                        result = Some(intersect);
                    }
                }
            }

            if let Some(right_index) = node.right {
                if let Some(intersect) = self.intersect_recursive(
                    ray,
                    Interval::new(ray_t.min, closest_so_far),
                    right_index,
                    primitives,
                    vertices,
                ) {
                    if intersect.t < closest_so_far {
                        closest_so_far = intersect.t;
                        result = Some(intersect);
                    }
                }
            }
        }

        result
    }

    // pub fn intersect(&self, ray: &Ray, ray_max: f32) -> Option<(f32, usize)> {
    //     if self.nodes.is_empty() {
    //         return None;
    //     }
    //
    //     let mut stack = vec![0usize]; // Start with the root node
    //     let mut closest_intersection = None;
    //
    //     while let Some(node_index) = stack.pop() {
    //         let node = &self.nodes[node_index];
    //
    //         if !node.bbox.intersect(ray) {
    //             continue;
    //         }
    //
    //         if node.left.is_none() && node.right.is_none() {
    //             // Leaf node
    //             for &prim_id in &self.primitives[node.begin..node.end] {
    //                 if let Some(distance) = self.verticies[prim_id].intersect(ray) {
    //                     if distance >= 0.0
    //                         && (closest_intersection.is_none()
    //                             || distance < closest_intersection.as_ref().unwrap().distance)
    //                     {
    //                         closest_intersection = Some(Intersection {
    //                             distance,
    //                             primitive_index: prim_id,
    //                         });
    //                     }
    //                 }
    //             }
    //         } else {
    //             // Inner node
    //             if let Some(left_index) = node.left {
    //                 stack.push(left_index);
    //             }
    //             if let Some(right_index) = node.right {
    //                 stack.push(right_index);
    //             }
    //         }
    //     }
    //
    //     closest_intersection
    // }
}

pub struct DebugInfo {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub inner_nodes: usize,
    pub min_depth: usize,
    pub max_depth: usize,
    pub total_depth: usize,
    pub min_sah_cost: f32,
    pub max_sah_cost: f32,
    pub total_sah_cost: f32,
}

impl DebugInfo {
    pub fn average_depth(&self) -> f32 {
        self.total_depth as f32 / self.total_nodes as f32
    }

    pub fn average_sah_cost(&self) -> f32 {
        self.total_sah_cost / self.total_nodes as f32
    }
}

impl<Prim: Primitive<f32>> BVH<Prim> {
    pub fn compute_debug_info(&self) -> DebugInfo {
        let mut info = DebugInfo {
            total_nodes: 0,
            leaf_nodes: 0,
            inner_nodes: 0,
            min_depth: usize::MAX,
            max_depth: 0,
            total_depth: 0,
            min_sah_cost: f32::MAX,
            max_sah_cost: f32::MIN,
            total_sah_cost: 0.0,
        };

        if !self.nodes.is_empty() {
            self.traverse_node(0, 0, &self.sah, &mut info);
        }

        info
    }

    fn traverse_node(&self, node_index: usize, depth: usize, sah: &SAH, info: &mut DebugInfo) {
        info.total_nodes += 1;
        info.total_depth += depth;
        info.min_depth = info.min_depth.min(depth);
        info.max_depth = info.max_depth.max(depth);

        let node = &self.nodes[node_index];

        if node.left.is_none() && node.right.is_none() {
            // Leaf node
            info.leaf_nodes += 1;
            let prim_count = node.end - node.begin;
            let leaf_sah_cost = sah.leaf_cost(prim_count, &node.bbox);
            info.total_sah_cost += leaf_sah_cost;
            info.min_sah_cost = info.min_sah_cost.min(leaf_sah_cost);
            info.max_sah_cost = info.max_sah_cost.max(leaf_sah_cost);
        } else {
            // Inner node
            info.inner_nodes += 1;
            let node_sah_cost = sah.node_cost(&node.bbox);
            info.total_sah_cost += node_sah_cost;
            info.min_sah_cost = info.min_sah_cost.min(node_sah_cost);
            info.max_sah_cost = info.max_sah_cost.max(node_sah_cost);

            if let Some(left_index) = node.left {
                self.traverse_node(left_index, depth + 1, sah, info);
            }
            if let Some(right_index) = node.right {
                self.traverse_node(right_index, depth + 1, sah, info);
            }
        }
    }

    pub fn traverse_and_draw_wireframe(
        &self,
        node_index: usize,
        depth: usize,
        depth_range: Range<usize>,
        img: &mut RgbaImage,
        colour: Colour,
        camera: &Camera,
    ) {

        let node = &self.nodes[node_index];

        if depth_range.contains(&depth) {
            node.bbox.draw_wireframe(img, colour, camera);
        }

        if depth < depth_range.end {
            if let Some(left_index) = node.left {
                self.traverse_and_draw_wireframe(left_index, depth + 1, depth_range.clone(), img, colour, camera);
            }
            if let Some(right_index) = node.right {
                self.traverse_and_draw_wireframe(right_index, depth + 1, depth_range.clone(), img, colour, camera);
            }
        }
    }
}

impl fmt::Display for DebugInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Debug Info:\n\
            Total Nodes: {}\n\
            Leaf Nodes: {}\n\
            Inner Nodes: {}\n\
            Min Depth: {}\n\
            Max Depth: {}\n\
            Total Depth: {}\n\
            Average Depth: {:.2}\n\
            Min SAH Cost: {:.2}\n\
            Max SAH Cost: {:.2}\n\
            Total SAH Cost: {:.2}\n\
            Average SAH Cost: {:.2}",
            self.total_nodes,
            self.leaf_nodes,
            self.inner_nodes,
            self.min_depth,
            self.max_depth,
            self.total_depth,
            self.average_depth(),
            self.min_sah_cost,
            self.max_sah_cost,
            self.total_sah_cost,
            self.average_sah_cost()
        )
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, Vector3};

    use super::{AABB, *};

    #[derive(Default)]
    struct MockPrimitive {
        bbox: AABB,
    }

    impl MockPrimitive {
        fn new(bbox: AABB) -> Self {
            Self { bbox }
        }
    }

    trait Primitive<T> {
        fn aabb(&self, vertices: &[Vector3<f32>]) -> AABB;
        fn centroid(&self, vertices: &[Vector3<f32>]) -> Vector3<f32>;
    }

    impl Primitive<f32> for MockPrimitive {
        fn aabb(&self, _vertices: &[Vector3<f32>]) -> AABB {
            self.bbox
        }

        fn centroid(&self, _vertices: &[Vector3<f32>]) -> Vector3<f32> {
            (self.bbox.min.coords + self.bbox.max.coords) / 2.0
        }
    }

    #[test]
    fn test_sah_leaf_and_node_cost() {
        let sah = SAH::new(1.0, 2.0);
        let bbox = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let leaf_cost = sah.leaf_cost(4, &bbox);
        let node_cost = sah.node_cost(&bbox);

        assert_eq!(leaf_cost, 2.0 * 4.0 * 6.0);
        assert_eq!(node_cost, 1.0 * 6.0);
    }

    #[test]
    fn test_binned_sah_builder_build() {
        let primitives = vec![
            MockPrimitive::new(AABB::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 1.0),
            )),
            MockPrimitive::new(AABB::new(
                Point3::new(1.0, 1.0, 1.0),
                Point3::new(2.0, 2.0, 2.0),
            )),
            MockPrimitive::new(AABB::new(
                Point3::new(2.0, 2.0, 2.0),
                Point3::new(3.0, 3.0, 3.0),
            )),
        ];

        let vertices = vec![];
        let sah = SAH::new(1.0, 2.0);
        let max_leaf_size = 1;

        let bboxes: Vec<AABB> = primitives.iter().map(|p| p.aabb(&vertices)).collect();
        let centers: Vec<Vector3<f32>> = primitives.iter().map(|p| p.centroid(&vertices)).collect();

        let mut builder = BinnedSahBuilder::<8>::new(&bboxes, &centers, sah, max_leaf_size);
        let nodes = builder.build();

        // Ensure that the nodes vector is not empty after building.
        assert!(!nodes.is_empty());

        // Check the root node
        let root = &nodes[0];
        assert!(root.left.is_some() && root.right.is_some());
        assert_eq!(root.begin, 0);
        assert_eq!(root.end, primitives.len());
    }

    #[test]
    fn test_fallback_split() {
        let primitives = vec![
            MockPrimitive::new(AABB::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 1.0),
            )),
            MockPrimitive::new(AABB::new(
                Point3::new(1.0, 1.0, 1.0),
                Point3::new(2.0, 2.0, 2.0),
            )),
        ];

        let vertices = vec![];
        let sah = SAH::new(1.0, 2.0);
        let max_leaf_size = 1;

        let bboxes: Vec<AABB> = primitives.iter().map(|p| p.aabb(&vertices)).collect();
        let centers: Vec<Vector3<f32>> = primitives.iter().map(|p| p.centroid(&vertices)).collect();

        let mut builder = BinnedSahBuilder::<8>::new(&bboxes, &centers, sah, max_leaf_size);
        let split_index = builder.fallback_split(0, 0, primitives.len());

        assert_eq!(split_index, 1);
    }

    #[test]
    fn test_partition_primitives() {
        let primitives = vec![
            MockPrimitive::new(AABB::new(
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 1.0),
            )),
            MockPrimitive::new(AABB::new(
                Point3::new(1.0, 1.0, 1.0),
                Point3::new(2.0, 2.0, 2.0),
            )),
            MockPrimitive::new(AABB::new(
                Point3::new(2.0, 2.0, 2.0),
                Point3::new(3.0, 3.0, 3.0),
            )),
        ];

        let vertices = vec![];
        let sah = SAH::new(1.0, 2.0);
        let max_leaf_size = 1;

        let bboxes: Vec<AABB> = primitives.iter().map(|p| p.aabb(&vertices)).collect();
        let centers: Vec<Vector3<f32>> = primitives.iter().map(|p| p.centroid(&vertices)).collect();

        let mut builder = BinnedSahBuilder::<8>::new(&bboxes, &centers, sah, max_leaf_size);
        let split_pos = 1.5;
        let axis = 0;

        let mid = builder.partition_primitives(axis, split_pos, 0, primitives.len());

        assert!(mid > 0 && mid < primitives.len());
    }

}
