use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;

use image::{RgbImage, RgbaImage};
use nalgebra::Vector3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::accelerators::aabb::AABB;
use crate::core::camera::Camera;
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::geometry::objects::mesh::{Primitive, TriangleIntersection};
use crate::geometry::objects::triangle::Vertex;
use crate::geometry::wireframe::WireFrame;
use crate::utils::colour::Colour;

#[derive(Copy, Clone)]
struct SahBVH2Node {
    bbox: AABB,
    left_i: Option<usize>,
    right_i: Option<usize>,
    first_prim: usize,
    prim_count: usize,
}

impl SahBVH2Node {
    pub fn new(
        bbox: AABB,
        left_i: Option<usize>,
        right_i: Option<usize>,
        first_prim: usize,
        prim_count: usize,
    ) -> Self {
        Self {
            bbox,
            left_i,
            right_i,
            first_prim,
            prim_count,
        }
    }
}

impl SahBVH2Node {
    fn is_leaf(&self) -> bool {
        return self.prim_count > 0;
    }
}

pub struct SahBVH2<Prim: Primitive<f32>> {
    nodes: Vec<SahBVH2Node>,
    phantom: PhantomData<Prim>
}

pub struct SahBVH2Builder<'a, Prim: Primitive<f32>> {
    bboxs: &'a [AABB],
    // centers: &'a [Vector3<f32>],
    primitives: &'a mut [Prim],
    vertices: &'a [Vertex],
    root_node_id: usize,
    nodes: Vec<SahBVH2Node>,
}

impl<'a, Prim: Primitive<f32>> SahBVH2Builder<'a, Prim> {
    pub fn new(bboxs: &'a [AABB], primitives: &'a mut [Prim], vertices: &'a [Vertex]) -> Self {
        let nodes: Vec<SahBVH2Node> = Vec::with_capacity(primitives.len() * 2 - 1);
        Self {
            bboxs,
            primitives,
            vertices,
            root_node_id: 0,
            nodes
        }
    }

    pub fn build(mut self) -> SahBVH2<Prim> {
        let amount_primitives = self.primitives.len();

        self.nodes.push(SahBVH2Node::new(
            self.node_bounds(0..amount_primitives),
            Some(0),
            Some(0),
            0,
            amount_primitives,
        ));

        self.subdivide_sah(0);
        println!("dead");

        SahBVH2 {
            nodes: self.nodes,
            phantom: PhantomData
        }
    }

    fn node_bounds(&self, range: Range<usize>) -> AABB {
        let mut aabb = AABB::default();
        for primitive in &self.primitives[range] {
            aabb.grow_bb_mut(&primitive.aabb(&self.vertices))
        }

        aabb
    }

    fn evaluate_sah(&self, node: &SahBVH2Node, axis: usize, pos: f32) -> f32 {
        let mut left_aabb = AABB::default();
        let mut right_aabb = AABB::default();
        let mut left_count = 0;
        let mut right_count = 0;
        for i in 0..node.prim_count {
            let prim = self.primitives[node.first_prim + i].centroid(self.vertices);
            if prim[axis] < pos {
                left_count += 1;
                left_aabb = left_aabb.grow(prim.into());
            } else {
                right_count += 1;
                right_aabb = right_aabb.grow(prim.into());
            }
        }

        let cost = left_count as f32 * left_aabb.half_area() + right_count as f32 * right_aabb.half_area();
        if cost > 0.0f32 {
            cost
        } else {
            f32::MAX
        }
    }

    fn subdivide_sah(&mut self, nodeid: usize) {
        let node = self.nodes[nodeid];

        // split pos plane thing
        // let mut best_axis = 0;
        // let mut best_pos = 0.;
        // let mut best_cost = f32::MAX;
        // for axis in 0..3 {
        //     for i in 0..node.prim_count {
        //         let candidate_pos = self.primitives[node.first_prim + i].centroid(self.vertices)[axis];
        //         let cost = self.evaluate_sah(&node, axis, candidate_pos);
        //         if cost < best_cost {
        //             // println!("cost: {}, {}", cost, best_cost);
        //             best_pos = candidate_pos;
        //             best_axis = axis;
        //             best_cost = cost;
        //         }
        //     }
        // }

        let (best_axis, best_pos, best_cost) = (0..3)
            .into_par_iter()
            .map(|axis| {
                let mut local_best_pos = 0.;
                let mut local_best_cost = f32::MAX;
                
                for i in 0..node.prim_count {
                    let candidate_pos = self.primitives[node.first_prim + i].centroid(self.vertices)[axis];
                    let cost = self.evaluate_sah(&node, axis, candidate_pos);
                    if cost < local_best_cost {
                        local_best_pos = candidate_pos;
                        local_best_cost = cost;
                    }
                }
                
                (axis, local_best_pos, local_best_cost)
            })
            .reduce(
                || (0, 0., f32::MAX),  // Initial value for the reduction
                |(axis1, pos1, cost1), (axis2, pos2, cost2)| {
                    if cost1 < cost2 {
                        (axis1, pos1, cost1)
                    } else {
                        (axis2, pos2, cost2)
                    }
                },
            );

        let axis = best_axis;
        let split_pos = best_pos;

        // better return
        let no_split_cost = node.prim_count as f32 * node.bbox.half_area();
        if best_cost >= no_split_cost {
            return;
        }

        if (node.prim_count < 20) {
            return;
        }

        let mut i = node.first_prim;
        let mut j = i + node.prim_count - 1;

        while i <= j {
            if self.primitives[i].centroid(self.vertices)[best_axis] < split_pos {
                i += 1;
            } else {
                self.primitives.swap(i, j);
                if j == 0 { break }
                j -= 1;
            }
        }

        let left_count = i - node.first_prim;

        if left_count == 0 || left_count == node.prim_count {
            return;
        }


        // Left child
        let left_child_i = self.nodes.len();
        self.nodes.push(SahBVH2Node::new(
            self.node_bounds(node.first_prim..(node.first_prim + left_count)),
            None,
            None,
            node.first_prim,
            left_count,
        ));


        let right_start = node.first_prim + left_count;
        let right_count = node.prim_count - left_count;
        // Right child
        let right_child_i = self.nodes.len();
        self.nodes.push(SahBVH2Node::new(
            self.node_bounds(right_start..(right_start + right_count)),
            None,
            None,
            right_start,
            right_count,
        ));

        // Update the current node
        self.nodes[nodeid].left_i = Some(left_child_i);
        self.nodes[nodeid].right_i = Some(right_child_i);
        self.nodes[nodeid].prim_count = 0;

        // Recursively subdivide child nodes
        self.subdivide_sah(left_child_i);
        self.subdivide_sah(right_child_i);
    }

}

impl<Prim: Primitive<f32>> SahBVH2<Prim> {
    pub fn build(primitives: &mut [Prim], vertices: &[Vertex]) -> Self {
        let bboxes: Vec<AABB> = primitives.iter().map(|prim| prim.aabb(vertices)).collect();
        
        let mut builder = SahBVH2Builder::new(&bboxes, primitives, vertices);
        builder.build()
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

        if node.left_i.is_none() && node.right_i.is_none() {
            // Leaf node: Check intersection with the contained primitives
            for prim_id in node.first_prim..(node.first_prim + node.prim_count) {
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

            if let Some(right_index) = node.right_i {
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

            if let Some(left_index) = node.left_i {
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
        }

        result
    }

    pub fn validate_traversal(&self) -> Result<(), Vec<usize>> {
        if self.nodes.is_empty() {
            // An empty BVH is trivially valid.
            return Ok(());
        }

        let total_nodes = self.nodes.len();
        let mut visited = vec![false; total_nodes];
        let mut stack = Vec::new();

        // Start traversal from the root node (assumed to be at index 0).
        stack.push(0);

        while let Some(node_index) = stack.pop() {
            if node_index >= total_nodes {
                // Invalid node index detected.
                // Depending on your preference, you can treat this as an error or skip.
                // Here, we'll skip invalid indices but you can choose to return an error instead.
                println!(
                    "Warning: Encountered invalid node index {}. Skipping.",
                    node_index
                );
                continue;
            }

            if visited[node_index] {
                // Already visited this node; possible cycle detected.
                // For a tree structure, this should not happen.
                println!("Warning: Detected multiple visits to node {}.", node_index);
                continue;
            }

            // Mark the node as visited.
            visited[node_index] = true;

            let node = &self.nodes[node_index];

            // If the node is an internal node, add its children to the stack.
            if node.left_i.is_some() || node.right_i.is_some() {
                if let Some(left_index) = node.left_i {
                    stack.push(left_index);
                }
                if let Some(right_index) = node.right_i {
                    stack.push(right_index);
                }
            }
            // Leaf nodes do not have children, so nothing to add.
        }

        // After traversal, check for any nodes that were not visited.
        let unreachable_nodes: Vec<usize> = visited
            .iter()
            .enumerate()
            .filter_map(|(idx, &was_visited)| if !was_visited { Some(idx) } else { None })
            .collect();

        if unreachable_nodes.is_empty() {
            Ok(())
        } else {
            Err(unreachable_nodes)
        }
    }
}


pub struct HUHDebugInfo {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub inner_nodes: usize,
    pub min_depth: usize,
    pub max_depth: usize,
    pub total_depth: usize,
    pub min_prims_in_leaf: usize,
    pub max_prims_in_leaf: usize,
    pub total_primitives_in_leaves: usize,

    pub leaf_node_min: usize,
    pub leaf_node_max: usize,
}

impl HUHDebugInfo {
    pub fn average_depth(&self) -> f32 {
        self.total_depth as f32 / self.total_nodes as f32
    }

    pub fn average_prims_in_leaf(&self) -> f32 {
        if self.leaf_nodes == 0 {
            0.0
        } else {
            self.total_primitives_in_leaves as f32 / self.leaf_nodes as f32
        }
    }
}

impl<Prim: Primitive<f32>> SahBVH2<Prim> {
    /// Computes detailed debug information about the BVH.
    pub fn compute_debug_info(&self) -> HUHDebugInfo {
        let mut info = HUHDebugInfo {
            total_nodes: 0,
            leaf_nodes: 0,
            inner_nodes: 0,
            min_depth: usize::MAX,
            max_depth: 0,
            total_depth: 0,
            min_prims_in_leaf: usize::MAX,
            max_prims_in_leaf: 0,
            total_primitives_in_leaves: 0,

            leaf_node_min: usize::MAX,
            leaf_node_max: 0,
        };

        if !self.nodes.is_empty() {
            self.traverse_node(0, 0, &mut info);
        }

        info
    }

    /// Recursively traverses the BVH nodes to gather debug information.
    fn traverse_node(&self, node_index: usize, depth: usize, info: &mut HUHDebugInfo) {
        // Update node counts and depth statistics
        info.total_nodes += 1;
        info.total_depth += depth;
        info.min_depth = info.min_depth.min(depth);
        info.max_depth = info.max_depth.max(depth);

        let node = &self.nodes[node_index];

        if node.left_i.is_none() && node.right_i.is_none() {
            // Leaf node
            info.leaf_nodes += 1;
            info.min_prims_in_leaf = info.min_prims_in_leaf.min(depth);
            info.max_prims_in_leaf = info.max_prims_in_leaf.max(depth);

            let prim_count = node.prim_count;
            info.total_primitives_in_leaves += prim_count;
            info.min_prims_in_leaf = info.min_prims_in_leaf.min(prim_count);
            info.max_prims_in_leaf = info.max_prims_in_leaf.max(prim_count);
        } else {
            // Inner node
            info.inner_nodes += 1;

            if let Some(left_index) = node.left_i {
                self.traverse_node(left_index, depth + 1, info);
            }
            if let Some(right_index) = node.right_i {
                self.traverse_node(right_index, depth + 1, info);
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


        if node.left_i.is_none() && node.right_i.is_none() {
            node.bbox.draw_wireframe(img, colour, camera);

        } else {
        // if depth < depth_range.end {
            if let Some(left_index) = node.left_i {
                self.traverse_and_draw_wireframe(left_index, depth + 1, depth_range.clone(), img, colour, camera);
            }
            if let Some(right_index) = node.right_i {
                self.traverse_and_draw_wireframe(right_index, depth + 1, depth_range.clone(), img, colour, camera);
            }
        }
    }
}

impl fmt::Display for HUHDebugInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== BVH Debug Information ===")?;
        writeln!(f, "Total Nodes: {}", self.total_nodes)?;
        writeln!(f, " - Leaf Nodes: {}", self.leaf_nodes)?;
        writeln!(f, " - Inner Nodes: {}", self.inner_nodes)?;
        writeln!(f, "Depth Statistics:")?;
        writeln!(f, " - Minimum Depth: {}", self.min_depth)?;
        writeln!(f, " - Maximum Depth: {}", self.max_depth)?;
        writeln!(f, " - Average Depth: {:.2}", self.average_depth())?;
        writeln!(f, " - Minimum Leaf Node Depth: {:.2}", self.min_prims_in_leaf)?;
        writeln!(f, " - Maximum Leaf Node Depth: {:.2}", self.max_prims_in_leaf)?;
        writeln!(f, "Primitive Statistics in Leaf Nodes:")?;
        writeln!(f, " - Minimum Primitives in Leaf Nodes: {}", self.min_prims_in_leaf)?;
        writeln!(f, " - Maximum Primitives in Leaf Nodes: {}", self.max_prims_in_leaf)?;
        writeln!(f, " - Total Primitives in Leaf Nodes: {}", self.total_primitives_in_leaves)?;
        writeln!(f, " - Average Primitives per Leaf Node: {:.2}", self.average_prims_in_leaf())?;
        Ok(())
    }
}



impl<'a, Prim: Primitive<f32>> SahBVH2Builder<'a, Prim> {
    /// WARN: legacy
    fn _subdivide(&mut self, nodeid: usize) {
        let node = self.nodes[nodeid];

        if node.prim_count <= 20 {
            return;
        }

        let axis = node.bbox.largest_axis();

        let range_start = node.first_prim;
        let range_end = range_start + node.prim_count;
        let primitives_slice = &mut self.primitives[range_start..range_end];

        let median_index = node.prim_count / 2;

        // Partition primitives around the median centroid
        primitives_slice.select_nth_unstable_by(median_index, |a, b| {
            let ca = a.centroid(&self.vertices)[axis];
            let cb = b.centroid(&self.vertices)[axis];
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let left_count = median_index;
        let right_count = node.prim_count - left_count;

        // If the partitioning failed, try a different axis or terminate
        if left_count == 0 || right_count == 0 {
            return;
        }

        // Left child
        let left_child_i = self.nodes.len();
        self.nodes.push(SahBVH2Node::new(
            self.node_bounds(range_start..(range_start + left_count)),
            None,
            None,
            range_start,
            left_count,
        ));

        // Right child
        let right_child_i = self.nodes.len();
        self.nodes.push(SahBVH2Node::new(
            self.node_bounds((range_start + left_count)..range_end),
            None,
            None,
            range_start + left_count,
            right_count,
        ));

        // Update the current node
        self.nodes[nodeid].left_i = Some(left_child_i);
        self.nodes[nodeid].right_i = Some(right_child_i);
        self.nodes[nodeid].prim_count = 0;

        // Recursively subdivide child nodes
        self._subdivide(left_child_i);
        self._subdivide(right_child_i);
    }


    /// WARN: legacy
    fn _subdivide_iterative(&mut self, root_nodeid: usize) {
        let mut stack = Vec::with_capacity(17000);
        stack.push(root_nodeid);

        while let Some(nodeid) = stack.pop() {
            let node = self.nodes[nodeid];

            if node.prim_count <= 20 {
                return;
            }

            let axis = node.bbox.largest_axis();

            let range_start = node.first_prim;
            let range_end = range_start + node.prim_count;
            let primitives_slice = &mut self.primitives[range_start..range_end];

            let median_index = node.prim_count / 2;

            // Partition primitives around the median centroid
            primitives_slice.select_nth_unstable_by(median_index, |a, b| {
                let ca = a.centroid(&self.vertices)[axis];
                let cb = b.centroid(&self.vertices)[axis];
                ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let left_count = median_index;
            let right_count = node.prim_count - left_count;

            // If the partitioning failed, try a different axis or terminate
            if left_count == 0 || right_count == 0 {
                return;
            }

            // Push left child node and get its index
            let left_child_i = self.nodes.len();
            self.nodes.push(SahBVH2Node::new(
                self.node_bounds(range_start..(range_start + left_count)),
                None,
                None,
                range_start,
                left_count,
            ));

            // Right child
            let right_child_i = self.nodes.len();
            self.nodes.push(SahBVH2Node::new(
                self.node_bounds((range_start + left_count)..range_end),
                None,
                None,
                range_start + left_count,
                right_count,
            ));

            // Assign child indices to the current node
            self.nodes[nodeid].left_i = Some(left_child_i);
            self.nodes[nodeid].right_i = Some(right_child_i);
            self.nodes[nodeid].prim_count = 0;

            // Push child nodes to the stack for further processing
            stack.push(left_child_i);
            stack.push(right_child_i);
        }
    }
}
