use core::f32;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Range;

use image::{RgbImage, RgbaImage};
use nalgebra::Vector3;

use crate::accelerators::aabb::AABB;
use crate::core::camera::Camera;
use crate::core::interval::Interval;
use crate::core::ray::Ray;
use crate::geometry::objects::mesh::{Primitive, TriangleIntersection};
use crate::geometry::objects::triangle::{self, Vertex};
use crate::geometry::wireframe::WireFrame;
use crate::utils::colour::Colour;

#[derive(Copy, Clone)]
struct BinSahBVH2Node {
    bbox: AABB,
    left_i: Option<usize>,
    right_i: Option<usize>,
    first_prim: usize,
    prim_count: usize,
}

impl BinSahBVH2Node {
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

impl BinSahBVH2Node {
    fn is_leaf(&self) -> bool {
        return self.prim_count > 0;
    }
}

#[derive(Default, Copy, Clone, Debug)]
struct Bin {
    bbox: AABB,
    prim_count: usize,
}

pub struct BinSahBVH2<Prim: Primitive<f32>> {
    nodes: Vec<BinSahBVH2Node>,
    phantom: PhantomData<Prim>
}

pub struct BinSahBVH2Builder<'a, Prim: Primitive<f32>> {
    bboxs: &'a [AABB],
    bin_count: usize,
    // centers: &'a [Vector3<f32>],
    primitives: &'a mut [Prim],
    vertices: &'a [Vertex],
    root_node_id: usize,
    nodes: Vec<BinSahBVH2Node>,
}

impl<'a, Prim: Primitive<f32>> BinSahBVH2Builder<'a, Prim> {
    pub fn new(bboxs: &'a [AABB], primitives: &'a mut [Prim], vertices: &'a [Vertex], bin_count: usize) -> Self {
        let nodes: Vec<BinSahBVH2Node> = Vec::with_capacity(primitives.len() * 2 - 1);
        Self {
            bboxs,
            bin_count,
            primitives,
            vertices,
            root_node_id: 0,
            nodes
        }
    }

    pub fn build(mut self) -> BinSahBVH2<Prim> {
        let amount_primitives = self.primitives.len();

        self.nodes.push(BinSahBVH2Node::new(
            self.node_bounds(0..amount_primitives),
            Some(0),
            Some(0),
            0,
            amount_primitives,
        ));

        self.subdivide_sah(0);

        BinSahBVH2 {
            nodes: self.nodes,
            phantom: PhantomData
        }
    }

    fn node_bounds(&self, range: Range<usize>) -> AABB {
        let mut aabb = AABB::default();
        for primitive in &self.primitives[range] {
            aabb.grow_bb_mut(&primitive.aabb(&self.vertices));
        }

        aabb
    }
    
    fn find_best_split_plane(&self, node: &BinSahBVH2Node) -> (usize, f32, f32) {
        let mut best_cost = f32::MAX;
        let mut best_axis = 0;
        let mut best_pos = 0.;
        for axis in 0..3 {
            let mut bounds_min = f32::MAX;
            let mut bounds_max = f32::MIN;
            for i in 0..node.prim_count {
                let centroid = self.primitives[node.first_prim + i].centroid(self.vertices)[axis];
                bounds_min = bounds_min.min(centroid);
                bounds_max = bounds_max.max(centroid);
            }

            if (bounds_min == bounds_max) { continue }

            // populate the bins
            let mut bin = vec![Bin::default(); self.bin_count];
            let scale = self.bin_count as f32 / (bounds_max - bounds_min);
            
            for i in 0..node.prim_count {
                let primitive = &self.primitives[node.first_prim + i];
                let bin_i = (self.bin_count - 1).min(((primitive.centroid(self.vertices)[axis] - bounds_min) * scale) as usize);
                bin[bin_i].prim_count += 1;
                bin[bin_i].bbox.grow_bb_mut(&primitive.aabb(self.vertices));
            }
            
            let mut left_count = vec![0usize; self.bin_count - 1];
            let mut right_count = vec![0usize; self.bin_count - 1];
            let mut left_area = vec![0.0f32; self.bin_count - 1];
            let mut right_area = vec![0.0f32; self.bin_count - 1];
            // dbg!(&bin);

            let mut left_sum = 0;
            let mut right_sum = 0;
            let mut left_box = AABB::default();
            let mut right_box = AABB::default();
            for i in 0..(self.bin_count - 1) {
                left_sum += bin[i].prim_count;
                left_count[i] = left_sum;
                left_box.grow_bb_mut(&bin[i].bbox);
                left_area[i] = left_box.half_area();

                let j = self.bin_count - 1 - i;
                right_sum += bin[j].prim_count;
                right_count[j - 1] = right_sum;
                right_box.grow_bb_mut(&bin[j].bbox);
                right_area[j - 1] = right_box.half_area();
            }


            let scale = (bounds_max - bounds_min) / self.bin_count as f32;
            for i in 0..(self.bin_count - 1) {
                let plane_cost = left_count[i] as f32 * left_area[i] + right_count[i] as f32 * right_area[i];
                if (plane_cost < best_cost) {
                    best_axis = axis;
                    best_pos = bounds_min + scale * (i as f32 + 1.);
                    best_cost = plane_cost;
                }
            }
        }

        (best_axis, best_pos, best_cost)
    }

    fn subdivide_sah(&mut self, nodeid: usize) {
        // Use a mutable reference to modify the node
        let node = self.nodes[nodeid];

        // split pos plane thing
        let (axis, split_pos, best_cost) = self.find_best_split_plane(&node);

        if (node.prim_count < 20) {
            return;
        }

        // better return
        let no_split_cost = node.prim_count as f32 * node.bbox.half_area();
        if best_cost >= no_split_cost {
            return;
        }

        let mut i = node.first_prim;
        let mut j = i + node.prim_count - 1;

        while i <= j {
            if self.primitives[i].centroid(self.vertices)[axis] < split_pos {
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
        self.nodes.push(BinSahBVH2Node::new(
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
        self.nodes.push(BinSahBVH2Node::new(
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

impl<Prim: Primitive<f32>> BinSahBVH2<Prim> {
    pub fn build(primitives: &mut [Prim], vertices: &[Vertex], bin_count: usize) -> Self {
        let bboxes: Vec<AABB> = primitives.iter().map(|prim| prim.aabb(vertices)).collect();
         
        
        let mut builder = BinSahBVH2Builder::new(&bboxes, primitives, vertices, bin_count);

        let start = std::time::Instant::now();
        let bvh = builder.build();
        let duration = start.elapsed();
        println!("BinSahBVH2 built in: {:?}", duration);

        bvh
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

impl<Prim: Primitive<f32>> BinSahBVH2<Prim> {
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
