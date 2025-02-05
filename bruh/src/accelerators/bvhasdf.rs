use glam::Vec3;

use crate::{AlignedVec3, Alignedu32, Triangle};

use super::{aabb::AABB, bvhfromotherland::BvhNode, Primitive};

const MAX_VAL: AlignedVec3 = AlignedVec3(Vec3 {
    x: 1e30,
    y: 1e30,
    z: 1e30,
});
const MIN_VAL: AlignedVec3 = AlignedVec3(Vec3 {
    x: -1e30,
    y: -1e30,
    z: -1e30,
});
const BVH_MAX_DEPTH: u32 = 16;

fn axis_min(tri: &Triangle, axis: usize) -> f32 {
    let mut min_val = tri.vertices[0].0[axis];
    for vtx in 1..3 {
        let val = tri.vertices[vtx].0[axis];
        if val < min_val {
            min_val = val;
        }
    }
    min_val
}

fn axis_max(tri: &Triangle, axis: usize) -> f32 {
    let mut max_val = tri.vertices[0].0[axis];
    for vtx in 1..3 {
        let val = tri.vertices[vtx].0[axis];
        if val > max_val {
            max_val = val;
        }
    }
    max_val
}

// impl Default for BvhNode {
//     fn default() -> Self {
//         Self {
//             idx: Alignedu32(0),
//             amt: Alignedu32(0),
//             left: Alignedu32(0),
//             min: MAX_VAL,
//             max: MIN_VAL,
//         }
//     }
// }

impl BvhNode {
    fn area(&self) -> f32 {
        if self.max.0 == MIN_VAL.0 || self.min.0 == MAX_VAL.0 {
            return 0.0;
        }
        let len = self.max.0 - self.min.0;
        2.0 * (len.x * len.y + len.x * len.z + len.y * len.z)
    }

    /// Expand the node's bounds to include the given triangle.
    fn expand(&mut self, tri: &Triangle) {
        self.min = AlignedVec3(self.min.0.min(tri.min_bound()));
        self.max = AlignedVec3(self.max.0.max(tri.max_bound()));
    }

    /// Initialize the node by expanding its bounds over the given list of triangle indices.
    /// Also set the offset and number of items.
    fn initialize(&mut self, triangles: &Vec<Triangle>, indices: &[u32], offset: u32) {
        for &i in indices {
            self.expand(&triangles[i as usize]);
        }
        self.idx = Alignedu32(offset);
        self.amt = Alignedu32(indices.len() as u32);
        self.left = Alignedu32(0); // 0 indicates a leaf
    }
}

pub struct BvhBuilder<'a> {
    bvh_list: Vec<BvhNode>,
    bboxes: Vec<AABB>,
    centers: Vec<Vec3>,
    triangles: &'a mut Vec<Triangle>,
}

impl<'a> BvhBuilder<'a> {
    /// Create a new builder.
    pub fn new(triangles: &'a mut Vec<Triangle>) -> Self {
        let bboxes: Vec<AABB> = triangles.iter().map(|prim| prim.aabb()).collect();
        let centers: Vec<Vec3> = triangles.iter().map(|prim| prim.centroid()).collect();
        Self {
            bvh_list: Vec::new(),
            bboxes,
            centers,
            triangles,
        }
    }

    /// Build the BVH and return the list of BVH nodes.
    pub fn build_bvh(&mut self) -> Vec<BvhNode> {
        // Create a list of triangle indices: 0, 1, 2, ...
        let mut indices: Vec<u32> = (0..self.triangles.len() as u32).collect();
        // Start with a root node.
        self.bvh_list.push(BvhNode::default());
        self.build_recursively(0, &mut indices, 0, 0, 1e30);

        // Reorder the triangles according to the computed ordering.
        Self::apply_ordering(self.triangles, &indices);

        // Return a copy of the BVH list.
        self.bvh_list.clone()
    }

    /// Recursively build the BVH.
    fn build_recursively(
        &mut self,
        node_idx: usize,
        indices: &mut [u32],
        depth: u32,
        offset: u32,
        parent_cost: f32,
    ) {
        self.bvh_list[node_idx].initialize(self.triangles, indices, offset);

        if depth >= BVH_MAX_DEPTH || indices.len() <= 2 {
            // Early termination for small nodes
            return;
        }

        if let Some((best_axis, best_split_k, best_cost)) = self.find_best_split(indices) {
            if best_cost >= parent_cost {
                return;
            }

            // Sort indices along best axis
            indices.sort_unstable_by(|&a, &b| {
                self.triangles[a as usize].centroid()[best_axis]
                    .partial_cmp(&self.triangles[b as usize].centroid()[best_axis])
                    .unwrap()
            });

            // Split into left and right
            let (left_indices, right_indices) = indices.split_at_mut(best_split_k);

            // Create child nodes
            let left_node_idx = self.bvh_list.len();
            self.bvh_list[node_idx].left = Alignedu32(left_node_idx as u32);
            self.bvh_list.push(BvhNode::default());
            self.bvh_list.push(BvhNode::default());

            // Recurse
            self.build_recursively(left_node_idx, left_indices, depth + 1, offset, best_cost);
            self.build_recursively(
                left_node_idx + 1,
                right_indices,
                depth + 1,
                offset + best_split_k as u32,
                best_cost,
            );
        }
    }

    /// Find the best split (axis and position) using a Surface Area Heuristic (SAH).
    fn find_best_split(&self, indices: &[u32]) -> Option<(usize, usize, f32)> {
        let mut best_axis = 0;
        let mut best_split_k = 0;
        let mut best_cost = f32::INFINITY;

        for axis in 0..3 {
            let mut sorted_indices = indices.to_vec();
            sorted_indices.sort_unstable_by(|&a, &b| {
                self.centers[a as usize][axis]
                    .partial_cmp(&self.triangles[b as usize].centroid()[axis])
                    .unwrap()
            });

            let n = sorted_indices.len();
            if n < 4 {
                // Don't split small nodes
                continue;
            }

            // Compute prefix and suffix bounds
            let (prefix_bounds, suffix_bounds) =
                self.compute_prefix_suffix(&sorted_indices);

            // Evaluate all possible splits
            let (current_best_cost, current_best_split) = (1..n)
                .map(|k| {
                    let left_area = prefix_bounds[k - 1].area();
                    let right_area = suffix_bounds[k].area();
                    (k as f32 * left_area + (n - k) as f32 * right_area, k)
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap_or((f32::INFINITY, 0));

            if current_best_cost < best_cost {
                best_cost = current_best_cost;
                best_axis = axis;
                best_split_k = current_best_split;
            }
        }

        if best_cost == f32::INFINITY {
            None
        } else {
            Some((best_axis, best_split_k, best_cost))
        }
    }

    fn compute_prefix_suffix(
        &self,
        sorted_indices: &[u32],
    ) -> (Vec<BvhNode>, Vec<BvhNode>) {
        let n = sorted_indices.len();
        let mut prefix_bounds = vec![BvhNode::default(); n];
        let mut suffix_bounds = vec![BvhNode::default(); n];

        // Compute prefix bounds
        for i in 0..n {
            let tri = &self.triangles[sorted_indices[i] as usize];
            if i == 0 {
                prefix_bounds[i].min = AlignedVec3(self.bboxes[i].min);
                prefix_bounds[i].max = AlignedVec3(self.bboxes[i].max);
            } else {
                prefix_bounds[i].min = AlignedVec3(prefix_bounds[i - 1].min.0.min(self.bboxes[i].min));
                prefix_bounds[i].max = AlignedVec3(prefix_bounds[i - 1].max.0.max(self.bboxes[i].max));
            }
        }

        // Compute suffix bounds
        for i in (0..n).rev() {
            let tri = &self.triangles[sorted_indices[i] as usize];
            if i == n - 1 {
                suffix_bounds[i].min = AlignedVec3(self.bboxes[i].min);
                suffix_bounds[i].max = AlignedVec3(self.bboxes[i].max);
            } else {
                suffix_bounds[i].min = AlignedVec3(suffix_bounds[i + 1].min.0.min(self.bboxes[i].min));
                suffix_bounds[i].max = AlignedVec3(suffix_bounds[i + 1].max.0.max(self.bboxes[i].max));
            }
        }

        (prefix_bounds, suffix_bounds)
    }

    /// Evaluate the cost of splitting at a given axis and position.
    /// Cost = (# left triangles * left area) + (# right triangles * right area)
    fn split_cost(&self, indices: &[u32], axis: usize, location: f32) -> f32 {
        let mut left_amount = 0;
        let mut right_amount = 0;
        let mut node_left = BvhNode::default();
        let mut node_right = BvhNode::default();

        for &idx in indices {
            let tri = &self.triangles[idx as usize];
            let center = (axis_min(tri, axis) + axis_max(tri, axis)) / 2.0;
            if center < location {
                node_left.expand(tri);
                left_amount += 1;
            } else {
                node_right.expand(tri);
                right_amount += 1;
            }
        }

        (left_amount as f32) * node_left.area() + (right_amount as f32) * node_right.area()
    }

    /// Reorder the triangles based on the computed ordering.
    fn apply_ordering(items: &mut Vec<Triangle>, ordering: &[u32]) {
        let sorted: Vec<Triangle> = ordering
            .iter()
            .map(|&i| items[i as usize].clone())
            .collect();
        *items = sorted;
    }
}
