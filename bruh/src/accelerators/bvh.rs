use glam::Vec3;

use crate::{AlignedVec3, Alignedu32, Triangle};

use super::bvhfromotherland::BvhNode;

const MAX_VAL: AlignedVec3 = AlignedVec3(Vec3 { x: 1e30, y: 1e30, z: 1e30 });
const MIN_VAL: AlignedVec3 = AlignedVec3(Vec3 { x: -1e30, y: -1e30, z: -1e30 });
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
    triangles: &'a mut Vec<Triangle>,
}

impl<'a> BvhBuilder<'a> {
    /// Create a new builder.
    pub fn new(triangles: &'a mut Vec<Triangle>) -> Self {
        Self {
            bvh_list: Vec::new(),
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

        if depth >= BVH_MAX_DEPTH || indices.len() <= 1 {
            return;
        }

        let (split_axis, split_pos) = self.find_best_split(indices);
        let best_cost = self.split_cost(indices, split_axis, split_pos);

        // If splitting does not improve the cost, do not subdivide.
        if best_cost >= parent_cost {
            return;
        }

        // Partition the indices based on the split:
        let mut left_idx = 0;
        let mut right_idx = indices.len() - 1;
        while left_idx <= right_idx {
            let tri = &self.triangles[indices[left_idx] as usize];
            let center = (axis_min(tri, split_axis) + axis_max(tri, split_axis)) / 2.0;
            if center < split_pos {
                left_idx += 1;
            } else {
                indices.swap(left_idx, right_idx);
                // Prevent underflow.
                if right_idx == 0 {
                    break;
                }
                right_idx -= 1;
            }
        }

        // Create two child nodes.
        let left_node_idx = self.bvh_list.len();
        self.bvh_list[node_idx].left = Alignedu32(left_node_idx as u32);
        self.bvh_list.push(BvhNode::default());
        self.bvh_list.push(BvhNode::default());

        // Split the indices into left and right sub-slices.
        let (left_indices, right_indices) = indices.split_at_mut(left_idx);
        self.build_recursively(left_node_idx, left_indices, depth + 1, offset, best_cost);
        self.build_recursively(
            left_node_idx + 1,
            right_indices,
            depth + 1,
            offset + left_idx as u32,
            best_cost,
        );
    }

    /// Find the best split (axis and position) using a Surface Area Heuristic (SAH).
    fn find_best_split(&self, indices: &[u32]) -> (usize, f32) {
        let mut split_axis: Option<usize> = None;
        let mut split_pos = 0.0;
        let mut best_cost = 1e30;

        // Try all three axes.
        for axis in 0..3 {
            // For each triangle, compute the center along this axis.
            for &tri_idx in indices {
                let tri = &self.triangles[tri_idx as usize];
                let center = (axis_min(tri, axis) + axis_max(tri, axis)) / 2.0;
                let cost = self.split_cost(indices, axis, center);
                if cost < best_cost {
                    best_cost = cost;
                    split_axis = Some(axis);
                    split_pos = center;
                }
            }
        }
        // dbg!(&split_axis, &split_pos, &best_cost);

        split_axis
            .map(|axis| (axis, split_pos))
            .expect("The axis has not been set")
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
