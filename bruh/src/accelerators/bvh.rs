use glam::Vec3;

use crate::{AlignedVec3, Alignedu32, Triangle};

const MAX_VAL: AlignedVec3 = AlignedVec3(Vec3 { x: 1e30, y: 1e30, z: 1e30 });
const MIN_VAL: AlignedVec3 = AlignedVec3(Vec3 { x: -1e30, y: -1e30, z: -1e30 });
const BVH_MAX_DEPTH: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BvhNode {
    min_idx: MinIdxUnion,
    max_amt: MaxAmtUnion,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union MinIdxUnion {
    min: AlignedVec3,
    idx: IdxStruct,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct IdxStruct {
    pad: [i32; 3],
    idx: Alignedu32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union MaxAmtUnion {
    max: AlignedVec3,
    amt: AmtStruct,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct AmtStruct {
    pad: [i32; 3],
    amt: Alignedu32,
}

impl Default for BvhNode {
    fn default() -> Self {
        Self {
            min_idx: MinIdxUnion { min: MAX_VAL },
            max_amt: MaxAmtUnion { max: MIN_VAL },
        }
    }
}

impl BvhNode {
    fn area(&self) -> f32 {
        unsafe {
            let min = self.min_idx.min.0;
            let max = self.max_amt.max.0;

            if max == MIN_VAL.0 || min == MAX_VAL.0 {
                return 0.0;
            }
            let len = max - min;
            2.0 * (len.x * len.y + len.x * len.z + len.y * len.z)
        }
    }

    /// Expand the node's bounds to include the given triangle.
    fn expand(&mut self, tri: &Triangle) {
        unsafe {
            let mut current_min = self.min_idx.min.0;
            let mut current_max = self.max_amt.max.0;

            current_min = current_min.min(tri.min_bound());
            current_max = current_max.max(tri.max_bound());

            self.min_idx.min = AlignedVec3(current_min);
            self.max_amt.max = AlignedVec3(current_max);
        }
    }

    /// Initialize the node by expanding its bounds over the given list of triangle indices.
    /// Also set the offset and number of items.
    fn initialize(&mut self, triangles: &Vec<Triangle>, indices: &[u32], offset: u32) {
        for &i in indices {
            self.expand(&triangles[i as usize]);
        }

        // Set idx and amt in the padding of min and max
        unsafe {
            let min_ptr = &mut self.min_idx.min as *mut AlignedVec3 as *mut u8;
            let idx_ptr = min_ptr.add(12) as *mut Alignedu32;
            *idx_ptr = Alignedu32(offset);

            let max_ptr = &mut self.max_amt.max as *mut AlignedVec3 as *mut u8;
            let amt_ptr = max_ptr.add(12) as *mut Alignedu32;
            *amt_ptr = Alignedu32(indices.len() as u32);
        }
    }

    /// Check if the node is a leaf (amt != 0)
    fn is_leaf(&self) -> bool {
        unsafe {
            let max_ptr = &self.max_amt.max as *const AlignedVec3 as *const u8;
            let amt_ptr = max_ptr.add(12) as *const Alignedu32;
            (*amt_ptr).0 != 0
        }
    }

    /// Get the left child index for internal nodes
    fn left(&self) -> u32 {
        unsafe {
            let min_ptr = &self.min_idx.min as *const AlignedVec3 as *const u8;
            let idx_ptr = min_ptr.add(12) as *const Alignedu32;
            (*idx_ptr).0
        }
    }

    /// Get the triangle offset for leaf nodes
    fn idx(&self) -> u32 {
        unsafe {
            let min_ptr = &self.min_idx.min as *const AlignedVec3 as *const u8;
            let idx_ptr = min_ptr.add(12) as *const Alignedu32;
            (*idx_ptr).0
        }
    }

    /// Get the triangle count for leaf nodes
    fn amt(&self) -> u32 {
        unsafe {
            let max_ptr = &self.max_amt.max as *const AlignedVec3 as *const u8;
            let amt_ptr = max_ptr.add(12) as *const Alignedu32;
            (*amt_ptr).0
        }
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
        let mut indices: Vec<u32> = (0..self.triangles.len() as u32).collect();
        self.bvh_list.push(BvhNode::default());
        self.build_recursively(0, &mut indices, 0, 0, 1e30);

        Self::apply_ordering(self.triangles, &indices);
        self.bvh_list.clone()
    }

    fn build_recursively(
        &mut self,
        node_idx: usize,
        indices: &mut [u32],
        depth: u32,
        offset: u32,
        parent_cost: f32,
    ) {
        let node = &mut self.bvh_list[node_idx];
        node.initialize(self.triangles, indices, offset);

        if depth >= BVH_MAX_DEPTH || indices.len() <= 1 {
            return;
        }

        let (split_axis, split_pos) = self.find_best_split(indices);
        let best_cost = self.split_cost(indices, split_axis, split_pos);

        if best_cost >= parent_cost {
            return;
        }

        let mut left_idx = 0;
        let mut right_idx = indices.len() - 1;
        while left_idx <= right_idx {
            let tri = &self.triangles[indices[left_idx] as usize];
            let center = (axis_min(tri, split_axis) + axis_max(tri, split_axis)) / 2.0;
            if center < split_pos {
                left_idx += 1;
            } else {
                indices.swap(left_idx, right_idx);
                if right_idx == 0 {
                    break;
                }
                right_idx -= 1;
            }
        }

        if left_idx == 0 || right_idx == indices.len() {
            return;
        }

        let left_node_idx = self.bvh_list.len();
        self.bvh_list.push(BvhNode::default());
        self.bvh_list.push(BvhNode::default());

        // Set left child index and mark as internal node
        unsafe {
            let node = &mut self.bvh_list[node_idx];
            let min_ptr = &mut node.min_idx.min as *mut AlignedVec3 as *mut u8;
            let idx_ptr = min_ptr.add(12) as *mut Alignedu32;
            *idx_ptr = Alignedu32(left_node_idx as u32);

            let max_ptr = &mut node.max_amt.max as *mut AlignedVec3 as *mut u8;
            let amt_ptr = max_ptr.add(12) as *mut Alignedu32;
            *amt_ptr = Alignedu32(0);
        }

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

    fn find_best_split(&self, indices: &[u32]) -> (usize, f32) {
        let mut split_axis = 0;
        let mut split_pos = 0.0;
        let mut best_cost = 1e30;

        for axis in 0..3 {
            for &tri_idx in indices {
                let tri = &self.triangles[tri_idx as usize];
                let center = (axis_min(tri, axis) + axis_max(tri, axis)) / 2.0;
                let cost = self.split_cost(indices, axis, center);
                if cost < best_cost {
                    best_cost = cost;
                    split_axis = axis;
                    split_pos = center;
                }
            }
        }

        (split_axis, split_pos)
    }

    fn split_cost(&self, indices: &[u32], axis: usize, location: f32) -> f32 {
        let mut left_amount = 0;
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
            }
        }

        let right_amount = indices.len() - left_amount;
        (left_amount as f32) * node_left.area() + (right_amount as f32) * node_right.area()
    }

    fn apply_ordering(items: &mut Vec<Triangle>, ordering: &[u32]) {
        let sorted: Vec<Triangle> = ordering.iter()
            .map(|&i| items[i as usize].clone())
            .collect();
        *items = sorted;
    }
}

fn axis_min(tri: &Triangle, axis: usize) -> f32 {
    tri.vertices.iter()
        .map(|v| v.0[axis])
        .fold(f32::INFINITY, |a, b| a.min(b))
}

fn axis_max(tri: &Triangle, axis: usize) -> f32 {
    tri.vertices.iter()
        .map(|v| v.0[axis])
        .fold(-f32::INFINITY, |a, b| a.max(b))
}
