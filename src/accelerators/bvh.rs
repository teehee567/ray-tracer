use glam::Vec3;

use crate::{AVec3, Au32, Material, Triangle};

const MAX_VAL: AVec3 = AVec3(Vec3 {
    x: 1e30,
    y: 1e30,
    z: 1e30,
});
const MIN_VAL: AVec3 = AVec3(Vec3 {
    x: -1e30,
    y: -1e30,
    z: -1e30,
});
const BVH_MAX_DEPTH: u32 = 64;
const SPLIT_ATTEMPTS: i32 = 8;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct BvhNode {
    min_idx: MinIdxUnion,
    max_amt: MaxAmtUnion,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union MinIdxUnion {
    min: AVec3,
    idx: IdxStruct,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct IdxStruct {
    pad: [i32; 3],
    idx: Au32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union MaxAmtUnion {
    max: AVec3,
    amt: AmtStruct,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct AmtStruct {
    pad: [i32; 3],
    amt: Au32,
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

    fn expand(&mut self, tri: &Triangle) {
        unsafe {
            let mut current_min = self.min_idx.min.0;
            let mut current_max = self.max_amt.max.0;
            current_min = current_min.min(tri.min_bound());
            current_max = current_max.max(tri.max_bound());
            self.min_idx.min = AVec3(current_min);
            self.max_amt.max = AVec3(current_max);
        }
    }

    fn initialize(&mut self, triangles: &Vec<Triangle>, indices: &[u32], offset: u32) {
        for &i in indices {
            self.expand(&triangles[i as usize]);
        }
        unsafe {
            let min_ptr = &mut self.min_idx.min as *mut AVec3 as *mut u8;
            let idx_ptr = min_ptr.add(12) as *mut Au32;
            *idx_ptr = Au32(offset);
            let max_ptr = &mut self.max_amt.max as *mut AVec3 as *mut u8;
            let amt_ptr = max_ptr.add(12) as *mut Au32;
            *amt_ptr = Au32(indices.len() as u32);
        }
    }

    fn is_leaf(&self) -> bool {
        unsafe {
            let max_ptr = &self.max_amt.max as *const AVec3 as *const u8;
            let amt_ptr = max_ptr.add(12) as *const Au32;
            (*amt_ptr).0 != 0
        }
    }

    fn left(&self) -> u32 {
        unsafe {
            let min_ptr = &self.min_idx.min as *const AVec3 as *const u8;
            let idx_ptr = min_ptr.add(12) as *const Au32;
            (*idx_ptr).0
        }
    }

    fn idx(&self) -> u32 {
        unsafe {
            let min_ptr = &self.min_idx.min as *const AVec3 as *const u8;
            let idx_ptr = min_ptr.add(12) as *const Au32;
            (*idx_ptr).0
        }
    }

    fn amt(&self) -> u32 {
        unsafe {
            let max_ptr = &self.max_amt.max as *const AVec3 as *const u8;
            let amt_ptr = max_ptr.add(12) as *const Au32;
            (*amt_ptr).0
        }
    }
}

pub struct BvhBuilder<'a> {
    bvh_list: Vec<BvhNode>,
    triangles: &'a mut Vec<Triangle>,
    materials: &'a mut Vec<Material>,
}

impl<'a> BvhBuilder<'a> {
    pub fn new(
        triangles: &'a mut Vec<Triangle>,
        materials: &'a mut Vec<Material>,
    ) -> Self {
        Self {
            bvh_list: Vec::new(),
            triangles,
            materials,
        }
    }

 
    pub fn build_bvh(mut self) -> Vec<BvhNode> {
        let mut indices: Vec<u32> = (0..self.triangles.len() as u32).collect();
        self.bvh_list.push(BvhNode::default());
        self.build_recursively(0, &mut indices, 0, 0, 1e30);

        Self::apply_ordering(self.triangles, &indices);

        // for node in &self.bvh_list {
        //     unsafe {
        //         println!("Node: {}", node.min_idx.idx.idx.0);
        //     }
        // }

        self.bvh_list
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

        let (split_axis, split_pos) = self.find_best_split(node_idx, indices);
        let best_cost = self.split_cost(indices, split_axis, split_pos);

        if best_cost >= parent_cost {
            return;
        }

        let mut left_count = 0;
        let mut right = indices.len() - 1;
        while left_count <= right {
            let tri = &self.triangles[indices[left_count] as usize];
            let center = (axis_min(tri, split_axis) + axis_max(tri, split_axis)) / 2.0;
            if center < split_pos {
                left_count += 1;
            } else {
                indices.swap(left_count, right);
                if right == 0 {
                    break;
                }
                right -= 1;
            }
        }

        if left_count == 0 || left_count == indices.len() {
            return;
        }

        let left_node_idx = self.bvh_list.len();
        self.bvh_list.push(BvhNode::default());
        self.bvh_list.push(BvhNode::default());

        unsafe {
            let node = &mut self.bvh_list[node_idx];
            let min_ptr = &mut node.min_idx.min as *mut AVec3 as *mut u8;
            let idx_ptr = min_ptr.add(12) as *mut Au32;
            *idx_ptr = Au32(left_node_idx as u32);

            let max_ptr = &mut node.max_amt.max as *mut AVec3 as *mut u8;
            let amt_ptr = max_ptr.add(12) as *mut Au32;
            *amt_ptr = Au32(0);
        }

        let (left_indices, right_indices) = indices.split_at_mut(left_count);
        self.build_recursively(left_node_idx, left_indices, depth + 1, offset, best_cost);
        self.build_recursively(
            left_node_idx + 1,
            right_indices,
            depth + 1,
            offset + left_count as u32,
            best_cost,
        );
    }

    fn find_best_split(&self, node_idx: usize, indices: &[u32]) -> (usize, f32) {
        let mut split_axis = 0;
        let mut split_pos = 0.0;
        let mut best_cost = 1e30;
        let node = &self.bvh_list[node_idx];
        unsafe {
            let node_min = node.min_idx.min.0;
            let node_max = node.max_amt.max.0;
            let dims = node_max - node_min;

            for axis in 0..3 {
                let step = dims[axis] / ((SPLIT_ATTEMPTS + 1) as f32);
                for attempt in 1..=SPLIT_ATTEMPTS {
                    let pos = node_min[axis] + step * attempt as f32;
                    let cost = self.split_cost(indices, axis, pos);
                    if cost < best_cost {
                        best_cost = cost;
                        split_axis = axis;
                        split_pos = pos;
                    }
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
        let sorted: Vec<Triangle> = ordering
            .iter()
            .map(|&i| items[i as usize].clone())
            .collect();
        *items = sorted;
    }
}


fn axis_min(tri: &Triangle, axis: usize) -> f32 {
    tri.vertices
        .iter()
        .map(|v| v.0[axis])
        .fold(f32::INFINITY, |a, b| a.min(b))
}


fn axis_max(tri: &Triangle, axis: usize) -> f32 {
    tri.vertices
        .iter()
        .map(|v| v.0[axis])
        .fold(-f32::INFINITY, |a, b| a.max(b))
}
