use std::marker::PhantomData;
use std::{cmp::Ordering, fmt};

use crate::{AlignedVec3, Alignedu32, Vertex};
use glam::Vec3;

use super::{aabb::AABB, Primitive};

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

#[derive(Copy, Clone)]
struct Bin {
    bbox: AABB,
    prim_count: usize,
}

#[derive(Copy, Clone)]
struct Split {
    bin_id: usize,
    cost: f32,
    axis: usize,
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

#[repr(C)]
#[derive(Copy, Clone, Debug)]
#[repr(align(16))]
pub struct BvhNode {
    pub idx: Alignedu32,
    pub amt: Alignedu32,
    pub left: Alignedu32,
    pub min: AlignedVec3,
    pub max: AlignedVec3,
}

impl Default for BvhNode {
    fn default() -> Self {
        Self {
            idx: Alignedu32(0),
            amt: Alignedu32(0),
            left: Alignedu32(0),
            min: AlignedVec3(Vec3::splat(1e30)),
            max: AlignedVec3(Vec3::splat(-1e30)),
        }
    }
}

pub struct BVH<Prim: Primitive> {
    pub nodes: Vec<BvhNode>,
    sah: SAH,
    phantom: PhantomData<Prim>,
}

impl<Prim: Primitive> BVH<Prim> {
    pub fn build(primitives: &[Prim], sah: SAH, max_leaf_size: usize) -> Self {
        let bboxes: Vec<AABB> = primitives.iter().map(|prim| prim.aabb()).collect();
        let centers: Vec<Vec3> = primitives.iter().map(|prim| prim.centroid()).collect();

        let mut builder = BinnedSahBuilder::<8>::new(&bboxes, &centers, sah, max_leaf_size);
        let nodes = builder.build();

        BVH {
            nodes,
            sah,
            phantom: PhantomData,
        }
    }
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

impl<Prim: Primitive> BVH<Prim> {
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

        if node.left.0 == 0 {
            // Leaf node
            info.leaf_nodes += 1;
            let leaf_sah_cost =
                sah.leaf_cost(node.amt.0 as usize, &AABB::new(node.min.0, node.max.0));
            info.total_sah_cost += leaf_sah_cost;
            info.min_sah_cost = info.min_sah_cost.min(leaf_sah_cost);
            info.max_sah_cost = info.max_sah_cost.max(leaf_sah_cost);
        } else {
            // Inner node
            info.inner_nodes += 1;
            let node_sah_cost = sah.node_cost(&AABB::new(node.min.0, node.max.0));
            info.total_sah_cost += node_sah_cost;
            info.min_sah_cost = info.min_sah_cost.min(node_sah_cost);
            info.max_sah_cost = info.max_sah_cost.max(node_sah_cost);

            // Traverse both children without checking indices, assuming correct BVH structure
            self.traverse_node(node.left.0 as usize, depth + 1, sah, info);
            self.traverse_node((node.left.0 + 1) as usize, depth + 1, sah, info);
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

pub struct BinnedSahBuilder<'a, const BIN_COUNT: usize> {
    bboxes: &'a [AABB],
    centers: &'a [Vec3],
    prim_ids: Vec<usize>,
    sah: SAH,
    max_leaf_size: usize,
}

impl<'a, const BIN_COUNT: usize> BinnedSahBuilder<'a, BIN_COUNT> {
    pub fn new(bboxes: &'a [AABB], centers: &'a [Vec3], sah: SAH, max_leaf_size: usize) -> Self {
        let prim_ids = (0..bboxes.len()).collect();
        BinnedSahBuilder {
            bboxes,
            centers,
            prim_ids,
            sah,
            max_leaf_size,
        }
    }

    pub fn build(&mut self) -> Vec<BvhNode> {
        let mut nodes = Vec::new();
        self.build_node(0, self.bboxes.len(), &mut nodes);
        nodes
    }

    fn build_node(&mut self, begin: usize, end: usize, nodes: &mut Vec<BvhNode>) -> usize {
        let bbox = self.compute_bbox(begin, end);
        let node_idx = nodes.len();
        nodes.push(BvhNode::default());

        if let Some(split_index) = self.try_split(&bbox, begin, end) {
            // Reserve two slots for children
            let left_child_idx = nodes.len();
            nodes.push(BvhNode::default());
            nodes.push(BvhNode::default());

            // Build left and right children into the reserved slots
            self.build_child(begin, split_index, nodes, left_child_idx);
            self.build_child(split_index, end, nodes, left_child_idx + 1);

            // Update current node
            nodes[node_idx] = BvhNode {
                idx: Alignedu32(begin as u32),
                amt: Alignedu32((end - begin) as u32),
                left: Alignedu32(left_child_idx as u32),
                min: AlignedVec3(bbox.min),
                max: AlignedVec3(bbox.max),
            };
        } else {
            // Leaf node
            nodes[node_idx] = BvhNode {
                idx: Alignedu32(begin as u32),
                amt: Alignedu32((end - begin) as u32),
                left: Alignedu32(0),
                min: AlignedVec3(bbox.min),
                max: AlignedVec3(bbox.max),
            };
        }
        node_idx
    }

    fn build_child(
        &mut self,
        begin: usize,
        end: usize,
        nodes: &mut Vec<BvhNode>,
        child_idx: usize,
    ) {
        let child_bbox = self.compute_bbox(begin, end);
        let child_node_idx = child_idx;

        if let Some(split_index) = self.try_split(&child_bbox, begin, end) {
            let left_child_idx = nodes.len();
            nodes.push(BvhNode::default());
            nodes.push(BvhNode::default());

            self.build_child(begin, split_index, nodes, left_child_idx);
            self.build_child(split_index, end, nodes, left_child_idx + 1);

            nodes[child_node_idx] = BvhNode {
                idx: Alignedu32(begin as u32),
                amt: Alignedu32((end - begin) as u32),
                left: Alignedu32(left_child_idx as u32),
                min: AlignedVec3(child_bbox.min),
                max: AlignedVec3(child_bbox.max),
            };
        } else {
            nodes[child_node_idx] = BvhNode {
                idx: Alignedu32(begin as u32),
                amt: Alignedu32((end - begin) as u32),
                left: Alignedu32(0),
                min: AlignedVec3(child_bbox.min),
                max: AlignedVec3(child_bbox.max),
            };
        }
    }
}

impl<'a, const BIN_COUNT: usize> BinnedSahBuilder<'a, BIN_COUNT> {
    fn compute_bbox(&self, begin: usize, end: usize) -> AABB {
        let mut bbox = AABB::default();
        for &prim_id in &self.prim_ids[begin..end] {
            bbox.grow_bb_mut(&self.bboxes[prim_id]);
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
        let bin_scale = Vec3::new(
            f32::from(BIN_COUNT as f32),
            f32::from(BIN_COUNT as f32),
            f32::from(BIN_COUNT as f32),
        ) * &bbox.diagonal();
        let bin_offset = bin_scale * &bbox.min * f32::from(-1.0);

        for &prim_id in &self.prim_ids[begin..end] {
            let center = self.centers[prim_id];
            let pos = (center * &bin_scale) + bin_offset;

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
