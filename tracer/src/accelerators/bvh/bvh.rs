#![allow(clippy::all)]
use std::{marker::PhantomData, ops::RangeBounds};
use nalgebra::Vector3;

use crate::{accelerators::aabb::AABB, geometry::objects::mesh::Primitive};

pub type Float = f32;

pub struct BVH2<Prim: Primitive<Float>> {
    pub nodes: Vec<BVH2Node>,
    phantom: PhantomData<Prim>
}

#[derive(Clone, Copy)]
pub struct SAH2 {
    log_cluster_size: usize,
    prim_offset: usize,
    cost_ratio: f32,
}

impl SAH2 {
    pub fn new(log_cluster_size: usize, cost_ratio: f32) -> Self {
        // Calculate the primitive offset based on the cluster size.
        // Equivalent to (1 << log_cluster_size) - 1 in C++.
        let prim_offset = if log_cluster_size > 0 {
            (1usize << log_cluster_size) - 1
        } else {
            0
        };
        
        SAH2 {
            log_cluster_size,
            prim_offset,
            cost_ratio,
        }
    }

    // pub fn leaf_cost(&self, prim_count: usize, bbox: &AABB) -> f32 {
    //     self.intersection_cost * prim_count as f32 * bbox.surface_area()
    // }
    //
    // pub fn node_cost(&self, bbox: &AABB) -> f32 {
    //     self.traversal_cost * bbox.surface_area()
    // }
}

// --- BVH Node Definition ---
#[derive(Default)]
pub struct BVH2Node {
    pub bbox: AABB,
    pub left: Option<usize>,
    pub right: Option<usize>,

    /// The index of the first primative in the vec
    pub first_index: usize,
    pub prim_count: usize,
}

// --- Bin and Split Definitions ---
#[derive(Copy, Clone, Default)]
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

struct BinnedSah2Task {
    node_id: usize,
    begin: usize,
    end: usize,
}

impl BinnedSah2Task {
    fn size(&self) -> usize {
        self.end - self.begin
    }
}

pub struct BinnedSah2Builder<'a, const BIN_COUNT: usize> {
    bboxes: &'a [AABB],
    centers: &'a [Vector3<Float>],
    prim_ids: Vec<usize>,

    sah: SAH2,
    max_leaf_size: usize,
    min_leaf_size: usize,
    phantom: PhantomData<Float>,
}

impl<'a, const BIN_COUNT: usize> BinnedSah2Builder<'a, BIN_COUNT> {
    pub fn new<Range: RangeBounds<usize>>(
        bboxes: &'a [AABB],
        centers: &'a [Vector3<Float>],
        sah: SAH2,
        leaf_size: Range,
    ) -> Self {
        let prim_ids = (0..bboxes.len()).collect();

        let min_leaf_size = match leaf_size.start_bound() {
            std::ops::Bound::Included(&start) => start,
            std::ops::Bound::Excluded(&start) => start,
            std::ops::Bound::Unbounded => 1,
        };

        let max_leaf_size = match leaf_size.end_bound() {
            std::ops::Bound::Included(&end) => end,
            std::ops::Bound::Excluded(&end) => end,
            std::ops::Bound::Unbounded => 1,
        };

        BinnedSah2Builder {
            bboxes,
            centers,
            prim_ids,
            min_leaf_size,
            sah,
            max_leaf_size,
            phantom: PhantomData,
        }
    }

    pub fn build<Prim: Primitive<Float>>(&mut self) -> BVH2<Prim> {
        todo!();
        let prim_count = self.bboxes.len();

        let nodes: Vec<BVH2Node> = Vec::with_capacity(2 * prim_count / self.min_leaf_size);

        let stack = vec![BinnedSah2Task { node_id: 0, begin: 0, end: prim_count }];

        // while (!stack.is_empty()) {
        //     let item = stack.pop().unwrap();
        //
        //     let node = &mut nodes[item.node_id];
        //
        //     if item.size() > self.min_leaf_size {
        //         if let Some(split_pos) = try_split() {
        //
        //         }
        //     }
        //
        // }
    }

    // pub fn try_split(&self, bbox: &AABB, begin: usize, end: usize) -> Option<usize> {
    //     let bins: [[Bin; BIN_COUNT]; 3] = self.fill_bins(bbox, begin, end);
    //
    //     let largest_axis = bbox.largest_axis();
    //     let best_split = Split { bin_id: BIN_COUNT / 2, cost: Float::MAX, axis: largest_axis };
    //     for axis in 0..3 {
    //         self.find_best_split(axis, per_axis_bins[axis], best_split);
    //     }
    // }
    //
    // fn fill_bins(&self, bbox: &AABB, begin: usize, end: usize) -> [[Bin; BIN_COUNT]; 3] {
    //     let bin_scale = Vector3::repeat(BIN_COUNT as f32).component_div(&bbox.diagonal());
    //     let bin_offset = -bbox.min.coords.component_mul(&bin_scale);
    //     let mut per_axis_bins = [[Bin::default(); BIN_COUNT]; 3];
    //
    //     for i in begin..end {
    //         let pos = self.centers[self.prim_ids[i]].component_mul(&bin_scale) + bin_offset;
    //
    //         for axis in 0..3 {
    //             let index = pos[axis]
    //                 .max(0.0f32)
    //                 .min(f32::from(BIN_COUNT as f32) - 1.0f32) as usize;
    //             per_axis_bins[axis][index].add(&self.bboxes[self.prim_ids[i]], 1);
    //         }
    //
    //     }
    //
    //     per_axis_bins
    // }
    //
    // fn find_best_split(&self, axis: usize, bins: &[Bin; BIN_COUNT], best_split: &mut Split) {
    //     let mut right_accum: Bin = Bin::default();
    //     let mut right_costs: [Float; BIN_COUNT];
    //     for i in (1..BIN_COUNT).rev() {
    //         right_accum.add_bin(&bins[i]);
    //         right_costs[i] = self.sah
    //     }
    // }
}
