use std::f32;
use glam::Vec3;

use crate::primitives::triangle::Triangle;

use super::aabb::AABB;

#[derive(Debug, Clone)]
pub struct BvhNode {
    pub bounding_box: AABB,
    pub next_id: i32,
    pub primitive_id: i32,
}

#[derive(Debug, Clone, Default)]
pub struct PackedBvhNode {
    pub min: [f32; 3],
    pub next_id: i32,
    pub max: [f32; 3],
    pub primitive_id: i32,
}

#[derive(Debug, Clone, Copy, Default)]
struct TempNode {
    bounding_box: AABB,
    next_id: i32,
    primitive_id: i32,
    left_id: i32,
    df_id: i32,
}

pub struct Bvh {
    temp_nodes: Vec<TempNode>,
    leafs: Vec<TempNode>,
    triangles: Vec<Triangle>,
    next_node_id: u32,
}

impl Bvh {
    pub fn new(triangles: Vec<Triangle>) -> (Self, Vec<PackedBvhNode>) {
        let mut bvh = Bvh {
            temp_nodes: Vec::with_capacity((triangles.len() * 2) - 1),
            leafs: Vec::with_capacity(triangles.len()),
            triangles,
            next_node_id: 0,
        };

        // Initialize leaf nodes
        for (id, tri) in bvh.triangles.iter().enumerate() {
            bvh.leafs.push(TempNode {
                bounding_box: AABB::from_triangle(tri),
                primitive_id: id as i32,
                next_id: -1,
                left_id: -1,
                df_id: 0,
            });
        }

        let root_bb = bvh.compute_bounds(0, bvh.leafs.len());
        bvh.temp_nodes.push(TempNode {
            bounding_box: root_bb,
            next_id: -1,
            primitive_id: -1,
            left_id: -1,
            df_id: 0,
        });

        bvh.subdivide(0, 0, bvh.leafs.len());
        bvh.set_depth_first_order();

        let packed_nodes = bvh.pack_nodes();
        (bvh, packed_nodes)
    }

    fn compute_bounds(&self, begin: usize, end: usize) -> AABB {
        let mut global_box = AABB::default();
        for leaf in &self.leafs[begin..end] {
            global_box.union_with(&leaf.bounding_box);
        }
        global_box
    }

    fn subdivide(&mut self, parent_id: usize, begin: usize, end: usize) {
        let count = end - begin;

        if count == 1 {
            self.temp_nodes[parent_id] = self.leafs[begin];
            return;
        }

        let left_id = self.temp_nodes.len();
        self.temp_nodes.resize(left_id + 2, TempNode::default());
        self.temp_nodes[parent_id].left_id = left_id as i32;

        if count == 2 {
            self.temp_nodes[left_id] = self.leafs[begin];
            self.temp_nodes[left_id + 1] = self.leafs[begin + 1];
            return;
        }

        let parent_bb = self.temp_nodes[parent_id].bounding_box;
        let split_axis = parent_bb.maximum_axis();

        if count <= 4 {
            self.leafs[begin..end].sort_by(|a, b| {
                let a_center = self.triangles[a.primitive_id as usize].center[split_axis];
                let b_center = self.triangles[b.primitive_id as usize].center[split_axis];
                a_center.partial_cmp(&b_center).unwrap()
            });

            let mid = begin + count / 2;
            self.temp_nodes[left_id].bounding_box = self.compute_bounds(begin, mid);
            self.temp_nodes[left_id + 1].bounding_box = self.compute_bounds(mid, end);
            self.subdivide(left_id, begin, mid);
            self.subdivide(left_id + 1, mid, end);
            return;
        }

        // SAH implementation
        const BUCKETS_COUNT: usize = 12;
        let mut buckets: [Bucket; BUCKETS_COUNT] = Default::default();
        let mut centroid_bounds = AABB::default();

        for leaf in &self.leafs[begin..end] {
            centroid_bounds.union_with_point(self.triangles[leaf.primitive_id as usize].center);
        }

        let split_axis_size = centroid_bounds.max[split_axis] - centroid_bounds.min[split_axis];
        let scale = 1.0 / (split_axis_size.max(f32::EPSILON));

        for leaf in &self.leafs[begin..end] {
            let centroid = self.triangles[leaf.primitive_id as usize].center[split_axis];
            let mut bucket_index = ((centroid - centroid_bounds.min[split_axis]) * scale * BUCKETS_COUNT as f32) as usize;
            bucket_index = bucket_index.clamp(0, BUCKETS_COUNT - 1);

            buckets[bucket_index].count += 1;
            buckets[bucket_index].bb.union_with(&leaf.bounding_box);
        }

        let (min_cost_bucket, _) = buckets[..BUCKETS_COUNT-1].iter().enumerate()
            .map(|(i, _)| {
                let (left_bb, left_count) = accumulate_buckets(&buckets[..=i]);
                let (right_bb, right_count) = accumulate_buckets(&buckets[i+1..]);
                let cost = 0.125 + (left_count * left_bb.surface_area() + right_count * right_bb.surface_area()) / parent_bb.surface_area();
                (i, cost)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let mid = begin + buckets[..=min_cost_bucket].iter().map(|b| b.count).sum::<usize>();
        let mid = mid.clamp(begin + 1, end - 1);

        self.temp_nodes[left_id].bounding_box = self.compute_bounds(begin, mid);
        self.temp_nodes[left_id + 1].bounding_box = self.compute_bounds(mid, end);
        self.subdivide(left_id, begin, mid);
        self.subdivide(left_id + 1, mid, end);
    }

    fn set_depth_first_order(&mut self) {
        let mut df_index = 0;
        self.depth_first_order(0, -1, &mut df_index);
    }

    fn depth_first_order(&mut self, id: usize, next_id: i32, df_index: &mut i32) {
        let left_id = self.temp_nodes[id].left_id as usize;
        self.temp_nodes[id].df_id = *df_index;
        *df_index += 1;
        self.temp_nodes[id].next_id = next_id;

        if self.temp_nodes[id].primitive_id == -1 {
            if left_id != usize::MAX {
                self.depth_first_order(left_id, (left_id + 1) as i32, df_index);
                self.depth_first_order(left_id + 1, self.temp_nodes[id].next_id, df_index);
            }
        }
    }

    fn pack_nodes(&self) -> Vec<PackedBvhNode> {
        let mut packed = vec![PackedBvhNode::default(); self.temp_nodes.len()];
        for node in &self.temp_nodes {
            let packed_node = &mut packed[node.df_id as usize];
            packed_node.min = node.bounding_box.min.into();
            packed_node.max = node.bounding_box.max.into();
            packed_node.primitive_id = node.primitive_id;
            packed_node.next_id = if node.next_id != -1 {
                self.temp_nodes[node.next_id as usize].df_id
            } else {
                -1
            };
        }
        packed
    }
}

#[derive(Default, Clone)]
struct Bucket {
    count: usize,
    bb: AABB,
}

fn accumulate_buckets(buckets: &[Bucket]) -> (AABB, f32) {
    let mut bb = AABB::default();
    let mut count = 0;
    for bucket in buckets {
        if bucket.count > 0 {
            bb.union_with(&bucket.bb);
            count += bucket.count;
        }
    }
    (bb, count as f32)
}
