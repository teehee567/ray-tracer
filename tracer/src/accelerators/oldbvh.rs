use std::cmp::Ordering;
use std::sync::Arc;

use crate::accelerators::aabb::AABB;
use crate::core::hittable::{HitRecord, Hittable};
use crate::core::interval::Interval;
use crate::core::ray::Ray;

enum BVHNode {
    Branch { left: Box<OLDBVH>, right: Box<OLDBVH> },
    Leaf(Box<dyn Hittable>),
}

pub struct OLDBVH {
    tree: BVHNode,
    bbox: AABB,
}

impl OLDBVH {
    pub fn new(mut objects: Vec<Box<dyn Hittable>>, time0: f32, time1: f32) -> Self {
        fn axis_range(hittable: &[Box<dyn Hittable>], time0: f32, time1: f32, axis: usize) -> f32 {
            let (min, max) = hittable
                .iter()
                .fold((f32::MAX, f32::MIN), |(bmin, bmax), hit| {
                    let aabb = hit.bounding_box();
                    let (axis_min, axis_max) = match axis {
                        0 => (aabb.min.x, aabb.max.x),
                        1 => (aabb.min.y, aabb.max.y),
                        2 => (aabb.min.z, aabb.max.z),
                        _ => unreachable!(),
                    };
                    (bmin.min(axis_min), bmax.max(axis_max))
                });
            max - min
        }

        let longest_axis = (0..3)
            .map(|a| (a, axis_range(&objects, time0, time1, a)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(axis, _)| axis)
            .unwrap_or(0);

        // Sort hittables along the chosen axis
        objects.sort_unstable_by(|a, b| {
            let abb = a.bounding_box();
            let bbb = b.bounding_box();
            let ac = match longest_axis {
                0 => abb.min.x + abb.max.x,
                1 => abb.min.y + abb.max.y,
                2 => abb.min.z + abb.max.z,
                _ => panic!("Axis out of range(it might be over)"),
            };
            let bc = match longest_axis {
                0 => bbb.min.x + bbb.max.x,
                1 => bbb.min.y + bbb.max.y,
                2 => bbb.min.z + bbb.max.z,
                _ => panic!("Axis out of range(it might be over)"),
            };
            ac.partial_cmp(&bc).unwrap()
        });

        let len = objects.len();
        match len {
            0 => panic!("No elements in scene"),
            1 => {
                let leaf = objects.pop().unwrap();
                let bbox = leaf.bounding_box().clone();
                Self {
                    tree: BVHNode::Leaf(leaf),
                    bbox,
                }
            }
            _ => {
                let right = Self::new(objects.drain(len / 2..).collect(), time0, time1);
                let left = Self::new(objects, time0, time1);
                let bbox = AABB::combine(&left.bbox, &right.bbox);
                Self {
                    tree: BVHNode::Branch {
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    bbox,
                }
            }
        }
    }
}

impl ToString for OLDBVH {
    fn to_string(&self) -> String {
        let mut output = String::new();

        let mut nodes: i32 = 0;

        fn build_string(bvh: &OLDBVH, level: i32) -> (ascii_tree::Tree, i32, i32) {
            match &bvh.tree {
                BVHNode::Branch { left, right } => {
                    let left_node = build_string(left, level + 1);
                    let right_node = build_string(right, level + 1);

                    (
                        ascii_tree::Tree::Node(
                            format!("level: {} ({}, {})", level, bvh.bbox.min, bvh.bbox.max),
                            vec![left_node.0, right_node.0],
                        ),
                        left_node.1 + right_node.1 + 1,
                        left_node.2 + left_node.2,
                    )
                }
                BVHNode::Leaf(_) => (
                    ascii_tree::Tree::Leaf(vec![format!(
                        "end: ({}, {})",
                        bvh.bbox.min, bvh.bbox.max
                    )]),
                    0,
                    1,
                ),
            }
        }

        let level = 0;
        let tree = build_string(self, level);
        let _ = ascii_tree::write_tree(&mut output, &tree.0);

        format!(
            "{}\ntotal nodes: {}, total leaves: {}",
            output, tree.1, tree.2
        )
    }
}

impl Hittable for OLDBVH {
    fn hit(&self, ray: &Ray, mut ray_t: Interval, rec: &mut HitRecord) -> bool {
        if self.bbox.hit(ray, ray_t) {
            match &self.tree {
                BVHNode::Leaf(leaf) => leaf.hit(ray, ray_t, rec),
                BVHNode::Branch { left, right } => {
                    let hit_left = left.hit(ray, ray_t, rec);
                    let hit_right = right.hit(
                        ray,
                        Interval::new(ray_t.min, if hit_left { rec.t } else { ray_t.max }),
                        rec,
                    );

                    hit_left || hit_right
                }
            }
        } else {
            false
        }
    }

    fn bounding_box(&self) -> &AABB {
        &self.bbox
    }
}
